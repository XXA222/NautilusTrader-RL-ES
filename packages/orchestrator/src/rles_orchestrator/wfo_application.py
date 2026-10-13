"""Leakage-safe WFO execution over the installed historical application."""
from __future__ import annotations

from dataclasses import asdict, replace
from hashlib import sha256
import json
import math
from pathlib import Path
import re
import statistics
from time import time_ns
from typing import Callable, cast

import numpy as np

from rles_data.core import CuratedStore, DataError, digest
from rles_data.coverage import CoveragePolicy
from rles_data.features import Scaler
from rles_data.folds import bind_fold_scaler, build_folds
from rles_data.io import read_curated
from rles_execution import ResolvedExecutionConfig
from rles_gym_bridge.historical import (
    HistoricalEnvironmentFactory,
    historical_feature_spec,
    load_rules,
)
from rles_research.baselines import BaselinePolicy, BaselineSpec
from rles_research.ledger import ResearchLedger
from rles_research.native_metrics import NATIVE_LIQUIDATION_COUNTS_AVAILABLE
from rles_research.selection import SelectionSpec, choose
from rles_research.statistics import (
    block_bootstrap,
    candidate_set_sha256,
    deflated_sharpe,
    pbo_cscv,
)

from .application import _file_hash, _verify_file, _worker_memory_policy, _write_verified, backtest, rollout, train


_CANDIDATE_FIELDS = {
    "candidate_id", "route", "algorithm", "steps", "population", "generations",
    "seed", "hidden_layers", "normalize_observation", "hyperparameters", "distillation",
    "deadband", "es_input_scaling", "feature_profile",
}


def _coverage(template: dict, instrument: str, start_ns: int, end_ns: int) -> dict:
    required = {
        "maximum_age_ns", "settlement_interval_minutes", "settlement_anchor_ns",
        "settlement_max_delay_ns", "provenance",
    }
    if (not required <= set(template)
            or set(template) - required - {"settlement_timestamp_tolerance_ns"}):
        raise DataError("WFO coverage template fields mismatch")
    interval_minutes = template["settlement_interval_minutes"]
    anchor = template["settlement_anchor_ns"]
    if type(interval_minutes) is not int or interval_minutes <= 0 or type(anchor) is not int or anchor < 0:
        raise DataError("invalid source-declared funding schedule")
    interval = interval_minutes * 60_000_000_000
    first = anchor + max(0, (start_ns - anchor + interval - 1) // interval) * interval
    # Settlement verification remains limited to this scoring window. The
    # separate source-declared calendar includes 24h of known future schedule
    # so an identical decision time gets the same causal countdown at fold edges.
    calendar = tuple(range(first, end_ns + 24 * 60 * 60 * 1_000_000_000, interval))
    settlements = tuple(timestamp for timestamp in calendar if timestamp < end_ns)
    return {
        "maximum_age_ns": template["maximum_age_ns"],
        "settlement_times_ns": {instrument: settlements},
        "known_schedule_times_ns": {instrument: calendar},
        "settlement_max_delay_ns": template["settlement_max_delay_ns"],
        "settlement_timestamp_tolerance_ns": template.get("settlement_timestamp_tolerance_ns", 0),
        "provenance": template["provenance"],
    }


def _require_window(store: CuratedStore, template: dict, instrument: str,
                    start_ns: int, end_ns: int) -> None:
    """Gate every read window, including read-only feature warmup."""
    policy = CoveragePolicy(**_coverage(template, instrument, start_ns, end_ns))
    report = store.quality(
        start_ns=start_ns,
        end_ns=end_ns,
        expected_instruments=(instrument,),
        coverage_policy=policy,
    )
    store.require_usable(report, "rl_es")


def _candidate(base: dict, values: dict, output: Path) -> dict:
    if not isinstance(values, dict) or set(values) - _CANDIDATE_FIELDS:
        raise DataError("unknown WFO candidate fields")
    if (not isinstance(values.get("es_input_scaling", "account_units"), str)
            or values.get("es_input_scaling", "account_units") not in {"none", "account_units"}):
        raise DataError("invalid WFO ES input scaling")
    identity = values.get("candidate_id")
    if not isinstance(identity, str) or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", identity) is None:
        raise DataError("invalid WFO candidate identity")
    result = {k: v for k, v in base.items() if k not in {"output", "evaluation", "wfo"}}
    result.update({k: v for k, v in values.items()
                   if k not in {"candidate_id", "deadband", "feature_profile"}})
    if "feature_profile" in values:
        profile = values["feature_profile"]
        if not isinstance(profile, str) or profile not in {
            "legacy", "causal_mtf_v1", "causal_mtf_v2", "causal_mtf_v3",
        }:
            raise DataError("invalid WFO candidate feature profile")
        environment = base.get("environment")
        if not isinstance(environment, dict):
            raise DataError("WFO candidate feature profile requires a historical environment")
        result["environment"] = {**environment, "feature_profile": profile}
    if "deadband" in values:
        environment = result.get("environment", base.get("environment"))
        if not isinstance(environment, dict) or not isinstance(environment.get("execution"), dict):
            raise DataError("WFO candidate deadband requires a resolved execution configuration")
        if type(values["deadband"]) not in (str, int):
            raise DataError("WFO candidate deadband must be a decimal string or integer")
        execution = {**environment["execution"], "deadband": values["deadband"]}
        try:
            ResolvedExecutionConfig.from_dict(execution)
        except (TypeError, ValueError) as exc:
            raise DataError("WFO candidate deadband is invalid for the execution configuration") from exc
        result["environment"] = {**environment, "execution": execution}
    result["output"] = str(output)
    return result


def _with_window(environment: dict, *, start_ns: int, end_ns: int, coverage_policy: dict) -> dict:
    if not isinstance(environment, dict):
        raise DataError("WFO candidate environment is invalid")
    return {**environment, "start_ns": start_ns, "end_ns": end_ns,
            "coverage_policy": coverage_policy}


def _returns(report: dict) -> list[float]:
    rows = report.get("account_rows")
    if not isinstance(rows, list) or len(rows) < 3:
        raise DataError("WFO report lacks regular native account rows")
    result = []
    for previous, current in zip(rows, rows[1:]):
        before, after = float(previous["equity"]), float(current["equity"])
        flow = float(current["external_cashflow"])
        value = (after - flow) / before - 1
        if not math.isfinite(value) or value <= -1:
            raise DataError("WFO account return is invalid")
        result.append(value)
    return result


def _period_sharpe(values: list[float]) -> float:
    if len(values) < 2:
        raise DataError("WFO Sharpe requires at least two period returns")
    deviation = statistics.stdev(values)
    if deviation <= 0:
        raise DataError("constant WFO returns cannot support statistical selection")
    return statistics.fmean(values) / deviation


def _aggregate_candidate(candidate_id: str, reports: list[dict], primary: str) -> dict:
    if not reports or any(
        report.get("liquidations_observed") is not True
        or type(report.get("liquidations")) is not int
        or report["liquidations"] < 0
        for report in reports
    ):
        raise DataError(
            f"candidate {candidate_id} cannot be selected: fold reports lack native liquidation facts"
        )
    if any(report.get("risk_intervention_components_known") is not True
           or type(report.get("risk_hard_interventions")) is not int
           or report["risk_hard_interventions"] < 0 for report in reports):
        raise DataError(f"candidate {candidate_id} cannot be selected: fold reports lack hard-risk attribution")
    metrics = [report["metrics"] for report in reports]
    field = {
        "median_fold_sortino": "sortino",
        "median_fold_net_return": "net_return",
        "median_fold_sharpe": "sharpe",
    }[primary]
    values = [row[field] for row in metrics]
    net_values = [row.get("net_return") for row in metrics]
    if any(value is None or not math.isfinite(float(value)) for value in (*values, *net_values)):
        raise DataError(f"candidate {candidate_id} has undefined {field}")
    return {
        "candidate_id": candidate_id,
        "liquidations": sum(int(report["liquidations"]) for report in reports),
        "max_drawdown": max(float(row["max_drawdown"]) for row in metrics),
        "turnover": max(float(row["turnover_ratio"]) for row in metrics),
        # Candidate feasibility constrains hard-guard changes. Continuous soft
        # turnover/funding shaping remains a separate reward and report metric.
        "risk_violations": sum(report["risk_hard_interventions"] for report in reports),
        primary: statistics.median(float(value) for value in values),
        "median_fold_net_return": statistics.median(float(value) for value in net_values),
    }


def _stress_validation_gate(reports: list[dict], spec: SelectionSpec) -> dict:
    """Fail closed unless every paired stress Validation clears frozen feasibility limits."""
    reasons: list[str] = []
    returns: list[float] = []
    hard_interventions = 0
    if not reports:
        return {"passed": False, "reasons": ["missing_stress_validation"], "fold_net_returns": []}
    for index, report in enumerate(reports):
        metrics = report.get("metrics") if isinstance(report, dict) else None
        if (not isinstance(metrics, dict)
                or report.get("execution_mode") != "stress"
                or report.get("liquidations_observed") is not True
                or type(report.get("liquidations")) is not int
                or type(report.get("risk_intervention_components_known")) is not bool
                or report.get("risk_intervention_components_known") is not True
                or type(report.get("risk_hard_interventions")) is not int
                or report.get("terminal_flat") is not True):
            reasons.append(f"fold_{index}:incomplete_stress_evidence")
            continue
        values = {name: metrics.get(name) for name in ("net_return", "max_drawdown", "turnover_ratio")}
        if any(isinstance(value, bool) or not isinstance(value, (int, float))
               or not math.isfinite(float(value)) for value in values.values()):
            reasons.append(f"fold_{index}:invalid_stress_metrics")
            continue
        net_return = float(cast(float, values["net_return"]))
        returns.append(net_return)
        hard_interventions += report["risk_hard_interventions"]
        if report["liquidations"] != 0:
            reasons.append(f"fold_{index}:liquidation")
        if net_return <= 0:
            reasons.append(f"fold_{index}:nonpositive_net_return")
        if net_return < spec.min_median_fold_net_return:
            reasons.append(f"fold_{index}:net_return_below_gate")
        if float(cast(float, values["max_drawdown"])) > spec.max_drawdown:
            reasons.append(f"fold_{index}:drawdown_limit")
        if float(cast(float, values["turnover_ratio"])) > spec.max_turnover:
            reasons.append(f"fold_{index}:turnover_limit")
    if hard_interventions > spec.max_risk_violations:
        reasons.append("total_hard_risk_interventions")
    return {
        "passed": not reasons,
        "reasons": reasons,
        "fold_net_returns": returns,
        "total_hard_risk_interventions": hard_interventions,
        "minimum_net_return_gate": spec.min_median_fold_net_return,
        "minimum_observed_net_return": min(returns) if returns else None,
    }


def _holdout_constraints_pass(report: dict, spec: SelectionSpec) -> bool:
    """Require complete, classified native facts before accepting Holdout gates."""
    metrics = report.get("metrics") if isinstance(report, dict) else None
    liquidations = report.get("liquidations") if isinstance(report, dict) else None
    hard_interventions = report.get("risk_hard_interventions") if isinstance(report, dict) else None
    if (not isinstance(metrics, dict) or report.get("terminal_flat") is not True
            or report.get("liquidations_observed") is not True
            or type(liquidations) is not int or liquidations < 0
            or report.get("risk_intervention_components_known") is not True
            or type(hard_interventions) is not int or hard_interventions < 0):
        return False
    drawdown, turnover = metrics.get("max_drawdown"), metrics.get("turnover_ratio")
    if any(isinstance(value, bool) or not isinstance(value, (int, float))
           or not math.isfinite(float(value)) for value in (drawdown, turnover)):
        return False
    return (
        liquidations == 0
        and 0 <= float(cast(float, drawdown)) <= spec.max_drawdown
        and 0 <= float(cast(float, turnover)) <= spec.max_turnover
        and hard_interventions <= spec.max_risk_violations
    )


def _holdout_return_gate(report: dict, spec: SelectionSpec) -> tuple[object, bool]:
    """Require the frozen minimum net return on the one-shot Holdout result."""
    metrics = report.get("metrics") if isinstance(report, dict) else None
    value = metrics.get("net_return") if isinstance(metrics, dict) else None
    passed = (
        isinstance(value, (int, float)) and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) > 0
        and float(value) >= spec.min_median_fold_net_return
    )
    return value, passed


def _source_tree_hash(root: Path) -> str:
    value = sha256()
    files = sorted(root.glob("packages/*/src/**/*.py"), key=lambda path: path.as_posix())
    entries = [(path.relative_to(root).as_posix(), path) for path in files]
    if not entries:
        # Wheels retain their Python sources but have no workspace packages/src
        # layout. Bind the code actually imported by this environment instead.
        from importlib.util import find_spec

        for module in (
            "rles_contracts", "rles_data", "rles_execution", "rles_risk",
            "rles_nautilus_adapter", "rles_gym_bridge", "rles_rl_sb3",
            "rles_es_evotorch", "rles_orchestrator", "rles_compute_runtime", "rles_research",
        ):
            spec = find_spec(module)
            if spec is None or spec.origin is None:
                raise DataError(f"installed WFO source unavailable for {module}")
            package = Path(spec.origin).resolve().parent
            sources = sorted(package.rglob("*.py"), key=lambda path: path.as_posix())
            if not sources or any(path.is_symlink() for path in sources):
                raise DataError(f"installed WFO source unavailable for {module}")
            entries.extend((f"{module}/{path.relative_to(package).as_posix()}", path) for path in sources)
    for name, path in sorted(entries):
        value.update(name.encode())
        value.update(bytes.fromhex(_file_hash(path)))
    return value.hexdigest()


def _scaler(bundle_root: Path) -> Scaler:
    _verify_file(bundle_root / "bundle.json")
    bundle = json.loads((bundle_root / "bundle.json").read_text(encoding="utf-8"))
    values = dict(bundle["preprocessing"]["scaler"])
    values["mean"], values["std"] = tuple(values["mean"]), tuple(values["std"])
    return Scaler(**values)


def _stress_validation_report(run: dict, bundle_root: Path, *, start_ns: int, end_ns: int,
                              coverage_policy: dict) -> dict:
    """Replay a completed candidate on the same Validation bars under frozen stress costs."""
    _verify_file(bundle_root / "bundle.json")
    bundle = json.loads((bundle_root / "bundle.json").read_text(encoding="utf-8"))
    environment = bundle.get("environment")
    if not isinstance(environment, dict):
        raise DataError("stress validation bundle lacks its frozen environment")
    factory = HistoricalEnvironmentFactory(**{
        **environment, "start_ns": start_ns, "end_ns": end_ns,
        "coverage_policy": coverage_policy, "fit_scaler": False,
        "split": "validation", "mode": "stress", "scaler": _scaler(bundle_root),
    })
    report = backtest(run, bundle_root, factory=factory, allow_execution_stress=True)
    if report.get("execution_mode") != "stress":
        raise DataError("stress validation report lacks stress execution identity")
    return report


def _baseline_reports(config: dict, bundle_root: Path, *, seed: int) -> dict[str, dict]:
    bundle = json.loads((bundle_root / "bundle.json").read_text(encoding="utf-8"))
    scaler = _scaler(bundle_root)
    factory = HistoricalEnvironmentFactory(**{
        **bundle["environment"], **config["evaluation"], "fit_scaler": False,
        "split": "test", "scaler": scaler,
    })
    output = {}
    for name in ("B0_no_trade", "B1_always_long", "B2_simple_momentum", "B3_random_action"):
        policy = BaselinePolicy(BaselineSpec(name, seed))
        cursor = 0
        close_cursor = 0
        closes: list[float] = []
        bars = tuple(
            (int(event.available_at_ns), float(event.payload["close"]))
            for event in factory.store.events
            if event.kind == "ohlcv_1m"
            and event.payload["instrument"] == factory.instrument
            and event.available_at_ns is not None
        )

        def predict(batch: np.ndarray) -> np.ndarray:
            nonlocal cursor, close_cursor
            if len(batch) != 1 or cursor >= len(factory.rows):
                raise DataError("baseline rollout cursor differs from native environment")
            decision = factory.rows[cursor].decision_ts_ns
            while close_cursor < len(bars) and bars[close_cursor][0] <= decision:
                closes.append(bars[close_cursor][1])
                close_cursor += 1
            cursor += 1
            return np.asarray([[policy.action(closes)]], dtype=np.float32)

        report, _ = rollout(factory, predict, seed=seed)
        output[name] = report
    return output


def _latest_final_retrain_checkpoint(run_root: Path, route: str) -> Path | None:
    """Return the latest completed, route-supported checkpoint for partial training.

    Integrity and recipe compatibility are rechecked by ``train`` before it
    resumes. This selector only chooses durable checkpoint boundaries and
    never treats a partially written directory as resumable.
    """
    route_stages = {
        "A0": ("R1",),
        "A1": ("E1",),
        "A2": ("R1", "E2"),
        "A3": ("E1", "R1"),
        "A4": ("E1", "R1", "E2"),
    }
    stages = route_stages.get(route)
    if stages is None:
        raise DataError("final retrain route has no supported checkpoint recovery path")
    manifest_path = run_root / "run.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise DataError("partial final retrain has no durable application run manifest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    attempt_id = manifest.get("attempt_id") if isinstance(manifest, dict) else None
    if (not isinstance(manifest, dict)
            or manifest.get("schema") != "rles.historical-run.v1"
            or not isinstance(attempt_id, str) or manifest.get("run_id") != attempt_id):
        raise DataError("partial final retrain run manifest has invalid attempt identity")

    for stage in reversed(stages):
        stage_root = run_root / ("R1-checkpoints" if stage == "R1" else stage)
        if stage_root.is_symlink() or not stage_root.is_dir():
            continue
        if stage == "R1":
            candidates = sorted(
                (path for path in stage_root.glob("step-[0-9]*")
                 if path.is_dir() and not path.is_symlink()
                 and path.name[5:].isdigit()
                 and (path / "COMPLETED").is_file()
                 and not (path / "COMPLETED").is_symlink()),
                key=lambda path: int(path.name[5:]), reverse=True,
            )
            if candidates:
                return candidates[0]
            continue
        generations = sorted(
            (path for path in stage_root.glob("generation-[0-9]*")
             if path.is_dir() and not path.is_symlink() and path.name[11:].isdigit()),
            key=lambda path: int(path.name[11:]), reverse=True,
        )
        for generation in generations:
            attempt = generation / attempt_id
            if (attempt.is_dir() and not attempt.is_symlink()
                    and (attempt / "manifest.json").is_file()
                    and not (attempt / "manifest.json").is_symlink()
                    and (attempt / "COMPLETED").is_file()
                    and not (attempt / "COMPLETED").is_symlink()):
                return attempt
    return None


def run_wfo(config: dict, *, train_fn: Callable[..., dict] = train,
            backtest_fn: Callable[..., dict] = backtest,
            stress_validation_fn: Callable[..., dict] | None = None,
            resume: bool = False) -> dict:
    """Execute Validation selection, one-shot fold Test, retrain and Holdout."""
    if type(resume) is not bool:
        raise DataError("WFO resume flag must be a boolean")
    settings = config.get("wfo")
    if not isinstance(settings, dict):
        raise DataError("WFO configuration required")
    required = {
        "protocol_id", "scheme", "start_ns", "end_ns", "data_start_ns", "warmup_ns",
        "execution_lag_guard_ns", "embargo_ns", "holdout_start_ns", "holdout_end_ns",
        "coverage_template", "candidates", "selection", "bootstrap", "pbo_max_partitions",
        "reviewer_approval_sha",
    }
    if set(settings) != required:
        optional = {"cost_stress_validation"}
        if set(settings) - required - optional or required - set(settings):
            raise DataError("WFO configuration fields mismatch")
    cost_stress_validation = settings.get("cost_stress_validation", True)
    if type(cost_stress_validation) is not bool:
        raise DataError("cost_stress_validation must be an explicit boolean")
    _worker_memory_policy(config)
    root = Path(config["output"]).resolve()
    if root.exists() and not resume:
        raise DataError("WFO output already exists; pass --resume to recover validation-stage work")
    if resume and not root.is_dir():
        raise DataError("WFO resume requires an existing output directory")
    candidates = settings["candidates"]
    if not isinstance(candidates, list) or len(candidates) < 2:
        raise DataError("WFO requires at least two frozen candidates")
    candidate_ids_raw = [item.get("candidate_id") for item in candidates if isinstance(item, dict)]
    if (len(candidate_ids_raw) != len(candidates)
            or any(not isinstance(identity, str) for identity in candidate_ids_raw)
            or len(set(candidate_ids_raw)) != len(candidate_ids_raw)):
        raise DataError("WFO candidate identities must be unique")
    candidate_ids = cast(list[str], candidate_ids_raw)
    # Validate every candidate recipe before creating a durable protocol.
    for values in candidates:
        _candidate(config, values, Path("candidate-validation-only"))
    holdout_start, holdout_end = settings["holdout_start_ns"], settings["holdout_end_ns"]
    if not (settings["start_ns"] < holdout_start < holdout_end <= settings["end_ns"]):
        raise DataError("WFO final Holdout boundaries are invalid")
    if train_fn is train and not NATIVE_LIQUIDATION_COUNTS_AVAILABLE:
        raise DataError(
            "WFO cannot start: installed native adapter does not expose liquidation event counts"
        )

    environment = dict(config["environment"])
    instrument = environment["instrument"]
    store = CuratedStore((*read_curated(environment["market_path"]).events,
                          *load_rules(environment["rules_path"])))
    template = settings["coverage_template"]
    default_profile = environment.get("feature_profile", "legacy")
    candidate_profiles = {
        values["candidate_id"]: values.get("feature_profile", default_profile)
        for values in candidates
    }
    if any(not isinstance(profile, str) or profile not in {
        "legacy", "causal_mtf_v1", "causal_mtf_v2", "causal_mtf_v3",
    } for profile in candidate_profiles.values()):
        raise DataError("WFO candidates contain an unknown feature profile")
    candidate_features = {
        identity: (historical_feature_spec(dict(template["maximum_age_ns"]))
                   if profile == "legacy" else
                   historical_feature_spec(dict(template["maximum_age_ns"]), profile=profile))
        for identity, profile in candidate_profiles.items()
    }
    candidate_feature_hashes = {
        identity: (digest({"profile": candidate_profiles[identity], "feature_spec": spec.hash})
                   if candidate_profiles[identity] == "causal_mtf_v3" else spec.hash)
        for identity, spec in candidate_features.items()
    }
    protocol_feature_hash = digest(candidate_feature_hashes)
    scheme = settings["scheme"]
    train_days = {"60/15/15": 60, "360/15/15": 360}.get(scheme)
    if train_days is None:
        raise DataError("unsupported WFO scheme")
    manifests = build_folds(
        start_ns=settings["start_ns"], end_ns=holdout_start, train_days=train_days,
        warmup_ns=settings["warmup_ns"],
        feature_lookback_ns=max(spec.max_lookback_ns for spec in candidate_features.values()),
        execution_lag_guard_ns=settings["execution_lag_guard_ns"],
        embargo_ns=settings["embargo_ns"], dataset_hash=store.snapshot_hash,
        feature_hash=protocol_feature_hash, data_start_ns=settings["data_start_ns"],
    )
    if not manifests:
        raise DataError("WFO window does not contain one complete approved fold")
    candidate_manifests = {
        identity: tuple(replace(manifest, feature_hash=candidate_feature_hashes[identity])
                        for manifest in manifests)
        for identity in candidate_ids
    }
    for manifest in manifests:
        for window in (manifest.train, manifest.validation, manifest.test):
            _require_window(store, template, instrument, window.warmup_start_ns,
                            window.score_end_ns)
    _require_window(store, template, instrument, settings["start_ns"] - settings["warmup_ns"],
                    holdout_start)
    _require_window(store, template, instrument, holdout_start - settings["warmup_ns"],
                    holdout_end)
    source_root = Path(__file__).resolve().parents[4]
    source_hash = _source_tree_hash(source_root)
    protocol_id = settings["protocol_id"]
    resume_manifest = {
        "schema": "rles.wfo-resume.v1",
        "protocol_id": protocol_id,
        "config_hash": digest({key: value for key, value in config.items() if key != "output"}),
        "source_hash": source_hash,
        "dataset_hash": store.snapshot_hash,
        "feature_hash": protocol_feature_hash,
        "candidate_feature_hashes": candidate_feature_hashes,
        "candidate_feature_profiles": candidate_profiles,
        "cost_stress_validation": cost_stress_validation,
        "fold_manifests": [asdict(value) for value in manifests],
        "candidate_fold_manifests": {
            identity: [asdict(value) for value in values]
            for identity, values in candidate_manifests.items()
        },
    }
    protocol_state = "DEVELOPMENT"
    if resume:
        ledger = ResearchLedger(root / "research.sqlite")
        manifest_path = root / "wfo-run.json"
        _verify_file(manifest_path)
        saved_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if saved_manifest != resume_manifest:
            raise DataError("WFO resume configuration, source, data, features, or folds changed")
        protocol_state = ledger.state(protocol_id)["state"]
        if protocol_state == "SUSPENDED":
            protocol_state = ledger.resume_suspended(protocol_id)
        elif protocol_state not in {"DEVELOPMENT", "RECIPE_FROZEN", "RETRAINED",
                                    "HOLDOUT_OPENED", "CLOSED"}:
            raise DataError("WFO protocol is not at a resumable stage")
        failure = root / "FAILED.json"
        if failure.exists():
            failure.replace(root / f"FAILED-resumed-{time_ns()}.json")
        completed_report = root / "wfo-report.json"
        if protocol_state == "CLOSED" and completed_report.exists():
            _verify_file(completed_report)
            report = json.loads(completed_report.read_text(encoding="utf-8"))
            if (report.get("protocol_id") != protocol_id
                    or report.get("dataset_hash") != store.snapshot_hash
                    or report.get("feature_hash") != protocol_feature_hash
                    or report.get("candidate_feature_hashes") != candidate_feature_hashes):
                raise DataError("closed WFO report differs from the frozen protocol identity")
            holdout_path = root / "holdout-report.json"
            _verify_file(holdout_path)
            holdout = json.loads(holdout_path.read_text(encoding="utf-8"))
            if digest(report.get("holdout")) != digest(holdout) or not ledger.verify_audit_chain():
                raise DataError("closed WFO Holdout artifact or ledger audit chain is inconsistent")
            return report
    else:
        root.mkdir(parents=True)
        ledger = ResearchLedger(root / "research.sqlite")
        validation_queries = len(manifests) * len(candidates) * (2 if cost_stress_validation else 1)
        ledger.create(protocol_id, protocol_id, validation_budget=validation_queries,
                      hpo_budget=0, es_evaluation_budget=0)
        _write_verified(root / "wfo-run.json", resume_manifest)
    validation: dict[str, list[dict]] = {str(identity): [] for identity in candidate_ids}
    stress_validation: dict[str, list[dict]] = {str(identity): [] for identity in candidate_ids}
    bundles: dict[tuple[int, str], Path] = {}
    bound_manifests: dict[str, list] = {identity: [] for identity in candidate_ids}

    try:
        for fold_index, manifest in enumerate(manifests):
            fold_id = f"{scheme.replace('/', '-')}-{fold_index:04d}"
            for values in candidates:
                candidate_id = values["candidate_id"]
                manifest = candidate_manifests[candidate_id][fold_index]
                output = root / "folds" / fold_id / "validation" / candidate_id
                run = _candidate(config, values, output)
                run["environment"] = _with_window(
                    run["environment"], start_ns=manifest.train.score_start_ns,
                    end_ns=manifest.train.score_end_ns,
                    coverage_policy=_coverage(template, instrument, manifest.train.score_start_ns,
                                              manifest.train.score_end_ns),
                )
                run["evaluation"] = {
                    "start_ns": manifest.validation.score_start_ns,
                    "end_ns": manifest.validation.score_end_ns, "split": "validation",
                    "coverage_policy": _coverage(template, instrument, manifest.validation.score_start_ns,
                                                   manifest.validation.score_end_ns),
                }
                query_id = f"validation-{fold_index}-{candidate_id}"
                existing_query = ledger.query(protocol_id, query_id)
                if existing_query is not None:
                    if (not resume or existing_query != {
                            "query_id": query_id, "kind": "validation",
                            "candidate": candidate_id, "fold_id": fold_id,
                    }):
                        raise DataError("WFO Validation reservation differs from the frozen fold/candidate")
                    if not output.is_dir():
                        raise DataError(
                            "WFO Validation was reserved but has no output; start a new protocol attempt"
                        )
                    report_path = output / "report.json"
                    bundle_path = output / "bundle.json"
                    completed_path = output / "COMPLETED.json"
                    if not completed_path.is_file():
                        if protocol_state != "DEVELOPMENT":
                            raise DataError("partial WFO Validation training is only resumable before recipe freeze")
                        # Bundle publication precedes Validation evaluation.
                        # Without a receipt after that boundary, its query may
                        # already have been consumed: never replay evaluation.
                        if any(path.exists() for path in (
                            bundle_path, report_path,
                            output / "bundle.json.sha256", output / "report.json.sha256",
                        )):
                            raise DataError("WFO Validation evaluation may have started without a receipt; refusing to replay")
                        checkpoint = _latest_final_retrain_checkpoint(output, str(run.get("route", "A0")))
                        if checkpoint is None:
                            raise DataError("partial WFO Validation training has no completed stage checkpoint; start a new protocol attempt")
                        trained = train_fn({**run, "resume_checkpoint": str(checkpoint)})
                        if Path(trained["output"]).resolve() != output.resolve():
                            raise DataError("resumed WFO Validation training returned a different output")
                    report_hash, bundle_hash = _verify_file(report_path), _verify_file(bundle_path)
                    if not completed_path.is_file():
                        raise DataError("resumed WFO Validation training has no completion receipt")
                    completed = json.loads(completed_path.read_text(encoding="utf-8"))
                    if (completed.get("report_sha256") != report_hash
                            or completed.get("bundle_sha256") != bundle_hash):
                        raise DataError("WFO Validation completion receipt does not match its artifacts")
                    trained = {"output": str(output)}
                else:
                    if output.exists():
                        raise DataError("WFO candidate output exists without its durable Validation reservation")
                    ledger.record_query(protocol_id, query_id, "validation", candidate_id, fold_id)
                    trained = train_fn(run)
                report_path = Path(trained["output"]) / "report.json"
                _verify_file(report_path)
                report = json.loads(report_path.read_text(encoding="utf-8"))
                if cost_stress_validation and report.get("execution_mode") != "base":
                    raise DataError("paired cost validation requires a base-mode Validation report")
                validation[candidate_id].append(report)
                bundle_root = Path(trained["output"])
                bundles[(fold_index, candidate_id)] = bundle_root
                if cost_stress_validation:
                    stress_root = root / "folds" / fold_id / "stress-validation" / candidate_id
                    stress_report_path = stress_root / "report.json"
                    stress_receipt_path = stress_root / "receipt.json"
                    stress_query_id = f"stress-validation-{fold_index}-{candidate_id}"
                    stress_query = ledger.query(protocol_id, stress_query_id)
                    expected_query = {
                        "query_id": stress_query_id, "kind": "validation",
                        "candidate": candidate_id, "fold_id": fold_id,
                    }
                    if stress_query is not None:
                        if not resume or stress_query != expected_query:
                            raise DataError("stress Validation reservation differs from the frozen fold/candidate")
                        if not stress_root.is_dir() or not stress_report_path.is_file() or not stress_receipt_path.is_file():
                            raise DataError("reserved stress Validation may have been consumed; refusing to replay")
                        stress_report_hash = _verify_file(stress_report_path)
                        _verify_file(stress_receipt_path)
                        stress_report = json.loads(stress_report_path.read_text(encoding="utf-8"))
                        stress_receipt = json.loads(stress_receipt_path.read_text(encoding="utf-8"))
                        expected_receipt = {
                            "schema": "rles.wfo-stress-validation.v1", "protocol_id": protocol_id,
                            "fold_id": fold_id, "candidate_id": candidate_id,
                            "bundle_sha256": _verify_file(bundle_root / "bundle.json"),
                            "report_sha256": stress_report_hash,
                            "start_ns": manifest.validation.score_start_ns,
                            "end_ns": manifest.validation.score_end_ns,
                            "mode": "stress",
                        }
                        if stress_receipt != expected_receipt or stress_report.get("execution_mode") != "stress":
                            raise DataError("stress Validation receipt/report identity mismatch")
                    else:
                        if stress_root.exists():
                            raise DataError("stress Validation artifacts exist without a durable query")
                        ledger.record_query(protocol_id, stress_query_id, "validation", candidate_id, fold_id)
                        coverage_policy = _coverage(
                            template, instrument, manifest.validation.score_start_ns,
                            manifest.validation.score_end_ns,
                        )
                        if stress_validation_fn is None:
                            stress_report = _stress_validation_report(
                                run, bundle_root, start_ns=manifest.validation.score_start_ns,
                                end_ns=manifest.validation.score_end_ns, coverage_policy=coverage_policy,
                            )
                        else:
                            stress_report = stress_validation_fn(
                                run, bundle_root, start_ns=manifest.validation.score_start_ns,
                                end_ns=manifest.validation.score_end_ns, coverage_policy=coverage_policy,
                            )
                        if not isinstance(stress_report, dict) or stress_report.get("execution_mode") != "stress":
                            raise DataError("stress Validation evaluator must report execution_mode=stress")
                        stress_root.mkdir(parents=True)
                        stress_report_hash = _write_verified(stress_report_path, stress_report)
                        _write_verified(stress_receipt_path, {
                            "schema": "rles.wfo-stress-validation.v1", "protocol_id": protocol_id,
                            "fold_id": fold_id, "candidate_id": candidate_id,
                            "bundle_sha256": _verify_file(bundle_root / "bundle.json"),
                            "report_sha256": stress_report_hash,
                            "start_ns": manifest.validation.score_start_ns,
                            "end_ns": manifest.validation.score_end_ns,
                            "mode": "stress",
                        })
                    stress_validation[candidate_id].append(stress_report)
            # Feature profiles are candidate-specific; bind each scaler to its
            # own Train window and feature identity before selection.
            for identity in candidate_ids:
                bound_manifests[identity].append(bind_fold_scaler(
                    candidate_manifests[identity][fold_index],
                    _scaler(bundles[(fold_index, identity)]),
                ))

        selection_spec = SelectionSpec(**settings["selection"])
        aggregate = [_aggregate_candidate(identity, validation[identity], selection_spec.primary_metric)
                     for identity in candidate_ids]
        stress_gates = ({identity: _stress_validation_gate(stress_validation[identity], selection_spec)
                         for identity in candidate_ids} if cost_stress_validation else {})
        stress_validation_artifacts = []
        if cost_stress_validation:
            for fold_index, _manifest in enumerate(manifests):
                fold_id = f"{scheme.replace('/', '-')}-{fold_index:04d}"
                for candidate_id in candidate_ids:
                    stress_report_path = (root / "folds" / fold_id / "stress-validation"
                                          / candidate_id / "report.json")
                    stress_validation_artifacts.append({
                        "fold_id": fold_id,
                        "candidate_id": candidate_id,
                        "report_path": str(stress_report_path.relative_to(root)),
                        "report_sha256": _verify_file(stress_report_path),
                        "gate": stress_gates[candidate_id],
                    })
        if cost_stress_validation:
            stress_rejected = {
                identity: "STRESS_VALIDATION_GATE"
                for identity, result in stress_gates.items() if not result["passed"]
            }
            eligible = [item for item in aggregate if item["candidate_id"] not in stress_rejected]
            if eligible:
                selected = choose(selection_spec, eligible)
                selected["rejected"] = {**selected.get("rejected", {}), **stress_rejected}
            else:
                selected = {
                    "status": "NO_FEASIBLE_MODEL", "rejected": stress_rejected,
                    "selection_spec_sha256": selection_spec.sha256(),
                }
        else:
            selected = choose(selection_spec, aggregate)
        if selected["status"] != "SELECTED":
            validation_artifacts = []
            for fold_index, _manifest in enumerate(manifests):
                fold_id = f"{scheme.replace('/', '-')}-{fold_index:04d}"
                for candidate_id in candidate_ids:
                    candidate_root = bundles[(fold_index, candidate_id)]
                    report_path = candidate_root / "report.json"
                    metrics = validation[candidate_id][fold_index].get("metrics", {})
                    validation_artifacts.append({
                        "fold_id": fold_id,
                        "candidate_id": candidate_id,
                        "report_path": str(report_path.relative_to(root)),
                        "report_sha256": _verify_file(report_path),
                        "metrics": {
                            name: metrics.get(name) for name in (
                                "net_return", "max_drawdown", "turnover_ratio", "sharpe", "sortino",
                            )
                        },
                        "liquidations": validation[candidate_id][fold_index].get("liquidations"),
                        "risk_hard_interventions": validation[candidate_id][fold_index].get(
                            "risk_hard_interventions"
                        ),
                    })
            ledger.abort(
                protocol_id,
                error_type="NoValidationCandidate",
                detail="WFO produced no feasible validation-selected candidate",
            )
            _write_verified(root / "wfo-report.json", {
                "schema": "rles.wfo-report.v1",
                "protocol_id": protocol_id,
                "dataset_hash": store.snapshot_hash,
                "feature_hash": protocol_feature_hash,
                "candidate_feature_hashes": candidate_feature_hashes,
                "fold_manifests": {
                    identity: [asdict(value) for value in values]
                    for identity, values in bound_manifests.items()
                },
                "validation_candidates": aggregate,
                "selection": selected,
                "fold_validation_artifacts": validation_artifacts,
                "cost_stress_validation": cost_stress_validation,
                "stress_validation_artifacts": stress_validation_artifacts,
                "stress_validation_candidates": stress_gates,
                "fold_tests": [],
                "baselines": [],
                "final_retrain": None,
                "holdout": None,
                "holdout_state": "LOCKED",
                "conclusion": "NO_VALIDATION_CANDIDATE",
                "ledger": ledger.evidence_summary(protocol_id),
            })
            raise DataError("WFO produced no feasible validation-selected candidate")
        selected_id = selected["candidate_id"]
        selected_values = next(values for values in candidates if values["candidate_id"] == selected_id)
        selected_bound_manifests = bound_manifests[selected_id]
        frozen_recipe = digest({"candidate": selected_values, "selection": asdict(selection_spec),
                                "folds": [asdict(value) for value in selected_bound_manifests],
                                "candidate_feature_hashes": candidate_feature_hashes,
                                "cost_stress_validation": cost_stress_validation,
                                "stress_validation_candidates": stress_gates,
                                "stress_validation_artifacts": stress_validation_artifacts})
        if _source_tree_hash(source_root) != source_hash:
            raise DataError("source tree changed during WFO Validation")
        selection_payload = {
            "schema": "rles.wfo-selection.v1",
            "protocol_id": protocol_id,
            "candidate_id": selected_id,
            "candidate": selected_values,
            "selection": selected,
            "selection_spec": asdict(selection_spec),
            "validation_candidates": aggregate,
            "cost_stress_validation": cost_stress_validation,
            "stress_validation_candidates": stress_gates,
            "stress_validation_artifacts": stress_validation_artifacts,
            "fold_manifests": [asdict(value) for value in selected_bound_manifests],
            "candidate_feature_hashes": candidate_feature_hashes,
            "recipe_sha": frozen_recipe,
            "code_sha": source_hash,
        }
        selection_path = root / "selection.json"
        if selection_path.exists():
            _verify_file(selection_path)
            saved_selection = json.loads(selection_path.read_text(encoding="utf-8"))
            if saved_selection != selection_payload:
                raise DataError("WFO frozen selection artifact differs from Validation evidence")
        else:
            _write_verified(selection_path, selection_payload)
        if protocol_state == "DEVELOPMENT":
            ledger.freeze_recipe(protocol_id, recipe_sha=frozen_recipe,
                                 code_sha=source_hash,
                                 selection_spec_sha=selection_spec.sha256())
            protocol_state = "RECIPE_FROZEN"
        else:
            frozen = ledger.state(protocol_id)
            if (frozen["state"] != protocol_state or frozen["state"] not in {
                    "RECIPE_FROZEN", "RETRAINED", "HOLDOUT_OPENED", "CLOSED",
            } or frozen["recipe_sha"] != frozen_recipe
                    or frozen["code_sha"] != source_hash
                    or frozen["selection_spec_sha"] != selection_spec.sha256()):
                raise DataError("WFO frozen ledger differs from persisted Validation selection")

        tests = []
        baselines = []
        for fold_index, manifest in enumerate(manifests):
            fold_id = f"{scheme.replace('/', '-')}-{fold_index:04d}"
            manifest = candidate_manifests[selected_id][fold_index]
            bundle_root = bundles[(fold_index, selected_id)]
            run = _candidate(config, selected_values, bundle_root)
            run["environment"] = json.loads((bundle_root / "bundle.json").read_text(encoding="utf-8"))["environment"]
            run["evaluation"] = {
                "start_ns": manifest.test.score_start_ns, "end_ns": manifest.test.score_end_ns,
                "split": "test", "coverage_policy": _coverage(
                    template, instrument, manifest.test.score_start_ns, manifest.test.score_end_ns),
            }
            query_id = f"test-{fold_index}-{selected_id}"
            test_root = root / "folds" / fold_id / "test"
            report_path = test_root / "report.json"
            baseline_path = test_root / "baselines.json"
            existing_query = ledger.query(protocol_id, query_id)
            if existing_query is not None:
                if (not resume or existing_query != {
                        "query_id": query_id, "kind": "test",
                        "candidate": selected_id, "fold_id": fold_id,
                }):
                    raise DataError("WFO Test reservation differs from the frozen fold/candidate")
                if not test_root.is_dir():
                    raise DataError("WFO Test was reserved without a durable report; one-shot Test cannot be replayed")
                _verify_file(report_path)
                report = json.loads(report_path.read_text(encoding="utf-8"))
            else:
                if test_root.exists():
                    raise DataError("WFO Test artifacts exist without the durable one-shot query")
                ledger.record_query(protocol_id, query_id, "test", selected_id, fold_id)
                report = backtest_fn(run, bundle_root)
                test_root.mkdir(parents=True)
                _write_verified(report_path, report)
            tests.append(report)
            if baseline_path.exists():
                _verify_file(baseline_path)
                baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
            else:
                baseline = _baseline_reports(run, bundle_root, seed=int(run.get("seed", 31)))
                _write_verified(baseline_path, baseline)
            baselines.append(baseline)

        # Finish all fold-level statistical work before the one-shot final
        # Holdout is opened. A malformed or unidentifiable statistic therefore
        # cannot consume the Holdout merely to fail report assembly afterward.
        test_returns = [value for report in tests for value in _returns(report)]
        validation_returns = {
            identity: [value for report in validation[identity] for value in _returns(report)]
            for identity in candidate_ids
        }
        bootstrap_cfg = settings["bootstrap"]
        bootstrap = block_bootstrap(test_returns, statistic=statistics.fmean, **bootstrap_cfg)
        trial_sharpes = [_period_sharpe(validation_returns[identity]) for identity in candidate_ids]
        dsr = deflated_sharpe(test_returns, effective_trials=len(candidate_ids),
                              trial_sharpes=trial_sharpes)
        pbo = None
        if len(manifests) >= 4 and len(manifests) % 2 == 0:
            sliced = [[_returns(report) for report in validation[identity]] for identity in candidate_ids]
            pbo = pbo_cscv(sliced, candidate_ids=candidate_ids,
                           candidate_set_hash=candidate_set_sha256(candidate_ids),
                           max_partitions=settings["pbo_max_partitions"])

        # Retrain the frozen winner on all pre-Holdout observations. No
        # Validation/Test factory is constructed in this phase.
        final_root = root / "final-retrain"
        final_run = _candidate(config, selected_values, final_root)
        final_run["environment"] = _with_window(
            final_run["environment"], start_ns=settings["start_ns"], end_ns=holdout_start,
            coverage_policy=_coverage(template, instrument, settings["start_ns"], holdout_start),
        )
        final_run.pop("evaluation", None)
        if _source_tree_hash(source_root) != source_hash:
            raise DataError("source tree changed after WFO recipe freeze")
        final_bundle_path = final_root / "bundle.json"
        final_receipt_path = root / "final-retrain.json"
        final = {"output": str(final_root)}

        def verify_final_retrain() -> str:
            """Validate a completed final-train artifact before it can open Holdout."""
            bundle_hash = _verify_file(final_bundle_path)
            bundle = json.loads(final_bundle_path.read_text(encoding="utf-8"))
            if digest(bundle.get("environment")) != digest(final_run["environment"]):
                raise DataError("final retrain bundle does not match the frozen pre-Holdout run")
            if bundle.get("training_end_ns") != holdout_start:
                raise DataError("final retrain bundle ends outside the pre-Holdout window")
            if (not isinstance(bundle.get("actor"), dict)
                    or not isinstance(bundle.get("lineage"), list)):
                raise DataError("final retrain bundle lacks actor or stage lineage")
            app_receipt_path = final_root / "COMPLETED.json"
            app_receipt = json.loads(app_receipt_path.read_text(encoding="utf-8"))
            if (app_receipt.get("bundle_sha256") != bundle_hash
                    or app_receipt.get("route") != final_run.get("route")
                    or app_receipt.get("algorithm") != final_run.get("algorithm")):
                raise DataError("final retrain completion receipt does not match its bundle")
            run_manifest_path = final_root / "run.json"
            run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
            recipe = {key: value for key, value in final_run.items()
                      if key not in {"output", "resume_checkpoint"}}
            if (run_manifest.get("schema") != "rles.historical-run.v1"
                    or run_manifest.get("recipe_hash") != digest(recipe)
                    or run_manifest.get("attempt_id") != run_manifest.get("run_id")):
                raise DataError("final retrain run manifest differs from the frozen run recipe")
            if final_receipt_path.exists():
                _verify_file(final_receipt_path)
                saved = json.loads(final_receipt_path.read_text(encoding="utf-8"))
                if saved != {
                    "schema": "rles.wfo-final-retrain.v1",
                    "protocol_id": protocol_id,
                    "recipe_sha": frozen_recipe,
                    "source_sha": source_hash,
                    "dataset_sha": store.snapshot_hash,
                    "bundle_sha256": bundle_hash,
                }:
                    raise DataError("final retrain receipt differs from the frozen WFO identity")
            else:
                _write_verified(final_receipt_path, {
                    "schema": "rles.wfo-final-retrain.v1",
                    "protocol_id": protocol_id,
                    "recipe_sha": frozen_recipe,
                    "source_sha": source_hash,
                    "dataset_sha": store.snapshot_hash,
                    "bundle_sha256": bundle_hash,
                })
            return bundle_hash

        if protocol_state == "RECIPE_FROZEN":
            if final_root.exists():
                application_completed = all(
                    (final_root / name).is_file()
                    for name in ("bundle.json", "bundle.json.sha256", "COMPLETED.json", "run.json")
                )
                if application_completed:
                    retrain_hash = verify_final_retrain()
                else:
                    checkpoint = _latest_final_retrain_checkpoint(
                        final_root, str(final_run.get("route", "")),
                    )
                    if checkpoint is None:
                        raise DataError(
                            "partial final retrain has no completed stage checkpoint; "
                            "start a new WFO protocol"
                        )
                    resumed_run = {**final_run, "resume_checkpoint": str(checkpoint)}
                    final = train_fn(resumed_run, evaluate=False)
                    if _source_tree_hash(source_root) != source_hash:
                        raise DataError("source tree changed while resuming final pre-Holdout retraining")
                    retrain_hash = verify_final_retrain()
            else:
                final = train_fn(final_run, evaluate=False)
                if _source_tree_hash(source_root) != source_hash:
                    raise DataError("source tree changed during final pre-Holdout retraining")
                retrain_hash = verify_final_retrain()
            ledger.register_retrain(protocol_id, retrain_hash, pre_holdout_end_ns=holdout_start)
            protocol_state = "RETRAINED"
        else:
            frozen = ledger.state(protocol_id)
            retrain_hash = verify_final_retrain()
            if frozen["retrain_sha"] != retrain_hash:
                raise DataError("registered final retrain hash differs from its durable bundle")

        holdout_dataset_hash = digest([store.snapshot_hash, holdout_start, holdout_end])
        holdout_was_open = protocol_state in {"HOLDOUT_OPENED", "CLOSED"}
        if protocol_state == "RETRAINED":
            ledger.open_holdout(protocol_id, holdout_dataset_sha=holdout_dataset_hash,
                                reviewer_approval_sha=settings["reviewer_approval_sha"],
                                holdout_start_ns=holdout_start)
            protocol_state = "HOLDOUT_OPENED"
        elif protocol_state not in {"HOLDOUT_OPENED", "CLOSED"}:
            raise DataError("WFO ledger is not ready to read the one-shot Holdout")
        final_run["evaluation"] = {
            "start_ns": holdout_start, "end_ns": holdout_end, "split": "holdout",
            "coverage_policy": _coverage(template, instrument, holdout_start, holdout_end),
        }
        holdout_path = root / "holdout-report.json"
        if holdout_path.exists():
            if protocol_state not in {"HOLDOUT_OPENED", "CLOSED"}:
                raise DataError("Holdout report exists without a durable one-shot ledger reservation")
            holdout_hash = _verify_file(holdout_path)
            holdout = json.loads(holdout_path.read_text(encoding="utf-8"))
        else:
            if protocol_state == "CLOSED":
                raise DataError("closed WFO is missing its durable Holdout report")
            if holdout_was_open and protocol_state == "HOLDOUT_OPENED":
                raise DataError("one-shot Holdout was opened without a durable report; refusing to replay")
            holdout = backtest_fn(final_run, final["output"])
            holdout_hash = _write_verified(holdout_path, holdout)
        ledger.record_holdout_report(protocol_id, holdout_hash)
        constraints_pass = _holdout_constraints_pass(holdout, selection_spec)
        holdout_return, strategy_return_pass = _holdout_return_gate(holdout, selection_spec)
        conclusion = "PASS" if constraints_pass and strategy_return_pass else "FAIL"
        if protocol_state == "HOLDOUT_OPENED":
            ledger.close_holdout(protocol_id, report_sha=holdout_hash,
                                 conclusion=conclusion)
            protocol_state = "CLOSED"
        report = {
            "schema": "rles.wfo-report.v1", "protocol_id": protocol_id,
            "dataset_hash": store.snapshot_hash, "feature_hash": protocol_feature_hash,
            "candidate_feature_hashes": candidate_feature_hashes,
            "fold_manifests": [asdict(value) for value in selected_bound_manifests],
            "validation_candidates": aggregate, "selection": selected,
            "cost_stress_validation": cost_stress_validation,
            "stress_validation_candidates": stress_gates,
            "stress_validation_artifacts": stress_validation_artifacts,
            "fold_tests": tests, "baselines": baselines,
            "bootstrap": bootstrap, "deflated_sharpe": dsr, "pbo": pbo,
            "final_retrain": final, "holdout": holdout,
            "operational_constraints_pass": constraints_pass,
            "holdout_return_gate": {
                "minimum_net_return": selection_spec.min_median_fold_net_return,
                "observed_net_return": holdout_return,
                "passed": strategy_return_pass,
            },
            "conclusion": conclusion,
            "ledger": ledger.evidence_summary(protocol_id),
        }
        _write_verified(root / "wfo-report.json", report)
        return report
    except BaseException as exc:
        detail = str(exc)[:2000] or "interrupted"
        try:
            protocol_state = ledger.state(protocol_id)["state"]
            if isinstance(exc, KeyboardInterrupt) and protocol_state in {
                    "DEVELOPMENT", "RECIPE_FROZEN", "RETRAINED", "HOLDOUT_OPENED",
            }:
                ledger.suspend(protocol_id, detail=detail)
            elif protocol_state not in {"CLOSED", "ABORTED", "SUSPENDED"}:
                ledger.abort(protocol_id, error_type=type(exc).__name__, detail=detail)
        except Exception:
            # Preserve the original WFO failure if the research database itself
            # is unavailable; FAILED.json remains the filesystem receipt.
            pass
        failure = root / "FAILED.json"
        if not failure.exists():
            failure.write_text(json.dumps({"error": type(exc).__name__, "detail": detail},
                                          ensure_ascii=False), encoding="utf-8")
        raise
