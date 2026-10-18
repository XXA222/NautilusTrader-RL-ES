"""Paired Train-inner R1-to-E2 native economics on one frozen A2 actor."""
from __future__ import annotations

import argparse
import ctypes
from hashlib import sha256
import json
import os
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BUNDLE = (ROOT / ".artifacts/audit-development-20261017/m12-three-objective"
                  / "constraint_priority")
DEFAULT_OUTPUT = ROOT / ".artifacts/audit-development-20261017/m13-stage-contribution-v1"
DEFAULT_REGISTRATION = ROOT / "docs/audit_evidence/20261017/m13_stage_contribution/preregistration.json"
MIN_FREE_BYTES = 2 * 1024**3
SOURCE_FILES = (
    "packages/orchestrator/src/rles_orchestrator/application.py",
    "packages/gym_bridge/src/rles_gym_bridge/env.py",
    "packages/gym_bridge/src/rles_gym_bridge/historical.py",
    "packages/nautilus_adapter/src/rles_nautilus_adapter/session.py",
    "packages/nautilus_adapter/src/rles_nautilus_adapter/perpetual.py",
    "packages/research/src/rles_research/native_metrics.py",
    "packages/rl_sb3/src/rles_rl_sb3/actor.py",
    "packages/risk/src/rles_risk/guards.py",
)


def _sha(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _available_memory_bytes() -> int:
    if os.name == "nt":
        class MemoryStatus(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

        status = MemoryStatus()
        status.dwLength = ctypes.sizeof(status)
        if not getattr(ctypes, "windll").kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            raise OSError("GlobalMemoryStatusEx failed")
        return int(status.ullAvailPhys)
    sysconf = getattr(os, "sysconf", None)
    if not callable(sysconf):
        raise OSError("platform does not expose available physical memory")
    return int(sysconf("SC_AVPHYS_PAGES") * sysconf("SC_PAGE_SIZE"))


def _json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
                    encoding="utf-8")


def _load_bundle(bundle_root: Path) -> tuple[dict, str]:
    bundle_path = bundle_root / "bundle.json"
    marker = bundle_path.with_suffix(".json.sha256")
    actual = _sha(bundle_path)
    if not marker.is_file() or marker.read_text(encoding="ascii").strip() != actual:
        raise SystemExit("bundle identity check failed")
    return json.loads(bundle_path.read_text(encoding="utf-8")), actual


def _preregister(bundle_root: Path, output: Path, registration: Path) -> dict:
    bundle_root = bundle_root.resolve()
    output = output.resolve()
    if output.exists() and any(output.iterdir()):
        raise SystemExit("M1.3 output path must be new or empty before preregistration")
    bundle, bundle_sha = _load_bundle(bundle_root)
    env = bundle["environment"]
    actor = bundle["actor"]
    actor_root = bundle_root / actor["path"]
    actor_hash = (actor_root / "COMPLETED").read_text(encoding="ascii").strip()
    if actor_hash != actor["actor_hash"]:
        raise SystemExit("R1 inference actor receipt mismatch")
    manifest_path = Path(env["market_path"]).parent / "import.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") not in {"rles.binance-archive.v1", "rles.binance-archive-range.v1"}:
        raise SystemExit("Train-inner fixture import manifest schema mismatch")
    for entry in manifest["sources"]:
        if _sha(manifest_path.parent / entry["path"]) != entry["sha256"]:
            raise SystemExit(f"source fixture checksum mismatch: {entry['path']}")
    selected = bundle["soft_multipliers"]
    baseline = {name: 1.0 for name in selected}
    if set(selected) != {"exposure_multiplier", "drawdown_throttle"}:
        raise SystemExit("M1.3 expects the current two-dimensional E2 schema")
    registration_payload = {
        "schema": "rles.m13-stage-contribution-preregistration.v1",
        "status": "frozen_before_native_replay",
        "stage_pair": "A2 R1 neutral SoftRisk -> selected E2 SoftRisk",
        "question": "What native economic change is associated with applying the frozen E2 SoftRisk multipliers to the exact frozen R1 actor?",
        "interpretation_limit": "single reused Train-inner window with scenario-only execution rules; descriptive stage diagnostic, not an independent strategy result",
        "split": "train_inner",
        "formal_evaluation_queries": {"validation": 0, "test": 0, "holdout": 0},
        "no_retraining_or_selection": True,
        "seed": 31,
        "modes": ["base", "stress"],
        "bundle_root": str(bundle_root),
        "bundle_sha256": bundle_sha,
        "run_id": json.loads((bundle_root / "COMPLETED.json").read_text(encoding="utf-8"))["run_id"],
        "actor_sha256": actor_hash,
        "lineage": bundle["lineage"],
        "training_end_ns": bundle["training_end_ns"],
        "window": {key: env[key] for key in ("start_ns", "end_ns")},
        "market_path": env["market_path"],
        "market_sha256": _sha(Path(env["market_path"])),
        "rules_path": env["rules_path"],
        "rules_sha256": _sha(Path(env["rules_path"])),
        "fixture_manifest_path": str(manifest_path.resolve()),
        "fixture_manifest_sha256": _sha(manifest_path),
        "execution": env["execution"],
        "hard_risk": env["hard_risk"],
        "soft_risk": env["soft_risk"],
        "starting_balance": env["starting_balance"],
        "risk_budget_notional": env["risk_budget_notional"],
        "r1_neutral_multipliers": baseline,
        "selected_e2_multipliers": selected,
        "primary_outcome": "native final equity difference from same initial capital, reported with fill/fee/funding/drawdown/turnover deltas",
        "resource_stop_free_bytes": MIN_FREE_BYTES,
        "output_root": str(output),
        "runner_sha256": _sha(Path(__file__).resolve()),
        "source_sha256": {relative: _sha(ROOT / relative) for relative in SOURCE_FILES},
    }
    if registration.exists():
        previous = json.loads(registration.read_text(encoding="utf-8"))
        if previous != registration_payload:
            raise SystemExit("frozen M1.3 registration already exists with different inputs")
    else:
        _json(registration, registration_payload)
        registration.with_suffix(registration.suffix + ".sha256").write_text(
            _sha(registration), encoding="ascii",
        )
    return registration_payload


def _verify_registration(registration: Path, output: Path) -> dict:
    marker = registration.with_suffix(registration.suffix + ".sha256")
    if not registration.is_file() or not marker.is_file() or marker.read_text(encoding="ascii").strip() != _sha(registration):
        raise SystemExit("M1.3 preregistration receipt mismatch")
    frozen = json.loads(registration.read_text(encoding="utf-8"))
    if _sha(Path(__file__).resolve()) != frozen["runner_sha256"]:
        raise SystemExit("M1.3 runner changed after preregistration")
    for relative, expected in frozen["source_sha256"].items():
        if _sha(ROOT / relative) != expected:
            raise SystemExit(f"M1.3 frozen source changed: {relative}")
    if output.resolve() != Path(frozen["output_root"]):
        raise SystemExit("M1.3 output path differs from preregistration")
    return frozen


def _run(registration: Path, output: Path) -> dict:
    frozen = _verify_registration(registration, output)
    if output.exists() and any(output.iterdir()):
        raise SystemExit("M1.3 output path must be empty before the frozen replay")
    bundle_root = Path(frozen["bundle_root"])
    bundle, bundle_sha = _load_bundle(bundle_root)
    if bundle_sha != frozen["bundle_sha256"]:
        raise SystemExit("M1.3 frozen bundle changed")
    if (_sha(Path(frozen["market_path"])) != frozen["market_sha256"]
            or _sha(Path(frozen["rules_path"])) != frozen["rules_sha256"]
            or _sha(Path(frozen["fixture_manifest_path"])) != frozen["fixture_manifest_sha256"]):
        raise SystemExit("M1.3 frozen market, rules, or fixture manifest changed")
    if _available_memory_bytes() < frozen["resource_stop_free_bytes"]:
        raise SystemExit("BLOCKED: available physical memory is below the frozen 2 GiB stop line")

    from rles_data.features import Scaler
    from rles_gym_bridge.historical import HistoricalEnvironmentFactory
    from rles_rl_sb3.actor import InferenceActor
    from rles_orchestrator.application import rollout
    from rles_orchestrator.research_access import record_development

    registry_config = {"output": str(output.resolve()), "environment": bundle["environment"]}
    record_development(registry_config,
                       {"start_ns": frozen["window"]["start_ns"],
                        "end_ns": frozen["window"]["end_ns"]},
                       label="m13-stage-contribution-train-inner")
    raw_scaler = dict(bundle["preprocessing"]["scaler"])
    raw_scaler["mean"] = tuple(raw_scaler["mean"])
    raw_scaler["std"] = tuple(raw_scaler["std"])
    scaler = Scaler(**raw_scaler)
    actor = bundle["actor"]
    inference = InferenceActor(bundle_root / actor["path"], verify=True)
    reports: dict[str, dict] = {}
    for mode in frozen["modes"]:
        if _available_memory_bytes() < frozen["resource_stop_free_bytes"]:
            raise SystemExit(f"BLOCKED: memory stop line reached before {mode} replay")
        factory = HistoricalEnvironmentFactory(
            **{**bundle["environment"], "split": "train_inner", "fit_scaler": False,
               "scaler": scaler, "mode": mode, "compact_report_snapshots": True},
        )
        pair = {}
        for stage, multipliers in (
            ("R1_neutral", frozen["r1_neutral_multipliers"]),
            ("E2_selected", frozen["selected_e2_multipliers"]),
        ):
            if _available_memory_bytes() < frozen["resource_stop_free_bytes"]:
                raise SystemExit(f"BLOCKED: memory stop line reached before {mode}/{stage} replay")
            report, _ = rollout(
                factory, inference.predict, seed=frozen["seed"],
                soft_parameters=multipliers, collect_observations=False,
                include_account_rows=True, include_execution_trace=True,
            )
            report["stage_label"] = stage
            report["soft_multipliers"] = multipliers
            pair[stage] = report
        base_rows, e2_rows = pair["R1_neutral"]["account_rows"], pair["E2_selected"]["account_rows"]
        base_by_ts = {row["ts_ns"]: row for row in base_rows}
        e2_by_ts = {row["ts_ns"]: row for row in e2_rows}
        if len(base_by_ts) != len(base_rows) or len(e2_by_ts) != len(e2_rows):
            raise SystemExit(f"{mode}: duplicate timestamp in native equity ledger")
        if set(base_by_ts) != set(e2_by_ts):
            raise SystemExit(f"{mode}: paired policies have different native decision timestamps")
        pair_summary = {
            "mode": mode,
            "decision_rows": len(base_by_ts),
            "same_native_timestamps": True,
            "initial_equity_r1": base_rows[0]["equity"],
            "initial_equity_e2": e2_rows[0]["equity"],
            "final_equity_r1": base_rows[-1]["equity"],
            "final_equity_e2": e2_rows[-1]["equity"],
            "native_equity_delta_e2_minus_r1": e2_rows[-1]["equity"] - base_rows[-1]["equity"],
            "metrics_r1": pair["R1_neutral"]["metrics"],
            "metrics_e2": pair["E2_selected"]["metrics"],
            "fills_r1": len(pair["R1_neutral"]["fills"]),
            "fills_e2": len(pair["E2_selected"]["fills"]),
            "execution_diagnostics_r1": pair["R1_neutral"]["execution_decision_diagnostics"],
            "execution_diagnostics_e2": pair["E2_selected"]["execution_decision_diagnostics"],
        }
        pair_summary["account_rows"] = [
            {"ts_ns": timestamp, "equity_r1": base_by_ts[timestamp]["equity"],
             "equity_e2": e2_by_ts[timestamp]["equity"],
             "equity_delta_e2_minus_r1": e2_by_ts[timestamp]["equity"] - base_by_ts[timestamp]["equity"],
             "fee_r1": base_by_ts[timestamp]["fee"], "fee_e2": e2_by_ts[timestamp]["fee"],
             "funding_r1": base_by_ts[timestamp]["funding"], "funding_e2": e2_by_ts[timestamp]["funding"],
             "slippage_cost_r1": base_by_ts[timestamp]["slippage_cost"],
             "slippage_cost_e2": e2_by_ts[timestamp]["slippage_cost"]}
            for timestamp in sorted(base_by_ts)
        ]
        reports[mode] = {"summary": pair_summary,
                         "r1_neutral": pair["R1_neutral"], "e2_selected": pair["E2_selected"]}
        del factory
    output.mkdir(parents=True, exist_ok=True)
    for mode, payload in reports.items():
        _json(output / f"{mode}.json", payload)
    summary = {
        "schema": "rles.m13-stage-contribution-analysis.v1",
        "status": "completed_train_inner_r1_to_e2_diagnostic",
        "preregistration_sha256": _sha(registration),
        "evaluation_queries": frozen["formal_evaluation_queries"],
        "scope": frozen["interpretation_limit"],
        "paired_stage_comparison": "same frozen R1 actor, same initial capital, same native environment recipe, same data timestamps; only E2 SoftRisk multipliers vary",
        "modes": {mode: value["summary"] for mode, value in reports.items()},
        "actor_sha256": frozen["actor_sha256"],
        "bundle_sha256": frozen["bundle_sha256"],
        "selected_e2_multipliers": frozen["selected_e2_multipliers"],
        "r1_neutral_multipliers": frozen["r1_neutral_multipliers"],
    }
    _json(output / "summary.json", summary)
    summary_path = ROOT / "docs/audit_evidence/20261017/m13_stage_contribution/analysis.json"
    _json(summary_path, summary)
    summary_path.with_suffix(".json.sha256").write_text(_sha(summary_path), encoding="ascii")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--registration", type=Path, default=DEFAULT_REGISTRATION)
    parser.add_argument("--preregister", action="store_true")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    bundle = args.bundle if args.bundle.is_absolute() else ROOT / args.bundle
    output = args.output if args.output.is_absolute() else ROOT / args.output
    registration = args.registration if args.registration.is_absolute() else ROOT / args.registration
    if args.preregister == args.run:
        raise SystemExit("select exactly one of --preregister or --run")
    if args.preregister:
        result = _preregister(bundle, output, registration)
    else:
        result = _run(registration, output)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
