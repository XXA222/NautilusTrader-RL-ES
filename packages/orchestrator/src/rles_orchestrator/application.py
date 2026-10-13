"""Configuration-driven training and native historical backtesting application."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from decimal import Decimal
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
from typing import Any, Callable, Literal, cast
import uuid

import numpy as np

from rles_compute_runtime.allocation import ComputeAllocation
from rles_compute_runtime.device import DeviceResolver
from rles_compute_runtime.errors import DeviceUnavailable
from rles_compute_runtime.resources import ResourceQuota
from rles_data.core import DataError, MINUTE_NS, digest
from rles_data.coverage import CoveragePolicy
from rles_data.features import Scaler
from rles_data.io import read_curated
from rles_es_evotorch.actor import ActorGenomeSpec, act, genome_hash
from rles_es_evotorch.fitness import FitnessSpec
from rles_es_evotorch.provider import EvolutionConfig
from rles_es_evotorch.soft_risk import SoftRiskDimension, SoftRiskSpace
from rles_execution import resolve_scenario
from rles_gym_bridge.historical import HistoricalEnvironmentFactory
from rles_research.ledger import ResearchLedger, SelectionLedger
from rles_research.native_metrics import native_report
from rles_rl_sb3.actor import InferenceActor
from rles_rl_sb3.config import PolicySpec, RLConfig, SeedSpec, TransitionFingerprint

from .es_stage import ESStageExecutor
from .diagnostics import ExecutionAttribution
from .rl_stage import RLStageExecutor
from .workflow import Orchestrator, ProviderBackedStageRunner, Route, RunSpec, STAGES, StageBudget


def _write(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as target:
        json.dump(value, target, indent=2, ensure_ascii=False, allow_nan=False)


def _file_hash(path: Path) -> str:
    value = sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def _write_verified(path: Path, value: Any) -> str:
    _write(path, value)
    identity = _file_hash(path)
    path.with_suffix(path.suffix + ".sha256").write_text(identity, encoding="ascii")
    return identity


def _verify_file(path: Path) -> str:
    marker = path.with_suffix(path.suffix + ".sha256")
    if not path.is_file() or not marker.is_file():
        raise DataError(f"artifact payload or digest missing: {path.name}")
    expected = marker.read_text(encoding="ascii")
    actual = _file_hash(path)
    if expected != actual:
        raise DataError(f"artifact payload integrity mismatch: {path.name}")
    return actual


def _persist_backtest_report(config: dict, bundle_path: str | Path, report: dict, *,
                             train_diagnostic: bool = False) -> dict:
    """Publish an immutable, checksummed CLI backtest report artifact."""
    bundle_root = Path(bundle_path).resolve()
    bundle_sha = _verify_file(bundle_root / "bundle.json")
    output = _backtest_output_path(config)
    schema = "rles.train-diagnostic.v1" if train_diagnostic else "rles.backtest-cli.v1"
    output.mkdir(parents=True)
    try:
        report_sha = _write_verified(output / "report.json", report)
        _write(output / "COMPLETED.json", {
            "schema": schema,
            "bundle_sha256": bundle_sha,
            "report_sha256": report_sha,
        })
    except Exception as exc:
        _write(output / "FAILED.json", {"error": type(exc).__name__, "detail": str(exc)[:1000]})
        raise
    return {
        "schema": schema,
        "output": str(output),
        "bundle_sha256": bundle_sha,
        "report_sha256": report_sha,
        **({"diagnostic_scope": report["diagnostic_scope"]} if train_diagnostic
           else {"evaluation": config.get("evaluation")}),
        "metrics": report.get("metrics"),
        "terminal_flat": report.get("terminal_flat"),
        "liquidations": report.get("liquidations"),
    }


def _backtest_output_path(config: dict) -> Path:
    output = Path(config["output"]).resolve()
    if output.is_symlink() or output.exists():
        raise DataError("backtest output already exists; use a new immutable output directory")
    return output


def _json_identity(value: Any) -> str:
    """Return the identity used after an application recipe crosses JSON storage."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def load_config(path: str | Path) -> dict:
    source = Path(path).resolve()
    config = json.loads(source.read_text(encoding="utf-8"))
    allowed = {"environment", "evaluation", "route", "algorithm", "steps", "population", "generations",
               "seed", "hidden_layers", "normalize_observation", "hyperparameters", "output", "distillation",
               "checkpoint_interval_steps", "wfo", "resume_checkpoint", "rl_device", "es_input_scaling",
               "enforce_worker_memory_limit", "worker_ram_bytes"}
    if not isinstance(config, dict) or set(config) - allowed:
        raise DataError("unknown application configuration fields")
    for name in ("market_path", "rules_path"):
        p = Path(config["environment"][name])
        config["environment"][name] = str((p if p.is_absolute() else source.parent / p).resolve())
    output = Path(config["output"])
    config["output"] = str((output if output.is_absolute() else source.parent / output).resolve())
    if config.get("resume_checkpoint") is not None:
        checkpoint = Path(config["resume_checkpoint"])
        candidate = checkpoint if checkpoint.is_absolute() else source.parent / checkpoint
        if candidate.is_symlink() or candidate.parent.is_symlink():
            raise DataError("resume checkpoint and its directory must not be symlinks")
        config["resume_checkpoint"] = str(candidate.resolve())
    return config


def prepare(config: dict) -> HistoricalEnvironmentFactory:
    return HistoricalEnvironmentFactory(**{**config["environment"], "split": "train_inner", "fit_scaler": True})


def audit_market(config: dict) -> dict:
    """Audit immutable market facts even when separately sourced rules are absent."""
    environment = config["environment"]
    evaluation = config.get("evaluation")
    if not isinstance(evaluation, dict):
        raise DataError("market audit requires explicit training and evaluation windows")
    store = read_curated(environment["market_path"])
    instrument = environment["instrument"]
    splits = {}
    for name, values in (("train", environment), ("evaluation", evaluation)):
        if not {"start_ns", "end_ns", "coverage_policy"} <= set(values):
            raise DataError(f"{name} market audit window is incomplete")
        report = store.quality(
            start_ns=values["start_ns"], end_ns=values["end_ns"],
            expected_instruments=(instrument,),
            coverage_policy=CoveragePolicy(**values["coverage_policy"]),
        )
        splits[name] = asdict(report)
    return {"schema": "rles.market-audit.v1", "instrument": instrument,
            "snapshot_hash": store.snapshot_hash, "splits": splits}


def rollout(factory: HistoricalEnvironmentFactory, predict, *, seed: int,
            soft_parameters: dict | None = None,
            collect_observations: bool = True) -> tuple[dict, np.ndarray]:
    env = factory.make_env(soft_multipliers=soft_parameters)
    observations = []
    requested_actions = []
    soft_targets = []
    risk_targets = []
    executed_actions = []
    plan_actions: Counter[str] = Counter()
    plan_reasons: Counter[str] = Counter()
    missing_plans = 0
    interventions = hard_interventions = soft_adjustment_steps = 0
    soft_adjustment_total = hard_adjustment_total = total_adjustment = 0.0
    components_known = True
    attribution = ExecutionAttribution()
    try:
        observation, _ = env.reset(seed=seed)
        if collect_observations:
            observations.append(observation.copy())
        while True:
            action = np.asarray(predict(observation[None, :]), dtype=np.float32).reshape(1)
            observation, _, terminated, truncated, info = env.step(action)
            if collect_observations:
                observations.append(observation.copy())
            requested_action = float(info["requested_action"])
            executed_action = float(info["policy_executed_action_pre_terminal"])
            if not np.isfinite(requested_action) or not np.isfinite(executed_action):
                raise DataError("native rollout action diagnostics are not finite")
            requested_actions.append(requested_action)
            soft_target = info.get("risk_soft_target")
            risk_target = info.get("risk_adjusted_target")
            if (soft_target is None) != (risk_target is None):
                raise DataError("native intermediate and final risk targets must be reported together")
            if soft_target is not None:
                if not np.isfinite(soft_target) or not np.isfinite(risk_target):
                    raise DataError("native risk targets are not finite")
                soft_targets.append(float(soft_target))
                risk_targets.append(float(risk_target))
            executed_actions.append(executed_action)
            plan_action = info.get("execution_plan_action")
            plan_reason = info.get("execution_plan_reason")
            if plan_action is None and plan_reason is None:
                missing_plans += 1
            elif (plan_action not in {"SUBMIT", "NOOP", "WAIT", "CANCEL"}
                  or not isinstance(plan_reason, str) or not plan_reason):
                raise DataError("native execution plan diagnostics are invalid")
            else:
                plan_actions[plan_action] += 1
                plan_reasons[plan_reason] += 1
            interventions += int(float(info["risk_guard_intervention"]) > 0)
            soft_amount = float(info["risk_soft_adjustment"])
            hard_amount = float(info["risk_hard_adjustment"])
            known = info["risk_intervention_components_known"]
            if (not np.isfinite(soft_amount) or soft_amount < 0 or
                    not np.isfinite(hard_amount) or hard_amount < 0 or type(known) is not bool):
                raise DataError("native risk adjustment attribution is invalid")
            components_known = components_known and known
            attribution.observe(requested=requested_action, executed=executed_action,
                                soft=soft_target if known else None,
                                risk=risk_target if known else None,
                                action=plan_action, reason=plan_reason)
            soft_adjustment_steps += int(soft_amount > 0)
            hard_interventions += int(hard_amount > 0)
            soft_adjustment_total += soft_amount
            hard_adjustment_total += hard_amount
            total_adjustment += float(info["risk_guard_intervention"])
            if terminated or truncated:
                break
        report = native_report(
            factory.snapshots, fill_references=factory.fill_references,
            sample_interval_ns=MINUTE_NS,
            allow_price_improvement=factory.execution.fill(factory.mode).data_granularity == "quote",
        )
        report["risk_guard_interventions"] = interventions
        report["risk_soft_adjustment_steps"] = soft_adjustment_steps
        report["risk_hard_interventions"] = hard_interventions
        report["risk_soft_adjustment_total"] = soft_adjustment_total
        report["risk_hard_adjustment_total"] = hard_adjustment_total
        report["risk_total_adjustment"] = total_adjustment
        report["risk_intervention_components_known"] = components_known
        requested = np.asarray(requested_actions, dtype=np.float64)
        executed = np.asarray(executed_actions, dtype=np.float64)
        action_gap = np.abs(requested - executed)
        report["action_diagnostics"] = {
            "count": int(requested.size),
            "requested_mean": float(np.mean(requested)),
            "requested_abs_mean": float(np.mean(np.abs(requested))),
            "requested_abs_p95": float(np.quantile(np.abs(requested), 0.95)),
            "requested_nonzero_fraction": float(np.mean(np.abs(requested) > 1e-6)),
            "requested_short_fraction": float(np.mean(requested < -1e-6)),
            "requested_long_fraction": float(np.mean(requested > 1e-6)),
            "requested_std": float(np.std(requested)),
            "executed_mean": float(np.mean(executed)),
            "executed_abs_mean": float(np.mean(np.abs(executed))),
            "executed_abs_p95": float(np.quantile(np.abs(executed), 0.95)),
            "executed_nonzero_fraction": float(np.mean(np.abs(executed) > 1e-6)),
            "requested_executed_gap_abs_mean": float(np.mean(action_gap)),
            "requested_executed_gap_abs_p95": float(np.quantile(action_gap, 0.95)),
            "adjusted_fraction": float(np.mean(action_gap > 1e-6)),
        }
        if soft_targets and len(soft_targets) == len(requested) and len(risk_targets) == len(requested):
            soft = np.asarray(soft_targets, dtype=np.float64)
            risk = np.asarray(risk_targets, dtype=np.float64)
            report["risk_target_diagnostics"] = {
                "components_known": True,
                "soft_target_abs_mean": float(np.mean(np.abs(soft))),
                "risk_adjusted_target_abs_mean": float(np.mean(np.abs(risk))),
                "requested_to_soft_gap_abs_mean": float(np.mean(np.abs(requested - soft))),
                "soft_to_risk_gap_abs_mean": float(np.mean(np.abs(soft - risk))),
                "risk_to_executed_gap_abs_mean": float(np.mean(np.abs(risk - executed))),
            }
        else:
            report["risk_target_diagnostics"] = {"components_known": False}
        report["execution_plan_diagnostics"] = {
            "components_known": missing_plans == 0,
            "count": int(requested.size),
            "missing_count": missing_plans,
            "action_counts": dict(sorted(plan_actions.items())),
            "reason_counts": dict(sorted(plan_reasons.items())),
        }
        report["execution_attribution"] = attribution.report()
        return report, (np.stack(observations) if collect_observations
                        else np.empty((0, observation.size), dtype=np.float32))
    finally:
        env.close()


def _evaluation_factory(config: dict, training: HistoricalEnvironmentFactory) -> HistoricalEnvironmentFactory:
    evaluation = config.get("evaluation")
    if not isinstance(evaluation, dict) or not {"start_ns", "end_ns", "coverage_policy"} <= set(evaluation):
        raise DataError("explicit disjoint evaluation window and coverage policy required")
    if set(evaluation) - {"start_ns", "end_ns", "coverage_policy", "mode", "split"}:
        raise DataError("evaluation cannot hot-swap frozen training risk/execution/data recipe")
    if evaluation.get("split", "validation") != "validation":
        raise DataError("training may query only the declared Validation split")
    if evaluation["start_ns"] < training.end_ns:
        raise DataError("evaluation window overlaps training")
    return HistoricalEnvironmentFactory(**{**config["environment"], **evaluation, "split": "validation",
                                            "fit_scaler": False, "scaler": training.scaler})


def _resolve_rl_device(requested: Any, route: Route) -> Literal["cpu", "cuda"]:
    """Resolve the public R1 device before creating run output or dispatching work."""
    if requested not in {"cpu", "cuda"}:
        raise DataError("rl_device must be explicitly cpu or cuda")
    if requested == "cuda" and "R1" not in STAGES[route]:
        raise DataError(f"route {route.value} has no R1 learner; CUDA request is unsupported")
    try:
        return cast(Literal["cpu", "cuda"],
                    DeviceResolver().resolve("gpu" if requested == "cuda" else "cpu").learner_device)
    except DeviceUnavailable as exc:
        raise DataError(f"requested R1 device is unavailable: {exc}") from exc


def _training_actor_spec(training: HistoricalEnvironmentFactory, hidden: tuple[int, ...],
                         config: dict, route: Route) -> ActorGenomeSpec:
    """Freeze causal account-unit scales for E1 without altering Gym facts."""
    legacy_resume = False
    manifest_path = Path(config["output"]) / "run.json"
    if config.get("resume_checkpoint") is not None and manifest_path.is_file():
        saved = json.loads(manifest_path.read_text(encoding="utf-8"))
        legacy_resume = isinstance(saved, dict) and "actor_spec" not in saved
    scaling = config.get("es_input_scaling", "none" if legacy_resume or "E1" not in STAGES[route]
                         else "account_units")
    if not isinstance(scaling, str) or scaling not in {"none", "account_units"}:
        raise DataError("es_input_scaling must be none or account_units")
    if scaling == "none" or "E1" not in STAGES[route]:
        return ActorGenomeSpec(training.schema.names, hidden)
    marks = [event for event in training.store.visible(training.rows[0].decision_ts_ns, "mark")
             if event.payload.get("instrument") == training.instrument]
    if not marks:
        raise DataError("ES quantity input scale requires an as-of Train mark")
    reference_mark = Decimal(str(marks[-1].payload["price"]))
    money_fields = {"account.equity", "account.peak_equity", "account.cash", "account.free_margin",
                    "account.initial_margin", "account.maintenance_margin", "account.unrealized_pnl"}
    scales = tuple(float(1 / training.capital) if name in money_fields else
                   float(reference_mark / training.budget) if name == "account.position_quantity" else 1.0
                   for name in training.schema.names)
    return ActorGenomeSpec(training.schema.names, hidden, version=2, input_scale=scales)


def _worker_memory_policy(config: dict) -> tuple[bool, int, dict]:
    """Validate resource policy before allocating data, queries, or workers."""
    enforce_worker_memory_limit = config.get("enforce_worker_memory_limit", os.name == "nt")
    worker_ram_bytes = config.get("worker_ram_bytes", 2 * 1024**3)
    if (type(enforce_worker_memory_limit) is not bool or type(worker_ram_bytes) is not int
            or not 0 < worker_ram_bytes <= 4 * 1024**3):
        raise DataError("worker memory policy requires a Boolean flag and bytes within the 4 GiB run quota")
    if enforce_worker_memory_limit and os.name != "nt":
        raise DataError("Windows Job worker memory enforcement is unavailable on this platform")
    worker_resources = {"enforcement": "windows_job_commit_memory" if enforce_worker_memory_limit else "quota_only",
                        "ram_bytes_per_worker": worker_ram_bytes}
    return enforce_worker_memory_limit, worker_ram_bytes, worker_resources


def train(config: dict, *, evaluate: bool = True) -> dict:
    import torch

    torch.set_num_threads(1)
    route = Route(config.get("route", "A0"))
    rl_device = _resolve_rl_device(config.get("rl_device", "cpu"), route)
    enforce_worker_memory_limit, worker_ram_bytes, worker_resources = _worker_memory_policy(config)
    training = prepare(config)
    evaluation = _evaluation_factory(config, training) if evaluate else None
    algorithm = config.get("algorithm", "PPO")
    steps, population, generations = config.get("steps", 32), config.get("population", 4), config.get("generations", 1)
    seed = config.get("seed", 31)
    hidden = tuple(config.get("hidden_layers", (8,)))
    actor_spec = _training_actor_spec(training, hidden, config, route)
    canonical_ppo_output = actor_spec.version == 2 and algorithm == "PPO" and "E1" in STAGES[route]
    previous_manifest = Path(config["output"]) / "run.json"
    if config.get("resume_checkpoint") is not None and previous_manifest.is_file():
        previous = json.loads(previous_manifest.read_text(encoding="utf-8"))
        canonical_ppo_output = previous.get("canonical_ppo_output", False)
        if type(canonical_ppo_output) is not bool:
            raise DataError("saved PPO output bridge flag must be Boolean")
    policy_hidden = (*hidden, 1) if canonical_ppo_output else hidden
    risk_space = SoftRiskSpace(tuple(SoftRiskDimension(name, 0.1, 1.0)
                                     for name in ("exposure_multiplier", "drawdown_throttle", "volatility_throttle")))
    action_hash = digest({"target_net_exposure": [-1, 1], "shape": [1]})
    fingerprint = TransitionFingerprint(
        env_contract=digest("NautilusBarrierEnv-v0.4.0"), dataset=training.store.snapshot_hash,
        feature=training.features.hash, feature_scaler=training.scaler.hash, observation=training.schema.hash,
        action=action_hash, action_transform=digest("soft-hard-native-quantity"), execution=training.execution.fingerprint(),
        hard_risk=training.hard.signature(),
        soft_risk=digest({name: str(value) for name, value in asdict(training.soft).items()}),
        reward=digest({"lambda_dd": 0, "lambda_guard": 0}),
        normalization=digest({"VecNormalize": config.get("normalize_observation", False)}),
        engine=digest("nautilus_trader-2.0.0rc6"), adapter=digest("rles.perpetual-native-v2"),
        fold=digest([training.start_ns, training.end_ns]), terminal=digest("force_flatten_with_cost"),
    )
    hyperparameters = config.get("hyperparameters", {"n_steps": 8, "batch_size": 8, "n_epochs": 1} if algorithm == "PPO" else
                                {"learning_starts": 8, "buffer_size": 256, "batch_size": 8, "train_freq": 1, "gradient_steps": 1})
    rl_config = RLConfig(algorithm, PolicySpec(len(training.schema.names), hidden_layers=policy_hidden, value_layers=hidden,
                                             observation_schema_hash=training.schema.hash, action_schema_hash=action_hash),
                         fingerprint, SeedSpec(seed, (seed + 1,), seed + 2, seed + 3),
                         hyperparameters=hyperparameters, device=rl_device,
                         normalize_observation=config.get("normalize_observation", False))
    budgets = {stage: StageBudget(max_rl_steps=steps) if stage == "R1" else
               StageBudget(max_population=population, max_generations=generations, max_evaluations=population * generations,
                           max_lane_requests=population * generations) for stage in STAGES[route]}
    output = Path(config["output"])
    resume_checkpoint = None
    resume_stage = None
    resume_rl_checkpoint = None
    if config.get("resume_checkpoint") is not None:
        requested_checkpoint = Path(config["resume_checkpoint"])
        if requested_checkpoint.is_symlink() or requested_checkpoint.parent.is_symlink():
            raise DataError("resume checkpoint and its directory must not be symlinks")
        resume_checkpoint = requested_checkpoint.resolve()
        output_root = output.resolve()
        if (resume_checkpoint.parent == output_root / "R1-checkpoints"
                and resume_checkpoint.name.startswith("step-")
                and resume_checkpoint.name[5:].isdigit()):
            resume_stage = "R1"
        elif (resume_checkpoint.parent.parent == output_root / "E1"
              and resume_checkpoint.parent.name.startswith("generation-")
              and resume_checkpoint.parent.name[11:].isdigit()):
            # ESCheckpointStore writes the attempt inside a generation root.
            resume_stage = "E1"
        elif (resume_checkpoint.parent == output_root / "E1"
              and resume_checkpoint.name.startswith("generation-")
              and resume_checkpoint.name[11:].isdigit()):
            resume_stage = "E1"
        elif (resume_checkpoint.parent.parent == output_root / "E2"
              and resume_checkpoint.parent.name.startswith("generation-")
              and resume_checkpoint.parent.name[11:].isdigit()):
            resume_stage = "E2"
        elif (resume_checkpoint.parent == output_root / "E2"
              and resume_checkpoint.name.startswith("generation-")
              and resume_checkpoint.name[11:].isdigit()):
            resume_stage = "E2"
        else:
            raise DataError("resume checkpoint must be a completed E1/E2 generation or R1 boundary in this run")
    run_manifest_path = output / "run.json"
    run_recipe = {key: value for key, value in config.items()
                  if key not in {"output", "resume_checkpoint"}}
    run_recipe_hash = digest(run_recipe)
    saved_run = None
    if output.exists():
        resumable_routes = ({"R1": {Route.A0, Route.A2, Route.A3, Route.A4},
                             "E1": {Route.A1, Route.A3, Route.A4},
                             "E2": {Route.A2, Route.A4}})
        if (resume_checkpoint is None or resume_stage not in resumable_routes
                or route not in resumable_routes[resume_stage] or not run_manifest_path.is_file()):
            raise DataError("existing output requires a matching resumable stage checkpoint and run manifest")
        if resume_stage == "R1" and any((output / name).exists()
                                        for name in ("COMPLETED.json", "bundle.json")):
            raise DataError("completed or finalized R1 run cannot be resumed")
        if resume_stage == "E1" and any((output / name).exists()
                                        for name in ("COMPLETED.json", "bundle.json", "r1")):
            raise DataError("completed or finalized E1 run cannot be resumed")
        if resume_stage == "E2" and any((output / name).exists()
                                        for name in ("COMPLETED.json", "bundle.json")):
            raise DataError("completed or finalized E2 run cannot be resumed")
        if resume_stage == "R1" and not (resume_checkpoint / "COMPLETED").is_file():
            raise DataError("R1 resume checkpoint is not a completed boundary")
        if resume_stage == "R1" and (output / "r1").exists():
            from rles_rl_sb3.checkpoint import checkpoint_verify

            try:
                rl_manifest = checkpoint_verify(resume_checkpoint, rl_config, trusted_source=True)
            except (OSError, RuntimeError, ValueError) as exc:
                raise DataError(f"R1 resume checkpoint failed integrity validation: {exc}") from exc
            if rl_manifest.get("num_timesteps") != steps:
                raise DataError("existing R1 actor can only be reused from a full-budget checkpoint")
        saved_run = json.loads(run_manifest_path.read_text(encoding="utf-8"))
        if (not isinstance(saved_run, dict)
                or set(saved_run) not in (
                    {"schema", "run_id", "attempt_id", "recipe_hash", "run_fingerprint"},
                    {"schema", "run_id", "attempt_id", "recipe_hash", "run_fingerprint", "actor_spec"},
                    {"schema", "run_id", "attempt_id", "recipe_hash", "run_fingerprint", "actor_spec", "canonical_ppo_output"},
                    {"schema", "run_id", "attempt_id", "recipe_hash", "run_fingerprint", "actor_spec", "canonical_ppo_output", "worker_resources"})
                or saved_run["schema"] != "rles.historical-run.v1"
                or saved_run["attempt_id"] != saved_run["run_id"]
                or saved_run["recipe_hash"] != run_recipe_hash):
            raise DataError("resume configuration differs from the saved application run")
        if "actor_spec" in saved_run and _json_identity(saved_run["actor_spec"]) != _json_identity(asdict(actor_spec)):
            raise DataError("resume actor input scaling differs from the saved application run")
        if "worker_resources" in saved_run and saved_run["worker_resources"] != worker_resources:
            raise DataError("resume worker memory policy differs from the saved application run")
        if resume_stage in {"E1", "E2"}:
            from rles_es_evotorch.checkpoint import ESCheckpointStore

            checkpoint_root = (resume_checkpoint.parent if resume_checkpoint.name == saved_run["run_id"]
                               else resume_checkpoint)
            expected_stage = cast(str, resume_stage)
            expected_parent = output.resolve() / expected_stage
            if (checkpoint_root.parent.resolve() != expected_parent
                    or not checkpoint_root.name.startswith("generation-")
                    or not checkpoint_root.name[11:].isdigit()):
                raise DataError(f"{expected_stage} resume checkpoint is outside this run's generation store")
            try:
                ESCheckpointStore(checkpoint_root).load(saved_run["run_id"])
            except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
                raise DataError(f"{expected_stage} resume checkpoint failed integrity validation: {exc}") from exc
        if resume_stage == "E2":
            from rles_rl_sb3.checkpoint import checkpoint_verify

            resume_rl_checkpoint = output / "R1-checkpoints" / f"step-{steps:012d}"
            if not resume_rl_checkpoint.is_dir():
                raise DataError("E2 resume requires the completed final R1 checkpoint")
            try:
                final_manifest = checkpoint_verify(resume_rl_checkpoint, rl_config, trusted_source=True)
            except (OSError, RuntimeError, ValueError) as exc:
                raise DataError(f"E2 resume R1 checkpoint failed integrity validation: {exc}") from exc
            if final_manifest.get("num_timesteps") != steps:
                raise DataError("E2 resume R1 checkpoint does not match the frozen full step budget")
        failed = output / "FAILED.json"
        if failed.exists():
            archived = output / f"FAILED-{uuid.uuid4().hex}.json"
            failed.replace(archived)
    else:
        if resume_checkpoint is not None:
            raise DataError("resume checkpoint has no existing application run output")
        output.mkdir(parents=True)
        run_id = "historical-" + uuid.uuid4().hex
    if saved_run is not None:
        run_id = saved_run["run_id"]
    spec = RunSpec(run_id=run_id, attempt_id=run_id, route=route, algorithm=algorithm, seed=seed,
                   data_hash=fingerprint.dataset, feature_hash=fingerprint.feature, config_hash=rl_config.hash,
                   hard_risk_hash=fingerprint.hard_risk, execution_hash=fingerprint.execution,
                   normalization_hash=fingerprint.normalization, soft_risk_schema_hash=risk_space.signature,
                   initial_soft_risk_hash=fingerprint.soft_risk, actor_signature=actor_spec.signature, stage_budget=budgets)
    if saved_run is not None and saved_run["run_fingerprint"] != spec.fingerprint:
        raise DataError("reconstructed RunSpec differs from the original application attempt")
    ledger = ResearchLedger(output / "research.sqlite")
    if saved_run is None:
        ledger.create(run_id, run_id, validation_budget=1, hpo_budget=1,
                      es_evaluation_budget=max(1, sum(b.max_lane_requests for b in budgets.values())))
        _write(run_manifest_path, {"schema": "rles.historical-run.v1", "run_id": run_id,
                                   "attempt_id": run_id, "recipe_hash": run_recipe_hash,
                                   "run_fingerprint": spec.fingerprint, "actor_spec": asdict(actor_spec),
                                   "canonical_ppo_output": canonical_ppo_output, "worker_resources": worker_resources})
    else:
        ledger_state = ledger.state(run_id)
        if ledger_state["run_id"] != run_id or ledger_state["state"] != "DEVELOPMENT":
            raise DataError("research ledger is not open for an in-place training resume")
    selection = SelectionLedger(ledger, run_id)
    state: dict[str, Any] = {}
    run_quota = ResourceQuota(output / "quota.sqlite", cpu_cores=1,
                               ram_bytes=4 * 1024**3, gpus=int(rl_device == "cuda"))
    frozen_recipe = {**config["environment"], "split": "train_inner", "fit_scaler": False}

    def execute_es(context):
        stage = context.stage
        worker_config = {
            "candidate_kind": "actor" if stage == "E1" else "soft_risk",
            "env_factory_ref": "rles_gym_bridge.historical:make_historical_es_env",
            "env_config": {"recipe": frozen_recipe, "scaler": asdict(training.scaler), "transition_hash": fingerprint.hash},
            "actor_spec": asdict(actor_spec),
            "actor_signature": actor_spec.signature, "transition_hash": fingerprint.hash, "episode_steps": len(training.rows) - 3,
        }
        if stage == "E2":
            worker_config.update({"risk_dimensions": [asdict(d) for d in risk_space.dimensions],
                                  "frozen_actor_hash": context.previous.actor_hash,
                                  "actor_inference": "l4_inference", "frozen_actor_artifact": str((output / "r1").resolve())})
        checkpoint_dir = output / stage
        resume_es = (stage == resume_stage and stage in {"E1", "E2"}) or (
            stage == "E1" and any(
                (path / run_id).is_dir()
                for path in checkpoint_dir.glob("generation-[0-9]*") if path.is_dir()
            )
        )
        executor = ESStageExecutor(
            stage=stage, artifact_id=f"{run_id}-{stage}", candidate_spec=actor_spec if stage == "E1" else risk_space,
            evolution=EvolutionConfig("pgpe" if stage == "E1" else "cmaes", population, generations, seed,
                                      max_evaluations=population * generations),
            selection=selection, quota=run_quota,
            factory_ref="rles_orchestrator.native_es_worker:make_native_es_session", factory_config=worker_config,
            payload_builder=lambda _request: {}, payload_builder_id=digest("historical-native-payload-v1"),
            fitness_spec=FitnessSpec(), tail_fraction=0.25, fold_id=run_id, episode_ids=("train-inner",),
            checkpoint_dir=checkpoint_dir, workers=1, lease_owner=run_id, ram_bytes_per_worker=worker_ram_bytes,
            startup_timeout_s=60, timeout_s=120, resume=resume_es,
            enforce_memory_limit=enforce_worker_memory_limit,
        )
        try:
            result = executor(context)
        finally:
            if executor.worker_resource_evidence:
                _write(output / f"{stage}-resources-{uuid.uuid4().hex}.json", {
                    "scope": "ES workers after Job assignment; excludes coordinator/learner",
                    "configured_policy": worker_resources,
                    "workers": executor.worker_resource_evidence,
                })
        state[stage] = result.evolution_result
        return result

    def execute_rl(context):
        previous = context.previous
        source = (None if previous is None else np.asarray(state["E1"].best_search_vector, dtype=np.float32))
        teacher = (lambda batch: np.zeros((len(batch), 1), dtype=np.float32)) if source is None else (lambda batch: act(actor_spec, source, batch))
        _, corpus = rollout(training, teacher, seed=seed)
        distillation = config.get("distillation", {})

        def restore_env_factory(replay_values):
            from rles_gym_bridge.episode import EpisodeReplay

            if len(replay_values) != rl_config.num_envs:
                raise DataError("R1 checkpoint replay count differs from frozen L4 environments")
            replays = tuple(EpisodeReplay.from_dict(value) for value in replay_values)

            def make_restored_env(rank, _env_seed, phase):
                replay = replays[rank] if phase == "train" else None
                return training.make_env(fingerprint.hash, replay=replay)

            return make_restored_env

        executor = RLStageExecutor(
            artifact_id=f"{run_id}-R1", config=rl_config,
            env_factory=lambda _rank, _seed, _phase: training.make_env(fingerprint.hash),
            gate_authorizer=lambda _gates, cfg: cfg.fingerprint.dataset == training.store.snapshot_hash,
            train_steps=steps, inference_dir=output / "r1", fixed_observations=corpus,
            actor_spec=actor_spec if previous else None, source_genome=source,
            training_observations=corpus if previous else None,
            parity_tolerance=distillation.get("tolerance", 0.001), distill_epochs=distillation.get("epochs", 3000),
            distill_early_stop=distillation.get("early_stop", True),
            distill_batch_size=distillation.get("batch_size", 8), distill_learning_rate=distillation.get("learning_rate", 0.01),
            checkpoint_dir=output / "R1-checkpoints",
            checkpoint_interval_steps=config.get("checkpoint_interval_steps", 1024),
            resume_from=(resume_checkpoint if resume_stage == "R1" else
                         resume_rl_checkpoint if resume_stage == "E2" else None),
            resume_env_factory=restore_env_factory if resume_stage in {"R1", "E2"} else None,
        )
        allocation = ComputeAllocation(
            quota=run_quota, owner=run_id, attempt_id=run_id, cpu_cores=1,
            ram_bytes=4 * 1024**3, learner_device=rl_device,
            gpu_lock_directory=(output / "GPU-leases") if rl_device == "cuda" else None,
        )
        with allocation:
            return executor(context)

    try:
        runners = {stage: ProviderBackedStageRunner(stage, execute_rl if stage == "R1" else execute_es, selection)
                   for stage in STAGES[route]}
        lineage = Orchestrator(runners, attest_usage=selection).run(spec)
        risk_parameters = dict(state["E2"].risk_parameters) if "E2" in state else None
        if route == Route.A1:
            genome = np.asarray(state["E1"].best_search_vector, dtype=np.float32)
            np.savez(output / "es_actor.npz", genome=genome)
            actor_metadata = {"kind": "es", "spec": asdict(actor_spec), "genome_hash": genome_hash(actor_spec, genome)}
        else:
            actor_metadata = {"kind": "rl", "path": "r1", "actor_hash": (output / "r1" / "COMPLETED").read_text(encoding="ascii")}
        bundle = {"schema": "rles.historical-bundle.v1", "actor": actor_metadata, "soft_multipliers": risk_parameters,
                  "rl_device": rl_device,
                  "runtime_resources": {"es_worker_policy": worker_resources, "coordinator_learner_enforcement": "quota_only"},
                  "environment": config["environment"], "training_end_ns": training.end_ns,
                  "preprocessing": training.metadata(),
                  "lineage": [{"stage": a.stage, "usage": asdict(a.usage), "actor_hash": a.actor_hash,
                               "soft_risk_hash": a.soft_risk_hash} for a in lineage]}
        bundle_hash = _write_verified(output / "bundle.json", bundle)
        report = backtest(config, output, factory=evaluation) if evaluation is not None else None
        report_hash = (_write_verified(output / "report.json", report)
                       if report is not None else None)
        _write(output / "COMPLETED.json", {
            "run_id": run_id, "route": route.value, "algorithm": algorithm, "rl_device": rl_device,
            "bundle_sha256": bundle_hash, "report_sha256": report_hash,
        })
        result = {"output": str(output), "route": route.value, "algorithm": algorithm,
                  "rl_device": rl_device}
        if report is not None:
            result.update({"metrics": report["metrics"], "terminal_flat": report["terminal_flat"]})
        return result
    except Exception as exc:
        _write(output / "FAILED.json", {"error": type(exc).__name__, "detail": str(exc)})
        raise


def backtest(config: dict, bundle_path: str | Path, *, factory: HistoricalEnvironmentFactory | None = None,
             allow_execution_stress: bool = False) -> dict:
    if type(allow_execution_stress) is not bool:
        raise DataError("execution stress authorization must be an explicit boolean")
    root = Path(bundle_path).resolve()
    _verify_file(root / "bundle.json")
    bundle = json.loads((root / "bundle.json").read_text(encoding="utf-8"))
    raw_scaler = dict(bundle["preprocessing"]["scaler"])
    raw_scaler["mean"], raw_scaler["std"] = tuple(raw_scaler["mean"]), tuple(raw_scaler["std"])
    if factory is None:
        if _json_identity(config["environment"]) != _json_identity(bundle["environment"]):
            raise DataError("backtest may not replace the trained environment recipe")
        if config["evaluation"]["start_ns"] < bundle["training_end_ns"]:
            raise DataError("backtest window overlaps training")
        requested = config["evaluation"].get("split", "test")
        if requested not in {"test", "holdout"}:
            raise DataError("backtest requires an explicit Test or Holdout split")
        factory = HistoricalEnvironmentFactory(**{**bundle["environment"], **config["evaluation"],
                                                   "fit_scaler": False, "split": requested,
                                                   "scaler": Scaler(**raw_scaler)})
    identities = bundle["preprocessing"].get("identities")
    if not isinstance(identities, dict) or set(identities) != {
        "dataset", "features", "schema", "scaler", "execution", "hard_risk", "config",
    }:
        raise DataError("bundle lacks complete frozen trading identities")
    actual_identities = {
        "dataset": factory.store.snapshot_hash,
        "features": factory.features.hash,
        "schema": factory.schema.hash,
        "scaler": factory.scaler.hash,
        "execution": factory.execution.fingerprint(),
        "hard_risk": factory.hard.signature(),
        "config": factory.config_hash,
    }
    if allow_execution_stress:
        if factory.mode != "stress":
            raise DataError("stress backtest authorization requires a stress-mode factory")
        actual_identities["config"] = factory.base_config_hash
    if actual_identities != identities:
        raise DataError("backtest data, preprocessing, risk, or execution identity differs from bundle")
    if factory.schema.hash != digest_schema(bundle["preprocessing"]["schema"]):
        raise DataError("backtest observation schema differs from the trained actor")
    actor = bundle["actor"]
    predict: Callable[[np.ndarray], np.ndarray]
    if actor["kind"] == "rl":
        inference = InferenceActor(root / actor["path"], verify=True)
        if (root / actor["path"] / "COMPLETED").read_text(encoding="ascii") != actor["actor_hash"]:
            raise DataError("bundle actor identity mismatch")
        predict = inference.predict
    else:
        values = actor["spec"]
        spec = ActorGenomeSpec(**{**values, "observation_names": tuple(values["observation_names"]), "hidden_sizes": tuple(values["hidden_sizes"]),
                                  "input_scale": None if values.get("input_scale") is None else tuple(values["input_scale"])})
        with np.load(root / "es_actor.npz", allow_pickle=False) as weights:
            genome = weights["genome"].copy()
        if genome_hash(spec, genome) != actor["genome_hash"]:
            raise DataError("ES actor integrity mismatch")
        predict = lambda batch: act(spec, genome, batch)
    report, _ = rollout(factory, predict, seed=config.get("seed", 31),
                        soft_parameters=bundle["soft_multipliers"], collect_observations=False)
    report["dataset_quality"] = asdict(factory.quality)
    report["execution_mode"] = factory.mode
    report["latency_model_id"] = factory.execution.latency(factory.mode).model_id
    report["fill_model_id"] = factory.execution.fill(factory.mode).model_id
    report["fee_multiplier"] = str(getattr(factory.execution.fill(factory.mode), "fee_multiplier", Decimal("1")))
    if hasattr(factory.execution, "schema_version"):
        report["execution_scenario_hash"] = resolve_scenario(factory.execution, factory.mode).scenario_hash
    else:
        # Test and embedded callers may provide a narrow execution facade.
        report["execution_scenario_hash"] = sha256(json.dumps({
            "config_hash": factory.execution.fingerprint(), "mode": factory.mode,
            "latency_model_id": factory.execution.latency(factory.mode).model_id,
            "fill_model_id": factory.execution.fill(factory.mode).model_id,
            "fee_multiplier": report["fee_multiplier"],
        }, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    report["market_data_provenance"] = sorted({
        e.provenance for e in factory.store.events if e.kind != "instrument_rule"
    })
    report["rule_provenance"] = sorted({
        e.provenance for e in factory.store.events if e.kind == "instrument_rule"
    })
    return report


def diagnose_train(config: dict, bundle_path: str | Path) -> dict:
    """Replay only the bundle's frozen Train window, without refitting or queries."""
    root = Path(bundle_path).resolve()
    _verify_file(root / "bundle.json")
    bundle = json.loads((root / "bundle.json").read_text(encoding="utf-8"))
    environment = bundle["environment"]
    if _json_identity(config["environment"]) != _json_identity(environment):
        raise DataError("Train diagnostics may not replace the trained environment recipe")
    if environment["end_ns"] != bundle["training_end_ns"]:
        raise DataError("Train diagnostic window differs from the bundle training boundary")
    raw_scaler = dict(bundle["preprocessing"]["scaler"])
    raw_scaler["mean"], raw_scaler["std"] = tuple(raw_scaler["mean"]), tuple(raw_scaler["std"])
    factory = HistoricalEnvironmentFactory(**{**environment, "split": "train_inner",
                                              "fit_scaler": False, "scaler": Scaler(**raw_scaler)})
    report = backtest(config, root, factory=factory)
    report["diagnostic_scope"] = {
        "split": "train_inner", "start_ns": factory.start_ns, "end_ns": factory.end_ns,
        "purpose": "exploratory mechanism diagnosis; not independent strategy acceptance",
        "evaluation_queries": 0, "scaler_refitted": False,
    }
    return report


def digest_schema(values: dict) -> str:
    from rles_gym_bridge.observation import ObservationSchema

    return ObservationSchema(**values).hash


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="rles", description=__doc__)
    parser.add_argument("command", choices=("market-audit", "validate", "train", "backtest", "diagnose-train", "wfo"))
    parser.add_argument("--config", required=True)
    parser.add_argument("--bundle")
    parser.add_argument("--resume", action="store_true",
                        help="resume WFO at verified training checkpoints and durable evaluation boundaries")
    args = parser.parse_args(argv)
    try:
        if args.resume and args.command != "wfo":
            raise DataError("--resume is supported only by the WFO command")
        config = load_config(args.config)
        if args.command == "market-audit":
            result = audit_market(config)
        elif args.command == "validate":
            result = prepare(config).metadata()
        elif args.command == "train":
            result = train(config)
        elif args.command == "wfo":
            from .wfo_application import run_wfo

            result = run_wfo(config, resume=args.resume)
        else:
            if not args.bundle:
                raise DataError(f"{args.command} requires --bundle")
            _backtest_output_path(config)
            train_diagnostic = args.command == "diagnose-train"
            report = (diagnose_train(config, args.bundle) if train_diagnostic
                      else backtest(config, args.bundle))
            result = _persist_backtest_report(config, args.bundle, report,
                                              train_diagnostic=train_diagnostic)
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        return 0
    except Exception as exc:
        print(json.dumps({"status": "BLOCKED", "error": type(exc).__name__, "detail": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2
