"""Configuration-driven training and native historical backtesting application."""
from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from decimal import Decimal
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import sys
from typing import Any, Callable, Literal, cast
import uuid

import numpy as np

from rles_compute_runtime.allocation import ComputeAllocation
from rles_compute_runtime.device import DeviceResolver
from rles_compute_runtime.errors import DeviceUnavailable
from rles_compute_runtime.resources import ResourceQuota
from rles_data.core import CuratedStore, DataError, MINUTE_NS, digest
from rles_data.coverage import CoveragePolicy
from rles_data.features import Scaler
from rles_es_evotorch.actor import ActorGenomeSpec, act, genome_hash
from rles_es_evotorch.fitness import (ConstraintPriorityFitnessSpec,
                                      FeasibilityFirstFitnessSpec, FitnessSpec)
from rles_es_evotorch.provider import EvolutionConfig
from rles_es_evotorch.soft_risk import SoftRiskDimension, SoftRiskSpace
from rles_execution import resolve_scenario
from rles_gym_bridge.historical import HistoricalEnvironmentFactory, load_rules
from rles_research.ledger import ResearchLedger, SelectionLedger
from rles_research.access import EvaluationAccess
from rles_research.native_metrics import native_report
from rles_rl_sb3.actor import InferenceActor
from rles_rl_sb3.config import PolicySpec, RLConfig, SeedSpec, TransitionFingerprint

from .es_stage import ESStageExecutor
from .diagnostics import ExecutionAttribution
from .research_access import (begin_application_trial, finish_application_trial,
                              read_registered_train_window,
                              read_registered_train_window_with_receipt,
                              RegisteredTrainRead, record_development, market_key)
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


def _unfilled_order_reason(order: Mapping[str, Any] | None, filled_qty: Decimal) -> str | None:
    """Describe why the submitted quantity remains unfilled from final native facts."""
    if order is None:
        if filled_qty != 0:
            raise DataError("native fill events have no matching order for execution attribution")
        return "unknown"
    try:
        ordered_qty = Decimal(str(order["quantity"]))
        native_filled_qty = Decimal(str(order["filled_qty"]))
    except (KeyError, ArithmeticError, TypeError, ValueError) as exc:
        raise DataError("native order quantity facts are invalid for execution attribution") from exc
    if (not ordered_qty.is_finite() or ordered_qty <= 0
            or not native_filled_qty.is_finite() or not 0 <= native_filled_qty <= ordered_qty
            or not filled_qty.is_finite() or filled_qty < 0):
        raise DataError("native order quantity is invalid for execution attribution")
    if native_filled_qty != filled_qty:
        raise DataError("native fill events disagree with cumulative order filled quantity")
    status = str(order.get("status", "unknown")).upper()
    if status == "FILLED" and filled_qty != ordered_qty:
        raise DataError("native FILLED status conflicts with cumulative fill quantity")
    if status == "PARTIALLY_FILLED" and not 0 < filled_qty < ordered_qty:
        raise DataError("native PARTIALLY_FILLED status conflicts with cumulative fill quantity")
    if filled_qty >= ordered_qty:
        return None
    if status in {"REJECTED", "DENIED"}:
        for key in ("reject_reason", "reason"):
            reason = order.get(key)
            if isinstance(reason, str) and reason.strip():
                return reason
        return status
    if status in {"CANCELED", "CANCELLED", "EXPIRED"}:
        state = "after_partial_fill" if filled_qty > 0 else "unfilled"
        return f"{status.lower()}_{state}"
    if status == "PARTIALLY_FILLED":
        return "partial_fill_open"
    if status == "PENDING_CANCEL":
        return "cancel_pending"
    if status == "PENDING_UPDATE":
        return "update_pending"
    if status in {"ACCEPTED", "SUBMITTED", "WORKING"}:
        return "awaiting_fill"
    if status in {"INITIALIZED", "EMULATED", "RELEASED", "TRIGGERED", "VOIDED"}:
        return f"{status.lower()}_unfilled"
    return "unknown"


def _fee_summary(fills: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate native commissions without adding unlike currency units."""
    by_currency: dict[str, Decimal] = {}
    for fill in fills:
        commission = fill.get("commission")
        if not isinstance(commission, str):
            raise DataError("native fill commission is invalid for execution attribution")
        parts = commission.split()
        if len(parts) != 2 or not parts[1]:
            raise DataError("native fill commission must include amount and currency")
        try:
            amount = Decimal(parts[0])
        except (ArithmeticError, TypeError, ValueError) as exc:
            raise DataError("native fill commission amount is invalid") from exc
        if not amount.is_finite():
            raise DataError("native fill commission amount must be finite")
        currency = parts[1]
        by_currency[currency] = by_currency.get(currency, Decimal(0)) + amount
    ordered = dict(sorted(by_currency.items()))
    total = sum(ordered.values(), Decimal(0))
    single_currency = next(iter(ordered), None) if len(ordered) <= 1 else None
    return {
        "fee": str(total) if len(ordered) <= 1 else None,
        "fee_currency": single_currency,
        "fees_by_currency": {currency: str(amount) for currency, amount in ordered.items()},
    }


def _related_native_order_trace(client_order_id: str, order: Mapping[str, Any] | None,
                                fills: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Expose final native outcome for an order referenced by a non-submit decision."""
    filled_qty = sum((Decimal(str(fill["last_qty"])) for fill in fills), Decimal(0))
    status = str(order.get("status", "unknown")).upper() if order is not None else "unknown"
    return {
        "client_order_id": client_order_id,
        "status": status,
        "order_quantity": str(order["quantity"]) if order is not None else None,
        "cumulative_filled_qty": str(filled_qty),
        "fill_event_ids": [fill["event_id"] for fill in fills],
        "fees": _fee_summary(fills),
        "unfilled_reason": _unfilled_order_reason(order, filled_qty),
    }


def _index_native_orders(orders: Sequence[Any]) -> dict[str, dict[str, Any]]:
    """Index final native order facts without coercing or overwriting identities."""
    if (not isinstance(orders, Sequence)
            or isinstance(orders, (str, bytes, bytearray))):
        raise DataError("native order history must be a sequence")
    indexed: dict[str, dict[str, Any]] = {}
    for raw_order in orders:
        if not isinstance(raw_order, Mapping):
            raise DataError("native order history fact must be a mapping")
        identity = raw_order.get("client_order_id")
        if not isinstance(identity, str) or not identity or identity != identity.strip():
            raise DataError("native order history lacks valid client identity")
        order = dict(raw_order)
        previous = indexed.get(identity)
        if previous is not None and previous != order:
            raise DataError("native order identity has conflicting final facts")
        indexed[identity] = order
    return indexed


def _persist_e1_stage_actor(path: Path, spec: ActorGenomeSpec, genome: np.ndarray,
                            lane_evidence_hash: str) -> dict[str, str]:
    vector = np.asarray(genome, dtype=np.float32)
    actor_hash = genome_hash(spec, vector)
    if (not isinstance(lane_evidence_hash, str) or len(lane_evidence_hash) != 64
            or any(char not in "0123456789abcdef" for char in lane_evidence_hash)):
        raise DataError("E1 stage actor requires a valid lane evidence identity")
    payload = {
        "schema": "rles.selected-stage-actor.v1",
        "stage": "E1",
        "actor_spec": asdict(spec),
        "actor_hash": actor_hash,
        "lane_evidence_hash": lane_evidence_hash,
        "genome": vector.tolist(),
    }
    payload = json.loads(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False))
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise DataError("existing E1 stage actor artifact is unreadable") from exc
        if not isinstance(existing, dict) or existing != payload:
            changed = (", ".join(sorted(key for key in payload if not isinstance(existing, dict)
                                        or existing.get(key) != payload[key]))
                       if isinstance(existing, dict) else "payload_shape")
            raise DataError(f"existing E1 stage actor artifact differs from the completed stage: {changed}")
        marker = path.with_suffix(path.suffix + ".sha256")
        if not marker.is_file():
            marker.write_text(_file_hash(path), encoding="ascii")
        identity = _verify_file(path)
    else:
        identity = _write_verified(path, payload)
    return {"path": path.name, "sha256": identity, "actor_hash": actor_hash,
            "lane_evidence_hash": lane_evidence_hash}


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


@dataclass(frozen=True)
class _HistoricalRLEnvFactory:
    """Create R1 environments by reusing one local factory or rebuilding frozen worker state."""
    recipe_json: str
    scaler_json: str
    transition_hash: str
    num_envs: int
    episode_steps: int | None
    replay_json: tuple[str, ...] = ()
    shared_factory: HistoricalEnvironmentFactory | None = None
    train_access_json: str | None = None

    def __call__(self, rank: int, _env_seed: int, phase: str):
        from rles_gym_bridge.episode import EpisodeReplay

        if (type(rank) is not int or not 0 <= rank < self.num_envs
                or phase not in {"train", "eval"}
                or self.replay_json and len(self.replay_json) != self.num_envs):
            raise DataError("invalid frozen historical R1 environment request")
        replay = (EpisodeReplay.from_dict(json.loads(self.replay_json[rank]))
                  if phase == "train" and self.replay_json else None)
        if (phase == "train" and self.shared_factory is not None and self.num_envs == 1
                and replay is None):
            return self.shared_factory.make_env(
                self.transition_hash, episode_rank=rank, episode_count=self.num_envs,
                episode_steps=self.episode_steps,
            )
        recipe = json.loads(self.recipe_json)
        scaler_values = json.loads(self.scaler_json)
        if not isinstance(recipe, dict) or not isinstance(scaler_values, dict):
            raise DataError("frozen historical R1 recipe is not a JSON object")
        scaler_values["mean"] = tuple(scaler_values["mean"])
        scaler_values["std"] = tuple(scaler_values["std"])
        prepared_store = None
        if self.train_access_json is not None:
            access_config = json.loads(self.train_access_json)
            if not isinstance(access_config, dict):
                raise DataError("frozen R1 Train access is not a JSON object")
            prepared_store = _read_registered_train_store(access_config)
        factory = HistoricalEnvironmentFactory(**{
            **recipe, "scaler": Scaler(**scaler_values), "prepared_store": prepared_store,
        })
        return factory.make_env(
            self.transition_hash, replay=replay, episode_rank=rank,
            episode_count=self.num_envs, episode_steps=self.episode_steps,
        )


def load_config(path: str | Path) -> dict:
    source = Path(path).resolve()
    config = json.loads(source.read_text(encoding="utf-8"))
    allowed = {"environment", "evaluation", "route", "algorithm", "steps", "population", "generations",
               "seed", "hidden_layers", "normalize_observation", "hyperparameters", "output", "distillation",
               "checkpoint_interval_steps", "wfo", "resume_checkpoint", "rl_device", "rl_gpu_memory_fraction", "es_input_scaling",
               "enforce_worker_memory_limit", "worker_ram_bytes", "reward_scale",
               "rl_num_envs", "rl_episode_steps", "rl_corpus_episodes", "rl_corpus_episode_steps",
               "es_fitness", "es_scenario_fold_id", "es_candidate_evidence", "train_data_access"}
    if not isinstance(config, dict) or set(config) - allowed:
        raise DataError("unknown application configuration fields")
    access = config.get("train_data_access")
    if "train_data_access" in config:
        if (not isinstance(access, dict)
                or set(access) not in ({"start_ns", "end_ns", "marker_sha256", "label"},
                                       {"start_ns", "end_ns", "marker_sha256", "label", "source_mode"})
                or type(access.get("start_ns")) is not int
                or type(access.get("end_ns")) is not int
                or access["start_ns"] >= access["end_ns"]
                or not isinstance(access.get("marker_sha256"), str)
                or not isinstance(access.get("label"), str)
                or not access["label"].startswith("train-")
                or not isinstance(access.get("source_mode", "row_group_scoped_v1"), str)
                or access.get("source_mode", "row_group_scoped_v1") not in {
                    "row_group_scoped_v1", "whole_snapshot_isolated_v1",
                }):
            raise DataError("invalid scoped Train data access configuration")
    _es_fitness_spec(config)
    fold_id = config.get("es_scenario_fold_id")
    if (fold_id is not None and (not isinstance(fold_id, str)
            or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", fold_id) is None)):
        raise DataError("es_scenario_fold_id must be a 1-128 character stable identifier")
    if type(config.get("es_candidate_evidence", False)) is not bool:
        raise DataError("es_candidate_evidence must be Boolean")
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


def _es_fitness_spec(config: dict) -> FitnessSpec | ConstraintPriorityFitnessSpec:
    raw = config.get("es_fitness", {})
    if not isinstance(raw, dict):
        raise DataError("es_fitness must be a JSON object")
    try:
        protocol = raw.get("protocol", "legacy_weighted_v1")
        if protocol == "constraint_priority_train_inner_v1":
            weights = {key: value for key, value in raw.items() if key != "protocol"}
            if set(weights) - (set(ConstraintPriorityFitnessSpec.__dataclass_fields__) - {"protocol"}):
                raise DataError("constraint-priority es_fitness has unknown fields")
            return ConstraintPriorityFitnessSpec(**weights)
        if protocol == "constraint_priority_train_inner_v2":
            weights = {key: value for key, value in raw.items() if key != "protocol"}
            if set(weights) - (set(FeasibilityFirstFitnessSpec.__dataclass_fields__) - {"protocol"}):
                raise DataError("feasibility-first es_fitness has unknown fields")
            return FeasibilityFirstFitnessSpec(**weights)
        if protocol != "legacy_weighted_v1" or set(raw) - set(FitnessSpec.__dataclass_fields__) - {"protocol"}:
            raise DataError("es_fitness must declare legacy_weighted_v1, constraint_priority_train_inner_v1, or constraint_priority_train_inner_v2")
        weights = {key: value for key, value in raw.items() if key != "protocol"}
        return FitnessSpec(**weights)
    except (TypeError, ValueError) as exc:
        raise DataError(f"invalid es_fitness: {exc}") from exc


def _training_factory_recipe_hash(config: dict) -> str:
    recipe = {**config["environment"], "split": "train_inner", "fit_scaler": True,
              "reward_scale": config.get("reward_scale", 1.0)}
    return digest(recipe)


def _read_registered_train_receipt(config: dict) -> RegisteredTrainRead:
    access = config.get("train_data_access")
    if (not isinstance(access, dict)
            or set(access) not in ({"start_ns", "end_ns", "marker_sha256", "label"},
                                   {"start_ns", "end_ns", "marker_sha256", "label", "source_mode"})
            or type(access.get("start_ns")) is not int
            or type(access.get("end_ns")) is not int
            or not isinstance(access.get("marker_sha256"), str)
            or not isinstance(access.get("label"), str)
            or not isinstance(access.get("source_mode", "row_group_scoped_v1"), str)
            or access.get("source_mode", "row_group_scoped_v1") not in {
                "row_group_scoped_v1", "whole_snapshot_isolated_v1",
            }):
        raise DataError("invalid scoped Train data access configuration")
    return read_registered_train_window_with_receipt(
        config, {"start_ns": access["start_ns"], "end_ns": access["end_ns"]},
        marker_sha256=access["marker_sha256"], label=access["label"],
        allow_legacy_isolated_snapshot=(
            access.get("source_mode", "row_group_scoped_v1") == "whole_snapshot_isolated_v1"
        ),
    )


def _read_registered_train_store(config: dict) -> CuratedStore:
    receipt = _read_registered_train_receipt(config)
    rules = load_rules(config["environment"]["rules_path"])
    return CuratedStore((*receipt.store.events, *rules))


def prepare(config: dict, *, fit_scaler: bool = True, scaler: Scaler | None = None,
            compact_report_snapshots: bool = False,
            require_scoped_train_access: bool = False) -> HistoricalEnvironmentFactory:
    access = config.get("train_data_access")
    if require_scoped_train_access and access is None:
        raise DataError("this Train-only operation requires registered scoped Train data access")
    if "train_data_access" in config and access is None:
        raise DataError("scoped Train data access must be an explicit object")
    prepared_store = None
    if access is None:
        record_development(config, config["environment"], label="train")
    else:
        registered_read = _read_registered_train_receipt(config)
        rules = load_rules(config["environment"]["rules_path"])
        prepared_store = CuratedStore((*registered_read.store.events, *rules))
    factory = HistoricalEnvironmentFactory(**{
        **config["environment"], "split": "train_inner", "fit_scaler": fit_scaler,
        "scaler": scaler, "compact_report_snapshots": compact_report_snapshots,
        "reward_scale": config.get("reward_scale", 1.0), "prepared_store": prepared_store,
    })
    factory.application_training_recipe_hash = _training_factory_recipe_hash(config)
    if access is not None:
        setattr(factory, "application_registered_train_access", digest([config["environment"], access]))
    return factory


def audit_market(config: dict) -> dict:
    """Audit immutable Train-inner market facts through the registered scoped reader."""
    if not isinstance(config, dict) or "evaluation" in config:
        raise DataError("market audit is Train-only and rejects evaluation configuration")
    access = config.get("train_data_access")
    if (not isinstance(access, dict)
            or set(access) not in ({"start_ns", "end_ns", "marker_sha256", "label"},
                                   {"start_ns", "end_ns", "marker_sha256", "label", "source_mode"})
            or type(access.get("start_ns")) is not int
            or type(access.get("end_ns")) is not int
            or not isinstance(access.get("marker_sha256"), str)
            or not isinstance(access.get("label"), str)
            or not isinstance(access.get("source_mode", "row_group_scoped_v1"), str)
            or access.get("source_mode", "row_group_scoped_v1") not in {
                "row_group_scoped_v1", "whole_snapshot_isolated_v1",
            }):
        raise DataError("market audit requires explicit registered Train data access")
    environment = config.get("environment")
    if (not isinstance(environment, dict)
            or not {"instrument", "start_ns", "end_ns", "coverage_policy"} <= set(environment)
            or environment.get("split", "train_inner") != "train_inner"):
        raise DataError("market audit requires a complete Train-inner market window")
    store = read_registered_train_window(
        config, {"start_ns": access["start_ns"], "end_ns": access["end_ns"]},
        marker_sha256=access["marker_sha256"], label=access["label"],
        allow_legacy_isolated_snapshot=(
            access.get("source_mode", "row_group_scoped_v1") == "whole_snapshot_isolated_v1"
        ),
    )
    instrument = environment["instrument"]
    report = store.quality(
        start_ns=environment["start_ns"], end_ns=environment["end_ns"],
        expected_instruments=(instrument,),
        coverage_policy=CoveragePolicy(**environment["coverage_policy"]),
    )
    return {"schema": "rles.market-audit.v2", "instrument": instrument,
            "snapshot_hash": store.snapshot_hash,
            "access_scope": {"split": "train_inner", "start_ns": environment["start_ns"],
                             "end_ns": environment["end_ns"]},
            "splits": {"train": asdict(report)}}


def _action_dynamics(values: np.ndarray) -> dict[str, float | int]:
    delta = np.diff(values)
    previous, current = values[:-1], values[1:]
    comparable = (np.abs(previous) > 1e-6) & (np.abs(current) > 1e-6)
    flips = previous[comparable] * current[comparable] < 0
    return {
        "change_abs_mean": float(np.mean(np.abs(delta))) if delta.size else 0.0,
        "direction_comparison_pair_count": int(np.count_nonzero(comparable)),
        "direction_flip_count": int(np.count_nonzero(flips)),
        "direction_flip_fraction": float(np.mean(flips)) if flips.size else 0.0,
    }


class _ActionFeatureResponse:
    """Streaming Pearson correlation between policy actions and visible market inputs."""

    def __init__(self, names: tuple[str, ...]) -> None:
        self.names = names
        self.count = 0
        self.mean_features = np.zeros(len(names), dtype=np.float64)
        self.m2_features = np.zeros(len(names), dtype=np.float64)
        self.mean_action = 0.0
        self.m2_action = 0.0
        self.co_moments = np.zeros(len(names), dtype=np.float64)

    def observe(self, values: np.ndarray, action: float) -> None:
        features = np.asarray(values, dtype=np.float64)
        if (features.shape != self.mean_features.shape or not np.isfinite(features).all()
                or not np.isfinite(action)):
            raise DataError("action/market-feature diagnostics are invalid")
        self.count += 1
        delta_features = features - self.mean_features
        delta_action = action - self.mean_action
        self.mean_features += delta_features / self.count
        self.mean_action += delta_action / self.count
        self.m2_features += delta_features * (features - self.mean_features)
        self.m2_action += delta_action * (action - self.mean_action)
        self.co_moments += delta_features * (action - self.mean_action)

    def report(self) -> dict[str, object]:
        denominator = np.sqrt(np.maximum(self.m2_features, 0) * max(self.m2_action, 0))
        correlations = {
            name: (float(self.co_moments[index] / denominator[index])
                   if denominator[index] > 0 else None)
            for index, name in enumerate(self.names)
        }
        return {
            "count": self.count,
            "pearson_correlation_with_requested_action": correlations,
            "interpretation": "descriptive Train/evaluation-window dependence; not predictive or causal evidence",
        }


def rollout(factory: HistoricalEnvironmentFactory, predict, *, seed: int,
            soft_parameters: dict | None = None,
            collect_observations: bool = True, episode_rank: int = 0,
            episode_count: int = 1, episode_steps: int | None = None,
            reset_options: dict | None = None,
            include_account_rows: bool = True,
            include_execution_trace: bool = False,
            predictor_factory: Callable[[Any, dict], Any] | None = None,
            decision_observer: Callable[[Any, np.ndarray, dict, bool, bool], None] | None = None
            ) -> tuple[dict, np.ndarray]:
    if type(include_execution_trace) is not bool:
        raise DataError("include_execution_trace must be Boolean")
    env_options: dict[str, Any] = {"soft_multipliers": soft_parameters}
    if episode_rank != 0 or episode_count != 1 or episode_steps is not None:
        env_options.update(episode_rank=episode_rank, episode_count=episode_count,
                           episode_steps=episode_steps)
    env = factory.make_env(**env_options)
    observations = []
    requested_actions = []
    soft_targets = []
    risk_targets = []
    rounded_targets: list[float | None] = []
    executed_actions = []
    plan_actions: Counter[str] = Counter()
    plan_reasons: Counter[str] = Counter()
    decision_trace: list[dict[str, Any]] = []
    terminal_settlement_trace: list[dict[str, Any]] = []
    submitted_client_order_ids: set[str] = set()
    decision_identity_errors = 0
    missing_plans = 0
    interventions = hard_interventions = soft_adjustment_steps = 0
    soft_adjustment_total = hard_adjustment_total = total_adjustment = 0.0
    components_known = True
    attribution = ExecutionAttribution()
    schema_names = tuple(getattr(factory.schema, "names", ()))
    market_indices = tuple(i for i, name in enumerate(schema_names) if name.startswith("market."))
    action_feature_response = _ActionFeatureResponse(tuple(schema_names[i] for i in market_indices))
    try:
        reset_result = (env.reset(seed=seed) if reset_options is None
                        else env.reset(seed=seed, options=reset_options))
        observation, reset_info = reset_result
        active_predict = predict if predictor_factory is None else predictor_factory(env, reset_info)
        if not callable(active_predict):
            raise DataError("rollout predictor factory must return a callable")
        if collect_observations:
            observations.append(observation.copy())
        while True:
            decision_observation = observation
            action = np.asarray(active_predict(observation[None, :]), dtype=np.float32).reshape(1)
            observation, _, terminated, truncated, info = env.step(action)
            if decision_observer is not None:
                decision_observer(env, decision_observation, info, bool(terminated), bool(truncated))
            if collect_observations:
                observations.append(observation.copy())
            terminal_settlement = info.get("terminal_settlement")
            if terminal_settlement is not None:
                if not isinstance(terminal_settlement, dict):
                    raise DataError("native terminal settlement trace is malformed")
                terminal_settlement_trace.append({
                    "decision_id": info.get("execution_decision_id"),
                    **terminal_settlement,
                })
            requested_action = float(info["requested_action"])
            executed_action = float(info["policy_executed_action_pre_terminal"])
            if not np.isfinite(requested_action) or not np.isfinite(executed_action):
                raise DataError("native rollout action diagnostics are not finite")
            requested_actions.append(requested_action)
            action_feature_response.observe(decision_observation[list(market_indices)], requested_action)
            soft_target = info.get("risk_soft_target")
            risk_target = info.get("risk_adjusted_target")
            if (soft_target is None) != (risk_target is None):
                raise DataError("native intermediate and final risk targets must be reported together")
            if soft_target is not None:
                if not np.isfinite(soft_target) or not np.isfinite(risk_target):
                    raise DataError("native risk targets are not finite")
                soft_targets.append(float(soft_target))
                risk_targets.append(float(risk_target))
            rounded_target = info.get("quantity_rounded_target_exposure")
            if rounded_target is not None and not np.isfinite(rounded_target):
                raise DataError("quantity-rounded target exposure is not finite")
            rounded_targets.append(None if rounded_target is None else float(rounded_target))
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
            decision_id = info.get("execution_decision_id")
            plan_id = info.get("execution_plan_id")
            idempotency_key = info.get("execution_idempotency_key")
            client_order_id = info.get("execution_client_order_id")
            if (not isinstance(decision_id, str) or not decision_id
                    or not isinstance(plan_id, str) or not plan_id
                    or not isinstance(idempotency_key, str) or not idempotency_key
                    or (plan_action == "SUBMIT") != isinstance(client_order_id, str)):
                decision_identity_errors += 1
            else:
                if plan_action == "SUBMIT":
                    expected_client_id = "RLES-" + sha256(
                        idempotency_key.encode("utf-8")).hexdigest()
                    if client_order_id != expected_client_id:
                        decision_identity_errors += 1
                    if isinstance(client_order_id, str):
                        submitted_client_order_ids.add(client_order_id)
                if include_execution_trace:
                    try:
                        policy_equity_before = Decimal(str(info["policy_equity_before"]))
                        policy_equity_after = Decimal(str(info["policy_equity_after"]))
                        policy_equity_change = Decimal(str(info["policy_equity_change"]))
                    except (KeyError, ArithmeticError, TypeError, ValueError) as exc:
                        raise DataError("native policy-barrier equity facts are invalid") from exc
                    policy_equity_currency = info.get("policy_equity_currency")
                    if (not policy_equity_before.is_finite() or not policy_equity_after.is_finite()
                            or not policy_equity_change.is_finite()
                            or policy_equity_after - policy_equity_before != policy_equity_change
                            or not isinstance(policy_equity_currency, str)
                            or not policy_equity_currency):
                        raise DataError("native policy-barrier equity facts do not reconcile")
                    decision_trace.append({
                        "decision_id": decision_id,
                        "decision_ts_ns": int(info["decision_ts_ns"]),
                        "plan_id": plan_id,
                        "idempotency_key": idempotency_key,
                        # Actor output is a normalized Box action, not an
                        # exposure target. Keep the legacy alias for existing
                        # diagnostic artifacts while making units explicit.
                        "actor_raw_action": requested_action,
                        "actor_raw_action_unit": "normalized_box[-1,1]",
                        "risk_target_unit": "fraction_of_risk_budget",
                        "actor_requested_target": requested_action,
                        "soft_target": info.get("risk_soft_target"),
                        "hard_risk_target": info.get("risk_adjusted_target"),
                        "quantity_rounded_target": rounded_target,
                        "hard_risk_adjustment": float(info["risk_hard_adjustment"]),
                        "action": plan_action,
                        "reason": plan_reason,
                        "delta_qty": info.get("execution_delta_qty"),
                        "client_order_id": client_order_id,
                        "related_order_ids": list(info.get("execution_related_order_ids", ())),
                        "executed_exposure": executed_action,
                        "net_exposure": float(info["net_exposure"]),
                        "policy_equity_before": str(policy_equity_before),
                        "policy_equity_after": str(policy_equity_after),
                        "policy_equity_change": str(policy_equity_change),
                        "equity_currency": policy_equity_currency,
                        "market_features": dict(info["market_features"]),
                    })
            interventions += int(float(info["risk_guard_intervention"]) > 0)
            soft_amount = float(info["risk_soft_adjustment"])
            hard_amount = float(info["risk_hard_adjustment"])
            known = info["risk_intervention_components_known"]
            if (not np.isfinite(soft_amount) or soft_amount < 0 or
                    not np.isfinite(hard_amount) or hard_amount < 0 or type(known) is not bool):
                raise DataError("native risk adjustment attribution is invalid")
            components_known = components_known and known
            attribution.observe(requested=requested_action, executed=executed_action,
                                rounded=None if rounded_target is None else float(rounded_target),
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
            include_account_rows=include_account_rows,
            linear_equity_rounding=Decimal("0.00000001"),
        )
        native_orders: dict[str, dict] = {}
        final_snapshot = factory.snapshots[-1] if factory.snapshots else None
        native_orders = _index_native_orders(getattr(final_snapshot, "all_orders", ()))
        fills_by_order: dict[str, list[dict]] = {}
        for fill in report.get("fills", ()):
            identity = fill.get("client_order_id")
            if not isinstance(identity, str) or not identity or identity != identity.strip():
                raise DataError("native fill history lacks client identity")
            fills_by_order.setdefault(identity, []).append(fill)
        if decision_identity_errors:
            report["execution_decision_diagnostics"] = {
                "components_known": False,
                "decision_count": len(requested_actions),
                "missing_or_invalid_identity_count": decision_identity_errors,
                "action_counts": dict(sorted(plan_actions.items())),
                "reason_counts": dict(sorted(plan_reasons.items())),
            }
        else:
            submitted_ids = submitted_client_order_ids
            decision_status_counts: Counter[str] = Counter()
            submitted_with_fills = 0
            unfilled_reasons: Counter[str] = Counter()
            for identity in submitted_ids:
                order = native_orders.get(identity)
                order_fills = fills_by_order.get(identity, [])
                filled_qty = sum((Decimal(str(fill["last_qty"]))
                                  for fill in order_fills), Decimal(0))
                status = str(order.get("status", "unknown")).upper() if order else "unknown"
                decision_status_counts[status] += 1
                submitted_with_fills += int(bool(order_fills))
                reason = _unfilled_order_reason(order, filled_qty)
                if reason is not None:
                    unfilled_reasons[reason] += 1
            if include_execution_trace:
                for row in decision_trace:
                    if row["action"] != "SUBMIT":
                        row["submission_status"] = "not_submitted"
                        row["native_order_status"] = None
                        row["related_native_orders"] = [
                            _related_native_order_trace(
                                identity, native_orders.get(identity), fills_by_order.get(identity, []),
                            ) for identity in row["related_order_ids"]
                        ]
                        row["fills"] = []
                        row["filled_qty"] = "0"
                        row.update(_fee_summary([]))
                        row["unfilled_reason"] = None
                        continue
                    identity = row["client_order_id"]
                    order = native_orders.get(identity)
                    order_fills = fills_by_order.get(identity, [])
                    filled_qty = sum((Decimal(str(fill["last_qty"]))
                                      for fill in order_fills), Decimal(0))
                    fee_summary = _fee_summary(order_fills)
                    status = str(order.get("status", "unknown")).upper() if order else "unknown"
                    row["submission_status"] = "native_order_observed" if order else "unknown"
                    row["native_order_status"] = status
                    row["fills"] = [{
                        "event_id": fill["event_id"], "trade_id": fill.get("trade_id"),
                        "quantity": fill["last_qty"], "price": fill["last_px"],
                        "fee": fill["commission"], "ts_event": fill["ts_event"],
                    } for fill in order_fills]
                    row["filled_qty"] = str(filled_qty)
                    row.update(fee_summary)
                    row["unfilled_reason"] = _unfilled_order_reason(order, filled_qty)
            report["execution_decision_diagnostics"] = {
                "components_known": decision_identity_errors == 0,
                "decision_count": len(requested_actions),
                "missing_or_invalid_identity_count": 0,
                "submit_decision_count": plan_actions["SUBMIT"],
                "native_order_count": sum(identity in native_orders for identity in submitted_ids),
                "submitted_decisions_with_fills": submitted_with_fills,
                "native_order_status_counts": dict(sorted(decision_status_counts.items())),
                "unfilled_reason_counts": dict(sorted(unfilled_reasons.items())),
                "action_counts": dict(sorted(plan_actions.items())),
                "reason_counts": dict(sorted(plan_reasons.items())),
                "trace_included": include_execution_trace,
            }
            if include_execution_trace:
                report["execution_decision_trace"] = decision_trace
                report["terminal_settlement_trace"] = terminal_settlement_trace
        report["risk_guard_interventions"] = interventions
        report["risk_soft_adjustment_steps"] = soft_adjustment_steps
        report["risk_hard_interventions"] = hard_interventions
        report["risk_soft_adjustment_total"] = soft_adjustment_total
        report["risk_hard_adjustment_total"] = hard_adjustment_total
        report["risk_total_adjustment"] = total_adjustment
        report["risk_intervention_components_known"] = components_known
        if (rounded_targets and len(rounded_targets) == len(requested_actions)
                and all(value is not None for value in rounded_targets)
                and len(risk_targets) == len(requested_actions)):
            rounded = np.asarray(rounded_targets, dtype=np.float64)
            risk_values = np.asarray(risk_targets, dtype=np.float64)
            report["quantity_rounding_diagnostics"] = {
                "components_known": True,
                "count": int(rounded.size),
                "risk_adjusted_target_abs_mean": float(np.mean(np.abs(risk_values))),
                "quantity_rounded_target_abs_mean": float(np.mean(np.abs(rounded))),
                "risk_to_rounded_target_gap_abs_mean": float(np.mean(np.abs(risk_values - rounded))),
                "rounded_target_to_actual_exposure_gap_abs_mean": float(np.mean(
                    np.abs(rounded - np.asarray(executed_actions, dtype=np.float64)))),
            }
        else:
            report["quantity_rounding_diagnostics"] = {
                "components_known": False,
                "count": sum(value is not None for value in rounded_targets),
            }
        requested = np.asarray(requested_actions, dtype=np.float64)
        executed = np.asarray(executed_actions, dtype=np.float64)
        action_gap = np.abs(requested - executed)
        requested_dynamics = _action_dynamics(requested)
        executed_dynamics = _action_dynamics(executed)
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
            "requested_target_change_abs_mean": requested_dynamics["change_abs_mean"],
            "requested_direction_comparison_pair_count": requested_dynamics["direction_comparison_pair_count"],
            "requested_direction_flip_count": requested_dynamics["direction_flip_count"],
            "requested_direction_flip_fraction": requested_dynamics["direction_flip_fraction"],
            "executed_exposure_change_abs_mean": executed_dynamics["change_abs_mean"],
            "executed_direction_comparison_pair_count": executed_dynamics["direction_comparison_pair_count"],
            "executed_direction_flip_count": executed_dynamics["direction_flip_count"],
            "executed_direction_flip_fraction": executed_dynamics["direction_flip_fraction"],
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
        report["action_market_response"] = action_feature_response.report()
        if isinstance(reset_info, dict) and "decision_ts_ns" in reset_info:
            report["initial_decision_ts_ns"] = int(reset_info["decision_ts_ns"])
            report["episode_window"] = reset_info.get("episode_window")
        return report, (np.stack(observations) if collect_observations
                        else np.empty((0, observation.size), dtype=np.float32))
    finally:
        env.close()


def _fixed_observation_corpus(factory: HistoricalEnvironmentFactory, predict, *, seed: int,
                              episodes: int, episode_steps: int) -> tuple[np.ndarray, dict[str, Any]]:
    """Build a bounded, deterministic parity/distillation corpus across exclusive Train lanes."""
    full_steps = len(factory.rows) - 3
    if (type(episodes) is not int or not 1 <= episodes <= 32
            or type(episode_steps) is not int or episode_steps < 2
            or episode_steps > full_steps // episodes):
        raise DataError("fixed observation corpus must fit in 1-32 exclusive Train lanes")
    offsets = _episode_start_offsets(
        train_window_steps=full_steps, episode_steps=episode_steps, num_envs=episodes,
    )
    batches = []
    observed_windows = []
    for rank in range(episodes):
        report, observations = rollout(
            factory, predict, seed=(seed + rank) % 2**32, episode_rank=rank,
            episode_count=episodes, episode_steps=episode_steps,
            reset_options={"rles_episode_index": 1},
        )
        if observations.shape[0] != episode_steps + 1:
            raise DataError("fixed observation corpus window ended before its registered length")
        decision_ts = int(report["initial_decision_ts_ns"])
        start_offset = int(np.searchsorted(
            [row.decision_ts_ns for row in factory.rows], decision_ts,
        ))
        observed_windows.append({"rank": rank, "initial_decision_ts_ns": decision_ts,
                                 "start_offset": start_offset,
                                 "episode_window": report["episode_window"]})
        batches.append(observations)
    corpus = np.ascontiguousarray(np.concatenate(batches, axis=0), dtype=np.float32)
    return corpus, {
        "sampling_policy": "rank_exclusive_deterministic_train_windows_v2",
        "episode_count": episodes,
        "episode_steps": episode_steps,
        "initial_start_offsets": list(offsets),
        "observed_windows": observed_windows,
        "unique_decision_timestamps": episodes * episode_steps,
        "observation_rows_including_window_starts": int(corpus.shape[0]),
        "observation_width": int(corpus.shape[1]),
        "observation_sha256": sha256(corpus.tobytes(order="C")).hexdigest(),
        "scope": "train-only fixed parity/distillation corpus",
    }


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
    record_development(config, evaluation, label="validation")
    return HistoricalEnvironmentFactory(**{**config["environment"], **evaluation, "split": "validation",
                                            "fit_scaler": False, "scaler": training.scaler,
                                            "reward_scale": training.reward_config.reward_scale})


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
    quantity_scale = reference_mark * training.contract_spec.multiplier / training.budget
    scales = tuple(float(1 / training.capital) if name in money_fields else
                   float(quantity_scale) if name == "account.position_quantity" else 1.0
                   for name in training.schema.names)
    return ActorGenomeSpec(training.schema.names, hidden, version=2, input_scale=scales)


def _episode_start_offsets(*, train_window_steps: int, episode_steps: int,
                           num_envs: int) -> tuple[int, ...]:
    if (type(train_window_steps) is not int or type(episode_steps) is not int
            or type(num_envs) is not int or train_window_steps < 1
            or not 1 <= episode_steps <= train_window_steps or num_envs < 1
            or episode_steps > train_window_steps // num_envs):
        raise DataError("RL episode must fit inside every rank-exclusive Train time lane")
    return tuple(rank * train_window_steps // num_envs for rank in range(num_envs))


def _rl_sampling_policy(config: dict) -> tuple[int, int | None]:
    """Validate bounded R1 parallelism before reading data or allocating output."""
    num_envs = config.get("rl_num_envs", 1)
    episode_steps = config.get("rl_episode_steps")
    if (type(num_envs) is not int or not 1 <= num_envs <= 4
            or (episode_steps is not None
                and (type(episode_steps) is not int or episode_steps < 2))):
        raise DataError("RL sampling requires 1-4 environments and episode steps >= 2")
    if num_envs > 1 and episode_steps is None:
        raise DataError("multiple RL environments require explicit fixed episode steps")
    return num_envs, episode_steps


def _fixed_corpus_policy(config: dict) -> tuple[int, int]:
    episodes = config.get("rl_corpus_episodes", 1)
    episode_steps = config.get("rl_corpus_episode_steps", 256)
    if (type(episodes) is not int or episodes != 1
            or type(episode_steps) is not int or episode_steps < 2):
        raise DataError("fixed observation corpus currently requires one episode and episode steps >= 2")
    return episodes, episode_steps


def _training_coverage_report(*, rl_steps: int, num_envs: int,
                              train_window_decision_steps: int,
                              episode_decision_steps: int,
                              episode_start_offsets: tuple[int, ...]) -> dict[str, Any]:
    """Report deterministic rotating-lane interaction and coverage bounds."""
    if (type(rl_steps) is not int or rl_steps < 0 or type(num_envs) is not int or num_envs < 1
            or type(train_window_decision_steps) is not int or train_window_decision_steps < 1
            or type(episode_decision_steps) is not int or episode_decision_steps < 1
            or type(episode_start_offsets) is not tuple or len(episode_start_offsets) != num_envs
            or any(type(value) is not int or value < 0
                   or value + episode_decision_steps > train_window_decision_steps
                   for value in episode_start_offsets)):
        raise DataError("invalid RL interaction coverage facts")
    expected_offsets = tuple(
        rank * train_window_decision_steps // num_envs for rank in range(num_envs)
    )
    if episode_start_offsets != expected_offsets:
        raise DataError("RL coverage offsets differ from rank-exclusive time lanes")
    quotient, remainder = divmod(rl_steps, num_envs)
    per_env = [quotient + int(rank < remainder) for rank in range(num_envs)]
    lane_sizes = [((rank + 1) * train_window_decision_steps // num_envs
                   - rank * train_window_decision_steps // num_envs)
                  for rank in range(num_envs)]
    unique_steps = [min(count, lane_size)
                    for count, lane_size in zip(per_env, lane_sizes, strict=True)]
    union_steps = sum(unique_steps)
    return {
        "rl_interaction_steps": rl_steps,
        "environment_count": num_envs,
        "train_window_decision_steps": train_window_decision_steps,
        "nominal_episode_decision_steps": episode_decision_steps,
        "sampling_policy": "rank_exclusive_coprime_rotating_windows_v1",
        "initial_episode_start_offsets": list(episode_start_offsets),
        "rank_lane_decision_steps": lane_sizes,
        "interaction_steps_per_environment": per_env,
        "upper_bound_unique_decision_steps_per_environment": unique_steps,
        "upper_bound_unique_train_window_steps": union_steps,
        "upper_bound_train_window_coverage_fraction": union_steps / train_window_decision_steps,
        "nominal_full_episode_passes_per_environment": [
            count // episode_decision_steps for count in per_env
        ],
        "nominal_partial_episode_steps_per_environment": [
            count % episode_decision_steps for count in per_env
        ],
        "interpretation": ("upper bound for deterministic rank-exclusive rotating Train episodes; "
                           "the immutable learning diagnostics record actual timestamp coverage"),
    }


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


def _gpu_lease_identity(learner_device: str) -> dict[str, Any] | None:
    """Return a user-wide lock identity shared by distinct run output folders."""
    if learner_device == "cpu":
        return None
    if learner_device != "cuda":
        raise DataError("GPU lease requires a resolved cpu/cuda learner device")
    override = os.environ.get("RLES_GPU_LEASE_DIRECTORY")
    if override is not None:
        if not override.strip():
            raise DataError("RLES_GPU_LEASE_DIRECTORY cannot be empty")
        root = Path(override).expanduser()
        if not root.is_absolute():
            raise DataError("RLES_GPU_LEASE_DIRECTORY must be an absolute shared path")
    elif os.name == "nt":
        local_app_data = os.environ.get("LOCALAPPDATA")
        root = (Path(local_app_data) if local_app_data else
                Path.home() / "AppData" / "Local")
        root = root / "NautilusTrader-RL-ES" / "GPU-leases"
    else:
        state_home = os.environ.get("XDG_STATE_HOME")
        root = (Path(state_home) if state_home else Path.home() / ".local" / "state")
        root = root / "nautilus-trader-rl-es" / "GPU-leases"
    return {"scope": "user_shared", "directory": str(root.resolve()), "gpu_index": 0}


def _rl_gpu_memory_fraction(value: Any, learner_device: str) -> float | None:
    """Validate the optional PyTorch caching-allocator budget."""
    if value is None:
        return None
    if (type(value) not in (int, float) or not np.isfinite(value)
            or not 0 < value <= 1):
        raise DataError("rl_gpu_memory_fraction must be finite numeric in (0,1]")
    if learner_device != "cuda":
        raise DataError("rl_gpu_memory_fraction requires rl_device=cuda")
    return float(value)


def train(config: dict, *, evaluate: bool = True,
          prepared_training: HistoricalEnvironmentFactory | None = None) -> dict:
    if "train_data_access" in config and evaluate:
        raise DataError("scoped Train data access requires evaluate=False")
    import torch

    torch.set_num_threads(1)
    fitness_spec = _es_fitness_spec(config)
    route = Route(config.get("route", "A0"))
    rl_device = _resolve_rl_device(config.get("rl_device", "cpu"), route)
    gpu_lease = _gpu_lease_identity(rl_device)
    rl_gpu_memory_fraction = _rl_gpu_memory_fraction(config.get("rl_gpu_memory_fraction"), rl_device)
    rl_num_envs, requested_episode_steps = _rl_sampling_policy(config)
    corpus_episodes, corpus_episode_steps = _fixed_corpus_policy(config)
    if "R1" not in STAGES[route] and (rl_num_envs != 1 or requested_episode_steps is not None):
        raise DataError(f"route {route.value} has no R1 learner; RL episode sampling is unsupported")
    enforce_worker_memory_limit, worker_ram_bytes, worker_resources = _worker_memory_policy(config)
    if prepared_training is None:
        training = prepare(config)
    else:
        if getattr(prepared_training, "application_training_recipe_hash", None) != _training_factory_recipe_hash(config):
            raise DataError("reused Train factory does not match this candidate's frozen recipe")
        training = prepared_training
    algorithm = config.get("algorithm", "PPO")
    steps, population, generations = config.get("steps", 32), config.get("population", 4), config.get("generations", 1)
    seed = config.get("seed", 31)
    train_window_steps = len(training.rows) - 3
    rl_episode_steps = train_window_steps if requested_episode_steps is None else requested_episode_steps
    episode_offsets = _episode_start_offsets(
        train_window_steps=train_window_steps, episode_steps=rl_episode_steps,
        num_envs=rl_num_envs,
    )
    evaluation = _evaluation_factory(config, training) if evaluate else None
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
                                     for name in ("exposure_multiplier", "drawdown_throttle")))
    action_hash = digest({"target_net_exposure": [-1, 1], "shape": [1]})
    fingerprint = TransitionFingerprint(
        env_contract=digest("NautilusBarrierEnv-v0.4.0"), dataset=training.store.snapshot_hash,
        feature=training.features.hash, feature_scaler=training.scaler.hash, observation=training.schema.hash,
        action=action_hash, action_transform=digest("soft-hard-native-quantity"), execution=training.execution.fingerprint(),
        hard_risk=training.hard.signature(),
        soft_risk=digest({name: str(value) for name, value in asdict(training.soft).items()}),
        reward=(digest({"lambda_dd": 0, "lambda_guard": 0})
                if training.reward_config.reward_scale == 1.0 else
                digest({"lambda_dd": 0, "lambda_guard": 0,
                        "reward_scale": training.reward_config.reward_scale})),
        normalization=digest({"VecNormalize": config.get("normalize_observation", False)}),
        engine=digest("nautilus_trader-2.0.0rc6"), adapter=digest("rles.perpetual-native-v2"),
        fold=(digest([training.start_ns, training.end_ns]) if rl_num_envs == 1
              and requested_episode_steps is None else digest({
                  "window": [training.start_ns, training.end_ns],
                  "rl_episode_sampling": {
                      "policy": "rank_exclusive_coprime_rotating_windows_v1",
                      "num_envs": rl_num_envs, "episode_steps": rl_episode_steps,
                      "initial_start_offsets": episode_offsets,
                  },
              })),
        terminal=digest("force_flatten_with_cost:finite_settlement_no_bootstrap_v1"),
    )
    hyperparameters = config.get("hyperparameters", {"n_steps": 8, "batch_size": 8, "n_epochs": 1} if algorithm == "PPO" else
                                {"learning_starts": 8, "buffer_size": 256, "batch_size": 8, "train_freq": 1, "gradient_steps": 1})
    rl_config = RLConfig(algorithm, PolicySpec(len(training.schema.names), hidden_layers=policy_hidden, value_layers=hidden,
                                             observation_schema_hash=training.schema.hash, action_schema_hash=action_hash),
                         fingerprint, SeedSpec(
                             seed, tuple((seed + rank + 1) % 2**32 for rank in range(rl_num_envs)),
                             (seed + rl_num_envs + 1) % 2**32,
                             (seed + rl_num_envs + 2) % 2**32,
                         ),
                         hyperparameters=hyperparameters, device=rl_device,
                         normalize_observation=config.get("normalize_observation", False),
                         cuda_allocator_fraction=rl_gpu_memory_fraction)
    budgets = {
        stage: (StageBudget(max_rl_steps=steps) if stage == "R1" else
                StageBudget(
                    max_population=population,
                    max_generations=generations,
                    max_evaluations=(population + int(stage == "E2")) * generations,
                    max_lane_requests=(population + int(stage == "E2")) * generations,
                ))
        for stage in STAGES[route]
    }
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
    e2_neutral_baseline = True
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
        allowed_manifest_sets = (
                    {"schema", "run_id", "attempt_id", "recipe_hash", "run_fingerprint"},
                    {"schema", "run_id", "attempt_id", "recipe_hash", "run_fingerprint", "actor_spec"},
                    {"schema", "run_id", "attempt_id", "recipe_hash", "run_fingerprint", "actor_spec", "canonical_ppo_output"},
                    {"schema", "run_id", "attempt_id", "recipe_hash", "run_fingerprint", "actor_spec", "canonical_ppo_output", "worker_resources"},
                    {"schema", "run_id", "attempt_id", "recipe_hash", "run_fingerprint", "actor_spec", "canonical_ppo_output", "worker_resources", "e2_neutral_protocol"},
                    {"schema", "run_id", "attempt_id", "recipe_hash", "run_fingerprint", "actor_spec", "canonical_ppo_output", "worker_resources", "gpu_lease"},
                    {"schema", "run_id", "attempt_id", "recipe_hash", "run_fingerprint", "actor_spec", "canonical_ppo_output", "worker_resources", "e2_neutral_protocol", "gpu_lease"})
        if (not isinstance(saved_run, dict)
                or (set(saved_run) not in allowed_manifest_sets
                    and set(saved_run) - {"es_scenario_fold_id"} not in allowed_manifest_sets)
                or saved_run["schema"] != "rles.historical-run.v1"
                or saved_run["attempt_id"] != saved_run["run_id"]
                or saved_run["recipe_hash"] != run_recipe_hash):
            raise DataError("resume configuration differs from the saved application run")
        e2_neutral_baseline = saved_run.get("e2_neutral_protocol") == "paired_generation_neutral_r1_v1"
        if "E2" in budgets and not e2_neutral_baseline:
            # Preserve the frozen lane/evaluation caps of pre-protocol checkpoints.
            budgets["E2"] = StageBudget(
                max_population=population, max_generations=generations,
                max_evaluations=population * generations,
                max_lane_requests=population * generations,
            )
        if "actor_spec" in saved_run and _json_identity(saved_run["actor_spec"]) != _json_identity(asdict(actor_spec)):
            raise DataError("resume actor input scaling differs from the saved application run")
        if "worker_resources" in saved_run and saved_run["worker_resources"] != worker_resources:
            raise DataError("resume worker memory policy differs from the saved application run")
        if "gpu_lease" in saved_run and saved_run["gpu_lease"] != gpu_lease:
            raise DataError("resume GPU lease scope differs from the saved application run")
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
    scenario_fold_id = config.get("es_scenario_fold_id", run_id)
    if (saved_run is not None
            and saved_run.get("es_scenario_fold_id", run_id) != scenario_fold_id):
        raise DataError("ES scenario fold differs from the saved application run")
    spec = RunSpec(run_id=run_id, attempt_id=run_id, route=route, algorithm=algorithm, seed=seed,
                   data_hash=fingerprint.dataset, feature_hash=fingerprint.feature, config_hash=rl_config.hash,
                   hard_risk_hash=fingerprint.hard_risk, execution_hash=fingerprint.execution,
                   normalization_hash=fingerprint.normalization, soft_risk_schema_hash=risk_space.signature,
                   initial_soft_risk_hash=fingerprint.soft_risk, actor_signature=actor_spec.signature, stage_budget=budgets)
    if saved_run is not None and saved_run["run_fingerprint"] != spec.fingerprint:
        legacy_space = SoftRiskSpace(tuple(SoftRiskDimension(name, 0.1, 1.0)
                                          for name in ("exposure_multiplier", "drawdown_throttle",
                                                       "volatility_throttle")))
        legacy_spec = replace(spec, soft_risk_schema_hash=legacy_space.signature)
        if saved_run["run_fingerprint"] == legacy_spec.fingerprint:
            risk_space, spec = legacy_space, legacy_spec
    if saved_run is not None and saved_run["run_fingerprint"] != spec.fingerprint:
        raise DataError("reconstructed RunSpec differs from the original application attempt")
    ledger = ResearchLedger(output / "research.sqlite")
    if saved_run is None:
        ledger.create(run_id, run_id, validation_budget=1, hpo_budget=1,
                      es_evaluation_budget=max(1, sum(b.max_lane_requests for b in budgets.values())))
        run_manifest = {"schema": "rles.historical-run.v1", "run_id": run_id,
                        "attempt_id": run_id, "recipe_hash": run_recipe_hash,
                        "run_fingerprint": spec.fingerprint, "actor_spec": asdict(actor_spec),
                        "canonical_ppo_output": canonical_ppo_output, "worker_resources": worker_resources,
                        "gpu_lease": gpu_lease,
                        "e2_neutral_protocol": "paired_generation_neutral_r1_v1"}
        if "es_scenario_fold_id" in config:
            run_manifest["es_scenario_fold_id"] = scenario_fold_id
        _write(run_manifest_path, run_manifest)
    else:
        ledger_state = ledger.state(run_id)
        if ledger_state["run_id"] != run_id or ledger_state["state"] != "DEVELOPMENT":
            raise DataError("research ledger is not open for an in-place training resume")
    selection = SelectionLedger(ledger, run_id)
    state: dict[str, Any] = {}
    run_quota = ResourceQuota(output / "quota.sqlite", cpu_cores=rl_num_envs,
                               ram_bytes=4 * 1024**3, gpus=int(rl_device == "cuda"))
    frozen_recipe = {
        **config["environment"], "split": "train_inner", "fit_scaler": False,
        "reward_scale": training.reward_config.reward_scale,
    }
    frozen_recipe_json = json.dumps(
        frozen_recipe, sort_keys=True, separators=(",", ":"), allow_nan=False,
    )
    frozen_scaler_json = json.dumps(
        asdict(training.scaler), sort_keys=True, separators=(",", ":"), allow_nan=False,
    )
    frozen_train_access_json = (None if "train_data_access" not in config else json.dumps({
        "output": config["output"], "environment": config["environment"],
        "train_data_access": config["train_data_access"],
    }, sort_keys=True, separators=(",", ":"), allow_nan=False))
    rl_env_factory = _HistoricalRLEnvFactory(
        frozen_recipe_json, frozen_scaler_json, fingerprint.hash, rl_num_envs,
        requested_episode_steps, shared_factory=training if rl_num_envs == 1 else None,
        train_access_json=frozen_train_access_json,
    )

    def execute_es(context):
        stage = context.stage
        worker_env_config = {
            "recipe": frozen_recipe, "scaler": asdict(training.scaler),
            "transition_hash": fingerprint.hash,
        }
        if frozen_train_access_json is not None:
            worker_env_config["registered_train_access"] = json.loads(frozen_train_access_json)
        worker_config = {
            "candidate_kind": "actor" if stage == "E1" else "soft_risk",
            "env_factory_ref": "rles_orchestrator.native_es_worker:make_registered_historical_es_env",
            "env_config": worker_env_config,
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
            fitness_spec=fitness_spec, tail_fraction=0.25,
            fold_id=scenario_fold_id, episode_ids=("train-inner",),
            checkpoint_dir=checkpoint_dir, workers=1, lease_owner=run_id, ram_bytes_per_worker=worker_ram_bytes,
            startup_timeout_s=60, timeout_s=120, resume=resume_es,
            enforce_memory_limit=enforce_worker_memory_limit,
            include_neutral_baseline=(stage == "E2" and e2_neutral_baseline),
            include_constraint_evidence=(
                isinstance(fitness_spec, ConstraintPriorityFitnessSpec)
                or config.get("es_candidate_evidence", False)
            ),
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
        if stage == "E1":
            if result.evolution_result is None:
                raise DataError("E1 stage completed without a selected optimizer result")
            selected_genome = np.asarray(result.evolution_result.best_search_vector, dtype=np.float32)
            state["E1_stage_actor_artifact"] = _persist_e1_stage_actor(
                output / "E1-selected-actor.json", actor_spec, selected_genome,
                result.metadata.get("lane_evidence_hash", ""),
            )
        return result

    def execute_rl(context):
        nonlocal fixed_corpus_metadata
        previous = context.previous
        source = (None if previous is None else np.asarray(state["E1"].best_search_vector, dtype=np.float32))
        teacher = (lambda batch: np.zeros((len(batch), 1), dtype=np.float32)) if source is None else (lambda batch: act(actor_spec, source, batch))
        corpus, fixed_corpus_metadata = _fixed_observation_corpus(
            training, teacher, seed=seed,
            episodes=corpus_episodes, episode_steps=corpus_episode_steps,
        )
        distillation = config.get("distillation", {})

        def restore_env_factory(replay_values):
            if len(replay_values) != rl_config.num_envs:
                raise DataError("R1 checkpoint replay count differs from frozen L4 environments")
            return _HistoricalRLEnvFactory(
                frozen_recipe_json, frozen_scaler_json, fingerprint.hash, rl_num_envs,
                requested_episode_steps, tuple(
                    json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
                    for value in replay_values
                ), train_access_json=frozen_train_access_json,
            )

        executor = RLStageExecutor(
            artifact_id=f"{run_id}-R1", config=rl_config,
            env_factory=rl_env_factory,
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
            quota=run_quota, owner=run_id, attempt_id=run_id, cpu_cores=rl_num_envs,
            ram_bytes=4 * 1024**3, learner_device=rl_device,
            gpu_lock_directory=Path(gpu_lease["directory"]) if gpu_lease is not None else None,
            gpu_index=gpu_lease["gpu_index"] if gpu_lease is not None else 0,
        )
        with allocation:
            return executor(context)

    fixed_corpus_metadata: dict[str, Any] | None = None
    trial_recipe = {
        "run_recipe": run_recipe,
        "dataset_sha256": fingerprint.dataset,
        "run_fingerprint": spec.fingerprint,
    }
    trial_attempt_key = begin_application_trial(
        run_id=run_id, recipe=trial_recipe, feature_sha=fingerprint.feature, seed=seed,
        resumed=resume_stage is not None,
        detail={"route": route.value, "algorithm": algorithm, "stage": resume_stage,
                "evaluate": evaluate, "dataset_sha256": fingerprint.dataset},
    )
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
            r1_artifact = next(artifact for artifact in lineage if artifact.stage == "R1")
            diagnostics_hash = r1_artifact.metadata.get("training_diagnostics_hash")
            if diagnostics_hash is None or diagnostics_hash != _verify_file(output / "R1-learning.json"):
                raise DataError("R1 learning diagnostics are missing or differ from the stage receipt")
            actor_metadata = {
                "kind": "rl", "path": "r1",
                "actor_hash": (output / "r1" / "COMPLETED").read_text(encoding="ascii"),
                "training_diagnostics_path": "R1-learning.json",
                "training_diagnostics_hash": diagnostics_hash,
            }
        actual_rl_steps = sum(a.usage.rl_steps for a in lineage if a.stage == "R1")
        training_coverage = _training_coverage_report(
            rl_steps=actual_rl_steps, num_envs=rl_config.num_envs,
            train_window_decision_steps=train_window_steps,
            episode_decision_steps=rl_episode_steps,
            episode_start_offsets=episode_offsets,
        )
        stage_actor_artifacts: dict[str, dict[str, str]] = {}
        e1_stage_actor_path = output / "E1-selected-actor.json"
        if e1_stage_actor_path.is_file():
            e1_stage_actor_hash = _verify_file(e1_stage_actor_path)
            e1_stage_actor = json.loads(e1_stage_actor_path.read_text(encoding="utf-8"))
            e1_lineage = next((artifact for artifact in lineage if artifact.stage == "E1"), None)
            if (not isinstance(e1_stage_actor, dict)
                    or e1_stage_actor.get("schema") != "rles.selected-stage-actor.v1"
                    or e1_stage_actor.get("stage") != "E1"
                    or e1_lineage is None
                    or e1_stage_actor.get("actor_hash") != e1_lineage.actor_hash
                    or e1_stage_actor.get("lane_evidence_hash")
                    != e1_lineage.metadata.get("lane_evidence_hash")
                    or e1_stage_actor.get("actor_spec")
                    != json.loads(json.dumps(asdict(actor_spec), sort_keys=True,
                                             separators=(",", ":"), allow_nan=False))):
                raise DataError("persisted E1 stage actor differs from the attested lineage")
            try:
                reconstructed_e1_hash = genome_hash(
                    actor_spec, np.asarray(e1_stage_actor["genome"], dtype=np.float32),
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise DataError("persisted E1 stage actor genome is malformed") from exc
            if reconstructed_e1_hash != e1_lineage.actor_hash:
                raise DataError("persisted E1 stage actor genome hash differs from the lineage")
            stage_actor_artifacts["E1"] = {
                "path": e1_stage_actor_path.name,
                "sha256": e1_stage_actor_hash,
                "actor_hash": str(e1_stage_actor["actor_hash"]),
                "lane_evidence_hash": str(e1_stage_actor["lane_evidence_hash"]),
            }
        bundle = {"schema": "rles.historical-bundle.v1", "actor": actor_metadata,
                  "es_scenario_fold_id": scenario_fold_id, "soft_multipliers": risk_parameters,
                  "stage_actor_artifacts": stage_actor_artifacts,
                  "es_candidate_evidence": config.get("es_candidate_evidence", False),
                  "reward_scale": training.reward_config.reward_scale,
                  "rl_device": rl_device,
                  "runtime_resources": {
                      "es_worker_policy": worker_resources,
                      "coordinator_learner_enforcement": "quota_only",
                      "gpu_lease": gpu_lease,
                      "cuda_allocator_fraction": rl_gpu_memory_fraction,
                  },
                  "training_coverage": training_coverage,
                  "fixed_observation_corpus": fixed_corpus_metadata,
                  "environment": config["environment"], "training_end_ns": training.end_ns,
                  "preprocessing": training.metadata(),
                  "lineage": [{"stage": a.stage, "usage": asdict(a.usage), "actor_hash": a.actor_hash,
                               "soft_risk_hash": a.soft_risk_hash,
                               "metadata": dict(a.metadata)} for a in lineage]}
        bundle_hash = _write_verified(output / "bundle.json", bundle)
        report = backtest(config, output, factory=evaluation) if evaluation is not None else None
        report_hash = (_write_verified(output / "report.json", report)
                       if report is not None else None)
        _write(output / "COMPLETED.json", {
            "run_id": run_id, "route": route.value, "algorithm": algorithm, "rl_device": rl_device,
            "bundle_sha256": bundle_hash, "report_sha256": report_hash,
        })
        result = {"output": str(output), "route": route.value, "algorithm": algorithm,
                  "rl_device": rl_device, "training_coverage": training_coverage}
        if fixed_corpus_metadata is not None:
            result["fixed_observation_corpus"] = fixed_corpus_metadata
        if report is not None:
            result.update({"metrics": report["metrics"], "terminal_flat": report["terminal_flat"]})
        finish_application_trial(
            trial_attempt_key, status="completed",
            detail={"route": route.value, "algorithm": algorithm, "run_fingerprint": spec.fingerprint,
                    "bundle_sha256": bundle_hash, "report_sha256": report_hash},
        )
        return result
    except BaseException as exc:
        trial_status = "interrupted" if isinstance(exc, (KeyboardInterrupt, SystemExit)) else "failed"
        finish_application_trial(
            trial_attempt_key, status=trial_status,
            detail={"error": type(exc).__name__, "detail": str(exc),
                    "route": route.value, "algorithm": algorithm},
        )
        if isinstance(exc, Exception):
            _write(output / "FAILED.json", {"error": type(exc).__name__, "detail": str(exc)})
        raise


def backtest(config: dict, bundle_path: str | Path, *, factory: HistoricalEnvironmentFactory | None = None,
             allow_execution_stress: bool = False, summary_only: bool = False,
             evaluation_access: EvaluationAccess | None = None,
             include_execution_trace: bool = False) -> dict:
    if "train_data_access" in config:
        access_identity = digest([config.get("environment"), config["train_data_access"]])
        if (factory is None or factory.split != "train_inner"
                or getattr(factory, "application_registered_train_access", None) != access_identity):
            raise DataError("scoped Train backtest requires a factory built by the registered Train reader")
    if type(allow_execution_stress) is not bool:
        raise DataError("execution stress authorization must be an explicit boolean")
    if type(summary_only) is not bool:
        raise DataError("summary-only report flag must be Boolean")
    if type(include_execution_trace) is not bool:
        raise DataError("execution trace flag must be Boolean")
    requested = (config.get("evaluation", {}).get("split", "test") if factory is None
                 else factory.split)
    formal = requested in {"test", "holdout"}
    if formal and not isinstance(evaluation_access, EvaluationAccess):
        raise DataError("formal Test/Holdout requires WFO research access; use diagnose-train for exploration")
    if requested not in {"test", "holdout", "train_inner", "validation"}:
        raise DataError("unknown evaluation split")
    root = Path(bundle_path).resolve()
    bundle_sha = _verify_file(root / "bundle.json")
    bundle = json.loads((root / "bundle.json").read_text(encoding="utf-8"))
    raw_scaler = dict(bundle["preprocessing"]["scaler"])
    raw_scaler["mean"], raw_scaler["std"] = tuple(raw_scaler["mean"]), tuple(raw_scaler["std"])
    if formal:
        window: dict = config["evaluation"] if factory is None else {
            "start_ns": factory.start_ns, "end_ns": factory.end_ns, "mode": factory.mode,
        }
        assert evaluation_access is not None
        evaluation_access.consume(bundle_sha=bundle_sha, split=requested,
                                  start_ns=window["start_ns"], end_ns=window["end_ns"],
                                  market=market_key(config["environment"]["instrument"] if factory is None
                                                    else factory.instrument),
                                  mode=window.get("mode", "base"))
    elif factory is not None and "train_data_access" not in config:
        record_development(config, {"start_ns": factory.start_ns, "end_ns": factory.end_ns},
                           label=requested)
    if factory is None:
        if _json_identity(config["environment"]) != _json_identity(bundle["environment"]):
            raise DataError("backtest may not replace the trained environment recipe")
        if config["evaluation"]["start_ns"] < bundle["training_end_ns"]:
            raise DataError("backtest window overlaps training")
        requested = config["evaluation"].get("split", "test")
        if requested not in {"test", "holdout", "validation"}:
            raise DataError("standalone exploration requires an explicit Validation split")
        bundle_reward_scale = bundle.get("reward_scale", 1.0)
        if config.get("reward_scale", bundle_reward_scale) != bundle_reward_scale:
            raise DataError("backtest reward scale differs from the trained bundle")
        if not formal:
            record_development(config, config["evaluation"], label="validation")
        factory = HistoricalEnvironmentFactory(**{**bundle["environment"], **config["evaluation"],
                                                   "fit_scaler": False, "split": requested,
                                                   "scaler": Scaler(**raw_scaler),
                                                   "reward_scale": bundle_reward_scale})
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
        diagnostics_path = actor.get("training_diagnostics_path")
        diagnostics_hash = actor.get("training_diagnostics_hash")
        if (diagnostics_path not in (None, "R1-learning.json")
                or (diagnostics_path is None) != (diagnostics_hash is None)
                or diagnostics_path is not None
                and _verify_file(root / diagnostics_path) != diagnostics_hash):
            raise DataError("bundle R1 learning diagnostics identity mismatch")
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
                        soft_parameters=bundle["soft_multipliers"], collect_observations=False,
                        include_account_rows=not summary_only,
                        include_execution_trace=include_execution_trace)
    report["dataset_quality"] = asdict(factory.quality)
    report["execution_mode"] = factory.mode
    report["latency_model_id"] = factory.execution.latency(factory.mode).model_id
    report["fill_model_id"] = factory.execution.fill(factory.mode).model_id
    report["fee_multiplier"] = str(getattr(factory.execution.fill(factory.mode), "fee_multiplier", Decimal("1")))
    report["simulated_depth_fraction"] = str(getattr(
        factory.execution.fill(factory.mode), "simulated_depth_fraction", Decimal("1"),
    ))
    if hasattr(factory.execution, "schema_version"):
        report["execution_scenario_hash"] = resolve_scenario(factory.execution, factory.mode).scenario_hash
    else:
        # Test and embedded callers may provide a narrow execution facade.
        report["execution_scenario_hash"] = sha256(json.dumps({
            "config_hash": factory.execution.fingerprint(), "mode": factory.mode,
            "latency_model_id": factory.execution.latency(factory.mode).model_id,
            "fill_model_id": factory.execution.fill(factory.mode).model_id,
            "fee_multiplier": report["fee_multiplier"],
            **({"simulated_depth_fraction": report["simulated_depth_fraction"]}
               if report["simulated_depth_fraction"] != "1" else {}),
        }, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    report["market_data_provenance"] = sorted({
        e.provenance for e in factory.store.events if e.kind != "instrument_rule"
    })
    report["rule_provenance"] = sorted({
        e.provenance for e in factory.store.events if e.kind == "instrument_rule"
    })
    report["evaluation_scope"] = {"split": requested, "formal": formal,
                                  "project_access_key": evaluation_access.key if evaluation_access else None}
    return report


def diagnose_train(config: dict, bundle_path: str | Path, *,
                   include_execution_trace: bool = False) -> dict:
    """Replay only the bundle's frozen Train window, without refitting or queries."""
    if not isinstance(config, dict) or "train_data_access" not in config:
        raise DataError("Train diagnostics require registered scoped Train data access")
    if config.get("evaluation") is not None:
        raise DataError("Train diagnostics reject Evaluation configuration")
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
    factory = prepare(config, fit_scaler=False, scaler=Scaler(**raw_scaler),
                      compact_report_snapshots=True, require_scoped_train_access=True)
    report = backtest(config, root, factory=factory, summary_only=True,
                      include_execution_trace=include_execution_trace)
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
    parser.add_argument("--execution-trace", action="store_true",
                        help="include per-decision native order and fill attribution in reports")
    parser.add_argument("--train-only", action="store_true",
                        help="run frozen Train stages and persist diagnostics without querying Validation")
    parser.add_argument("--resume", action="store_true",
                        help="resume WFO at verified training checkpoints and durable evaluation boundaries")
    args = parser.parse_args(argv)
    try:
        if args.resume and args.command != "wfo":
            raise DataError("--resume is supported only by the WFO command")
        if args.train_only and args.command != "train":
            raise DataError("--train-only is supported only by the train command")
        config = load_config(args.config)
        if ("train_data_access" in config
                and args.command not in {"validate", "train", "diagnose-train"}):
            raise DataError("scoped Train data access is limited to Train-only commands")
        if args.command == "market-audit":
            result = audit_market(config)
        elif args.command == "validate":
            result = prepare(config).metadata()
        elif args.command == "train":
            result = train(config, evaluate=not args.train_only)
        elif args.command == "wfo":
            from .wfo_application import run_wfo

            result = run_wfo(config, resume=args.resume)
        else:
            if not args.bundle:
                raise DataError(f"{args.command} requires --bundle")
            _backtest_output_path(config)
            train_diagnostic = args.command == "diagnose-train"
            report = (diagnose_train(config, args.bundle,
                                     include_execution_trace=args.execution_trace)
                      if train_diagnostic else
                      backtest(config, args.bundle,
                               include_execution_trace=args.execution_trace))
            result = _persist_backtest_report(config, args.bundle, report,
                                              train_diagnostic=train_diagnostic)
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        return 0
    except Exception as exc:
        print(json.dumps({"status": "BLOCKED", "error": type(exc).__name__, "detail": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2
