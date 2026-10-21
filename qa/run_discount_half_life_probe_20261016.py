"""Run a frozen, Train-only paired PPO gamma half-life screen.

Each seed has a baseline and six-hour-discount arm with identical budgets and
training fields. No Test/Holdout access or candidate promotion is possible.
"""
from __future__ import annotations

import argparse
import ctypes
import gc
import json
from pathlib import Path
import sys
import time

import numpy as np

# The root virtual environment was synchronized with non-editable packages.
# Research runs must import the source tree that is checksummed in this recipe.
_ROOT = Path(__file__).resolve().parents[1]
for _package_root in (_ROOT / "packages").iterdir():
    _source = _package_root / "src"
    if _source.is_dir():
        sys.path.insert(0, str(_source))

from rles_data.core import DataError  # noqa: E402
from rles_data.features import Scaler  # noqa: E402
from rles_gym_bridge.historical import HistoricalEnvironmentFactory  # noqa: E402
from rles_orchestrator.application import (  # noqa: E402
    _file_hash,
    _verify_file,
    _write_verified,
    backtest,
    prepare,
    rollout,
    train,
)
from rles_orchestrator.research_access import record_development  # noqa: E402


def available_memory() -> int:
    class MemoryStatus(ctypes.Structure):
        _fields_ = [
            ("length", ctypes.c_ulong), ("load", ctypes.c_ulong),
            *[(name, ctypes.c_ulonglong) for name in (
                "total", "available", "total_page", "available_page",
                "total_virtual", "available_virtual", "available_extended",
            )],
        ]

    status = MemoryStatus()
    status.length = ctypes.sizeof(status)
    if not getattr(ctypes, "windll").kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise OSError("physical memory status unavailable")
    return int(status.available)


def coverage_for_window(original: dict, window: dict) -> dict:
    coverage = json.loads(json.dumps(original))
    for key in ("settlement_times_ns", "known_schedule_times_ns"):
        if key in coverage:
            margin = 24 * 60 * 60 * 1_000_000_000 if key == "known_schedule_times_ns" else 0
            coverage[key] = {
                symbol: [stamp for stamp in times
                         if window["start_ns"] - margin <= stamp < window["end_ns"] + margin]
                for symbol, times in coverage[key].items()
            }
    return coverage


def run(plan_path: Path, output: Path) -> dict:
    root = _ROOT
    plan_path = plan_path.resolve()
    plan_sha = _verify_file(plan_path)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if (plan.get("schema") != "rles.discount-half-life-probe.v2"
            or plan.get("status") != "preregistered_before_training"):
        raise DataError("a preregistered discount probe plan is required")
    if output.exists():
        raise DataError("output must be a new immutable directory")
    bundle_root = (root / plan["source_bundle"]).resolve()
    bundle_sha = _verify_file(bundle_root / "bundle.json")
    if bundle_sha != plan["source_bundle_sha256"]:
        raise DataError("frozen seed-102 source bundle changed")
    market_path = root / plan["source_market"]["path"]
    rules_path = root / plan["source_rules"]["path"]
    if (_file_hash(market_path) != plan["source_market"]["sha256"]
            or _file_hash(rules_path) != plan["source_rules"]["sha256"]):
        raise DataError("frozen market or scenario rules changed")
    if available_memory() < plan["minimum_start_free_bytes"]:
        raise MemoryError("less than the preregistered free-memory start threshold")

    bundle = json.loads((bundle_root / "bundle.json").read_text(encoding="utf-8"))
    environment = dict(bundle["environment"])
    environment["market_path"] = str(market_path)
    environment["rules_path"] = str(rules_path)
    environment["start_ns"] = plan["training_interval"]["start_ns"]
    environment["end_ns"] = plan["training_interval"]["end_ns"]
    environment["coverage_policy"] = coverage_for_window(
        environment["coverage_policy"], plan["training_interval"],
    )
    if not (bundle["environment"]["start_ns"] <= environment["start_ns"]
            < environment["end_ns"] <= bundle["environment"]["end_ns"]):
        raise DataError("preregistered training interval falls outside the archived source recipe")
    output.mkdir(parents=True)
    (output / "recipes").mkdir()
    (output / "reports").mkdir()
    _write_verified(output / "plan.json", {**plan, "plan_file_sha256": plan_sha})
    started = time.perf_counter()
    try:
        # One verified immutable source load is shared by the six sequential
        # learners. The original Train source snapshot remains hash-identical.
        base_config = {
            "environment": environment,
            "reward_scale": plan["training"]["reward_scale"],
            "output": str(output / "shared-training-recipe"),
        }
        prepared = prepare(base_config)
        if prepared.store.snapshot_hash != bundle["preprocessing"]["identities"]["dataset"]:
            raise DataError("prepared source identity differs from the frozen candidate")
        if available_memory() < plan["stop_free_bytes"]:
            raise MemoryError("resource floor reached after source preparation")
        records = []
        arms = plan["arms"]
        training = plan["training"]
        for seed in plan["seeds"]:
            for arm_name, arm in arms.items():
                if available_memory() < plan["stop_free_bytes"]:
                    raise MemoryError("resource floor approached; stopped before next training run")
                run_output = output / "runs" / f"{arm_name}-seed-{seed}"
                config = {
                    "environment": environment,
                    "route": training["route"],
                    "algorithm": training["algorithm"],
                    "steps": training["steps_per_seed_and_arm"],
                    "seed": seed,
                    "hidden_layers": training["hidden_layers"],
                    "normalize_observation": training["normalize_observation"],
                    "hyperparameters": {**training["hyperparameters"], "gamma": arm["gamma"]},
                    "output": str(run_output),
                    "reward_scale": training["reward_scale"],
                    "rl_device": training["rl_device"],
                    "rl_num_envs": training["rl_num_envs"],
                    "rl_episode_steps": training["rl_episode_steps"],
                    "checkpoint_interval_steps": 8192,
                    "enforce_worker_memory_limit": True,
                    "worker_ram_bytes": 2 * 1024**3,
                }
                _write_verified(output / "recipes" / f"{arm_name}-seed-{seed}.json", config)
                tick = time.perf_counter()
                result = train(config, evaluate=False, prepared_training=prepared)
                result_bundle_sha = _verify_file(run_output / "bundle.json")
                records.append({
                    "arm": arm_name,
                    "seed": seed,
                    "gamma": arm["gamma"],
                    "discount_half_life_minutes": arm["discount_half_life_minutes"],
                    "training_seconds": time.perf_counter() - tick,
                    "bundle": str(run_output.relative_to(output)),
                    "bundle_sha256": result_bundle_sha,
                    "training_coverage": result["training_coverage"],
                    "learning_sha256": _verify_file(run_output / "R1-learning.json"),
                    "actor_hash": json.loads((run_output / "bundle.json").read_text(encoding="utf-8"))["actor"]["actor_hash"],
                })
                del result
                gc.collect()

        returns = []
        flat_returns = {}
        diagnostic_config = {
            "environment": environment,
            "seed": plan["seeds"][0],
            "output": str(output / "development-access"),
        }
        for window_index, window in enumerate(plan["diagnostic_windows"]):
            if not (environment["end_ns"] <= window["start_ns"] < window["end_ns"]
                    <= bundle["training_end_ns"]):
                raise DataError("diagnostic window is not post-training Train data")
            record_development(diagnostic_config, window, label="gamma-half-life-probe")
            for mode in plan["execution_modes"]:
                if available_memory() < plan["stop_free_bytes"]:
                    raise MemoryError("resource floor approached before no-trade diagnostic")
                flat_factory = HistoricalEnvironmentFactory(**{
                    **environment,
                    **window,
                    "coverage_policy": coverage_for_window(environment["coverage_policy"], window),
                    "split": "train_inner",
                    "mode": mode,
                    "fit_scaler": False,
                    "scaler": prepared.scaler,
                    "reward_scale": training["reward_scale"],
                    "compact_report_snapshots": True,
                    "prepared_store": prepared.store,
                })
                flat_report, _ = rollout(
                    flat_factory,
                    lambda batch: np.zeros((len(batch), 1), dtype=np.float32),
                    seed=plan["seeds"][0], collect_observations=False,
                    include_account_rows=False,
                )
                flat_returns[(window_index, mode)] = flat_report["metrics"]["net_return"]

        for trained in records:
            trained_root = output / trained["bundle"]
            trained_bundle = json.loads((trained_root / "bundle.json").read_text(encoding="utf-8"))
            scaler_values = dict(trained_bundle["preprocessing"]["scaler"])
            scaler_values["mean"] = tuple(scaler_values["mean"])
            scaler_values["std"] = tuple(scaler_values["std"])
            scaler = Scaler(**scaler_values)
            if scaler.hash != prepared.scaler.hash:
                raise DataError("paired learner scaler differs from shared Train fit")
            for window_index, window in enumerate(plan["diagnostic_windows"]):
                for mode in plan["execution_modes"]:
                    if available_memory() < plan["stop_free_bytes"]:
                        raise MemoryError("resource floor approached; stopped before next diagnostic")
                    factory = HistoricalEnvironmentFactory(**{
                        **environment,
                        **window,
                        "coverage_policy": coverage_for_window(environment["coverage_policy"], window),
                        "split": "train_inner",
                        "mode": mode,
                        "fit_scaler": False,
                        "scaler": scaler,
                        "reward_scale": training["reward_scale"],
                        "compact_report_snapshots": True,
                        "prepared_store": prepared.store,
                    })
                    config = {**diagnostic_config, "seed": trained["seed"]}
                    tick = time.perf_counter()
                    report = backtest(
                        config, trained_root, factory=factory,
                        allow_execution_stress=mode == "stress", summary_only=True,
                    )
                    path = output / "reports" / (
                        f"{trained['arm']}-seed-{trained['seed']}-window-{window_index}-{mode}.json"
                    )
                    report_sha = _write_verified(path, report)
                    returns.append({
                        "arm": trained["arm"], "seed": trained["seed"],
                        "window": window_index, "mode": mode,
                        "net_return": report["metrics"]["net_return"],
                        "max_drawdown": report["metrics"]["max_drawdown"],
                        "total_fee": report["metrics"]["total_fee"],
                        "total_funding": report["metrics"]["total_funding"],
                        "total_slippage_cost": report["metrics"]["total_slippage_cost"],
                        "terminal_flat": report["terminal_flat"],
                        "report": str(path.relative_to(output)),
                        "report_sha256": report_sha,
                        "seconds": time.perf_counter() - tick,
                    })
        by_key = {(r["arm"], r["seed"], r["window"], r["mode"]): r["net_return"] for r in returns}
        paired = []
        for seed in plan["seeds"]:
            for mode in plan["execution_modes"]:
                deltas = [
                    by_key[("six_hour_half_life", seed, window_index, mode)]
                    - by_key[("control", seed, window_index, mode)]
                    for window_index in range(len(plan["diagnostic_windows"]))
                ]
                paired.append({"seed": seed, "mode": mode, "window_deltas": deltas,
                               "mean_delta": sum(deltas) / len(deltas),
                               "positive_mean": sum(deltas) > 0})
        flat_checks = []
        for mode in plan["execution_modes"]:
            for window_index in range(len(plan["diagnostic_windows"])):
                mean_return = sum(
                    by_key[("six_hour_half_life", seed, window_index, mode)] for seed in plan["seeds"]
                ) / len(plan["seeds"])
                flat = flat_returns[(window_index, mode)]
                flat_checks.append({"window": window_index, "mode": mode,
                                    "treatment_seed_mean": mean_return,
                                    "no_trade_return": flat,
                                    "beats_no_trade": mean_return > flat})
        paired_gate = all(row["positive_mean"] for row in paired)
        flat_gate = all(sum(row["beats_no_trade"] for row in flat_checks if row["mode"] == mode) >= 2
                        for mode in plan["execution_modes"])
        summary = {
            "schema": "rles.discount-half-life-probe-result.v1",
            "plan_sha256": plan_sha,
            "source_bundle_sha256": bundle_sha,
            "market_sha256": _file_hash(market_path),
            "rules_sha256": _file_hash(rules_path),
            "split": "train_inner",
            "formal_evaluation_queries": 0,
            "scaler_refitted_for_diagnostics": False,
            "historical_economics": "scenario-only",
            "training_runs": records,
            "returns": returns,
            "paired_deltas": paired,
            "no_trade_checks": flat_checks,
            "gate": "PASS_C_MECHANISM_TO_FIVE_SEED_CONFIRMATION" if paired_gate and flat_gate
                    else "FAIL_NO_CROSS_WINDOW_MECHANISM_ADVANTAGE",
            "candidate_promotion": False,
            "budget_expansion_allowed": False,
            "elapsed_seconds": time.perf_counter() - started,
        }
        _write_verified(output / "summary.json", summary)
        return summary
    except BaseException as exc:
        _write_verified(output / "FAILED.json", {"error": type(exc).__name__, "detail": str(exc)[:2000],
                                                  "completed_training_runs": len(locals().get("records", [])),
                                                  "elapsed_seconds": time.perf_counter() - started})
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.plan, args.output)
    print(json.dumps({"gate": result["gate"], "training_runs": len(result["training_runs"]),
                      "diagnostics": len(result["returns"])}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
