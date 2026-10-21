"""Evaluate frozen paired gamma-probe models on preregistered Train windows."""
from __future__ import annotations

import argparse
import ctypes
import json
from pathlib import Path
import sqlite3
import sys
import time

import numpy as np

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
)
from rles_orchestrator.research_access import market_key, record_development  # noqa: E402


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


def _assert_no_formal_overlap(registry_path: Path, market: str, windows: list[dict]) -> None:
    connection = sqlite3.connect(f"file:{registry_path.as_posix()}?mode=ro", uri=True)
    try:
        formal = connection.execute(
            "SELECT DISTINCT start_ns,end_ns FROM access WHERE market=? "
            "AND purpose IN ('test','holdout')", (market,),
        ).fetchall()
    finally:
        connection.close()
    for window in windows:
        if any(window["start_ns"] < end and window["end_ns"] > start for start, end in formal):
            raise DataError("preregistered Train interval overlaps a reserved Test/Holdout")


def evaluate(plan_path: Path, output: Path) -> dict:
    plan_path = plan_path.resolve()
    plan_sha = _verify_file(plan_path)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if (plan.get("schema") != "rles.discount-half-life-evaluation.v1"
            or plan.get("status") != "preregistered_before_any_model_diagnostic"):
        raise DataError("a frozen evaluation amendment is required")
    if output.exists():
        raise DataError("output must be a new immutable directory")
    training_plan_path = _ROOT / plan["training_plan"]
    training_plan_sha = _verify_file(training_plan_path)
    if training_plan_sha != plan["training_plan_sha256"]:
        raise DataError("paired training plan changed after model creation")
    manifest_path = _ROOT / plan["trained_run_manifest"]
    manifest_sha = _verify_file(manifest_path)
    if manifest_sha != plan["trained_run_manifest_sha256"]:
        raise DataError("trained model manifest changed")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    trained_root = (_ROOT / plan["trained_runs_root"]).resolve()
    source_plan = json.loads(training_plan_path.read_text(encoding="utf-8"))
    bundle_root = (_ROOT / source_plan["source_bundle"]).resolve()
    if _verify_file(bundle_root / "bundle.json") != source_plan["source_bundle_sha256"]:
        raise DataError("frozen source bundle changed")
    market_path = _ROOT / source_plan["source_market"]["path"]
    rules_path = _ROOT / source_plan["source_rules"]["path"]
    for path, key in ((market_path, "source_market"), (rules_path, "source_rules")):
        if _file_hash(path) != source_plan[key]["sha256"]:
            raise DataError("frozen source market or rules changed")

    full_environment = json.loads((bundle_root / "bundle.json").read_text(encoding="utf-8"))["environment"]
    full_environment = dict(full_environment)
    full_environment["market_path"] = str(market_path)
    full_environment["rules_path"] = str(rules_path)
    train_environment = dict(full_environment)
    train_environment["start_ns"] = source_plan["training_interval"]["start_ns"]
    train_environment["end_ns"] = source_plan["training_interval"]["end_ns"]
    train_environment["coverage_policy"] = coverage_for_window(
        full_environment["coverage_policy"], source_plan["training_interval"],
    )
    if available_memory() < plan["minimum_start_free_bytes"]:
        raise MemoryError("less than the preregistered free-memory start threshold")
    _assert_no_formal_overlap(
        _ROOT / ".artifacts/research-access.sqlite", market_key(full_environment["instrument"]),
        [source_plan["training_interval"], *plan["diagnostic_windows"]],
    )

    output.mkdir(parents=True)
    (output / "reports").mkdir()
    _write_verified(output / "plan.json", {**plan, "plan_file_sha256": plan_sha})
    started = time.perf_counter()
    try:
        prepare_config = {
            "environment": train_environment,
            "reward_scale": source_plan["training"]["reward_scale"],
            "output": str(output / "development-access"),
        }
        prepared = prepare(prepare_config)
        if prepared.store.snapshot_hash != "258834aba27e77b0c97fbb4a5ca9fc6c18699f2f8d0d56c77812d1946bbc1adb":
            raise DataError("prepared market/rules identity differs from the registered source")
        if available_memory() < plan["stop_free_bytes"]:
            raise MemoryError("resource floor reached after source preparation")

        run_records = []
        for row in manifest["runs"]:
            run_root = (_ROOT / row["bundle"]).parent
            try:
                run_root.relative_to(trained_root)
            except ValueError as exc:
                raise DataError("trained run escaped the preregistered artifact root") from exc
            bundle_path = run_root / "bundle.json"
            if _verify_file(bundle_path) != row["bundle_sha256"]:
                raise DataError("paired training bundle changed after manifest freeze")
            learning_path = run_root / "R1-learning.json"
            if _verify_file(learning_path) != row["learning_sha256"]:
                raise DataError("paired learning diagnostics changed after manifest freeze")
            completed = json.loads((run_root / "COMPLETED.json").read_text(encoding="utf-8"))
            if completed["bundle_sha256"] != row["bundle_sha256"]:
                raise DataError("completed run marker does not bind the frozen bundle")
            run_records.append({**row, "run_root": str(run_root)})

        diagnostics_config = {
            "environment": full_environment,
            "seed": source_plan["seeds"][0],
            "output": str(output / "development-access"),
        }
        quality = []
        for window_index, window in enumerate(plan["diagnostic_windows"]):
            record_development(diagnostics_config, window, label="gamma-half-life-probe")
            factory = HistoricalEnvironmentFactory(**{
                **full_environment,
                **window,
                "coverage_policy": coverage_for_window(full_environment["coverage_policy"], window),
                "split": "train_inner",
                "fit_scaler": False,
                "scaler": prepared.scaler,
                "reward_scale": source_plan["training"]["reward_scale"],
                "compact_report_snapshots": True,
                "prepared_store": prepared.store,
            })
            grade = factory.quality.grade.value
            quality.append({"window": window_index, "start_ns": window["start_ns"],
                            "end_ns": window["end_ns"], "grade": grade,
                            "findings": [str(item) for item in factory.quality.findings]})
            if grade != "DQ-B":
                raise DataError(f"diagnostic window quality {grade} differs from the frozen DQ-B eligibility")
        _write_verified(output / "quality-preflight.json", {"windows": quality})

        flat_returns = {}
        for window_index, window in enumerate(plan["diagnostic_windows"]):
            for mode in plan["execution_modes"]:
                if available_memory() < plan["stop_free_bytes"]:
                    raise MemoryError("resource floor approached before no-trade rollout")
                factory = HistoricalEnvironmentFactory(**{
                    **full_environment, **window,
                    "coverage_policy": coverage_for_window(full_environment["coverage_policy"], window),
                    "split": "train_inner", "mode": mode, "fit_scaler": False,
                    "scaler": prepared.scaler,
                    "reward_scale": source_plan["training"]["reward_scale"],
                    "compact_report_snapshots": True, "prepared_store": prepared.store,
                })
                report, _ = rollout(
                    factory, lambda batch: np.zeros((len(batch), 1), dtype=np.float32),
                    seed=source_plan["seeds"][0], collect_observations=False,
                    include_account_rows=False,
                )
                flat_returns[(window_index, mode)] = report["metrics"]["net_return"]

        returns = []
        for trained in run_records:
            trained_bundle = json.loads((_ROOT / trained["bundle"]).read_text(encoding="utf-8"))
            scaler_values = dict(trained_bundle["preprocessing"]["scaler"])
            scaler_values["mean"] = tuple(scaler_values["mean"])
            scaler_values["std"] = tuple(scaler_values["std"])
            scaler = Scaler(**scaler_values)
            if scaler.hash != prepared.scaler.hash:
                raise DataError("trained model scaler differs from the frozen safe Train interval")
            for window_index, window in enumerate(plan["diagnostic_windows"]):
                for mode in plan["execution_modes"]:
                    if available_memory() < plan["stop_free_bytes"]:
                        raise MemoryError("resource floor approached; stopped before next model diagnostic")
                    factory = HistoricalEnvironmentFactory(**{
                        **full_environment, **window,
                        "coverage_policy": coverage_for_window(full_environment["coverage_policy"], window),
                        "split": "train_inner", "mode": mode, "fit_scaler": False,
                        "scaler": scaler,
                        "reward_scale": source_plan["training"]["reward_scale"],
                        "compact_report_snapshots": True, "prepared_store": prepared.store,
                    })
                    tick = time.perf_counter()
                    report = backtest(
                        diagnostics_config | {"seed": trained["seed"]}, trained["run_root"],
                        factory=factory, allow_execution_stress=mode == "stress",
                        summary_only=True,
                    )
                    report_path = output / "reports" / (
                        f"{trained['arm']}-seed-{trained['seed']}-window-{window_index}-{mode}.json"
                    )
                    report_sha = _write_verified(report_path, report)
                    returns.append({
                        "arm": trained["arm"], "seed": trained["seed"],
                        "window": window_index, "mode": mode,
                        "net_return": report["metrics"]["net_return"],
                        "max_drawdown": report["metrics"]["max_drawdown"],
                        "total_fee": report["metrics"]["total_fee"],
                        "total_funding": report["metrics"]["total_funding"],
                        "total_slippage_cost": report["metrics"]["total_slippage_cost"],
                        "terminal_flat": report["terminal_flat"],
                        "report": str(report_path.relative_to(output)),
                        "report_sha256": report_sha,
                        "seconds": time.perf_counter() - tick,
                    })

        by_key = {(row["arm"], row["seed"], row["window"], row["mode"]): row["net_return"]
                  for row in returns}
        paired = []
        for seed in source_plan["seeds"]:
            for mode in plan["execution_modes"]:
                deltas = [by_key[("six_hour_half_life", seed, i, mode)]
                          - by_key[("control", seed, i, mode)]
                          for i in range(len(plan["diagnostic_windows"]))]
                paired.append({"seed": seed, "mode": mode, "window_deltas": deltas,
                               "mean_delta": sum(deltas) / len(deltas),
                               "positive_mean": sum(deltas) > 0})
        flat_checks = []
        for mode in plan["execution_modes"]:
            for i in range(len(plan["diagnostic_windows"])):
                treatment_mean = sum(by_key[("six_hour_half_life", s, i, mode)]
                                     for s in source_plan["seeds"]) / len(source_plan["seeds"])
                flat_checks.append({"window": i, "mode": mode,
                                    "treatment_seed_mean": treatment_mean,
                                    "no_trade_return": flat_returns[(i, mode)],
                                    "beats_no_trade": treatment_mean > flat_returns[(i, mode)]})
        passed = (all(row["positive_mean"] for row in paired)
                  and all(sum(row["beats_no_trade"] for row in flat_checks if row["mode"] == mode) >= 2
                          for mode in plan["execution_modes"]))
        summary = {
            "schema": "rles.discount-half-life-probe-result.v1",
            "evaluation_plan_sha256": plan_sha,
            "training_plan_sha256": training_plan_sha,
            "trained_run_manifest_sha256": manifest_sha,
            "source_dataset_hash": prepared.store.snapshot_hash,
            "split": "train_inner",
            "formal_evaluation_queries": 0,
            "quality_preflight": quality,
            "scaler_refitted_for_diagnostics": False,
            "historical_economics": "scenario-only",
            "training_runs": run_records,
            "returns": returns,
            "paired_deltas": paired,
            "no_trade_checks": flat_checks,
            "gate": "PASS_C_MECHANISM_TO_FIVE_SEED_CONFIRMATION" if passed
                    else "FAIL_NO_CROSS_WINDOW_MECHANISM_ADVANTAGE",
            "candidate_promotion": False,
            "budget_expansion_allowed": False,
            "elapsed_seconds": time.perf_counter() - started,
        }
        _write_verified(output / "summary.json", summary)
        return summary
    except BaseException as exc:
        _write_verified(output / "FAILED.json", {"error": type(exc).__name__,
                                                  "detail": str(exc)[:2000],
                                                  "elapsed_seconds": time.perf_counter() - started})
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate(args.plan, args.output)
    print(json.dumps({"gate": result["gate"], "training_runs": len(result["training_runs"]),
                      "diagnostics": len(result["returns"])}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
