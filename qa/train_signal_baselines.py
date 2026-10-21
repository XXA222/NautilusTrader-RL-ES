"""Bounded, preregistered Train-only comparisons of frozen RL and causal rules.

No fit, optimizer update, Test or Holdout evaluation is performed. Full source
identity is verified once, then immutable history is shared between factories.
"""
from __future__ import annotations

import argparse
from collections import deque
import ctypes
from dataclasses import asdict
import json
from pathlib import Path
import time

import numpy as np

from rles_data.core import CuratedStore, DataError, MINUTE_NS
from rles_data.io import read_curated
from rles_gym_bridge.historical import HistoricalEnvironmentFactory, load_rules
from rles_orchestrator.application import _file_hash, _verify_file, _write_verified, backtest, rollout
from rles_orchestrator.research_access import record_development
from rles_orchestrator.wfo_application import _scaler
from rles_research.baselines import BaselinePolicy, BaselineSpec


def available_memory() -> int:
    class MemoryStatus(ctypes.Structure):
        _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong),
                    *[(name, ctypes.c_ulonglong) for name in (
                        "total", "available", "total_page", "available_page",
                        "total_virtual", "available_virtual", "available_extended")]]
    status = MemoryStatus()
    status.length = ctypes.sizeof(status)
    if not getattr(ctypes, "windll").kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise OSError("physical memory status unavailable")
    return int(status.available)


def coverage_for_window(original: dict, window: dict) -> dict:
    coverage = json.loads(json.dumps(original))
    for key in ("settlement_times_ns", "known_schedule_times_ns"):
        if key in coverage:
            margin = 24 * 60 * MINUTE_NS if key == "known_schedule_times_ns" else 0
            coverage[key] = {symbol: [stamp for stamp in times
                if window["start_ns"] - margin <= stamp < window["end_ns"] + margin]
                for symbol, times in coverage[key].items()}
    return coverage


def compare(plan_path: Path, output: Path) -> dict:
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if plan["split"] != "train_inner" or output.exists():
        raise DataError("Train-only plan and a new immutable output directory required")
    bundle_root = Path(plan["bundle"])
    if _verify_file(bundle_root / "bundle.json") != plan["bundle_sha256"]:
        raise DataError("preregistered bundle changed")
    bundle = json.loads((bundle_root / "bundle.json").read_text(encoding="utf-8"))
    environment = bundle["environment"]
    config = {"environment": environment, "output": str(output), "seed": plan["seed"]}
    for window in plan["windows"]:
        if not environment["start_ns"] <= window["start_ns"] < window["end_ns"] <= bundle["training_end_ns"]:
            raise DataError("diagnostic window crosses frozen Train boundary")
        record_development(config, window, label="train-signal-baselines")
    if available_memory() < plan["minimum_start_free_bytes"]:
        raise MemoryError("insufficient free memory for verified source loading")
    output.mkdir(parents=True)
    _write_verified(output / "plan.json", {**plan, "plan_file_sha256": _file_hash(plan_path)})
    try:
        source = plan["source"]
        for name in ("market", "rules"):
            if _file_hash(Path(source[name + "_path"])) != source[name + "_sha256"]:
                raise DataError("archived source checksum changed")
        started = time.perf_counter()
        rules = load_rules(source["rules_path"])
        store = CuratedStore((*read_curated(source["market_path"]).events, *rules))
        if store.snapshot_hash != bundle["preprocessing"]["identities"]["dataset"]:
            raise DataError("archived market/rules do not match the frozen model")
        print(json.dumps({"stage": "source-verified", "seconds": time.perf_counter() - started,
                          "events": len(store.events), "free_bytes": available_memory()}), flush=True)
        scal = _scaler(bundle_root)
        results = []
        for window_index, window in enumerate(plan["windows"]):
            for mode in plan["modes"]:
                coverage = coverage_for_window(environment["coverage_policy"], window)
                factory = HistoricalEnvironmentFactory(**{
                    **environment, **window, "coverage_policy": coverage,
                    "split": "train_inner", "mode": mode, "scaler": scal, "fit_scaler": False,
                    "reward_scale": bundle.get("reward_scale", 1.0),
                    "compact_report_snapshots": True, "prepared_store": store,
                })
                for policy_name in plan["policies"]:
                    if available_memory() < plan["stop_free_bytes"]:
                        raise MemoryError("resource floor approached; stopped before the next episode")
                    tick = time.perf_counter()
                    if policy_name == "frozen_rl":
                        report = backtest(config, bundle_root, factory=factory,
                                          allow_execution_stress=mode == "stress", summary_only=False,
                                          include_execution_trace=True)
                    else:
                        policy = BaselinePolicy(BaselineSpec(policy_name, plan["seed"],
                                                target_long=plan["target"],
                                                momentum_lookback=plan["momentum_lookback"]))
                        visible_bars = iter(event for event in store.events
                                            if event.kind == "ohlcv_1m"
                                            and event.payload["instrument"] == factory.instrument)
                        next_bar = next(visible_bars, None)
                        closes: deque[float] = deque(maxlen=plan["momentum_lookback"] + 1)
                        cursor = 0

                        def predict(batch):
                            nonlocal next_bar, cursor
                            if len(batch) != 1:
                                raise DataError("baseline requires one deterministic account")
                            timestamp = factory.rows[cursor].decision_ts_ns
                            while (next_bar is not None and next_bar.available_at_ns is not None
                                   and next_bar.available_at_ns <= timestamp):
                                closes.append(float(next_bar.payload["close"]))
                                next_bar = next(visible_bars, None)
                            cursor += 1
                            return np.array([[policy.action(list(closes))]], dtype=np.float32)

                        report, _ = rollout(factory, predict, seed=plan["seed"],
                                            soft_parameters=bundle["soft_multipliers"],
                                            collect_observations=False, include_account_rows=True,
                                            include_execution_trace=True)
                        report["dataset_quality"] = asdict(factory.quality)
                    path = output / f"window-{window_index}-{mode}-{policy_name}.json"
                    report_sha = _write_verified(path, report)
                    row = {"window": window_index, "mode": mode, "policy": policy_name,
                           "metrics": report["metrics"], "terminal_flat": report["terminal_flat"],
                           "action_diagnostics": report["action_diagnostics"],
                           "report_sha256": report_sha, "report": path.name,
                           "seconds": time.perf_counter() - tick}
                    results.append(row)
                    print(json.dumps({key: row[key] for key in ("window", "mode", "policy", "seconds")}), flush=True)
        pairs = []
        for index, _window in enumerate(plan["windows"]):
            for mode in plan["modes"]:
                subset = {row["policy"]: row["metrics"]["net_return"] for row in results
                          if row["window"] == index and row["mode"] == mode}
                pairs.append({"window": index, "mode": mode, "net_returns": subset,
                              "rl_beats_flat": subset["frozen_rl"] > subset["B0_no_trade"],
                              "rl_beats_all_baselines": subset["frozen_rl"] > max(
                                  value for name, value in subset.items() if name != "frozen_rl")})
        passed = all(pair["rl_beats_all_baselines"] for pair in pairs)
        summary = {"schema": "rles.train-signal-baselines.v1", "split": "train_inner",
                   "formal_evaluation_queries": 0, "scaler_refitted": False,
                   "scope": "exploratory frozen-policy diagnostics; internal data are development data",
                   "historical_economics": "scenario-only; historical rule validity unproven",
                   "risk_comparison": "same budget/caps; realized exposures reported separately",
                   "candidate_gate": "PASS_BASELINE_SCREEN_ONLY" if passed else "FAIL_NO_CROSS_WINDOW_BASELINE_ADVANTAGE",
                   "budget_expansion_allowed": passed, "pairs": pairs, "runs": results}
        _write_verified(output / "summary.json", summary)
        return summary
    except BaseException as exc:
        _write_verified(output / "FAILED.json", {"error": type(exc).__name__, "detail": str(exc)})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = compare(args.plan, args.output)
    print(json.dumps({"candidate_gate": summary["candidate_gate"], "runs": len(summary["runs"])}))


if __name__ == "__main__":
    main()
