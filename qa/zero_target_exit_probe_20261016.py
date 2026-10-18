"""Run a deterministic Train-only contrast for zero-target execution semantics."""
from __future__ import annotations

import argparse
from collections import Counter
import gc
import json
from pathlib import Path

import numpy as np

from rles_data.core import CuratedStore, DataError
from rles_data.io import read_curated
from rles_gym_bridge.historical import HistoricalEnvironmentFactory, load_rules
from rles_orchestrator.application import _file_hash, _verify_file, _write_verified
from rles_orchestrator.research_access import record_development
from rles_orchestrator.wfo_application import _scaler
from rles_research.native_metrics import native_report

from profile_native_history_growth_20261016 import available_memory, coverage_for_window


def target_at(step: int, action_path: dict) -> float:
    if step == 0:
        return float(action_path["step_0"])
    if step <= 21:
        return float(action_path["steps_1_through_21"])
    return float(action_path["step_22_onward"])


def one_run(plan: dict, store: CuratedStore, bundle: dict, scaler, *, policy: str, seed: int,
            output: Path) -> dict:
    environment = bundle["environment"]
    window = plan["window"]
    coverage = coverage_for_window(environment["coverage_policy"], window)
    execution = {**environment["execution"], "zero_target_policy": policy}
    factory = HistoricalEnvironmentFactory(**{
        **environment, "start_ns": window["start_ns"], "end_ns": window["end_ns"],
        "coverage_policy": coverage, "execution": execution,
        "split": "train_inner", "mode": "base", "scaler": scaler,
        "fit_scaler": False, "compact_report_snapshots": True,
        "prepared_store": store,
    })
    env = factory.make_env(soft_multipliers=bundle["soft_multipliers"])
    trajectory = []
    plan_counts: Counter[str] = Counter()
    zero_noop_count = 0
    close_plan_count = 0
    first_post_zero_flat_step = None
    final_info = None
    try:
        observation, _reset_info = env.reset(seed=seed)
        scoring_steps = len(factory.rows) - 3
        for step in range(scoring_steps):
            raw = target_at(step, plan["action_path"])
            observation, _reward, terminated, truncated, info = env.step(
                np.asarray([raw], dtype=np.float32))
            action = info["execution_plan_action"]
            reason = info["execution_plan_reason"]
            plan_counts[f"{action}/{reason}"] += 1
            actual = float(info["policy_executed_action_pre_terminal"])
            if step >= 22 and step < 22 + plan["max_probe_steps_after_zero"]:
                if reason == "deadband":
                    zero_noop_count += 1
                if reason == "policy_flat":
                    close_plan_count += 1
                if first_post_zero_flat_step is None and abs(actual) < 0.01 and not info["terminal_settled"]:
                    first_post_zero_flat_step = step
            if step < 30 or step < 22 + plan["max_probe_steps_after_zero"]:
                trajectory.append({
                    "step": step, "requested": raw,
                    "soft_target": info["risk_soft_target"],
                    "hard_risk_target": info["risk_adjusted_target"],
                    "quantity_rounded_target_exposure": info["quantity_rounded_target_exposure"],
                    "plan_action": action, "plan_reason": reason,
                    "actual_native_exposure": actual,
                    "terminal_settled": bool(info["terminal_settled"]),
                })
            final_info = info
            if terminated or truncated:
                break
        if final_info is None:
            raise DataError("zero-target probe emitted no native decisions")
        report = native_report(
            factory.snapshots, fill_references=factory.fill_references,
            sample_interval_ns=60_000_000_000,
            allow_price_improvement=False, include_account_rows=False,
        )
        step21 = next((row for row in trajectory if row["step"] == 21), None)
        step22 = next((row for row in trajectory if row["step"] == 22), None)
        profile = {
            "policy": policy, "seed": seed, "action_steps": len(factory.snapshots) - 3,
            "terminal_flat": report["terminal_flat"], "fill_count": len(report["fills"]),
            "fee_slippage_funding_metrics": report["metrics"],
            "step21_state_before_zero": step21, "step22_state_after_zero_request": step22,
            "first_flat_step_after_zero_before_terminal": first_post_zero_flat_step,
            "deadband_noop_steps_after_zero_within_probe": zero_noop_count,
            "explicit_policy_flat_plans_within_probe": close_plan_count,
            "plan_counts": dict(sorted(plan_counts.items())),
            "trajectory_first_82_steps": trajectory,
            "interpretation": "semantic control only; terminal settlement is excluded from early-exit evidence",
        }
        _write_verified(output / f"{policy}-seed-{seed}.json", profile)
        return profile
    finally:
        env.close()


def run(plan_path: Path, output: Path) -> dict:
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if plan.get("split") != "train_inner" or output.exists():
        raise DataError("Train-inner preregistration and a new output directory are required")
    marker = plan_path.with_name(plan_path.name + ".sha256")
    if not marker.is_file() or marker.read_text(encoding="ascii").strip() != _file_hash(plan_path):
        raise DataError("zero-target probe plan checksum mismatch")
    bundle_root = Path(plan["bundle"])
    if _verify_file(bundle_root / "bundle.json") != plan["bundle_sha256"]:
        raise DataError("frozen bundle checksum changed")
    bundle = json.loads((bundle_root / "bundle.json").read_text(encoding="utf-8"))
    if plan["window"]["end_ns"] > bundle["training_end_ns"] or available_memory() < 10 * 1024**3:
        raise MemoryError("probe exceeds frozen Train or start memory boundary")
    for name in ("market", "rules"):
        if _file_hash(Path(plan["source"][name + "_path"])) != plan["source"][name + "_sha256"]:
            raise DataError(f"frozen {name} source checksum changed")
    output.mkdir(parents=True)
    _write_verified(output / "preregistration.json", {
        "plan_path": str(plan_path.resolve()), "plan_sha256": _file_hash(plan_path), "plan": plan})
    try:
        rules = load_rules(plan["source"]["rules_path"])
        store = CuratedStore((*read_curated(plan["source"]["market_path"]).events, *rules))
        if store.snapshot_hash != plan["dataset_sha256"]:
            raise DataError("source snapshot does not match frozen bundle")
        record_development({"environment": bundle["environment"], "output": str(output), "seed": 0},
                           {"start_ns": plan["window"]["source_start_ns"],
                            "end_ns": plan["window"]["end_ns"]},
                           label="zero-target-exit-semantic-probe")
        scaler = _scaler(bundle_root)
        runs = []
        for seed in plan["seeds"]:
            for policy in plan["modes"]:
                if available_memory() < plan["stop_free_bytes"]:
                    raise MemoryError("resource stop floor reached before next paired episode")
                gc.collect()
                profile = one_run(plan, store, bundle, scaler, policy=policy, seed=seed, output=output)
                runs.append(profile)
                print(json.dumps({
                    "seed": seed, "policy": policy, "fills": profile["fill_count"],
                    "zero_state": None if profile["step21_state_before_zero"] is None else
                    profile["step21_state_before_zero"]["actual_native_exposure"],
                    "step22_reason": None if profile["step22_state_after_zero_request"] is None else
                    profile["step22_state_after_zero_request"]["plan_reason"],
                    "early_flat_step": profile["first_flat_step_after_zero_before_terminal"],
                }), flush=True)
        comparisons = []
        for seed in plan["seeds"]:
            pair = {row["policy"]: row for row in runs if row["seed"] == seed}
            legacy, explicit = pair["deadband"], pair["close"]
            before = legacy["step21_state_before_zero"]
            comparisons.append({
                "seed": seed,
                "same_pre_zero_exposure": (
                    before is not None and explicit["step21_state_before_zero"] is not None
                    and abs(before["actual_native_exposure"]
                            - explicit["step21_state_before_zero"]["actual_native_exposure"]) < 1e-9),
                "deadband_zero_plan": (None if legacy["step22_state_after_zero_request"] is None else
                                        legacy["step22_state_after_zero_request"]["plan_reason"]),
                "close_zero_plan": (None if explicit["step22_state_after_zero_request"] is None else
                                    explicit["step22_state_after_zero_request"]["plan_reason"]),
                "deadband_early_exit": legacy["first_flat_step_after_zero_before_terminal"],
                "close_early_exit": explicit["first_flat_step_after_zero_before_terminal"],
                "semantic_condition_met": (
                    before is not None and 0.01 < abs(before["actual_native_exposure"]) < 0.10
                    and legacy["step22_state_after_zero_request"] is not None
                    and legacy["step22_state_after_zero_request"]["plan_reason"] == "deadband"
                    and explicit["step22_state_after_zero_request"] is not None
                    and explicit["step22_state_after_zero_request"]["plan_reason"] == "policy_flat"
                    and explicit["first_flat_step_after_zero_before_terminal"] is not None),
            })
        summary = {
            "schema": "rles.zero-target-exit-probe-results.v1",
            "split": "train_inner", "scope": "deterministic execution semantics; not strategy performance",
            "bundle_sha256": plan["bundle_sha256"], "plan_sha256": _file_hash(plan_path),
            "dataset_snapshot_hash": store.snapshot_hash,
            "comparison_count": len(comparisons), "comparisons": comparisons,
            "semantic_probe_passed": len(comparisons) == len(plan["seeds"])
                and all(row["semantic_condition_met"] for row in comparisons),
            "runs": [{k: row[k] for k in (
                "policy", "seed", "fill_count", "terminal_flat", "plan_counts",
                "step21_state_before_zero", "step22_state_after_zero_request",
                "first_flat_step_after_zero_before_terminal",
            )} for row in runs],
            "limitations": [
                "The action path is deterministic and hand-specified; it is not an RL policy.",
                "All data and execution economics are previously used Train/scenario-only inputs.",
                "No profitability or independent generalization claim is made.",
            ],
        }
        _write_verified(output / "summary.json", summary)
        return summary
    except BaseException as exc:
        _write_verified(output / "FAILED.json", {"error": type(exc).__name__, "detail": str(exc)})
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.plan, args.output)
    print(json.dumps({"semantic_probe_passed": result["semantic_probe_passed"],
                      "comparisons": result["comparison_count"]}))


if __name__ == "__main__":
    main()
