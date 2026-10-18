"""Compare continuous and daily force-flat accounts on the same frozen Train-inner path."""
from __future__ import annotations

import argparse
from decimal import Decimal
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from rles_data.core import CuratedStore, DataError
from rles_data.io import read_curated
from rles_gym_bridge.historical import (
    HistoricalEnvironmentFactory,
    load_rules,
    partition_decision_lanes,
)
from rles_orchestrator.application import _file_hash, _verify_file, _write_verified, rollout
from rles_orchestrator.research_access import record_development
from rles_orchestrator.wfo_application import _scaler
from profile_native_history_growth_20261016 import available_memory, coverage_for_window

DAY_NS = 86_400 * 1_000_000_000
TARGET = 0.25
EPISODE_COUNT = 90


def _predict(_batch: np.ndarray) -> np.ndarray:
    return np.full((1, 1), TARGET, dtype=np.float32)


def _held_seconds(snapshots) -> float:
    held_ns = 0
    for current, following in zip(snapshots, snapshots[1:]):
        if current.positions:
            held_ns += max(0, int(following.decision_ts_ns) - int(current.decision_ts_ns))
    return held_ns / 1_000_000_000


def _position_identities(snapshots) -> set[str]:
    return {str(position.get("position_id", "")) for snapshot in snapshots
            for position in snapshot.positions if position.get("position_id")}


def run(plan_path: Path, output: Path) -> dict:
    plan_bytes = plan_path.read_bytes()
    plan_sha = sha256(plan_bytes).hexdigest()
    plan_sha_path = plan_path.with_suffix(".json.sha256")
    if not plan_sha_path.is_file() or plan_sha_path.read_text(encoding="ascii") != plan_sha:
        raise DataError("frozen episode-horizon preregistration checksum mismatch")
    plan = json.loads(plan_bytes)
    if (plan.get("schema") != "rles.episode-horizon-full-coverage-prereg.v1"
            or plan.get("split") != "train_inner" or output.exists()):
        raise DataError("requires frozen Train-inner inputs and a new output directory")
    root = Path(__file__).resolve().parents[1]
    for relative, expected_sha in plan["code_sha256"].items():
        if _file_hash(root / relative) != expected_sha:
            raise DataError(f"frozen code changed: {relative}")
    frozen_plan_path = Path(plan["frozen_plan_path"])
    if _file_hash(frozen_plan_path) != plan["frozen_plan_sha256"]:
        raise DataError("frozen episode-horizon source plan checksum changed")
    frozen_plan = json.loads(frozen_plan_path.read_text(encoding="utf-8"))
    source_plan_path = Path(plan["source_plan_path"])
    if (_file_hash(source_plan_path) != plan["source_plan_sha256"]
            or Path(frozen_plan["source_plan"]).resolve() != source_plan_path.resolve()
            or frozen_plan["source_plan_sha256"] != plan["source_plan_sha256"]
            or frozen_plan["source"] != plan["source"]):
        raise DataError("frozen source plan checksum changed")
    source_plan = json.loads(source_plan_path.read_text(encoding="utf-8"))
    free_bytes = available_memory()
    if free_bytes < int(source_plan["minimum_start_free_bytes"]):
        raise MemoryError("free memory is below the frozen Train-inner profile start floor")
    bundle_root = Path(source_plan["bundle"])
    if _verify_file(bundle_root / "bundle.json") != plan["bundle_sha256"]:
        raise DataError("frozen policy bundle checksum changed")
    bundle = json.loads((bundle_root / "bundle.json").read_text(encoding="utf-8"))
    seed = int(plan["seed"])
    source = plan["source"]
    start_ns = int(source_plan["windows"][0]["start_ns"])
    end_ns = start_ns + 90 * DAY_NS
    if end_ns > int(bundle["training_end_ns"]):
        raise DataError("episode-horizon comparison exceeds the frozen Train boundary")
    window = {"start_ns": start_ns, "end_ns": end_ns}
    if (start_ns != int(plan["start_ns"]) or end_ns != int(plan["end_ns"])
            or plan["episode_count"] != EPISODE_COUNT
            or Decimal(str(plan["target_exposure"])) != Decimal(str(TARGET))
            or plan["partition_policy"] != "contiguous_rank_lanes_full_cover_v1"):
        raise DataError("frozen episode-horizon scope or partition policy changed")

    environment = bundle["environment"]
    config = {"environment": environment, "output": str(output), "seed": seed}
    record_development(config, {"start_ns": start_ns, "end_ns": end_ns},
                       label=plan["experiment_id"])
    # Reserve the Train-inner interval before hashing or opening any market/rule payload.
    for key in ("market", "rules"):
        if _file_hash(Path(source[f"{key}_path"])) != source[f"{key}_sha256"]:
            raise DataError(f"frozen {key} source checksum changed")
    for relative, expected_sha in plan["code_sha256"].items():
        if _file_hash(root / relative) != expected_sha:
            raise DataError(f"frozen code changed after reservation: {relative}")
    output.mkdir(parents=True)
    _write_verified(output / "plan.json", {
        "schema": "rles.episode-horizon-plan.v2",
        "source_plan": str(plan_path.resolve()), "source_plan_sha256": _file_hash(plan_path),
        "bundle_sha256": plan["bundle_sha256"],
        "source_hashes": {key: source[f"{key}_sha256"] for key in ("market", "rules")},
        "code_sha256": plan["code_sha256"],
        "split": "train_inner", "start_ns": start_ns, "end_ns": end_ns,
        "policy": {"kind": "fixed_target", "target_exposure": TARGET, "seed": seed},
        "arms": {"T0": "one continuous 90-day account, terminal close only",
                 "T1": f"{EPISODE_COUNT} contiguous rank-exclusive lanes, each terminally force-flat"},
        "partition_policy": plan["partition_policy"],
        "expected_decision_steps": plan["expected_decision_steps"],
        "economics": "scenario-only; dollar PnL sums independent episode accounts and is not compounded",
        "purpose": "execution-horizon mechanism diagnosis; no model selection or profitability gate",
    })

    store = CuratedStore((*read_curated(source["market_path"]).events,
                          *load_rules(source["rules_path"])))
    if store.snapshot_hash != bundle["preprocessing"]["identities"]["dataset"]:
        raise DataError("frozen market/rule sources differ from the policy dataset identity")
    scaler = _scaler(bundle_root)
    coverage = coverage_for_window(environment["coverage_policy"], window)
    base_factory_args = {
        **environment,
        "coverage_policy": coverage,
        "start_ns": start_ns,
        "end_ns": end_ns,
        "split": "train_inner",
        "mode": "base",
        "scaler": scaler,
        "fit_scaler": False,
        "reward_scale": bundle.get("reward_scale", 1.0),
        "compact_report_snapshots": True,
        "prepared_store": store,
    }

    continuous_factory = HistoricalEnvironmentFactory(**base_factory_args)
    continuous_report, _ = rollout(
        continuous_factory, _predict, seed=seed,
        soft_parameters=bundle["soft_multipliers"], collect_observations=False,
        include_account_rows=False,
    )
    continuous_snapshots = tuple(continuous_factory.snapshots)
    continuous_initial = Decimal(str(continuous_snapshots[0].equity))
    continuous_final = Decimal(str(continuous_snapshots[-1].equity))
    continuous = {
        "decision_steps": continuous_report["action_diagnostics"]["count"],
        "initial_equity": str(continuous_initial), "final_equity": str(continuous_final),
        "net_pnl": str(continuous_final - continuous_initial),
        "fee": continuous_report["metrics"]["total_fee"],
        "funding": continuous_report["metrics"]["total_funding"],
        "fill_count": len(continuous_report["fills"]),
        "funding_event_count": len(continuous_report["funding"]),
        "held_seconds": _held_seconds(continuous_snapshots),
        "position_count": len(_position_identities(continuous_snapshots)),
        "terminal_flat": continuous_report["terminal_flat"],
    }

    segmented_factory = HistoricalEnvironmentFactory(**base_factory_args)
    full_steps = len(segmented_factory.rows) - 3
    if (full_steps != continuous_report["action_diagnostics"]["count"]
            or full_steps != int(plan["expected_decision_steps"])):
        raise DataError("frozen decision-step count differs from the Train-inner source")
    lanes = partition_decision_lanes(full_steps, EPISODE_COUNT)
    if (lanes[0][0] != 0 or lanes[-1][1] != full_steps
            or any(left[1] != right[0] for left, right in zip(lanes, lanes[1:]))):
        raise DataError("rank lanes do not exactly partition all Train-inner decisions")
    episodes = []
    minimum_free_bytes = free_bytes
    for rank, (lane_start, lane_end) in enumerate(lanes):
        episode_steps = lane_end - lane_start
        free_bytes = available_memory()
        minimum_free_bytes = min(minimum_free_bytes, free_bytes)
        if free_bytes < int(source_plan["stop_free_bytes"]):
            raise MemoryError("resource stop floor reached before the next daily episode")
        report, _ = rollout(
            segmented_factory, _predict, seed=(seed + rank) % 2**32,
            soft_parameters=bundle["soft_multipliers"], collect_observations=False,
            episode_rank=rank, episode_count=EPISODE_COUNT,
            episode_steps=episode_steps, include_account_rows=False,
            include_execution_trace=True,
        )
        snapshots = tuple(segmented_factory.snapshots)
        if (report["episode_window"]["start_offset"] != lane_start
                or report["episode_window"]["decision_steps"] != episode_steps):
            raise DataError("native Episode window differs from its frozen rank lane")
        initial = Decimal(str(snapshots[0].equity))
        final = Decimal(str(snapshots[-1].equity))
        policy_order_ids = {
            row["client_order_id"] for row in report["execution_decision_trace"]
            if row["action"] == "SUBMIT" and row["client_order_id"]
        }
        forced_fills = [fill for fill in report["fills"]
                        if fill["client_order_id"] not in policy_order_ids]
        episodes.append({
            "rank": rank, "window": report["episode_window"],
            "decision_steps": report["action_diagnostics"]["count"],
            "initial_equity": str(initial), "final_equity": str(final),
            "net_pnl": str(final - initial),
            "fee": report["metrics"]["total_fee"],
            "funding": report["metrics"]["total_funding"],
            "fill_count": len(report["fills"]), "funding_event_count": len(report["funding"]),
            "forced_terminal_close_fill_count": len(forced_fills),
            "forced_terminal_close_fee": str(sum(
                (Decimal(str(fill["commission"]).split()[0]) for fill in forced_fills),
                Decimal(0),
            )),
            "held_seconds": _held_seconds(snapshots),
            "position_count": len(_position_identities(snapshots)),
            "submit_decisions": report["execution_decision_diagnostics"]["submit_decision_count"],
            "filled_submit_decisions": report["execution_decision_diagnostics"][
                "submitted_decisions_with_fills"],
            "unfilled_reason_counts": report["execution_decision_diagnostics"][
                "unfilled_reason_counts"],
            "terminal_flat": report["terminal_flat"],
        })

    segmented_pnl = sum((Decimal(row["net_pnl"]) for row in episodes), Decimal(0))
    segmented_fee = sum((Decimal(str(row["fee"])) for row in episodes), Decimal(0))
    segmented_funding = sum((Decimal(str(row["funding"])) for row in episodes), Decimal(0))
    summary = {
        "schema": "rles.episode-horizon-comparison.v2",
        "split": "train_inner", "source_plan_sha256": _file_hash(plan_path),
        "bundle_sha256": plan["bundle_sha256"],
        "code_sha256": plan["code_sha256"],
        "source_hashes": {key: source[f"{key}_sha256"] for key in ("market", "rules")},
        "start_ns": start_ns, "end_ns": end_ns,
        "minimum_free_bytes_observed": minimum_free_bytes,
        "policy": {"kind": "fixed_target", "target_exposure": TARGET,
                   "soft_multipliers": bundle["soft_multipliers"]},
        "continuous": continuous,
        "segmented": {
            "episode_count": len(episodes),
            "episode_steps_min": min(row["decision_steps"] for row in episodes),
            "episode_steps_max": max(row["decision_steps"] for row in episodes),
            "episode_steps_by_rank": [row["decision_steps"] for row in episodes],
            "decision_steps": sum(row["decision_steps"] for row in episodes),
            "unique_decision_steps": sum(row["decision_steps"] for row in episodes),
            "decision_lane_coverage": {
                "full_steps": full_steps, "lane_count": len(lanes),
                "first_start_offset": lanes[0][0], "last_end_offset": lanes[-1][1],
                "contiguous": all(left[1] == right[0]
                                   for left, right in zip(lanes, lanes[1:])),
            },
            "independent_episode_pnl_sum": str(segmented_pnl),
            "fee": str(segmented_fee), "funding": str(segmented_funding),
            "fill_count": sum(row["fill_count"] for row in episodes),
            "funding_event_count": sum(row["funding_event_count"] for row in episodes),
            "forced_terminal_close_fill_count": sum(
                row["forced_terminal_close_fill_count"] for row in episodes),
            "forced_terminal_close_fee": str(sum(
                (Decimal(row["forced_terminal_close_fee"]) for row in episodes), Decimal(0))),
            "held_seconds_sum": sum(row["held_seconds"] for row in episodes),
            "terminal_flat_episodes": sum(row["terminal_flat"] is True for row in episodes),
            "episodes": episodes,
        },
        "differences": {
            "decision_steps_delta": sum(row["decision_steps"] for row in episodes)
            - continuous["decision_steps"],
            "net_pnl_sum_minus_continuous_pnl": str(
                segmented_pnl - (continuous_final - continuous_initial)),
            "fee_sum_minus_continuous_fee": str(
                segmented_fee - Decimal(str(continuous["fee"]))),
            "funding_sum_minus_continuous_funding": str(
                segmented_funding - Decimal(str(continuous["funding"]))),
            "fill_count_delta": sum(row["fill_count"] for row in episodes) - continuous["fill_count"],
            "funding_event_count_delta": sum(row["funding_event_count"] for row in episodes)
            - continuous["funding_event_count"],
        },
        "limitations": [
            "Economics use scenario-only fee/rule assumptions, not verified historical exchange rules.",
            "Segment PnL is the sum of independent episodes, each with the same starting balance; it is not compounded.",
            "This fixed-action mechanism comparison is not policy selection or an alpha test.",
            "T1 covers every continuous-account decision exactly once using variable lane lengths of at most one decision step difference.",
        ],
        "formal_evaluation_queries": 0,
    }
    _write_verified(output / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True,
                        help="existing frozen Train-inner native history growth plan")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.plan, args.output)
    print(json.dumps({"episodes": result["segmented"]["episode_count"],
                      "summary": str(args.output / "summary.json")}))


if __name__ == "__main__":
    main()
