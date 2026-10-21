"""Compare Bar Base and Stress execution costs on a frozen Train-inner path."""
from __future__ import annotations

import argparse
from decimal import Decimal
import json
from pathlib import Path

import numpy as np

from rles_data.core import CuratedStore, DataError
from rles_data.io import read_curated
from rles_gym_bridge.historical import HistoricalEnvironmentFactory, load_rules
from rles_orchestrator.application import _file_hash, _verify_file, _write_verified, rollout
from rles_orchestrator.research_access import record_development
from rles_orchestrator.wfo_application import _scaler
from profile_native_history_growth_20261016 import available_memory, coverage_for_window

DAY_NS = 86_400 * 1_000_000_000
TARGET = 0.25
MODES = ("base", "stress")
SOURCE_FILES = (
    "qa/compare_bar_cost_modes_20261017.py",
    "qa/profile_native_history_growth_20261016.py",
    "packages/gym_bridge/src/rles_gym_bridge/historical.py",
    "packages/gym_bridge/src/rles_gym_bridge/env.py",
    "packages/orchestrator/src/rles_orchestrator/application.py",
    "packages/orchestrator/src/rles_orchestrator/research_access.py",
    "packages/research/src/rles_research/access.py",
    "packages/research/src/rles_research/native_metrics.py",
    "packages/execution/src/rles_execution/planner.py",
    "packages/risk/src/rles_risk/guards.py",
)


def _predict(_batch: np.ndarray) -> np.ndarray:
    return np.full((1, 1), TARGET, dtype=np.float32)


def _order_metrics(orders, fills) -> dict:
    fills_by_order: dict[str, list[dict]] = {}
    for fill in fills:
        client_order_id = str(fill.get("client_order_id", ""))
        if not client_order_id:
            raise DataError("native fill lacks client_order_id")
        fills_by_order.setdefault(client_order_id, []).append(fill)

    ordered_qty = Decimal(0)
    filled_qty = Decimal(0)
    filled_orders = partial_orders = unfilled_orders = 0
    order_ids: set[str] = set()
    for order in orders:
        client_order_id = str(order.get("client_order_id", ""))
        if not client_order_id:
            raise DataError("native order lacks client_order_id")
        if client_order_id in order_ids:
            raise DataError("duplicate client_order_id in native orders")
        order_ids.add(client_order_id)
        order_qty = Decimal(str(order["quantity"]))
        if order_qty <= 0:
            raise DataError("native order quantity must be positive")
        order_fills = fills_by_order.get(client_order_id, [])
        order_filled_qty = sum((Decimal(str(fill["last_qty"])) for fill in order_fills), Decimal(0))
        ordered_qty += order_qty
        filled_qty += order_filled_qty
        if order_filled_qty >= order_qty:
            filled_orders += 1
        elif order_filled_qty > 0:
            partial_orders += 1
        else:
            unfilled_orders += 1

    if set(fills_by_order) - order_ids:
        raise DataError("native fill has no matching submitted order")
    if filled_qty > ordered_qty:
        raise DataError("native fills exceed submitted order quantity")
    fill_count = len(fills)
    filled_notional = sum(
        (Decimal(str(fill["last_qty"])) * Decimal(str(fill["last_px"])) for fill in fills),
        Decimal(0),
    )
    return {
        "native_order_count": len(orders),
        "filled_orders": filled_orders,
        "partially_filled_orders": partial_orders,
        "unfilled_orders": unfilled_orders,
        "filled_order_ratio": filled_orders / len(orders) if orders else None,
        "ordered_qty": str(ordered_qty),
        "filled_qty": str(filled_qty),
        "quantity_fill_ratio": str(filled_qty / ordered_qty) if ordered_qty else None,
        "fill_count": fill_count,
        "filled_quote_notional": str(filled_notional),
    }


def run(plan_path: Path, output: Path) -> dict:
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if plan.get("split") != "train_inner" or output.exists():
        raise DataError("requires frozen Train-inner inputs and a new output directory")
    source_plan_path = Path(plan["source_plan"])
    if _file_hash(source_plan_path) != plan["source_plan_sha256"]:
        raise DataError("frozen Train-inner source plan checksum changed")
    source_plan = json.loads(source_plan_path.read_text(encoding="utf-8"))
    bundle_root = Path(source_plan["bundle"])
    if _verify_file(bundle_root / "bundle.json") != plan["bundle_sha256"]:
        raise DataError("frozen policy bundle checksum changed")
    bundle = json.loads((bundle_root / "bundle.json").read_text(encoding="utf-8"))
    source = plan["source"]
    for key in ("market", "rules"):
        if _file_hash(Path(source[f"{key}_path"])) != source[f"{key}_sha256"]:
            raise DataError(f"frozen {key} source checksum changed")

    start_ns = int(source_plan["windows"][0]["start_ns"])
    end_ns = start_ns + 90 * DAY_NS
    if end_ns > int(bundle["training_end_ns"]):
        raise DataError("Base/Stress comparison exceeds frozen Train boundary")
    window = {"start_ns": start_ns, "end_ns": end_ns}
    seed = int(plan.get("seed", source_plan["seed"]))
    environment = bundle["environment"]
    config = {"environment": environment, "output": str(output), "seed": seed}
    record_development(config, window, label="bar-cost-base-stress-train-inner")

    output.mkdir(parents=True)
    source_code_hashes = {
        path: _file_hash(Path(path)) for path in SOURCE_FILES
    }
    _write_verified(output / "plan.json", {
        "schema": "rles.bar-cost-comparison-plan.v1",
        "source_plan": str(plan_path.resolve()),
        "source_plan_sha256": _file_hash(plan_path),
        "source_train_plan_sha256": _file_hash(source_plan_path),
        "bundle_sha256": plan["bundle_sha256"],
        "source_hashes": {key: source[f"{key}_sha256"] for key in ("market", "rules")},
        "source_code_sha256": source_code_hashes,
        "split": "train_inner",
        "window": window,
        "seed": seed,
        "policy": {"kind": "fixed_target", "target_exposure": TARGET,
                   "soft_multipliers": bundle["soft_multipliers"]},
        "modes": {mode: environment["execution"][f"{mode}_fill"] for mode in MODES},
        "metrics": [
            "net_pnl", "net_return", "max_drawdown", "turnover_ratio", "filled_qty",
            "filled_quote_notional", "filled_order_ratio", "quantity_fill_ratio",
            "total_slippage_cost", "total_fee", "total_funding",
        ],
        "purpose": "Base/Stress Bar execution-cost sensitivity; no selection or profitability gate",
        "liquidity_assumption": "Bar model provides order-sized simulated depth; not historical depth evidence",
    })

    rules = load_rules(source["rules_path"])
    store = CuratedStore((*read_curated(source["market_path"]).events, *rules))
    if store.snapshot_hash != bundle["preprocessing"]["identities"]["dataset"]:
        raise DataError("frozen market/rule sources differ from the policy dataset identity")
    scaler = _scaler(bundle_root)
    coverage = coverage_for_window(environment["coverage_policy"], window)
    common_factory_args = {
        **environment,
        "coverage_policy": coverage,
        "start_ns": start_ns,
        "end_ns": end_ns,
        "split": "train_inner",
        "scaler": scaler,
        "fit_scaler": False,
        "reward_scale": bundle.get("reward_scale", 1.0),
        "compact_report_snapshots": True,
        "prepared_store": store,
    }

    results = {}
    minimum_free_bytes = available_memory()
    for mode in MODES:
        free_bytes = available_memory()
        minimum_free_bytes = min(minimum_free_bytes, free_bytes)
        if free_bytes < int(source_plan["stop_free_bytes"]):
            raise MemoryError("resource stop floor reached before the next cost mode")
        factory = HistoricalEnvironmentFactory(**common_factory_args, mode=mode)
        report, _ = rollout(
            factory, _predict, seed=seed,
            soft_parameters=bundle["soft_multipliers"],
            collect_observations=False, include_account_rows=False,
        )
        snapshots = tuple(factory.snapshots)
        if not snapshots or not report["terminal_flat"]:
            raise DataError(f"{mode} run did not produce a flat terminal account")
        initial = Decimal(str(snapshots[0].equity))
        final = Decimal(str(snapshots[-1].equity))
        order_metrics = _order_metrics(snapshots[-1].all_orders, report["fills"])
        metrics = report["metrics"]
        results[mode] = {
            "decision_steps": report["action_diagnostics"]["count"],
            "initial_equity": str(initial),
            "final_equity": str(final),
            "net_pnl": str(final - initial),
            "net_return": metrics["net_return"],
            "max_drawdown": metrics["max_drawdown"],
            "turnover_ratio": metrics["turnover_ratio"],
            "gross_return_before_reported_costs": metrics["gross_return_before_reported_costs"],
            "total_slippage_cost": metrics["total_slippage_cost"],
            "total_fee": metrics["total_fee"],
            "total_funding": metrics["total_funding"],
            "funding_event_count": len(report["funding"]),
            "terminal_flat": report["terminal_flat"],
            "order_metrics": order_metrics,
        }

    numeric_fields = (
        "net_pnl", "net_return", "max_drawdown", "turnover_ratio",
        "total_slippage_cost", "total_fee", "total_funding",
    )
    summary = {
        "schema": "rles.bar-cost-comparison.v1",
        "split": "train_inner",
        "source_plan_sha256": _file_hash(plan_path),
        "bundle_sha256": plan["bundle_sha256"],
        "source_hashes": {key: source[f"{key}_sha256"] for key in ("market", "rules")},
        "window": window,
        "seed": seed,
        "policy": {"kind": "fixed_target", "target_exposure": TARGET,
                   "soft_multipliers": bundle["soft_multipliers"]},
        "minimum_free_bytes_observed": minimum_free_bytes,
        "stop_free_bytes": int(source_plan["stop_free_bytes"]),
        "modes": results,
        "stress_minus_base": {
            key: (float(results["stress"][key]) - float(results["base"][key]))
            for key in numeric_fields
        },
        "limitations": [
            "Costs and market rules are scenario assumptions, not verified historical exchange rules.",
            "Bar fill simulation provides depth sized to the order and cannot certify executable market depth.",
            "Filled notional and quantity fill ratios describe simulated native fills, not live liquidity.",
            "A single fixed-action Train-inner sensitivity run is not model selection or alpha evidence.",
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
    print(json.dumps({"modes": list(result["modes"]),
                      "summary": str(args.output / "summary.json")}))


if __name__ == "__main__":
    main()
