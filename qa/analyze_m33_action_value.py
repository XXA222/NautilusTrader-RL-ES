from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/audit_evidence/20261017/m33_action_value"
PREREG = EVIDENCE / "preregistration_revision5.json"
OUTPUT = EVIDENCE / "analysis_final.json"
MINUTE_NS = 60_000_000_000
VOL_CUTS = (0.00025, 0.0005, 0.001)
VOL_LABELS = ("le_0.00025", "0.00025_to_0.0005", "0.0005_to_0.001", "gt_0.001")
POLICIES = ("B0_no_trade", "B1_always_long", "B4_always_short",
            "B2_simple_momentum", "B5_simple_reversal", "frozen_rl")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _regimes(features: dict[str, Any]) -> dict[str, str]:
    if "volatility_5m" not in features:
        raise ValueError("M3.3 decision trace is missing volatility_5m")
    volatility = float(features["volatility_5m"])
    index = int(np.searchsorted(VOL_CUTS, volatility, side="left"))
    result = {"volatility": VOL_LABELS[index]}
    trend_name = next((name for name in ("return_15m", "return_5m", "return_1m")
                       if name in features), None)
    if trend_name is None:
        raise ValueError("M3.3 decision trace is missing trend return features")
    trend = float(features[trend_name])
    funding = float(features["last_settled_funding"])
    result["trend"] = "negative" if trend < 0 else "positive" if trend > 0 else "zero"
    result["funding"] = "negative" if funding < 0 else "positive" if funding > 0 else "zero"
    return result


def _exposure_durations(rows: list[dict[str, Any]]) -> dict[str, Any]:
    segments: list[tuple[str, float]] = []
    start_ns: int | None = None
    direction: str | None = None
    last_ns: int | None = None
    for row in rows:
        exposure = float(row["net_exposure"])
        current = "flat" if abs(exposure) <= 1e-12 else "long" if exposure > 0 else "short"
        timestamp = int(row["ts_ns"])
        if current != direction:
            if direction in {"long", "short"} and start_ns is not None and last_ns is not None:
                segments.append((direction, (last_ns - start_ns + MINUTE_NS) / MINUTE_NS))
            start_ns = timestamp if current in {"long", "short"} else None
            direction = current
        last_ns = timestamp
    if direction in {"long", "short"} and start_ns is not None and last_ns is not None:
        segments.append((direction, (last_ns - start_ns + MINUTE_NS) / MINUTE_NS))
    durations = np.asarray([item[1] for item in segments], dtype=np.float64)
    return {
        "contiguous_same_direction_exposure_segments": len(segments),
        "mean_minutes": float(np.mean(durations)) if durations.size else 0.0,
        "median_minutes": float(np.median(durations)) if durations.size else 0.0,
        "p95_minutes": float(np.quantile(durations, 0.95)) if durations.size else 0.0,
        "long_segments": sum(direction == "long" for direction, _duration in segments),
        "short_segments": sum(direction == "short" for direction, _duration in segments),
        "interpretation": "contiguous decision-bar net exposure duration; position identity changes are not available in this trace",
    }


def _step_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    account_rows = report.get("account_rows")
    trace_rows = report.get("execution_decision_trace")
    if not isinstance(account_rows, list) or len(account_rows) < 2 or not isinstance(trace_rows, list):
        raise ValueError("M3.3 replay lacks account rows or execution trace")
    traces: dict[int, dict[str, Any]] = {}
    for trace in trace_rows:
        timestamp = int(trace["decision_ts_ns"])
        if timestamp in traces:
            raise ValueError("M3.3 replay has duplicate execution trace timestamps")
        if not isinstance(trace.get("market_features"), dict):
            raise ValueError("M3.3 replay trace lacks raw L2 market features")
        traces[timestamp] = trace
    ordered_trace_times = sorted(traces)
    if len(ordered_trace_times) != len(trace_rows):
        raise ValueError("M3.3 trace timestamp identity is not unique")
    fills_by_time: dict[int, list[dict[str, Any]]] = defaultdict(list)
    fill_ids: set[str] = set()
    for fill in report.get("fills", []):
        identity = str(fill.get("event_id", ""))
        if not identity or identity in fill_ids:
            raise ValueError("M3.3 native fill identity is empty or duplicated")
        fill_ids.add(identity)
        fills_by_time[int(fill["ts_event"])].append(fill)
    previous_equity = float(account_rows[0]["equity"])
    output: list[dict[str, Any]] = []
    for account in account_rows[1:]:
        account_timestamp = int(account["ts_ns"])
        timestamp = account_timestamp
        equity = float(account["equity"])
        delta = equity - previous_equity
        previous_equity = equity
        terminal_drain = timestamp not in traces
        if terminal_drain:
            # Terminal drain snapshots can land after the final policy decision.
            # Attribute only such trailing settlement facts to the last decision.
            if timestamp > ordered_trace_times[-1]:
                timestamp = ordered_trace_times[-1]
            else:
                raise ValueError(f"M3.3 account barrier has no matching decision trace: {timestamp}")
        trace = traces[timestamp]
        fee = float(account["fee"])
        funding = float(account["funding"])
        if float(account.get("external_cashflow", 0.0)) != 0.0:
            raise ValueError("M3.3 scenario excludes external account cashflows")
        # Fill prices already carry slippage. Gross here reverses only fee and
        # signed funding attribution from actual native-equity change.
        gross = delta + fee - funding
        fills = fills_by_time.get(account_timestamp, [])
        step = {
            "ts_ns": timestamp,
            "net_pnl_usdt": delta,
            "gross_marked_price_pnl_usdt": gross,
            "fee_usdt": fee,
            "funding_usdt": funding,
            "slippage_attribution_usdt": float(account["slippage_cost"]),
            "turnover_notional_usdt": float(account["turnover_notional"]),
            "net_exposure": float(trace["net_exposure"]),
            "requested_action": float(trace["actor_requested_target"]),
            "soft_target": trace.get("soft_target"),
            "hard_risk_target": trace.get("hard_risk_target"),
            "quantity_rounded_target": trace.get("quantity_rounded_target"),
            "executed_action": float(trace["executed_exposure"]),
            "plan_action": trace["action"],
            "plan_reason": trace["reason"],
            "native_fill_count": len(fills),
            "filled_quantity": sum(float(fill["last_qty"]) for fill in fills),
            "native_order_status": trace.get("native_order_status"),
            "terminal_drain_economics": terminal_drain,
            "trace_linked_fill_count": len(trace.get("fills", [])),
            "market_features": trace["market_features"],
            "regimes": _regimes(trace["market_features"]),
        }
        if terminal_drain and output and output[-1]["ts_ns"] == timestamp:
            previous = output[-1]
            for metric in ("net_pnl_usdt", "gross_marked_price_pnl_usdt", "fee_usdt",
                           "funding_usdt", "slippage_attribution_usdt", "turnover_notional_usdt",
                           "filled_quantity"):
                previous[metric] += step[metric]
            previous["native_fill_count"] += step["native_fill_count"]
            previous["terminal_drain_economics"] = True
        else:
            output.append(step)
    if not output:
        raise ValueError("M3.3 replay produced no attributed policy steps")
    if not np.isclose(sum(row["net_pnl_usdt"] for row in output),
                      float(account_rows[-1]["equity"]) - float(account_rows[0]["equity"]),
                      rtol=0.0, atol=1e-7):
        raise ValueError("M3.3 per-step equity changes do not reconcile to native terminal equity")
    if sum(row["native_fill_count"] for row in output) != len(fill_ids):
        raise ValueError("M3.3 per-step fill events do not reconcile to native fill facts")
    return output


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(rows)
    actions = Counter(row["plan_action"] for row in rows)
    reasons = Counter(str(row["plan_reason"]) for row in rows)
    target_summaries = {}
    for name in ("requested_action", "soft_target", "hard_risk_target",
                 "quantity_rounded_target", "executed_action", "net_exposure"):
        values = [float(row[name]) for row in rows if row[name] is not None]
        target_summaries[name] = {
            "observed_steps": len(values),
            "mean": float(np.mean(values)) if values else None,
            "mean_abs": float(np.mean(np.abs(values))) if values else None,
        }
    return {
        "decision_steps": count,
        "net_pnl_usdt": sum(row["net_pnl_usdt"] for row in rows),
        "gross_marked_price_pnl_usdt": sum(row["gross_marked_price_pnl_usdt"] for row in rows),
        "fee_usdt": sum(row["fee_usdt"] for row in rows),
        "funding_usdt": sum(row["funding_usdt"] for row in rows),
        "slippage_attribution_usdt": sum(row["slippage_attribution_usdt"] for row in rows),
        "turnover_notional_usdt": sum(row["turnover_notional_usdt"] for row in rows),
        "submit_decisions": actions.get("SUBMIT", 0),
        "native_fill_count": sum(row["native_fill_count"] for row in rows),
        "trace_linked_fill_count": sum(row["trace_linked_fill_count"] for row in rows),
        "terminal_drain_economic_decisions": sum(row["terminal_drain_economics"] for row in rows),
        "filled_quantity": sum(row["filled_quantity"] for row in rows),
        "plan_action_counts": dict(sorted(actions.items())),
        "plan_reason_counts": dict(sorted(reasons.items())),
        "mean_requested_action": float(np.mean([row["requested_action"] for row in rows])),
        "mean_executed_action": float(np.mean([row["executed_action"] for row in rows])),
        "mean_abs_net_exposure": float(np.mean([abs(row["net_exposure"]) for row in rows])),
        "risk_and_execution_target_summaries": target_summaries,
        "nonzero_exposure_fraction": sum(abs(row["net_exposure"]) > 1e-12 for row in rows) / count,
        "exposure_durations": _exposure_durations(rows),
    }


def _bootstrap_interval(rows_left: list[dict[str, Any]], rows_right: list[dict[str, Any]], *,
                        start_ns: int, block_minutes: int, replicates: int,
                        seed: int) -> dict[str, Any]:
    left = {row["ts_ns"]: row["net_pnl_usdt"] for row in rows_left}
    right = {row["ts_ns"]: row["net_pnl_usdt"] for row in rows_right}
    if left.keys() != right.keys():
        raise ValueError("M3.3 paired policies do not share identical decision timestamps")
    block_ns = block_minutes * MINUTE_NS
    blocks: dict[int, float] = defaultdict(float)
    for timestamp in left:
        block = max(0, (timestamp - start_ns) // block_ns)
        blocks[int(block)] += left[timestamp] - right[timestamp]
    values = np.asarray([blocks[index] for index in range(max(blocks) + 1)], dtype=np.float64)
    if values.size < 6:
        return {"status": "insufficient_blocks", "block_count": int(values.size),
                "observed_delta_usdt": float(np.sum(values)), "ci_95_usdt": None}
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, values.size, size=(replicates, values.size))
    draws = np.sum(values[sampled], axis=1)
    lower, upper = np.quantile(draws, (0.025, 0.975), method="linear")
    return {
        "status": "descriptive_block_bootstrap",
        "block_minutes": block_minutes,
        "block_count": int(values.size),
        "replicates": replicates,
        "seed": seed,
        "observed_delta_usdt": float(np.sum(values)),
        "ci_95_usdt": [float(lower), float(upper)],
        "bootstrap_positive_fraction": float(np.mean(draws > 0)),
    }


def analyze() -> dict[str, Any]:
    prereg = load(PREREG)
    if (prereg.get("schema") != "rles.m33-action-value-preregistration.v5"
            or prereg.get("status") != "analysis_binding_revision_after_replay"
            or prereg.get("evaluation_queries") != {
                "validation": 0, "test": 0, "holdout": 0, "outer_2026": 0,
            }):
        raise ValueError("M3.3 preregistration or evaluation guard is invalid")
    supersedes = prereg["supersedes"]
    for entry in supersedes.values():
        if sha(Path(entry["path"])) != entry["sha256"]:
            raise ValueError("M3.3 superseded preregistration changed")
    if sha(Path(prereg["frozen_plan"]["path"])) != prereg["frozen_plan"]["sha256"]:
        raise ValueError("M3.3 frozen replay plan changed")
    for relative, expected in prereg["source_code_sha256"].items():
        if sha(ROOT / relative) != expected:
            raise ValueError(f"M3.3 source changed after preregistration: {relative}")
    if sha(ROOT / "qa/analyze_m33_action_value.py") != prereg["analysis_code_sha256"]:
        raise ValueError("M3.3 analysis implementation changed after its binding revision")
    for identity_name in ("prior_baseline_plan", "prior_baseline_summary", "bundle"):
        entry = prereg[identity_name]
        if sha(Path(entry["path"])) != entry["sha256"]:
            raise ValueError(f"M3.3 prerequisite changed: {identity_name}")
    for entry in prereg["market_and_rules"].values():
        if sha(Path(entry["path"])) != entry["sha256"]:
            raise ValueError("M3.3 market/rules source changed")

    output = Path(prereg["replay_output"])
    summary = load(output / "summary.json")
    if (summary.get("formal_evaluation_queries") != 0
            or len(summary.get("runs", [])) != prereg["prior_baseline_reports"]
            or summary.get("scope") != "exploratory frozen-policy diagnostics; internal data are development data"):
        raise ValueError("M3.3 replay summary is incomplete or outside Train-only scope")
    plan = load(Path(prereg["frozen_plan"]["path"]))
    start_by_window = {index: int(window["start_ns"])
                       for index, window in enumerate(plan["windows"])}
    grouped: dict[tuple[int, str, str], list[dict[str, Any]]] = {}
    run_records = []
    for run in summary["runs"]:
        report_path = output / run["report"]
        if sha(report_path) != run["report_sha256"]:
            raise ValueError(f"M3.3 report SHA mismatch: {run['report']}")
        report = load(report_path)
        rows = _step_rows(report)
        if len(rows) != int(report["action_diagnostics"]["count"]):
            raise ValueError(f"M3.3 per-step/account row count mismatch: {run['report']}")
        run_key = (int(run["window"]), str(run["mode"]), str(run["policy"]))
        if run_key in grouped:
            raise ValueError(f"duplicate M3.3 policy run: {run_key}")
        grouped[run_key] = rows
        run_records.append({
            "window": run_key[0], "mode": run_key[1], "policy": run_key[2],
            "report_sha256": run["report_sha256"],
            "metrics": run["metrics"],
            "per_decision": _summarize(rows),
            "timestamp_start_ns": min(row["ts_ns"] for row in rows),
            "timestamp_end_ns": max(row["ts_ns"] for row in rows),
        })
    expected_runs = len(plan["windows"]) * len(plan["modes"]) * len(POLICIES)
    if len(grouped) != expected_runs:
        raise ValueError("M3.3 policy matrix is incomplete")

    regimes: dict[str, Any] = {}
    paired_uncertainty: list[dict[str, Any]] = []
    comparisons = ("B0_no_trade", "B1_always_long", "B4_always_short",
                   "B2_simple_momentum", "B5_simple_reversal")
    bootstrap = prereg["paired_uncertainty"]
    for window in range(len(plan["windows"])):
        for mode in plan["modes"]:
            actor_rows = grouped[(window, mode, "frozen_rl")]
            for baseline in comparisons:
                paired_uncertainty.append({
                    "window": window, "mode": mode, "actor": "frozen_rl", "baseline": baseline,
                    **_bootstrap_interval(
                        actor_rows, grouped[(window, mode, baseline)],
                        start_ns=start_by_window[window],
                        block_minutes=int(bootstrap["block_minutes"]),
                        replicates=int(bootstrap["replicates"]),
                        seed=int(bootstrap["seed"]) + window * 100 + len(paired_uncertainty),
                    ),
                })
            for policy in POLICIES:
                rows = grouped[(window, mode, policy)]
                accum: dict[str, list[dict[str, Any]]] = defaultdict(list)
                for row in rows:
                    for family, label in row["regimes"].items():
                        accum[f"{family}:{label}"].append(row)
                for name, selected in sorted(accum.items()):
                    regimes[f"window-{window}:{mode}:{policy}:{name}"] = _summarize(selected)

    return {
        "schema": "rles.m33-action-value-analysis.v1",
        "status": "passed_train_inner_action_value_diagnostics",
        "analyzer_sha256": sha(ROOT / "qa/analyze_m33_action_value.py"),
        "preregistration_sha256": sha(PREREG),
        "replay_summary_sha256": sha(output / "summary.json"),
        "evaluation_queries": prereg["evaluation_queries"],
        "regime_policy": prereg["regime_policy"],
        "totals": {
            "policy_runs": len(run_records),
            "decision_steps": sum(row["per_decision"]["decision_steps"] for row in run_records),
            "unique_policy_window_mode_cells": len(grouped),
            "all_native_equity_changes_reconciled": True,
            "slippage_deducted_from_native_equity": False,
        },
        "paired_uncertainty": paired_uncertainty,
        "runs": run_records,
        "regime_contributions": regimes,
        "limitations": prereg["limitations"],
    }


if __name__ == "__main__":
    payload = json.dumps(analyze(), ensure_ascii=False, indent=2) + "\n"
    with OUTPUT.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(payload)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    OUTPUT.with_suffix(OUTPUT.suffix + ".sha256").write_text(digest + "\n", encoding="ascii")
    print(json.dumps({"output": str(OUTPUT), "sha256": digest}, ensure_ascii=False))
