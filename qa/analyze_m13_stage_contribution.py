"""Independently verify and summarize frozen M1.3 native replay reports."""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRATION = ROOT / "docs/audit_evidence/20261017/m13_stage_contribution/preregistration_revision2.json"
DEFAULT_OUTPUT_ROOT = ROOT / ".artifacts/audit-development-20261017/m13-stage-contribution-v1"
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
METRICS = (
    "net_return", "max_drawdown", "sharpe", "sortino", "turnover_ratio",
    "total_fee", "total_funding", "total_slippage_cost",
    "gross_return_before_reported_costs",
)


def _sha(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _verify_preregistration(registration: Path, output_root: Path) -> dict[str, Any]:
    marker = registration.with_suffix(registration.suffix + ".sha256")
    if not marker.is_file() or marker.read_text(encoding="ascii").strip() != _sha(registration):
        raise ValueError("M1.3 preregistration SHA receipt mismatch")
    frozen = _load_json(registration)
    if frozen.get("status") != "frozen_before_native_replay":
        raise ValueError("M1.3 preregistration was not frozen before replay")
    if frozen.get("formal_evaluation_queries") != {"validation": 0, "test": 0, "holdout": 0}:
        raise ValueError("M1.3 preregistration includes a formal evaluation query")
    if Path(str(frozen.get("output_root"))).resolve() != output_root.resolve():
        raise ValueError("M1.3 report directory differs from preregistration")
    runner = ROOT / "qa/m13_stage_contribution_20261017.py"
    if _sha(runner) != frozen.get("runner_sha256"):
        raise ValueError("M1.3 replay runner differs from the frozen receipt")
    source_hashes = frozen.get("source_sha256")
    if not isinstance(source_hashes, dict):
        raise ValueError("M1.3 source identity list is missing")
    for relative in SOURCE_FILES:
        if _sha(ROOT / relative) != source_hashes.get(relative):
            raise ValueError(f"frozen source identity mismatch: {relative}")
    return frozen


def _verify_equity_reconciliation(report: dict[str, Any], label: str) -> None:
    if report.get("costs_already_in_native_equity") is not True:
        raise ValueError(f"{label}: native equity cost inclusion is not confirmed")
    reconciliation = report.get("equity_reconciliation")
    if not isinstance(reconciliation, dict) or not isinstance(reconciliation.get("rows"), list):
        raise ValueError(f"{label}: event-level equity reconciliation is missing")
    if report.get("terminal_flat") is not True:
        raise ValueError(f"{label}: terminal account is not confirmed flat")
    for row in reconciliation["rows"]:
        residual = abs(float(row["residual"]))
        tolerance = float(row["tolerance"])
        if residual > tolerance:
            raise ValueError(f"{label}: equity residual exceeds tolerance at {row['ts_ns']}")


def summarize_mode(mode: str, payload: dict[str, Any]) -> dict[str, Any]:
    r1_value = payload.get("r1_neutral")
    e2_value = payload.get("e2_selected")
    if not isinstance(r1_value, dict) or not isinstance(e2_value, dict):
        raise ValueError(f"{mode}: one or more native arm reports are missing")
    r1, e2 = r1_value, e2_value
    reports = {"r1_neutral": r1, "e2_selected": e2}
    _verify_equity_reconciliation(r1, f"{mode}/R1")
    _verify_equity_reconciliation(e2, f"{mode}/E2")

    r1_rows, e2_rows = r1.get("account_rows"), e2.get("account_rows")
    if not isinstance(r1_rows, list) or not isinstance(e2_rows, list) or not r1_rows:
        raise ValueError(f"{mode}: paired native account rows are missing")
    if len(r1_rows) != len(e2_rows):
        raise ValueError(f"{mode}: arm account row counts differ")
    timestamps = [row["ts_ns"] for row in r1_rows]
    if timestamps != [row["ts_ns"] for row in e2_rows] or any(
        left >= right for left, right in zip(timestamps, timestamps[1:])
    ):
        raise ValueError(f"{mode}: native account timestamps are not strictly paired")

    stored = payload.get("summary")
    if not isinstance(stored, dict) or stored.get("same_native_timestamps") is not True:
        raise ValueError(f"{mode}: runner summary does not attest paired timestamps")
    first_r1, first_e2 = r1_rows[0], e2_rows[0]
    last_r1, last_e2 = r1_rows[-1], e2_rows[-1]
    initial_r1, initial_e2 = float(first_r1["equity"]), float(first_e2["equity"])
    final_r1, final_e2 = float(last_r1["equity"]), float(last_e2["equity"])
    if initial_r1 != initial_e2 or final_r1 != float(stored["final_equity_r1"]):
        raise ValueError(f"{mode}: initial or R1 final equity disagrees with raw account rows")
    if final_e2 != float(stored["final_equity_e2"]):
        raise ValueError(f"{mode}: E2 final equity disagrees with raw account rows")
    if abs((final_e2 - final_r1) - float(stored["native_equity_delta_e2_minus_r1"])) > 1e-8:
        raise ValueError(f"{mode}: final paired equity delta is inconsistent")

    r1_metrics, e2_metrics = r1.get("metrics"), e2.get("metrics")
    if not isinstance(r1_metrics, dict) or not isinstance(e2_metrics, dict):
        raise ValueError(f"{mode}: native economic metrics are missing")
    deltas = {key: float(e2_metrics[key]) - float(r1_metrics[key]) for key in METRICS}
    r1_exec = r1.get("execution_decision_diagnostics", {})
    e2_exec = e2.get("execution_decision_diagnostics", {})
    if not isinstance(r1_exec, dict) or not isinstance(e2_exec, dict):
        raise ValueError(f"{mode}: execution diagnostics are malformed")
    r1_decisions = int(r1_exec.get("decision_count", 0))
    e2_decisions = int(e2_exec.get("decision_count", 0))
    if r1_decisions != e2_decisions:
        raise ValueError(f"{mode}: policy decision counts differ")
    return {
        "mode": mode,
        "paired_account_rows": len(timestamps),
        "policy_decisions_per_arm": r1_decisions,
        "same_native_timestamps": True,
        "initial_equity": initial_r1,
        "final_equity_r1": final_r1,
        "final_equity_e2": final_e2,
        "native_equity_delta_e2_minus_r1": final_e2 - final_r1,
        "metric_deltas_e2_minus_r1": deltas,
        "execution_deltas_e2_minus_r1": {
            "fills": len(e2.get("fills", [])) - len(r1.get("fills", [])),
            "submit_decisions": int(e2_exec.get("submit_decision_count", 0))
            - int(r1_exec.get("submit_decision_count", 0)),
            "native_orders": int(e2_exec.get("native_order_count", 0))
            - int(r1_exec.get("native_order_count", 0)),
            "hard_risk_interventions": int(e2.get("risk_hard_interventions", 0))
            - int(r1.get("risk_hard_interventions", 0)),
        },
        "r1_metrics": {key: r1_metrics[key] for key in METRICS},
        "e2_metrics": {key: e2_metrics[key] for key in METRICS},
        "equity_reconciliation": {
            arm: {
                "rows": len(report["equity_reconciliation"]["rows"]),
                "max_abs_residual": report["equity_reconciliation"]["max_abs_residual"],
                "maximum_tolerance": max(
                    float(row["tolerance"]) for row in report["equity_reconciliation"]["rows"]
                ),
                "terminal_flat": report.get("terminal_flat"),
            }
            for arm, report in reports.items()
        },
        "report_sha256": None,
    }


def analyze(registration: Path, output_root: Path, result_path: Path) -> dict[str, Any]:
    frozen = _verify_preregistration(registration, output_root)
    modes: dict[str, dict[str, Any]] = {}
    report_hashes: dict[str, str] = {}
    for mode in frozen["modes"]:
        report_path = output_root / f"{mode}.json"
        payload = _load_json(report_path)
        report_hashes[mode] = _sha(report_path)
        modes[mode] = summarize_mode(mode, payload)
    return {
        "schema": "rles.m13-stage-contribution-independent-analysis.v1",
        "status": "verified_train_inner_diagnostic",
        "preregistration_path": registration.relative_to(ROOT).as_posix(),
        "preregistration_sha256": _sha(registration),
        "report_sha256": report_hashes,
        "bundle_sha256": frozen["bundle_sha256"],
        "actor_sha256": frozen["actor_sha256"],
        "stage_pair": frozen["stage_pair"],
        "selected_e2_multipliers": frozen["selected_e2_multipliers"],
        "r1_neutral_multipliers": frozen["r1_neutral_multipliers"],
        "interpretation_limit": frozen["interpretation_limit"],
        "formal_evaluation_queries": frozen["formal_evaluation_queries"],
        "modes": modes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", type=Path, default=DEFAULT_REGISTRATION)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--result", type=Path,
        default=ROOT / "docs/audit_evidence/20261017/m13_stage_contribution/analysis_verified.json",
    )
    args = parser.parse_args()
    result_path = args.result if args.result.is_absolute() else ROOT / args.result
    result = analyze(args.registration, args.output_root, result_path)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
                           encoding="utf-8")
    result_path.with_suffix(result_path.suffix + ".sha256").write_text(_sha(result_path), encoding="ascii")
    print(json.dumps({"status": result["status"], "result": str(result_path),
                      "modes": {key: value["metric_deltas_e2_minus_r1"]
                                for key, value in result["modes"].items()}},
                     indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
