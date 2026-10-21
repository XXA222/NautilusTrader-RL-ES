from __future__ import annotations

import pytest

from qa.analyze_m13_stage_contribution import summarize_mode


def _arm(*, equity: tuple[float, float], fee: float, fills: int, submits: int) -> dict:
    metrics = {
        "net_return": equity[-1] / 10_000 - 1,
        "max_drawdown": 0.01,
        "sharpe": -0.5,
        "sortino": -0.8,
        "turnover_ratio": 0.2,
        "total_fee": fee,
        "total_funding": 0.1,
        "total_slippage_cost": 0.05,
        "gross_return_before_reported_costs": 0.002,
    }
    return {
        "account_rows": [
            {"ts_ns": 10, "equity": equity[0]},
            {"ts_ns": 20, "equity": equity[1]},
        ],
        "costs_already_in_native_equity": True,
        "equity_reconciliation": {
            "rows": [{"ts_ns": 20, "residual": "0", "tolerance": "0.0000001"}],
            "max_abs_residual": "0",
        },
        "terminal_flat": True,
        "metrics": metrics,
        "fills": [{} for _ in range(fills)],
        "execution_decision_diagnostics": {
            "decision_count": 1,
            "submit_decision_count": submits,
            "native_order_count": submits,
        },
        "risk_hard_interventions": 1,
    }


def _mode_payload(*, e2_timestamps: tuple[int, int] = (10, 20)) -> dict:
    r1 = _arm(equity=(10_000, 9_990), fee=2, fills=4, submits=3)
    e2 = _arm(equity=(10_000, 9_995), fee=1, fills=2, submits=2)
    e2["account_rows"][0]["ts_ns"], e2["account_rows"][1]["ts_ns"] = e2_timestamps
    return {
        "r1_neutral": r1,
        "e2_selected": e2,
        "summary": {
            "same_native_timestamps": True,
            "final_equity_r1": 9_990,
            "final_equity_e2": 9_995,
            "native_equity_delta_e2_minus_r1": 5,
        },
    }


def test_summarize_mode_recomputes_paired_economic_and_execution_deltas() -> None:
    result = summarize_mode("base", _mode_payload())

    assert result["paired_account_rows"] == 2
    assert result["policy_decisions_per_arm"] == 1
    assert result["native_equity_delta_e2_minus_r1"] == 5
    assert result["metric_deltas_e2_minus_r1"]["total_fee"] == -1
    assert result["metric_deltas_e2_minus_r1"]["net_return"] == pytest.approx(0.0005)
    assert result["execution_deltas_e2_minus_r1"] == {
        "fills": -2,
        "submit_decisions": -1,
        "native_orders": -1,
        "hard_risk_interventions": 0,
    }


def test_summarize_mode_rejects_unpaired_timestamps() -> None:
    with pytest.raises(ValueError, match="timestamps are not strictly paired"):
        summarize_mode("base", _mode_payload(e2_timestamps=(10, 21)))
