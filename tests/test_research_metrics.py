from __future__ import annotations

from decimal import Decimal
from types import SimpleNamespace

import pytest

from rles_research.metrics import MetricsError, from_account_facts
from rles_research.native_metrics import native_report

pytestmark = pytest.mark.unit


def test_account_metrics_use_period_flows_and_native_signed_funding_once() -> None:
    facts = (
        {"ts_ns": 1, "equity": 1_000.0, "external_cashflow": 0.0,
         "turnover_notional": 0.0, "fee": 0.0, "funding": 0.0, "slippage_cost": 0.0},
        {"ts_ns": 11, "equity": 1_100.0, "external_cashflow": 100.0,
         "turnover_notional": 200.0, "fee": 5.0, "funding": -2.0, "slippage_cost": 3.0},
        {"ts_ns": 21, "equity": 990.0, "external_cashflow": 0.0,
         "turnover_notional": 110.0, "fee": 1.0, "funding": 1.0, "slippage_cost": 1.0},
    )

    result = from_account_facts(facts, periods_per_year=2, sample_interval_ns=10)

    # The first sample's boundary deposit does not appear as trading return.
    assert result["net_return"] == pytest.approx(-0.1)
    assert result["turnover_ratio"] == pytest.approx(0.3)
    assert result["total_fee"] == pytest.approx(6.0)
    assert result["total_funding"] == pytest.approx(-1.0)
    assert result["total_slippage_cost"] == pytest.approx(4.0)
    # Net equity already includes these flows. The gross metric strips each
    # period cost once and removes the signed funding cashflow once.
    assert result["gross_return_before_reported_costs"] == pytest.approx(-0.099)


def test_account_metrics_only_compute_profit_factor_from_closed_trade_pnls() -> None:
    facts = tuple(
        {"ts_ns": index * 10, "equity": equity, "external_cashflow": 0.0,
         "turnover_notional": 0.0, "fee": 0.0, "funding": 0.0, "slippage_cost": 0.0}
        for index, equity in enumerate((1_000.0, 1_010.0, 1_005.0), start=1)
    )

    without_closed_trades = from_account_facts(
        facts, periods_per_year=1, sample_interval_ns=10
    )
    with_closed_trades = from_account_facts(
        facts, periods_per_year=1, sample_interval_ns=10,
        realized_trade_pnls=(4.0, -2.0, -2.0),
    )

    assert "trade_profit_factor" not in without_closed_trades
    assert with_closed_trades["trade_profit_factor"] == pytest.approx(1.0)
    assert with_closed_trades["closed_trade_count"] == 3


def test_account_metrics_reject_irregular_sampling() -> None:
    facts = tuple(
        {"ts_ns": timestamp, "equity": 1_000.0, "external_cashflow": 0.0,
         "turnover_notional": 0.0, "fee": 0.0, "funding": 0.0, "slippage_cost": 0.0}
        for timestamp in (1, 11, 22)
    )
    with pytest.raises(MetricsError, match="irregular account sampling"):
        from_account_facts(facts, periods_per_year=1, sample_interval_ns=10)


def test_quote_execution_can_report_realized_price_improvement_as_signed_slippage() -> None:
    facts = (
        {"ts_ns": 1, "equity": 1_000.0, "external_cashflow": 0.0,
         "turnover_notional": 0.0, "fee": 0.0, "funding": 0.0, "slippage_cost": 0.0},
        {"ts_ns": 11, "equity": 1_001.0, "external_cashflow": 0.0,
         "turnover_notional": 100.0, "fee": 0.1, "funding": 0.0, "slippage_cost": -0.2},
        {"ts_ns": 21, "equity": 1_001.0, "external_cashflow": 0.0,
         "turnover_notional": 0.0, "fee": 0.0, "funding": 0.0, "slippage_cost": 0.0},
    )
    with pytest.raises(MetricsError, match="invalid account values"):
        from_account_facts(facts, periods_per_year=1, sample_interval_ns=10)

    result = from_account_facts(facts, periods_per_year=1, sample_interval_ns=10,
                                allow_price_improvement=True)
    assert result["total_slippage_cost"] == pytest.approx(-0.2)


def test_native_report_reassembles_incremental_fill_and_funding_facts() -> None:
    fill = {"event_id": "fill-1", "client_order_id": "order-1", "order_side": "BUY",
            "last_qty": "1", "last_px": "100", "commission": "0.1 USDT"}
    funding = {"event_id": "funding-1", "pnl_change": "-0.2 USDT"}

    def snapshot(timestamp, equity, fills=(), adjustments=()):
        return SimpleNamespace(
            decision_ts_ns=timestamp, equity=equity, currency="USDT", fills=fills,
            funding=adjustments, positions=(), orders=(), liquidation_capability=True,
            liquidation_orders=(),
        )

    report = native_report(
        (snapshot(1, 1_000.0), snapshot(11, 999.9, (fill,)),
         snapshot(21, 999.7, (), (funding,))),
        fill_references={"order-1": Decimal("100")}, sample_interval_ns=10,
    )

    assert report["fills"] == [fill]
    assert report["funding"] == [funding]
    assert report["metrics"]["total_fee"] == pytest.approx(0.1)
    assert report["metrics"]["total_funding"] == pytest.approx(-0.2)
