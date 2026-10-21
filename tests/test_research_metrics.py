from __future__ import annotations

from decimal import Decimal
from types import SimpleNamespace

import pytest

from rles_research.metrics import MetricsError, from_account_facts
from rles_research.native_metrics import account_rows, native_report, reconcile_linear_equity
from rles_research.statistics import (
    deflated_sharpe, deflated_sharpe_trial_count_sensitivity,
)

pytestmark = pytest.mark.unit


def test_dsr_trial_count_sensitivity_brackets_dependence_without_claiming_correlation() -> None:
    returns = [0.01, -0.004, 0.012, 0.0, -0.003, 0.009, 0.002, -0.006]
    trial_sharpes = [0.2, 0.5, 0.1]
    sensitivity = deflated_sharpe_trial_count_sensitivity(
        returns, trial_sharpes=trial_sharpes,
    )
    perfect_dependence, independence = sensitivity["scenarios"]
    assert sensitivity["status"] == "scenario_sensitivity_only"
    assert sensitivity["effective_trial_count_range"] == [1, 3]
    assert sensitivity["correlation_estimated"] is False
    assert sensitivity["project_historical_trials"] == "not_estimated"
    assert perfect_dependence["effective_trials"] == 1
    assert independence["effective_trials"] == 3
    assert perfect_dependence["dsr_probability"] >= independence["dsr_probability"]
    declared = deflated_sharpe(
        returns, effective_trials=3, trial_sharpes=trial_sharpes,
    )
    assert independence["dsr_probability"] == declared["dsr_probability"]


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
    compact_report = native_report(
        (snapshot(1, 1_000.0), snapshot(11, 999.9, (fill,)),
         snapshot(21, 999.7, (), (funding,))),
        fill_references={"order-1": Decimal("100")}, sample_interval_ns=10,
        include_account_rows=False,
    )

    assert report["fills"] == [fill]
    assert report["funding"] == [funding]
    assert report["metrics"]["total_fee"] == pytest.approx(0.1)
    assert report["metrics"]["total_funding"] == pytest.approx(-0.2)
    assert compact_report["metrics"] == report["metrics"]
    assert "account_rows" not in compact_report


def test_native_report_reconciles_signed_maker_fee_rebates() -> None:
    opening = {"event_id": "fill-open", "client_order_id": "order-open", "order_side": "BUY",
               "last_qty": "1", "last_px": "100", "commission": "-0.1 USDT"}
    closing = {"event_id": "fill-close", "client_order_id": "order-close", "order_side": "SELL",
               "last_qty": "1", "last_px": "101", "commission": "-0.1 USDT"}

    def snapshot(timestamp, equity, *, mark=None, positions=(), fills=()):
        return SimpleNamespace(
            decision_ts_ns=timestamp, equity=Decimal(equity), currency="USDT", mark=mark,
            fills=fills, funding=(), positions=positions, orders=(),
            liquidation_capability=True, liquidation_orders=(), contract_multiplier=Decimal(1),
        )

    position = ({"side": "LONG", "quantity": "1", "multiplier": "1",
                 "is_inverse": False, "settlement_currency": "USDT"},)
    report = native_report(
        (snapshot(1, "1000"), snapshot(11, "1000.1", mark=Decimal(100),
                                      positions=position, fills=(opening,)),
         snapshot(21, "1001.2", fills=(opening, closing))),
        fill_references={"order-open": Decimal(100), "order-close": Decimal(101)},
        sample_interval_ns=10, linear_equity_rounding=Decimal(0),
    )

    assert report["metrics"]["total_fee"] == pytest.approx(-0.2)
    assert report["metrics"]["gross_return_before_reported_costs"] == pytest.approx(0.001)
    assert report["equity_reconciliation"]["terminal_checked"] is True
    assert report["equity_reconciliation"]["max_abs_residual"] == "0"


@pytest.mark.parametrize("field, amount", (("commission", "1,2 USDT"),
                                            ("pnl_change", "1,2 USDT")))
def test_native_money_rejects_malformed_grouping_separator(field: str, amount: str) -> None:
    base = dict(currency="USDT", fills=(), funding=(), positions=(), orders=())
    fill = {"event_id": "fill-1", "client_order_id": "order-1", "order_side": "BUY",
            "last_qty": "1", "last_px": "100", "commission": "0.1 USDT"}
    funding = {"event_id": "funding-1", "pnl_change": "-0.1 USDT"}
    if field == "commission":
        fill[field] = amount
    else:
        funding[field] = amount
    snapshots = (
        SimpleNamespace(decision_ts_ns=1, equity=Decimal("1000"), **base),
        SimpleNamespace(decision_ts_ns=11, equity=Decimal("1000"),
                         **{**base, "fills": (fill,) if field == "commission" else (),
                            "funding": (funding,) if field == "pnl_change" else ()}),
    )

    with pytest.raises(MetricsError, match="invalid native ledger amount"):
        account_rows(snapshots, fill_references={"order-1": Decimal("100")})


def test_native_money_accepts_nautilus_decimal_amount_serialization() -> None:
    from nautilus_trader.model import Money

    base = dict(currency="USDT", fills=(), funding=(), positions=(), orders=())
    amount = str(Money.from_str("1234.50 USDT"))
    fill = {"event_id": "fill-1", "client_order_id": "order-1", "order_side": "BUY",
            "last_qty": "1", "last_px": "100", "commission": amount}
    snapshots = (
        SimpleNamespace(decision_ts_ns=1, equity=Decimal("1000"), **base),
        SimpleNamespace(decision_ts_ns=11, equity=Decimal("999"),
                         **{**base, "fills": (fill,)}),
    )

    rows = account_rows(snapshots, fill_references={"order-1": Decimal("100")})

    assert rows[1]["fee"] == pytest.approx(1234.5)


def test_native_report_merges_same_timestamp_facts_and_uses_terminal_equity() -> None:
    fill = {"event_id": "fill-1", "client_order_id": "order-1", "order_side": "BUY",
            "last_qty": "1", "last_px": "100", "commission": "0.1 USDT"}
    funding = {"event_id": "funding-1", "pnl_change": "-0.2 USDT"}
    base = dict(currency="USDT", positions=(), orders=(),
                liquidation_capability=True, liquidation_orders=())
    snapshots = (
        SimpleNamespace(decision_ts_ns=1, equity=1000.0, fills=(), funding=(), **base),
        SimpleNamespace(decision_ts_ns=11, equity=999.9, fills=(fill,), funding=(), **base),
        SimpleNamespace(decision_ts_ns=11, equity=999.7, fills=(), funding=(funding,), **base),
        SimpleNamespace(decision_ts_ns=21, equity=999.7, fills=(), funding=(), **base),
    )
    report = native_report(snapshots, fill_references={"order-1": Decimal("100")},
                           sample_interval_ns=10)
    assert len(report["account_rows"]) == 3
    assert report["account_rows"][1]["equity"] == 999.7
    assert report["metrics"]["total_fee"] == pytest.approx(0.1)
    assert report["metrics"]["total_funding"] == pytest.approx(-0.2)
    assert report["fills"] == [fill]
    assert report["funding"] == [funding]


def test_native_report_counts_identical_event_ids_once_across_same_timestamp_barriers() -> None:
    fill = {"event_id": "fill-1", "client_order_id": "order-1", "order_side": "BUY",
            "last_qty": "1", "last_px": "100", "commission": "0.1 USDT"}
    funding = {"event_id": "funding-1", "pnl_change": "-0.2 USDT"}
    base = dict(currency="USDT", positions=(), orders=(),
                liquidation_capability=True, liquidation_orders=())
    snapshots = (
        SimpleNamespace(decision_ts_ns=1, equity=Decimal("1000"), fills=(), funding=(), **base),
        SimpleNamespace(decision_ts_ns=11, equity=Decimal("999.9"),
                        fills=(fill,), funding=(funding,), **base),
        # Native cumulative barriers can repeat the same immutable event facts.
        SimpleNamespace(decision_ts_ns=11, equity=Decimal("999.7"),
                        fills=(fill,), funding=(funding,), **base),
        SimpleNamespace(decision_ts_ns=21, equity=Decimal("999.7"),
                        fills=(), funding=(), **base),
    )

    report = native_report(snapshots, fill_references={"order-1": Decimal("100")},
                           sample_interval_ns=10)

    assert report["account_rows"][-1]["equity"] == 999.7
    assert report["metrics"]["total_fee"] == pytest.approx(0.1)
    assert report["metrics"]["total_funding"] == pytest.approx(-0.2)
    assert report["fills"] == [fill]
    assert report["funding"] == [funding]


def test_same_timestamp_native_fill_identity_conflict_is_rejected() -> None:
    base = dict(currency="USDT", funding=(), positions=(), orders=(),
                liquidation_capability=True, liquidation_orders=())
    first = {"event_id": "fill-1", "client_order_id": "order-1", "order_side": "BUY",
             "last_qty": "1", "last_px": "100", "commission": "0.1 USDT"}
    changed = {**first, "last_px": "101"}
    snapshots = (
        SimpleNamespace(decision_ts_ns=1, equity=1000.0, fills=(), **base),
        SimpleNamespace(decision_ts_ns=11, equity=999.9, fills=(first,), **base),
        SimpleNamespace(decision_ts_ns=11, equity=999.8, fills=(changed,), **base),
    )
    with pytest.raises(MetricsError, match="conflicting facts"):
        native_report(snapshots, fill_references={"order-1": Decimal("100")},
                      sample_interval_ns=10)


def test_account_metrics_consume_one_shot_iterable_without_row_materialization() -> None:
    facts = [
        {"ts_ns": index * 10, "equity": equity, "external_cashflow": 0.0,
         "turnover_notional": 0.0, "fee": 0.0, "funding": 0.0, "slippage_cost": 0.0}
        for index, equity in enumerate((1000.0, 1010.0, 1005.0, 1020.0), start=1)
    ]
    from_list = from_account_facts(facts, periods_per_year=2, sample_interval_ns=10)
    from_iterator = from_account_facts(iter(facts), periods_per_year=2, sample_interval_ns=10)
    assert from_iterator == from_list


def _linear_ledger():
    """Hand-audited long round trip, then short round trip, with signed funding."""
    fills = (
        dict(event_id="buy-open", client_order_id="b1", order_side="BUY",
             last_qty="2", last_px="101", commission="0.200 USDT"),
        dict(event_id="sell-close", client_order_id="s1", order_side="SELL",
             last_qty="2", last_px="104", commission="0.208 USDT"),
        dict(event_id="sell-open", client_order_id="s2", order_side="SELL",
             last_qty="1", last_px="103", commission="0.103 USDT"),
        dict(event_id="buy-close", client_order_id="b2", order_side="BUY",
             last_qty="1", last_px="100", commission="0.100 USDT"),
    )
    funding = (dict(event_id="f1", pnl_change="-0.500 USDT"),
               dict(event_id="f2", pnl_change="0.100 USDT"))

    def snapshot(ts, equity, fill_rows=(), adjustments=(), positions=()):
        return SimpleNamespace(
            decision_ts_ns=ts, equity=Decimal(equity), currency="USDT", fills=fill_rows,
            funding=adjustments, positions=positions, orders=(),
            liquidation_capability=True, liquidation_orders=(),
        )

    snapshots = (
        snapshot(1, "1000"),
        snapshot(11, "997.800", fills[:1], positions=({"side": "LONG"},)),
        snapshot(11, "1005.192", fills[:2], funding),
        snapshot(21, "1005.089", fills[2:3], positions=({"side": "SHORT"},)),
        snapshot(21, "1007.989", fills, funding),
        snapshot(31, "1007.989", fills, funding),
    )
    return snapshots, {"b1": Decimal(100), "s1": Decimal(105),
                       "s2": Decimal(104), "b2": Decimal(99)}


@pytest.mark.parametrize(
    "field, replacement, message",
    (("fills", (None,), "native fill history fact must be a mapping"),
     ("fills", ({"event_id": "bad-fill", "client_order_id": "b1",
                  "order_side": "BUY", "last_qty": "1", "last_px": "100"},),
      "native fill history fact lacks required fields"),
     ("funding", ({"event_id": "bad-funding"},),
      "native funding history fact lacks required fields")),
)
def test_account_rows_fail_closed_on_malformed_native_history_facts(field, replacement, message) -> None:
    snapshots, references = _linear_ledger()
    target = snapshots[1 if field == "fills" else 2]
    malformed = SimpleNamespace(**{**vars(target), field: replacement})
    with pytest.raises(MetricsError, match=message):
        account_rows((snapshots[0], malformed, snapshots[-1]), fill_references=references)


def test_linear_ledger_reconciles_every_flat_barrier_without_double_costs() -> None:
    snapshots, references = _linear_ledger()
    report = native_report(snapshots, fill_references=references, sample_interval_ns=10,
                           linear_equity_rounding=Decimal(0))
    check = report["equity_reconciliation"]
    assert check["checked_flat_barriers"] == 4
    assert check["unchecked_open_barriers"] == 2
    assert check["terminal_checked"] is True
    assert Decimal(check["max_abs_residual"]) == 0
    last = check["rows"][-1]
    assert Decimal(last["realized_price_pnl"]) == 9
    assert Decimal(last["fee"]) == Decimal("0.611")
    assert Decimal(last["funding"]) == Decimal("-0.400")
    assert Decimal(last["expected_equity"]) == Decimal("1007.989")
    assert report["metrics"]["net_return"] == pytest.approx(0.007989)
    assert report["metrics"]["total_fee"] == pytest.approx(0.611)
    assert report["metrics"]["total_funding"] == pytest.approx(-0.4)
    assert report["metrics"]["total_slippage_cost"] == 6
    assert len(report["fills"]) == 4 and len(report["funding"]) == 2
    compact = native_report(snapshots, fill_references=references, sample_interval_ns=10,
                            linear_equity_rounding=Decimal(0), include_account_rows=False)
    assert compact["metrics"] == report["metrics"]
    assert "rows" not in compact["equity_reconciliation"]
    assert {k: v for k, v in check.items() if k != "rows"} == compact["equity_reconciliation"]


def test_native_ledger_reconciles_explicit_source_identified_deposit_and_withdrawal() -> None:
    base = dict(currency="USDT", fills=(), funding=(), positions=(), orders=(),
                liquidation_capability=True, liquidation_orders=())
    snapshots = (
        SimpleNamespace(decision_ts_ns=1, equity=Decimal("1000"), **base),
        SimpleNamespace(decision_ts_ns=11, equity=Decimal("1100"), **base),
        SimpleNamespace(decision_ts_ns=21, equity=Decimal("1050"), **base),
    )
    deposit = {"event_id": "transfer-1", "decision_ts_ns": 11, "currency": "USDT",
               "kind": "DEPOSIT", "amount": "100", "source_id": "bank-statement-2026-10",
               "source_sha256": "a" * 64}
    withdrawal = {"event_id": "transfer-2", "decision_ts_ns": 21, "currency": "USDT",
                  "kind": "WITHDRAWAL", "amount": "50", "source_id": "bank-statement-2026-10",
                  "source_sha256": "a" * 64}

    report = native_report(
        snapshots, fill_references={}, sample_interval_ns=10,
        external_cashflows=(deposit, deposit, withdrawal),
        linear_equity_rounding=Decimal(0),
    )

    assert report["external_cashflows"] == [deposit, withdrawal]
    assert [row["external_cashflow"] for row in report["account_rows"]] == [0.0, 100.0, -50.0]
    assert report["metrics"]["net_return"] == pytest.approx(0.0)
    assert report["metrics"]["gross_return_before_reported_costs"] == pytest.approx(0.0)
    check = report["equity_reconciliation"]
    assert check["unique_external_cashflows"] == 2
    assert check["net_external_cashflow"] == "50"
    assert check["max_abs_residual"] == "0"


def test_native_ledger_rejects_unverifiable_or_ambiguous_external_cashflow_facts() -> None:
    base = dict(currency="USDT", fills=(), funding=(), positions=(), orders=(),
                liquidation_capability=True, liquidation_orders=())
    snapshots = (
        SimpleNamespace(decision_ts_ns=1, equity=Decimal("1000"), **base),
        SimpleNamespace(decision_ts_ns=11, equity=Decimal("1100"), **base),
        SimpleNamespace(decision_ts_ns=21, equity=Decimal("1100"), **base),
    )
    flow = {"event_id": "transfer-1", "decision_ts_ns": 11, "currency": "USDT",
            "kind": "DEPOSIT", "amount": "100", "source_id": "statement",
            "source_sha256": "b" * 64}
    changed = {**flow, "amount": "101"}
    with pytest.raises(MetricsError, match="identity has conflicting facts"):
        account_rows(snapshots, fill_references={}, external_cashflows=(flow, changed))
    with pytest.raises(MetricsError, match="source SHA-256 required"):
        account_rows(snapshots, fill_references={},
                     external_cashflows=({**flow, "source_sha256": "unknown"},))
    duplicate_timestamp = (snapshots[0], snapshots[1],
                           SimpleNamespace(decision_ts_ns=11, equity=Decimal("1100"), **base),
                           snapshots[2])
    with pytest.raises(MetricsError, match="duplicate native barriers"):
        account_rows(duplicate_timestamp, fill_references={}, external_cashflows=(flow,))


@pytest.mark.parametrize("source_time", [True, 0, 12, "10"])
def test_native_ledger_rejects_invalid_external_cashflow_source_time(source_time: object) -> None:
    base = dict(currency="USDT", fills=(), funding=(), positions=(), orders=(),
                liquidation_capability=True, liquidation_orders=())
    snapshots = (
        SimpleNamespace(decision_ts_ns=1, equity=Decimal("1000"), **base),
        SimpleNamespace(decision_ts_ns=11, equity=Decimal("1100"), **base),
        SimpleNamespace(decision_ts_ns=21, equity=Decimal("1100"), **base),
    )
    flow = {"event_id": "transfer-1", "decision_ts_ns": 11,
            "source_event_time_ns": source_time, "currency": "USDT",
            "kind": "DEPOSIT", "amount": "100", "source_id": "statement",
            "source_sha256": "b" * 64}

    with pytest.raises(MetricsError, match="source time must precede or equal"):
        account_rows(snapshots, fill_references={}, external_cashflows=(flow,))


@pytest.mark.parametrize("kind", [[], {}, None, 1, "TRANSFER"])
def test_native_ledger_rejects_invalid_external_cashflow_kind(kind: object) -> None:
    base = dict(currency="USDT", fills=(), funding=(), positions=(), orders=(),
                liquidation_capability=True, liquidation_orders=())
    snapshots = (
        SimpleNamespace(decision_ts_ns=1, equity=Decimal("1000"), **base),
        SimpleNamespace(decision_ts_ns=11, equity=Decimal("1000"), **base),
    )
    flow = {"event_id": "transfer-1", "decision_ts_ns": 11, "currency": "USDT",
            "kind": kind, "amount": "1", "source_id": "statement",
            "source_sha256": "d" * 64}

    with pytest.raises(MetricsError, match="cashflow kind must be DEPOSIT or WITHDRAWAL"):
        account_rows(snapshots, fill_references={}, external_cashflows=(flow,))


def test_explicit_external_cashflow_is_reconciled_while_linear_position_is_open() -> None:
    opening = {"event_id": "open", "client_order_id": "open-order", "order_side": "BUY",
               "last_qty": "1", "last_px": "100", "commission": "0.1 USDT"}
    closing = {"event_id": "close", "client_order_id": "close-order", "order_side": "SELL",
               "last_qty": "1", "last_px": "115", "commission": "0.115 USDT"}
    position = {"side": "LONG", "quantity": "1", "multiplier": "1", "is_inverse": False,
                "settlement_currency": "USDT"}
    base = dict(currency="USDT", funding=(), orders=(), liquidation_capability=True,
                liquidation_orders=())
    snapshots = (
        SimpleNamespace(decision_ts_ns=1, equity=Decimal("1000"), fills=(), positions=(), **base),
        SimpleNamespace(decision_ts_ns=11, equity=Decimal("999.9"), fills=(opening,),
                         positions=(position,), mark=Decimal("100"), **base),
        SimpleNamespace(decision_ts_ns=21, equity=Decimal("1109.9"), fills=(),
                         positions=(position,), mark=Decimal("110"), **base),
        SimpleNamespace(decision_ts_ns=31, equity=Decimal("1114.785"), fills=(closing,),
                         positions=(), mark=Decimal("115"), **base),
    )
    flow = {"event_id": "transfer", "decision_ts_ns": 21, "currency": "USDT",
            "kind": "DEPOSIT", "amount": "100", "source_id": "statement",
            "source_sha256": "c" * 64}

    report = native_report(
        snapshots, fill_references={"open-order": Decimal(100), "close-order": Decimal(115)},
        sample_interval_ns=10, external_cashflows=(flow,),
        linear_equity_rounding=Decimal(0),
    )

    check = report["equity_reconciliation"]
    assert check["checked_open_barriers"] == 2
    assert check["unchecked_open_barriers"] == 0
    assert check["net_external_cashflow"] == "100"
    assert Decimal(check["max_abs_residual"]) == 0
    assert check["rows"][2]["external_cashflow"] == "100"
    expected_time_weighted_return = (
        (999.9 / 1000.0) * ((1109.9 - 100.0) / 999.9) * (1114.785 / 1109.9) - 1
    )
    assert report["metrics"]["net_return"] == pytest.approx(expected_time_weighted_return)


def test_linear_reconciliation_checks_open_marked_equity_and_rejects_mismatch() -> None:
    opened = {"event_id": "open-1", "client_order_id": "open-order", "order_side": "BUY",
              "last_qty": "2", "last_px": "100", "commission": "0.2 USDT"}
    closed = {"event_id": "close-1", "client_order_id": "close-order", "order_side": "SELL",
              "last_qty": "2", "last_px": "103", "commission": "0.206 USDT"}
    funding = {"event_id": "funding-1", "pnl_change": "-0.1 USDT"}
    position = {"side": "LONG", "quantity": "2", "multiplier": "1",
                "is_inverse": False, "settlement_currency": "USDT"}

    def snapshot(ts, equity, fills=(), adjustments=(), positions=(), mark=None):
        return SimpleNamespace(
            decision_ts_ns=ts, equity=Decimal(equity), currency="USDT", fills=fills,
            funding=adjustments, positions=positions, orders=(), mark=mark,
        )

    snapshots = (
        snapshot(1, "1000"),
        snapshot(11, "1003.8", (opened,), positions=(position,), mark=Decimal("102")),
        snapshot(21, "1005.494", (closed,), (funding,)),
    )
    result = reconcile_linear_equity(snapshots, rounding_per_event=Decimal(0))
    assert result["checked_flat_barriers"] == 2
    assert result["checked_open_barriers"] == 1
    assert result["open_barriers"] == 1
    assert result["unchecked_open_barriers"] == 0
    assert result["terminal_checked"] is True
    open_row = result["rows"][1]
    assert Decimal(open_row["price_pnl_at_mark"]) == 4
    assert Decimal(open_row["expected_equity"]) == Decimal("1003.8")

    bad_open = snapshot(11, "1003.9", (opened,), positions=(position,), mark=Decimal("102"))
    with pytest.raises(MetricsError, match="equity mismatch at open barrier"):
        reconcile_linear_equity((snapshots[0], bad_open, snapshots[-1]),
                                rounding_per_event=Decimal(0))


def test_native_metrics_reconcile_non_unit_linear_contract_multiplier() -> None:
    multiplier = Decimal("0.001")
    opening = {"event_id": "open-m", "client_order_id": "open-order", "order_side": "BUY",
               "last_qty": "10", "last_px": "100", "commission": "0.01 USDT"}
    closing = {"event_id": "close-m", "client_order_id": "close-order", "order_side": "SELL",
               "last_qty": "10", "last_px": "102", "commission": "0.0102 USDT"}
    position = {"side": "LONG", "quantity": "10", "multiplier": str(multiplier),
                "is_inverse": False, "settlement_currency": "USDT"}

    def snapshot(ts, equity, fills=(), positions=(), mark=None):
        return SimpleNamespace(
            decision_ts_ns=ts, equity=Decimal(equity), currency="USDT", fills=fills,
            funding=(), positions=positions, orders=(), mark=mark,
            contract_multiplier=multiplier,
        )

    snapshots = (
        snapshot(1, "1000"),
        snapshot(11, "1000", (opening,), (position,), Decimal("101")),
        snapshot(21, "999.9998", (closing,)),
    )
    report = native_report(
        snapshots,
        fill_references={"open-order": Decimal("99"), "close-order": Decimal("103")},
        sample_interval_ns=10,
        linear_equity_rounding=Decimal(0),
    )

    assert sum(row["turnover_notional"] for row in report["account_rows"]) == pytest.approx(2.02)
    assert report["metrics"]["total_slippage_cost"] == pytest.approx(0.02)
    assert report["equity_reconciliation"]["max_abs_residual"] == "0"
    assert report["equity_reconciliation"]["checked_open_barriers"] == 1
    assert report["equity_reconciliation"]["rows"][1]["price_pnl_at_mark"] == "0.010"


def test_linear_liquidation_facts_and_terminal_equity_reconcile_together() -> None:
    opened = {"event_id": "open-1", "client_order_id": "open-order", "order_side": "BUY",
              "last_qty": "1", "last_px": "100", "commission": "0.1 USDT"}
    liquidation_fill = {
        "event_id": "liquidation-fill-1", "client_order_id": "LIQUIDATION-BINANCE-1",
        "order_side": "SELL", "last_qty": "1", "last_px": "75",
        "commission": "0.075 USDT",
    }
    liquidation_order = {"client_order_id": "LIQUIDATION-BINANCE-1",
                         "tags": ["LIQUIDATION_BINANCE_CLOSE"]}
    funding = {"event_id": "funding-1", "pnl_change": "0.2 USDT"}
    position = {"side": "LONG", "quantity": "1", "multiplier": "1",
                "is_inverse": False, "settlement_currency": "USDT"}

    def snapshot(ts, equity, fills=(), adjustments=(), positions=(), liquidations=()):
        return SimpleNamespace(
            decision_ts_ns=ts, equity=Decimal(equity), currency="USDT", fills=fills,
            funding=adjustments, positions=positions, orders=(), mark=Decimal("80"),
            liquidation_capability=True, liquidation_orders=liquidations,
        )

    snapshots = (
        snapshot(1, "1000"),
        snapshot(11, "979.9", (opened,), positions=(position,)),
        snapshot(21, "975.025", (liquidation_fill,), (funding,),
                 liquidations=(liquidation_order,)),
    )
    report = native_report(
        snapshots, fill_references={"open-order": Decimal("100"),
                                    "LIQUIDATION-BINANCE-1": Decimal("80")},
        sample_interval_ns=10, linear_equity_rounding=Decimal(0),
    )
    check = report["equity_reconciliation"]
    assert check["checked_open_barriers"] == 1
    assert check["unchecked_open_barriers"] == 0
    assert check["terminal_checked"] is True
    assert check["max_abs_residual"] == "0"
    assert report["terminal_flat"] is True
    assert report["liquidations_observed"] is True
    assert report["liquidations"] == 1
    assert report["metrics"]["total_fee"] == pytest.approx(0.175)
    assert report["metrics"]["total_funding"] == pytest.approx(0.2)
    assert report["metrics"]["total_slippage_cost"] == pytest.approx(5.0)


def test_native_report_deduplicates_liquidation_order_lifecycle_snapshots() -> None:
    initial = {"client_order_id": "LIQUIDATION-BINANCE-1", "instrument_id": "BTCUSDT.BINANCE",
               "order_side": "SELL", "quantity": "1", "tags": ["LIQUIDATION_BINANCE_CLOSE"],
               "status": "ACCEPTED", "filled_qty": "0", "leaves_qty": "1"}
    updated = {**initial, "status": "FILLED", "filled_qty": "1", "leaves_qty": "0"}

    def snapshot(timestamp, liquidation_orders):
        return SimpleNamespace(
            decision_ts_ns=timestamp, equity=Decimal("1000"), currency="USDT", fills=(),
            funding=(), positions=(), orders=(), liquidation_capability=True,
            liquidation_orders=liquidation_orders,
        )

    report = native_report(
        (snapshot(1, ()), snapshot(2, (initial,)), snapshot(3, (updated,))),
        fill_references={}, sample_interval_ns=1,
    )

    assert report["liquidations"] == 1
    assert report["liquidation_orders"] == [updated]


@pytest.mark.parametrize(
    "first_filled, next_filled, message",
    (("0.5", "0.4", "cumulative fill quantity regressed"),
     ("0.5", "1.1", "exceeds order quantity")),
)
def test_native_report_rejects_invalid_cumulative_liquidation_fills(
        first_filled: str, next_filled: str, message: str) -> None:
    first = {
        "client_order_id": "LIQUIDATION-BINANCE-1", "instrument_id": "BTCUSDT.BINANCE",
        "order_side": "SELL", "quantity": "1", "tags": ["LIQUIDATION_BINANCE_CLOSE"],
        "status": "PARTIALLY_FILLED", "filled_qty": first_filled,
        "leaves_qty": str(Decimal("1") - Decimal(first_filled)),
    }
    next_order = {
        **first, "status": "PARTIALLY_FILLED", "filled_qty": next_filled,
        "leaves_qty": str(Decimal("1") - Decimal(next_filled)),
    }

    def snapshot(timestamp: int, liquidation_orders: tuple[dict, ...]):
        return SimpleNamespace(
            decision_ts_ns=timestamp, equity=Decimal("1000"), currency="USDT", fills=(),
            funding=(), positions=(), orders=(), liquidation_capability=True,
            liquidation_orders=liquidation_orders,
        )

    with pytest.raises(MetricsError, match=message):
        native_report(
            (snapshot(0, ()), snapshot(1, (first,)), snapshot(2, (next_order,))),
            fill_references={}, sample_interval_ns=1,
        )


def test_native_report_orders_facts_and_liquidation_state_by_snapshot_time() -> None:
    buy = {"event_id": "buy-1", "client_order_id": "buy-order", "order_side": "BUY",
           "last_qty": "1", "last_px": "100", "commission": "0.1 USDT"}
    sell = {"event_id": "sell-1", "client_order_id": "sell-order", "order_side": "SELL",
            "last_qty": "1", "last_px": "100", "commission": "0.1 USDT"}
    funding_early = {"event_id": "funding-early", "pnl_change": "-0.2 USDT"}
    funding_late = {"event_id": "funding-late", "pnl_change": "0.1 USDT"}
    initial_order = {
        "client_order_id": "LIQUIDATION-BINANCE-1", "instrument_id": "BTCUSDT.BINANCE",
        "order_side": "SELL", "quantity": "1", "tags": ["LIQUIDATION_BINANCE_CLOSE"],
        "status": "ACCEPTED", "filled_qty": "0", "leaves_qty": "1",
    }
    updated_order = {**initial_order, "status": "FILLED", "filled_qty": "1", "leaves_qty": "0"}

    def snapshot(timestamp, equity, *, fills=(), funding=(), liquidation_orders=()):
        return SimpleNamespace(
            decision_ts_ns=timestamp, equity=Decimal(equity), currency="USDT", fills=fills,
            funding=funding, positions=(), orders=(), liquidation_capability=True,
            liquidation_orders=liquidation_orders,
        )

    # The t=21 barrier arrives before t=11. Report facts and cumulative order
    # lifecycle state must still follow the authoritative timestamp order.
    snapshots = (
        snapshot(1, "1000"),
        snapshot(21, "999.8", fills=(sell,), funding=(funding_late,),
                 liquidation_orders=(updated_order,)),
        snapshot(11, "999.9", fills=(buy,), funding=(funding_early,),
                 liquidation_orders=(initial_order,)),
    )
    report = native_report(
        snapshots, fill_references={"buy-order": Decimal(100), "sell-order": Decimal(100)},
        sample_interval_ns=10,
    )

    assert [row["ts_ns"] for row in report["account_rows"]] == [1, 11, 21]
    assert report["fills"] == [buy, sell]
    assert report["funding"] == [funding_early, funding_late]
    assert report["liquidation_orders"] == [updated_order]
    assert report["metrics"]["total_fee"] == pytest.approx(0.2)
    assert report["metrics"]["total_funding"] == pytest.approx(-0.1)


@pytest.mark.parametrize("field, value", (("quantity", "2"),
                                          ("tags", ["LIQUIDATION_OTHER_CLOSE"])))
def test_native_report_rejects_conflicting_liquidation_order_identity(field, value) -> None:
    order = {"client_order_id": "LIQUIDATION-BINANCE-1", "instrument_id": "BTCUSDT.BINANCE",
             "order_side": "SELL", "quantity": "1", "tags": ["LIQUIDATION_BINANCE_CLOSE"],
             "status": "ACCEPTED"}
    changed = {**order, field: value}

    def snapshot(timestamp, liquidation_orders):
        return SimpleNamespace(
            decision_ts_ns=timestamp, equity=Decimal("1000"), currency="USDT", fills=(),
            funding=(), positions=(), orders=(), liquidation_capability=True,
            liquidation_orders=liquidation_orders,
        )

    with pytest.raises(MetricsError, match="liquidation order identity has conflicting facts"):
        native_report(
            (snapshot(1, (order,)), snapshot(2, (changed,)), snapshot(3, ())),
            fill_references={}, sample_interval_ns=1,
        )


def test_native_report_rejects_noncanonical_liquidation_order_identity() -> None:
    order = {"client_order_id": " LIQUIDATION-BINANCE-1 ",
             "instrument_id": "BTCUSDT.BINANCE", "order_side": "SELL", "quantity": "1",
             "tags": ["LIQUIDATION_BINANCE_CLOSE"], "status": "ACCEPTED"}
    snapshots = tuple(SimpleNamespace(
        decision_ts_ns=timestamp, equity=Decimal("1000"), currency="USDT", fills=(),
        funding=(), positions=(), orders=(), liquidation_capability=True,
        liquidation_orders=orders,
    ) for timestamp, orders in ((1, ()), (2, (order,)), (3, ())))

    with pytest.raises(MetricsError, match="native liquidation order lacks identity"):
        native_report(snapshots, fill_references={}, sample_interval_ns=1)


def test_linear_reconciliation_detects_missing_cost_or_extra_cashflow() -> None:
    snapshots, _ = _linear_ledger()
    changed = SimpleNamespace(**{**vars(snapshots[-1]), "equity": Decimal("1008.089")})
    with pytest.raises(MetricsError, match="equity mismatch"):
        reconcile_linear_equity((*snapshots[:-1], changed))
    with pytest.raises(MetricsError, match="flat event-free initial"):
        reconcile_linear_equity(snapshots[1:])


def test_reported_account_state_does_not_infer_external_cashflow() -> None:
    base = dict(currency="USDT", fills=(), funding=(), positions=(), orders=())
    snapshots = (
        SimpleNamespace(decision_ts_ns=1, equity=Decimal("1000"), reported=False, **base),
        SimpleNamespace(decision_ts_ns=11, equity=Decimal("1100"), reported=True, **base),
    )

    # AccountState.reported has no transfer semantics. Without a separately
    # sourced cashflow fact, the unexplained balance change must fail closed.
    with pytest.raises(MetricsError, match="linear native equity mismatch"):
        reconcile_linear_equity(snapshots, rounding_per_event=Decimal(0))


@pytest.mark.parametrize("label", ("fill", "funding"))
@pytest.mark.parametrize("identity", (None, "", " ", " fill-1 ", 0))
def test_account_rows_reject_missing_event_identity_before_summarizing(label, identity) -> None:
    snapshots, references = _linear_ledger()
    changed = dict(snapshots[1].fills[0] if label == "fill" else snapshots[2].funding[0])
    changed["event_id"] = identity
    bad = SimpleNamespace(**{**vars(snapshots[1]),
                             "fills": (changed,) if label == "fill" else (),
                             "funding": (changed,) if label == "funding" else ()})
    with pytest.raises(MetricsError, match="lacks identity"):
        account_rows((snapshots[0], bad), fill_references=references)


@pytest.mark.parametrize("field, value", (("order_side", "UNKNOWN"), ("order_side", []),
                                         ("order_side", {}), ("last_qty", "0"),
                                         ("last_qty", "NaN"), ("last_px", "-1"),
                                         ("last_px", "broken")))
def test_account_rows_reject_invalid_native_fill_economics(field, value) -> None:
    snapshots, references = _linear_ledger()
    changed = {**snapshots[1].fills[0], field: value}
    bad = SimpleNamespace(**{**vars(snapshots[1]), "fills": (changed,)})
    with pytest.raises(MetricsError, match="native fill"):
        account_rows((snapshots[0], bad), fill_references=references)


@pytest.mark.parametrize("side", [[], {}, None, "UNKNOWN"])
def test_linear_reconciliation_rejects_invalid_native_position_side(side: object) -> None:
    fill = {"event_id": "open", "client_order_id": "open-order", "order_side": "BUY",
            "last_qty": "1", "last_px": "100", "commission": "0.1 USDT"}
    position = {"side": side, "quantity": "1", "multiplier": "1", "is_inverse": False,
                "settlement_currency": "USDT"}
    base = dict(currency="USDT", funding=(), orders=(), liquidation_capability=True,
                liquidation_orders=())
    snapshots = (
        SimpleNamespace(decision_ts_ns=1, equity=Decimal("1000"), fills=(), positions=(), **base),
        SimpleNamespace(decision_ts_ns=11, equity=Decimal("999.9"), fills=(fill,),
                         positions=(position,), mark=Decimal("100"), **base),
    )

    with pytest.raises(MetricsError, match="native linear position side and quantity are invalid"):
        reconcile_linear_equity(snapshots, rounding_per_event=Decimal(0))


def test_funding_identity_conflict_across_timestamps_is_rejected() -> None:
    snapshots, references = _linear_ledger()
    changed = {**snapshots[2].funding[0], "pnl_change": "1 USDT"}
    bad = SimpleNamespace(**{**vars(snapshots[-1]), "funding": (changed,)})
    with pytest.raises(MetricsError, match="funding identity has conflicting"):
        account_rows((*snapshots[:-1], bad), fill_references=references)


def test_snapshot_currency_change_and_invalid_timestamp_are_rejected() -> None:
    snapshots, references = _linear_ledger()
    for change, message in (({"currency": "USD"}, "conflicting currencies"),
                            ({"decision_ts_ns": True}, "timestamp")):
        bad = SimpleNamespace(**{**vars(snapshots[-1]), **change})
        with pytest.raises(MetricsError, match=message):
            account_rows((*snapshots[:-1], bad), fill_references=references)


@pytest.mark.parametrize(
    "snapshot, message",
    (
        (SimpleNamespace(decision_ts_ns=1), "lacks currency"),
        (SimpleNamespace(decision_ts_ns=1, currency="USDT"), "lacks timestamp, currency, or equity"),
        (SimpleNamespace(decision_ts_ns=1, currency="USDT", equity=float("nan")), "nonfinite native equity"),
    ),
)
def test_native_snapshot_shape_and_equity_fail_closed(snapshot, message) -> None:
    with pytest.raises(MetricsError, match=message):
        account_rows((snapshot,), fill_references={})


@pytest.mark.parametrize("snapshots", (None, 1, "not-snapshots"))
def test_native_snapshot_collection_requires_a_sequence(snapshots) -> None:
    with pytest.raises(MetricsError, match="native account snapshots required"):
        account_rows(snapshots, fill_references={})


@pytest.mark.parametrize("field, value, label", [
    ("fills", None, "fill"), ("funding", 7, "funding"),
])
def test_native_snapshot_economic_history_requires_sequences(field, value, label) -> None:
    snapshots, references = _linear_ledger()
    malformed = SimpleNamespace(**{**vars(snapshots[1]), field: value})
    with pytest.raises(MetricsError, match=f"native {label} history must be a sequence"):
        account_rows((snapshots[0], malformed), fill_references=references)


@pytest.mark.parametrize("field, value, label", [
    ("positions", None, "position"), ("orders", "not-orders", "order"),
])
def test_native_report_terminal_fact_collections_require_sequences(field, value, label) -> None:
    snapshots, references = _linear_ledger()
    malformed = SimpleNamespace(**{**vars(snapshots[-1]), field: value})
    with pytest.raises(MetricsError, match=f"native {label} history must be a sequence"):
        native_report((*snapshots[:-1], malformed), fill_references=references,
                      sample_interval_ns=10)


def test_native_report_liquidation_order_history_requires_sequence() -> None:
    snapshots, references = _linear_ledger()
    malformed = SimpleNamespace(**{**vars(snapshots[-1]), "liquidation_orders": 1})
    with pytest.raises(MetricsError, match="native liquidation order history must be a sequence"):
        native_report((*snapshots[:-1], malformed), fill_references=references,
                      sample_interval_ns=10)


@pytest.mark.parametrize("client_order_id", [None, 1, [], {}, " order-1 "])
def test_native_fill_requires_string_client_order_identity(client_order_id) -> None:
    snapshots, references = _linear_ledger()
    fill = {**snapshots[1].fills[0], "client_order_id": client_order_id}
    malformed = SimpleNamespace(**{**vars(snapshots[1]), "fills": (fill,)})
    with pytest.raises(MetricsError, match="client order identity is invalid"):
        account_rows((snapshots[0], malformed), fill_references=references)


@pytest.mark.parametrize("cashflows", [None, 1, "not-cashflows"])
def test_external_cashflow_collection_requires_sequence(cashflows) -> None:
    snapshots, _ = _linear_ledger()
    with pytest.raises(MetricsError, match="external cashflows must be a sequence"):
        reconcile_linear_equity(snapshots, external_cashflows=cashflows)


@pytest.mark.parametrize("references", [None, [], "not-a-mapping"])
def test_native_fill_references_require_mapping(references) -> None:
    snapshots, _ = _linear_ledger()
    with pytest.raises(MetricsError, match="fill references must be a mapping"):
        account_rows(snapshots, fill_references=references)
