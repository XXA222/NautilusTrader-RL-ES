from __future__ import annotations

import hashlib
import json
from decimal import Decimal
from types import SimpleNamespace

import pytest

from rles_research.binance_cashflows import (
    BinanceIncomePage,
    import_binance_usdm_transfer_income,
    import_binance_usdm_transfer_income_pages,
)
from rles_research.metrics import MetricsError
from rles_research.native_metrics import native_report

pytestmark = pytest.mark.unit


def test_binance_transfer_export_maps_to_native_cashflow_facts_and_reconciles() -> None:
    payload = json.dumps([
        {"incomeType": "TRANSFER", "income": "125.50", "asset": "USDT",
         "time": 15, "tranId": 9001},
        {"incomeType": "TRANSFER", "income": "-25.25", "asset": "USDT",
         "time": 25, "tranId": 9002},
    ], separators=(",", ":"))
    imported = import_binance_usdm_transfer_income(
        payload, account_scope_id="subaccount-7", currency="USDT",
        sample_barriers_ns=(10_000_000, 20_000_000, 30_000_000),
    )

    assert imported.source_sha256 == hashlib.sha256(payload.encode()).hexdigest()
    assert imported.source_id.endswith(imported.source_sha256)
    assert [fact["kind"] for fact in imported.facts] == ["DEPOSIT", "WITHDRAWAL"]
    assert [fact["amount"] for fact in imported.facts] == ["125.50", "25.25"]
    assert [fact["decision_ts_ns"] for fact in imported.facts] == [20_000_000, 30_000_000]
    assert [fact["source_event_time_ns"] for fact in imported.facts] == [15_000_000, 25_000_000]
    assert imported.event_time_range_ns == (15_000_000, 25_000_000)

    base = dict(currency="USDT", fills=(), funding=(), positions=(), orders=(),
                liquidation_capability=True, liquidation_orders=())
    snapshots = tuple(
        SimpleNamespace(decision_ts_ns=timestamp, equity=Decimal(equity), **base)
        for timestamp, equity in ((10_000_000, "1000"), (20_000_000, "1125.50"),
                                  (30_000_000, "1100.25"))
    )
    with pytest.raises(MetricsError, match="explicit sample-barrier attribution opt-in"):
        native_report(
            snapshots, fill_references={}, sample_interval_ns=10_000_000,
            external_cashflows=imported.facts, linear_equity_rounding=Decimal(0),
        )
    report = native_report(
        snapshots, fill_references={}, sample_interval_ns=10_000_000,
        external_cashflows=imported.facts,
        allow_cashflow_barrier_attribution=True,
        linear_equity_rounding=Decimal(0),
    )
    assert report["equity_reconciliation"]["net_external_cashflow"] == "100.25"
    assert report["equity_reconciliation"]["max_abs_residual"] == "0"
    expected_timing = {
        "source_event_time_count": 2,
        "source_event_time_complete": True,
        "source_event_time_exact_barrier_count": 0,
        "max_attribution_lag_ns": 5_000_000,
        "exact_sample_boundary_timing": False,
        "return_timing_mode": "sample_barrier_attribution_only",
    }
    assert report["cashflow_timing"] == expected_timing
    assert report["equity_reconciliation"]["cashflow_timing"] == expected_timing
    assert any("not exact time-weighted" in item for item in imported.limitations)


def test_binance_transfer_exactly_on_sample_barrier_uses_default_twr_path() -> None:
    payload = json.dumps([{
        "incomeType": "TRANSFER", "income": "100", "asset": "USDT",
        "time": 20, "tranId": 9003,
    }], separators=(",", ":"))
    imported = import_binance_usdm_transfer_income(
        payload, account_scope_id="subaccount-7", currency="USDT",
        sample_barriers_ns=(10_000_000, 20_000_000, 30_000_000),
    )
    assert imported.facts[0]["source_event_time_ns"] == 20_000_000
    assert imported.facts[0]["decision_ts_ns"] == 20_000_000

    base = dict(currency="USDT", fills=(), funding=(), positions=(), orders=(),
                liquidation_capability=True, liquidation_orders=())
    snapshots = tuple(
        SimpleNamespace(decision_ts_ns=timestamp, equity=Decimal(equity), **base)
        for timestamp, equity in ((10_000_000, "1000"), (20_000_000, "1100"),
                                  (30_000_000, "1100"))
    )
    report = native_report(
        snapshots, fill_references={}, sample_interval_ns=10_000_000,
        external_cashflows=imported.facts, linear_equity_rounding=Decimal(0),
    )

    assert report["cashflow_timing"] == {
        "source_event_time_count": 1,
        "source_event_time_complete": True,
        "source_event_time_exact_barrier_count": 1,
        "max_attribution_lag_ns": 0,
        "exact_sample_boundary_timing": True,
        "return_timing_mode": "exact_sample_boundary",
    }
    assert report["metrics"]["net_return"] == pytest.approx(0.0)


def test_binance_off_barrier_transfer_uses_synchronous_equity_for_exact_twr() -> None:
    payload = json.dumps([{
        "incomeType": "TRANSFER", "income": "100", "asset": "USDT",
        "time": 15, "tranId": 9004,
    }], separators=(",", ":"))
    imported = import_binance_usdm_transfer_income(
        payload, account_scope_id="subaccount-7", currency="USDT",
        sample_barriers_ns=(10_000_000, 20_000_000, 30_000_000),
    )
    base = dict(currency="USDT", fills=(), funding=(), positions=(), orders=(),
                liquidation_capability=True, liquidation_orders=())
    snapshots = tuple(
        SimpleNamespace(decision_ts_ns=timestamp, equity=Decimal(equity), **base)
        for timestamp, equity in ((10_000_000, "1000"), (20_000_000, "1060"),
                                  (30_000_000, "1100"))
    )
    point = {
        "decision_ts_ns": 15_000_000,
        "equity_after_cashflow": "1050",
        "currency": "USDT",
        "source_id": "account-equity-at-transfer",
        "source_sha256": "e" * 64,
    }
    with pytest.raises(MetricsError, match="source-synchronous event equity"):
        native_report(
            snapshots, fill_references={}, sample_interval_ns=10_000_000,
            external_cashflows=imported.facts,
        )
    with pytest.raises(MetricsError, match="event equity points a sequence"):
        native_report(
            snapshots, fill_references={}, sample_interval_ns=10_000_000,
            external_cashflows=imported.facts, cashflow_equity_points=None,  # type: ignore[arg-type]
        )
    with pytest.raises(MetricsError, match="must match an off-barrier flow time"):
        native_report(
            snapshots, fill_references={}, sample_interval_ns=10_000_000,
            external_cashflows=imported.facts,
            cashflow_equity_points=({**point, "decision_ts_ns": 16_000_000},),
        )
    report = native_report(
        snapshots, fill_references={}, sample_interval_ns=10_000_000,
        external_cashflows=imported.facts, cashflow_equity_points=(point,),
    )

    expected_first_period = (Decimal("950") / Decimal("1000")
                             * Decimal("1060") / Decimal("1050") - 1)
    expected_total = (Decimal("950") / Decimal("1000")
                      * Decimal("1060") / Decimal("1050")
                      * Decimal("1100") / Decimal("1060") - 1)
    assert report["cashflow_return_timing"] == {
        "mode": "exact_event_time_equity", "event_equity_point_count": 1,
    }
    assert report["account_rows"][1]["cashflow_adjusted_return"] == pytest.approx(
        float(expected_first_period),
    )
    assert report["metrics"]["net_return"] == pytest.approx(float(expected_total))
    assert report["metrics"]["net_return"] != pytest.approx(
        float(Decimal("960") / Decimal("1000")
              * Decimal("1100") / Decimal("1060") - 1),
    )


@pytest.mark.parametrize("payload", [
    b"{bad json",
    b"{}",
    b'[{"incomeType":"COMMISSION","income":"-1","asset":"USDT","time":15,"tranId":1}]',
    b'[{"incomeType":"TRANSFER","income":"1","asset":"BTC","time":15,"tranId":1}]',
    b'[{"incomeType":"TRANSFER","income":"0","asset":"USDT","time":15,"tranId":1}]',
    b'[{"incomeType":"TRANSFER","income":"1","asset":"USDT","time":5,"tranId":1}]',
])
def test_binance_transfer_import_fails_closed_on_bad_source_or_alignment(payload: bytes) -> None:
    with pytest.raises(MetricsError):
        import_binance_usdm_transfer_income(
            payload, account_scope_id="account", currency="USDT",
            sample_barriers_ns=(10_000_000, 20_000_000, 30_000_000),
        )


def test_binance_transfer_import_rejects_conflicting_ids_and_invalid_barriers() -> None:
    conflict = json.dumps([
        {"incomeType": "TRANSFER", "income": "1", "asset": "USDT", "time": 15, "tranId": 1},
        {"incomeType": "TRANSFER", "income": "2", "asset": "USDT", "time": 15, "tranId": 1},
    ])
    with pytest.raises(MetricsError, match="conflicting facts"):
        import_binance_usdm_transfer_income(
            conflict, account_scope_id="account", currency="USDT",
            sample_barriers_ns=(10_000_000, 20_000_000),
        )
    with pytest.raises(MetricsError, match="strictly increasing"):
        import_binance_usdm_transfer_income(
            b"[]", account_scope_id="account", currency="USDT",
            sample_barriers_ns=(10, 10),
        )


def test_binance_transfer_import_rejects_duplicate_json_object_keys() -> None:
    payload = (
        b'[{"incomeType":"TRANSFER","incomeType":"TRANSFER","income":"1",'
        b'"asset":"USDT","time":15,"tranId":1}]'
    )
    with pytest.raises(MetricsError, match="duplicate object key"):
        import_binance_usdm_transfer_income(
            payload, account_scope_id="account", currency="USDT",
            sample_barriers_ns=(10_000_000, 20_000_000),
        )


def test_binance_paged_import_binds_exact_pages_and_contiguous_query_scope() -> None:
    page1 = b'[{"incomeType":"TRANSFER","income":"10","asset":"USDT","time":15,"tranId":1}]'
    page2 = b'[{"incomeType":"TRANSFER","income":"-2","asset":"USDT","time":25,"tranId":2}]'
    imported = import_binance_usdm_transfer_income_pages(
        (
            BinanceIncomePage(10, 19, 2, page1),
            BinanceIncomePage(20, 30, 2, page2),
        ),
        query_start_ms=10,
        query_end_ms=30,
        account_scope_id="account-3",
        currency="USDT",
        sample_barriers_ns=(10_000_000, 20_000_000, 30_000_000),
    )

    assert imported.query_time_range_ms == (10, 30)
    assert imported.page_sha256 == tuple(hashlib.sha256(page).hexdigest() for page in (page1, page2))
    assert imported.source_sha256 not in {
        hashlib.sha256(page1).hexdigest(), hashlib.sha256(page2).hexdigest(),
    }
    assert all(fact["source_sha256"] == imported.source_sha256 for fact in imported.facts)
    assert [fact["kind"] for fact in imported.facts] == ["DEPOSIT", "WITHDRAWAL"]
    assert [fact["source_event_time_ns"] for fact in imported.facts] == [15_000_000, 25_000_000]
    assert imported.source_id.endswith(imported.source_sha256)
    assert any("do not authenticate" in item for item in imported.limitations)


@pytest.mark.parametrize(
    ("pages", "query_start_ms", "query_end_ms", "message"),
    [
        ((BinanceIncomePage(11, 19, 2, b"[]"),), 10, 19, "contiguously"),
        ((BinanceIncomePage(10, 19, 2, b"[]"),), 10, 20, "do not reach"),
        ((BinanceIncomePage(10, 19, 2,
          b'[{"incomeType":"TRANSFER","income":"1","asset":"USDT","time":15,"tranId":1},'
          b'{"incomeType":"TRANSFER","income":"2","asset":"USDT","time":16,"tranId":2}]'),),
         10, 19, "may be truncated"),
        ((BinanceIncomePage(10, 19, 5,
          b'[{"incomeType":"TRANSFER","income":"1","asset":"USDT","time":20,"tranId":1}]'),),
         10, 19, "outside its declared page range"),
    ],
)
def test_binance_paged_import_rejects_incomplete_or_truncated_coverage(
    pages: tuple[BinanceIncomePage, ...],
    query_start_ms: int,
    query_end_ms: int,
    message: str,
) -> None:
    with pytest.raises(MetricsError, match=message):
        import_binance_usdm_transfer_income_pages(
            pages,
            query_start_ms=query_start_ms,
            query_end_ms=query_end_ms,
            account_scope_id="account",
            currency="USDT",
            sample_barriers_ns=(10, 20_000_000, 30_000_000),
        )


def test_binance_paged_import_rejects_duplicate_json_object_keys() -> None:
    page = BinanceIncomePage(
        10, 19, 2,
        b'[{"incomeType":"TRANSFER","income":"1","asset":"USDT",'
        b'"time":15,"time":16,"tranId":1}]',
    )
    with pytest.raises(MetricsError, match="duplicate object key"):
        import_binance_usdm_transfer_income_pages(
            (page,), query_start_ms=10, query_end_ms=19,
            account_scope_id="account", currency="USDT",
            sample_barriers_ns=(10, 20_000_000, 30_000_000),
        )
