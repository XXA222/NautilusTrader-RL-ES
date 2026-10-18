"""A real Nautilus streaming-engine fixture for liquidation-order identity only."""
from contextlib import redirect_stdout
from decimal import Decimal
import io

import pytest

from nautilus_trader.backtest import BacktestEngine
from nautilus_trader.config import BacktestEngineConfig
from nautilus_trader.execution import MakerTakerFeeModel
from nautilus_trader.model import (
    AccountId, AccountType, Currency, Money, OmsType, OrderSide, Price, Quantity,
    QuoteTick, TimeInForce,
)
from nautilus_trader.testkit.providers import TestInstrumentProvider
from nautilus_trader.trading import Strategy

from rles_nautilus_adapter.session import NativeFactsReader, _compact_position_record

pytestmark = pytest.mark.micro_golden


_INSTRUMENT = TestInstrumentProvider.btcusd_bybit()


class _OpenLiquidationExposure(Strategy):
    """Synthetic stress fixture using only the installed native public API."""

    def __init__(self):
        super().__init__()
        self.submitted = False

    def on_start(self):
        self.subscribe_quotes(_INSTRUMENT.id)

    def on_quote(self, quote):
        if not self.submitted:
            self.submitted = True
            self.submit_order(self.order_factory.market(
                instrument_id=_INSTRUMENT.id, order_side=OrderSide.BUY,
                quantity=Quantity.from_int(10_000_000), time_in_force=TimeInForce.GTC,
            ))


def _stress_quote(price: str, timestamp: int) -> QuoteTick:
    return QuoteTick(
        instrument_id=_INSTRUMENT.id, bid_price=Price.from_str(price),
        ask_price=Price.from_str(price), bid_size=Quantity.from_int(10_000_000),
        ask_size=Quantity.from_int(10_000_000), ts_event=timestamp, ts_init=timestamp,
    )


def test_native_reader_counts_streamed_liquidation_order_from_exchange_tags():
    btc = Currency.from_str("BTC")
    engine = BacktestEngine(config=BacktestEngineConfig(bypass_logging=True, run_analysis=False))
    engine.add_venue(
        venue=_INSTRUMENT.id.venue, oms_type=OmsType.NETTING, account_type=AccountType.MARGIN,
        base_currency=btc, starting_balances=[Money(1.0, btc)],
        fee_model=MakerTakerFeeModel(Decimal("-0.00025"), Decimal("0.00075")),
        liquidation_enabled=True, liquidation_trigger_ratio=1.0,
        liquidation_cancel_open_orders=True,
    )
    engine.add_instrument(_INSTRUMENT)
    engine.add_strategy(_OpenLiquidationExposure())
    try:
        with redirect_stdout(io.StringIO()):
            engine.add_data([_stress_quote("40000.0", 0)])
            engine.run(streaming=True)

        position = engine.cache.positions_open(
            instrument_id=_INSTRUMENT.id, account_id=AccountId.from_str("BYBIT-001"),
        )[0]
        full_position = position.to_dict()
        compact_position = _compact_position_record(position)
        historical_fields = {"events", "adjustments", "commissions", "venue_order_ids", "trade_ids"}
        unavailable_aggregate_fields = {"buy_qty", "sell_qty"}
        assert compact_position == {
            key: value for key, value in full_position.items()
            if key not in historical_fields | unavailable_aggregate_fields
        }
        assert not historical_fields.intersection(compact_position)

        compact_facts = NativeFactsReader(
            account_id=AccountId.from_str("BYBIT-001"), instrument_id=_INSTRUMENT.id,
            currency="BTC", compact_history=True,
        ).read(engine, 0)
        assert compact_facts.positions == (compact_position,)

        with redirect_stdout(io.StringIO()):
            engine.clear_data()
            engine.add_data([_stress_quote("20000.0", 1)])
            engine.run(streaming=True)

        facts = NativeFactsReader(
            account_id=AccountId.from_str("BYBIT-001"), instrument_id=_INSTRUMENT.id,
            currency="BTC", liquidation_capability=True,
        ).read(engine, 1)

        assert facts.liquidation_orders is not None
        assert len(facts.liquidation_orders) == 1
        order = facts.liquidation_orders[0]
        assert order["client_order_id"].startswith("LIQUIDATION-BYBIT-")
        assert "LIQUIDATION_BYBIT_CLOSE" in order["tags"]
        assert any(fill["client_order_id"] == order["client_order_id"] for fill in facts.fills)
    finally:
        engine.dispose()


def test_native_linear_perpetual_liquidation_reconciles_to_flat_equity():
    from types import SimpleNamespace

    from nautilus_trader.model import (
        CryptoPerpetual, InstrumentId, MarkPriceUpdate, StandardMarginModel, Symbol, Venue,
    )
    from rles_research.native_metrics import native_report, reconcile_linear_equity

    usdt = Currency.from_str("USDT")
    instrument = CryptoPerpetual(
        instrument_id=InstrumentId.from_str("ETHUSDT-PERP.SIM"),
        raw_symbol=Symbol("ETHUSDT"), base_currency=Currency.from_str("ETH"),
        quote_currency=usdt, settlement_currency=usdt, is_inverse=False,
        price_precision=2, size_precision=3, price_increment=Price.from_str("0.01"),
        size_increment=Quantity.from_str("0.001"), ts_event=1, ts_init=1,
        min_notional=Money.from_str("10 USDT"), margin_init=Decimal("0.05"),
        margin_maint=Decimal("0.04"),
    )

    class LargeLong(Strategy):
        def __init__(self):
            super().__init__()
            self.submitted = False
            self.quotes_seen = 0

        def on_start(self):
            self.subscribe_quotes(instrument.id)

        def on_quote(self, quote):
            self.quotes_seen += 1
            if not self.submitted and self.quotes_seen >= 2:
                self.submitted = True
                self.submit_order(self.order_factory.market(
                    instrument_id=instrument.id, order_side=OrderSide.BUY,
                    quantity=Quantity.from_str("49.000"), time_in_force=TimeInForce.GTC,
                ))

    def quote(price: str, timestamp: int) -> QuoteTick:
        return QuoteTick(
            instrument_id=instrument.id, bid_price=Price.from_str(price),
            ask_price=Price.from_str(price), bid_size=Quantity.from_str("100.000"),
            ask_size=Quantity.from_str("100.000"), ts_event=timestamp, ts_init=timestamp,
        )

    def mark(price: str, timestamp: int) -> MarkPriceUpdate:
        return MarkPriceUpdate(instrument.id, Price.from_str(price), timestamp, timestamp)

    engine = BacktestEngine(config=BacktestEngineConfig(bypass_logging=True, run_analysis=False))
    engine.add_venue(
        venue=Venue.from_str("SIM"), oms_type=OmsType.NETTING,
        account_type=AccountType.MARGIN, base_currency=usdt,
        starting_balances=[Money.from_str("10000 USDT")],
        margin_model=StandardMarginModel(),
        fee_model=MakerTakerFeeModel(Decimal("0.001"), Decimal("0.001")),
        liquidation_enabled=True, liquidation_trigger_ratio=1.0,
        liquidation_cancel_open_orders=True,
    )
    engine.add_instrument(instrument)
    engine.add_strategy(LargeLong())
    account_id = AccountId.from_str("SIM-001")
    reader = NativeFactsReader(account_id=account_id, instrument_id=instrument.id,
                               currency="USDT", liquidation_capability=True)
    try:
        with redirect_stdout(io.StringIO()):
            engine.add_data([quote("2000.00", 1)])
            engine.run(streaming=True)
        initial = reader.read(engine, 1)
        assert initial.equity == Decimal("10000")

        with redirect_stdout(io.StringIO()):
            engine.clear_data()
            engine.add_data([quote("2000.00", 2), mark("2000.00", 2)])
            engine.run(streaming=True)
        opened = reader.read(engine, 2)
        assert opened.positions
        assert opened.mark_price == Decimal("2000")

        with redirect_stdout(io.StringIO()):
            engine.clear_data()
            engine.add_data([quote("1800.00", 3), mark("1800.00", 3)])
            engine.run(streaming=True)
        liquidated = reader.read(engine, 3)
        assert liquidated.mark_price == Decimal("1800")
        assert liquidated.equity == Decimal("13.8")
        assert len(liquidated.fills) == 2
        assert liquidated.liquidation_orders is not None
        assert len(liquidated.liquidation_orders) == 1
        assert liquidated.liquidation_orders[0]["client_order_id"].startswith("LIQUIDATION-SIM-")
        assert not liquidated.positions
        repeated_liquidation = reader.read(engine, 4)

        def ledger_snapshot(facts):
            return SimpleNamespace(
                decision_ts_ns=facts.timestamp_ns, equity=facts.equity,
                currency=facts.currency, positions=facts.positions,
                orders=facts.orders, fills=facts.fills,
                funding=facts.funding_adjustments, mark=facts.mark_price,
                liquidation_capability=facts.liquidation_capability,
                liquidation_orders=facts.liquidation_orders,
            )

        snapshots = (initial, opened, liquidated, repeated_liquidation)
        report = native_report(
            tuple(map(ledger_snapshot, snapshots)),
            fill_references={fill["client_order_id"]: Decimal(fill["last_px"])
                             for fill in liquidated.fills},
            sample_interval_ns=1,
        )
        assert report["liquidations"] == 1
        assert report["liquidation_orders"] == list(repeated_liquidation.liquidation_orders)
        assert len(report["fills"]) == 2
        assert report["metrics"]["total_fee"] > 0

        report = reconcile_linear_equity(
            tuple(map(ledger_snapshot, (initial, opened, liquidated))),
            rounding_per_event=Decimal("0.00000001"),
        )
        assert report["terminal_checked"] is True
        assert report["checked_open_barriers"] >= 1
        assert report["unchecked_open_barriers"] == 0
        assert report["unique_fills"] == 2
        assert Decimal(report["max_abs_residual"]) == 0
    finally:
        engine.dispose()
