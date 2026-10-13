"""A real Nautilus streaming-engine fixture for liquidation-order identity only."""
from contextlib import redirect_stdout
from decimal import Decimal
import io
from pathlib import Path
import runpy

import pytest

from nautilus_trader.backtest import BacktestEngine
from nautilus_trader.config import BacktestEngineConfig
from nautilus_trader.execution import MakerTakerFeeModel
from nautilus_trader.model import AccountId, AccountType, Money, OmsType

from rles_nautilus_adapter.session import NativeFactsReader

pytestmark = pytest.mark.micro_golden


def test_native_reader_counts_streamed_liquidation_order_from_exchange_tags():
    demo_path = (Path(__file__).resolve().parents[1]
                 / "nautilus_trader" / "examples" / "backtest" / "liquidation_demo.py")
    demo = runpy.run_path(str(demo_path))
    engine = BacktestEngine(config=BacktestEngineConfig(bypass_logging=True, run_analysis=False))
    engine.add_venue(
        venue=demo["BYBIT"], oms_type=OmsType.NETTING, account_type=AccountType.MARGIN,
        base_currency=demo["BTC"], starting_balances=[Money(1.0, demo["BTC"])],
        fee_model=MakerTakerFeeModel(Decimal("-0.00025"), Decimal("0.00075")),
        liquidation_enabled=True, liquidation_trigger_ratio=1.0,
        liquidation_cancel_open_orders=True,
    )
    engine.add_instrument(demo["BTCUSD"])
    engine.add_strategy(demo["MarketBuyOnStart"](
        instrument_id=demo["BTCUSD"].id,
        trade_size=demo["Quantity"].from_int(10_000_000),
    ))
    try:
        with redirect_stdout(io.StringIO()):
            engine.add_data([demo["_make_quote"](40_000.0, ts=0)])
            engine.run(streaming=True)
            engine.clear_data()
            engine.add_data([demo["_make_quote"](20_000.0, ts=1)])
            engine.run(streaming=True)

        facts = NativeFactsReader(
            account_id=AccountId.from_str("BYBIT-001"), instrument_id=demo["BTCUSD"].id,
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
