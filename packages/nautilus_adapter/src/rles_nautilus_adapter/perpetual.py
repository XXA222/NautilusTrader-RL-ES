"""Production linear-perpetual bindings using official native extension APIs."""
from __future__ import annotations

from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR
from hashlib import sha256
from types import SimpleNamespace
from typing import Any, Callable, Mapping

from .bootstrap import NativeBootConfiguration, PublicBacktestBuilder
from .market_data import NativeMarketEventDecoder
from .session import EngineBoundaryError, EngineSession, NativeFactsReader
from .submitter import NativePlanSubmitter


class ConservativeBpsFillModel:
    """Provide adverse prices to Nautilus' fill simulation, never match locally.

    A bar cannot establish actual depth. This is an explicit conservative
    research model with sufficient simulated depth for the requested order;
    the native engine owns matching, fees, funding and account changes.
    """
    def __init__(self, slippage_bps: Decimal, record_reference: Callable | None = None,
                 simulated_depth_fraction: Decimal = Decimal("1")) -> None:
        if not slippage_bps.is_finite() or not 0 < slippage_bps < 10_000:
            raise ValueError("bar execution requires positive slippage below 10000 bps")
        if (not simulated_depth_fraction.is_finite()
                or not Decimal(0) <= simulated_depth_fraction <= Decimal(1)):
            raise ValueError("simulated depth fraction must be in [0, 1]")
        self.slippage = slippage_bps / 10_000
        self.simulated_depth_fraction = simulated_depth_fraction
        self._record_reference = record_reference

    def is_limit_filled(self) -> bool:
        return True

    def is_slipped(self) -> bool:
        return False  # The orderbook below already includes adverse slippage.

    def fill_limit_inside_spread(self) -> bool:
        return False

    def get_orderbook_for_fill_simulation(self, instrument, order, best_bid, best_ask):
        from nautilus_trader.model import BookOrder, BookType, OrderBook, OrderSide

        if best_bid is None or best_ask is None:
            raise EngineBoundaryError("bar fill model requires an observable native market price")
        tick = instrument.price_increment.as_decimal()
        bid = (best_bid.as_decimal() * (1 - self.slippage) / tick).to_integral_value(rounding=ROUND_FLOOR) * tick
        ask = (best_ask.as_decimal() * (1 + self.slippage) / tick).to_integral_value(rounding=ROUND_CEILING) * tick
        if bid <= 0 or ask < bid:
            raise EngineBoundaryError("invalid adverse fill prices")
        if self._record_reference is not None:
            reference = best_ask if order.side == OrderSide.BUY else best_bid
            self._record_reference(str(order.client_order_id), reference.as_decimal())
        book = OrderBook(instrument.id, BookType.L2_MBP)
        step = instrument.size_increment.as_decimal()
        depth_units = (order.quantity.as_decimal() * self.simulated_depth_fraction / step)
        depth = depth_units.to_integral_value(rounding=ROUND_FLOOR) * step
        if depth > 0:
            simulated_size = instrument.make_qty(depth)
            book.add(BookOrder(OrderSide.BUY, instrument.make_price(bid), simulated_size, 1), 0, 1, 0)
            book.add(BookOrder(OrderSide.SELL, instrument.make_price(ask), simulated_size, 2), 0, 2, 0)
        return book


QUOTE_MAX_AGE_NS = 2_000_000_000


def _effective_fee_rates(rule: Mapping[str, Any], execution: Any, mode: str) -> tuple[Decimal, Decimal]:
    """Resolve charged fee rates while retaining the source rule as an immutable fact."""
    multiplier = execution.fill(mode).fee_multiplier if mode == "stress" else Decimal("1")
    maker, taker = Decimal(rule["maker_fee"]), Decimal(rule["taker_fee"])
    # A cost shock can raise a charged fee but never expands an exchange rebate.
    return (maker * multiplier if maker > 0 else maker,
            taker * multiplier if taker > 0 else taker)


def make_perpetual_session(*, rule: Mapping[str, Any], execution: Any, mode: str,
                           starting_balance: Decimal, base_currency: str = "ETH",
                           quote_currency: str = "USDT",
                           record_snapshot: Callable[[Any], None] | None = None,
                           record_fill_reference: Callable | None = None,
                           compact_native_history: bool = False) -> EngineSession:
    """Fresh native models/engine for one episode; caller supplies verified rules."""
    from nautilus_trader.execution import DefaultFillModel, MakerTakerFeeModel, StaticLatencyModel
    from nautilus_trader.model import (
        AccountId, AccountType, BarType, CryptoPerpetual, Currency, InstrumentId,
        Money, OmsType, Price, Quantity, StandardMarginModel, Symbol,
    )
    from nautilus_trader.trading import Strategy

    price_tick, size_step = Decimal(rule["tick_size"]), Decimal(rule["step_size"])
    price_exponent, size_exponent = price_tick.as_tuple().exponent, size_step.as_tuple().exponent
    if not isinstance(price_exponent, int) or not isinstance(size_exponent, int):
        raise ValueError("finite decimal increments required")
    instrument_id = InstrumentId.from_str(rule["instrument"])
    quote = Currency.from_str(quote_currency)
    if str(instrument_id.venue) != "SIM" or starting_balance <= 0:
        raise ValueError("offline perpetual replay requires SIM venue and positive capital")
    instrument = CryptoPerpetual(
        instrument_id=instrument_id, raw_symbol=Symbol(base_currency + quote_currency),
        base_currency=Currency.from_str(base_currency), quote_currency=quote,
        settlement_currency=quote, is_inverse=False,
        price_precision=max(0, -price_exponent),
        size_precision=max(0, -size_exponent),
        price_increment=Price.from_str(str(price_tick)), size_increment=Quantity.from_str(str(size_step)),
        min_quantity=Quantity.from_str(str(rule["min_qty"])),
        min_notional=Money.from_str(f"{rule['min_notional']} {quote_currency}"),
        margin_init=Decimal(rule["initial_margin_rate"]), margin_maint=Decimal(rule["maintenance_margin_rate"]),
        ts_event=rule["effective_ns"], ts_init=rule["effective_ns"],
    )
    latency, fill = execution.latency(mode), execution.fill(mode)
    maker_fee, taker_fee = _effective_fee_rates(rule, execution, mode)
    quote_execution = fill.data_granularity == "quote"
    if fill.data_granularity not in {"bar", "quote"}:
        raise ValueError("perpetual factory requires explicit bar or quote execution")
    if quote_execution and fill.slippage_bps != 0:
        raise ValueError("quote execution cannot add synthetic BPS slippage to native displayed depth")
    submitter = NativePlanSubmitter(
        instrument_id=str(instrument_id), quantity_unit=base_currency, quantity_increment=size_step,
        execution_spec_hash=execution.fingerprint(), latency_model_id=latency.model_id,
    )

    def prepare_quote_reference(engine: Any, plan: Any, decision_ts_ns: int) -> None:
        if not quote_execution or plan.action != "SUBMIT":
            return
        quote_tick = engine.cache.quote(instrument_id)
        if quote_tick is None or quote_tick.ts_init > decision_ts_ns:
            raise EngineBoundaryError("quote execution requires an as-of native quote at the decision barrier")
        if decision_ts_ns - quote_tick.ts_init > QUOTE_MAX_AGE_NS:
            raise EngineBoundaryError("native best quote is stale at order submission")
        bid, bid_size = quote_tick.bid_price, quote_tick.bid_size
        ask, ask_size = quote_tick.ask_price, quote_tick.ask_size
        reference, size = (ask, ask_size) if plan.delta_qty > 0 else (bid, bid_size)
        if size.as_decimal() <= 0:
            raise EngineBoundaryError("native best quote has no displayed liquidity")
        if record_fill_reference is not None:
            client_id = "RLES-" + sha256(plan.idempotency_key.encode("utf-8")).hexdigest()
            record_fill_reference(client_id, reference.as_decimal())

    builder = PublicBacktestBuilder(plan_submitter=submitter, config=NativeBootConfiguration(
        venue_kwargs=({
            "venue": instrument_id.venue, "oms_type": OmsType.NETTING, "account_type": AccountType.MARGIN,
            "starting_balances": [Money.from_str(f"{starting_balance} {quote_currency}")], "base_currency": quote,
            "margin_model": StandardMarginModel(),
            "fee_model": MakerTakerFeeModel(maker_fee, taker_fee),
            "fill_model": (DefaultFillModel(prob_fill_on_limit=1.0, prob_slippage=0.0, random_seed=0)
                           if quote_execution else
                           ConservativeBpsFillModel(fill.slippage_bps, record_fill_reference,
                                                    fill.simulated_depth_fraction)),
            "latency_model": StaticLatencyModel(base_latency_nanos=latency.decision_ns,
                                                insert_latency_nanos=latency.submit_ns + latency.venue_ns,
                                                update_latency_nanos=latency.submit_ns + latency.venue_ns,
                                                cancel_latency_nanos=latency.submit_ns + latency.venue_ns),
            "bar_execution": not quote_execution, "liquidity_consumption": quote_execution,
            "liquidation_enabled": True,
        },), instrument_builders=(lambda: instrument,), strategy_builder=Strategy,
    ))

    def snapshot_factory(**facts):
        snapshot = SimpleNamespace(**facts)
        if record_snapshot is not None:
            record_snapshot(snapshot)
        return snapshot

    return EngineSession(
        build_native=builder, submit_plan=submitter,
        decode_event=NativeMarketEventDecoder(instrument=instrument,
                                             bar_type=BarType.from_str(f"{instrument.id}-1-MINUTE-LAST-EXTERNAL")),
        facts_reader=NativeFactsReader(account_id=AccountId.from_str("SIM-001"),
                                       instrument_id=instrument.id, currency=quote_currency,
                                       liquidation_capability=True,
                                       compact_history=compact_native_history),
        contract_snapshot_factory=snapshot_factory,
        certified_barrier_protocol=EngineSession.BARRIER_PROTOCOL,
        prepare_submit=prepare_quote_reference if quote_execution else None,
    )
