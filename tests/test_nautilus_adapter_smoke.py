from __future__ import annotations

import re
from decimal import Decimal
from types import SimpleNamespace

import pytest

from nautilus_trader.execution import DefaultFillModel, MakerTakerFeeModel, StaticLatencyModel
from nautilus_trader.model import (
    AccountType,
    AccountId,
    Bar,
    BarType,
    Currency,
    CurrencyPair,
    InstrumentId,
    Money,
    OmsType,
    Price,
    Quantity,
    StandardMarginModel,
    Symbol,
    Venue,
)
from nautilus_trader.trading import Strategy, StrategyConfig

from rles_nautilus_adapter.bootstrap import NativeBootConfiguration, PublicBacktestBuilder
from rles_nautilus_adapter.session import EngineBoundaryError, EngineSession, NativeFactsReader
from rles_contracts.dto import DataQualityGrade, ExecutionPlan, MarketEvent, OrderType, PendingOrderPolicy, TimeInForce
from rles_contracts.frozen import FrozenFields
from rles_nautilus_adapter import NativeMarketEventDecoder, NativePlanSubmitter, VisibleBatch

pytestmark = pytest.mark.micro_golden


def _instrument() -> CurrencyPair:
    return CurrencyPair(
        instrument_id=InstrumentId(Symbol("BTC/USD"), Venue("SIM")),
        raw_symbol=Symbol("BTC/USD"),
        base_currency=Currency.from_str("BTC"),
        quote_currency=Currency.from_str("USD"),
        price_precision=2,
        size_precision=4,
        price_increment=Price(0.01, 2),
        size_increment=Quantity(0.0001, 4),
        ts_event=1,
        ts_init=1,
    )


class _NoopStrategy(Strategy):
    def __init__(self) -> None:
        super().__init__(StrategyConfig())


def _builder(submitter: NativePlanSubmitter | None = None) -> PublicBacktestBuilder:
    return PublicBacktestBuilder(
        plan_submitter=submitter,
        config=NativeBootConfiguration(
            venue_kwargs=(
                {
                    "venue": Venue("SIM"),
                    "oms_type": OmsType.NETTING,
                    "account_type": AccountType.MARGIN,
                    "starting_balances": [Money(100_000, Currency.from_str("USD"))],
                    "base_currency": Currency.from_str("USD"),
                    "margin_model": StandardMarginModel(),
                    "fee_model": MakerTakerFeeModel(Decimal("0.001"), Decimal("0.001")),
                    "fill_model": DefaultFillModel(),
                    "latency_model": StaticLatencyModel(),
                    "bar_execution": True,
                    "liquidation_enabled": False,
                },
            ),
            instrument_builders=(_instrument,),
            strategy_builder=_NoopStrategy,
        )
    )


def test_engine_session_opens_and_disposes_a_real_pinned_backtest_engine() -> None:
    instrument = _instrument()
    builder = _builder()
    session = EngineSession(
        build_native=builder,
        submit_plan=lambda _engine, _plan: None,
        decode_event=lambda event: event,
        facts_reader=NativeFactsReader(
            account_id="SIM-ACCOUNT",
            instrument_id=instrument.id,
            currency="USD",
        ),
        contract_snapshot_factory=lambda **_facts: None,
        certified_barrier_protocol=EngineSession.BARRIER_PROTOCOL,
    )

    fingerprint = session.open(SimpleNamespace(
        dataset_hash="a" * 64,
        fold_hash="b" * 64,
        config_hash="c" * 64,
    ))
    try:
        assert session.state == "READY"
        assert re.fullmatch(r"[0-9a-f]{64}", fingerprint)
        assert session.fingerprint == fingerprint
    finally:
        session.close()
    assert session.state == "CLOSED"


def test_native_builder_refuses_cross_fold_execution_model_reuse() -> None:
    builder = _builder()
    engine = builder(SimpleNamespace())
    engine.dispose()
    with pytest.raises(EngineBoundaryError, match="models cannot be reused"):
        builder(SimpleNamespace())


def test_real_rc6_engine_streams_bars_and_reads_native_account_facts(capfd) -> None:
    instrument = _instrument()
    bar_type = BarType.from_str(f"{instrument.id}-1-MINUTE-LAST-EXTERNAL")

    def event(number: int) -> SimpleNamespace:
        timestamp = number * 60_000_000_000
        bar = Bar(
            bar_type=bar_type,
            open=instrument.make_price(100.0 + number),
            high=instrument.make_price(101.0 + number),
            low=instrument.make_price(99.0 + number),
            close=instrument.make_price(100.5 + number),
            volume=instrument.make_qty(100),
            ts_event=timestamp,
            ts_init=timestamp,
        )
        return SimpleNamespace(
            event_id=f"native-bar-{number}", source_hash="a" * 64,
            ts_event_ns=timestamp, ts_init_ns=timestamp,
            available_at_ns=timestamp, native_bar=bar,
        )

    from rles_nautilus_adapter.session import VisibleBatch

    session = EngineSession(
        build_native=_builder(),
        submit_plan=lambda _engine, _plan: pytest.fail("NOOP must not submit a native order"),
        decode_event=lambda item: item.native_bar,
        facts_reader=NativeFactsReader(
            account_id=AccountId.from_str("SIM-001"),
            instrument_id=instrument.id,
            currency="USD",
        ),
        contract_snapshot_factory=lambda **facts: SimpleNamespace(**facts),
        certified_barrier_protocol=EngineSession.BARRIER_PROTOCOL,
    )
    session.open(SimpleNamespace(dataset_hash="a" * 64, fold_hash="b" * 64,
                                 config_hash="c" * 64))
    first = session.bootstrap((VisibleBatch(60_000_000_000, (event(1),), True),),
                              decision_ts_ns=60_000_000_000)
    assert first.equity == Decimal("100000.00")
    assert first.currency == "USD"
    assert first.positions == ()
    with pytest.raises(EngineBoundaryError, match="native equity currency differs"):
        NativeFactsReader(account_id=AccountId.from_str("SIM-001"),
                          instrument_id=instrument.id, currency="EUR").read(
                              session._engine, 60_000_000_000)
    plan = ExecutionPlan(
        decision_id="decision-1", plan_id="noop-1", instrument_id=str(instrument.id),
        quantity_unit="BTC", quantity_increment=Decimal("0.0001"),
        latency_model_id="latency-1", execution_spec_hash="d" * 64,
        order_type=OrderType.MARKET_LIKE, tif=TimeInForce.GTC,
        cancel_policy=PendingOrderPolicy.NONE,
        action="NOOP", reason="native account already flat",
        idempotency_key="noop-1", delta_qty=Decimal(0),
    )
    second = session.advance(plan,
                             batches=(VisibleBatch(120_000_000_000, (event(2),), True),),
                             next_decision_ts_ns=120_000_000_000)
    assert second.equity == first.equity
    assert second.decision_ts_ns == 120_000_000_000
    session.close()
    assert session.state == "CLOSED"
    assert "[INFO]" not in capfd.readouterr().out


def test_engine_session_rejects_unhashed_transition_identity_before_build() -> None:
    session = EngineSession(
        build_native=lambda _episode: pytest.fail("invalid identity must not build native engine"),
        submit_plan=lambda _engine, _plan: None,
        decode_event=lambda event: event,
        facts_reader=NativeFactsReader(
            account_id="SIM-ACCOUNT",
            instrument_id=_instrument().id,
            currency="USD",
        ),
        contract_snapshot_factory=lambda **_facts: None,
        certified_barrier_protocol=EngineSession.BARRIER_PROTOCOL,
    )
    with pytest.raises(EngineBoundaryError, match="lowercase SHA-256"):
        session.open(SimpleNamespace(
            dataset_hash="dataset-fixture",
            fold_hash="b" * 64,
            config_hash="c" * 64,
        ))
    assert session.state == "CLOSED"


def _native_order_session(submitter: NativePlanSubmitter) -> EngineSession:
    session = EngineSession(
        build_native=_builder(submitter), submit_plan=submitter,
        decode_event=lambda item: item.native_bar,
        facts_reader=NativeFactsReader(account_id=AccountId.from_str("SIM-001"),
                                       instrument_id=_instrument().id, currency="USD"),
        contract_snapshot_factory=lambda **facts: SimpleNamespace(**facts),
        certified_barrier_protocol=EngineSession.BARRIER_PROTOCOL,
    )
    session.open(SimpleNamespace(dataset_hash="a" * 64, fold_hash="b" * 64,
                                 config_hash="c" * 64))
    return session


def _order_batch(number: int) -> VisibleBatch:
    instrument = _instrument()
    timestamp = number * 60_000_000_000
    bar = Bar(
        bar_type=BarType.from_str(f"{instrument.id}-1-MINUTE-LAST-EXTERNAL"),
        open=instrument.make_price(100), high=instrument.make_price(100),
        low=instrument.make_price(100), close=instrument.make_price(100),
        volume=instrument.make_qty(100), ts_event=timestamp, ts_init=timestamp,
    )
    return VisibleBatch(timestamp, (SimpleNamespace(
        event_id=f"order-bar-{number}", source_hash="a" * 64,
        ts_event_ns=timestamp, ts_init_ns=timestamp, available_at_ns=timestamp,
        native_bar=bar),), True)


def _submitter() -> NativePlanSubmitter:
    return NativePlanSubmitter(instrument_id=str(_instrument().id), quantity_unit="BTC",
                               quantity_increment=Decimal("0.0001"),
                               execution_spec_hash="d" * 64, latency_model_id="latency-1")


def _order_plan(key: str, quantity: str) -> ExecutionPlan:
    return ExecutionPlan(
        decision_id=key, plan_id=key, instrument_id=str(_instrument().id),
        quantity_unit="BTC", quantity_increment=Decimal("0.0001"),
        latency_model_id="latency-1", execution_spec_hash="d" * 64,
        order_type=OrderType.MARKET_LIKE, tif=TimeInForce.GTC,
        cancel_policy=PendingOrderPolicy.NONE, action="SUBMIT", reason="native order test",
        idempotency_key=key, delta_qty=Decimal(quantity),
    )


def _perpetual_instrument():
    from nautilus_trader.model import CryptoPerpetual

    usdt = Currency.from_str("USDT")
    return CryptoPerpetual(
        instrument_id=InstrumentId.from_str("ETHUSDT-PERP.SIM"),
        raw_symbol=Symbol("ETHUSDT"), base_currency=Currency.from_str("ETH"),
        quote_currency=usdt, settlement_currency=usdt, is_inverse=False,
        price_precision=2, size_precision=3, price_increment=Price.from_str("0.01"),
        size_increment=Quantity.from_str("0.001"), ts_event=1, ts_init=1,
        min_notional=Money.from_str("10 USDT"), margin_init=Decimal("1.0"),
        margin_maint=Decimal("0.35"),
    )


def _perpetual_builder(submitter: NativePlanSubmitter, *, bar_execution: bool = True) -> PublicBacktestBuilder:
    instrument = _perpetual_instrument()
    usdt = Currency.from_str("USDT")
    return PublicBacktestBuilder(
        plan_submitter=submitter,
        config=NativeBootConfiguration(
            venue_kwargs=({
                "venue": Venue("SIM"), "oms_type": OmsType.NETTING,
                "account_type": AccountType.MARGIN,
                "starting_balances": [Money(100_000, usdt)], "base_currency": usdt,
                "margin_model": StandardMarginModel(),
                "fee_model": MakerTakerFeeModel(Decimal("0.001"), Decimal("0.001")),
                "fill_model": DefaultFillModel(), "latency_model": StaticLatencyModel(),
                "bar_execution": bar_execution, "liquidity_consumption": not bar_execution,
                "liquidation_enabled": False,
            },),
            instrument_builders=(lambda: instrument,), strategy_builder=_NoopStrategy,
        ),
    )


def _perpetual_batch(number: int, *extra_events: tuple[str, dict]) -> VisibleBatch:
    instrument = _perpetual_instrument()
    timestamp = number * 60_000_000_000
    payloads = [("ohlcv_1m", {
        "instrument": str(instrument.id), "open": "1000.00", "high": "1000.00",
        "low": "1000.00", "close": "1000.00", "volume": "100.000",
    }), *extra_events]
    events = tuple(MarketEvent(
        event_id=f"perpetual-{number}-{index}", ts_event_ns=timestamp,
        ts_init_ns=timestamp, available_at_ns=timestamp,
        source_hash=f"{index + 1:064x}", quality_grade=DataQualityGrade.B,
        event_type=kind, payload=FrozenFields.from_mapping(payload),
    ) for index, (kind, payload) in enumerate(payloads))
    return VisibleBatch(timestamp, events, True)


def test_native_session_rejects_duplicate_funding_settlement_before_decoding() -> None:
    instrument = _perpetual_instrument()
    decoded: list[str] = []

    def decode(event: MarketEvent) -> SimpleNamespace:
        decoded.append(event.event_id)
        return SimpleNamespace(ts_init=event.ts_init_ns)

    session = EngineSession(
        build_native=lambda _episode: None,
        submit_plan=lambda _engine, _plan: None,
        decode_event=decode,
        facts_reader=NativeFactsReader(
            account_id=AccountId.from_str("SIM-001"), instrument_id=instrument.id,
            currency="USDT",
        ),
        contract_snapshot_factory=lambda **facts: SimpleNamespace(**facts),
        certified_barrier_protocol=EngineSession.BARRIER_PROTOCOL,
    )
    session._engine = object()
    settlement = {
        "instrument": str(instrument.id), "rate": "0.001",
        "settlement_ns": 120_000_000_000,
    }
    batch = _perpetual_batch(
        2, ("funding_settlement", settlement), ("funding_settlement", settlement),
    )

    with pytest.raises(EngineBoundaryError, match="duplicate native funding settlement boundary"):
        session._preflight((batch,), end_ts_ns=batch.timestamp_ns)

    assert decoded == []

    class _FakeEngine:
        def add_data(self, _events: list[SimpleNamespace]) -> None:
            pass

        def run(self, *, streaming: bool) -> None:
            assert streaming is True

        def clear_data(self) -> None:
            pass

    session._engine = _FakeEngine()
    first_settlement = _perpetual_batch(2, ("funding_settlement", settlement))
    prepared = session._preflight((first_settlement,), end_ts_ns=first_settlement.timestamp_ns)
    session._replay(prepared)
    decoded_after_first = len(decoded)
    repeated_settlement = _perpetual_batch(3, ("funding_settlement", settlement))
    with pytest.raises(EngineBoundaryError, match="duplicate native funding settlement boundary"):
        session._preflight((repeated_settlement,), end_ts_ns=repeated_settlement.timestamp_ns)
    assert len(decoded) == decoded_after_first


def _quote_payload(bid: str, bid_qty: str, ask: str, ask_qty: str) -> tuple[str, dict]:
    return "best_quote", {"instrument": "ETHUSDT-PERP.SIM", "bid_price": bid,
                           "bid_qty": bid_qty, "ask_price": ask, "ask_qty": ask_qty,
                           "update_id": 1, "transaction_time_ns": 60_000_000_000}


def _perpetual_plan(key: str, quantity: str) -> ExecutionPlan:
    instrument = _perpetual_instrument()
    return ExecutionPlan(
        decision_id=key, plan_id=key, instrument_id=str(instrument.id),
        quantity_unit="ETH", quantity_increment=Decimal("0.001"),
        latency_model_id="latency-1", execution_spec_hash="e" * 64,
        order_type=OrderType.MARKET_LIKE, tif=TimeInForce.GTC,
        cancel_policy=PendingOrderPolicy.NONE, action="SUBMIT", reason="funding golden",
        idempotency_key=key, delta_qty=Decimal(quantity),
    )


@pytest.mark.parametrize("entry_qty,exit_qty", [("1.0000", "-1.0000"), ("-1.0000", "1.0000")])
def test_native_submitter_executes_market_fees_and_terminal_flatten(entry_qty: str, exit_qty: str) -> None:
    session = _native_order_session(_submitter())
    try:
        initial = session.bootstrap((_order_batch(1),), decision_ts_ns=60_000_000_000)
        entered = session.advance(_order_plan("entry", entry_qty), batches=(_order_batch(2),),
                                  next_decision_ts_ns=120_000_000_000)
        assert len(entered.positions) == 1
        assert entered.orders == ()
        assert len(entered.fills) == 1
        assert entered.equity == initial.equity - Decimal("0.10")
        terminal = session.drain_and_close(
            "force_flatten_with_cost", flatten_plan=_order_plan("exit", exit_qty),
            drain_batches=(_order_batch(3),), terminal_ts_ns=180_000_000_000,
        )
        assert terminal.positions == terminal.orders == ()
        assert len(terminal.fills) == 2
        assert len(terminal.all_orders) == 2
        assert sum(order["is_reduce_only"] for order in terminal.all_orders) == 1
        assert terminal.equity == initial.equity - Decimal("0.20")
        assert session.state == "CLOSED"
    finally:
        session.close()


@pytest.mark.parametrize("sign", [1, -1])
def test_native_hard_risk_partial_reduction_is_reduce_only(sign: int) -> None:
    from dataclasses import replace

    session = _native_order_session(_submitter())
    try:
        initial = session.bootstrap((_order_batch(1),), decision_ts_ns=60_000_000_000)
        session.advance(_order_plan("entry", str(sign)), batches=(_order_batch(2),),
                        next_decision_ts_ns=120_000_000_000)
        reduced = session.advance(
            replace(_order_plan("hard-reduce", str(Decimal("-0.5") * sign)), reason="hard_risk_reduce"),
            batches=(_order_batch(3),), next_decision_ts_ns=180_000_000_000,
        )
        assert len(reduced.positions) == 1
        assert Decimal(reduced.positions[0]["signed_qty"]) == Decimal("0.5") * sign
        assert len(reduced.fills) == 2
        assert reduced.equity == initial.equity - Decimal("0.15")
        assert sum(order["is_reduce_only"] for order in reduced.all_orders) == 1
        terminal = session.drain_and_close(
            "force_flatten_with_cost", flatten_plan=_order_plan("exit", str(Decimal("-0.5") * sign)),
            drain_batches=(_order_batch(4),), terminal_ts_ns=240_000_000_000,
        )
        assert terminal.positions == terminal.orders == ()
        assert terminal.equity == initial.equity - Decimal("0.20")
    finally:
        session.close()


@pytest.mark.parametrize("quantity", ["1", "-2"])
def test_native_hard_risk_rejects_expansion_and_reversal_before_submission(quantity: str) -> None:
    from dataclasses import replace

    submitter = _submitter()
    session = _native_order_session(submitter)
    try:
        session.bootstrap((_order_batch(1),), decision_ts_ns=60_000_000_000)
        session.advance(_order_plan("entry", "1"), batches=(_order_batch(2),),
                        next_decision_ts_ns=120_000_000_000)
        with pytest.raises(EngineBoundaryError, match="must reduce the owned"):
            submitter(session._engine, replace(_order_plan("bad-reduce", quantity), reason="hard_risk_reduce"))
        assert len(session.snapshot().all_orders) == 1
    finally:
        session.close()


@pytest.mark.parametrize("sign", [-1, 1])
def test_native_voluntary_flat_is_a_full_reduce_only_exit(sign):
    from dataclasses import replace
    submitter = _submitter()
    session = _native_order_session(submitter)
    try:
        initial = session.bootstrap((_order_batch(1),), decision_ts_ns=60_000_000_000)
        session.advance(_order_plan("entry", str(sign)), batches=(_order_batch(2),),
                        next_decision_ts_ns=120_000_000_000)
        with pytest.raises(EngineBoundaryError, match="quantity differs from native position"):
            submitter(session._engine, replace(_order_plan("bad-flat", str(-0.5 * sign)), reason="policy_flat"))
        flat = session.advance(replace(_order_plan("policy-exit", str(-sign)), reason="policy_flat"),
                               batches=(_order_batch(3),), next_decision_ts_ns=180_000_000_000)
        assert flat.positions == ()
        assert flat.equity == initial.equity - Decimal("0.20")
        assert sum(order["is_reduce_only"] for order in flat.all_orders) == 1
    finally:
        session.close()


@pytest.mark.parametrize("quantity", ["-2.0000", "-0.5000", "1.0000"])
def test_native_terminal_rejects_wrong_close_quantity_before_submitting(quantity: str) -> None:
    submitter = _submitter()
    session = _native_order_session(submitter)
    try:
        session.bootstrap((_order_batch(1),), decision_ts_ns=60_000_000_000)
        session.advance(_order_plan("entry", "1.0000"), batches=(_order_batch(2),),
                        next_decision_ts_ns=120_000_000_000)
        with pytest.raises(EngineBoundaryError, match="quantity differs from native position"):
            submitter.submit_flatten(session._engine, _order_plan("wrong-exit", quantity))
        assert len(session.snapshot().all_orders) == 1
    finally:
        session.close()


@pytest.mark.parametrize("field,value", [
    ("execution_spec_hash", "e" * 64), ("latency_model_id", "other-latency"),
    ("quantity_unit", "ETH"), ("quantity_increment", Decimal("0.001")),
    ("instrument_id", "ETH/USD.SIM"),
])
def test_native_submitter_rejects_contract_drift_before_order(field: str, value: object) -> None:
    from dataclasses import replace

    session = _native_order_session(_submitter())
    try:
        session.bootstrap((_order_batch(1),), decision_ts_ns=60_000_000_000)
        with pytest.raises(EngineBoundaryError, match="frozen native execution contract"):
            session.advance(replace(_order_plan("drift", "1.0000"), **{field: value}),
                            batches=(_order_batch(2),), next_decision_ts_ns=120_000_000_000)
        assert session.state == "CLOSED"
    finally:
        session.close()


@pytest.mark.parametrize("terminal", [False, True])
def test_native_submitter_cancels_live_strategy_order_and_observes_ack(terminal: bool) -> None:
    from dataclasses import replace
    from nautilus_trader.model import ClientOrderId, OrderSide

    submitter = _submitter()
    session = _native_order_session(submitter)
    try:
        session.bootstrap((_order_batch(1),), decision_ts_ns=60_000_000_000)
        # Native limit order creates a deterministic pending command without
        # emulating an exchange or introducing another production order type.
        order = submitter._strategy.order_factory.limit(
            instrument_id=_instrument().id, order_side=OrderSide.BUY,
            quantity=Quantity.from_str("1.0000"), price=Price.from_str("50.00"),
            client_order_id=ClientOrderId("test-pending-native"),
        )
        submitter._strategy.submit_order(order)
        wait = replace(_order_plan("wait-pending", "1.0000"), action="WAIT",
                       delta_qty=Decimal(0), cancel_policy=PendingOrderPolicy.WAIT)
        pending = session.advance(wait, batches=(_order_batch(2),),
                                  next_decision_ts_ns=120_000_000_000)
        assert len(pending.orders) == 1
        cancel = replace(_order_plan("cancel-pending", "1.0000"), action="CANCEL",
                         delta_qty=Decimal(0), cancel_policy=PendingOrderPolicy.CANCEL_REPLACE,
                         cancel_order_ids=("test-pending-native",), requires_cancel_ack=True)
        if terminal:
            flat = replace(_order_plan("already-flat", "1.0000"), action="NOOP", delta_qty=Decimal(0))
            acknowledged = session.drain_and_close(
                "force_flatten_with_cost", cancel_plan=cancel,
                cancel_batches=(_order_batch(3),), cancel_ts_ns=180_000_000_000,
                flatten_plan=flat, drain_batches=(_order_batch(4),), terminal_ts_ns=240_000_000_000,
            )
            assert session.state == "CLOSED"
        else:
            acknowledged = session.advance(cancel, batches=(_order_batch(3),),
                                           next_decision_ts_ns=180_000_000_000)
        assert acknowledged.orders == ()
        assert acknowledged.positions == acknowledged.fills == ()
        assert acknowledged.all_orders[0]["status"] == "CANCELED"
        assert acknowledged.equity == Decimal("100000.00")
    finally:
        session.close()


def test_native_submitter_rejects_reused_action_and_cross_episode_binding() -> None:
    submitter = _submitter()
    session = _native_order_session(submitter)
    try:
        session.bootstrap((_order_batch(1),), decision_ts_ns=60_000_000_000)
        plan = _order_plan("once", "1.0000")
        session.advance(plan, batches=(_order_batch(2),), next_decision_ts_ns=120_000_000_000)
        with pytest.raises(EngineBoundaryError, match="already used"):
            submitter(session._engine, plan)
        assert len(session.snapshot().all_orders) == 1
        with pytest.raises(EngineBoundaryError, match="cannot be rebound"):
            submitter.bind(session._engine, submitter._strategy)
    finally:
        session.close()


def test_native_fills_retain_closed_roundtrip_after_reopening_position() -> None:
    session = _native_order_session(_submitter())
    try:
        session.bootstrap((_order_batch(1),), decision_ts_ns=60_000_000_000)
        for number, quantity in [(2, "1.0000"), (3, "-1.0000"), (4, "1.0000")]:
            snapshot = session.advance(_order_plan(f"roundtrip-{number}", quantity),
                                       batches=(_order_batch(number),),
                                       next_decision_ts_ns=number * 60_000_000_000)
            assert len(snapshot.fills) == number - 1
        terminal = session.drain_and_close(
            "force_flatten_with_cost", flatten_plan=_order_plan("roundtrip-exit", "-1.0000"),
            drain_batches=(_order_batch(5),), terminal_ts_ns=300_000_000_000,
        )
        assert len(terminal.fills) == len(terminal.all_orders) == 4
        assert terminal.equity == Decimal("99999.60")
    finally:
        session.close()


@pytest.mark.parametrize(
    ("quantity", "funding_change", "equity"),
    [("1.000", Decimal("-1.00"), Decimal("99998.00")),
     ("-1.000", Decimal("1.00"), Decimal("100000.00"))],
)
@pytest.mark.parametrize("compact_history", [False, True])
def test_native_perpetual_funding_rate_settles_once_in_account(
    quantity: str, funding_change: Decimal, equity: Decimal, compact_history: bool,
) -> None:
    from dataclasses import replace
    from nautilus_trader.model import BarType

    instrument = _perpetual_instrument()
    submitter = NativePlanSubmitter(
        instrument_id=str(instrument.id), quantity_unit="ETH",
        quantity_increment=Decimal("0.001"), execution_spec_hash="e" * 64,
        latency_model_id="latency-1",
    )
    bar_type = BarType.from_str(f"{instrument.id}-1-MINUTE-LAST-EXTERNAL")
    decoder = NativeMarketEventDecoder(instrument=instrument, bar_type=bar_type)
    session = EngineSession(
        build_native=_perpetual_builder(submitter), submit_plan=submitter,
        decode_event=decoder,
        facts_reader=NativeFactsReader(
            account_id=AccountId.from_str("SIM-001"), instrument_id=instrument.id,
            currency="USDT", compact_history=compact_history,
        ),
        contract_snapshot_factory=lambda **facts: SimpleNamespace(**facts),
        certified_barrier_protocol=EngineSession.BARRIER_PROTOCOL,
    )
    session.open(SimpleNamespace(dataset_hash="a" * 64, fold_hash="b" * 64,
                                 config_hash="c" * 64))
    try:
        initial = session.bootstrap((_perpetual_batch(1),), decision_ts_ns=60_000_000_000)
        entered = session.advance(
            _perpetual_plan("funding-entry", quantity),
            batches=(_perpetual_batch(2),), next_decision_ts_ns=120_000_000_000,
        )
        assert len(entered.positions) == len(entered.fills) == 1
        funding_event_ns = 180_000_000_000
        settlement_ns = 240_000_000_000
        mark = ("mark", {"instrument": str(instrument.id), "price": "1000.00"})
        rate = ("funding_update", {"instrument": str(instrument.id), "rate": "0.001",
                                    "interval_minutes": 480,
                                    "next_funding_ns": settlement_ns})
        wait = replace(_perpetual_plan("funding-observe", "1.000"),
                       action="NOOP", delta_qty=Decimal(0))
        session.advance(
            wait, batches=(_perpetual_batch(3, mark, rate),),
            next_decision_ts_ns=funding_event_ns,
        )
        settlement_event = ("funding_settlement", {
            "instrument": str(instrument.id), "rate": "0.001",
            "interval_minutes": 480, "settlement_ns": settlement_ns,
        })
        funding_boundary = session.advance(
            replace(wait, decision_id="funding-boundary", plan_id="funding-boundary",
                    idempotency_key="funding-boundary"),
            batches=(_perpetual_batch(4, settlement_event),),
            next_decision_ts_ns=settlement_ns,
        )
        settled = session.advance(
            replace(wait, decision_id="funding-settled", plan_id="funding-settled",
                    idempotency_key="funding-settled"),
            batches=(_perpetual_batch(5),), next_decision_ts_ns=300_000_000_000,
        )
        assert settled.mark == Decimal("1000.00")
        assert len(funding_boundary.funding) == 1
        adjustment = funding_boundary.funding[0]
        assert Money.from_str(adjustment["pnl_change"]).as_decimal() == funding_change
        assert settled.equity == initial.equity - Decimal("1.00") + funding_change
        assert settled.equity == equity
        assert adjustment["ts_event"] == settlement_ns
        assert settled.margins == entered.margins
        if compact_history:
            assert settled.funding == ()
            assert session.snapshot().funding == ()
        else:
            assert len(settled.funding) == 1
    finally:
        session.close()


def test_compact_funding_reader_skips_unchanged_positions_and_emits_only_new_facts() -> None:
    reader = NativeFactsReader(account_id="SIM-001", instrument_id="ETHUSDT-PERP.SIM",
                               currency="USDT", compact_history=True)
    first_record = {"event_id": "funding-1", "ts_init": 10, "ts_event": 10}
    second_record = {"event_id": "funding-2", "ts_init": 20, "ts_event": 20}

    class Adjustment:
        adjustment_type = "FUNDING"

        def __init__(self, record):
            self.record = record

        def to_dict(self):
            return self.record

    class Position:
        id = "position-1"
        instrument_id = "ETHUSDT-PERP.SIM"
        ts_opened = 1

        def __init__(self, timestamp, adjustments):
            self.ts_last = timestamp
            self.records = list(adjustments)
            self.adjustment_reads = 0

        def adjustments(self):
            self.adjustment_reads += 1
            return tuple(Adjustment(record) for record in self.records)

    archived = Position(10, (first_record,))
    live = Position(10, (first_record,))
    cache = SimpleNamespace(position_snapshots=lambda **_kwargs: (archived,))

    first = reader._compact_funding_updates((live,), cache, "FUNDING")
    unchanged = reader._compact_funding_updates((live,), cache, "FUNDING")
    assert first == (first_record,)
    assert unchanged == ()
    assert live.adjustment_reads == 1
    assert archived.adjustment_reads == 0

    live.ts_last = 20
    live.records.append(second_record)
    next_update = reader._compact_funding_updates((live,), cache, "FUNDING")
    assert next_update == (second_record,)
    assert live.adjustment_reads == 2


def test_native_quote_execution_uses_displayed_book_and_records_submission_reference() -> None:
    from rles_execution import FillProfile, LatencyProfile, ResolvedExecutionConfig
    from rles_nautilus_adapter.perpetual import make_perpetual_session

    latency = LatencyProfile("quote-base", 0, 0, 0)
    stress_latency = LatencyProfile("quote-stress", 0, 0, 1_000_000)
    execution = ResolvedExecutionConfig(
        "0.4.0", Decimal("0.01"), "wait", "MARKET", "GTC", "force_flatten_with_cost",
        latency, stress_latency, FillProfile("native-book", Decimal(0), "quote"),
        FillProfile("native-book-stress", Decimal(0), "quote"),
    )
    rule = {
        "instrument": "ETHUSDT-PERP.SIM", "tick_size": "0.01", "step_size": "0.001",
        "min_qty": "0.001", "min_notional": "10", "maker_fee": "0.001",
        "taker_fee": "0.001", "initial_margin_rate": "0.01",
        "maintenance_margin_rate": "0.005", "effective_ns": 1,
    }
    references: dict[str, Decimal] = {}
    session = make_perpetual_session(
        rule=rule, execution=execution, mode="base", starting_balance=Decimal("100000"),
        record_fill_reference=lambda client_id, reference: references.__setitem__(client_id, reference),
    )
    session.open(SimpleNamespace(dataset_hash="a" * 64, fold_hash="b" * 64,
                                 config_hash=execution.fingerprint()))
    try:
        from dataclasses import replace

        session.bootstrap((_perpetual_batch(
            1, ("best_quote", {"instrument": rule["instrument"], "bid_price": "1000.00",
                               "bid_qty": "1.000", "ask_price": "1000.01", "ask_qty": "0.100",
                               "update_id": 1, "transaction_time_ns": 60_000_000_000}),
        ),), decision_ts_ns=60_000_000_000)
        entered = session.advance(
            replace(_perpetual_plan("quote-entry", "0.500"),
                    latency_model_id=latency.model_id,
                    execution_spec_hash=execution.fingerprint()),
            batches=(_perpetual_batch(
                2, ("best_quote", {"instrument": rule["instrument"], "bid_price": "1000.04",
                                   "bid_qty": "1.000", "ask_price": "1000.05", "ask_qty": "0.400",
                                   "update_id": 2, "transaction_time_ns": 120_000_000_000}),
            ),), next_decision_ts_ns=120_000_000_000,
        )
        assert sum(Decimal(fill["last_qty"]) for fill in entered.fills) == Decimal("0.500")
        at_best_ask = sum(Decimal(fill["last_qty"]) for fill in entered.fills
                          if Decimal(fill["last_px"]) == Decimal("1000.01"))
        assert at_best_ask <= Decimal("0.100")
        assert max(Decimal(fill["last_px"]) for fill in entered.fills) <= Decimal("1000.05")
        assert all(references[fill["client_order_id"]] == Decimal("1000.01") for fill in entered.fills)
        assert entered.orders == ()
    finally:
        session.close()


def test_native_bar_depth_stress_limits_fill_size_to_frozen_synthetic_fraction() -> None:
    from dataclasses import replace
    from rles_execution import FillProfile, LatencyProfile, ResolvedExecutionConfig
    from rles_nautilus_adapter.perpetual import make_perpetual_session

    latency = LatencyProfile("depth-base", 0, 0, 0)
    stress_latency = LatencyProfile("depth-stress", 0, 0, 0)
    execution = ResolvedExecutionConfig(
        "0.4.0", Decimal("0.01"), "wait", "MARKET", "GTC", "force_flatten_with_cost",
        latency, stress_latency,
        FillProfile("bar-base", Decimal("1")),
        FillProfile("bar-limited-depth", Decimal("2"), simulated_depth_fraction=Decimal("0.4")),
    )
    rule = {
        "instrument": "ETHUSDT-PERP.SIM", "tick_size": "0.01", "step_size": "0.001",
        "min_qty": "0.001", "min_notional": "10", "maker_fee": "0.001",
        "taker_fee": "0.001", "initial_margin_rate": "0.01",
        "maintenance_margin_rate": "0.005", "effective_ns": 1,
    }
    session = make_perpetual_session(
        rule=rule, execution=execution, mode="stress", starting_balance=Decimal("100000"),
    )
    session.open(SimpleNamespace(dataset_hash="c" * 64, fold_hash="d" * 64,
                                 config_hash=execution.fingerprint()))
    try:
        session.bootstrap((_perpetual_batch(1),), decision_ts_ns=60_000_000_000)
        plan = replace(_perpetual_plan("partial-depth", "0.500"),
                       latency_model_id=stress_latency.model_id,
                       execution_spec_hash=execution.fingerprint())
        snapshot = session.advance(plan, batches=(_perpetual_batch(2),),
                                   next_decision_ts_ns=120_000_000_000)
        filled = sum(Decimal(fill["last_qty"]) for fill in snapshot.fills)
        assert filled == Decimal("0.200")
        assert filled / Decimal("0.500") == Decimal("0.4")
        assert len(snapshot.orders) == 1
    finally:
        session.close()


def test_native_market_event_decoder_checks_index_quality_scope_and_funding_conflicts() -> None:
    from nautilus_trader.model import BarType, IndexPriceUpdate, QuoteTick

    instrument = _perpetual_instrument()
    decoder = NativeMarketEventDecoder(
        instrument=instrument,
        bar_type=BarType.from_str(f"{instrument.id}-1-MINUTE-LAST-EXTERNAL"),
    )

    def market_event(event_id: str, kind: str, timestamp: int, payload: dict,
                     quality: DataQualityGrade = DataQualityGrade.B,
                     available_at: int | None = None):
        initialized = timestamp if available_at is None else available_at
        return MarketEvent(
            event_id=event_id, ts_event_ns=timestamp, ts_init_ns=initialized,
            available_at_ns=initialized, source_hash="c" * 64, quality_grade=quality,
            event_type=kind, payload=FrozenFields.from_mapping(payload),
        )

    index = decoder(market_event(
        "index-1", "index", 100,
        {"instrument": str(instrument.id), "price": "1000.25"},
    ))
    assert isinstance(index, IndexPriceUpdate)
    assert index.value == instrument.make_price(1000.25)
    quote = decoder(market_event("quote-1", "best_quote", 100, {
        "instrument": str(instrument.id), "bid_price": "1000.20", "bid_qty": "0.100",
        "ask_price": "1000.30", "ask_qty": "0.200", "update_id": 1,
        "transaction_time_ns": 99,
    }))
    assert isinstance(quote, QuoteTick)
    assert quote.bid_price == instrument.make_price(1000.20)
    assert quote.ask_size == instrument.make_qty(0.200)
    with pytest.raises(EngineBoundaryError, match="unavailable or has unusable quality"):
        decoder(market_event(
            "low-quality-mark", "mark", 101,
            {"instrument": str(instrument.id), "price": "1000"},
            DataQualityGrade.D,
        ))
    with pytest.raises(EngineBoundaryError, match="differs from native adapter scope"):
        decoder(market_event(
            "wrong-instrument", "index", 102,
            {"instrument": "BTCUSDT-PERP.SIM", "price": "1000"},
        ))

    scheduled = market_event("rate-1", "funding_update", 110, {
        "instrument": str(instrument.id), "rate": "0.001", "next_funding_ns": 200,
    })
    assert decoder(scheduled).next_funding_ns == 200
    late_settlement = market_event("published-settlement", "funding_settlement", 130, {
        "instrument": str(instrument.id), "rate": "0.001", "settlement_ns": 130,
        "interval_minutes": 480, "settlement_mark_price": "2000.00",
    }, available_at=140)
    decoded_mark, decoded_settlement = decoder(late_settlement)
    assert decoded_mark.value == instrument.make_price(2000.00)
    assert decoded_mark.ts_event == 130
    assert decoded_mark.ts_init == 140
    assert decoded_settlement.ts_event == 130
    assert decoded_settlement.ts_init == 140
    assert decoded_settlement.next_funding_ns == 141
    settled_feature = decoder(market_event("settled-feature", "funding_update", 480, {
        "instrument": str(instrument.id), "rate": "0.001",
        "rate_semantics": "last_settled_rate", "interval_minutes": 480,
    }, available_at=490))
    assert settled_feature.interval is None
    assert settled_feature.next_funding_ns is None
    with pytest.raises(EngineBoundaryError, match="last settled funding rate cannot schedule"):
        decoder(market_event("invalid-settled-feature", "funding_update", 500, {
            "instrument": str(instrument.id), "rate": "0.001",
            "rate_semantics": "last_settled_rate", "next_funding_ns": 600,
        }))
    with pytest.raises(EngineBoundaryError, match="unsupported native funding rate semantics"):
        decoder(market_event("unknown-rate-semantics", "funding_update", 501, {
            "instrument": str(instrument.id), "rate": "0.001",
            "rate_semantics": "settled_last_rate", "interval_minutes": 480,
        }))
    with pytest.raises(EngineBoundaryError, match="conflicting funding rates"):
        decoder(market_event("settlement-conflict", "funding_settlement", 200, {
            "instrument": str(instrument.id), "rate": "0.002", "settlement_ns": 200,
        }))


@pytest.mark.parametrize("compact_history", [False, True])
def test_delayed_rest_funding_settlement_is_charged_once_at_settlement_mark(
    compact_history: bool,
) -> None:
    from dataclasses import replace

    instrument = _perpetual_instrument()
    submitter = NativePlanSubmitter(
        instrument_id=str(instrument.id), quantity_unit="ETH",
        quantity_increment=Decimal("0.001"), execution_spec_hash="e" * 64,
        latency_model_id="latency-1",
    )
    decoder = NativeMarketEventDecoder(
        instrument=instrument,
        bar_type=BarType.from_str(f"{instrument.id}-1-MINUTE-LAST-EXTERNAL"),
    )
    session = EngineSession(
        build_native=_perpetual_builder(submitter), submit_plan=submitter,
        decode_event=decoder,
        facts_reader=NativeFactsReader(
            account_id=AccountId.from_str("SIM-001"), instrument_id=instrument.id,
            currency="USDT", compact_history=compact_history,
        ),
        contract_snapshot_factory=lambda **facts: SimpleNamespace(**facts),
        certified_barrier_protocol=EngineSession.BARRIER_PROTOCOL,
    )
    session.open(SimpleNamespace(dataset_hash="a" * 64, fold_hash="b" * 64,
                                 config_hash="c" * 64))
    try:
        initial = session.bootstrap((_perpetual_batch(1),), decision_ts_ns=60_000_000_000)
        entered = session.advance(
            _perpetual_plan("delayed-funding-entry", "1.000"),
            batches=(_perpetual_batch(2),), next_decision_ts_ns=120_000_000_000,
        )
        assert len(entered.positions) == 1
        wait = replace(_perpetual_plan("delayed-funding-wait", "1.000"),
                       action="NOOP", delta_qty=Decimal(0))
        session.advance(
            wait,
            batches=(_perpetual_batch(3, ("mark", {
                "instrument": str(instrument.id), "price": "1000.00",
            })),),
            next_decision_ts_ns=180_000_000_000,
        )

        settlement_ns = 480 * 60 * 1_000_000_000
        delay_ns = 1_000_000
        source_batch = _perpetual_batch(480, ("funding_settlement", {
            "instrument": str(instrument.id), "rate": "0.001",
            "interval_minutes": 480, "settlement_ns": settlement_ns,
            "settlement_mark_price": "2000.123",
        }), ("funding_update", {
            "instrument": str(instrument.id), "rate": "0.001",
            "interval_minutes": 480, "rate_semantics": "last_settled_rate",
        }))
        delayed_events = tuple(replace(
            event, ts_init_ns=settlement_ns + delay_ns,
            available_at_ns=settlement_ns + delay_ns,
        ) for event in source_batch.events)
        available_ns = settlement_ns + delay_ns
        pending = session.advance(
            replace(wait, decision_id="delayed-funding-publish",
                    plan_id="delayed-funding-publish", idempotency_key="delayed-funding-publish"),
            batches=(VisibleBatch(available_ns, delayed_events, True),),
            next_decision_ts_ns=available_ns,
        )
        assert pending.funding == ()

        settled = session.advance(
            replace(wait, decision_id="delayed-funding-post-settlement",
                    plan_id="delayed-funding-post-settlement",
                    idempotency_key="delayed-funding-post-settlement"),
            batches=(_perpetual_batch(481),),
            next_decision_ts_ns=481 * 60 * 1_000_000_000,
        )
        assert settled.mark == Decimal("2000.123")
        assert len(settled.funding) == 1
        assert Money.from_str(settled.funding[0]["pnl_change"]).as_decimal() == Decimal("-2.000123")
        assert settled.equity == (initial.equity + Decimal("1000.123")
                                  - Decimal("1.00") - Decimal("2.000123"))
    finally:
        session.close()


def test_native_funding_event_deduplication_rejects_conflicting_records() -> None:
    from nautilus_trader.core import UUID4
    from nautilus_trader.model import PositionAdjusted, PositionAdjustmentType, PositionId, StrategyId, TraderId

    # This test covers extraction deduplication; the neighboring replay test
    # verifies the account-side funding payment through the native engine.
    event = PositionAdjusted(
        trader_id=TraderId("TRADER-001"), strategy_id=StrategyId("test-001"),
        instrument_id=_instrument().id, position_id=PositionId("test-position"),
        account_id=AccountId.from_str("SIM-001"), adjustment_type=PositionAdjustmentType.FUNDING,
        quantity_change=None, pnl_change=Money.from_str("-0.01 USD"), reason="test extraction",
        event_id=UUID4(), ts_event=60_000_000_000, ts_init=60_000_000_000,
    )
    record = event.to_dict()
    identical = PositionAdjusted.from_dict(record)
    assert NativeFactsReader._unique_events((event, identical)) == (record,)
    conflict = PositionAdjusted.from_dict({**record, "pnl_change": "-0.02 USD"})
    with pytest.raises(EngineBoundaryError, match="conflicting facts"):
        NativeFactsReader._unique_events((event, conflict))
