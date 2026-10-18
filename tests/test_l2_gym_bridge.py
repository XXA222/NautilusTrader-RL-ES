from __future__ import annotations

import numpy as np
import pytest

from rles_data.core import (
    Availability, CuratedStore, DataError, Event, FeatureField, FeatureSpec, MINUTE_NS,
)
from rles_data.features import Scaler, aggregate_closed, make_feature_row
from rles_gym_bridge import (
    L2FrozenScalerAdapter,
    ObservationBuilder,
    ObservationSchema,
    visible_market_features,
)


pytestmark = pytest.mark.contracts


def _bar(event_id: str, minute: int, close: float, available_minute: int | None = None) -> Event:
    timestamp = minute * MINUTE_NS
    available = (available_minute or minute) * MINUTE_NS
    value = str(close)
    return Event(event_id, "ohlcv_1m", timestamp, timestamp, available,
                 Availability.NATIVE, "a" * 64,
                 {"instrument": "BTC/USD", "open": value, "high": value,
                  "low": value, "close": value, "volume": "1"}, "test-source")


def test_bounded_ohlcv_feature_uses_event_time_order_and_availability() -> None:
    spec = FeatureSpec("0.4.0", (
        FeatureField("return_1m", "float64", "ratio", "ohlcv_1m", 2, 1, "return"),
    ))
    # Input and availability order differ from event time; the bounded store
    # window must still produce the same chronological feature row.
    store = CuratedStore((_bar("bar-3", 3, 103.0),
                          _bar("bar-4", 4, 104.0),
                          _bar("bar-1", 1, 101.0),
                          _bar("bar-2", 2, 102.0, available_minute=3)))

    row = make_feature_row(store, spec, 4 * MINUTE_NS, instrument="BTC/USD")

    assert row.values == pytest.approx((104.0 / 102.0 - 1.0,))
    assert row.available_at_ns == 4 * MINUTE_NS


def test_bounded_ohlcv_feature_still_rejects_duplicate_close_timestamps() -> None:
    spec = FeatureSpec("0.4.0", (
        FeatureField("return_1m", "float64", "ratio", "ohlcv_1m", 2, 1, "return"),
    ))
    store = CuratedStore((_bar("bar-1", 1, 101.0),
                          _bar("bar-2a", 2, 102.0),
                          _bar("bar-2b", 2, 102.0),
                          _bar("bar-3", 3, 103.0)))

    with pytest.raises(DataError, match="non-unique 1m close timestamp"):
        make_feature_row(store, spec, 4 * MINUTE_NS, instrument="BTC/USD")


def test_direct_multi_timeframe_aggregation_preserves_closed_and_delayed_bar_rules() -> None:
    events = []
    for minute in range(1, 16):
        close = 100.0 + minute
        ts = minute * MINUTE_NS
        available = (minute + 1) * MINUTE_NS if minute == 14 else ts
        events.append(Event(
            f"agg-{minute}", "ohlcv_1m", ts, ts, available, Availability.NATIVE,
            "a" * 64,
            {"instrument": "BTC/USD", "open": str(close - 0.5), "high": str(close + 1),
             "low": str(close - 1), "close": str(close), "volume": str(minute)},
            "test-source",
        ))
    store = CuratedStore(reversed(events))

    before_close = aggregate_closed(store, 14 * MINUTE_NS, 5,
                                    instrument="BTC/USD", lookback_buckets=3)
    at_close = aggregate_closed(store, 15 * MINUTE_NS, 5,
                                instrument="BTC/USD", lookback_buckets=3)

    assert tuple(row["close_ns"] for row in before_close) == (5 * MINUTE_NS, 10 * MINUTE_NS)
    assert tuple(row["close_ns"] for row in at_close) == (5 * MINUTE_NS, 10 * MINUTE_NS, 15 * MINUTE_NS)
    assert at_close == (
        {"close_ns": 5 * MINUTE_NS, "available_at_ns": 5 * MINUTE_NS,
         "open": 100.5, "high": 106.0, "low": 100.0, "close": 105.0, "volume": 15.0},
        {"close_ns": 10 * MINUTE_NS, "available_at_ns": 10 * MINUTE_NS,
         "open": 105.5, "high": 111.0, "low": 105.0, "close": 110.0, "volume": 40.0},
        {"close_ns": 15 * MINUTE_NS, "available_at_ns": 15 * MINUTE_NS,
         "open": 110.5, "high": 116.0, "low": 110.0, "close": 115.0, "volume": 65.0},
    )
    at_close[0]["close"] = -1.0
    assert aggregate_closed(store, 15 * MINUTE_NS, 5,
                            instrument="BTC/USD", lookback_buckets=3)[0]["close"] == 105.0


def test_direct_multi_timeframe_aggregation_rejects_duplicate_close_timestamp() -> None:
    events = tuple(_bar(f"bar-{minute}", minute, 100.0 + minute) for minute in range(1, 6))
    store = CuratedStore((*events, _bar("bar-3-copy", 3, 103.0)))

    with pytest.raises(DataError, match="non-unique 1m close timestamp"):
        aggregate_closed(store, 5 * MINUTE_NS, 5,
                         instrument="BTC/USD", lookback_buckets=1)


def _feature_inputs():
    spec = FeatureSpec("0.4.0", (
        FeatureField("basis", "float64", "ratio", "mark", 1, 1, "basis", 1_000),
        FeatureField("index", "float64", "USD", "index", 1, 1, "latest", 1_000),
    ))
    events = (
        Event("mark-1", "mark", 1_000, 1_000, 1_100, Availability.NATIVE,
              "a" * 64, {"instrument": "BTC/USD", "price": "105"}, "test-source"),
        Event("index-1", "index", 900, 900, 1_200, Availability.NATIVE,
              "b" * 64, {"instrument": "BTC/USD", "price": "100"}, "test-source"),
    )
    schema = ObservationSchema("0.4.0", (
        "account.equity", "market.index", "account.drawdown", "market.basis",
        "account.peak_equity", "risk.guard_intervention",
        "execution.prev_executed_target",
    ), ("USD", "USD", "ratio", "ratio", "USD", "ratio", "ratio"))
    return CuratedStore(events), spec, schema


def test_basis_is_not_visible_before_later_index_and_carries_latest_availability() -> None:
    store, spec, schema = _feature_inputs()
    with pytest.raises(DataError, match="index price not available"):
        make_feature_row(store, spec, 1_150, instrument="BTC/USD")
    row = make_feature_row(store, spec, 1_300, instrument="BTC/USD")
    assert row.available_at_ns == 1_200
    assert row.values == pytest.approx((0.05, 100.0))
    visible = visible_market_features(row, spec, schema,
                                      instrument="BTC/USD", decision_ts_ns=1_300)
    assert visible["basis"].available_at_ns == 1_200
    assert visible["index"].unit == "USD"
    with pytest.raises(ValueError, match="identity, clock or shape"):
        visible_market_features(row, spec, schema,
                                instrument="ETH/USD", decision_ts_ns=1_300)


def test_l2_scaler_only_normalizes_market_columns_in_l3_observation() -> None:
    store, spec, schema = _feature_inputs()
    row = make_feature_row(store, spec, 1_300, instrument="BTC/USD")
    scaler = Scaler(spec.hash, (0.02, 90.0), (0.01, 5.0), "c" * 64, 10)
    adapter = L2FrozenScalerAdapter(schema, spec, scaler, expected_scaler_hash=scaler.hash)
    builder = ObservationBuilder(schema, scaler=adapter, expected_scaler_hash=scaler.hash)
    obs = builder.build(
        decision_ts_ns=1_300,
        market=visible_market_features(row, spec, schema,
                                       instrument="BTC/USD", decision_ts_ns=1_300),
        account={"equity": 1000.0, "drawdown": 0.1, "peak_equity": 1100.0},
        risk={"guard_intervention": 0.0},
        execution={"prev_executed_target": 0.25},
        time={},
    )
    assert obs.dtype == np.float32
    assert obs.tolist() == pytest.approx([1000.0, 2.0, 0.1, 3.0, 1100.0, 0.0, 0.25])
    with pytest.raises(ValueError, match="scaler hash"):
        L2FrozenScalerAdapter(schema, spec, scaler, expected_scaler_hash="d" * 64)


@pytest.mark.parametrize("read_hash_first", [False, True])
@pytest.mark.parametrize("field,value", [
    ("mean", (1.02, 90.0)), ("std", (1.0, 5.0)), ("feature_hash", "d" * 64),
    ("train_window_hash", "e" * 64), ("training_only", False), ("sample_count", 11),
])
def test_scaler_rejects_low_level_mutation_before_and_after_hash_cache(read_hash_first, field, value):
    from rles_data.core import digest
    _, spec, _ = _feature_inputs()
    scaler = Scaler(spec.hash, (0.02, 90.0), (0.01, 5.0), "c" * 64, 10)
    if read_hash_first:
        assert scaler.hash == digest(scaler)
    object.__setattr__(scaler, field, value)
    with pytest.raises(DataError, match="scaler mutated"):
        _ = scaler.hash
    with pytest.raises(DataError, match="scaler mutated"):
        scaler.transform((0.05, 100.0), spec.hash)


def test_restored_scaler_and_transform_internal_mutation_are_detected():
    import pickle
    _, spec, schema = _feature_inputs()

    class MutatingScaler(Scaler):
        def transform(self, vector, feature_hash):
            result = super().transform(vector, feature_hash)
            object.__setattr__(self, "mean", (1.02, 90.0))
            return result

    scaler = Scaler(spec.hash, (0.02, 90.0), (0.01, 5.0), "c" * 64, 10)
    restored = pickle.loads(pickle.dumps(scaler))
    assert restored.hash == scaler.hash
    object.__setattr__(restored, "std", (2.0, 5.0))
    with pytest.raises(DataError, match="scaler mutated"):
        L2FrozenScalerAdapter(schema, spec, restored, expected_scaler_hash=scaler.hash)
    mutating = MutatingScaler(spec.hash, scaler.mean, scaler.std, "c" * 64, 10)
    adapter = L2FrozenScalerAdapter(schema, spec, mutating, expected_scaler_hash=mutating.hash)
    with pytest.raises((DataError, ValueError), match="scaler mutated"):
        adapter.transform(np.ones((1, len(schema.names))))


def test_native_margin_and_liquidation_facts_are_classified_only_when_certified():
    from types import SimpleNamespace

    from rles_gym_bridge.env import _native_constraint_facts

    healthy = SimpleNamespace(
        equity="100", liquidation_capability=True,
        margins=({"maintenance": "50"},), liquidation_orders=(),
    )
    breached = SimpleNamespace(
        equity="50", liquidation_capability=True,
        margins=({"maintenance": "50"},), liquidation_orders=(),
    )
    liquidated = SimpleNamespace(
        equity="100", liquidation_capability=True,
        margins=(), liquidation_orders=({"client_order_id": "LIQUIDATION-X"},),
    )
    unknown = SimpleNamespace(equity="100", margins=(), liquidation_orders=None)
    assert _native_constraint_facts(healthy) == (True, False, False)
    assert _native_constraint_facts(breached) == (True, True, False)
    assert _native_constraint_facts(liquidated) == (True, False, True)
    assert _native_constraint_facts(unknown) == (False, False, False)


def test_terminal_settlement_rejects_equity_currency_change() -> None:
    from types import SimpleNamespace

    from rles_gym_bridge.env import ContractNotReady, _terminal_settlement_facts

    before = SimpleNamespace(
        all_orders=(), fills=(), funding=(), equity="100", currency="USDT",
        decision_ts_ns=10, positions=(),
    )
    after = SimpleNamespace(
        all_orders=(), fills=(), funding=(), equity="99", currency="USDT",
        decision_ts_ns=11, positions=(),
    )
    facts = _terminal_settlement_facts(before, after, policy_exposure=0.5)
    assert facts["equity_delta"] == "-1"
    assert facts["equity_currency"] == "USDT"

    changed_currency = SimpleNamespace(**{**vars(after), "currency": "BNB"})
    with pytest.raises(ContractNotReady, match="terminal settlement account facts"):
        _terminal_settlement_facts(before, changed_currency, policy_exposure=0.5)
