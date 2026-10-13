from __future__ import annotations

import numpy as np
import pytest

from rles_data.core import Availability, CuratedStore, DataError, Event, FeatureField, FeatureSpec
from rles_data.features import Scaler, make_feature_row
from rles_gym_bridge import (
    L2FrozenScalerAdapter,
    ObservationBuilder,
    ObservationSchema,
    visible_market_features,
)


pytestmark = pytest.mark.contracts


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
