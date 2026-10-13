from __future__ import annotations

import pytest

from rles_data.core import Availability, CuratedStore, DataError, Event, MINUTE_NS, digest
from rles_data.schemas import validate_payload

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("count", [0, 1, 3])
def test_streamed_snapshot_retains_canonical_identity_for_all_availability_states(count) -> None:
    events = tuple(
        Event(f'行情-"{index}"', "mark", (index + 1) * MINUTE_NS, (index + 1) * MINUTE_NS,
              None if source == Availability.MISSING else (index + 1) * MINUTE_NS + index,
              source, "a" * 64, {"instrument": "ETHUSDT", "price": 1e-12 * (index + 1)},
              '行情源/quote"/backslash\\/newline\n')
        for index, source in enumerate((Availability.NATIVE, Availability.DERIVED, Availability.MISSING)[:count])
    )
    # Snapshot order and duplicate elimination are part of the old identity.
    store = CuratedStore((*reversed(events), *events[:1]))
    assert store.snapshot_hash == digest(store.events)
    assert store.duplicate_count == int(count > 0)


def test_funding_settlement_requires_integer_native_timestamp() -> None:
    with pytest.raises(DataError, match="settlement_ns must be an integer"):
        validate_payload(
            "funding_settlement",
            {"instrument": "ETHUSDT", "rate": 0.001, "settlement_ns": "60000000000"},
            60_000_000_000,
        )


def test_instrument_rule_rejects_noninteger_effective_timestamp() -> None:
    rule = {
        "instrument": "ETHUSDT",
        "tick_size": 0.01,
        "step_size": 0.001,
        "min_notional": 5,
        "min_qty": 0.001,
        "maker_fee": 0,
        "taker_fee": 0.001,
        "initial_margin_rate": 0.1,
        "maintenance_margin_rate": 0.05,
        "effective_ns": "60000000000",
        "rule_version": "v1",
    }
    with pytest.raises(DataError, match="effective_ns must be an integer"):
        validate_payload("instrument_rule", rule, 60_000_000_000)


def test_instrument_rule_requires_boolean_reduce_only_min_notional_capability() -> None:
    rule = {
        "instrument": "ETHUSDT",
        "tick_size": 0.01,
        "step_size": 0.001,
        "min_notional": 20,
        "min_qty": 0.001,
        "maker_fee": 0.0002,
        "taker_fee": 0.0005,
        "initial_margin_rate": 1,
        "maintenance_margin_rate": 0.004,
        "effective_ns": 60_000_000_000,
        "rule_version": "v1",
        "reduce_only_min_notional_exempt": "true",
    }
    with pytest.raises(DataError, match="reduce_only_min_notional_exempt must be bool"):
        validate_payload("instrument_rule", rule, 60_000_000_000)
