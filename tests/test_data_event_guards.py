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


def test_latest_rule_index_matches_visible_filter_sort_and_conflict_semantics() -> None:
    instrument = "ETHUSDT"

    def rule(event_id: str, effective_minute: int, available_minute: int,
             *, maker_fee: str = "0.0002", valid_until_minute: int | None = None) -> Event:
        effective = effective_minute * MINUTE_NS
        payload = {
            "instrument": instrument, "tick_size": "0.01", "step_size": "0.001",
            "min_notional": "5", "min_qty": "0.001", "maker_fee": maker_fee,
            "taker_fee": "0.0005", "initial_margin_rate": "0.1",
            "maintenance_margin_rate": "0.05", "effective_ns": effective,
            "rule_version": event_id,
        }
        if valid_until_minute is not None:
            payload["valid_until_ns"] = valid_until_minute * MINUTE_NS
        available = available_minute * MINUTE_NS
        event_time = min(effective, available)
        return Event(event_id, "instrument_rule", event_time, event_time, available,
                     Availability.NATIVE, "a" * 64, payload, "test-source")

    store = CuratedStore((
        rule("base", 1, 1),
        rule("late-older", 10, 20),
        rule("early-newer", 15, 15),
        rule("expiring", 30, 25, valid_until_minute=40),
        rule("same-effective-conflict", 30, 35, maker_fee="0.0003"),
    ))

    def reference(decision_ns: int) -> str:
        matches = [event for event in store.visible(decision_ns, "instrument_rule")
                   if event.payload.get("instrument") == instrument
                   and event.payload.get("effective_ns", event.ts_event_ns) <= decision_ns]
        if not matches:
            raise DataError("instrument rule unavailable at this decision time")
        matches.sort(key=lambda event: (event.payload.get("effective_ns", event.ts_event_ns),
                                        event.available_at_ns, event.event_id))
        latest = matches[-1]
        if (latest.payload.get("valid_until_ns") is not None
                and decision_ns >= latest.payload["valid_until_ns"]):
            raise DataError("latest known rule has expired; fallback to older version forbidden")
        same = [event for event in matches if event.payload.get("effective_ns", event.ts_event_ns)
                == latest.payload.get("effective_ns", latest.ts_event_ns)]
        if len({digest(event.payload) for event in same}) > 1:
            raise DataError("conflicting instrument rules at decision time")
        return latest.event_id

    def outcome(query) -> tuple[str, str]:
        try:
            return ("value", query())
        except DataError as exc:
            return ("error", str(exc))

    for minute in range(0, 45):
        decision_ns = minute * MINUTE_NS
        assert outcome(lambda: reference(decision_ns)) == outcome(
            lambda: store.latest_rule(decision_ns, instrument).event_id)
