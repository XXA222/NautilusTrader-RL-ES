"""Synthetic negative/positive coverage cases; not real-market acceptance."""
from dataclasses import replace

import pytest

from rles_data.core import Availability, CuratedStore, DataError, Event, Grade, MINUTE_NS
from rles_data.coverage import CoveragePolicy
from rles_gym_bridge.observation import ObservationSchema

pytestmark = pytest.mark.unit
SYMBOL = "ETHUSDT-PERP"


def events():
    result = []
    for i in range(1, 5):
        ts = i * MINUTE_NS
        for kind, payload in {
            "ohlcv_1m": {"open": "100", "high": "100", "low": "100", "close": "100", "volume": "1"},
            "mark": {"price": "100"}, "index": {"price": "100"},
            "funding_update": {"rate": "0.001"},
        }.items():
            result.append(Event(f"{kind}-{i}", kind, ts, ts, ts, Availability.NATIVE,
                                "a" * 64, {"instrument": SYMBOL, **payload}, "synthetic:coverage-regression"))
    for kind, payload in {
        "funding_settlement": {"rate": "0.001", "settlement_ns": MINUTE_NS},
        "instrument_rule": {"tick_size": "0.1", "step_size": "0.1", "min_qty": "1", "min_notional": "1",
                            "maker_fee": "0.001", "taker_fee": "0.001", "initial_margin_rate": "0.1",
                            "maintenance_margin_rate": "0.05", "effective_ns": MINUTE_NS, "rule_version": "v1"},
    }.items():
        result.append(Event(kind, kind, MINUTE_NS, MINUTE_NS, MINUTE_NS, Availability.NATIVE,
                            "a" * 64, {"instrument": SYMBOL, **payload}, "synthetic:coverage-regression"))
    return result


def policy(settlements=(MINUTE_NS,)):
    return CoveragePolicy({k: MINUTE_NS for k in ("mark", "index", "funding_update")},
                          {SYMBOL: settlements}, "synthetic:explicit-schedule")


def quality(store, coverage):
    return store.quality(start_ns=MINUTE_NS, end_ns=5 * MINUTE_NS,
                         expected_instruments=(SYMBOL,), coverage_policy=coverage)


def test_complete_interval_and_schedule_authorize_settlement():
    store = CuratedStore(events())
    report = quality(store, policy())
    assert report.grade == Grade.A
    store.require_usable(report, "settlement")


def test_presence_without_temporal_policy_cannot_authorize_full_capabilities():
    store = CuratedStore(events())
    report = quality(store, None)
    assert report.grade == Grade.C
    for purpose in ("rl_es", "leveraged", "liquidation", "settlement"):
        with pytest.raises(DataError):
            store.require_usable(report, purpose)


@pytest.mark.parametrize("kind", ("mark", "index", "funding_update"))
def test_single_initial_fact_does_not_cover_whole_interval(kind):
    store = CuratedStore(e for e in events() if e.kind != kind or e.ts_event_ns == MINUTE_NS)
    report = quality(store, policy())
    assert f"{kind}_coverage_gap" in report.findings
    with pytest.raises(DataError):
        store.require_usable(report, "liquidation")


def test_rule_expiry_mid_interval_is_not_hidden_by_presence():
    store = CuratedStore(replace(e, payload={**e.payload, "valid_until_ns": 2 * MINUTE_NS})
                         if e.kind == "instrument_rule" else e for e in events())
    report = quality(store, policy())
    assert "instrument_rule_coverage_gap" in report.findings
    assert report.grade == Grade.C


def test_missing_expected_settlement_blocks_even_when_another_exists():
    store = CuratedStore(events())
    report = quality(store, policy((MINUTE_NS, 3 * MINUTE_NS)))
    assert report.grade == Grade.D
    assert "funding_settlement_coverage_gap" in report.findings


def test_no_settlement_due_is_explicitly_allowed_in_short_interval():
    store = CuratedStore(e for e in events() if e.kind != "funding_settlement")
    report = quality(store, policy(()))
    assert report.grade == Grade.A
    store.require_usable(report, "settlement")


def test_source_settlement_timestamp_jitter_requires_explicit_tolerance():
    shifted_ns = MINUTE_NS + 1_000_000
    shifted = [
        replace(event, ts_event_ns=shifted_ns, ts_init_ns=shifted_ns,
                available_at_ns=shifted_ns,
                payload={**dict(event.payload), "settlement_ns": shifted_ns})
        if event.kind == "funding_settlement" else event
        for event in events()
    ]
    store = CuratedStore(shifted)
    assert "funding_settlement_coverage_gap" in quality(store, policy()).findings
    tolerant = CoveragePolicy(
        {k: MINUTE_NS for k in ("mark", "index", "funding_update")},
        {SYMBOL: (MINUTE_NS,)}, "synthetic:explicit-settlement-schedule",
        settlement_max_delay_ns=MINUTE_NS,
        settlement_timestamp_tolerance_ns=1_000_000,
    )
    assert quality(store, tolerant).grade == Grade.A


def test_observation_schema_owns_immutable_string_sequences():
    names = ["account.equity", "account.peak_equity", "account.drawdown", "risk.guard_intervention",
             "execution.prev_executed_target"]
    units = ["USD", "USD", "ratio", "ratio", "ratio"]
    schema = ObservationSchema("0.4.0", names, units)
    previous = schema.hash
    names[0] = "changed"
    units[0] = "changed"
    assert schema.hash == previous and type(schema.units) is tuple
    with pytest.raises(ValueError):
        ObservationSchema("0.4.0", (None,), (1,))
