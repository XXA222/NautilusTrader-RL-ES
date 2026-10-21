from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal
import hashlib

import pytest

from rles_contracts.dto import DataQualityGrade, MarketEvent
from rles_contracts.frozen import FrozenFields
from rles_contracts.serialization import (
    canonical_bytes,
    from_arrow_ipc,
    from_canonical_bytes,
    to_arrow_ipc,
)
from rles_data.core import Availability, CuratedStore, DataError, Event
from rles_data.contracts import to_market_event
from rles_data.features import Scaler
from rles_execution.planner import InstrumentRules, TargetQuantityPlanner
from rles_execution.spec import FillProfile, LatencyProfile, ResolvedExecutionConfig
from rles_research.statistics import (
    StatisticsError,
    block_bootstrap,
    candidate_set_sha256,
    deflated_sharpe,
    pbo_cscv,
)
from rles_research.wfo import plan_folds
from rles_research.artifacts import ArtifactStore, ArtifactError

pytestmark = pytest.mark.unit


def test_l1_canonical_event_roundtrip_and_future_guard() -> None:
    event = MarketEvent(
        event_id="evt-1",
        ts_event_ns=100,
        ts_init_ns=101,
        available_at_ns=102,
        source_hash="a" * 64,
        quality_grade=DataQualityGrade.A,
        event_type="mark",
        payload=FrozenFields.from_mapping({"price": Decimal("12.50")}),
    )
    restored = from_canonical_bytes(canonical_bytes(event))
    assert restored == event
    with pytest.raises(ValueError, match="future event"):
        event.require_available(101)
    event.require_available(102)


@pytest.mark.parametrize(
    ("event_ts", "init_ts", "available_ts"),
    ((102, 101, 101), (101, 102, 101)),
)
def test_l1_market_event_rejects_availability_before_causal_inputs(
    event_ts: int, init_ts: int, available_ts: int
) -> None:
    with pytest.raises(ValueError, match="cannot precede event or initialization time"):
        MarketEvent(
            event_id="causality-event",
            ts_event_ns=event_ts,
            ts_init_ns=init_ts,
            available_at_ns=available_ts,
            source_hash="c" * 64,
            quality_grade=DataQualityGrade.A,
            event_type="mark",
            payload=FrozenFields.from_mapping({"price": Decimal("12.50")}),
        )


def test_l1_arrow_ipc_roundtrip_uses_canonical_payload_hashes() -> None:
    event = MarketEvent(
        event_id="arrow-event-1",
        ts_event_ns=100,
        ts_init_ns=101,
        available_at_ns=102,
        source_hash="e" * 64,
        quality_grade=DataQualityGrade.A,
        event_type="mark",
        payload=FrozenFields.from_mapping({"instrument": "ETHUSDT-PERP", "price": Decimal("2450.125")}),
    )
    assert from_arrow_ipc(to_arrow_ipc((event,))) == (event,)


def test_l2_visibility_is_decision_time_bounded() -> None:
    minute = 60_000_000_000
    event = Event(
        event_id="bar-1",
        kind="ohlcv_1m",
        ts_event_ns=minute,
        ts_init_ns=minute + 1,
        available_at_ns=minute + 5,
        availability_source=Availability.DERIVED,
        source_hash="b" * 64,
        payload={
            "instrument": "ETHUSDT-PERP",
            "open": "100",
            "high": "103",
            "low": "99",
            "close": "102",
            "volume": "4",
        },
        provenance="fixture:source-sha256",
    )
    store = CuratedStore((event,))
    assert store.visible(minute + 4) == ()
    assert store.visible(minute + 5) == (event,)
    with pytest.raises(TypeError, match="immutable market payload"):
        event.payload["close"] = "999"  # type: ignore[index]


def test_indexed_visibility_preserves_global_order_kind_filters_and_availability_boundary() -> None:
    minute = 60_000_000_000
    events = (
        Event("bar-late", "ohlcv_1m", minute, minute, minute + 5, Availability.DERIVED, "b" * 64,
              {"instrument": "ETHUSDT-PERP", "open": "100", "high": "101", "low": "99",
               "close": "100", "volume": "1"}, "fixture:indexed-visibility"),
        Event("mark-early", "mark", minute, minute, minute + 4, Availability.NATIVE, "c" * 64,
              {"instrument": "ETHUSDT-PERP", "price": "100"}, "fixture:indexed-visibility"),
        Event("bar-later", "ohlcv_1m", minute, minute, minute + 7, Availability.NATIVE, "d" * 64,
              {"instrument": "ETHUSDT-PERP", "open": "100", "high": "102", "low": "99",
               "close": "101", "volume": "2"}, "fixture:indexed-visibility"),
        Event("mark-unavailable", "mark", minute, minute, None, Availability.MISSING, "e" * 64,
              {"instrument": "ETHUSDT-PERP", "price": "99"}, "fixture:indexed-visibility"),
    )
    store = CuratedStore(events)
    for decision_ns in (minute + 3, minute + 4, minute + 5, minute + 7, minute + 8):
        expected = tuple(event for event in store.events if event.visible(decision_ns))
        assert store.visible(decision_ns) == expected
        assert store.visible(decision_ns, "ohlcv_1m") == tuple(
            event for event in expected if event.kind == "ohlcv_1m")


def test_event_time_window_is_bounded_and_filters_late_availability() -> None:
    def price(event_id: str, event_ns: int, available_ns: int) -> Event:
        return Event(event_id, "mark", event_ns, event_ns, available_ns, Availability.DERIVED,
                     (event_id[0].lower() * 64),
                     {"instrument": "ETHUSDT-PERP", "price": "100"},
                     "fixture:event-time-window")

    events = (price("a-old", 20, 20), price("b-in-window", 50, 60),
              price("c-not-yet-available", 65, 75), price("d-boundary", 70, 70))
    store = CuratedStore(events)

    assert store.visible_event_window(70, 45, "mark") == (events[1], events[3])
    assert store.visible_event_window(74, 45, "mark") == (events[1], events[3])
    assert store.available_event_window(60, 70, kind="mark") == (events[1], events[3])
    assert store.available_event_window(60, 75) == (events[1], events[3], events[2])


def test_l2_quality_counts_ohlcv_gaps_per_instrument() -> None:
    minute = 60_000_000_000
    events = []
    for symbol, timestamps in (
        ("ETHUSDT-PERP", (minute, minute * 2, minute * 3)),
        ("BTCUSDT-PERP", (minute, minute * 3)),
    ):
        for index, timestamp in enumerate(timestamps):
            events.append(Event(
                event_id=f"{symbol}-{index}", kind="ohlcv_1m",
                ts_event_ns=timestamp, ts_init_ns=timestamp,
                available_at_ns=timestamp, availability_source=Availability.NATIVE,
                source_hash=f"{index + 1:x}" * 64,
                payload={"instrument": symbol, "open": "100", "high": "101",
                         "low": "99", "close": "100", "volume": "1"},
                provenance="fixture:source-sha256",
            ))
    report = CuratedStore(events).quality(
        required=("ohlcv_1m",), start_ns=minute, end_ns=minute * 4,
        expected_instruments=("BTCUSDT-PERP", "ETHUSDT-PERP"),
    )
    assert report.grade.value == "DQ-D"
    assert report.missing_minutes == 1
    assert "critical_ohlcv_gap" in report.findings
    with pytest.raises(TypeError, match="immutable market payload"):
        report.counts["ohlcv_1m"] = 0  # type: ignore[index]


def test_l2_quality_blocks_an_expected_instrument_with_no_bars() -> None:
    minute = 60_000_000_000
    event = Event(
        event_id="eth-only", kind="ohlcv_1m", ts_event_ns=minute, ts_init_ns=minute,
        available_at_ns=minute, availability_source=Availability.NATIVE,
        source_hash="f" * 64,
        payload={"instrument": "ETHUSDT-PERP", "open": "100", "high": "101",
                 "low": "99", "close": "100", "volume": "1"},
        provenance="fixture:source-sha256",
    )
    report = CuratedStore((event,)).quality(
        required=("ohlcv_1m",), expected_instruments=("ETHUSDT-PERP", "BTCUSDT-PERP"),
    )
    assert report.grade.value == "DQ-D"
    assert "expected_instrument_ohlcv_missing" in report.findings


def test_l2_quality_rejects_unexpected_supplemental_instrument_facts() -> None:
    minute = 60_000_000_000
    events = [
        Event(
            event_id="eth-bar", kind="ohlcv_1m", ts_event_ns=minute, ts_init_ns=minute,
            available_at_ns=minute, availability_source=Availability.NATIVE,
            source_hash="a" * 64,
            payload={"instrument": "ETHUSDT-PERP", "open": "100", "high": "101",
                     "low": "99", "close": "100", "volume": "1"},
            provenance="fixture:source-sha256",
        ),
        Event(
            event_id="btc-mark", kind="mark", ts_event_ns=minute, ts_init_ns=minute,
            available_at_ns=minute, availability_source=Availability.NATIVE,
            source_hash="b" * 64,
            payload={"instrument": "BTCUSDT-PERP", "price": "65000"},
            provenance="fixture:source-sha256",
        ),
    ]
    with pytest.raises(DataError, match="unexpected market-fact instruments"):
        CuratedStore(events).quality(expected_instruments=("ETHUSDT-PERP",))


@pytest.mark.parametrize(
    ("omission", "expected_grade"),
    (("funding_settlement", "DQ-D"), ("mark", "DQ-C"), ("instrument_rule", "DQ-C")),
)
def test_l2_supplemental_quality_is_checked_per_expected_instrument(
    omission: str, expected_grade: str
) -> None:
    minute = 60_000_000_000
    required = ("ohlcv_1m", "mark", "index", "funding_update",
                "funding_settlement", "instrument_rule")
    events = []
    for symbol in ("ETHUSDT-PERP", "BTCUSDT-PERP"):
        payloads = {
            "ohlcv_1m": {"open": "100", "high": "101", "low": "99", "close": "100", "volume": "1"},
            "mark": {"price": "100"},
            "index": {"price": "100"},
            "funding_update": {"rate": "0.001"},
            "funding_settlement": {"rate": "0.001", "settlement_ns": minute},
            "instrument_rule": {
                "tick_size": "0.1", "step_size": "0.1", "min_notional": "10", "min_qty": "1",
                "maker_fee": "0.001", "taker_fee": "0.001", "initial_margin_rate": "0.1",
                "maintenance_margin_rate": "0.05", "effective_ns": minute, "rule_version": "v1",
            },
        }
        for kind in required:
            if symbol == "BTCUSDT-PERP" and kind == omission:
                continue
            payload = {"instrument": symbol, **payloads[kind]}
            events.append(Event(
                event_id=f"{symbol}-{kind}", kind=kind, ts_event_ns=minute, ts_init_ns=minute,
                available_at_ns=minute, availability_source=Availability.NATIVE,
                source_hash="9" * 64, payload=payload, provenance="fixture:source-sha256",
            ))
    report = CuratedStore(events).quality(
        required=required, start_ns=minute, end_ns=minute * 2,
        expected_instruments=("BTCUSDT-PERP", "ETHUSDT-PERP"),
    )
    assert report.grade.value == expected_grade
    if omission == "funding_settlement":
        assert "funding_untrusted" in report.findings
    else:
        assert "mark_index_or_rule_missing" in report.findings


@pytest.mark.parametrize(
    "kwargs",
    (
        {"sample_count": True},
        {"training_only": 1},
        {"std": [1.0]},
    ),
)
def test_l2_scaler_rejects_mutable_or_weakly_typed_metadata(kwargs: dict) -> None:
    values = {
        "feature_hash": "a" * 64,
        "mean": (0.0,),
        "std": (1.0,),
        "train_window_hash": "b" * 64,
        "sample_count": 1,
        "training_only": True,
    }
    values.update(kwargs)
    with pytest.raises(DataError):
        Scaler(**values)


def test_l2_event_maps_to_l1_without_inventing_time_or_quality() -> None:
    minute = 60_000_000_000
    event = Event(
        event_id="bar-contract-1",
        kind="ohlcv_1m",
        ts_event_ns=minute,
        ts_init_ns=minute + 1,
        available_at_ns=minute + 5,
        availability_source=Availability.DERIVED,
        source_hash="c" * 64,
        payload={"instrument": "ETHUSDT-PERP", "open": "100", "high": "103",
                 "low": "99", "close": "102", "volume": "4"},
        provenance="fixture:source-sha256",
    )
    mapped = to_market_event(event, quality_grade=DataQualityGrade.B)
    assert (mapped.event_id, mapped.ts_event_ns, mapped.ts_init_ns) == (
        event.event_id, event.ts_event_ns, event.ts_init_ns
    )
    assert mapped.available_at_ns == event.available_at_ns
    assert mapped.source_hash == event.source_hash
    assert mapped.quality_grade is DataQualityGrade.B
    assert mapped.payload.get("close") == "102"


@pytest.mark.parametrize("case", ("missing_availability", "dq_d", "nested_payload"))
def test_l2_to_l1_mapping_rejects_unproven_or_unrepresentable_events(case: str) -> None:
    minute = 60_000_000_000
    payload = {"instrument": "ETHUSDT-PERP", "open": "100", "high": "103",
               "low": "99", "close": "102", "volume": "4"}
    available_at_ns = minute + 5
    if case == "missing_availability":
        available_at_ns = None
    elif case == "nested_payload":
        payload["metadata"] = {"source": "nested"}
    event = Event(
        event_id=f"bar-contract-{case}",
        kind="ohlcv_1m",
        ts_event_ns=minute,
        ts_init_ns=minute + 1,
        available_at_ns=available_at_ns,
        availability_source=Availability.MISSING if available_at_ns is None else Availability.DERIVED,
        source_hash="d" * 64,
        payload=payload,
        provenance="fixture:source-sha256",
    )
    grade = DataQualityGrade.D if case == "dq_d" else DataQualityGrade.B
    with pytest.raises(ValueError):
        to_market_event(event, quality_grade=grade)


def test_l3_planner_quantizes_exposure_and_never_opens_beyond_target() -> None:
    config = ResolvedExecutionConfig(
        schema_version="0.4.0",
        deadband=Decimal("0"),
        pending_order_policy="cancel_replace",
        order_type="MARKET",
        tif="GTC",
        terminal_policy="force_flatten_with_cost",
        base_latency=LatencyProfile("base", 0, 1, 1),
        stress_latency=LatencyProfile("stress", 0, 2, 2),
        base_fill=FillProfile("base-fill", Decimal("1")),
        stress_fill=FillProfile("stress-fill", Decimal("2")),
    )
    planner = TargetQuantityPlanner(config)
    plan = planner.plan(
        decision_id="decision-1",
        instrument=InstrumentRules(
            "ETHUSDT-PERP", Decimal("0.1"), Decimal("0.1"), Decimal("10"), Decimal("1")
        ),
        target_exposure=Decimal("0.5"),
        risk_budget_notional=Decimal("1000"),
        equity=Decimal("1000"),
        mark_price=Decimal("100"),
        current_qty=Decimal("0"),
        pending=(),
    )
    assert plan.action == "SUBMIT"
    assert plan.delta_qty == Decimal("5")
    assert plan.target_qty == Decimal("5")


@pytest.mark.short_wfo
def test_l6_wfo_respects_holdout_and_bootstrap_is_seeded() -> None:
    start = datetime(2020, 1, 1, tzinfo=timezone.utc)
    holdout = start + timedelta(days=180)
    folds = plan_folds(
        start=start,
        end=start + timedelta(days=240),
        scheme="60/15/15",
        warmup=timedelta(days=5),
        purge=timedelta(days=1),
        embargo=timedelta(days=1),
        final_holdout_start=holdout,
        data_available_from=start - timedelta(days=5),
    )
    holdout_ns = int(holdout.timestamp() * 1_000_000_000)
    assert folds and all(fold.test_end_ns <= holdout_ns for fold in folds)
    values = [0.01, -0.002, 0.004, 0.003, -0.001, 0.005]
    kwargs = dict(
        block_length=2,
        draws=100,
        seed=31,
        confidence=0.95,
        statistic=lambda sample: sum(sample) / len(sample),
    )
    assert block_bootstrap(values, **kwargs) == block_bootstrap(values, **kwargs)


def test_l6_pbo_rejects_candidate_identity_drift() -> None:
    candidate_ids = ("a", "b")
    data = [[[0.1, 0.2], [0.0, 0.1], [0.2, 0.0], [0.1, 0.1]],
            [[0.0, 0.1], [0.2, 0.2], [0.1, 0.0], [0.0, 0.1]]]
    with pytest.raises(ValueError, match="candidate set changed"):
        pbo_cscv(data, candidate_ids=candidate_ids, candidate_set_hash=hashlib.sha256(b"wrong").hexdigest())
    report = pbo_cscv(
        data,
        candidate_ids=candidate_ids,
        candidate_set_hash=candidate_set_sha256(candidate_ids),
    )
    assert report["candidate_count"] == 2


@pytest.mark.parametrize(
    "override",
    (
        {"block_length": True},
        {"draws": True},
        {"seed": True},
        {"returns": [0.01, True, -0.01, 0.02]},
        {"statistic": lambda _sample: True},
    ),
)
def test_l6_bootstrap_rejects_boolean_values_and_weak_budgets(override: dict) -> None:
    params = {
        "returns": [0.01, -0.02, 0.015, 0.005],
        "block_length": 2,
        "draws": 100,
        "seed": 7,
        "confidence": 0.95,
        "statistic": lambda sample: sum(sample) / len(sample),
    }
    params.update(override)
    with pytest.raises(StatisticsError):
        block_bootstrap(**params)


def test_l6_research_statistics_reject_boolean_financial_inputs() -> None:
    with pytest.raises(StatisticsError, match="Sharpe inputs"):
        deflated_sharpe([0.01, True, -0.02, 0.03], effective_trials=1, trial_sharpes=[0.2])
    ids = ("a", "b")
    bad = [[[0.01, True], [0.02, 0.01], [0.03, 0.02], [0.01, -0.01]],
           [[0.02, 0.01], [0.01, 0.02], [-0.01, 0.01], [0.02, 0.03]]]
    with pytest.raises(StatisticsError, match="invalid slice returns"):
        pbo_cscv(bad, candidate_ids=ids, candidate_set_hash=candidate_set_sha256(ids))
    with pytest.raises(StatisticsError, match="max_partitions"):
        pbo_cscv(bad, candidate_ids=ids, candidate_set_hash=candidate_set_sha256(ids),
                 max_partitions=True)


@pytest.mark.artifact_restore
def test_l6_artifact_commit_verify_restore_and_tamper_rejection(tmp_path) -> None:
    hashes = {
        name: f"{index:x}" * 64
        for index, name in enumerate(
            ("data_hash", "code_hash", "config_hash", "seed_hash", "deps_hash",
             "adapter_hash", "compute_hash", "execution_hash", "risk_hash"),
            start=1,
        )
    }
    store = ArtifactStore(tmp_path / "artifacts", trusted_sources={"qa-local"})
    writer = store.begin("run-1", "attempt-1")
    writer.add("policy.txt", b"deterministic-test-policy")
    target = writer.complete(
        l1_manifest=hashes,
        required_files=["policy.txt"],
        trusted_source_id="qa-local",
    )
    record = store.verify("run-1", "attempt-1")
    assert record["source_id"] == "qa-local"
    restored = store.read_bytes(
        "run-1", "attempt-1", "policy.txt", expected_manifest_hash=record["manifest_sha256"]
    )
    assert restored == b"deterministic-test-policy"
    (target / "unindexed.txt").write_bytes(b"outside frozen inventory")
    with pytest.raises(ArtifactError, match="unindexed artifact files"):
        store.verify("run-1", "attempt-1")
    (target / "unindexed.txt").unlink()
    (target / "policy.txt").write_bytes(b"modified")
    with pytest.raises(ArtifactError, match="hash mismatch"):
        store.verify("run-1", "attempt-1")
