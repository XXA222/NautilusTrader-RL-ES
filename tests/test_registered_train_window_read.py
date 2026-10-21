from __future__ import annotations

from hashlib import sha256
from pathlib import Path

import pytest

from rles_data.core import Availability, CuratedStore, DataError, Event, MINUTE_NS
from rles_data.io import read_curated_window, write_curated
from rles_orchestrator import research_access


def _store() -> CuratedStore:
    events = []
    for minute in range(1, 5):
        price = str(100 + minute)
        timestamp = minute * MINUTE_NS
        events.append(Event(
            f"bar-{minute}", "ohlcv_1m", timestamp, timestamp, timestamp,
            Availability.NATIVE, "a" * 64,
            {"instrument": "BTC/USD", "open": price, "high": price,
             "low": price, "close": price, "volume": "1"},
            "synthetic:registered-window-test",
        ))
    return CuratedStore(events)


def test_registered_train_reader_reserves_exact_scope_before_row_group_decode(
    monkeypatch, tmp_path: Path,
) -> None:
    market_path = tmp_path / "market.parquet"
    write_curated(market_path, _store(), row_group_size=2)
    marker = market_path.with_name(market_path.name + ".COMPLETED.json")
    marker_sha = sha256(marker.read_bytes()).hexdigest()
    events = []

    def reserve(config, window, *, label, source_identity=None):
        events.append(("reserve", config["environment"]["curated_marker_sha256"],
                       config["environment"]["curated_read_mode"], dict(window), label,
                       source_identity))
        return "reservation-1"

    def read(path, start_ns, end_ns, *, expected_marker_sha256,
             allow_legacy_isolated_snapshot=False):
        events.append(("decode", expected_marker_sha256, allow_legacy_isolated_snapshot,
                       start_ns, end_ns))
        return read_curated_window(path, start_ns, end_ns,
                                   expected_marker_sha256=expected_marker_sha256,
                                   allow_legacy_isolated_snapshot=allow_legacy_isolated_snapshot)

    monkeypatch.setattr(research_access, "record_development", reserve)
    monkeypatch.setattr(research_access, "read_curated_window", read)
    result = research_access.read_registered_train_window_with_receipt(
        {"output": str(tmp_path / "run"),
         "environment": {"market_path": str(market_path), "instrument": "BTC/USD",
                         "start_ns": 3 * MINUTE_NS, "end_ns": 4 * MINUTE_NS}},
        {"start_ns": 3 * MINUTE_NS, "end_ns": 5 * MINUTE_NS},
        marker_sha256=marker_sha,
        label="train-stage69-scoped-feature-profile-v1",
    )

    assert [event[0] for event in events] == ["reserve", "decode"]
    assert events[0][1] == events[1][1] == marker_sha
    assert events[0][2] == "row_group_scoped_v1" and events[1][2] is False
    assert [event.event_id for event in result.store.events] == ["bar-3", "bar-4"]
    assert result.reservation_id == "reservation-1"
    assert result.source_identity["marker_sha256"] == marker_sha
    assert result.registered_before_read is True and result.split == "train"


def test_registered_train_reader_rejects_evaluation_overlap_before_reservation(
    monkeypatch, tmp_path: Path,
) -> None:
    market_path = tmp_path / "unused.parquet"
    reserved = False

    def reserve(*_args, **_kwargs):
        nonlocal reserved
        reserved = True

    monkeypatch.setattr(research_access, "record_development", reserve)
    try:
        research_access.read_registered_train_window(
            {"output": str(tmp_path / "run"),
             "environment": {"market_path": str(market_path), "instrument": "BTC/USD",
                             "start_ns": MINUTE_NS, "end_ns": 4 * MINUTE_NS},
             "evaluation": {"start_ns": 4 * MINUTE_NS, "end_ns": 6 * MINUTE_NS}},
            {"start_ns": MINUTE_NS, "end_ns": 5 * MINUTE_NS},
            marker_sha256="a" * 64,
            label="train-scoped-read",
        )
    except ValueError as exc:
        assert "overlaps the evaluation interval" in str(exc)
    else:
        raise AssertionError("evaluation overlap was accepted")
    assert reserved is False


def test_registered_train_read_roundtrips_against_project_registry(tmp_path, monkeypatch):
    from rles_research.access import ProjectResearchRegistry, ResearchLockError

    market_path = tmp_path / "market.parquet"
    write_curated(market_path, _store(), row_group_size=2)
    marker = market_path.with_name(market_path.name + ".COMPLETED.json")
    marker_sha = sha256(marker.read_bytes()).hexdigest()
    registry = ProjectResearchRegistry(tmp_path / "research.sqlite")
    monkeypatch.setattr(research_access, "project_registry", lambda: registry)
    config = {
        "output": str(tmp_path / "train-run"),
        "environment": {
            "market_path": str(market_path), "instrument": "BTC/USD",
            "start_ns": 3 * MINUTE_NS, "end_ns": 4 * MINUTE_NS,
        },
    }

    read = research_access.read_registered_train_window_with_receipt(
        config, {"start_ns": 3 * MINUTE_NS, "end_ns": 5 * MINUTE_NS},
        marker_sha256=marker_sha, label="train-window-smoke",
    )

    assert research_access.verify_registered_train_read(read) is True
    forged = type(read)(
        store=read.store, reservation_id="forged-token",
        reservation_key=read.reservation_key, owner=read.owner,
        source_identity=read.source_identity,
        reservation_identity=read.reservation_identity, market=read.market,
        family=read.family, start_ns=read.start_ns, end_ns=read.end_ns,
    )
    with pytest.raises(ResearchLockError, match="absent or mismatched"):
        research_access.verify_registered_train_read(forged)
    with pytest.raises(DataError, match="do not match"):
        research_access.verify_registered_train_read(
            type(read)(
                store=read.store, reservation_id=read.reservation_id,
                reservation_key=read.reservation_key, owner=read.owner,
                source_identity={**read.source_identity, "marker_sha256": "d" * 64},
                reservation_identity=read.reservation_identity, market=read.market,
                family=read.family, start_ns=read.start_ns, end_ns=read.end_ns,
            ),
        )
