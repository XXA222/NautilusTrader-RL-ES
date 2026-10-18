from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

import pytest

from rles_data.core import Availability, CuratedStore, DataError, Event, MINUTE_NS
import rles_data.io as curated_io
from rles_data.io import curated_source_identity, read_curated, read_curated_window, write_curated


def _store() -> CuratedStore:
    events = []
    for minute in range(1, 5):
        price = str(100 + minute)
        timestamp = minute * MINUTE_NS
        events.append(Event(
            event_id=f"bar-{minute}", kind="ohlcv_1m",
            ts_event_ns=timestamp, ts_init_ns=timestamp, available_at_ns=timestamp,
            availability_source=Availability.NATIVE, source_hash="a" * 64,
            payload={"instrument": "BTC/USD", "open": price, "high": price,
                     "low": price, "close": price, "volume": "1"},
            provenance="synthetic:curated-window-test",
        ))
    return CuratedStore(events)


def _marker_sha(path: Path) -> str:
    marker = path.with_name(path.name + ".COMPLETED.json")
    return sha256(marker.read_bytes()).hexdigest()


def test_curated_source_identity_reads_marker_without_decoding_payload(monkeypatch,
                                                                       tmp_path: Path) -> None:
    path = tmp_path / "market.parquet"
    source = _store()
    write_curated(path, source, row_group_size=2)
    monkeypatch.setattr(curated_io, "read_curated",
                        lambda *_args: pytest.fail("payload reader was called"))

    identity = curated_source_identity(path)

    assert identity == {
        "schema_version": "0.4.1", "snapshot_hash": source.snapshot_hash,
        "file_sha256": json.loads(path.with_name(path.name + ".COMPLETED.json").read_text())["file_sha256"],
        "marker_sha256": _marker_sha(path), "events": len(source.events),
    }


def test_curated_window_reads_only_fully_contained_verified_row_groups(tmp_path: Path) -> None:
    path = tmp_path / "market.parquet"
    source = _store()
    write_curated(path, source, row_group_size=2)
    marker_sha = _marker_sha(path)

    full = read_curated(path)
    scoped = read_curated_window(path, 3 * MINUTE_NS, 5 * MINUTE_NS,
                                 expected_marker_sha256=marker_sha)

    assert full.snapshot_hash == source.snapshot_hash
    assert [event.event_id for event in scoped.events] == ["bar-3", "bar-4"]
    assert all(3 * MINUTE_NS <= event.ts_event_ns < 5 * MINUTE_NS
               for event in scoped.events)


def test_curated_window_fails_when_window_cuts_a_row_group(tmp_path: Path) -> None:
    path = tmp_path / "market.parquet"
    write_curated(path, _store(), row_group_size=2)

    with pytest.raises(DataError, match="crosses the registered window boundary"):
        read_curated_window(path, 2 * MINUTE_NS, 4 * MINUTE_NS,
                            expected_marker_sha256=_marker_sha(path))


def test_curated_window_binds_the_manifest_digest_before_decoding(tmp_path: Path) -> None:
    path = tmp_path / "market.parquet"
    write_curated(path, _store(), row_group_size=2)
    marker = path.with_name(path.name + ".COMPLETED.json")
    payload = json.loads(marker.read_text(encoding="utf-8"))
    payload["row_groups"][1]["events_sha256"] = "f" * 64
    marker.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")),
                      encoding="utf-8")
    # The stale receipt must reject the changed manifest before any row-group read.
    with pytest.raises(DataError, match="receipt mismatch"):
        read_curated_window(path, 3 * MINUTE_NS, 5 * MINUTE_NS,
                            expected_marker_sha256=_marker_sha(path))


def test_curated_window_rejects_legacy_snapshot_without_scoped_receipts(tmp_path: Path) -> None:
    path = tmp_path / "legacy.parquet"
    write_curated(path, _store(), row_group_size=2)
    marker = path.with_name(path.name + ".COMPLETED.json")
    marker_payload = json.loads(marker.read_text(encoding="utf-8"))
    legacy_payload = {
        "schema_version": "0.4.0",
        "snapshot_hash": marker_payload["snapshot_hash"],
        "file_sha256": marker_payload["file_sha256"],
        "events": marker_payload["events"],
    }
    marker.write_text(json.dumps(legacy_payload, sort_keys=True, separators=(",", ":")),
                      encoding="utf-8")
    marker.with_suffix(marker.suffix + ".sha256").write_text(_marker_sha(path), encoding="ascii")

    with pytest.raises(DataError, match="row-group indexed 0.4.1"):
        read_curated_window(path, 3 * MINUTE_NS, 5 * MINUTE_NS,
                            expected_marker_sha256=_marker_sha(path))


def _rewrite_as_legacy(path: Path) -> None:
    marker = path.with_name(path.name + ".COMPLETED.json")
    current = json.loads(marker.read_text(encoding="utf-8"))
    marker.write_text(json.dumps({
        "schema_version": "0.4.0", "snapshot_hash": current["snapshot_hash"],
        "file_sha256": current["file_sha256"], "events": current["events"],
    }, sort_keys=True, separators=(",", ":")), encoding="utf-8")


def test_curated_window_allows_legacy_only_when_every_row_group_is_isolated(
    tmp_path: Path,
) -> None:
    path = tmp_path / "legacy-isolated.parquet"
    source = _store()
    write_curated(path, source, row_group_size=2)
    _rewrite_as_legacy(path)

    result = read_curated_window(
        path, MINUTE_NS, 5 * MINUTE_NS, expected_marker_sha256=_marker_sha(path),
        allow_legacy_isolated_snapshot=True,
    )

    assert result.snapshot_hash == source.snapshot_hash
    assert [event.event_id for event in result.events] == ["bar-1", "bar-2", "bar-3", "bar-4"]


def test_curated_window_rejects_unisolated_legacy_before_decoding(
    monkeypatch, tmp_path: Path,
) -> None:
    path = tmp_path / "legacy-crossing.parquet"
    write_curated(path, _store(), row_group_size=2)
    _rewrite_as_legacy(path)
    monkeypatch.setattr(curated_io, "read_curated",
                        lambda *_args: pytest.fail("unisolated legacy payload was decoded"))

    with pytest.raises(DataError, match="not physically isolated"):
        read_curated_window(
            path, 2 * MINUTE_NS, 5 * MINUTE_NS,
            expected_marker_sha256=_marker_sha(path),
            allow_legacy_isolated_snapshot=True,
        )
