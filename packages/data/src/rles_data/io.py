"""Append-only RAW partitions, streaming Parquet ingress and commit-verified CURATED."""
from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path
import re
import shutil
import uuid
from typing import Any, Iterable

from .core import (Availability, AvailabilityPolicy, CuratedStore, DataError, Event,
                   KINDS, canonical_bytes, require_hash, require_ns)
from .schemas import validate_payload

SOURCE_RE = re.compile(r"[A-Za-z0-9_-]{1,96}\Z")
CURATED_ROW_GROUP_SIZE = 16_384


def _curated_row_group(events: list[Event] | tuple[Event, ...], index: int) -> dict[str, Any]:
    available = [event.available_at_ns for event in events]
    known_available = [value for value in available if value is not None]
    return {
        "index": index,
        "events": len(events),
        "min_ts_event_ns": min(event.ts_event_ns for event in events),
        "max_ts_event_ns": max(event.ts_event_ns for event in events),
        "min_available_at_ns": min(known_available) if len(known_available) == len(events) else None,
        "max_available_at_ns": max(known_available) if len(known_available) == len(events) else None,
        "events_sha256": sha256(canonical_bytes(events)).hexdigest(),
    }


def _marker_sha(marker: Path) -> str:
    receipt = marker.with_suffix(marker.suffix + ".sha256")
    if not receipt.is_file():
        raise DataError("curated scoped manifest receipt missing")
    expected = receipt.read_text(encoding="ascii").strip()
    if not re.fullmatch(r"[0-9a-f]{64}", expected) or _file_sha(marker) != expected:
        raise DataError("curated scoped manifest receipt mismatch")
    return expected


def curated_source_identity(path: str | Path) -> dict[str, Any]:
    """Read and validate the immutable curated marker without decoding event payloads.

    Research runners can freeze this identity in their access ledger before
    calling a payload reader. The Parquet file hash is verified by the reader
    itself before any row is decoded.
    """
    target = Path(path)
    marker = target.with_name(target.name + ".COMPLETED.json")
    if not target.is_file() or not marker.is_file():
        raise DataError("curated snapshot not COMPLETE")
    marker_bytes = marker.read_bytes()
    try:
        metadata = json.loads(marker_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DataError("invalid curated commit metadata") from exc
    schema_version = metadata.get("schema_version") if isinstance(metadata, dict) else None
    expected_fields = ({"schema_version", "snapshot_hash", "file_sha256", "events"}
                       if schema_version == "0.4.0" else
                       {"schema_version", "snapshot_hash", "file_sha256", "events", "row_groups"})
    if (not isinstance(metadata, dict) or schema_version not in {"0.4.0", "0.4.1"}
            or set(metadata) != expected_fields or type(metadata.get("events")) is not int
            or metadata["events"] <= 0):
        raise DataError("invalid curated commit metadata")
    snapshot_hash = require_hash(metadata["snapshot_hash"])
    file_hash = require_hash(metadata["file_sha256"])
    marker_hash = (_marker_sha(marker) if schema_version == "0.4.1"
                   else sha256(marker_bytes).hexdigest())
    return {"schema_version": schema_version, "snapshot_hash": snapshot_hash,
            "file_sha256": file_hash, "marker_sha256": marker_hash,
            "events": metadata["events"]}


def _event_from_row(row: dict[str, Any]) -> Event:
    payload = json.loads(row["payload_json"])
    # Event.__post_init__ performs the strict payload validation after freezing.
    return Event(row["event_id"], row["kind"], row["ts_event_ns"], row["ts_init_ns"],
                 row["available_at_ns"], Availability(row["availability_source"]),
                 row["source_hash"], payload, row["provenance"])


def _file_sha(path: Path) -> str:
    h = sha256()
    with path.open("rb") as src:
        while chunk := src.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def _sync_dir(path: Path) -> None:
    if os.name == "posix":
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def _publish_new(path: Path, data: bytes) -> None:
    """Stage then hard-link once; never expose truncated final files."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name("." + path.name + "." + uuid.uuid4().hex + ".staging")
    try:
        with temp.open("xb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.link(temp, path)  # O_EXCL semantics, including simultaneous writers
        _sync_dir(path.parent)
    finally:
        temp.unlink(missing_ok=True)


def _safe_source(source: str, kind: str, suffix: str) -> None:
    if not isinstance(source, str) or SOURCE_RE.fullmatch(source) is None:
        raise DataError("source must be a path-safe identifier")
    if kind not in KINDS or suffix not in (".parquet", ".arrow"):
        raise DataError("unsupported raw source format or kind")


def store_raw(root: str | Path, source: str, kind: str, raw: bytes,
              suffix: str = ".parquet") -> dict[str, str]:
    _safe_source(source, kind, suffix)
    if not isinstance(raw, bytes) or not raw:
        raise DataError("RAW bytes required")
    hash_value = sha256(raw).hexdigest()
    target = Path(root) / "RAW" / source / kind / hash_value[:2] / (hash_value + suffix)
    try:
        _publish_new(target, raw)
    except FileExistsError:
        if _file_sha(target) != hash_value:
            raise DataError("immutable RAW hash mismatch") from None
    return {"path": str(target), "source_hash": hash_value, "kind": kind, "source": source}


def store_raw_file(root: str | Path, source: str, kind: str, input_path: str | Path,
                   suffix: str = ".parquet") -> dict[str, str]:
    """Bounded-memory RAW import for multi-GB source files."""
    _safe_source(source, kind, suffix)
    original = Path(input_path)
    hash_value = _file_sha(original)
    if original.stat().st_size == 0:
        raise DataError("empty RAW input")
    target = Path(root) / "RAW" / source / kind / hash_value[:2] / (hash_value + suffix)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_name("." + target.name + "." + uuid.uuid4().hex + ".staging")
    try:
        with original.open("rb") as src, temp.open("xb") as dst:
            shutil.copyfileobj(src, dst, length=1024 * 1024)
            dst.flush()
            os.fsync(dst.fileno())
        if _file_sha(temp) != hash_value:
            raise DataError("RAW changed during import")
        try:
            os.link(temp, target)
            _sync_dir(target.parent)
        except FileExistsError:
            if _file_sha(target) != hash_value:
                raise DataError("immutable RAW hash mismatch") from None
    finally:
        temp.unlink(missing_ok=True)
    return {"path": str(target), "source_hash": hash_value, "kind": kind, "source": source}


def ingest_parquet(path: str | Path, source_hash: str, kind: str, provenance: str,
                   conservative_delay_ns: int | None = None, *, batch_size: int = 16384) -> CuratedStore:
    """Require event_id/ts_event_ns/ts_init_ns/payload_json, plus optional available_at_ns.

    ts_event_ns on 1m OHLCV is the close timestamp.  No default funding/mark/rules.
    Parquet is streamed in batches; source is verified over bytes before import.
    """
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise DataError("pyarrow is required for Parquet ingestion") from exc
    if kind not in KINDS or not provenance or type(batch_size) is not int or batch_size <= 0:
        raise DataError("invalid kind, provenance or batch size")
    require_hash(source_hash)
    policy = AvailabilityPolicy(conservative_delay_ns)
    location = Path(path)
    if _file_sha(location) != source_hash:
        raise DataError("unverified source SHA-256")
    parquet = pq.ParquetFile(location)
    required = {"event_id", "ts_event_ns", "ts_init_ns", "payload_json"}
    columns = set(parquet.schema_arrow.names)
    if not required <= columns:
        raise DataError(f"missing Parquet columns: {sorted(required - columns)}")
    if parquet.metadata.num_rows == 0:
        raise DataError("empty Parquet input")
    return _ingest_batches(
        parquet.iter_batches(batch_size=batch_size,
                             columns=sorted(required | ({"available_at_ns"} & columns))),
        kind, source_hash, provenance, policy,
    )


def _unique_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    obj: dict[str, Any] = {}
    for key, value in pairs:
        if key in obj:
            raise DataError(f"duplicate payload JSON key: {key}")
        obj[key] = value
    return obj


def _ingest_batches(batches: Iterable[Any], kind: str, source_hash: str,
                    provenance: str, policy: AvailabilityPolicy) -> CuratedStore:
    events: list[Event] = []
    for batch in batches:
        for row in batch.to_pylist():
            timestamp = require_ns(row["ts_event_ns"])
            initialized = require_ns(row["ts_init_ns"])
            available, origin = policy.resolve(timestamp, initialized, row.get("available_at_ns"))
            payload_json = row["payload_json"]
            if not isinstance(payload_json, str):
                raise DataError("payload_json must be UTF-8 JSON text")
            try:
                payload = json.loads(payload_json, object_pairs_hook=_unique_json_keys,
                                     parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite")))
            except (ValueError, TypeError) as exc:
                raise DataError("invalid payload_json") from exc
            validate_payload(kind, payload, timestamp)
            if not isinstance(row["event_id"], str) or not row["event_id"]:
                raise DataError("event_id must be a nonempty string")
            events.append(Event(row["event_id"], kind, timestamp, initialized, available,
                                origin, source_hash, payload, provenance))
    if not events:
        raise DataError("empty input event stream")
    return CuratedStore(events)


def ingest_arrow(path: str | Path, source_hash: str, kind: str, provenance: str,
                 conservative_delay_ns: int | None = None) -> CuratedStore:
    """Arrow IPC file ingestion, sharing exactly the Parquet validation rules."""
    try:
        import pyarrow.ipc as ipc
    except ImportError as exc:
        raise DataError("pyarrow required for Arrow ingestion") from exc
    if kind not in KINDS or not provenance:
        raise DataError("invalid Arrow input kind/provenance")
    require_hash(source_hash)
    location = Path(path)
    if _file_sha(location) != source_hash:
        raise DataError("unverified Arrow SHA-256")
    policy = AvailabilityPolicy(conservative_delay_ns)
    with location.open("rb") as stream:
        reader = ipc.open_file(stream)
        columns = set(reader.schema.names)
        required = {"event_id", "ts_event_ns", "ts_init_ns", "payload_json"}
        if not required <= columns:
            raise DataError(f"missing Arrow columns: {sorted(required - columns)}")
        return _ingest_batches((reader.get_batch(i) for i in range(reader.num_record_batches)),
                               kind, source_hash, provenance, policy)


def write_curated(path: str | Path, store: CuratedStore, *,
                  row_group_size: int = CURATED_ROW_GROUP_SIZE) -> str:
    """Write immutable Parquet, then a verified COMPLETE marker; crash remains blocked."""
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise DataError("pyarrow required for curated Parquet") from exc
    target = Path(path)
    marker = target.with_name(target.name + ".COMPLETED.json")
    marker_receipt = marker.with_suffix(marker.suffix + ".sha256")
    if target.exists() or marker.exists() or marker_receipt.exists():
        raise DataError("curated snapshot path already occupied")
    if type(row_group_size) is not int or row_group_size <= 0:
        raise DataError("curated row group size must be a positive integer")
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_name("." + target.name + "." + uuid.uuid4().hex + ".staging")
    try:
        if not store.events:
            raise DataError("cannot export empty curated snapshot")
        schema = pa.schema([
            ("event_id", pa.string()), ("kind", pa.string()),
            ("ts_event_ns", pa.int64()), ("ts_init_ns", pa.int64()),
            ("available_at_ns", pa.int64()), ("availability_source", pa.string()),
            ("source_hash", pa.string()), ("payload_json", pa.string()),
            ("provenance", pa.string()),
        ])
        row_groups = []
        with pq.ParquetWriter(temp, schema, compression="zstd") as writer:
            for index, offset in enumerate(range(0, len(store.events), row_group_size)):
                event_group = store.events[offset: offset + row_group_size]
                rows = []
                for e in event_group:
                    rows.append({"event_id": e.event_id, "kind": e.kind,
                                 "ts_event_ns": e.ts_event_ns, "ts_init_ns": e.ts_init_ns,
                                 "available_at_ns": e.available_at_ns,
                                 "availability_source": e.availability_source.value,
                                 "source_hash": e.source_hash,
                                 "payload_json": canonical_bytes(e.payload).decode("utf-8"),
                                 "provenance": e.provenance})
                writer.write_table(pa.Table.from_pylist(rows, schema=schema))
                row_groups.append(_curated_row_group(event_group, index))
        # Windows FlushFileBuffers requires a writable handle.
        with temp.open("r+b") as fh:
            os.fsync(fh.fileno())
        physical_hash = _file_sha(temp)
        os.link(temp, target)
        _sync_dir(target.parent)
        marker_bytes = canonical_bytes({
            "schema_version": "0.4.1",
            "snapshot_hash": store.snapshot_hash,
            "file_sha256": physical_hash,
            "events": len(store.events),
            "row_groups": row_groups,
        })
        _publish_new(marker, marker_bytes)
        _publish_new(marker_receipt, sha256(marker_bytes).hexdigest().encode("ascii"))
    finally:
        temp.unlink(missing_ok=True)
    return store.snapshot_hash


def read_curated(path: str | Path) -> CuratedStore:
    """Reject orphan data, missing marker, hash mismatch or tampered semantic content."""
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise DataError("pyarrow required for curated Parquet") from exc
    target = Path(path)
    marker = target.with_name(target.name + ".COMPLETED.json")
    if not target.is_file() or not marker.is_file():
        raise DataError("curated snapshot not COMPLETE")
    metadata = json.loads(marker.read_text(encoding="utf-8"))
    schema_version = metadata.get("schema_version") if isinstance(metadata, dict) else None
    expected_fields = ({"schema_version", "snapshot_hash", "file_sha256", "events"}
                       if schema_version == "0.4.0" else
                       {"schema_version", "snapshot_hash", "file_sha256", "events", "row_groups"})
    if (not isinstance(metadata, dict) or schema_version not in {"0.4.0", "0.4.1"}
            or set(metadata) != expected_fields):
        raise DataError("invalid curated commit metadata")
    if schema_version == "0.4.1":
        _marker_sha(marker)
    require_hash(metadata["snapshot_hash"])
    if _file_sha(target) != require_hash(metadata["file_sha256"]):
        raise DataError("curated file hash mismatch")
    parquet = pq.ParquetFile(target)
    events = []
    if schema_version == "0.4.1":
        row_groups = metadata["row_groups"]
        if (not isinstance(row_groups, list) or len(row_groups) != parquet.metadata.num_row_groups
                or not row_groups):
            raise DataError("curated row group manifest mismatch")
        for index, entry in enumerate(row_groups):
            if not isinstance(entry, dict) or entry.get("index") != index:
                raise DataError("invalid curated row group manifest entry")
            rows = parquet.read_row_group(index).to_pylist()
            group_events = [_event_from_row(row) for row in rows]
            expected_group = _curated_row_group(group_events, index)
            if expected_group != entry:
                raise DataError(f"curated row group content mismatch: {index}")
            events.extend(group_events)
    else:
        for batch in parquet.iter_batches(batch_size=16384):
            events.extend(_event_from_row(row) for row in batch.to_pylist())
    store = CuratedStore(events)
    if store.snapshot_hash != metadata["snapshot_hash"] or len(store.events) != metadata["events"]:
        raise DataError("curated semantic manifest mismatch")
    return store


def read_curated_window(path: str | Path, start_ns: int, end_ns: int, *,
                        expected_marker_sha256: str,
                        allow_legacy_isolated_snapshot: bool = False) -> CuratedStore:
    """Read only complete, integrity-verified Parquet row groups inside a frozen window.

    The caller must reserve the exact access window before calling this function
    and freeze the COMPLETE marker digest in the preregistration. This function
    reads footer metadata for every row group, but decodes only row groups whose
    event and availability timestamp bounds are wholly contained in the window.
    If a row group cuts across a requested boundary, it fails closed instead of
    decoding potentially out-of-scope rows or silently returning partial data.
    Legacy 0.4.0 snapshots are rejected by default. The explicit isolated-snapshot
    mode accepts one only when footer statistics prove every row group is wholly
    inside the registered window; it then verifies the complete file hash before
    decoding the physically isolated Train-only snapshot.
    """
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise DataError("pyarrow required for Parquet") from exc
    require_ns(start_ns)
    require_ns(end_ns)
    if start_ns >= end_ns:
        raise DataError("invalid curated read window")
    if type(allow_legacy_isolated_snapshot) is not bool:
        raise DataError("legacy isolated-snapshot mode must be Boolean")
    require_hash(expected_marker_sha256)
    target = Path(path)
    marker = target.with_name(target.name + ".COMPLETED.json")
    if not target.is_file() or not marker.is_file():
        raise DataError("curated snapshot not COMPLETE")
    marker_bytes = marker.read_bytes()
    metadata = json.loads(marker_bytes)
    schema_version = metadata.get("schema_version") if isinstance(metadata, dict) else None
    actual_marker_sha256 = (_marker_sha(marker) if schema_version == "0.4.1"
                            else sha256(marker_bytes).hexdigest())
    if actual_marker_sha256 != expected_marker_sha256:
        raise DataError("curated COMPLETE marker differs from preregistration")
    if schema_version == "0.4.0" and allow_legacy_isolated_snapshot:
        if (set(metadata) != {"schema_version", "snapshot_hash", "file_sha256", "events"}
                or type(metadata.get("events")) is not int or metadata["events"] <= 0):
            raise DataError("invalid legacy curated commit metadata")
        require_hash(metadata["snapshot_hash"])
        require_hash(metadata["file_sha256"])
        parquet = pq.ParquetFile(target)
        if parquet.metadata.num_rows != metadata["events"] or parquet.metadata.num_row_groups < 1:
            raise DataError("legacy curated row count differs from its commit metadata")
        columns = {name: index for index, name in enumerate(parquet.schema_arrow.names)}
        if not {"ts_event_ns", "available_at_ns"}.issubset(columns):
            raise DataError("legacy curated row-group time columns are missing")
        for index in range(parquet.metadata.num_row_groups):
            row_group = parquet.metadata.row_group(index)
            for column_name in ("ts_event_ns", "available_at_ns"):
                stats = row_group.column(columns[column_name]).statistics
                if (stats is None or not stats.has_min_max or stats.null_count != 0
                        or not start_ns <= int(stats.min) <= int(stats.max) < end_ns):
                    raise DataError(
                        f"legacy curated row group is not physically isolated in the registered window: {index}",
                    )
        # This legacy escape hatch is safe only for a complete snapshot wholly
        # inside the Train scope. read_curated verifies the whole-file hash before
        # it decodes any row payload.
        store = read_curated(target)
        if any(event.available_at_ns is None
               or not start_ns <= event.ts_event_ns < end_ns
               or not start_ns <= event.available_at_ns < end_ns
               for event in store.events):
            raise DataError("legacy curated payload escaped its registered Train window")
        return store
    if (not isinstance(metadata, dict) or metadata.get("schema_version") != "0.4.1"
            or set(metadata) != {"schema_version", "snapshot_hash", "file_sha256", "events", "row_groups"}):
        raise DataError("scoped curated reads require a row-group indexed 0.4.1 snapshot")
    require_hash(metadata["snapshot_hash"])
    require_hash(metadata["file_sha256"])
    if type(metadata["events"]) is not int or metadata["events"] <= 0:
        raise DataError("invalid curated event count")
    row_groups = metadata["row_groups"]
    parquet = pq.ParquetFile(target)
    if (not isinstance(row_groups, list) or not row_groups
            or len(row_groups) != parquet.metadata.num_row_groups):
        raise DataError("curated row group manifest mismatch")
    columns = {name: index for index, name in enumerate(parquet.schema_arrow.names)}
    if not {"ts_event_ns", "available_at_ns"}.issubset(columns):
        raise DataError("curated row group time columns are missing")
    selected: list[tuple[int, dict[str, Any]]] = []
    for index, entry in enumerate(row_groups):
        if not isinstance(entry, dict) or entry.get("index") != index:
            raise DataError("invalid curated row group manifest entry")
        row_group = parquet.metadata.row_group(index)
        if type(entry.get("events")) is not int or entry["events"] != row_group.num_rows:
            raise DataError(f"curated row group count mismatch: {index}")
        bounds: dict[str, tuple[int, int]] = {}
        for column_name in ("ts_event_ns", "available_at_ns"):
            column = row_group.column(columns[column_name])
            stats = column.statistics
            if stats is None or not stats.has_min_max:
                raise DataError(f"curated row group lacks timestamp statistics: {index}")
            bounds[column_name] = (int(stats.min), int(stats.max))
        if (entry.get("min_ts_event_ns"), entry.get("max_ts_event_ns")) != bounds["ts_event_ns"]:
            raise DataError(f"curated event-time metadata mismatch: {index}")
        available_bounds = (entry.get("min_available_at_ns"), entry.get("max_available_at_ns"))
        if available_bounds[0] is None or available_bounds[1] is None:
            event_touches = bounds["ts_event_ns"][1] >= start_ns and bounds["ts_event_ns"][0] < end_ns
            if event_touches:
                raise DataError(f"curated row group has unscopable availability: {index}")
            continue
        if (type(available_bounds[0]) is not int or type(available_bounds[1]) is not int
                or available_bounds != bounds["available_at_ns"]):
            raise DataError(f"curated availability metadata mismatch: {index}")
        event_min, event_max = bounds["ts_event_ns"]
        available_min, available_max = bounds["available_at_ns"]
        event_touches = event_max >= start_ns and event_min < end_ns
        available_touches = available_max >= start_ns and available_min < end_ns
        if not (event_touches or available_touches):
            continue
        fully_contained = (
            event_min >= start_ns and event_max < end_ns
            and available_min >= start_ns and available_max < end_ns
        )
        if not fully_contained:
            raise DataError(f"curated row group crosses the registered window boundary: {index}")
        if not re.fullmatch(r"[0-9a-f]{64}", str(entry.get("events_sha256", ""))):
            raise DataError(f"curated row group digest is malformed: {index}")
        selected.append((index, entry))
    if not selected:
        raise DataError("registered window contains no complete curated row groups")
    events: list[Event] = []
    for index, entry in selected:
        rows = parquet.read_row_group(index).to_pylist()
        group_events = [_event_from_row(row) for row in rows]
        if _curated_row_group(group_events, index) != entry:
            raise DataError(f"curated row group content mismatch: {index}")
        if any(event.available_at_ns is None
               or not start_ns <= event.ts_event_ns < end_ns
               or not start_ns <= event.available_at_ns < end_ns
               for event in group_events):
            raise DataError(f"curated row group escaped the registered window: {index}")
        events.extend(group_events)
    return CuratedStore(events)


def write_fact_json(path: str | Path, data: Any) -> str:
    """Append-only canonical manifest writer for DatasetCard/FeatureSpec/Fold/DQ."""
    encoded = canonical_bytes(data)
    try:
        _publish_new(Path(path), encoded)
    except FileExistsError as exc:
        raise DataError("immutable fact manifest already exists") from exc
    return sha256(encoded).hexdigest()
