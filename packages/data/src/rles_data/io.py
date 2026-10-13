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


def write_curated(path: str | Path, store: CuratedStore) -> str:
    """Write immutable Parquet, then a verified COMPLETE marker; crash remains blocked."""
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise DataError("pyarrow required for curated Parquet") from exc
    target = Path(path)
    marker = target.with_name(target.name + ".COMPLETED.json")
    if target.exists() or marker.exists():
        raise DataError("curated snapshot path already occupied")
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
        with pq.ParquetWriter(temp, schema, compression="zstd") as writer:
            for offset in range(0, len(store.events), 16_384):
                rows = []
                for e in store.events[offset: offset + 16_384]:
                    rows.append({"event_id": e.event_id, "kind": e.kind,
                                 "ts_event_ns": e.ts_event_ns, "ts_init_ns": e.ts_init_ns,
                                 "available_at_ns": e.available_at_ns,
                                 "availability_source": e.availability_source.value,
                                 "source_hash": e.source_hash,
                                 "payload_json": canonical_bytes(e.payload).decode("utf-8"),
                                 "provenance": e.provenance})
                writer.write_table(pa.Table.from_pylist(rows, schema=schema))
        # Windows FlushFileBuffers requires a writable handle.
        with temp.open("r+b") as fh:
            os.fsync(fh.fileno())
        physical_hash = _file_sha(temp)
        os.link(temp, target)
        _sync_dir(target.parent)
        _publish_new(marker, canonical_bytes({"schema_version": "0.4.0", "snapshot_hash": store.snapshot_hash,
                                              "file_sha256": physical_hash, "events": len(store.events)}))
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
    if set(metadata) != {"schema_version", "snapshot_hash", "file_sha256", "events"} or metadata["schema_version"] != "0.4.0":
        raise DataError("invalid curated commit metadata")
    require_hash(metadata["snapshot_hash"])
    if _file_sha(target) != require_hash(metadata["file_sha256"]):
        raise DataError("curated file hash mismatch")
    events = []
    parquet = pq.ParquetFile(target)
    for batch in parquet.iter_batches(batch_size=16384):
        for row in batch.to_pylist():
            payload = json.loads(row["payload_json"])
            validate_payload(row["kind"], payload, row["ts_event_ns"])
            events.append(Event(row["event_id"], row["kind"], row["ts_event_ns"], row["ts_init_ns"],
                                row["available_at_ns"], Availability(row["availability_source"]),
                                row["source_hash"], payload, row["provenance"]))
    store = CuratedStore(events)
    if store.snapshot_hash != metadata["snapshot_hash"] or len(store.events) != metadata["events"]:
        raise DataError("curated semantic manifest mismatch")
    return store


def write_fact_json(path: str | Path, data: Any) -> str:
    """Append-only canonical manifest writer for DatasetCard/FeatureSpec/Fold/DQ."""
    encoded = canonical_bytes(data)
    try:
        _publish_new(Path(path), encoded)
    except FileExistsError as exc:
        raise DataError("immutable fact manifest already exists") from exc
    return sha256(encoded).hexdigest()
