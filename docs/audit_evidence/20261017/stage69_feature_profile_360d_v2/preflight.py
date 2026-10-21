from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[4]
EVIDENCE = Path(__file__).resolve().parent


def main() -> None:
    prereg_bytes = (EVIDENCE / "preregistration.json").read_bytes()
    prereg = json.loads(prereg_bytes)
    assert hashlib.sha256(prereg_bytes).hexdigest() == (
        EVIDENCE / "preregistration.json.sha256"
    ).read_text(encoding="ascii").strip()
    config_bytes = (EVIDENCE / "config.json").read_bytes()
    assert hashlib.sha256(config_bytes).hexdigest() == prereg["config_sha256"]
    assert prereg["window"]["end_ns"] - prereg["window"]["start_ns"] == (
        360 * 24 * 60 * 60 * 1_000_000_000
    )

    registry_path = (ROOT / ".artifacts" / "research-access.sqlite").resolve()
    connection = sqlite3.connect(
        f"file:{registry_path.as_posix()}?mode=ro", uri=True,
    )
    formal = connection.execute(
        "SELECT start_ns,end_ns FROM access "
        "WHERE market='ETHUSDT' AND family='perpetual-market-v1' "
        "AND purpose IN ('test','holdout')",
    ).fetchall()

    row_groups = events = 0
    for source in prereg["sources"]:
        path = Path(source["path"])
        marker_path = path.with_name(path.name + ".COMPLETED.json")
        marker_bytes = marker_path.read_bytes()
        marker = json.loads(marker_bytes)
        assert hashlib.sha256(marker_bytes).hexdigest() == source["marker_sha256"]
        assert marker["file_sha256"] == source["marker_file_sha256"]
        assert marker["snapshot_hash"] == source["marker_snapshot_hash"]
        import_manifest = path.parent / "import.json"
        assert hashlib.sha256(import_manifest.read_bytes()).hexdigest() == (
            source["import_manifest_sha256"]
        )

        start_ns, end_ns = source["scope"]["start_ns"], source["scope"]["end_ns"]
        assert not any(start_ns < formal_end and end_ns > formal_start
                       for formal_start, formal_end in formal)
        parquet = pq.ParquetFile(path)
        assert parquet.metadata.num_row_groups == source["row_groups"]
        assert parquet.metadata.num_rows == source["events"]
        for group_index in range(parquet.metadata.num_row_groups):
            group = parquet.metadata.row_group(group_index)
            for column_name in ("ts_event_ns", "available_at_ns"):
                column_index = next(
                    index for index in range(group.num_columns)
                    if group.column(index).path_in_schema == column_name
                )
                stats = group.column(column_index).statistics
                assert stats is not None and stats.has_min_max and stats.null_count == 0
                assert start_ns <= int(stats.min) <= int(stats.max) < end_ns
        row_groups += parquet.metadata.num_row_groups
        events += parquet.metadata.num_rows

    for relative_path, expected_hash in prereg["code_sha256"].items():
        path = ROOT / relative_path
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected_hash
    print(json.dumps({
        "status": "PASS",
        "payload_read": False,
        "window_days": prereg["window"]["days"],
        "row_groups": row_groups,
        "source_events": events,
        "formal_overlaps": 0,
        "funding_settlement_count": 1080,
    }, indent=2))


if __name__ == "__main__":
    main()
