"""Metadata-only validation for the Stage69 Episode Reset preregistration."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sqlite3

import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[4]
EVIDENCE = Path(__file__).resolve().parent


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    prereg_path = EVIDENCE / "preregistration.json"
    prereg_raw = prereg_path.read_bytes()
    prereg = json.loads(prereg_raw)
    assert sha256_file(prereg_path) == (
        EVIDENCE / "preregistration.json.sha256"
    ).read_text(encoding="ascii").strip()
    config_path = EVIDENCE / "config.json"
    assert sha256_file(config_path) == prereg["config_sha256"]
    config = json.loads(config_path.read_bytes())
    assert config["environment"]["rules_path"] == prereg["rules_path"]
    assert sha256_file(Path(prereg["rules_path"])) == prereg["rules_sha256"]
    assert prereg["minimum_available_memory_bytes"] == 2 * 1024**3
    assert [row["days"] for row in prereg["reset_windows"]] == [60, 360]

    registry_path = ROOT / ".artifacts" / "research-access.sqlite"
    connection = sqlite3.connect(f"file:{registry_path.as_posix()}?mode=ro", uri=True)
    formal = connection.execute(
        "SELECT start_ns,end_ns FROM access WHERE market='ETHUSDT' "
        "AND family='perpetual-market-v1' AND purpose IN ('test','holdout')",
    ).fetchall()
    connection.close()

    total_events = total_groups = 0
    for source in prereg["sources"]:
        assert source["label"].startswith("train-")
        path = Path(source["path"])
        marker_path = path.with_name(path.name + ".COMPLETED.json")
        marker_raw = marker_path.read_bytes()
        marker = json.loads(marker_raw)
        assert hashlib.sha256(marker_raw).hexdigest() == source["marker_sha256"]
        assert marker["file_sha256"] == source["file_sha256"]
        assert marker["snapshot_hash"] == source["snapshot_hash"]
        import_manifest = path.parent / "import.json"
        assert sha256_file(import_manifest) == source["import_manifest_sha256"]
        scope = source["scope"]
        assert not any(scope["start_ns"] < end and scope["end_ns"] > start
                       for start, end in formal)
        parquet = pq.ParquetFile(path)
        assert parquet.metadata.num_rows == source["events"]
        assert parquet.metadata.num_row_groups == source["row_groups"]
        columns = {name: index for index, name in enumerate(parquet.schema_arrow.names)}
        for group_index in range(parquet.metadata.num_row_groups):
            row_group = parquet.metadata.row_group(group_index)
            for name in ("ts_event_ns", "available_at_ns"):
                stats = row_group.column(columns[name]).statistics
                assert stats is not None and stats.has_min_max and stats.null_count == 0
                assert scope["start_ns"] <= int(stats.min) <= int(stats.max) < scope["end_ns"]
        total_events += parquet.metadata.num_rows
        total_groups += parquet.metadata.num_row_groups

    for relative_path, expected_hash in prereg["source_code_sha256"].items():
        assert sha256_file(ROOT / relative_path) == expected_hash
    funding = config["environment"]["coverage_policy"]["settlement_times_ns"]
    funding_points = sum(len(values) for values in funding.values())
    print(json.dumps({
        "status": "PASS",
        "payload_read": False,
        "physical_train_sources": len(prereg["sources"]),
        "row_groups": total_groups,
        "source_events": total_events,
        "formal_scope_overlaps": 0,
        "funding_schedule_points": funding_points,
        "reset_windows_days": [60, 360],
        "stop_line_bytes": prereg["minimum_available_memory_bytes"],
    }, indent=2))


if __name__ == "__main__":
    main()
