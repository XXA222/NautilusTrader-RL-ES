from __future__ import annotations

import hashlib
import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from rles_data.__main__ import main

pytestmark = pytest.mark.unit


def test_l2_cli_blocks_expected_instrument_missing_from_manifest(tmp_path, capsys) -> None:
    minute = 60_000_000_000
    parquet_path = tmp_path / "eth.parquet"
    table = pa.table({
        "event_id": ["eth-1"],
        "ts_event_ns": [minute],
        "ts_init_ns": [minute],
        "available_at_ns": [minute],
        "payload_json": [json.dumps({
            "instrument": "ETHUSDT-PERP", "open": "100", "high": "101",
            "low": "99", "close": "100", "volume": "1",
        })],
    })
    pq.write_table(table, parquet_path)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({
        "inputs": [{
            "path": str(parquet_path),
            "source_hash": hashlib.sha256(parquet_path.read_bytes()).hexdigest(),
            "kind": "ohlcv_1m",
            "provenance": "fixture:source-sha256",
        }],
        "start_ns": minute,
        "end_ns": minute * 2,
        "expected_instruments": ["ETHUSDT-PERP", "BTCUSDT-PERP"],
    }), encoding="utf-8")

    exit_code = main(["validate", "--manifest-json", str(manifest_path)])
    output = json.loads(capsys.readouterr().out)
    assert exit_code == 2
    assert output["status"] == "BLOCKED"
    assert output["quality"]["missing_minutes"] == 1
    assert "expected_instrument_ohlcv_missing" in output["quality"]["findings"]


def test_l2_cli_rejects_duplicate_manifest_keys(tmp_path, capsys) -> None:
    manifest_path = tmp_path / "duplicate.json"
    manifest_path.write_text('{"inputs":[],"inputs":[{}]}', encoding="utf-8")

    exit_code = main(["validate", "--manifest-json", str(manifest_path)])
    error = json.loads(capsys.readouterr().err)
    assert exit_code == 2
    assert error["error"] == "DataError"
    assert "duplicate validation manifest key" in error["detail"]
