"""Fixed L2 encoding budget; Python allocation evidence, not whole-process RSS."""
from __future__ import annotations

import gc
import json
from time import perf_counter
import tracemalloc

import pytest

from rles_data.core import Availability, Event, MINUTE_NS, _snapshot_digest, digest

pytestmark = pytest.mark.perf


def test_snapshot_encoding_has_bounded_temporary_memory(tmp_path, record_property):
    # Allocate input before measuring: the regression was an extra full JSON copy.
    events = tuple(
        Event(f"mark-{i:06d}", "mark", (i + 1) * MINUTE_NS, (i + 1) * MINUTE_NS,
              (i + 1) * MINUTE_NS, Availability.NATIVE, "a" * 64,
              {"instrument": "ETHUSDT-PERP.SIM", "price": "2300.25"}, "synthetic:perf")
        for i in range(10_000)
    )
    _snapshot_digest(events[:1])  # Warm Python's encoding path outside measurement.
    samples = {}
    identities = []
    for name, encode in (("stream", _snapshot_digest), ("materialized_reference", digest)):
        gc.collect()
        tracemalloc.start()
        started = perf_counter()
        try:
            identities.append(encode(events))
            wall_sec = perf_counter() - started
            _, peak_bytes = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        samples[name] = {"wall_sec": wall_sec, "peak_python_bytes": peak_bytes}
    report = {
        "schema": "rles.snapshot-perf.v1", "event_count": len(events), "samples": samples,
        "max_stream_python_bytes": 256 * 1024, "max_stream_wall_sec": 20,
        "scope": "canonical snapshot encoding only; input excluded; not RSS or 90-day acceptance",
    }
    (tmp_path / "snapshot-perf.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    for name, sample in samples.items():
        for metric, value in sample.items():
            record_property(f"snapshot_{name}_{metric}", value)
    assert identities[0] == identities[1]
    assert samples["stream"]["peak_python_bytes"] < report["max_stream_python_bytes"]
    assert samples["stream"]["peak_python_bytes"] < samples["materialized_reference"]["peak_python_bytes"] / 10
    # A generous watchdog catches severe throughput regressions without ranking CI hosts.
    assert 0 < samples["stream"]["wall_sec"] < report["max_stream_wall_sec"]
