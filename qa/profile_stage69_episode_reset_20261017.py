"""Profile cold/warm Episode Reset on preregistered, isolated Train snapshots."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

from rles_data.core import CuratedStore, DataError
from rles_gym_bridge.historical import HistoricalEnvironmentFactory, load_rules
from rles_orchestrator.research_access import read_registered_train_window, record_development
from profile_native_history_growth_20261016 import (
    DAY_NS,
    RssSampler,
    available_memory,
    coverage_for_window,
    process_rss,
)

STOP_FLOOR_BYTES = 2 * 1024**3
SOURCE_CODE = (
    "qa/profile_stage69_episode_reset_20261017.py",
    "qa/profile_native_history_growth_20261016.py",
    "packages/data/src/rles_data/core.py",
    "packages/data/src/rles_data/io.py",
    "packages/gym_bridge/src/rles_gym_bridge/historical.py",
    "packages/gym_bridge/src/rles_gym_bridge/env.py",
    "packages/orchestrator/src/rles_orchestrator/research_access.py",
    "packages/research/src/rles_research/access.py",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_verified(path: Path, payload: dict) -> str:
    raw = json.dumps(payload, sort_keys=True, indent=2, allow_nan=False).encode() + b"\n"
    path.write_bytes(raw)
    receipt = hashlib.sha256(raw).hexdigest()
    path.with_name(path.name + ".sha256").write_text(receipt + "\n", encoding="ascii")
    return receipt


def check_memory() -> int:
    available = available_memory()
    if available <= STOP_FLOOR_BYTES:
        raise MemoryError(
            f"system available memory {available} reached the {STOP_FLOOR_BYTES}-byte stop line",
        )
    return available


def _fingerprint(factory: HistoricalEnvironmentFactory, observation, info: dict) -> dict:
    snapshot = factory.snapshots[0]
    return {
        "observation_sha256": hashlib.sha256(observation.tobytes()).hexdigest(),
        "decision_ts_ns": info["decision_ts_ns"],
        "initial_equity": info["initial_equity"],
        "episode_window": info["episode_window"],
        "native_equity": str(snapshot.equity),
        "position_count": len(snapshot.positions),
        "open_order_count": len(snapshot.orders),
        "snapshot_count_after_reset": len(factory.snapshots),
    }


def _reset_case(factory: HistoricalEnvironmentFactory, *, days: int, seed: int) -> dict:
    check_memory()
    rss_before = process_rss()
    env_started = time.perf_counter()
    env = factory.make_env()
    make_env_seconds = time.perf_counter() - env_started
    samples = []
    fingerprints = []
    with RssSampler() as sampler:
        for index in range(2):
            check_memory()
            started = time.perf_counter()
            observation, info = env.reset(seed=seed, options={"rles_episode_index": 0})
            samples.append(time.perf_counter() - started)
            fingerprints.append(_fingerprint(factory, observation, info))
    env.close()
    if fingerprints[0] != fingerprints[1]:
        raise DataError(f"{days}-day same-seed cold/warm initial-state fingerprints differ")
    return {
        "days": days,
        "decision_rows": len(factory.rows),
        "make_env_seconds": make_env_seconds,
        "cold_reset_seconds": samples[0],
        "warm_reset_seconds": samples[1],
        "warm_reset_speedup": samples[0] / samples[1] if samples[1] else None,
        "cached_visible_batch_count": len(factory._market_batch_index or ()),
        "cached_row_frame_count": len(factory._market_batch_groups_by_row),
        "cached_barrier_frame_count": len(factory._barrier_frames_by_row or ()),
        "reset_rss_start_bytes": rss_before,
        "reset_rss_peak_bytes": sampler.peak,
        "reset_rss_peak_delta_bytes": max(0, sampler.peak - rss_before),
        "same_seed_initial_state_reproducible": True,
        "initial_state_fingerprint": fingerprints[0],
        "available_memory_after_bytes": available_memory(),
    }


def run(prereg_path: Path, config_path: Path, output_dir: Path) -> dict:
    if output_dir.exists():
        raise DataError("profile output already exists; refusing to overwrite evidence")
    prereg_raw = prereg_path.read_bytes()
    prereg = json.loads(prereg_raw)
    config_raw = config_path.read_bytes()
    config = json.loads(config_raw)
    prereg_receipt = prereg_path.with_name(prereg_path.name + ".sha256")
    if hashlib.sha256(prereg_raw).hexdigest() != prereg_receipt.read_text(encoding="ascii").strip():
        raise DataError("preregistration checksum mismatch")
    if hashlib.sha256(config_raw).hexdigest() != prereg["config_sha256"]:
        raise DataError("frozen reset config checksum mismatch")
    if prereg["window"]["end_ns"] - prereg["window"]["start_ns"] != 360 * DAY_NS:
        raise DataError("frozen profile window is not exactly 360 days")
    if prereg["minimum_available_memory_bytes"] != STOP_FLOOR_BYTES:
        raise DataError("profile stop line differs from the required 2 GiB")
    for relative, expected in prereg["source_code_sha256"].items():
        if sha256_file(Path(relative)) != expected:
            raise DataError(f"frozen source code changed: {relative}")

    environment = config["environment"]
    if sha256_file(Path(environment["rules_path"])) != prereg["rules_sha256"]:
        raise DataError("scenario rules source checksum changed")
    seed = int(config["seed"])
    full_window = prereg["window"]
    initial_free = check_memory()

    # Register every complete, physically isolated Train source before opening
    # any Parquet payload. Labels and marker identities are frozen in prereg.
    registered = []
    for source in prereg["sources"]:
        source_path = Path(source["path"])
        marker_path = source_path.with_name(source_path.name + ".COMPLETED.json")
        if sha256_file(marker_path) != source["marker_sha256"]:
            raise DataError(f"{source['name']} completion marker changed")
        if sha256_file(source_path.parent / "import.json") != source["import_manifest_sha256"]:
            raise DataError(f"{source['name']} import manifest changed")
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        if (marker.get("snapshot_hash") != source["snapshot_hash"]
                or marker.get("file_sha256") != source["file_sha256"]):
            raise DataError(f"{source['name']} marker identity differs from preregistration")
        scope = source["scope"]
        intersection = {
            "start_ns": max(full_window["start_ns"], scope["start_ns"]),
            "end_ns": min(full_window["end_ns"], scope["end_ns"]),
        }
        if intersection["start_ns"] >= intersection["end_ns"]:
            raise DataError(f"{source['name']} has no overlap with the frozen Train window")
        source_config = json.loads(json.dumps(config))
        source_config["output"] = str(output_dir.resolve())
        source_config["environment"].update({
            "market_path": str(source_path.resolve()),
            "start_ns": intersection["start_ns"],
            "end_ns": intersection["end_ns"],
            "curated_marker_sha256": source["marker_sha256"],
            "curated_read_mode": source["source_mode"],
        })
        record_development(source_config, scope, label=source["label"])
        registered.append((source_config, source))

    check_memory()
    load_started = time.perf_counter()
    stores = []
    for source_config, source in registered:
        check_memory()
        stores.append(read_registered_train_window(
            source_config,
            source["scope"],
            marker_sha256=source["marker_sha256"],
            label=source["label"],
            allow_legacy_isolated_snapshot=True,
        ))
    market_events = tuple(event for store in stores for event in store.events)
    rule_events = tuple(load_rules(environment["rules_path"]))
    store = CuratedStore((*market_events, *rule_events))
    source_load_seconds = time.perf_counter() - load_started
    if store.snapshot_hash != prereg["expected_snapshot_hash"]:
        raise DataError("merged isolated Train snapshot differs from verified Stage69 profile")
    del stores, market_events, rule_events

    results = []
    scaler = None
    windows = sorted(prereg["reset_windows"], key=lambda item: item["days"], reverse=True)
    for window in windows:
        check_memory()
        coverage = coverage_for_window(environment["coverage_policy"], window)
        factory_args = {
            **environment,
            "market_path": prereg["sources"][0]["path"],
            **{key: window[key] for key in ("start_ns", "end_ns")},
            "coverage_policy": coverage,
            "split": "train_inner",
            "mode": "base",
            "scaler": scaler,
            "fit_scaler": scaler is None,
            "compact_report_snapshots": True,
            "prepared_store": store,
        }
        factory_rss_start = process_rss()
        factory_started = time.perf_counter()
        with RssSampler() as factory_sampler:
            factory = HistoricalEnvironmentFactory(**factory_args)
        case = _reset_case(factory, days=window["days"], seed=seed)
        case.update({
            "factory_build_seconds": time.perf_counter() - factory_started,
            "factory_rss_peak_bytes": factory_sampler.peak,
            "factory_rss_start_bytes": factory_rss_start,
            "factory_rss_peak_delta_bytes": max(0, factory_sampler.peak - factory_rss_start),
            "store_event_count": len(store.events),
            "store_snapshot_hash": store.snapshot_hash,
            "data_quality_grade": factory.quality.grade.value,
            "rule_provenance": factory.rule.provenance,
        })
        if scaler is None:
            scaler = factory.scaler
        results.append(case)
        del factory

    results.sort(key=lambda item: item["days"])
    by_days = {item["days"]: item for item in results}
    short_fp = by_days[60]["initial_state_fingerprint"]
    long_fp = by_days[360]["initial_state_fingerprint"]
    for key in ("observation_sha256", "decision_ts_ns", "initial_equity", "native_equity",
                "position_count", "open_order_count"):
        if short_fp[key] != long_fp[key]:
            raise DataError(f"60/360-day initial state differs for {key}")

    output_dir.mkdir(parents=True)
    summary = {
        "schema": "rles.stage69.episode-reset-profile.v1",
        "status": "PASS",
        "purpose": "Train-inner cold/warm Reset and cache-construction engineering measurements",
        "preregistration_sha256": hashlib.sha256(prereg_raw).hexdigest(),
        "config_sha256": hashlib.sha256(config_raw).hexdigest(),
        "source_code_sha256": prereg["source_code_sha256"],
        "source_labels": [source["label"] for source in prereg["sources"]],
        "window": full_window,
        "source_load_seconds": source_load_seconds,
        "initial_available_memory_bytes": initial_free,
        "minimum_available_memory_bytes_observed": min(
            [initial_free, *(case["available_memory_after_bytes"] for case in results)],
        ),
        "minimum_available_memory_stop_line_bytes": STOP_FLOOR_BYTES,
        "cases": results,
        "formal_evaluation_queries": 0,
        "interpretation": [
            "Only the preregistered physically isolated Train snapshots were decoded.",
            "The same Train-fitted scaler and first economic timestamp were used for both windows.",
            "Cold/warm timings measure Episode Reset and visible-batch/frame construction, not training.",
            "Rule assumptions are scenario-only; no PnL or strategy claim is made.",
        ],
    }
    write_verified(output_dir / "profile-result.json", summary)
    print(json.dumps(summary, sort_keys=True, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.preregistration, args.config, args.output)


if __name__ == "__main__":
    main()
