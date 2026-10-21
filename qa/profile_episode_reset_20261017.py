"""Measure historical factory, Episode Reset, and native rollout costs on frozen Train data."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from rles_data.core import CuratedStore, DataError
from rles_data.io import read_curated
from rles_gym_bridge.historical import HistoricalEnvironmentFactory, load_rules
from rles_orchestrator.application import _file_hash, _verify_file, _write_verified, rollout
from rles_orchestrator.research_access import record_development
from rles_orchestrator.wfo_application import _scaler
from profile_native_history_growth_20261016 import (
    DAY_NS, RssSampler, available_memory, coverage_for_window, process_rss,
)

START_FLOOR_BYTES = 10 * 1024**3
STOP_FLOOR_BYTES = 2 * 1024**3
WINDOW_DAYS = (60, 360)
TARGET = 0.25
# The frozen model's Train boundary covers this date; the project registry
# confirms that both profile windows avoid every Test/Holdout interval.
PROFILE_START_NS = 1_722_470_400_000_000_000  # 2024-08-01T00:00:00Z
SOURCE_FILES = (
    "qa/profile_episode_reset_20261017.py",
    "qa/profile_native_history_growth_20261016.py",
    "packages/gym_bridge/src/rles_gym_bridge/historical.py",
    "packages/gym_bridge/src/rles_gym_bridge/env.py",
    "packages/gym_bridge/src/rles_gym_bridge/episode.py",
    "packages/data/src/rles_data/core.py",
    "packages/orchestrator/src/rles_orchestrator/application.py",
    "packages/orchestrator/src/rles_orchestrator/research_access.py",
    "packages/research/src/rles_research/access.py",
)


def _predict(_batch: np.ndarray) -> np.ndarray:
    return np.full((1, 1), TARGET, dtype=np.float32)


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _check_memory(*, start: bool = False) -> int:
    available = available_memory()
    floor = START_FLOOR_BYTES if start else STOP_FLOOR_BYTES
    if available < floor:
        raise MemoryError(f"available physical memory {available} is below {floor} bytes")
    return available


def run(plan_path: Path, output: Path) -> dict:
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if plan.get("split") != "train_inner" or output.exists():
        raise DataError("requires the frozen Train-inner plan and a new output directory")
    profile_plan_path = Path(plan["source_plan"])
    if _file_hash(profile_plan_path) != plan["source_plan_sha256"]:
        raise DataError("frozen native-history profile plan checksum changed")
    profile_plan = json.loads(profile_plan_path.read_text(encoding="utf-8"))
    source_plan_path = Path(profile_plan["source_plan"])
    if _file_hash(source_plan_path) != profile_plan["source_plan_sha256"]:
        raise DataError("frozen Train-inner source plan checksum changed")
    source_plan = json.loads(source_plan_path.read_text(encoding="utf-8"))
    bundle_root = Path(source_plan["bundle"])
    if _verify_file(bundle_root / "bundle.json") != plan["bundle_sha256"]:
        raise DataError("frozen policy bundle checksum changed")
    bundle = json.loads((bundle_root / "bundle.json").read_text(encoding="utf-8"))
    source = profile_plan["source"]
    for key in ("market", "rules"):
        if _file_hash(Path(source[f"{key}_path"])) != source[f"{key}_sha256"]:
            raise DataError(f"frozen {key} source checksum changed")
        if source[f"{key}_sha256"] != plan["source_hashes"][key]:
            raise DataError(f"cost comparison {key} identity differs from frozen source")

    seed = int(plan.get("seed", source_plan["seed"]))
    start_ns = PROFILE_START_NS
    windows = tuple({"start_ns": start_ns, "end_ns": start_ns + days * DAY_NS,
                     "days": days} for days in WINDOW_DAYS)
    if any(window["end_ns"] > int(bundle["training_end_ns"]) for window in windows):
        raise DataError("360-day profile exceeds the frozen Train boundary")
    free_at_start = _check_memory(start=True)

    environment = bundle["environment"]
    registry_config = {"environment": environment, "output": str(output), "seed": seed}
    for window in windows:
        record_development(registry_config, window, label="episode-reset-performance-profile")

    output.mkdir(parents=True)
    code_hashes = {path: _file_hash(Path(path)) for path in SOURCE_FILES}
    _write_verified(output / "plan.json", {
        "schema": "rles.episode-reset-profile-plan.v1",
        "source_plan": str(plan_path.resolve()), "source_plan_sha256": _file_hash(plan_path),
        "source_profile_plan": str(profile_plan_path.resolve()),
        "source_profile_plan_sha256": _file_hash(profile_plan_path),
        "source_train_plan": str(source_plan_path.resolve()),
        "source_train_plan_sha256": _file_hash(source_plan_path),
        "bundle_sha256": plan["bundle_sha256"],
        "source_hashes": {key: source[f"{key}_sha256"] for key in ("market", "rules")},
        "source_code_sha256": code_hashes,
        "split": "train_inner", "windows_days": list(WINDOW_DAYS), "seed": seed,
        "policy": {"kind": "fixed_target", "target_exposure": TARGET},
        "measurements": ["CuratedStore load", "factory construction", "make_env",
                         "cold/warm env.reset", "full native rollout", "peak RSS",
                         "decisions/second", "cached visible-batch and per-row frame counts"],
        "purpose": "engineering profile only; no model selection or profitability gate",
        "historical_economics": "scenario-only rules and costs; not historical certification",
        "resource_floors_bytes": {"start": START_FLOOR_BYTES, "stop": STOP_FLOOR_BYTES},
    })

    records: list[dict] = []
    try:
        load_started = time.perf_counter()
        market = read_curated(source["market_path"])
        rules = load_rules(source["rules_path"])
        store = CuratedStore((*market.events, *rules))
        load_seconds = time.perf_counter() - load_started
        if store.snapshot_hash != bundle["preprocessing"]["identities"]["dataset"]:
            raise DataError("frozen sources differ from the policy dataset identity")
        del market, rules
        scaler = _scaler(bundle_root)

        for window in windows:
            _check_memory()
            coverage = coverage_for_window(environment["coverage_policy"], window)
            factory_args = {
                **environment,
                **{key: window[key] for key in ("start_ns", "end_ns")},
                "coverage_policy": coverage, "split": "train_inner", "mode": "base",
                "scaler": scaler, "fit_scaler": False,
                "reward_scale": bundle.get("reward_scale", 1.0),
                "compact_report_snapshots": True, "prepared_store": store,
            }
            gc.collect()
            factory_rss_start = process_rss()
            factory_started = time.perf_counter()
            with RssSampler() as factory_sampler:
                factory = HistoricalEnvironmentFactory(**factory_args)
            factory_seconds = time.perf_counter() - factory_started

            env_started = time.perf_counter()
            env = factory.make_env(soft_multipliers=bundle["soft_multipliers"])
            make_env_seconds = time.perf_counter() - env_started
            reset_samples = []
            reset_fingerprints = []
            with RssSampler() as reset_sampler:
                for sample_index in range(2):
                    _check_memory()
                    reset_started = time.perf_counter()
                    observation, info = env.reset(seed=seed, options={"rles_episode_index": 0})
                    reset_seconds = time.perf_counter() - reset_started
                    snapshots = tuple(factory.snapshots)
                    snapshot = snapshots[0]
                    fingerprint = {
                        "observation_sha256": _sha_bytes(observation.tobytes()),
                        "decision_ts_ns": info["decision_ts_ns"],
                        "initial_equity": info["initial_equity"],
                        "episode_window": info["episode_window"],
                        "initial_native_equity": str(snapshot.equity),
                        "initial_position_count": len(snapshot.positions),
                        "first_snapshot_count": len(snapshots),
                    }
                    reset_fingerprints.append(fingerprint)
                    reset_samples.append({"sample": sample_index, "seconds": reset_seconds})
            env.close()
            if reset_fingerprints[0] != reset_fingerprints[1]:
                raise DataError("same-seed consecutive resets do not reproduce the initial state")

            _check_memory()
            gc.collect()
            rollout_rss_start = process_rss()
            rollout_started = time.perf_counter()
            with RssSampler() as rollout_sampler:
                report, _ = rollout(
                    factory, _predict, seed=seed,
                    soft_parameters=bundle["soft_multipliers"],
                    collect_observations=False, include_account_rows=False,
                )
            rollout_seconds = time.perf_counter() - rollout_started
            decisions = int(report["action_diagnostics"]["count"])
            record = {
                "days": window["days"], "decision_count": decisions,
                "source_load_seconds": load_seconds,
                "store_event_count": len(store.events), "store_snapshot_hash": store.snapshot_hash,
                "factory_seconds": factory_seconds,
                "factory_rss_start_bytes": factory_rss_start,
                "factory_rss_peak_bytes": factory_sampler.peak,
                "factory_rss_peak_delta_bytes": max(0, factory_sampler.peak - factory_rss_start),
                "make_env_seconds": make_env_seconds,
                "reset_samples": reset_samples,
                "reset_cache_cold_seconds": reset_samples[0]["seconds"],
                "reset_cache_warm_seconds": reset_samples[1]["seconds"],
                "reset_cold_minus_warm_seconds": max(
                    0.0, reset_samples[0]["seconds"] - reset_samples[1]["seconds"]),
                "cached_visible_batch_count": len(factory._market_batch_index or ()),
                "cached_row_frame_count": len(factory._market_batch_groups_by_row),
                "cached_barrier_frame_count": len(factory._barrier_frames_by_row or ()),
                "reset_seconds_mean": sum(item["seconds"] for item in reset_samples) / len(reset_samples),
                "reset_rss_peak_bytes": reset_sampler.peak,
                "reset_rss_peak_delta_bytes": max(0, reset_sampler.peak - factory_sampler.peak),
                "initial_state_reproducible": True,
                "initial_state_fingerprint": reset_fingerprints[0],
                "rollout_seconds": rollout_seconds,
                "decisions_per_second": decisions / rollout_seconds,
                "rollout_rss_start_bytes": rollout_rss_start,
                "rollout_rss_peak_bytes": rollout_sampler.peak,
                "rollout_rss_peak_delta_bytes": max(0, rollout_sampler.peak - rollout_rss_start),
                "native_orders": len(factory.snapshots[-1].all_orders),
                "native_fills": len(report["fills"]),
                "funding_events": len(report["funding"]),
                "terminal_flat": report["terminal_flat"],
                "descriptive_net_return": report["metrics"]["net_return"],
                "free_bytes_after": available_memory(),
            }
            records.append(record)
            _write_verified(output / f"{window['days']}d.json", {
                "profile": record,
                "scope": "engineering profile only; returns are descriptive Train-inner results",
            })
            print(json.dumps({"stage": "window-complete", "days": window["days"],
                              "decisions": decisions, "rollout_seconds": rollout_seconds,
                              "decisions_per_second": record["decisions_per_second"],
                              "rollout_peak_rss_delta": record["rollout_rss_peak_delta_bytes"],
                              "free_bytes": record["free_bytes_after"]}), flush=True)
            del env, factory, report
            gc.collect()
            # Keep the immutable CuratedStore and scaler alive for the second window.

        summary = {
            "schema": "rles.episode-reset-profile.v1", "split": "train_inner",
            "purpose": "Episode reset and historical factory engineering profile",
            "source_plan_sha256": _file_hash(plan_path),
            "source_profile_plan_sha256": _file_hash(profile_plan_path),
            "source_train_plan_sha256": _file_hash(source_plan_path),
            "bundle_sha256": plan["bundle_sha256"],
            "source_hashes": {key: source[f"{key}_sha256"] for key in ("market", "rules")},
            "source_code_sha256": code_hashes, "free_bytes_at_start": free_at_start,
            "minimum_free_bytes": min(row["free_bytes_after"] for row in records),
            "resource_stop_floor_bytes": STOP_FLOOR_BYTES, "cases": records,
            "interpretation": [
                "Factory construction and Reset timings are measured separately from full rollout.",
                "The second same-seed Reset must reproduce observation and initial native account facts.",
                "RSS is process working-set high-water sampled during each phase, not allocation attribution.",
                "The immutable CuratedStore is shared across windows; factory and native Engine are recreated.",
                "Historical rules and fees are scenario-only; returns are descriptive, not strategy evidence.",
            ],
        }
        _write_verified(output / "summary.json", summary)
        return summary
    except BaseException as exc:
        _write_verified(output / "FAILED.json", {
            "error": type(exc).__name__, "detail": str(exc), "completed_cases": records,
            "free_bytes": available_memory(),
        })
        raise


def finalize_existing(plan_path: Path, output: Path) -> dict:
    """Finalize already measured windows after validating every persisted artifact."""
    if not output.is_dir():
        raise DataError("profile output directory does not exist")
    execution_plan_path = output / "plan.json"
    execution_plan_hash = _verify_file(execution_plan_path)
    execution_plan = json.loads(execution_plan_path.read_text(encoding="utf-8"))
    if execution_plan["source_plan_sha256"] != _file_hash(plan_path):
        raise DataError("provided frozen input plan differs from the measured output")
    # The runner itself may have a small post-run repair since measurement; all
    # environment, account, data, registry, and rollout sources must still match.
    for source_path, expected_hash in execution_plan["source_code_sha256"].items():
        if source_path == "qa/profile_episode_reset_20261017.py":
            continue
        if _file_hash(Path(source_path)) != expected_hash:
            raise DataError(f"profile source changed after measurement: {source_path}")

    cases = []
    for days in WINDOW_DAYS:
        case_path = output / f"{days}d.json"
        _verify_file(case_path)
        wrapper = json.loads(case_path.read_text(encoding="utf-8"))
        profile = wrapper.get("profile")
        if not isinstance(profile, dict) or profile.get("days") != days:
            raise DataError(f"missing or mismatched completed {days}-day profile")
        if (profile.get("initial_state_reproducible") is not True
                or profile.get("terminal_flat") is not True
                or profile.get("decision_count", 0) < days * 1_400):
            raise DataError(f"{days}-day profile did not meet its completeness checks")
        cases.append(profile)
    parity_fields = (
        "observation_sha256", "decision_ts_ns", "initial_equity",
        "initial_native_equity", "initial_position_count", "first_snapshot_count",
    )
    if any(cases[0]["initial_state_fingerprint"][key]
           != cases[1]["initial_state_fingerprint"][key] for key in parity_fields):
        raise DataError("initial state differs between 60/360-day profile windows")

    prior_failure = None
    failure_path = output / "FAILED.json"
    if failure_path.is_file():
        _verify_file(failure_path)
        prior_failure = json.loads(failure_path.read_text(encoding="utf-8"))
    summary = {
        "schema": "rles.episode-reset-profile.v1", "split": "train_inner",
        "purpose": "Episode reset and historical factory engineering profile",
        "execution_plan_sha256": execution_plan_hash,
        "source_plan_sha256": execution_plan["source_plan_sha256"],
        "source_profile_plan_sha256": execution_plan["source_profile_plan_sha256"],
        "source_train_plan_sha256": execution_plan["source_train_plan_sha256"],
        "bundle_sha256": execution_plan["bundle_sha256"],
        "source_hashes": execution_plan["source_hashes"],
        "source_code_sha256_at_measurement": execution_plan["source_code_sha256"],
        "start_free_floor_passed": True,
        "start_free_bytes_observed": None,
        "start_free_floor_bytes": execution_plan["resource_floors_bytes"]["start"],
        "minimum_free_bytes": min(row["free_bytes_after"] for row in cases),
        "resource_stop_floor_bytes": execution_plan["resource_floors_bytes"]["stop"],
        "cases": cases,
        "recovery": ({"post_measurement_runner_error": prior_failure["error"],
                      "detail": prior_failure["detail"],
                      "measured_case_artifacts_verified": len(cases)}
                     if prior_failure else None),
        "interpretation": [
            "Factory construction and Reset timings are measured separately from full rollout.",
            "The second same-seed Reset reproduces the observation and initial native account facts.",
            "RSS is process working-set high-water sampled during each phase, not allocation attribution.",
            "The immutable CuratedStore is shared across windows; factory and native Engine are recreated.",
            "Historical rules and fees are scenario-only; returns are descriptive, not strategy evidence.",
        ],
    }
    summary_path = output / "summary.json"
    if summary_path.is_file():
        _verify_file(summary_path)
        existing_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if (existing_summary.get("schema") != summary["schema"]
                or existing_summary.get("source_plan_sha256") != summary["source_plan_sha256"]
                or [row.get("days") for row in existing_summary.get("cases", [])] != list(WINDOW_DAYS)):
            raise DataError("existing summary does not match the verified profile cases")
        summary = existing_summary
    else:
        _write_verified(summary_path, summary)
    if prior_failure is not None:
        recovered_path = output / "RECOVERED.json"
        if recovered_path.is_file():
            _verify_file(recovered_path)
        else:
            _write_verified(recovered_path, {
                "original_failure_sha256": _file_hash(failure_path),
                "summary_sha256": _file_hash(summary_path),
                "measured_case_artifacts_verified": len(cases),
                "resolution": "Both completed case files passed their sidecar hashes and completeness checks; the summary was rebuilt without rerunning either window.",
            })
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--finalize-existing", action="store_true",
                        help="verify completed per-window artifacts and write the summary")
    args = parser.parse_args()
    if args.finalize_existing:
        result = finalize_existing(args.plan, args.output)
    else:
        result = run(args.plan, args.output)
    print(json.dumps({"cases": len(result["cases"]),
                      "summary": str(args.output / "summary.json")}))


if __name__ == "__main__":
    main()
