"""Profile native replay growth on frozen Train history, without strategy selection.

Compares flat, persistent-long, and periodic-rebalance action paths over 1/30/90
days. Reports process RSS high-water sampled during factory construction and
rollout, elapsed time, event counts, and verified source identities. Historical
execution rules remain scenario assumptions; this is engineering evidence only.
"""
from __future__ import annotations

import argparse
import ctypes
from ctypes import wintypes
import gc
import json
from pathlib import Path
import threading
import time

import numpy as np

from rles_data.core import CuratedStore, DataError, MINUTE_NS
from rles_data.io import read_curated
from rles_gym_bridge.historical import HistoricalEnvironmentFactory, load_rules
from rles_orchestrator.application import _file_hash, _verify_file, _write_verified, rollout
from rles_orchestrator.research_access import record_development
from rles_orchestrator.wfo_application import _scaler


def available_memory() -> int:
    class MemoryStatus(ctypes.Structure):
        _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong),
                    *[(name, ctypes.c_ulonglong) for name in (
                        "total", "available", "total_page", "available_page",
                        "total_virtual", "available_virtual", "available_extended")]]
    status = MemoryStatus()
    status.length = ctypes.sizeof(status)
    if not getattr(ctypes, "windll").kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise OSError("physical memory status unavailable")
    return int(status.available)


def coverage_for_window(original: dict, window: dict) -> dict:
    coverage = json.loads(json.dumps(original))
    for key in ("settlement_times_ns", "known_schedule_times_ns"):
        if key in coverage:
            margin = 24 * 60 * MINUTE_NS if key == "known_schedule_times_ns" else 0
            coverage[key] = {symbol: [stamp for stamp in times
                if window["start_ns"] - margin <= stamp < window["end_ns"] + margin]
                for symbol, times in coverage[key].items()}
    return coverage


DAY_NS = 86_400 * 1_000_000_000


def process_rss() -> int:
    """Return current process working set (RSS) on Windows."""
    class Counters(ctypes.Structure):
        _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong),
                    ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
    counters = Counters()
    counters.cb = ctypes.sizeof(counters)
    kernel32 = getattr(ctypes, "WinDLL")("kernel32", use_last_error=True)
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    psapi = getattr(ctypes, "WinDLL")("psapi", use_last_error=True)
    psapi.GetProcessMemoryInfo.argtypes = (wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD)
    psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
    if not psapi.GetProcessMemoryInfo(kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
        raise OSError("GetProcessMemoryInfo failed")
    return int(counters.WorkingSetSize)


class RssSampler:
    def __init__(self) -> None:
        self.peak = process_rss()
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self) -> None:
        while not self.stop.wait(0.05):
            self.peak = max(self.peak, process_rss())

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_exc):
        self.stop.set()
        self.thread.join()
        self.peak = max(self.peak, process_rss())


def make_predictor(workload: str, rows):
    cursor = 0

    def predict(batch):
        nonlocal cursor
        if len(batch) != 1 or cursor >= len(rows):
            raise DataError("profile requires one deterministic account and bounded rows")
        if workload == "flat":
            target = 0.0
        elif workload == "long":
            target = 0.25
        else:
            # A meaningful but bounded inventory change every six hours.
            target = 0.25 if (cursor // 360) % 2 == 0 else 0.50
        cursor += 1
        return np.array([[target]], dtype=np.float32)

    return predict


def run(plan_path: Path, output: Path) -> dict:
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if plan.get("split") != "train_inner" or output.exists():
        raise DataError("requires the frozen Train plan and a new output directory")
    bundle_root = Path(plan["bundle"])
    if _verify_file(bundle_root / "bundle.json") != plan["bundle_sha256"]:
        raise DataError("frozen candidate bundle checksum changed")
    bundle = json.loads((bundle_root / "bundle.json").read_text(encoding="utf-8"))
    source = plan["source"]
    for key in ("market", "rules"):
        if _file_hash(Path(source[key + "_path"])) != source[key + "_sha256"]:
            raise DataError(f"frozen {key} source checksum changed")
    base = plan["windows"][0]["start_ns"]
    windows = [{"start_ns": base, "end_ns": base + days * DAY_NS, "days": days}
               for days in (1, 30, 90)]
    if any(window["end_ns"] > bundle["training_end_ns"] for window in windows):
        raise DataError("long profile window exceeds frozen Train boundary")
    if available_memory() < plan["minimum_start_free_bytes"]:
        raise MemoryError("free memory below frozen plan start floor")
    environment = bundle["environment"]
    registry_config = {"environment": environment, "output": str(output), "seed": plan["seed"]}
    for window in windows:
        record_development(registry_config, window, label="f7-native-history-growth")

    output.mkdir(parents=True)
    _write_verified(output / "plan.json", {
        "schema": "rles.native-history-growth-plan.v1",
        "source_plan": str(plan_path.resolve()), "source_plan_sha256": _file_hash(plan_path),
        "bundle_sha256": plan["bundle_sha256"], "source": source,
        "split": "train_inner", "windows_days": [1, 30, 90],
        "workloads": {"flat": 0.0, "long": 0.25,
                      "rebalance": "alternate target 0.25/0.50 every 360 decision steps"},
        "purpose": "engineering performance profile only; no model selection or return gate",
        "historical_economics": "scenario assumptions; not verified historical trading rules",
    })
    records = []
    try:
        started = time.perf_counter()
        rules = load_rules(source["rules_path"])
        store = CuratedStore((*read_curated(source["market_path"]).events, *rules))
        if store.snapshot_hash != bundle["preprocessing"]["identities"]["dataset"]:
            raise DataError("archived source does not match frozen model data identity")
        source_seconds = time.perf_counter() - started
        scaler = _scaler(bundle_root)
        print(json.dumps({"stage": "sources-verified", "seconds": source_seconds,
                          "events": len(store.events), "free_bytes": available_memory(),
                          "rss_bytes": process_rss()}), flush=True)

        for window in windows:
            coverage = coverage_for_window(environment["coverage_policy"], window)
            for workload in ("flat", "long", "rebalance"):
                if available_memory() < plan["stop_free_bytes"]:
                    raise MemoryError("resource stop floor reached before next profile case")
                gc.collect()
                begin_rss = process_rss()
                start = time.perf_counter()
                with RssSampler() as sampler:
                    factory = HistoricalEnvironmentFactory(**{
                        **environment, **{k: window[k] for k in ("start_ns", "end_ns")},
                        "coverage_policy": coverage, "split": "train_inner", "mode": "base",
                        "scaler": scaler, "fit_scaler": False,
                        "reward_scale": bundle.get("reward_scale", 1.0),
                        "compact_report_snapshots": True, "prepared_store": store,
                    })
                    factory_seconds = time.perf_counter() - start
                    rollout_start = time.perf_counter()
                    report, _ = rollout(factory, make_predictor(workload, factory.rows),
                                        seed=plan["seed"],
                                        soft_parameters=bundle["soft_multipliers"],
                                        collect_observations=False, include_account_rows=False)
                    rollout_seconds = time.perf_counter() - rollout_start
                    peak_rss = sampler.peak
                snapshots = factory.snapshots
                fill_count = sum(len(snapshot.fills) for snapshot in snapshots)
                funding_count = sum(len(snapshot.funding) for snapshot in snapshots)
                record = {
                    "days": window["days"], "workload": workload,
                    "decision_steps": report["action_diagnostics"]["count"],
                    "snapshot_count": len(snapshots), "fill_facts": fill_count,
                    "funding_facts": funding_count,
                    "factory_seconds": factory_seconds, "rollout_seconds": rollout_seconds,
                    "total_seconds": time.perf_counter() - start,
                    "rss_start_bytes": begin_rss, "rss_peak_bytes": peak_rss,
                    "rss_peak_delta_bytes": max(0, peak_rss - begin_rss),
                    "rss_end_bytes": process_rss(), "free_bytes_after": available_memory(),
                    "terminal_flat": report["terminal_flat"],
                    "net_return_descriptive_only": report["metrics"]["net_return"],
                }
                records.append(record)
                case = output / f"{window['days']}d-{workload}.json"
                _write_verified(case, {"profile": record, "scope": "engineering only"})
                del factory, snapshots, report
                gc.collect()
                print(json.dumps({"case": f"{window['days']}d-{workload}",
                                  "steps": record["decision_steps"],
                                  "seconds": record["total_seconds"],
                                  "rss_peak_delta_bytes": record["rss_peak_delta_bytes"],
                                  "free_bytes": available_memory()}), flush=True)

        summary = {
            "schema": "rles.native-history-growth.v1", "split": "train_inner",
            "scope": "engineering performance only; not strategy evaluation",
            "historical_economics": "scenario assumptions; validity remains unproven",
            "source_seconds": source_seconds, "source_event_count": len(store.events),
            "source_snapshot_hash": store.snapshot_hash,
            "bundle_sha256": plan["bundle_sha256"], "source_hashes": {
                key: source[key + "_sha256"] for key in ("market", "rules")},
            "cases": records,
            "limitations": [
                "RSS is sampled current process working-set peak, not isolated allocation attribution.",
                "Each case creates a new factory but reuses the verified immutable in-process store.",
                "Factory snapshots and exact risk/report traces remain material memory consumers.",
                "Historical execution rules remain scenario-only; returns are descriptive and not a gate.",
            ],
        }
        _write_verified(output / "summary.json", summary)
        return summary
    except BaseException as exc:
        _write_verified(output / "FAILED.json", {"error": type(exc).__name__, "detail": str(exc),
                                                   "completed_cases": records})
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.plan, args.output)
    print(json.dumps({"cases": len(result["cases"]), "summary": str(args.output / "summary.json")}))


if __name__ == "__main__":
    main()
