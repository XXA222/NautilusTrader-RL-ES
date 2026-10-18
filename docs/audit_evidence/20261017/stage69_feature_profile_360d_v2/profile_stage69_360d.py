from __future__ import annotations
import ctypes
from ctypes import wintypes
from hashlib import sha256
import json
import os
from pathlib import Path
import struct
import threading
import time

from rles_data.core import CuratedStore
from rles_orchestrator.application import load_config
from rles_orchestrator.research_access import read_registered_train_window, record_development
from rles_gym_bridge.historical import HistoricalEnvironmentFactory
from rles_orchestrator.application import load_rules

class _MemoryStatus(ctypes.Structure):
    _fields_ = [('length', ctypes.c_ulong), ('memory_load', ctypes.c_ulong),
        ('total_phys', ctypes.c_ulonglong), ('avail_phys', ctypes.c_ulonglong),
        ('total_page_file', ctypes.c_ulonglong), ('avail_page_file', ctypes.c_ulonglong),
        ('total_virtual', ctypes.c_ulonglong), ('avail_virtual', ctypes.c_ulonglong),
        ('avail_extended_virtual', ctypes.c_ulonglong)]
class _ProcessMemoryCounters(ctypes.Structure):
    _fields_ = [('cb', ctypes.c_ulong), ('page_fault_count', ctypes.c_ulong),
        ('peak_working_set_size', ctypes.c_size_t), ('working_set_size', ctypes.c_size_t),
        ('quota_peak_paged_pool_usage', ctypes.c_size_t), ('paged_pool_usage', ctypes.c_size_t),
        ('quota_peak_non_paged_pool_usage', ctypes.c_size_t), ('non_paged_pool_usage', ctypes.c_size_t),
        ('pagefile_usage', ctypes.c_size_t), ('peak_pagefile_usage', ctypes.c_size_t)]
def available_memory() -> int:
    status = _MemoryStatus(); status.length = ctypes.sizeof(status)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise OSError('GlobalMemoryStatusEx failed')
    return int(status.avail_phys)
def process_memory() -> _ProcessMemoryCounters:
    counters = _ProcessMemoryCounters(); counters.cb = ctypes.sizeof(counters)
    kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
    psapi = ctypes.WinDLL('psapi', use_last_error=True)
    kernel32.GetCurrentProcess.argtypes = []; kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(_ProcessMemoryCounters), wintypes.DWORD]
    psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
    if not psapi.GetProcessMemoryInfo(kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
        raise ctypes.WinError(ctypes.get_last_error())
    return counters
def sha_file(path: Path) -> str:
    h = sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''): h.update(block)
    return h.hexdigest()
def main() -> None:
    out = Path(__file__).resolve().parent
    prereg_bytes = (out/'preregistration.json').read_bytes()
    prereg_sha = sha256(prereg_bytes).hexdigest()
    if (out/'preregistration.json.sha256').read_text(encoding='ascii').strip() != prereg_sha:
        raise SystemExit('preregistration receipt mismatch')
    prereg = json.loads(prereg_bytes)
    config_path = out/'config.json'; config_bytes = config_path.read_bytes()
    if sha256(config_bytes).hexdigest() != prereg['config_sha256']:
        raise SystemExit('frozen config changed')
    config = load_config(config_path)
    for relative, expected in prereg['code_sha256'].items():
        if sha_file(Path(relative)) != expected: raise SystemExit(f'frozen code changed: {relative}')
    if sha_file(Path(config['environment']['rules_path'])) != prereg['rules_sha256']:
        raise SystemExit('frozen rules source changed')
    for source in prereg['sources']:
        p = Path(source['path']); marker = p.with_name(p.name+'.COMPLETED.json')
        if sha_file(marker) != source['marker_sha256']:
            raise SystemExit(f"marker changed for {source['name']}")
        if sha_file(p.parent/'import.json') != source['import_manifest_sha256']:
            raise SystemExit(f"source manifest changed for {source['name']}")
        md = json.loads(marker.read_text(encoding='utf-8'))
        if md.get('file_sha256') != source['marker_file_sha256'] or md.get('snapshot_hash') != source['marker_snapshot_hash']:
            raise SystemExit(f"market marker identity changed for {source['name']}")
    stop_line = prereg['stop_line']['minimum_system_available_memory_bytes']
    initial_available = available_memory()
    if initial_available <= stop_line: raise SystemExit(f'memory stop line active before payload read: {initial_available}')
    peak_rss = int(process_memory().peak_working_set_size); min_available = initial_available; done = threading.Event()
    def monitor() -> None:
        nonlocal peak_rss, min_available
        while not done.wait(0.05):
            peak_rss = max(peak_rss, int(process_memory().peak_working_set_size))
            free = available_memory(); min_available = min(min_available, free)
            if free <= stop_line: os._exit(78)
    watcher = threading.Thread(target=monitor, daemon=True); watcher.start()
    # Persist both exact source reservations before opening either Parquet payload.
    scoped = []
    for source in prereg['sources']:
        one = json.loads(json.dumps(config))
        one['environment']['market_path'] = source['path']
        one['environment']['start_ns'] = max(prereg['window']['start_ns'], source['scope']['start_ns'])
        one['environment']['end_ns'] = min(prereg['window']['end_ns'], source['scope']['end_ns'])
        if one['environment']['start_ns'] >= one['environment']['end_ns']:
            raise SystemExit(f"empty Train intersection for {source['name']}")
        scope_config = {**one, 'environment': {**one['environment'],
            'curated_marker_sha256': source['marker_sha256'],
            'curated_read_mode': source['source_mode']}}
        record_development(scope_config, source['scope'], label=source['label'])
        scoped.append((one, source))
    started = time.perf_counter()
    stores = []
    for one, source in scoped:
        stores.append(read_registered_train_window(
            one, source['scope'], marker_sha256=source['marker_sha256'],
            label=source['label'], allow_legacy_isolated_snapshot=True))
    market_events = tuple(event for store in stores for event in store.events)
    rules = tuple(load_rules(config['environment']['rules_path']))
    merged = CuratedStore((*market_events, *rules))
    env = config['environment']
    factory = HistoricalEnvironmentFactory(
        **{**env, 'split':'train_inner', 'fit_scaler':True, 'prepared_store':merged})
    elapsed = time.perf_counter() - started
    feature_hash = sha256()
    for row in factory.rows:
        feature_hash.update(struct.pack('>qq', row.decision_ts_ns, row.available_at_ns))
        feature_hash.update(struct.pack(f'>{len(row.values)}d', *row.values))
    done.set(); watcher.join(timeout=1)
    after = process_memory(); peak_rss=max(peak_rss, int(after.peak_working_set_size)); min_available=min(min_available,available_memory())
    result = {
      'schema':'rles.stage69.train-feature-profile-result.v1','status':'PASS',
      'config_sha256':prereg['config_sha256'],'preregistration_sha256':prereg_sha,
      'window':prereg['window'],'days':360,'source_events_loaded':len(market_events),
      'merged_events_with_rules':len(merged.events),'feature_rows':len(factory.rows),
      'feature_matrix_sha256':feature_hash.hexdigest(),'feature_schema_hash':factory.schema.hash,
      'prepare_and_feature_build_wall_seconds':elapsed,
      'process_peak_rss_bytes':peak_rss,'process_working_set_after_bytes':int(after.working_set_size),
      'system_available_memory_initial_bytes':initial_available,
      'system_available_memory_min_bytes':min_available,'minimum_available_stop_line_bytes':stop_line,
      'snapshot_hash':merged.snapshot_hash,'data_quality_grade':factory.quality.grade.value,
      'instrument_rule_provenance':factory.rule.provenance,'formal_evaluation_queries':0,
      'source_access_labels':[s['label'] for s in prereg['sources']],
    }
    raw = json.dumps(result, sort_keys=True, indent=2).encode()+b'\n'
    (out/'profile-result.json').write_bytes(raw)
    (out/'profile-result.json.sha256').write_text(sha256(raw).hexdigest(),encoding='ascii')
    print(json.dumps(result,indent=2))
if __name__=='__main__': main()
