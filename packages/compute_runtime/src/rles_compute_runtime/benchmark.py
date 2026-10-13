"""Replicated, comparable end-to-end device benchmarking; never use GPU utilisation."""
from __future__ import annotations

import hashlib
import json
import statistics
from collections.abc import Callable, Mapping
from typing import TypeAlias, cast

from .errors import ComputeError
from .validation import finite_real

_FIELDS = ('wall_sec','p95_action_latency_ms','peak_rss_bytes','peak_vram_bytes',
           'pcie_txrx_bytes','numerical_max_abs_error','env_steps','es_evaluations','model_params')
Number: TypeAlias = int | float
Sample: TypeAlias = dict[str, Number]
_COUNTERS = frozenset({'env_steps', 'es_evaluations', 'model_params'})


def _sample(value: object) -> Sample:
    if not isinstance(value, Mapping) or set(value) != set(_FIELDS):
        raise ComputeError(f'benchmark evidence requires exactly {_FIELDS}')
    parsed: Sample = {}
    for field in _FIELDS:
        raw = value[field]
        if not finite_real(raw) or raw < 0:
            raise ComputeError('non-finite/invalid benchmark reading')
        if field in _COUNTERS and type(raw) is not int:
            raise ComputeError(f'{field} must be an integer counter')
        parsed[field] = raw
    if parsed['wall_sec'] <= 0 or parsed['env_steps'] <= 0 or parsed['model_params'] <= 0:
        raise ComputeError('benchmark must include real steps, model params and elapsed time')
    return parsed


def benchmark(modes: list[str], run_once: Callable[[str, int], Mapping[str, object]], *,
              frozen_fingerprints: Mapping[str, str], repetitions: int = 3) -> dict[str, object]:
    """Caller provides actual end-to-end *train/evaluate* probes, not synthetic timings."""
    expected_fingerprints={'dataset_sha256','seed_tree_sha256','actor_signature_sha256','execution_sha256','risk_sha256','algorithm_config_sha256'}
    if set(frozen_fingerprints)!=expected_fingerprints or any(not isinstance(v,str) or len(v)!=64 or any(c not in '0123456789abcdef' for c in v) for v in frozen_fingerprints.values()):
        raise ComputeError('same frozen dataset/seed/actor/execution/risk/algorithm fingerprints required')
    if (type(repetitions) is not int or repetitions < 3 or not isinstance(modes, list)
        or not modes or any(not isinstance(mode, str) for mode in modes)
        or len(modes) != len(set(modes)) or 'cpu' not in modes):
        raise ValueError('CPU baseline and at least 3 independent repeats are required')
    if set(modes) - {'cpu','gpu','hybrid'}:
        raise ValueError('invalid topology')
    out: dict[str, list[Sample]] = {}
    for mode in modes:
        samples: list[Sample] = []
        for trial in range(repetitions):
            samples.append(_sample(run_once(mode, trial)))
        out[mode] = samples
    budgets = ('env_steps','es_evaluations','model_params')
    reference = tuple(out['cpu'][0][k] for k in budgets)
    if any(tuple(s[k] for k in budgets) != reference for samples in out.values() for s in samples):
        raise ComputeError('benchmark budgets/model parameters differ; comparison forbidden')
    summary = {mode:{'median_wall_sec':statistics.median(x['wall_sec'] for x in samples),
                    'max_p95_latency_ms':max(x['p95_action_latency_ms'] for x in samples),
                    'peak_rss_bytes':max(x['peak_rss_bytes'] for x in samples),
                    'peak_vram_bytes':max(x['peak_vram_bytes'] for x in samples),
                    'median_pcie_bytes':statistics.median(x['pcie_txrx_bytes'] for x in samples),
                    'max_numerical_error':max(x['numerical_max_abs_error'] for x in samples)}
               for mode,samples in out.items()}
    report={'schema_version':1,'frozen_fingerprints':dict(frozen_fingerprints),'repetitions':repetitions,'sample_budget':dict(zip(budgets,reference)),
            'samples':out,'summary':summary}
    validate_benchmark_report(report)
    return report


def validate_benchmark_report(report: Mapping[str, object]) -> str:
    """Independently recompute all device benchmark aggregation and its digest.

    The probe remains an externally attested real workload; this protects
    against stale or inconsistent report summaries, not fabricated measurements.
    """
    required={'schema_version','frozen_fingerprints','repetitions','sample_budget','samples','summary'}
    if (not isinstance(report,Mapping) or set(report)!=required
        or type(report['schema_version']) is not int or report['schema_version']!=1):
        raise ComputeError('device benchmark schema drift')
    fingerprints=report['frozen_fingerprints']
    required_fps={'dataset_sha256','seed_tree_sha256','actor_signature_sha256',
                  'execution_sha256','risk_sha256','algorithm_config_sha256'}
    if (not isinstance(fingerprints,Mapping) or set(fingerprints)!=required_fps
        or any(not isinstance(x,str) or len(x)!=64 or any(c not in '0123456789abcdef' for c in x)
               for x in fingerprints.values())):
        raise ComputeError('incomplete frozen benchmark fingerprints')
    count=report['repetitions']
    if type(count) is not int or count<3 or not isinstance(report['samples'],Mapping) or 'cpu' not in report['samples']:
        raise ComputeError('benchmark lacks complete CPU baseline')
    if not set(report['samples']).issubset({'cpu','gpu','hybrid'}):
        raise ComputeError('unknown compute topology')
    raw_samples=report['samples']
    samples: dict[str, list[Sample]] = {}
    for mode, group in raw_samples.items():
        if not isinstance(mode, str) or not isinstance(group, list) or len(group) != count:
            raise ComputeError('unbalanced repeated benchmark')
        samples[mode] = [_sample(item) for item in group]
    budgets={'env_steps','es_evaluations','model_params'}
    if not isinstance(report['sample_budget'],Mapping) or set(report['sample_budget'])!=budgets:
        raise ComputeError('sample budgets missing')
    sample_budget = report['sample_budget']
    numeric_budget: dict[str, Number] = {}
    for key in budgets:
        raw = sample_budget[key]
        if type(raw) is not int or not finite_real(raw) or raw < 0:
            raise ComputeError('invalid benchmark sample budget')
        numeric_budget[key] = raw
    summary={}
    for mode, group in samples.items():
        for sample in group:
            if any(sample[k] != numeric_budget[k] for k in budgets):
                raise ComputeError('incomparable benchmark run/sampling budget')
        summary[mode]={'median_wall_sec':statistics.median(x['wall_sec'] for x in group),
                       'max_p95_latency_ms':max(x['p95_action_latency_ms'] for x in group),
                       'peak_rss_bytes':max(x['peak_rss_bytes'] for x in group),
                       'peak_vram_bytes':max(x['peak_vram_bytes'] for x in group),
                       'median_pcie_bytes':statistics.median(x['pcie_txrx_bytes'] for x in group),
                       'max_numerical_error':max(x['numerical_max_abs_error'] for x in group)}
    declared_summary = report['summary']
    if not isinstance(declared_summary, Mapping) or set(declared_summary) != set(summary):
        raise ComputeError('benchmark summary does not agree with immutable samples')
    for mode, expected in summary.items():
        declared = declared_summary[mode]
        if (not isinstance(declared, Mapping) or set(declared) != set(expected)
            or any(not finite_real(value) for value in declared.values())
            or declared != expected):
            raise ComputeError('benchmark summary does not agree with immutable samples')
    try:
        encoded = json.dumps(report,sort_keys=True,allow_nan=False,separators=(',',':')).encode()
    except (TypeError, ValueError, OverflowError) as exc:
        raise ComputeError('benchmark report must be JSON serializable') from exc
    return hashlib.sha256(encoded).hexdigest()


def device_decision(report: Mapping[str, object], *, max_p95_latency_ms: float,
                    max_numeric_error: float, min_speedup_fraction: float = 0.01,
                    reviewer_approved: bool=False) -> dict[str, str]:
    """GPU/hybrid wins only with a measured meaningful E2E speedup and bounded error."""
    if reviewer_approved is not True:
        raise ComputeError('auto-device benchmark decision requires explicit human approval')
    if (any(not finite_real(x)
            for x in (max_p95_latency_ms,max_numeric_error,min_speedup_fraction))
        or not (0 <= min_speedup_fraction < 1 and max_p95_latency_ms > 0 and max_numeric_error >= 0)):
        raise ValueError('invalid predeclared acceptance threshold')
    digest=validate_benchmark_report(report)
    s = cast(Mapping[str, Mapping[str, Number]], report['summary'])
    cpu = s['cpu']['median_wall_sec']
    acceptable = [m for m in ('gpu','hybrid') if m in s and
                  s[m]['median_wall_sec'] < cpu*(1-min_speedup_fraction) and
                  s[m]['max_p95_latency_ms'] <= max_p95_latency_ms and
                  s[m]['max_numerical_error'] <= max_numeric_error]
    selected = min(acceptable, key=lambda m:s[m]['median_wall_sec']) if acceptable else 'cpu'
    frozen_fingerprints = report['frozen_fingerprints']
    if not isinstance(frozen_fingerprints, Mapping):
        raise ComputeError('benchmark fingerprints missing')
    frozen=json.dumps(dict(frozen_fingerprints),sort_keys=True,separators=(',',':')).encode()
    return {'status':'APPROVED','selected_topology':selected,
            'fingerprints_sha256':hashlib.sha256(frozen).hexdigest(),
            'benchmark_sha256':digest,
            'decision_rule':'min_e2e_median_with_latency_and_parity_bounds_v1'}


def worker_scaling_benchmark(run_once: Callable[[int,int],Mapping[str,object]], *,
                             frozen_fingerprints: Mapping[str,str],
                             workers: tuple[int,...]=(1,2,4), repetitions:int=3,
                             max_peak_rss_bytes:int) -> dict:
    """Compare 1/2/4 independent CPU Engines at identical sample budget.

    The caller supplies an actual end-to-end CPU-engine probe for each worker count.
    """
    if (type(workers) is not tuple or workers!=(1,2,4)
        or any(type(worker) is not int for worker in workers)
        or type(max_peak_rss_bytes) is not int or not finite_real(max_peak_rss_bytes)
        or max_peak_rss_bytes<=0):
        raise ComputeError('frozen worker-scaling protocol requires 1/2/4 and explicit RSS cap')
    if set(frozen_fingerprints)!={'dataset_sha256','seed_tree_sha256','actor_signature_sha256',
                                 'execution_sha256','risk_sha256','algorithm_config_sha256'}:
        raise ComputeError('worker scale probe requires frozen experiment fingerprints')
    for fingerprint in frozen_fingerprints.values():
        if not isinstance(fingerprint,str) or len(fingerprint)!=64 or any(ch not in '0123456789abcdef' for ch in fingerprint):
            raise ComputeError('invalid experiment fingerprint')
    if type(repetitions) is not int or repetitions<3:
        raise ComputeError('at least 3 repeats per worker-count')
    results: dict[int, list[dict[str, Number]]] = {}
    budgets=None
    for count in workers:
        samples: list[dict[str, Number]] = []
        for trial in range(repetitions):
            raw_sample = dict(run_once(count,trial))
            if set(raw_sample)!={'wall_sec','peak_rss_bytes','env_steps','model_params'}:
                raise ComputeError('incomplete/invalid scaling evidence')
            v: dict[str, Number] = {}
            for key, item in raw_sample.items():
                if not finite_real(item) or item <= 0:
                    raise ComputeError('incomplete/invalid scaling evidence')
                if key in _COUNTERS and type(item) is not int:
                    raise ComputeError(f'{key} must be an integer counter')
                v[key] = item
            if v['peak_rss_bytes']>max_peak_rss_bytes:
                raise ComputeError('worker scaling exceeded declared RSS ceiling')
            unit=(v['env_steps'],v['model_params'])
            if budgets is None: budgets=unit
            if unit!=budgets:raise ComputeError('worker-scaling budget/model drift')
            samples.append(v)
        results[count]=samples
    median={count:statistics.median(s['wall_sec'] for s in samples) for count,samples in results.items()}
    selected=min(workers,key=lambda c:(median[c],c))
    return {'frozen_fingerprints':dict(frozen_fingerprints),'repetitions':repetitions,
            'workers':list(workers),'samples':results,'median_wall_sec':median,
            'selected_cpu_workers':selected,'max_peak_rss_bytes':max_peak_rss_bytes}
