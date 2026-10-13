"""Resolve a run's device exactly once from audited benchmark evidence."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Mapping

from .errors import ComputeError, DeviceUnavailable


@dataclass(frozen=True, slots=True)
class ResolvedDevice:
    requested: str
    topology: str
    learner_device: str
    engine_device: str = 'cpu'
    benchmark_digest: str | None = None


class DeviceResolver:
    """A single-use resolver; a new attempt must use a new instance."""

    def __init__(self) -> None:
        self._resolved: ResolvedDevice | None = None
        self._attempted = False

    def resolve(self, requested: str, *, decision: Mapping[str, object] | None = None,
                current_fingerprints: Mapping[str, str] | None = None,
                benchmark_report: Mapping[str, object] | None = None) -> ResolvedDevice:
        if self._attempted:
            raise ComputeError('DeviceResolver may only attempt resolution once per run_attempt')
        # Even an unsuccessful CUDA probe must terminate this attempt; never
        # allow a catch-and-switch to CPU without a new attempt identity.
        self._attempted = True
        if requested not in {'cpu', 'gpu', 'hybrid', 'auto'}:
            raise ValueError('topology must be cpu, gpu, hybrid or auto')
        if requested == 'auto':
            if decision is None or decision.get('status') != 'APPROVED':
                raise DeviceUnavailable('auto requires an APPROVED end-to-end benchmark decision')
            topology = decision.get('selected_topology')
            digest = decision.get('benchmark_sha256')
            if topology not in {'cpu', 'gpu', 'hybrid'} or not isinstance(digest, str) or len(digest) != 64:
                raise DeviceUnavailable('malformed auto device decision')
            if any(c not in '0123456789abcdef' for c in digest):
                raise DeviceUnavailable('benchmark digest must be lowercase SHA256')
            if current_fingerprints is None or benchmark_report is None:
                raise DeviceUnavailable('auto requires the original benchmark report and run fingerprints')
            from .benchmark import validate_benchmark_report
            if validate_benchmark_report(benchmark_report) != digest:
                raise DeviceUnavailable('auto decision does not match benchmark report bytes')
            report_fingerprints = benchmark_report['frozen_fingerprints']
            if not isinstance(report_fingerprints, Mapping):
                raise DeviceUnavailable('benchmark report fingerprints missing')
            if dict(report_fingerprints) != dict(current_fingerprints):
                raise DeviceUnavailable('benchmark report fingerprint mismatch')
            report_summary = benchmark_report['summary']
            if not isinstance(report_summary, Mapping) or topology not in report_summary:
                raise DeviceUnavailable('selected topology absent from report')
            fp=hashlib.sha256(json.dumps(dict(current_fingerprints),sort_keys=True,separators=(',',':')).encode()).hexdigest()
            if fp!=decision.get('fingerprints_sha256'):
                raise DeviceUnavailable('auto device decision belongs to a different experiment')
        else:
            topology, digest = requested, None
        learner = 'cpu' if topology == 'cpu' else 'cuda'
        if learner == 'cuda':
            try:
                import torch
                if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
                    raise DeviceUnavailable('CUDA not available: refusing CPU fallback')
            except ImportError as exc:
                raise DeviceUnavailable('CUDA mode requires external torch dependency') from exc
        self._resolved = ResolvedDevice(requested, topology, learner, benchmark_digest=digest)
        return self._resolved

    @property
    def resolved(self) -> ResolvedDevice:
        if self._resolved is None:
            raise ComputeError('device has not been resolved')
        return self._resolved
