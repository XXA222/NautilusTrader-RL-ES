"""Fault policy: fail-fast or explicit approved new-attempt CPU retry."""
from __future__ import annotations

from dataclasses import dataclass
from .errors import ComputeError


@dataclass(frozen=True)
class FaultDecision:
    action: str
    new_attempt_required: bool
    comparability_warning: str | None


def decide_fault(fault: str, *, fallback_policy: str, reviewer_approved: bool = False) -> FaultDecision:
    if fault not in {'worker_crash','worker_timeout','cuda_oom','cuda_failure','disk_full'}:
        raise ValueError('unrecognized fault')
    if fallback_policy not in {'fail_fast','approved_cpu_retry'}:
        raise ValueError('unknown fallback policy')
    if fallback_policy == 'approved_cpu_retry' and reviewer_approved and fault in {'cuda_oom','cuda_failure'}:
        return FaultDecision('NEW_CPU_ATTEMPT',True,
            'Device changed; new attempt_id, seed tree and full accounting required; no in-place resume')
    return FaultDecision('ABORT',False,None)


class FaultInjector:
    """Explicit, nonproduction acceptance hooks. Never enabled for live engines."""

    def __init__(self, *, research_sandbox: bool, reviewer_approved: bool):
        if not research_sandbox or not reviewer_approved:
            raise ComputeError('fault injection requires approved research sandbox')

    def terminate_worker(self, pool, index: int) -> None:
        """Inject actual worker loss; next dispatch must abort the run_attempt."""
        if not pool._processes or pool._closed or not (0 <= index < pool.workers):
            raise ComputeError('cannot inject into inactive/unallocated worker')
        worker=pool._processes[index]
        worker.terminate()
        worker.join(timeout=3)
        if worker.is_alive():
            worker.kill();worker.join(timeout=3)

    @staticmethod
    def inject_cuda_oom_boundary() -> None:
        """Exercise policy handling only; NOT a true CUDA allocator OOM test."""
        raise ComputeError('INJECTED_CUDA_OOM_BOUNDARY')
