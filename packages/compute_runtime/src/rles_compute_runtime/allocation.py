"""Attempt-scoped combined CPU/RAM/GPU quota and exclusive CUDA ownership.

This is a local run-time reservation, not an OS cgroup/Job Object limit. The
integrator must still enforce real process resource ceilings.
"""
from __future__ import annotations

from pathlib import Path

from .errors import ComputeError
from .resources import GPULearnerLease, Lease, ResourceQuota


class ComputeAllocation:
    """Acquire the quota before GPU ownership; unwind safely on contention.

    No automatic stale-lock breaking. Failure to relinquish the GPU ownership
    prevents releasing its quota, so another learner cannot silently supersede
    an unknown CUDA context.
    """

    def __init__(self, *, quota: ResourceQuota, owner: str, attempt_id: str,
                 cpu_cores: int, ram_bytes: int, learner_device: str,
                 gpu_lock_directory: str | Path | None = None, gpu_index: int = 0):
        if not isinstance(quota,ResourceQuota) or not owner or not attempt_id:
            raise ValueError('trusted quota, owner and attempt ID required')
        if learner_device not in {'cpu','cuda'}:
            raise ValueError('learner device must be the resolved cpu/cuda value')
        if type(cpu_cores) is not int or cpu_cores<1 or type(ram_bytes) is not int or ram_bytes<1:
            raise ValueError('positive CPU/RAM quota required')
        if learner_device=='cuda' and gpu_lock_directory is None:
            raise ValueError('CUDA requires a durable GPU lock directory')
        self.quota=quota
        self.owner=owner
        self.attempt_id=attempt_id
        self.cpu_cores=cpu_cores
        self.ram_bytes=ram_bytes
        self.learner_device=learner_device
        self.gpu_lock=(GPULearnerLease(gpu_lock_directory,gpu_index)
                       if learner_device=='cuda' and gpu_lock_directory is not None else None)
        self.lease: Lease | None = None
        self._used=False

    def acquire(self) -> Lease:
        if self._used:
            raise ComputeError('ComputeAllocation is single-use within an attempt')
        self._used=True
        reservation=self.quota.acquire(self.owner,cpu_cores=self.cpu_cores,
                                      ram_bytes=self.ram_bytes,gpus=int(self.gpu_lock is not None))
        self.lease=reservation
        if self.gpu_lock:
            try:
                self.gpu_lock.acquire(attempt_id=self.attempt_id)
            except BaseException:
                # Exclusivity acquisition failed; quota can safely be released
                # because no CUDA context could have been admitted by this owner.
                self.quota.release(reservation)
                self.lease=None
                raise
        return reservation

    def release(self) -> None:
        if self.lease is None:
            raise ComputeError('allocation is not active')
        if self.gpu_lock:
            self.gpu_lock.release()  # May raise; do NOT give its quota to somebody else.
        self.quota.release(self.lease)
        self.lease=None

    def __enter__(self) -> ComputeAllocation:
        self.acquire()
        return self

    def __exit__(self, *_:object) -> None:
        self.release()
