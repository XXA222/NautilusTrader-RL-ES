from __future__ import annotations

import json
import multiprocessing as mp
import time

import pytest
import torch

from rles_compute_runtime.allocation import ComputeAllocation
from rles_compute_runtime.device import DeviceResolver
from rles_compute_runtime.errors import ComputeError, DeviceUnavailable, QuotaExceeded
from rles_compute_runtime.resources import GPULearnerLease, ResourceQuota
from rles_data.core import DataError
import rles_orchestrator.application as application
from rles_orchestrator.application import load_config


pytestmark = pytest.mark.unit


def _hold_gpu_lease(directory: str, ready) -> None:
    lease = GPULearnerLease(directory, 0)
    lease.acquire(attempt_id='hard-exit-owner')
    ready.send(True)
    while True:
        time.sleep(1)


def test_cpu_device_resolution_is_concrete_and_single_use() -> None:
    resolver = DeviceResolver()

    resolved = resolver.resolve("cpu")

    assert resolved.learner_device == "cpu"
    assert resolved.engine_device == "cpu"
    assert resolver.resolved == resolved
    with pytest.raises(ComputeError, match="only attempt resolution once"):
        resolver.resolve("cpu")


def test_cuda_resolution_fails_closed_without_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    resolver = DeviceResolver()

    with pytest.raises(DeviceUnavailable, match="refusing CPU fallback"):
        resolver.resolve("gpu")
    with pytest.raises(ComputeError, match="only attempt resolution once"):
        resolver.resolve("cpu")


def test_auto_device_requires_approved_benchmark_decision() -> None:
    resolver = DeviceResolver()

    with pytest.raises(DeviceUnavailable, match="APPROVED end-to-end benchmark"):
        resolver.resolve("auto")


@pytest.mark.parametrize("field", ["device", "es_device"])
def test_application_config_rejects_unimplemented_device_selection(tmp_path, field: str) -> None:
    config_path = tmp_path / f"{field}-config.json"
    config_path.write_text(json.dumps({field: "cuda"}), encoding="utf-8")

    with pytest.raises(DataError, match="unknown application configuration fields"):
        load_config(config_path)


def test_gpu_lease_conflict_releases_loser_quota_and_preserves_owner(tmp_path) -> None:
    lock_dir = tmp_path / "gpu-locks"
    owner_quota = ResourceQuota(tmp_path / "owner.sqlite", cpu_cores=1, ram_bytes=1024, gpus=1)
    contender_quota = ResourceQuota(tmp_path / "contender.sqlite", cpu_cores=1, ram_bytes=1024, gpus=1)
    owner = ComputeAllocation(
        quota=owner_quota, owner="owner", attempt_id="attempt-owner", cpu_cores=1,
        ram_bytes=1024, learner_device="cuda", gpu_lock_directory=lock_dir,
    )
    contender = ComputeAllocation(
        quota=contender_quota, owner="contender", attempt_id="attempt-contender", cpu_cores=1,
        ram_bytes=1024, learner_device="cuda", gpu_lock_directory=lock_dir,
    )

    owner.acquire()
    try:
        with pytest.raises(QuotaExceeded, match="single CUDA learner owner"):
            contender.acquire()
        assert len(owner_quota.active()) == 1
        assert contender_quota.active() == []
        assert (lock_dir / "gpu-0.owner").is_file()
    finally:
        owner.release()

    assert owner_quota.active() == []
    lock_file = lock_dir / "gpu-0.owner"
    assert lock_file.is_file()  # Stable inode prevents split-lock races.
    assert lock_file.stat().st_size == 0


def test_gpu_kernel_lease_recovers_after_hard_process_exit(tmp_path) -> None:
    context = mp.get_context("spawn")
    receive, send = context.Pipe(duplex=False)
    process = context.Process(target=_hold_gpu_lease, args=(str(tmp_path), send))
    process.start()
    send.close()
    try:
        assert receive.poll(15), "GPU lease owner did not acquire its lock"
        assert receive.recv() is True
        contender = GPULearnerLease(tmp_path, 0)
        with pytest.raises(QuotaExceeded, match="single CUDA learner owner"):
            contender.acquire(attempt_id="contender-before-exit")

        process.kill()
        process.join(timeout=10)
        assert process.exitcode is not None and process.exitcode != 0

        recovered = GPULearnerLease(tmp_path, 0)
        recovered.acquire(attempt_id="contender-after-exit")
        recovered.release()
    finally:
        if process.is_alive():
            process.kill()
            process.join(timeout=5)
        receive.close()


def test_gpu_kernel_lease_keeps_legacy_owner_file_fail_closed(tmp_path) -> None:
    lock_file = tmp_path / "gpu-0.owner"
    lock_file.write_text(json.dumps({"token": "old", "attempt_id": "legacy", "pid": 123}),
                         encoding="utf-8")

    with pytest.raises(ComputeError, match="legacy GPU owner file requires manual recovery"):
        GPULearnerLease(tmp_path, 0).acquire(attempt_id="new-owner")
    assert json.loads(lock_file.read_text(encoding="utf-8"))["token"] == "old"


def test_application_cuda_lease_uses_one_user_shared_path_across_run_directories(
    tmp_path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    shared_lock = tmp_path / "user-gpu-leases"
    monkeypatch.setenv("RLES_GPU_LEASE_DIRECTORY", str(shared_lock))
    run_one = application._gpu_lease_identity("cuda")
    run_two = application._gpu_lease_identity("cuda")

    assert run_one == run_two == {
        "scope": "user_shared", "directory": str(shared_lock.resolve()), "gpu_index": 0,
    }
    assert application._gpu_lease_identity("cpu") is None


def test_application_cuda_lease_rejects_relative_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RLES_GPU_LEASE_DIRECTORY", ".\\run-local-gpu-locks")

    with pytest.raises(DataError, match="absolute shared path"):
        application._gpu_lease_identity("cuda")


@pytest.mark.parametrize("value", [True, 0, 1.01, float("nan"), "0.5"])
def test_application_rejects_invalid_cuda_allocator_fraction(value) -> None:
    with pytest.raises(DataError, match="rl_gpu_memory_fraction"):
        application._rl_gpu_memory_fraction(value, "cuda")


def test_application_cuda_allocator_fraction_requires_cuda() -> None:
    assert application._rl_gpu_memory_fraction(None, "cpu") is None
    assert application._rl_gpu_memory_fraction(0.75, "cuda") == 0.75
    with pytest.raises(DataError, match="requires rl_device=cuda"):
        application._rl_gpu_memory_fraction(0.75, "cpu")
