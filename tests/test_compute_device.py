from __future__ import annotations

import json

import pytest
import torch

from rles_compute_runtime.allocation import ComputeAllocation
from rles_compute_runtime.device import DeviceResolver
from rles_compute_runtime.errors import ComputeError, DeviceUnavailable, QuotaExceeded
from rles_compute_runtime.resources import ResourceQuota
from rles_data.core import DataError
from rles_orchestrator.application import load_config


pytestmark = pytest.mark.unit


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
    assert not (lock_dir / "gpu-0.owner").exists()
