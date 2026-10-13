"""Actual Windows allocation/termination tests, not market-data goldens."""
from __future__ import annotations

from ctypes import wintypes as wt
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from rles_compute_runtime.errors import ComputeProtocolError, WorkerFailure
from rles_compute_runtime.faults import FaultInjector
from rles_compute_runtime.resources import ResourceQuota
from rles_compute_runtime.windows_job import WindowsWorkerJob, _kernel
from rles_compute_runtime.workers import CPUWorkerPool

pytestmark = [pytest.mark.micro_golden, pytest.mark.skipif(os.name != "nt", reason="Windows Job API required")]
MIB = 1024**2


class AllocationSession:
    def __init__(self, config):
        if config.get("marker"):
            Path(config["marker"]).write_text("factory started")

    def reset(self, **_kwargs):
        return {"pid": os.getpid()}

    def step(self, *, allocate_bytes=0):
        memory = bytearray(allocate_bytes)
        return {"allocated": len(memory)}

    def close(self):
        pass


def make_session(config):
    return AllocationSession(config)


def _key(step):
    return {"run_id": "limits-run", "attempt_id": "limits-attempt", "env_id": 0,
            "episode_id": 1, "step_id": step, "candidate_id": "none", "model_id": "model",
            "schema_version": "0.4.0", "dtype": "float32", "shape": [1]}


def _pool(tmp_path, *, memory_bytes=256 * MIB, marker=None):
    quota = ResourceQuota(tmp_path / "quota.sqlite", cpu_cores=1, ram_bytes=512 * MIB, gpus=0)
    pool = CPUWorkerPool(workers=1, factory_ref=f"{__name__}:make_session",
                         factory_config={} if marker is None else {"marker": str(marker)},
                         mode="rl", quota=quota, lease_owner="owned-limit-test",
                         ram_bytes_per_worker=memory_bytes, startup_timeout_s=5,
                         enforce_memory_limit=True)
    return pool, quota


def test_windows_job_denies_real_commit_and_reports_peak():
    code = '''import json,sys
sys.stdin.readline()
small=bytearray(16*1024**2)
try:
    large=bytearray(256*1024**2)
except MemoryError:
    print(json.dumps({'small':len(small),'large':'MemoryError'}),flush=True)
else:
    print(json.dumps({'large':'unexpected success'}),flush=True)
'''
    child = subprocess.Popen([sys.executable, "-c", code], stdin=subprocess.PIPE,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                             creationflags=subprocess.CREATE_NO_WINDOW)
    job = WindowsWorkerJob(96 * MIB)
    try:
        job.attach(child.pid)
        stdout, stderr = child.communicate("go\n", timeout=15)
        assert child.returncode == 0, stderr
        assert json.loads(stdout) == {"small": 16 * MIB, "large": "MemoryError"}
        report = job.close()
        assert report["active_processes"] == 0
        assert report["process_limit_bytes"] == report["tree_limit_bytes"] == 96 * MIB
        assert 0 < report["peak_process_commit_bytes"] <= 96 * MIB
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=5)
        job.close()


def test_coordinator_hard_exit_kills_owned_worker_tree():
    controller_code = '''import os,subprocess,sys
from rles_compute_runtime.windows_job import WindowsWorkerJob
child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)'],
    stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,
    creationflags=subprocess.CREATE_NO_WINDOW)
job=WindowsWorkerJob(128*1024**2)
try:
    job.attach(child.pid)
except BaseException:
    child.kill();child.wait();raise
print(child.pid,flush=True)
sys.stdin.readline()
os._exit(73)
'''
    controller = subprocess.Popen([sys.executable, "-c", controller_code], stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                                  creationflags=subprocess.CREATE_NO_WINDOW)
    handle = None
    kernel = _kernel()
    try:
        assert controller.stdout is not None
        child_pid = int(controller.stdout.readline().strip())
        handle = kernel.OpenProcess(0x100000, False, child_pid)  # SYNCHRONIZE; retain exact process identity.
        assert handle
        controller.communicate("exit\n", timeout=10)
        assert controller.returncode == 73
        kernel.WaitForSingleObject.argtypes = [wt.HANDLE, wt.DWORD]
        kernel.WaitForSingleObject.restype = wt.DWORD
        assert kernel.WaitForSingleObject(handle, 10000) == 0
    finally:
        if controller.poll() is None:
            controller.kill()
            controller.wait(timeout=5)
        if handle:
            kernel.CloseHandle(handle)


def test_descendant_commit_counts_against_the_same_tree_limit():
    code = '''import json,subprocess,sys
sys.stdin.readline()
held=bytearray(64*1024**2)
child_code="try:\\n memory=bytearray(64*1024**2)\\nexcept MemoryError:\\n print('tree-limited',flush=True)\\nelse:\\n print('unexpected success',flush=True)"
child=subprocess.run([sys.executable,'-c',child_code],capture_output=True,text=True,
    creationflags=subprocess.CREATE_NO_WINDOW,timeout=10)
print(json.dumps({'child_exit':child.returncode,'child_output':child.stdout.strip(),'held':len(held)}),flush=True)
'''
    parent = subprocess.Popen([sys.executable, "-c", code], stdin=subprocess.PIPE,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                              creationflags=subprocess.CREATE_NO_WINDOW)
    job = WindowsWorkerJob(128 * MIB)
    try:
        job.attach(parent.pid)
        stdout, stderr = parent.communicate("go\n", timeout=15)
        assert parent.returncode == 0, stderr
        assert json.loads(stdout) == {"child_exit": 0, "child_output": "tree-limited", "held": 64 * MIB}
        report = job.close()
        # Windows venv redirectors may add launcher processes to the tree.
        assert report["total_processes"] >= 2 and report["active_processes"] == 0
        assert report["peak_process_commit_bytes"] < 128 * MIB
    finally:
        if parent.poll() is None:
            parent.kill()
            parent.wait(timeout=5)
        job.close()


def test_memory_limited_pool_aborts_on_oom_and_releases_only_after_tree_exit(tmp_path):
    pool, quota = _pool(tmp_path)
    with pool:
        assert pool.dispatch([(0, _key(0), "reset", {})], timeout_s=5)
        assert pool.dispatch([(0, _key(1), "step", {"allocate_bytes": 16 * MIB})], timeout_s=5)
        with pytest.raises(ComputeProtocolError, match="MemoryError"):
            pool.dispatch([(0, _key(2), "step", {"allocate_bytes": 512 * MIB})], timeout_s=5)
    assert quota.active() == []
    evidence = pool.resource_evidence()[0]
    assert evidence["enforcement"] == "windows_job_commit_memory"
    assert evidence["active_processes"] == 0
    assert evidence["peak_tree_commit_bytes"] <= 256 * MIB
    with pytest.raises(WorkerFailure, match="not running"):
        pool.dispatch([(0, _key(3), "step", {})], timeout_s=5)


def test_limited_worker_force_kill_does_not_reuse_stale_actions(tmp_path):
    pool, quota = _pool(tmp_path)
    with pool:
        pool.dispatch([(0, _key(0), "reset", {})], timeout_s=5)
        FaultInjector(research_sandbox=True, reviewer_approved=True).terminate_worker(pool, 0)
        with pytest.raises(WorkerFailure, match="dead before submit"):
            pool.dispatch([(0, _key(1), "step", {})], timeout_s=5)
    assert quota.active() == []
    assert pool.resource_evidence()[0]["active_processes"] == 0


def test_limit_startup_failure_never_opens_factory_or_leaves_lease(tmp_path):
    marker = tmp_path / "factory-started.txt"
    pool, quota = _pool(tmp_path, memory_bytes=MIB, marker=marker)
    with pytest.raises(WorkerFailure):
        pool.start()
    assert not marker.exists()
    assert quota.active() == []
    assert all(row["active_processes"] == 0 for row in pool.resource_evidence())


def test_job_refuses_to_own_its_coordinator():
    with WindowsWorkerJob(128 * MIB) as job:
        with pytest.raises(WorkerFailure, match="coordinator"):
            job.attach(os.getpid())
