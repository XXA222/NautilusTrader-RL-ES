"""OS owner-death recovery for owned Windows CPU trees, not strategy evidence."""
from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import multiprocessing as mp
import os
import sqlite3
import subprocess
import sys
import time

import pytest

from rles_compute_runtime.errors import ComputeError, QuotaExceeded
from rles_compute_runtime.recovery import inspect_quota, main
from rles_compute_runtime.resources import Lease, ResourceQuota
from rles_compute_runtime.workers import CPUWorkerPool

pytestmark = [pytest.mark.artifact_restore]
WINDOWS = pytest.mark.skipif(os.name != 'nt', reason='Windows owner and Job lifetime APIs required')
MIB = 1024**2


class Session:
    def __init__(self, config):
        self.child = None
        if config.get('descendant'):
            self.child = subprocess.Popen(
                [sys.executable, '-c', 'import time; time.sleep(120)'],
                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )

    def close(self):
        if self.child is not None:
            self.child.kill()
            self.child.wait(timeout=5)


def make_session(config):
    return Session(config)


def _controller(database, ready, stop, descendant):
    quota = ResourceQuota(database, cpu_cores=1, ram_bytes=512 * MIB, gpus=0)
    pool = CPUWorkerPool(
        workers=1, factory_ref=f'{__name__}:make_session', factory_config={'descendant': descendant},
        mode='rl', quota=quota, lease_owner='hard-exit-owner', ram_bytes_per_worker=256 * MIB,
        enforce_memory_limit=True, startup_timeout_s=15,
    )
    try:
        pool.start()
        ready.send({'lease': quota.active()[0], 'jobs': pool.resource_evidence(), 'pid': os.getpid()})
        stop.recv()
    finally:
        pool.close()


def _start(tmp_path, *, descendant=False):
    context = mp.get_context('spawn')
    receive, send = context.Pipe(duplex=False)
    stop_receive, stop_send = context.Pipe(duplex=False)
    database = tmp_path / 'quota.sqlite'
    process = context.Process(target=_controller, args=(str(database), send, stop_receive, descendant))
    process.start()
    send.close()
    stop_receive.close()
    try:
        assert receive.poll(20), 'controller did not report its sealed lease'
        state = receive.recv()
        row = state['lease']
        lease = Lease(row['token'], row['cpu'], row['ram'], row['gpu'], row['owner'])
        return process, stop_send, database, state, lease
    except BaseException:
        process.kill()
        process.join(timeout=5)
        stop_send.close()
        raise
    finally:
        receive.close()


def _cleanup(process, stop):
    if process.is_alive():
        try:
            stop.send('stop')
        except (OSError, BrokenPipeError):
            pass
        process.join(timeout=5)
    if process.is_alive():
        process.kill()
        process.join(timeout=5)
    stop.close()


def _quota(database):
    return ResourceQuota(database, cpu_cores=1, ram_bytes=512 * MIB, gpus=0)


def _recover_competitor(database, lease, gate, result):
    quota = _quota(database)
    gate.wait(10)
    try:
        quota.recover_cpu_lease(lease, reason='concurrent explicit recovery')
        result.send('recovered')
    except ComputeError:
        result.send('blocked')
    finally:
        result.close()


def _wait_empty(state):
    from rles_compute_runtime.windows_recovery import require_empty_job

    deadline = time.monotonic() + 10
    while True:
        try:
            for job in state['jobs']:
                require_empty_job(job['job_name'], 256 * MIB)
            return
        except ComputeError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(0.02)


def test_inspection_is_read_only_and_does_not_create_missing_database(tmp_path):
    path = tmp_path / 'quota.sqlite'
    with pytest.raises(ComputeError, match='existing regular'):
        inspect_quota(path)
    assert not path.exists()
    quota = _quota(path)
    lease = quota.acquire('read-only', cpu_cores=1, ram_bytes=1024)
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    state = inspect_quota(path)
    assert state['leases'][0]['recovery_state'] == 'unmanaged_or_legacy'
    assert state['leases'][0]['token'] == lease.token
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before
    quota.release(lease)


def test_gpu_and_legacy_lease_recovery_fail_closed(tmp_path):
    quota = ResourceQuota(tmp_path / 'quota.sqlite', cpu_cores=2, ram_bytes=4096, gpus=1)
    gpu = quota.acquire('gpu-owner', cpu_cores=1, ram_bytes=1024, gpus=1)
    legacy = quota.acquire('old-owner', cpu_cores=1, ram_bytes=1024)
    with pytest.raises(ComputeError, match='GPU lease recovery'):
        quota.recover_cpu_lease(gpu, reason='must preserve GPU')
    with pytest.raises(ComputeError, match='no completed'):
        quota.recover_cpu_lease(legacy, reason='legacy lacks lifetime evidence')
    assert len(quota.active()) == 2
    assert inspect_quota(quota.path)['recoveries'] == []
    quota.release(gpu)
    quota.release(legacy)


def test_old_database_inspection_does_not_migrate_schema(tmp_path):
    path = tmp_path / 'old.sqlite'
    with sqlite3.connect(path) as connection:
        connection.execute('CREATE TABLE quota_meta (id INTEGER PRIMARY KEY,cpu INTEGER,ram INTEGER,gpu INTEGER)')
        connection.execute('INSERT INTO quota_meta VALUES(1,1,1024,0)')
        connection.execute('CREATE TABLE lease (token TEXT,owner TEXT,cpu INTEGER,ram INTEGER,gpu INTEGER,acquired REAL)')
        connection.execute('INSERT INTO lease VALUES(?,?,?,?,?,?)', ('a' * 32, 'legacy', 1, 1024, 0, 1))
    before = path.read_bytes()
    state = inspect_quota(path)
    assert state['leases'][0]['recovery_state'] == 'unmanaged_or_legacy'
    assert state['recoveries'] == []
    assert path.read_bytes() == before


@WINDOWS
def test_live_owner_and_wrong_descriptor_cannot_release_cpu_quota(tmp_path):
    process, stop, database, state, lease = _start(tmp_path)
    quota = _quota(database)
    try:
        assert process.pid == state['pid']
        for descriptor in (replace(lease, owner='foreign'), replace(lease, ram_bytes=1),
                           replace(lease, token='a' * 32)):
            with pytest.raises(ComputeError, match='does not exist or does not belong'):
                quota.recover_cpu_lease(descriptor, reason='wrong lease')
        with pytest.raises(ComputeError, match='still alive'):
            quota.recover_cpu_lease(lease, reason='owner has not exited')
        assert len(quota.active()) == 1
        with pytest.raises(QuotaExceeded):
            quota.acquire('contender', cpu_cores=1, ram_bytes=1024)
    finally:
        _cleanup(process, stop)
    assert quota.active() == []


@WINDOWS
def test_hard_killed_coordinator_recovers_exact_lease_and_cli_receipt(tmp_path, capsys):
    process, stop, database, state, lease = _start(tmp_path, descendant=True)
    quota = _quota(database)
    try:
        process.kill()  # Actual OS termination, no finally/KeyboardInterrupt cleanup.
        process.join(timeout=5)
        assert process.exitcode is not None and process.exitcode != 0
        _wait_empty(state)
        assert len(quota.active()) == 1  # Kernel cleanup did not silently refund SQLite.
        assert main(['recover', '--database', str(database), '--token', lease.token,
                     '--owner', lease.owner, '--cpu-cores', str(lease.cpu_cores),
                     '--ram-bytes', str(lease.ram_bytes), '--reason', 'owned coordinator OS kill']) == 0
        receipt = json.loads(capsys.readouterr().out)
        assert receipt['schema'] == 'rles.cpu-lease-recovery.v1'
        assert all(job['active_processes'] == 0 for job in receipt['jobs'])
        assert quota.active() == []
        assert inspect_quota(database)['recoveries'] == [receipt]
        with pytest.raises(ComputeError, match='does not exist'):
            quota.recover_cpu_lease(lease, reason='replay must fail')
        renewed = quota.acquire('new-owner', cpu_cores=1, ram_bytes=256 * MIB)
        quota.release(renewed)
        with sqlite3.connect(database) as connection:
            assert connection.execute('SELECT count(*) FROM lease_recovery_log').fetchone()[0] == 1
    finally:
        _cleanup(process, stop)


@WINDOWS
def test_surviving_owned_tree_blocks_recovery_until_last_job_handle_closes(tmp_path):
    from rles_compute_runtime.windows_job import _kernel

    process, stop, database, state, lease = _start(tmp_path, descendant=True)
    quota = _quota(database)
    kernel = _kernel()
    handle = kernel.OpenJobObjectW(0x4, False, state['jobs'][0]['job_name'])
    assert handle
    try:
        # Retaining this exact owned Job handle intentionally prevents kill-on-close.
        process.kill()
        process.join(timeout=5)
        with pytest.raises(ComputeError, match='active processes'):
            quota.recover_cpu_lease(lease, reason='tree intentionally still alive')
        assert len(quota.active()) == 1
        assert inspect_quota(database)['recoveries'] == []
        kernel.CloseHandle(handle)
        handle = None
        _wait_empty(state)
        quota.recover_cpu_lease(lease, reason='all owned trees now exited')
        assert quota.active() == []
    finally:
        if handle:
            kernel.CloseHandle(handle)
        _cleanup(process, stop)


@WINDOWS
def test_incomplete_startup_and_corrupt_evidence_are_not_reclaimed(tmp_path):
    quota = _quota(tmp_path / 'quota.sqlite')
    lease = quota.acquire('incomplete', cpu_cores=1, ram_bytes=256 * MIB)
    quota.enable_windows_recovery(lease, workers=1, memory_bytes=256 * MIB)
    with pytest.raises(ComputeError, match='no completed'):
        quota.recover_cpu_lease(lease, reason='startup incomplete')
    quota.seal_windows_recovery(lease)
    with sqlite3.connect(quota.path) as connection:
        connection.execute("UPDATE lease_recovery SET jobs_json='[\"foreign-job\"]'")
    with pytest.raises(ComputeError, match='invalid owned Job'):
        quota.recover_cpu_lease(lease, reason='corrupt job list')
    assert len(quota.active()) == 1
    quota.release(lease)


@WINDOWS
def test_concurrent_recovery_commits_once(tmp_path):
    process, stop, database, state, lease = _start(tmp_path)
    competitors = []
    receivers = []
    try:
        process.kill()
        process.join(timeout=5)
        _wait_empty(state)
        context = mp.get_context('spawn')
        gate = context.Event()
        for _ in range(2):
            receive, send = context.Pipe(duplex=False)
            competitor = context.Process(target=_recover_competitor, args=(str(database), lease, gate, send))
            competitor.start()
            send.close()
            competitors.append(competitor)
            receivers.append(receive)
        gate.set()
        outcomes = []
        for receive in receivers:
            assert receive.poll(15)
            outcomes.append(receive.recv())
        assert sorted(outcomes) == ['blocked', 'recovered']
        assert len(inspect_quota(database)['recoveries']) == 1
        assert _quota(database).active() == []
    finally:
        for receive in receivers:
            receive.close()
        for competitor in competitors:
            competitor.join(timeout=5)
            if competitor.is_alive():
                competitor.kill()
                competitor.join(timeout=5)
        _cleanup(process, stop)


@WINDOWS
def test_owner_witness_rejects_foreign_scope_and_distinguishes_pid_reuse(monkeypatch):
    import ctypes as ct
    from types import SimpleNamespace
    import rles_compute_runtime.windows_recovery as recovery

    identity = recovery.owner_identity()
    with pytest.raises(ComputeError, match='still alive'):
        recovery.require_owner_exited(identity)
    for change in ({'machine': 'another-machine'}, {'session': identity['session'] + 1}):
        with pytest.raises(ComputeError, match='original machine and Windows session'):
            recovery.require_owner_exited({**identity, **change})
    # Classifier probe only: immutable enrollment always captured the true birth time.
    assert recovery.require_owner_exited({**identity, 'created': identity['created'] - 1}) == {
        'owner_state': 'original_exited_pid_reused',
    }
    monkeypatch.setattr(recovery, '_machine', lambda: identity['machine'])
    monkeypatch.setattr(recovery, '_session', lambda _pid: identity['session'])
    def denied(*_args):
        ct.set_last_error(5)
        return 0
    monkeypatch.setattr(recovery, '_kernel', lambda: SimpleNamespace(OpenProcess=denied))
    with pytest.raises(ComputeError, match='cannot establish owner exit'):
        recovery.require_owner_exited(identity)


@WINDOWS
def test_named_job_cannot_attach_to_preexisting_object():
    from uuid import uuid4
    from rles_compute_runtime.errors import WorkerFailure
    from rles_compute_runtime.windows_job import WindowsWorkerJob

    name = 'Local\\RLES-collision-' + uuid4().hex
    with WindowsWorkerJob(256 * MIB, name=name) as job:
        with pytest.raises(WorkerFailure, match='already exists'):
            WindowsWorkerJob(128 * MIB, name=name)
        assert job.snapshot()['tree_limit_bytes'] == 256 * MIB
