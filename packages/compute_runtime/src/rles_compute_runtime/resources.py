"""SQLite-backed cross-process quota accounting and hard exclusive GPU lease."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
import json
import importlib
import os
import socket
import sqlite3
import threading
import time
import uuid

from .errors import ComputeError, QuotaExceeded


@dataclass(frozen=True, slots=True)
class Lease:
    token: str
    cpu_cores: int
    ram_bytes: int
    gpu_count: int
    owner: str


class ResourceQuota:
    """Atomic leases shared by Windows/Linux processes via an on-disk SQLite database.

    Leases are intentionally NOT automatically reclaimed by PID: an unknown process
    could still own a live CUDA context. A failed owner requires explicit recovery.
    """

    def __init__(self, db_path: str | Path, *, cpu_cores: int, ram_bytes: int, gpus: int = 1):
        if any(type(v) is not int for v in (cpu_cores, ram_bytes, gpus)) or cpu_cores < 1 or ram_bytes < 1 or gpus < 0:
            raise ValueError('invalid resource budgets')
        self.path = Path(db_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.limits = cpu_cores, ram_bytes, gpus
        with self._db() as conn:
            conn.execute('CREATE TABLE IF NOT EXISTS quota_meta (id INTEGER PRIMARY KEY CHECK(id=1), cpu INTEGER NOT NULL, ram INTEGER NOT NULL, gpu INTEGER NOT NULL)')
            conn.execute('CREATE TABLE IF NOT EXISTS lease (token TEXT PRIMARY KEY, owner TEXT NOT NULL, cpu INTEGER NOT NULL, ram INTEGER NOT NULL, gpu INTEGER NOT NULL, acquired REAL NOT NULL)')
            conn.execute('CREATE TABLE IF NOT EXISTS lease_recovery (token TEXT PRIMARY KEY, state TEXT NOT NULL, identity_json TEXT NOT NULL, jobs_json TEXT NOT NULL, memory_bytes INTEGER NOT NULL)')
            conn.execute('CREATE TABLE IF NOT EXISTS lease_recovery_log (token TEXT PRIMARY KEY, owner TEXT NOT NULL, recovered REAL NOT NULL, reason TEXT NOT NULL, receipt_json TEXT NOT NULL)')
            old = conn.execute('SELECT cpu, ram, gpu FROM quota_meta WHERE id=1').fetchone()
            if old is None:
                conn.execute('INSERT INTO quota_meta VALUES (1,?,?,?)', self.limits)
            elif tuple(old) != self.limits:
                raise ComputeError('quota limits already frozen in this database')

    @contextmanager
    def _db(self):
        conn = sqlite3.connect(self.path, timeout=15, isolation_level=None)
        try:
            conn.execute('PRAGMA busy_timeout=15000')
            conn.execute('BEGIN IMMEDIATE')
            yield conn
            conn.execute('COMMIT')
        except BaseException:
            if conn.in_transaction:
                conn.execute('ROLLBACK')
            raise
        finally:
            conn.close()

    def acquire(self, owner: str, *, cpu_cores: int, ram_bytes: int, gpus: int = 0) -> Lease:
        if not isinstance(owner,str) or not owner or any(type(v) is not int for v in (cpu_cores, ram_bytes, gpus)) or cpu_cores < 0 or ram_bytes < 0 or gpus < 0 or (cpu_cores == ram_bytes == gpus == 0):
            raise ValueError('invalid resource request')
        requested = (cpu_cores, ram_bytes, gpus)
        with self._db() as conn:
            used = conn.execute('SELECT COALESCE(SUM(cpu),0),COALESCE(SUM(ram),0),COALESCE(SUM(gpu),0) FROM lease').fetchone()
            if any(a + b > maximum for a, b, maximum in zip(used, requested, self.limits)):
                raise QuotaExceeded(f'cannot lease requested cpu/ram/gpu; currently used={tuple(used)}')
            token = uuid.uuid4().hex
            conn.execute('INSERT INTO lease VALUES (?,?,?,?,?,?)', (token, owner, *requested, time.time()))
        return Lease(token, *requested, owner)

    def release(self, lease: Lease) -> None:
        with self._db() as conn:
            self._require_lease(conn, lease)
            recovery = conn.execute(
                'SELECT state,identity_json,jobs_json,memory_bytes FROM lease_recovery WHERE token=?',
                (lease.token,),
            ).fetchone()
            if recovery is not None:
                self._require_safe_enrolled_release(lease, recovery)
            cur = conn.execute('DELETE FROM lease WHERE token=? AND owner=? AND cpu=? AND ram=? AND gpu=?',
                               (lease.token, lease.owner, lease.cpu_cores, lease.ram_bytes, lease.gpu_count))
            if cur.rowcount != 1:
                raise ComputeError('lease does not exist or does not belong to caller')
            conn.execute('DELETE FROM lease_recovery WHERE token=?', (lease.token,))

    @staticmethod
    def _require_safe_enrolled_release(lease: Lease, recovery: tuple) -> None:
        """Do not refund an enrolled lease until its owned execution is quiescent."""
        from .windows_recovery import (
            owner_identity, require_empty_job, require_owned_job,
        )

        if os.name != 'nt':
            raise ComputeError('enrolled Windows lease release requires Windows ownership proof')
        state, identity_json, evidence_json, memory_bytes = recovery
        try:
            identity, evidence = json.loads(identity_json), json.loads(evidence_json)
        except (TypeError, ValueError) as exc:
            raise ComputeError('corrupted lease release evidence') from exc
        if identity != owner_identity():
            raise ComputeError('enrolled lease release must come from its original owner')

        if lease.gpu_count == 0 and state in {'preparing', 'ready'}:
            if (not isinstance(evidence, list) or not 1 <= len(evidence) <= 128
                    or evidence != [f'Local\\RLES-{lease.token}-{index}'
                                    for index in range(len(evidence))]
                    or type(memory_bytes) is not int or memory_bytes <= 0
                    or memory_bytes * len(evidence) != lease.ram_bytes):
                raise ComputeError('invalid enrolled CPU release evidence')
            for name in evidence:
                require_empty_job(name, memory_bytes)
            return

        if lease.gpu_count == 1 and state in {'application_preparing', 'application_ready'}:
            required = {'kind', 'job_name', 'memory_bytes', 'gpu_lock_directory', 'gpu_index'}
            if (not isinstance(evidence, dict) or set(evidence) != required
                    or evidence.get('kind') != 'supervised_gpu'
                    or not isinstance(evidence.get('job_name'), str)
                    or type(memory_bytes) is not int or memory_bytes <= 0
                    or evidence.get('memory_bytes') != memory_bytes
                    or not isinstance(evidence.get('gpu_lock_directory'), str)
                    or not Path(evidence['gpu_lock_directory']).is_absolute()
                    or type(evidence.get('gpu_index')) is not int
                    or evidence['gpu_index'] < 0):
                raise ComputeError('invalid supervised GPU release evidence')
            job = require_owned_job(evidence['job_name'], memory_bytes, identity['pid'])
            probe = GPULearnerLease(evidence['gpu_lock_directory'], evidence['gpu_index'])
            probe.acquire(attempt_id=f'release-{lease.token}')
            probe.release()
            process_ids = job['process_ids']
            owner_ancestors = job['owner_ancestor_pids']
            allowed_processes = {identity['pid'], *owner_ancestors}
            if (not isinstance(process_ids, tuple) or identity['pid'] not in process_ids
                    or any(pid not in allowed_processes for pid in process_ids)):
                raise ComputeError(
                    f"supervised application descendants remain active "
                    f"{process_ids}; release refused",
                )
            return

        raise ComputeError('enrolled lease state is not releasable; ownership remains reserved')

    @staticmethod
    def _require_lease(conn, lease: Lease) -> None:
        if (type(lease) is not Lease or not isinstance(lease.token, str)
                or len(lease.token) != 32 or any(c not in '0123456789abcdef' for c in lease.token)
                or not isinstance(lease.owner, str) or not lease.owner
                or any(type(v) is not int or v < 0 for v in
                       (lease.cpu_cores, lease.ram_bytes, lease.gpu_count))):
            raise ComputeError('invalid exact lease descriptor')
        row = conn.execute('SELECT token,owner,cpu,ram,gpu FROM lease WHERE token=?',
                           (lease.token,)).fetchone()
        if row != (lease.token, lease.owner, lease.cpu_cores, lease.ram_bytes, lease.gpu_count):
            raise ComputeError('lease does not exist or does not belong to caller')

    def enable_windows_recovery(self, lease: Lease, *, workers: int, memory_bytes: int) -> tuple[str, ...]:
        """Enroll before creating any worker. Incomplete startup is never reclaimed."""
        from .windows_recovery import owner_identity

        if (type(workers) is not int or not 1 <= workers <= 128
                or type(memory_bytes) is not int or memory_bytes <= 0
                or lease.gpu_count != 0 or lease.ram_bytes != workers * memory_bytes
                or lease.cpu_cores < workers):
            raise ComputeError('recovery enrollment requires exact CPU worker tree budgets')
        identity = owner_identity()  # Requires actual local Windows lifetime evidence.
        names = tuple(f'Local\\RLES-{lease.token}-{index}' for index in range(workers))
        with self._db() as conn:
            self._require_lease(conn, lease)
            if conn.execute('SELECT 1 FROM lease_recovery WHERE token=?', (lease.token,)).fetchone():
                raise ComputeError('lease recovery already enrolled')
            conn.execute('INSERT INTO lease_recovery VALUES (?,?,?,?,?)',
                         (lease.token, 'preparing', json.dumps(identity), json.dumps(names), memory_bytes))
        return names

    def seal_windows_recovery(self, lease: Lease) -> None:
        """Only the enrolling coordinator may certify completed worker startup."""
        from .windows_recovery import owner_identity

        identity = owner_identity()
        with self._db() as conn:
            self._require_lease(conn, lease)
            row = conn.execute('SELECT state,identity_json FROM lease_recovery WHERE token=?',
                               (lease.token,)).fetchone()
            if row is None or row[0] != 'preparing' or json.loads(row[1]) != identity:
                raise ComputeError('worker recovery enrollment is not owned or already sealed')
            conn.execute("UPDATE lease_recovery SET state='ready' WHERE token=?", (lease.token,))

    def enable_supervised_gpu_recovery(self, lease: Lease, *, job_name: str,
                                       memory_bytes: int, gpu_lock_directory: str | Path,
                                       gpu_index: int) -> None:
        """Enroll a supervised app GPU lease before learner/worker creation."""
        from .windows_recovery import owner_identity, require_owned_job

        directory = Path(gpu_lock_directory)
        if (lease.gpu_count != 1 or type(memory_bytes) is not int or memory_bytes <= 0
                or type(gpu_index) is not int or gpu_index < 0 or not directory.is_absolute()):
            raise ComputeError('supervised GPU recovery requires exact GPU and Job identities')
        identity = owner_identity()
        require_owned_job(job_name, memory_bytes, identity['pid'])
        evidence = {
            'kind': 'supervised_gpu', 'job_name': job_name, 'memory_bytes': memory_bytes,
            'gpu_lock_directory': str(directory.resolve()), 'gpu_index': gpu_index,
        }
        with self._db() as conn:
            self._require_lease(conn, lease)
            if conn.execute('SELECT 1 FROM lease_recovery WHERE token=?', (lease.token,)).fetchone():
                raise ComputeError('lease recovery already enrolled')
            conn.execute('INSERT INTO lease_recovery VALUES (?,?,?,?,?)',
                         (lease.token, 'application_preparing', json.dumps(identity),
                          json.dumps(evidence), memory_bytes))

    def seal_supervised_gpu_recovery(self, lease: Lease) -> None:
        """Seal enrollment after rechecking owner membership in the named Job."""
        from .windows_recovery import owner_identity, require_owned_job

        identity = owner_identity()
        with self._db() as conn:
            self._require_lease(conn, lease)
            row = conn.execute('SELECT state,identity_json,jobs_json,memory_bytes '
                               'FROM lease_recovery WHERE token=?', (lease.token,)).fetchone()
            if row is None or row[0] != 'application_preparing' or json.loads(row[1]) != identity:
                raise ComputeError('supervised GPU recovery enrollment is not owned or already sealed')
            try:
                evidence = json.loads(row[2])
            except (TypeError, ValueError) as exc:
                raise ComputeError('corrupted supervised GPU recovery evidence') from exc
            if (not isinstance(evidence, dict) or evidence.get('kind') != 'supervised_gpu'
                    or evidence.get('memory_bytes') != row[3]):
                raise ComputeError('invalid supervised GPU recovery evidence')
            job_name = evidence.get('job_name')
            if not isinstance(job_name, str):
                raise ComputeError('invalid supervised application Job name')
            require_owned_job(job_name, row[3], identity['pid'])
            conn.execute("UPDATE lease_recovery SET state='application_ready' WHERE token=?",
                         (lease.token,))

    def recover_supervised_gpu_lease(self, lease: Lease, *, reason: str) -> dict:
        """Reclaim one exact GPU allocation after its supervised process tree exits."""
        from .windows_recovery import require_empty_job, require_owner_exited

        if not isinstance(reason, str) or not reason.strip() or len(reason) > 1000:
            raise ValueError('a nonempty bounded recovery reason is required')
        with self._db() as conn:
            self._require_lease(conn, lease)
            if lease.gpu_count != 1:
                raise ComputeError('supervised GPU recovery requires exactly one GPU')
            row = conn.execute('SELECT state,identity_json,jobs_json,memory_bytes '
                               'FROM lease_recovery WHERE token=?', (lease.token,)).fetchone()
            if row is None or row[0] != 'application_ready':
                raise ComputeError('GPU lease lacks sealed supervised process-tree evidence')
            try:
                identity, evidence = json.loads(row[1]), json.loads(row[2])
            except (TypeError, ValueError) as exc:
                raise ComputeError('corrupted supervised GPU recovery evidence') from exc
            if (not isinstance(evidence, dict) or set(evidence) != {
                    'kind', 'job_name', 'memory_bytes', 'gpu_lock_directory', 'gpu_index'}
                    or evidence['kind'] != 'supervised_gpu'
                    or not isinstance(evidence['job_name'], str)
                    or not evidence['job_name'].startswith('Local\\RLES-Supervisor-')
                    or type(row[3]) is not int or row[3] <= 0
                    or evidence['memory_bytes'] != row[3]
                    or not isinstance(evidence['gpu_lock_directory'], str)
                    or not Path(evidence['gpu_lock_directory']).is_absolute()
                    or type(evidence['gpu_index']) is not int or evidence['gpu_index'] < 0):
                raise ComputeError('invalid supervised GPU recovery evidence')
            owner_proof = require_owner_exited(identity)
            job = require_empty_job(evidence['job_name'], row[3])
            # Probe the shared kernel lock while the quota transaction is held.
            # This also refuses legacy owner files that lack safe OS-lock evidence.
            probe = GPULearnerLease(evidence['gpu_lock_directory'], evidence['gpu_index'])
            probe.acquire(attempt_id=f'recovery-{lease.token}')
            probe.release()
            receipt = {
                'schema': 'rles.supervised-gpu-lease-recovery.v1',
                'lease': {'token': lease.token, 'owner': lease.owner,
                          'cpu_cores': lease.cpu_cores, 'ram_bytes': lease.ram_bytes,
                          'gpu_count': lease.gpu_count},
                'reason': reason, 'owner_identity': identity, 'owner_proof': owner_proof,
                'job': job, 'gpu_lock_probe': 'acquired_and_released',
                'recovered_at': time.time(),
                'scope': 'exact supervised application process tree exited; no CUDA VRAM estimate',
            }
            conn.execute('INSERT INTO lease_recovery_log VALUES (?,?,?,?,?)',
                         (lease.token, lease.owner, receipt['recovered_at'], reason, json.dumps(receipt)))
            conn.execute('DELETE FROM lease_recovery WHERE token=?', (lease.token,))
            conn.execute('DELETE FROM lease WHERE token=?', (lease.token,))
        return receipt

    def recover_cpu_lease(self, lease: Lease, *, reason: str) -> dict:
        """Explicitly reclaim a dead coordinator's sealed Windows Job CPU lease.

        Checks and deletion share an IMMEDIATE transaction. GPU, legacy,
        incomplete startup, active owner/tree and unknown status fail closed.
        This never kills processes, removes GPU locks or refunds research queries.
        """
        from .windows_recovery import require_empty_job, require_owner_exited

        if not isinstance(reason, str) or not reason.strip() or len(reason) > 1000:
            raise ValueError('a nonempty bounded recovery reason is required')
        with self._db() as conn:
            self._require_lease(conn, lease)
            if lease.gpu_count:
                raise ComputeError('GPU lease recovery is not supported; ownership remains reserved')
            row = conn.execute('SELECT state,identity_json,jobs_json,memory_bytes FROM lease_recovery WHERE token=?',
                               (lease.token,)).fetchone()
            if row is None or row[0] != 'ready':
                raise ComputeError('lease has no completed Windows Job recovery evidence')
            try:
                identity, names = json.loads(row[1]), json.loads(row[2])
            except (TypeError, ValueError) as exc:
                raise ComputeError('corrupted lease recovery evidence') from exc
            if (not isinstance(names, list) or not 1 <= len(names) <= 128
                    or names != [f'Local\\RLES-{lease.token}-{index}' for index in range(len(names))]
                    or type(row[3]) is not int or row[3] <= 0
                    or row[3] * len(names) != lease.ram_bytes):
                raise ComputeError('invalid owned Job recovery evidence')
            owner_proof = require_owner_exited(identity)
            jobs = [require_empty_job(name, row[3]) for name in names]
            receipt = {
                'schema': 'rles.cpu-lease-recovery.v1', 'lease': {
                    'token': lease.token, 'owner': lease.owner, 'cpu_cores': lease.cpu_cores,
                    'ram_bytes': lease.ram_bytes, 'gpu_count': lease.gpu_count,
                }, 'reason': reason, 'owner_identity': identity, 'owner_proof': owner_proof,
                'jobs': jobs, 'recovered_at': time.time(),
                'scope': 'CPU quota only; research reservations unchanged; not an OS memory limit',
            }
            conn.execute('INSERT INTO lease_recovery_log VALUES (?,?,?,?,?)',
                         (lease.token, lease.owner, receipt['recovered_at'], reason, json.dumps(receipt)))
            conn.execute('DELETE FROM lease_recovery WHERE token=?', (lease.token,))
            conn.execute('DELETE FROM lease WHERE token=?', (lease.token,))
        return receipt

    def active(self) -> list[dict[str, object]]:
        with self._db() as conn:
            return [dict(zip(('token','owner','cpu','ram','gpu','acquired'), row)) for row in
                    conn.execute('SELECT token,owner,cpu,ram,gpu,acquired FROM lease ORDER BY acquired').fetchall()]


class GPULearnerLease:
    """Process-lifetime exclusive lock per GPU, with durable owner metadata.

    The kernel releases the byte-range lock when its last owning process handle
    closes, including after a hard process exit. The stable lock file is never
    unlinked, avoiding split-lock races on POSIX. Legacy owner files without a
    protocol marker remain manual-recovery-only.
    """

    _local_guard = threading.Lock()
    _local_owners: set[str] = set()
    _protocol = 'rles.gpu-lease.v2'

    def __init__(self, directory: str | Path, gpu_index: int):
        if type(gpu_index) is not int or gpu_index < 0:
            raise ValueError('invalid GPU index')
        self.gpu_index = gpu_index
        self.path = Path(directory) / f'gpu-{gpu_index}.owner'
        self.token: str | None = None
        self._fd: int | None = None
        self._path_key: str | None = None

    @staticmethod
    def _try_lock(fd: int) -> bool:
        os.lseek(fd, 0, os.SEEK_SET)
        if os.name == 'nt':
            msvcrt = importlib.import_module('msvcrt')

            try:
                msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
            except OSError:
                return False
        else:
            fcntl = importlib.import_module('fcntl')

            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                return False
        return True

    @staticmethod
    def _unlock(fd: int) -> None:
        os.lseek(fd, 0, os.SEEK_SET)
        if os.name == 'nt':
            msvcrt = importlib.import_module('msvcrt')

            msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
        else:
            fcntl = importlib.import_module('fcntl')

            fcntl.flock(fd, fcntl.LOCK_UN)

    @staticmethod
    def _release_local_key(key: str) -> None:
        with GPULearnerLease._local_guard:
            GPULearnerLease._local_owners.discard(key)

    def acquire(self, *, attempt_id: str) -> str:
        if self.token is not None or not attempt_id:
            raise ComputeError('GPU lease already acquired or attempt_id missing')
        self.path.parent.mkdir(parents=True, exist_ok=True)
        key = os.path.normcase(str(self.path.resolve()))
        with self._local_guard:
            if key in self._local_owners:
                raise QuotaExceeded('single CUDA learner owner already allocated')
            self._local_owners.add(key)
        token = uuid.uuid4().hex
        fd: int | None = None
        locked = False
        try:
            fd = os.open(self.path, os.O_CREAT | os.O_RDWR, 0o600)
            if os.fstat(fd).st_size == 0:
                os.write(fd, b'\0')
                os.fsync(fd)
            locked = self._try_lock(fd)
            if not locked:
                raise QuotaExceeded('single CUDA learner owner already allocated')
            os.lseek(fd, 0, os.SEEK_SET)
            previous_size = os.fstat(fd).st_size
            if previous_size > 65536:
                raise ComputeError('GPU owner file exceeds the supported metadata size')
            previous = os.read(fd, max(previous_size, 1))
            if previous.strip(b'\0\r\n '):
                try:
                    old_state = json.loads(previous.decode('utf-8'))
                except (UnicodeDecodeError, ValueError):
                    raise ComputeError('legacy or corrupted GPU owner file requires manual recovery')
                if not isinstance(old_state, dict) or old_state.get('lock_protocol') != self._protocol:
                    raise ComputeError('legacy GPU owner file requires manual recovery')
            state = json.dumps({
                'lock_protocol': self._protocol, 'token': token, 'attempt_id': attempt_id,
                'pid': os.getpid(), 'machine': socket.gethostname(),
            }, separators=(',', ':')).encode('utf-8')
            os.ftruncate(fd, 0)
            os.lseek(fd, 0, os.SEEK_SET)
            os.write(fd, state)
            os.fsync(fd)
        except BaseException:
            try:
                if fd is not None and locked:
                    self._unlock(fd)
            finally:
                if fd is not None:
                    os.close(fd)
                self._release_local_key(key)
            raise
        self.token = token
        self._fd = fd
        self._path_key = key
        return token

    def release(self) -> None:
        if self.token is None:
            raise ComputeError('GPU lease not owned')
        try:
            if self._fd is None:
                raise OSError('GPU lease handle is missing')
            if os.fstat(self._fd).st_size > 65536:
                raise ValueError('GPU owner metadata exceeds supported size')
            os.lseek(self._fd, 0, os.SEEK_SET)
            state = json.loads(os.read(self._fd, os.fstat(self._fd).st_size).decode('utf-8'))
        except (OSError, UnicodeDecodeError, ValueError) as exc:
            raise ComputeError('GPU lease corrupted; manual recovery required') from exc
        if (state.get('lock_protocol') != self._protocol or state.get('token') != self.token
                or state.get('pid') != os.getpid()):
            raise ComputeError('GPU lease ownership changed; refusing delete')
        assert self._fd is not None and self._path_key is not None
        os.ftruncate(self._fd, 0)
        os.fsync(self._fd)
        self._unlock(self._fd)
        os.close(self._fd)
        self._release_local_key(self._path_key)
        self.token = None
        self._fd = None
        self._path_key = None

    def __enter__(self):
        raise ComputeError('call acquire(attempt_id=...) before entering GPU lease')
