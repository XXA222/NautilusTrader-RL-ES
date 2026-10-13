"""SQLite-backed cross-process quota accounting and hard exclusive GPU lease."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
import json
import os
import sqlite3
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
            cur = conn.execute('DELETE FROM lease WHERE token=? AND owner=? AND cpu=? AND ram=? AND gpu=?',
                               (lease.token, lease.owner, lease.cpu_cores, lease.ram_bytes, lease.gpu_count))
            if cur.rowcount != 1:
                raise ComputeError('lease does not exist or does not belong to caller')
            conn.execute('DELETE FROM lease_recovery WHERE token=?', (lease.token,))

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
    """Exclusive lock file per GPU; never delete a foreign/stale-looking lock."""

    def __init__(self, directory: str | Path, gpu_index: int):
        if type(gpu_index) is not int or gpu_index < 0:
            raise ValueError('invalid GPU index')
        self.path = Path(directory) / f'gpu-{gpu_index}.owner'
        self.token: str | None = None

    def acquire(self, *, attempt_id: str) -> str:
        if self.token is not None or not attempt_id:
            raise ComputeError('GPU lease already acquired or attempt_id missing')
        self.path.parent.mkdir(parents=True, exist_ok=True)
        token = uuid.uuid4().hex
        try:
            fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError as exc:
            raise QuotaExceeded('single CUDA learner owner already allocated') from exc
        try:
            with os.fdopen(fd, 'w', encoding='utf-8') as stream:
                json.dump({'token':token, 'attempt_id':attempt_id, 'pid':os.getpid()},stream)
                stream.flush()
                os.fsync(stream.fileno())
        except BaseException:
            self.path.unlink(missing_ok=True)
            raise
        self.token = token
        return token

    def release(self) -> None:
        if self.token is None:
            raise ComputeError('GPU lease not owned')
        try:
            state = json.loads(self.path.read_text(encoding='utf-8'))
        except (OSError, ValueError) as exc:
            raise ComputeError('GPU lease corrupted; manual recovery required') from exc
        if state.get('token') != self.token or state.get('pid') != os.getpid():
            raise ComputeError('GPU lease ownership changed; refusing delete')
        self.path.unlink()
        self.token = None

    def __enter__(self):
        raise ComputeError('call acquire(attempt_id=...) before entering GPU lease')
