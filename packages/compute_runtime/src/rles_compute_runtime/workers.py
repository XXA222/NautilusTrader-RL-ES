"""Spawn-only isolated CPU engines supplied by an external trusted L3 factory."""
from __future__ import annotations

import importlib
import multiprocessing as mp
import os
import queue
import time
import threading
from collections.abc import Mapping, Sequence
from multiprocessing.queues import Queue as MPQueue
from multiprocessing.process import BaseProcess

from .errors import ComputeProtocolError, DeviceUnavailable, WorkerFailure
from .resources import ResourceQuota, Lease
from .transport import enforce_response, small_json, validate_key
from .validation import valid_timeout
from .windows_job import WindowsWorkerJob


def _json_object(value: object, label: str) -> dict[str, object]:
    parsed = small_json(value)
    if not isinstance(parsed, Mapping) or any(not isinstance(key, str) for key in parsed):
        raise ComputeProtocolError(f'{label} must be a string-keyed JSON object')
    return dict(parsed)


def _load_factory(ref: str):
    if not isinstance(ref, str) or ref.count(':') != 1:
        raise ComputeProtocolError('factory must be installed module:callable')
    module, attribute = ref.split(':')
    if not module or not attribute or attribute.startswith('_'):
        raise ComputeProtocolError('invalid factory reference')
    obj = importlib.import_module(module)
    for part in attribute.split('.'):
        if part.startswith('_'):
            raise ComputeProtocolError('private factory attributes prohibited')
        obj = getattr(obj, part)
    if not callable(obj):
        raise ComputeProtocolError('factory is not callable')
    return obj


def _main(inbox, outbox, factory_ref: str, factory_config: dict, mode: str, threads: int, startup_gate=None):
    os.environ['OMP_NUM_THREADS'] = str(threads)
    os.environ['MKL_NUM_THREADS'] = str(threads)
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # CPU Engine worker is never a CUDA learner
    # No torch/CUDA initialisation in the parent, no fork, and no engine sharing.
    session = None
    try:
        if startup_gate is not None:
            while not startup_gate.wait(0.1):
                parent = mp.parent_process()
                if parent is not None and not parent.is_alive():
                    return
        factory = _load_factory(factory_ref)
        if mode == 'rl':
            # READY means the actual EngineSession exists, not just that a module imported.
            session = factory(small_json(factory_config))
        outbox.put({'control': 'READY'})
        while True:
            msg = inbox.get()
            if msg is None:
                break
            key = validate_key(msg['key'])
            operation = msg['op']
            payload = _json_object(msg['payload'], 'worker operation payload')
            try:
                if mode == 'rl':
                    if session is None:
                        raise ComputeProtocolError('RL engine disappeared after startup')
                    if operation == 'reset':
                        result = session.reset(**payload)
                    elif operation == 'step':
                        result = session.step(**payload)
                    else:
                        raise ComputeProtocolError('unsupported RL operation')
                else:
                    if operation != 'evaluate':
                        raise ComputeProtocolError('ES worker only accepts evaluate')
                    # Fresh L3 engine per candidate, including on error paths.
                    single = factory(small_json(factory_config))
                    try:
                        result = single.evaluate(**payload)
                    finally:
                        close = getattr(single, 'close', None)
                        if callable(close):
                            close()
                outbox.put({'key':key, 'status':'OK', 'payload':small_json(result)})
            except BaseException as exc:
                outbox.put({'key':key, 'status':'ERROR', 'payload':{'type':type(exc).__name__}})
                break  # fail closed: never reuse contaminated engine
    except BaseException:
        try:
            outbox.put({'control':'START_FAILED'})
        except Exception:
            pass
    finally:
        if session is not None:
            close = getattr(session, 'close', None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass


class CPUWorkerPool:
    """One independent Engine per RL worker; ES candidate sessions are isolated.

    This is an execution primitive, not a replacement for the upstream L3 VecEnv.
    The caller must supply *frozen* contract-valid action/observation payloads.
    """

    def __init__(self, *, workers: int, factory_ref: str, factory_config: Mapping[str, object],
                 mode: str, quota: ResourceQuota, lease_owner: str,
                 ram_bytes_per_worker: int, cpu_cores_per_worker: int = 1,
                 startup_timeout_s: float = 30, torch_threads: int = 1,
                 enforce_memory_limit: bool = False):
        if (type(workers) is not int or not 1 <= workers <= 128
            or mode not in {'rl', 'es'} or type(torch_threads) is not int or torch_threads < 1):
            raise ValueError('invalid worker pool specification')
        if (not isinstance(quota, ResourceQuota) or not isinstance(lease_owner, str) or not lease_owner
            or type(ram_bytes_per_worker) is not int or ram_bytes_per_worker < 1
            or type(cpu_cores_per_worker) is not int or cpu_cores_per_worker < 1):
            raise ValueError('explicit resource quota and per-worker resource budget required')
        if not valid_timeout(startup_timeout_s):
            raise ValueError('startup timeout must be positive, finite and platform-supported')
        if type(enforce_memory_limit) is not bool:
            raise ValueError('memory enforcement flag must be Boolean')
        if enforce_memory_limit and os.name != 'nt':
            raise DeviceUnavailable('Windows Job memory enforcement requested on an unsupported platform')
        self.enforce_memory_limit = enforce_memory_limit
        self._jobs: list[WindowsWorkerJob] = []
        self.quota, self.lease_owner = quota, lease_owner
        self.ram_bytes_per_worker, self.cpu_cores_per_worker = ram_bytes_per_worker, cpu_cores_per_worker
        self._lease: Lease | None = None
        self.workers = workers
        self.factory_ref = factory_ref
        self.factory_config = _json_object(dict(factory_config), 'worker factory configuration')
        self.mode = mode
        self.timeout = startup_timeout_s
        self.threads = torch_threads
        self._ctx = mp.get_context('spawn')
        self._processes: list[BaseProcess] = []
        self._inboxes: list[MPQueue] = []
        self._outboxes: list[MPQueue] = []
        self._closed = False
        self._dispatch_lock = threading.RLock()
        self._dispatching = False

    def start(self) -> None:
        if self._processes or self._closed:
            raise WorkerFailure('pool can be started only once')
        try:
            self._lease = self.quota.acquire(self.lease_owner, cpu_cores=self.workers*self.cpu_cores_per_worker,
                ram_bytes=self.workers*self.ram_bytes_per_worker)
            job_names = (self.quota.enable_windows_recovery(
                self._lease, workers=self.workers, memory_bytes=self.ram_bytes_per_worker,
            ) if self.enforce_memory_limit else ())
            for idx in range(self.workers):
                inbox, outbox = self._ctx.Queue(maxsize=1), self._ctx.Queue(maxsize=1)
                self._inboxes.append(inbox); self._outboxes.append(outbox)
                config = dict(self.factory_config)
                config['worker_index'] = idx
                gate = self._ctx.Event() if self.enforce_memory_limit else None
                job = (WindowsWorkerJob(self.ram_bytes_per_worker, name=job_names[idx])
                       if self.enforce_memory_limit else None)
                if job is not None:
                    self._jobs.append(job)
                process = self._ctx.Process(target=_main,
                    args=(inbox,outbox,self.factory_ref,config,self.mode,self.threads,gate), daemon=True)
                process.start()
                self._processes.append(process)
                if job is not None:
                    assert process.pid is not None and gate is not None
                    job.attach(process.pid)
                    gate.set()
            deadline = time.monotonic() + self.timeout
            for index, outbox in enumerate(self._outboxes):
                try:
                    ready = outbox.get(timeout=max(0.001, deadline - time.monotonic()))
                except queue.Empty as exc:
                    raise WorkerFailure(f'worker {index} failed to start before deadline') from exc
                if ready != {'control':'READY'} or not self._processes[index].is_alive():
                    raise WorkerFailure(f'worker {index} startup rejected')
            if self.enforce_memory_limit:
                self.quota.seal_windows_recovery(self._lease)
        except BaseException:
            self.close()
            raise

    def dispatch(self, jobs: Sequence[tuple[int, Mapping[str, object], str, Mapping[str, object]]],
                 *, timeout_s: float) -> list[object]:
        """Never allow simultaneous callers to interleave keyed responses."""
        if not self._dispatch_lock.acquire(blocking=False):
            raise WorkerFailure('worker pool already has an active dispatch')
        if self._dispatching:
            self._dispatch_lock.release()
            raise WorkerFailure('worker pool already has an active dispatch')
        self._dispatching = True
        try:
            return self._dispatch_barrier(jobs, timeout_s=timeout_s)
        finally:
            self._dispatching = False
            self._dispatch_lock.release()

    def _dispatch_barrier(self, jobs, *, timeout_s):
        """Dispatch ≤one in-flight task per worker; return in input order."""
        if self.mode != 'rl' or not self._processes or self._closed or not valid_timeout(timeout_s):
            raise WorkerFailure('pool not running or invalid timeout')
        ids = [j[0] for j in jobs]
        if (any(type(i) is not int or i < 0 or i >= self.workers for i in ids)
            or len(ids) != len(set(ids))):
            raise ComputeProtocolError('duplicate/unallocated worker in batch')
        if not jobs:
            return []
        deadline = time.monotonic() + timeout_s
        validated = [(worker, validate_key(key), op, small_json(dict(payload))) for worker,key,op,payload in jobs]
        try:
            for worker,key,op,payload in validated:
                if not self._processes[worker].is_alive():
                    raise WorkerFailure(f'worker {worker} dead before submit')
                self._inboxes[worker].put({'key':key,'op':op,'payload':payload},
                                         timeout=max(0.001, deadline-time.monotonic()))
            outputs = []
            for worker,key,_,_ in validated:
                try:
                    message = self._outboxes[worker].get(timeout=max(0.001,deadline-time.monotonic()))
                except queue.Empty as exc:
                    raise WorkerFailure(f'worker {worker} timeout (no stale action retry)') from exc
                outputs.append(enforce_response(key, message))
            return outputs
        except BaseException:
            self.close()  # release all Workers; never keep an out-of-sync RL barrier
            raise

    def dispatch_candidates(self, candidates, *, timeout_s: float) -> dict[str, object]:
        """Reject reentrant dispatch; preserve candidate attribution."""
        if not self._dispatch_lock.acquire(blocking=False):
            raise WorkerFailure('worker pool already has an active dispatch')
        if self._dispatching:
            self._dispatch_lock.release()
            raise WorkerFailure('worker pool already has an active dispatch')
        self._dispatching = True
        try:
            return self._dispatch_candidates(candidates, timeout_s=timeout_s)
        finally:
            self._dispatching = False
            self._dispatch_lock.release()

    def _dispatch_candidates(self, candidates, *, timeout_s: float) -> dict[str, object]:
        """ES completion-driven scheduling, one in-flight candidate per worker."""
        if self.mode != 'es' or self._closed or not self._processes or not valid_timeout(timeout_s):
            raise WorkerFailure('ES pool unavailable')
        candidates = [(validate_key(key), small_json(dict(payload))) for key,payload in candidates]
        ids = [key['candidate_id'] for key,_ in candidates]
        if any(not isinstance(cid,str) or cid == 'none' for cid in ids) or len(ids)!=len(set(ids)):
            raise ComputeProtocolError('duplicate/invalid ES candidate IDs')
        pending = iter(candidates)
        inflight = {}
        answers = {}
        def offer(worker: int):
            try:
                key,payload = next(pending)
            except StopIteration:
                return
            if not self._processes[worker].is_alive():
                raise WorkerFailure('ES worker died')
            self._inboxes[worker].put({'key':key,'op':'evaluate','payload':payload},timeout=timeout_s)
            inflight[worker]=(key,time.monotonic()+timeout_s)
        try:
            for worker in range(self.workers):
                offer(worker)
            while inflight:
                progress = False
                for worker,(key,deadline) in list(inflight.items()):
                    try:
                        msg=self._outboxes[worker].get_nowait()
                    except queue.Empty:
                        # multiprocessing.Queue.empty() can report True while
                        # a completed result is still crossing its feeder
                        # pipe, even after the worker has exited. The request
                        # deadline is the authoritative failure boundary.
                        if time.monotonic() >= deadline:
                            raise WorkerFailure(f'ES worker {worker} candidate deadline exceeded')
                        continue
                    answers[key['candidate_id']]=enforce_response(key,msg)
                    del inflight[worker]
                    offer(worker)
                    progress=True
                if not progress:
                    time.sleep(0.005)
            return answers
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        # Safe against an external shutdown racing with barrier/candidate IPC.
        with self._dispatch_lock:
            self._close_unlocked()

    def resource_evidence(self) -> list[dict]:
        if not self.enforce_memory_limit:
            return [{"enforcement": "quota_only", "workers": self.workers,
                     "workers_started": len(self._processes), "ram_bytes_per_worker": self.ram_bytes_per_worker}]
        return [job.snapshot() for job in self._jobs]

    def _close_unlocked(self) -> None:
        if self._closed:
            return
        self._closed = True
        for inbox in self._inboxes:
            try:
                inbox.put_nowait(None)
            except (queue.Full, ValueError, OSError):
                pass
        for process in self._processes:
            process.join(timeout=1)
            if process.is_alive():
                process.terminate()
                process.join(timeout=3)
                if process.is_alive():
                    process.kill()
                    process.join(timeout=3)
        # If any engine survived all termination attempts, KEEP the resource
        # reservation: releasing it here would allow another job to oversell RAM.
        job_errors = []
        for job in self._jobs:
            try:
                job.close()
            except Exception as exc:
                job_errors.append(str(exc))
        for process in self._processes:
            process.join(timeout=0.1)
        survivors = [process.pid for process in self._processes if process.is_alive()]
        for stream in (*self._inboxes, *self._outboxes):
            stream.cancel_join_thread()
            stream.close()
            stream.join_thread()
        if survivors or job_errors:
            raise WorkerFailure(f'workers/tree cleanup incomplete {survivors}: {job_errors}; quota lease retained for manual recovery')
        if self._lease is not None:
            self.quota.release(self._lease)
            self._lease = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *_):
        self.close()
