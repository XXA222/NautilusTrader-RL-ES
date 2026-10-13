"""Spawn-only, fail-closed synchronous barrier for independent Gymnasium envs.

This is NOT an SB3 VecEnv. A separate, reviewed adaptation must supply SB3's
reset-on-done and terminal_observation semantics without skipping barriers.
"""
from __future__ import annotations

import multiprocessing as mp
from multiprocessing.connection import wait
from multiprocessing.connection import Connection
import pickle
import time
from typing import Any, Callable, Sequence, cast

import numpy as np


class BatchBarrierFailed(RuntimeError):
    """Discard all trajectories for this attempt; workers cannot continue."""


def _worker_main(pipe: Any, env_factory: Callable[[], Any], env_id: int) -> None:
    env = None
    request_id = -1
    episode_id = -1
    try:
        env = env_factory()
        while True:
            command, request_id, episode_id, step_id, payload = pipe.recv()
            if command == "CLOSE":
                pipe.send((env_id, request_id, episode_id, step_id, "OK", None))
                break
            if command == "RESET":
                if payload["do_reset"]:
                    obs, info = env.reset(seed=payload["seed"], options=payload["options"])
                    result = (obs, info)
                else:
                    result = None
            elif command == "STEP":
                result = env.step(payload["action"])
            elif command == "GET_ATTR":
                result = env.get_wrapper_attr(payload["name"]) if payload["do_query"] else None
            elif command == "SET_ATTR":
                if payload["do_query"]:
                    env.set_wrapper_attr(payload["name"], payload["value"])
                result = None
            elif command == "ENV_METHOD":
                result = (env.get_wrapper_attr(payload["name"])(*payload["args"], **payload["kwargs"])
                          if payload["do_query"] else None)
            elif command == "IS_WRAPPED":
                wrapped = False
                current = env
                if payload["do_query"]:
                    import gymnasium as gym
                    while True:
                        wrapped = wrapped or isinstance(current, payload["wrapper_class"])
                        if not isinstance(current, gym.Wrapper):
                            break
                        current = current.env
                result_wrapped = wrapped if payload["do_query"] else None
            else:
                raise BatchBarrierFailed(f"unknown command: {command!r}")
            pipe.send((env_id, request_id, episode_id, step_id, "OK",
                       result_wrapped if command == "IS_WRAPPED" else result))
    except BaseException as exc:
        try:
            pipe.send((env_id, request_id, episode_id, step_id, "FAIL",
                       (type(exc).__name__, str(exc))))
        except (OSError, EOFError, BrokenPipeError):
            pass
    finally:
        try:
            if env is not None:
                env.close()
        finally:
            pipe.close()


class SyncVectorBarrier:
    @staticmethod
    def _valid_optional_seed(value: Any) -> bool:
        if value is None:
            return True
        if type(value) is not int:
            return False
        return value >= 0

    def __init__(self, env_factories: Sequence[Callable[[], Any]], *, timeout_s: float,
                 max_envs: int = 16) -> None:
        if (not env_factories or type(max_envs) is not int or max_envs <= 0
                or len(env_factories) > max_envs or type(timeout_s) not in (float, int)
                or not 0 < timeout_s <= 3600):
            raise ValueError("invalid bounded worker count/timeout")
        self.num_envs = len(env_factories)
        self._timeout_s = float(timeout_s)
        self._workers: list[Any] = []
        self._pipes: list[Any] = []
        self._closed = False
        self._active = False
        self._request_id = 0
        self._episode_id = 0
        self._step_id = 0
        self._observations: list[Any] = [None] * self.num_envs
        self._reset_infos: list[dict[str, Any]] = [{} for _ in range(self.num_envs)]
        self._needs_reset: set[int] = set()
        ctx = mp.get_context("spawn")
        try:
            for idx, fn in enumerate(env_factories):
                if not callable(fn):
                    raise TypeError("worker factory is not callable")
                pickle.dumps(fn)  # spawn-safe, not a transport permission for untrusted pickle
                parent, child = ctx.Pipe(duplex=True)
                proc = ctx.Process(target=_worker_main, args=(child, fn, idx), daemon=True)
                try:
                    proc.start()
                finally:
                    child.close()
                self._workers.append(proc)
                self._pipes.append(parent)
        except BaseException:
            self._abort()
            raise

    def _abort(self) -> None:
        self._closed = True
        self._active = False
        for conn in self._pipes:
            try:
                conn.close()
            except OSError:
                pass
        for proc in self._workers:
            if proc.is_alive():
                proc.terminate()
        for proc in self._workers:
            proc.join(timeout=1.0)
            if proc.is_alive():
                proc.kill()
                proc.join(timeout=1.0)

    def _exchange(self, command: str, payloads: Sequence[dict[str, Any]]) -> list[Any]:
        if self._closed or len(payloads) != self.num_envs:
            raise BatchBarrierFailed("closed attempt or incomplete action batch")
        request_id, episode_id, step_id = self._request_id, self._episode_id, self._step_id
        deadline = time.monotonic() + self._timeout_s
        try:
            for idx, (pipe, payload, proc) in enumerate(zip(self._pipes, payloads, self._workers)):
                if not proc.is_alive():
                    raise BatchBarrierFailed(f"worker {idx} exited before barrier")
                pipe.send((command, request_id, episode_id, step_id, payload))
            pending = {pipe: i for i, pipe in enumerate(self._pipes)}
            out: list[Any] = [None] * self.num_envs
            while pending:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise BatchBarrierFailed("worker barrier deadline exceeded")
                ready = wait(tuple(pending), timeout=remaining)
                if not ready:
                    raise BatchBarrierFailed("worker barrier deadline exceeded")
                for ready_pipe in ready:
                    pipe = cast(Connection, ready_pipe)
                    idx = pending.pop(pipe)
                    response = pipe.recv()
                    if not isinstance(response, tuple) or len(response) != 6:
                        raise BatchBarrierFailed(f"malformed worker {idx} response")
                    wid, rid, eid, sid, status, value = response
                    if (wid, rid, eid, sid) != (idx, request_id, episode_id, step_id) or status != "OK":
                        raise BatchBarrierFailed(f"worker {idx} misrouted/failed: {value!r}")
                    out[idx] = value
            self._request_id += 1
            return out
        except BaseException:
            self._abort()
            raise

    def reset(self, *, seeds: Sequence[int | None], options: Sequence[dict] | None = None):
        if (self._closed or len(seeds) != self.num_envs
                or any(not self._valid_optional_seed(seed) for seed in seeds)):
            raise BatchBarrierFailed("one valid optional seed required per worker")
        opts = list(options) if options is not None else [{} for _ in seeds]
        if len(opts) != self.num_envs or any(not isinstance(o, dict) for o in opts):
            raise BatchBarrierFailed("one options dictionary required per worker")
        self._episode_id += 1
        self._step_id = 0
        try:
            results = self._exchange("RESET", [{"seed": s, "options": o, "do_reset": True}
                                                for s, o in zip(seeds, opts)])
            if any(not isinstance(r, tuple) or len(r) != 2 or not isinstance(r[1], dict) for r in results):
                raise BatchBarrierFailed("invalid worker reset result")
            if any(not self._transportable(obs) or not self._transportable(info) for obs, info in results):
                raise BatchBarrierFailed("unsafe/native values in reset response")
            obs = np.stack([r[0] for r in results])
            if not np.isfinite(obs).all():
                raise BatchBarrierFailed("non-finite reset observation")
            self._observations = [np.array(r[0], copy=True) for r in results]
            self._reset_infos = [dict(r[1]) for r in results]
            self._active = True
            return obs, [r[1] for r in results]
        except BaseException:
            self._abort()
            raise

    def step(self, actions: Sequence[Any]):
        if self._closed or not self._active or len(actions) != self.num_envs:
            raise BatchBarrierFailed("reset required or action batch incomplete")
        try:
            results = self._exchange("STEP", [{"action": a} for a in actions])
            if any(not isinstance(r, tuple) or len(r) != 5 for r in results):
                raise BatchBarrierFailed("non-Gymnasium worker step result")
            obs, rewards, terminated, truncated, infos = zip(*results)
            if any(not self._transportable(o) or not self._transportable(info)
                   for o, info in zip(obs, infos)):
                raise BatchBarrierFailed("unsafe/native values in step response")
            if any(type(v) not in (bool, np.bool_) for v in (*terminated, *truncated)):
                raise BatchBarrierFailed("invalid done flag")
            if any(not isinstance(v, dict) for v in infos):
                raise BatchBarrierFailed("malformed worker info")
            batch = np.stack(obs)
            rw = np.asarray(rewards, dtype=np.float64)
            if not np.isfinite(batch).all() or not np.isfinite(rw).all():
                raise BatchBarrierFailed("non-finite step batch")
            self._step_id += 1
            self._observations = [np.array(item, copy=True) for item in obs]
            # L3 intentionally does not auto-reset any worker. The caller must
            # explicitly reset the entire cohort to avoid silent account reuse.
            self._needs_reset = {i for i, (term, trunc) in enumerate(zip(terminated, truncated))
                                 if term or trunc}
            if self._needs_reset:
                self._active = False
            else:
                self._active = True
            return batch, rw.astype(np.float32), np.asarray(terminated, dtype=bool), \
                np.asarray(truncated, dtype=bool), list(infos)
        except BaseException:
            self._abort()
            raise

    def reset_subset(self, indices: Sequence[int], *, seeds: Sequence[int | None],
                     options: Sequence[dict[str, Any]] | None = None):
        """Reset only completed workers at the SB3 auto-reset boundary."""
        chosen = tuple(indices)
        if (self._closed or not chosen or not self._needs_reset or not set(chosen) <= self._needs_reset or
                any(type(i) is not int or not 0 <= i < self.num_envs for i in chosen) or
                len(set(chosen)) != len(chosen) or len(seeds) != self.num_envs or
                any(not self._valid_optional_seed(seeds[i]) for i in chosen)):
            raise BatchBarrierFailed("invalid selective worker reset")
        opts = list(options) if options is not None else [{} for _ in seeds]
        if len(opts) != self.num_envs or any(type(o) is not dict for o in opts):
            raise BatchBarrierFailed("one options dictionary required per worker")
        selected = set(chosen)
        results = self._exchange("RESET", [
            {"seed": seeds[i], "options": opts[i], "do_reset": i in selected}
            for i in range(self.num_envs)
        ])
        for i in chosen:
            item = results[i]
            if not isinstance(item, tuple) or len(item) != 2 or not isinstance(item[1], dict):
                self._abort()
                raise BatchBarrierFailed("invalid selective reset result")
            if not self._transportable(item[0]) or not self._transportable(item[1]):
                self._abort()
                raise BatchBarrierFailed("unsafe/native values in reset response")
            self._observations[i] = np.array(item[0], copy=True)
            self._reset_infos[i] = dict(item[1])
            self._needs_reset.remove(i)
        if any(item is None for item in self._observations):
            self._abort()
            raise BatchBarrierFailed("uninitialized worker observation after selective reset")
        self._active = not self._needs_reset
        return np.stack(self._observations), [dict(item) for item in self._reset_infos]

    def _query(self, command: str, payloads: Sequence[dict[str, Any]], indices: Sequence[int]) -> list[Any]:
        chosen = tuple(indices)
        if (any(type(i) is not int or not 0 <= i < self.num_envs for i in chosen)
                or len(set(chosen)) != len(chosen)):
            raise BatchBarrierFailed("invalid worker query indices")
        if not chosen:
            return []
        selected = set(chosen)
        payloads = tuple(dict(payload, do_query=(i in selected)) for i, payload in enumerate(payloads))
        results = self._exchange(command, payloads)
        values = [results[i] for i in chosen]
        for value in values:
            if not self._transportable(value):
                self._abort()
                raise BatchBarrierFailed("native object or unsafe value cannot cross the worker boundary")
        return values

    @staticmethod
    def _transportable(value: Any) -> bool:
        if value is None or type(value) in (str, int, float, bool):
            return not (type(value) is float and not np.isfinite(value))
        if isinstance(value, np.ndarray):
            return value.dtype != object and bool(np.isfinite(value).all())
        if type(value) in (tuple, list):
            return all(SyncVectorBarrier._transportable(item) for item in value)
        if type(value) is dict:
            return all(type(key) is str and SyncVectorBarrier._transportable(item)
                       for key, item in value.items())
        try:
            from gymnasium import spaces
            return isinstance(value, spaces.Space)
        except ImportError:
            return False

    def get_attr(self, name: str, indices: Sequence[int]):
        if type(name) is not str:
            raise BatchBarrierFailed("worker attribute name must be a string")
        return self._query("GET_ATTR", [{"name": name} for _ in range(self.num_envs)], indices)

    def set_attr(self, name: str, value: Any, indices: Sequence[int]):
        if type(name) is not str or not self._transportable(value):
            raise BatchBarrierFailed("worker attribute updates must use transport-safe values")
        payload = {"name": name, "value": value}
        return self._query("SET_ATTR", [payload for _ in range(self.num_envs)], indices)

    def env_method(self, name: str, args: tuple, kwargs: dict, indices: Sequence[int]):
        if (type(name) is not str or type(args) is not tuple or type(kwargs) is not dict
                or not self._transportable(args) or not self._transportable(kwargs)):
            raise BatchBarrierFailed("worker method calls must use transport-safe arguments")
        payload = {"name": name, "args": args, "kwargs": kwargs}
        return self._query("ENV_METHOD", [payload for _ in range(self.num_envs)], indices)

    def env_is_wrapped(self, wrapper_class: type, indices: Sequence[int]):
        from gymnasium import Wrapper
        if not isinstance(wrapper_class, type) or not issubclass(wrapper_class, Wrapper):
            raise BatchBarrierFailed("worker wrapper query requires a Gymnasium Wrapper class")
        payload = {"wrapper_class": wrapper_class}
        return self._query("IS_WRAPPED", [payload for _ in range(self.num_envs)], indices)

    def close(self) -> None:
        if self._closed:
            return
        try:
            # Cooperative close gives native sessions a chance to dispose.
            # It is still bounded by the same hard barrier deadline.
            self._exchange("CLOSE", [{} for _ in range(self.num_envs)])
            self._closed = True
            self._active = False
            for proc in self._workers:
                proc.join(timeout=0.5)
            if any(proc.is_alive() for proc in self._workers):
                self._abort()
            else:
                for conn in self._pipes:
                    conn.close()
        except BaseException:
            self._abort()
            raise

    def __enter__(self) -> SyncVectorBarrier:
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()
