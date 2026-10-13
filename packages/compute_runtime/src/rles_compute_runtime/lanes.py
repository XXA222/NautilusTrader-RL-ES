"""Barrier-preserving RL scheduling and candidate-isolated ES scheduling."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from hashlib import sha256
import json

from .errors import ComputeProtocolError
from .transport import validate_key
from .workers import CPUWorkerPool


class RLTrainLane:
    """Do not implement SB3 learning; only schedule a full, synchronous VecEnv step."""

    def __init__(self, pool: CPUWorkerPool, *, timeout_s: float):
        if pool.mode != 'rl' or pool.workers < 1:
            raise ValueError('RLTrainLane requires RL workers')
        self.pool, self.timeout = pool, timeout_s
        self._current_episode: dict[int, object] = {}
        self._expected_step: dict[int, int] = {}
        self._run_identity: tuple[object, ...] | None = None
        self._seen_episodes: dict[int, set[object]] = {}

    def barrier(self, keys: Sequence[Mapping[str, object]],
                actions: Sequence[Mapping[str, object]], *, operation: str = 'step') -> list[object]:
        if len(keys) != self.pool.workers or len(actions) != len(keys) or operation not in {'reset','step'}:
            raise ComputeProtocolError('RL barrier requires exactly one request per environment')
        jobs = []
        identities = set()
        for idx, (key, action) in enumerate(zip(keys, actions)):
            k = validate_key(key)
            if k['env_id'] != idx or k['candidate_id'] != 'none':
                raise ComputeProtocolError('environment/candidate key mismatch')
            # The entire VecEnv barrier belongs to one frozen transition function.
            shape_value = k['shape']
            if not isinstance(shape_value, list):
                raise ComputeProtocolError('invalid tensor shape in validated worker key')
            identities.add(tuple(k[field] if field != 'shape' else tuple(shape_value)
                for field in ('run_id','attempt_id','model_id','schema_version','dtype','shape')))
            if operation == 'reset':
                if k['step_id'] != 0 or k['episode_id'] in self._seen_episodes.get(idx,set()):
                    raise ComputeProtocolError('reset needs a never-used episode ID and step zero')
            elif idx not in self._current_episode or self._current_episode[idx] != k['episode_id'] or k['step_id'] != self._expected_step[idx]:
                raise ComputeProtocolError('stale/unknown RL episode or noncontiguous step')
            jobs.append((idx,k,operation,action))
        if len(identities) != 1:
            raise ComputeProtocolError('mixed run/attempt/model/schema in RL barrier')
        identity = next(iter(identities))
        if self._run_identity is not None and identity != self._run_identity:
            raise ComputeProtocolError('run/attempt/model/schema changed inside RL lane')
        result = self.pool.dispatch(jobs, timeout_s=self.timeout)
        self._run_identity = identity
        for idx, (key, _) in enumerate(zip(keys, actions)):
            k = validate_key(key)
            if operation == 'reset':
                self._current_episode[idx] = k['episode_id']
                self._seen_episodes.setdefault(idx,set()).add(k['episode_id'])
                self._expected_step[idx] = 0
            else:
                self._expected_step[idx] += 1
        return result


class ESEvalLane:
    """Bounded candidate batches; results are always keyed by candidate_id."""

    def __init__(self, pool: CPUWorkerPool, *, timeout_s: float):
        if pool.mode != 'es':
            raise ValueError('ES lane requires ES workers')
        self.pool, self.timeout = pool, timeout_s
        self._identity: tuple[object, ...] | None = None

    @property
    def runtime_signature(self) -> str:
        """Frozen worker recipe used by L5 checkpoint and dispatch identity."""
        recipe = {
            'schema': 1,
            'factory_ref': self.pool.factory_ref,
            'factory_config': self.pool.factory_config,
            'workers': self.pool.workers,
            'torch_threads': self.pool.threads,
            'timeout_s': self.timeout,
        }
        return sha256(json.dumps(recipe, sort_keys=True, separators=(',', ':'),
                                 allow_nan=False).encode()).hexdigest()

    def evaluate(self, candidates: Sequence[tuple[Mapping[str, object], Mapping[str, object]]]) -> dict[str, object]:
        seen = set()
        identities = set()
        for key, _ in candidates:
            k = validate_key(key)
            if not isinstance(k['candidate_id'], str) or k['candidate_id'] == 'none' or k['candidate_id'] in seen:
                raise ComputeProtocolError('candidate IDs must be unique and nonempty')
            seen.add(k['candidate_id'])
            identities.add((k['run_id'], k['attempt_id'], k['model_id'], k['schema_version']))
        if len(identities) > 1:
            raise ComputeProtocolError('ES population mixed run/attempt/model/schema identities')
        identity = next(iter(identities)) if identities else None
        if identity and self._identity is not None and identity != self._identity:
            raise ComputeProtocolError('ES lane cannot change frozen run/attempt/model/schema')
        result = self.pool.dispatch_candidates(candidates, timeout_s=self.timeout)
        if identity:
            self._identity = identity
        return result
