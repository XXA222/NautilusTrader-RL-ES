from __future__ import annotations

from dataclasses import dataclass
import time

import gymnasium as gym
import numpy as np
import pytest
from gymnasium import spaces

from rles_rl_sb3.config import PolicySpec
from rles_rl_sb3.envs import make_sync_env

pytestmark = pytest.mark.micro_golden


class _ProcessEnv(gym.Env):
    def __init__(self, rank: int, *, max_steps: int, delay: float = 0.0):
        super().__init__()
        self.rank, self.max_steps, self.delay = rank, max_steps, delay
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float32)
        self.action_space = spaces.Box(-1, 1, shape=(1,), dtype=np.float32)
        self.render_mode = None
        self.steps = 0
        self.value = 0

    @property
    def rles_transition_hash(self) -> str:
        return "a" * 64

    def state(self):
        return self.rank, self.steps, self.value

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        return np.array([self.rank * 10 + self.steps], dtype=np.float32), {"rank": self.rank}

    def step(self, action):
        if self.delay:
            time.sleep(self.delay)
        self.steps += 1
        observation = np.array([self.rank * 10 + self.steps], dtype=np.float32)
        truncated = self.steps >= self.max_steps
        return observation, 1.0, False, truncated, {"rank": self.rank, "steps": self.steps}


@dataclass(frozen=True)
class _Factory:
    def __call__(self, rank, _seed, _phase):
        return _ProcessEnv(rank, max_steps=rank + 1)


@dataclass(frozen=True)
class _SlowFactory:
    def __call__(self, rank, _seed, _phase):
        return _ProcessEnv(rank, max_steps=100, delay=4.0 if rank == 0 else 0)


def _policy():
    return PolicySpec(observation_dim=1, observation_schema_hash="b" * 64,
                      action_schema_hash="c" * 64, hidden_layers=(2,), value_layers=(2,))


def test_spawn_vecenv_resets_only_done_workers_and_preserves_terminal_observations():
    vec = make_sync_env(_Factory(), (11, 12), "train", _policy(), timeout_s=10)
    try:
        assert vec.get_attr("rles_transition_hash") == ["a" * 64, "a" * 64]
        vec.seed(999)  # SB3 algorithm seed must not replace the L1 environment seed tree.
        assert vec.reset().tolist() == [[0], [10]]
        assert vec.get_attr("rles_seed_schedule_state") == [
            {"root_seed": 11, "next_episode": 1},
            {"root_seed": 12, "next_episode": 1},
        ]
        vec.env_method("rles_restore_seed_schedule_state",
                       {"root_seed": 11, "next_episode": 7}, indices=0)
        vec.reset()
        assert vec.get_attr("rles_seed_schedule_state")[0] == {
            "root_seed": 11, "next_episode": 8,
        }
        vec.set_attr("value", 7, indices=0)
        assert vec.env_method("state", indices=1) == [(1, 0, 0)]
        observations, rewards, dones, infos = vec.step(np.zeros((2, 1), dtype=np.float32))
        assert observations.tolist() == [[0], [11]]  # Env 0 auto-reset; env 1 advanced.
        assert rewards.tolist() == [1, 1]
        assert dones.tolist() == [True, False]
        assert infos[0]["terminal_observation"].tolist() == [1]
        assert infos[0]["TimeLimit.truncated"] is True
        assert vec.get_attr("steps") == [0, 1]
        assert vec.get_attr("rles_seed_schedule_state")[0] == {
            "root_seed": 11, "next_episode": 9,
        }
        observations, _rewards, dones, infos = vec.step(np.zeros((2, 1), dtype=np.float32))
        assert observations.tolist() == [[0], [10]]
        assert dones.tolist() == [True, True]
        assert [item["terminal_observation"].tolist() for item in infos] == [[1], [12]]
    finally:
        vec.close()


def test_spawn_vecenv_worker_timeout_kills_the_entire_attempt():
    vec = make_sync_env(_SlowFactory(), (1, 2), "train", _policy(), timeout_s=0.2)
    vec.reset()
    vec.step_async(np.zeros((2, 1), dtype=np.float32))
    with pytest.raises(RuntimeError, match="deadline exceeded"):
        vec.step_wait()
    assert all(not process.is_alive() for process in vec._barrier._workers)
    vec.close()
