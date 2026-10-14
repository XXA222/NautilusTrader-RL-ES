"""L3-owned Gym environments, bound to SB3's *synchronous* VecEnv API."""
from __future__ import annotations

from collections.abc import Callable
from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
from typing import Literal
import numpy as np

from .config import PolicySpec
from .errors import ContractError

# The callable is supplied by L3; L4 never implements Gym step/reset/settlement.
EnvFactory = Callable[[int, int, Literal["train", "eval"]], object]


def _episode_seed(root_seed: int, episode_index: int) -> int:
    payload = f"rles-env-seed-v1:{root_seed}:{episode_index}".encode("ascii")
    return int.from_bytes(sha256(payload).digest()[:4], "big")


def _spawn_worker(factory: EnvFactory, rank: int, seed: int,
                  phase: Literal["train", "eval"], spec: PolicySpec):
    from gymnasium import Env, Wrapper

    class SeedWrapper(Wrapper):
        def __init__(self, env):
            super().__init__(env)
            self._root_seed = seed
            self._next_episode = 0
            self._first = True

        def reset(self, *, seed=None, options=None):
            if seed is not None and (type(seed) is not int or not 0 <= seed < 2**32):
                raise ContractError("episode reset seed must be uint32")
            if options is not None and not isinstance(options, dict):
                raise ContractError("episode reset options must be a dictionary")
            episode_options = {} if options is None else dict(options)
            if "rles_episode_index" in episode_options:
                raise ContractError("rles_episode_index is reserved by the immutable seed schedule")
            if self._first:
                seed = self._root_seed
                self._first = False
                episode_index = 0
                self._next_episode = 1
            else:
                # L1 owns the immutable environment seed tree. SB3 may pass its
                # algorithm seed through VecEnv.seed(); it must not replace the
                # independently frozen environment stream.
                episode_index = self._next_episode
                seed = _episode_seed(self._root_seed, episode_index)
                self._next_episode += 1
            episode_options["rles_episode_index"] = episode_index
            return self.env.reset(seed=seed, options=episode_options)

        @property
        def rles_seed_schedule_state(self):
            return {"root_seed": self._root_seed, "next_episode": self._next_episode}

        def rles_restore_seed_schedule_state(self, state):
            if (not isinstance(state, dict) or set(state) != {"root_seed", "next_episode"}
                    or type(state["root_seed"]) is not int or state["root_seed"] != self._root_seed
                    or type(state["next_episode"]) is not int or state["next_episode"] < 1):
                raise ContractError("invalid environment seed-schedule checkpoint")
            self._next_episode = state["next_episode"]

    env = factory(rank, seed, phase)
    if not isinstance(env, Env):
        raise ContractError("L3 factory must supply a Gymnasium Env")
    assert_spaces(env, spec)
    return SeedWrapper(env)


@dataclass(frozen=True)
class _SpawnEnvFactory:
    factory_payload: bytes
    rank: int
    seed: int
    phase: Literal["train", "eval"]
    spec: PolicySpec

    @classmethod
    def create(cls, factory: EnvFactory, rank: int, seed: int,
               phase: Literal["train", "eval"], spec: PolicySpec):
        import cloudpickle
        return cls(cloudpickle.dumps(factory), rank, seed, phase, spec)

    def __call__(self):
        import cloudpickle
        factory = cloudpickle.loads(self.factory_payload)
        return _spawn_worker(factory, self.rank, self.seed, self.phase, self.spec)


class BoundedSubprocessVecEnv:
    """SB3 VecEnv semantics over L3's deadline-bounded spawn barrier."""

    def __new__(cls, env_fns: Sequence[Callable[[], object]], policy: PolicySpec,
                *, timeout_s: float = 300.0):
        from stable_baselines3.common.vec_env.base_vec_env import VecEnv
        from gymnasium import spaces
        from .errors import ContractError
        from rles_gym_bridge.vector import BatchBarrierFailed, SyncVectorBarrier

        class _Adapter(VecEnv):
            def __init__(self):
                self._barrier = SyncVectorBarrier(env_fns, timeout_s=max(timeout_s, 30.0))
                self._pending_actions: np.ndarray | None = None
                self._closed = False
                obs_space = spaces.Box(-np.inf, np.inf, shape=(policy.observation_dim,), dtype=np.float32)
                action_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
                try:
                    super().__init__(len(env_fns), obs_space, action_space)
                    if self.render_mode is not None:
                        raise ContractError("native training environments must be non-rendering")
                    # Process startup/import has its own grace interval; decision
                    # and reset barriers use the caller's configured deadline.
                    self._barrier._timeout_s = float(timeout_s)
                except BaseException:
                    self._barrier.close()
                    self._closed = True
                    raise

            def reset(self):
                if self._closed:
                    raise BatchBarrierFailed("closed vector environment")
                results = self._barrier.reset(seeds=self._seeds, options=self._options)
                observations, infos = results
                self.reset_infos = infos
                self._reset_seeds()
                self._reset_options()
                return self._observations(observations)

            @staticmethod
            def _observations(values):
                result = np.asarray(values)
                if (result.dtype != np.float32 or result.ndim != 2 or
                        not np.isfinite(result).all()):
                    raise ContractError("spawn workers returned invalid flat float32 observations")
                return result.copy()

            def step_async(self, actions):
                if self._closed or self._pending_actions is not None:
                    raise BatchBarrierFailed("step already pending or vector environment closed")
                value = np.asarray(actions)
                if (value.shape != (self.num_envs, 1) or value.dtype != np.float32 or
                        not np.isfinite(value).all() or np.any(value < -1) or np.any(value > 1)):
                    raise ContractError("SB3 action batch must be finite float32 [n_envs,1] in [-1,1]")
                self._pending_actions = value.copy()

            def step_wait(self):
                if self._closed or self._pending_actions is None:
                    raise BatchBarrierFailed("step_wait requires a pending action batch")
                actions, self._pending_actions = self._pending_actions, None
                observations, rewards, terminated, truncated, infos = self._barrier.step(actions)
                out = self._observations(observations)
                if (rewards.dtype != np.float32 or rewards.shape != (self.num_envs,) or
                        terminated.shape != (self.num_envs,) or truncated.shape != (self.num_envs,)):
                    raise ContractError("spawn worker result shape/dtype mismatch")
                infos = deepcopy(infos)
                completed = []
                for index in range(self.num_envs):
                    done = bool(terminated[index] or truncated[index])
                    infos[index]["TimeLimit.truncated"] = bool(truncated[index] and not terminated[index])
                    if done:
                        infos[index]["terminal_observation"] = out[index].copy()
                        completed.append(index)
                if completed:
                    # SB3 expects per-env autoreset. The L3 barrier resets only
                    # completed processes at this same synchronous batch edge.
                    out, reset_infos = self._barrier.reset_subset(completed, seeds=self._seeds,
                                                                  options=self._options)
                    for index in completed:
                        self.reset_infos[index] = reset_infos[index]
                    for index in completed:
                        self._seeds[index] = None
                        self._options[index] = {}
                return out, rewards.copy(), np.logical_or(terminated, truncated), infos

            def close(self):
                if not self._closed:
                    try:
                        self._barrier.close()
                    finally:
                        self._closed = True

            def get_attr(self, attr_name, indices=None):
                return self._barrier.get_attr(attr_name, self._get_indices(indices))

            def set_attr(self, attr_name, value, indices=None):
                self._barrier.set_attr(attr_name, value, self._get_indices(indices))

            def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
                return self._barrier.env_method(method_name, method_args, method_kwargs,
                                                self._get_indices(indices))

            def env_is_wrapped(self, wrapper_class, indices=None):
                return self._barrier.env_is_wrapped(wrapper_class, self._get_indices(indices))

            def get_images(self):
                if self.render_mode != "rgb_array":
                    return [None] * self.num_envs
                return self.env_method("render")

        return _Adapter()


def assert_spaces(env: object, spec: PolicySpec) -> None:
    from gymnasium import spaces
    obs = getattr(env, "observation_space", None)
    action = getattr(env, "action_space", None)
    if not isinstance(obs, spaces.Box) or obs.shape != (spec.observation_dim,):
        raise ContractError("L3 must expose a fixed flat Box observation")
    if obs.dtype != np.dtype("float32"):
        raise ContractError("L3 observation dtype must be float32")
    if not isinstance(action, spaces.Box) or action.shape != (1,):
        raise ContractError("L3 action must be one-dimensional Box")
    if action.dtype != np.dtype("float32") or not np.array_equal(action.low, np.array([-1], dtype=np.float32)) or not np.array_equal(action.high, np.array([1], dtype=np.float32)):
        raise ContractError("L3 action bounds/dtype do not match [-1,1] float32")


def make_sync_env(factory: EnvFactory, seeds: tuple[int, ...], phase: Literal["train", "eval"],
                  spec: PolicySpec, *, timeout_s: float = 300.0):
    from stable_baselines3.common.vec_env import DummyVecEnv
    env_fns = [_SpawnEnvFactory.create(factory, i, seed, phase, spec)
               for i, seed in enumerate(seeds)]
    if len(seeds) == 1:
        def creator():
            return _spawn_worker(factory, 0, seeds[0], phase, spec)
        # A single native episode benefits from direct calls and retains the
        # same independently issued environment seed.
        return DummyVecEnv([creator])
    # NautilusTrader's rc6 component/message registries are process scoped;
    # each native environment receives its own spawned Python process.
    return BoundedSubprocessVecEnv(env_fns, spec, timeout_s=timeout_s)


def check_budget(algorithm: str, params: dict, n_envs: int, timesteps: int) -> None:
    if type(timesteps) is not int or timesteps <= 0:
        raise ContractError("total_timesteps must be positive integer")
    if algorithm == "PPO":
        block = int(params.get("n_steps", 2048)) * n_envs
    else:
        block = int(params.get("train_freq", 1)) * n_envs
    if timesteps % block:
        raise ContractError(f"timesteps={timesteps} must divide synchronous rollout block={block}")


def assert_transition_proof(vec, expected_hash: str) -> None:
    """Fail before replay storage if L3's effective transition semantics drift."""
    try:
        observed = vec.get_attr("rles_transition_hash")
    except (AttributeError, ValueError) as exc:
        raise ContractError("L3-021 must expose rles_transition_hash on each Env (ICR-L4-001)") from exc
    if len(observed) != vec.num_envs or any(x != expected_hash for x in observed):
        raise ContractError("In-flight Execution/Risk/Action/Env transition fingerprint drift")


def assert_resume_proof(vec, expected_tokens: tuple[str, ...]) -> None:
    try:
        tokens = vec.get_attr("rles_boundary_token")
    except (AttributeError, ValueError) as exc:
        raise ContractError("L3 must expose its restored boundary tokens (ICR-L4-002)") from exc
    if tuple(tokens) != expected_tokens:
        raise ContractError("L3 runtime state does not match checkpoint-boundary tokens")



def boundary_raw_observations(vec, observation_dim: int) -> np.ndarray:
    """Read L3-owned post-barrier observations; never reset or step L3 here.

    Contract extension ICR-L4-002: rles_boundary_observation returns the
    current *raw* float32 observation associated with rles_boundary_token.
    Exact equality is necessary for an uninterrupted native SB3 continuation.
    """
    try:
        values = vec.get_attr("rles_boundary_observation")
    except (AttributeError, ValueError) as exc:
        raise ContractError("L3 lacks approved rles_boundary_observation resume interface") from exc
    if len(values) != vec.num_envs:
        raise ContractError("L3 boundary observation count differs from n_envs")
    for item in values:
        if (not isinstance(item, np.ndarray) or item.dtype != np.float32 or
                item.shape != (observation_dim,) or not np.isfinite(item).all()):
            raise ContractError("L3 boundary observation must be finite float32 [features]")
    return np.stack(values, axis=0).copy()
