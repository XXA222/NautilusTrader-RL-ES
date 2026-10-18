"""Strict, serializable L4 configuration; no algorithm hyperparameter fallback."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from hashlib import sha256
import json
import math
from types import MappingProxyType
from typing import Any, Literal, Mapping

from .errors import ContractError

AlgoName = Literal["PPO", "SAC", "TD3", "DDPG"]
Mode = Literal["cpu", "cuda"]

_COMMON = frozenset({"learning_rate", "gamma"})
_ON = frozenset({
    "n_steps", "batch_size", "n_epochs", "gae_lambda", "clip_range", "ent_coef",
    "vf_coef", "max_grad_norm", "target_kl", "log_std_init", "zero_init_action_head",
})
_OFF = frozenset({"buffer_size", "learning_starts", "batch_size", "tau", "train_freq", "gradient_steps"})
_SAC = frozenset({"ent_coef", "target_entropy"})
_TD3 = frozenset({"policy_delay", "target_policy_noise", "target_noise_clip"})
_ALLOWED = {
    "PPO": _COMMON | _ON,
    "SAC": _COMMON | _OFF | _SAC,
    "TD3": _COMMON | _OFF | _TD3,
    "DDPG": _COMMON | _OFF,  # No borrowed TD3-only controls.
}
_POSITIVE_INTEGERS = frozenset({"n_steps", "batch_size", "n_epochs", "buffer_size", "train_freq", "policy_delay"})
_NONNEG_INTEGERS = frozenset({"learning_starts"})
_SCALAR_FINITE = frozenset({"learning_rate", "gamma", "gae_lambda", "clip_range", "vf_coef", "max_grad_norm", "tau", "target_policy_noise", "target_noise_clip"})


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def digest(value: Any) -> str:
    return sha256(canonical_json(value)).hexdigest()


def _ensure_exact_fields(cls: type, values: Mapping[str, Any]) -> dict[str, Any]:
    accepted = {f.name for f in fields(cls)}
    extra = set(values) - accepted
    if extra:
        raise ContractError(f"Unknown {cls.__name__} keys: {sorted(extra)}")
    return dict(values)


@dataclass(frozen=True)
class PolicySpec:
    """MLP-only, fixed float32 vector input / bounded one-dimensional action."""
    observation_dim: int
    hidden_layers: tuple[int, ...] = (64, 64)
    value_layers: tuple[int, ...] = (64, 64)
    activation: Literal["tanh", "relu"] = "tanh"
    observation_dtype: Literal["float32"] = "float32"
    action_dtype: Literal["float32"] = "float32"
    action_shape: tuple[int, ...] = (1,)
    action_low: float = -1.0
    action_high: float = 1.0
    observation_schema_hash: str = ""
    action_schema_hash: str = ""

    def __post_init__(self) -> None:
        if type(self.observation_dim) is not int or self.observation_dim <= 0:
            raise ContractError("observation_dim must be a positive integer")
        if type(self.hidden_layers) is not tuple or type(self.value_layers) is not tuple or type(self.action_shape) is not tuple:
            raise ContractError("Policy layer sizes and action shape must be immutable tuples")
        if not self.hidden_layers or any(type(x) is not int or x <= 0 for x in self.hidden_layers):
            raise ContractError("nonempty, positive hidden_layers required")
        if not self.value_layers or any(type(x) is not int or x <= 0 for x in self.value_layers):
            raise ContractError("nonempty, positive value_layers required")
        if self.action_shape != (1,) or self.action_low != -1.0 or self.action_high != 1.0:
            raise ContractError("v0.4.0 action must be Box([-1,1], shape=(1,))")
        if self.activation not in ("tanh", "relu"):
            raise ContractError("Only tanh/relu MLP activations are supported")
        if self.observation_dtype != "float32" or self.action_dtype != "float32":
            raise ContractError("Observation/action dtype must be float32")
        for k in ("observation_schema_hash", "action_schema_hash"):
            if not _is_hash(getattr(self, k)):
                raise ContractError(f"{k} must be a SHA256 hex digest from L1/L3")

    @classmethod
    def parse(cls, data: Mapping[str, Any]) -> PolicySpec:
        obj = _ensure_exact_fields(cls, data)
        for key in ("hidden_layers", "value_layers", "action_shape"):
            if key in obj:
                obj[key] = tuple(obj[key])
        return cls(**obj)


def _is_hash(v: Any) -> bool:
    return isinstance(v, str) and len(v) == 64 and all(c in "0123456789abcdef" for c in v)


@dataclass(frozen=True)
class TransitionFingerprint:
    """Every effective transition-function input must be frozen by the owner."""
    env_contract: str
    dataset: str
    feature: str
    feature_scaler: str
    observation: str
    action: str
    action_transform: str
    execution: str
    hard_risk: str
    soft_risk: str
    reward: str
    normalization: str
    engine: str
    adapter: str
    fold: str
    terminal: str

    def __post_init__(self) -> None:
        for f in fields(self):
            if not _is_hash(getattr(self, f.name)):
                raise ContractError(f"TransitionFingerprint.{f.name} requires sha256 hex")

    @property
    def hash(self) -> str:
        return digest(asdict(self))

    @classmethod
    def parse(cls, data: Mapping[str, Any]) -> TransitionFingerprint:
        return cls(**_ensure_exact_fields(cls, data))


@dataclass(frozen=True)
class SeedSpec:
    """Injected by L1 SeedTree; replay RNG follows SB3 global NumPy RNG."""
    algorithm: int
    environments: tuple[int, ...]
    noise: int
    replay: int

    def __post_init__(self) -> None:
        if type(self.environments) is not tuple:
            raise ContractError("SeedSpec.environments must be an immutable tuple")
        if any(type(x) is not int or not 0 <= x < 2**32 for x in (self.algorithm, self.noise, self.replay, *self.environments)):
            raise ContractError("seed entries must be uint32 integers")
        if not self.environments:
            raise ContractError("at least one environment seed required")

    @classmethod
    def parse(cls, data: Mapping[str, Any]) -> SeedSpec:
        obj = _ensure_exact_fields(cls, data)
        if "environments" in obj:
            obj["environments"] = tuple(obj["environments"])
        return cls(**obj)


def _valid_auto_coefficient(value: str) -> bool:
    try:
        parsed = float(value)
    except ValueError:
        return False
    return math.isfinite(parsed) and parsed > 0


@dataclass(frozen=True)
class RLConfig:
    algorithm: AlgoName
    policy: PolicySpec
    fingerprint: TransitionFingerprint
    seeds: SeedSpec
    hyperparameters: Mapping[str, Any] = field(default_factory=dict)
    device: Mode = "cpu"       # "auto" resolved by L6 before construction.
    normalize_observation: bool = False
    normalize_reward: bool = False
    noise_sigma: float = 0.1
    version: str = "0.4.0"
    cuda_allocator_fraction: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.policy, PolicySpec) or not isinstance(self.fingerprint, TransitionFingerprint) or not isinstance(self.seeds, SeedSpec):
            raise ContractError("PolicySpec/TransitionFingerprint/SeedSpec must be parsed frozen DTOs")
        if self.algorithm not in _ALLOWED:
            raise ContractError(f"Unsupported algorithm: {self.algorithm}")
        if self.version != "0.4.0":
            raise ContractError("L4 contract version mismatch")
        if self.device not in ("cpu", "cuda"):
            raise ContractError("device must be concretely resolved before L4")
        if self.cuda_allocator_fraction is not None:
            if (type(self.cuda_allocator_fraction) not in (int, float)
                    or not math.isfinite(self.cuda_allocator_fraction)
                    or not 0 < self.cuda_allocator_fraction <= 1):
                raise ContractError("cuda_allocator_fraction must be finite and in (0,1]")
            if self.device != "cuda":
                raise ContractError("cuda_allocator_fraction requires CUDA device")
        if type(self.normalize_observation) is not bool or type(self.normalize_reward) is not bool:
            raise ContractError("Normalization switches must be bool")
        if type(self.noise_sigma) not in (float, int):
            raise ContractError("noise_sigma must be real numeric, not bool")
        if self.normalize_reward:
            raise ContractError("Reward normalization is disabled in v0.4.0")
        if len(self.seeds.environments) < 1:
            raise ContractError("missing environment seeds")
        if not 0 <= self.noise_sigma <= 2 or not math.isfinite(self.noise_sigma):
            raise ContractError("invalid noise_sigma")
        params = self.hyperparameters
        if not isinstance(params, Mapping):
            raise ContractError("hyperparameters must be a mapping")
        extra = set(params) - _ALLOWED[self.algorithm]
        if extra:
            raise ContractError(f"Unknown or forbidden {self.algorithm} hyperparameters: {sorted(extra)}")
        for k, v in params.items():
            if k in _POSITIVE_INTEGERS and (type(v) is not int or v <= 0):
                raise ContractError(f"{k} must be positive integer")
            if k in _NONNEG_INTEGERS and (type(v) is not int or v < 0):
                raise ContractError(f"{k} must be nonnegative integer")
            if k == "gradient_steps" and (type(v) is not int or (v < 1 and v != -1)):
                raise ContractError("gradient_steps must be >=1 or -1")
            if k in _SCALAR_FINITE and (type(v) not in (int, float) or not math.isfinite(v)):
                raise ContractError(f"{k} must be finite numeric")
            if k == "ent_coef":
                auto = self.algorithm == "SAC" and isinstance(v, str) and (
                    v == "auto" or v.startswith("auto_") and len(v) > 5
                    and _valid_auto_coefficient(v[5:]))
                finite_positive = type(v) in (int, float) and math.isfinite(v) and v >= 0
                if not (auto or finite_positive):
                    raise ContractError("invalid ent_coef")
            if k == "target_entropy" and not (v == "auto" or type(v) in (float, int) and math.isfinite(v)):
                raise ContractError("invalid target_entropy")
            if k == "log_std_init" and (type(v) not in (int, float)
                                         or not math.isfinite(v) or not -20 <= v <= 2):
                raise ContractError("PPO log_std_init must be finite and in [-20,2]")
            if k == "zero_init_action_head" and type(v) is not bool:
                raise ContractError("PPO zero_init_action_head must be Boolean")
        if "learning_rate" in params and not params["learning_rate"] > 0:
            raise ContractError("learning_rate must be > 0")
        if "gamma" in params and not 0 <= params["gamma"] <= 1:
            raise ContractError("gamma outside [0,1]")
        if "tau" in params and not 0 < params["tau"] <= 1:
            raise ContractError("tau outside (0,1]")
        if "gae_lambda" in params and not 0 <= params["gae_lambda"] <= 1:
            raise ContractError("gae_lambda outside [0,1]")
        if "clip_range" in params and not 0 < params["clip_range"] <= 1:
            raise ContractError("PPO clip_range outside (0,1]")
        for k in ("target_policy_noise", "target_noise_clip", "vf_coef", "max_grad_norm"):
            if k in params and params[k] < 0:
                raise ContractError(f"{k} must be nonnegative")
        if self.algorithm == "PPO":
            steps = params.get("n_steps", 2048) * len(self.seeds.environments)
            if steps <= 1:
                raise ContractError("PPO rollout must have at least two transitions")
            batch = params.get("batch_size", 64)
            if steps % batch != 0:
                raise ContractError("PPO minibatch must divide n_steps*n_envs (no truncated batch)")
        if self.algorithm in ("SAC", "PPO") and self.noise_sigma != 0.1:
            raise ContractError("noise_sigma has no effect in PPO/SAC; do not specify a misleading value")
        if self.algorithm in ("TD3", "DDPG") and self.noise_sigma == 0:
            raise ContractError("deterministic off-policy exploration requires nonzero noise")
        if self.algorithm != "PPO" and params.get("buffer_size", 1_000_000) < params.get("batch_size", 256):
            raise ContractError("replay buffer capacity below minibatch size")

        # A frozen dataclass with a mutable dict is not actually frozen.
        # Reject mutation of the live SB3 recipe after fingerprint creation.
        object.__setattr__(self, "hyperparameters", MappingProxyType(dict(params)))

    @property
    def hash(self) -> str:
        identity = {
            "algorithm": self.algorithm, "policy": asdict(self.policy),
            "fingerprint": asdict(self.fingerprint), "seeds": asdict(self.seeds),
            "hyperparameters": dict(self.hyperparameters), "device": self.device,
            "normalize_observation": self.normalize_observation,
            "normalize_reward": self.normalize_reward, "noise_sigma": self.noise_sigma,
            "version": self.version,
        }
        # Preserve pre-budget identities for existing checkpoints and run manifests.
        if self.cuda_allocator_fraction is not None:
            identity["cuda_allocator_fraction"] = self.cuda_allocator_fraction
        return digest(identity)

    @property
    def num_envs(self) -> int:
        return len(self.seeds.environments)

    @classmethod
    def parse(cls, data: Mapping[str, Any]) -> RLConfig:
        obj = _ensure_exact_fields(cls, data)
        obj["policy"] = PolicySpec.parse(obj["policy"])
        obj["fingerprint"] = TransitionFingerprint.parse(obj["fingerprint"])
        obj["seeds"] = SeedSpec.parse(obj["seeds"])
        return cls(**obj)
