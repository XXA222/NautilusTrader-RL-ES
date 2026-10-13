"""Canonical deterministic actor and exact float32/C-order genome codec.

No stable-baselines3, NautilusTrader, or engine implementation is imported.
The L4 adapter must certify parameter structural equivalence independently.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math
from typing import Mapping

import numpy as np


def _digest(value: object) -> str:
    return sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                             separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ActorGenomeSpec:
    observation_names: tuple[str, ...]
    hidden_sizes: tuple[int, ...]
    activation: str = "tanh"
    output_size: int = 1
    action_scale: float = 1.0
    dtype: str = "float32"
    version: int = 1
    input_scale: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        if (not isinstance(self.observation_names, tuple) or not self.observation_names
                or any(not isinstance(n, str) or not n.strip() for n in self.observation_names)
                or len(set(self.observation_names)) != len(self.observation_names)):
            raise ValueError("observation_names must be a nonempty ordered unique tuple")
        if (not isinstance(self.hidden_sizes, tuple) or not self.hidden_sizes
                or any(type(n) is not int or n <= 0 for n in self.hidden_sizes)):
            raise ValueError("hidden_sizes must be a tuple of positive integers")
        if self.activation not in ("tanh", "relu") or self.output_size != 1:
            raise ValueError("unsupported canonical architecture")
        if (self.dtype != "float32" or type(self.version) is not int or self.version not in (1, 2)
                or isinstance(self.action_scale, bool) or not math.isfinite(self.action_scale)
                or not (0 < self.action_scale <= 1)):
            raise ValueError("invalid canonical scaling, dtype, or schema version")
        if self.version == 1:
            if self.input_scale is not None:
                raise ValueError("version 1 actor must retain identity input scaling")
        elif (type(self.input_scale) is not tuple or len(self.input_scale) != len(self.observation_names)
              or any(type(value) not in (int, float) or not math.isfinite(value) or value <= 0
                     or not np.isfinite(np.float32(value)) or np.float32(value) <= 0
                     for value in self.input_scale)):
            raise ValueError("version 2 actor requires finite positive float32 input scales for every observation")

    @property
    def shapes(self) -> tuple[tuple[str, tuple[int, ...]], ...]:
        dims = (len(self.observation_names), *self.hidden_sizes, self.output_size)
        return tuple(
            pair
            for index, (input_size, output_size) in enumerate(zip(dims[:-1], dims[1:]))
            for pair in (
                (f"layers.{index}.weight", (output_size, input_size)),
                (f"layers.{index}.bias", (output_size,)),
            )
        )

    @property
    def size(self) -> int:
        return sum(math.prod(shape) for _, shape in self.shapes)

    @property
    def signature(self) -> str:
        values = {
            "version": self.version,
            "observation_names": self.observation_names,
            "hidden_sizes": self.hidden_sizes,
            "activation": self.activation,
            "output_size": self.output_size,
            "action_scale": self.action_scale,
            "dtype": self.dtype,
            "parameter_shapes": self.shapes,
            "parameter_layout": "C",
            "output_activation": "tanh",
        }
        if self.version == 2:
            values["input_scale"] = self.input_scale
        return _digest(values)


def encode(spec: ActorGenomeSpec, state: Mapping[str, np.ndarray]) -> np.ndarray:
    """Strict state -> vector; preserves float32 bit patterns."""
    if not isinstance(spec, ActorGenomeSpec):
        raise TypeError("spec must be ActorGenomeSpec")
    if set(state.keys()) != {name for name, _ in spec.shapes}:
        raise ValueError("missing or additional actor parameter keys")
    pieces = []
    for name, shape in spec.shapes:
        value = np.asarray(state[name])
        if value.shape != shape or value.dtype != np.dtype("float32"):
            raise ValueError(f"parameter shape or dtype mismatch: {name}")
        if not np.isfinite(value).all():
            raise ValueError(f"nonfinite actor parameter: {name}")
        pieces.append(value.ravel(order="C"))
    return np.concatenate(pieces).astype(np.float32, copy=True)


def decode(spec: ActorGenomeSpec, genome: np.ndarray) -> dict[str, np.ndarray]:
    if not isinstance(spec, ActorGenomeSpec):
        raise TypeError("spec must be ActorGenomeSpec")
    value = np.asarray(genome)
    if (value.dtype != np.dtype("float32") or value.shape != (spec.size,)
            or not np.isfinite(value).all()):
        raise ValueError("genome must be finite float32 with exact schema size")
    state: dict[str, np.ndarray] = {}
    start = 0
    for name, shape in spec.shapes:
        length = math.prod(shape)
        state[name] = value[start:start + length].reshape(shape, order="C").copy()
        start += length
    return state


def genome_hash(spec: ActorGenomeSpec, genome: np.ndarray) -> str:
    value = np.asarray(genome)
    if value.dtype != np.dtype("float32") or value.shape != (spec.size,) or not np.isfinite(value).all():
        raise ValueError("invalid genome for hashing")
    data = bytes.fromhex(spec.signature) + value.astype("<f4", copy=False).tobytes(order="C")
    return sha256(data).hexdigest()


def policy_inputs(spec: ActorGenomeSpec, observations: np.ndarray) -> np.ndarray:
    """Apply the frozen policy-only scale; native observation facts stay raw."""
    obs = np.asarray(observations)
    if (obs.dtype != np.dtype("float32") or obs.ndim != 2
            or obs.shape[1] != len(spec.observation_names) or obs.shape[0] == 0
            or not np.isfinite(obs).all()):
        raise ValueError("observations must be nonempty finite float32 [batch, features]")
    if spec.input_scale is None:
        return obs
    scaled = obs * np.asarray(spec.input_scale, dtype=np.float32)
    if not np.isfinite(scaled).all():
        raise FloatingPointError("canonical actor input scaling produced nonfinite values")
    return scaled


def act(spec: ActorGenomeSpec, genome: np.ndarray, observations: np.ndarray) -> np.ndarray:
    obs = policy_inputs(spec, observations)
    parameters = decode(spec, genome)
    activations = obs
    for layer in range(len(spec.hidden_sizes) + 1):
        activations = activations @ parameters[f"layers.{layer}.weight"].T
        activations = activations + parameters[f"layers.{layer}.bias"]
        if layer < len(spec.hidden_sizes):
            activations = np.tanh(activations) if spec.activation == "tanh" else np.maximum(activations, 0)
    actions = np.tanh(activations) * np.float32(spec.action_scale)
    if not np.isfinite(actions).all():
        raise FloatingPointError("canonical actor produced nonfinite output")
    return np.clip(actions, -spec.action_scale, spec.action_scale).astype(np.float32, copy=False)
