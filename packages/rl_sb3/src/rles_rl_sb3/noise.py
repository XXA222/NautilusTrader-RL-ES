"""SB3-compatible exploration noise with independent, restorable seed streams."""
from __future__ import annotations
from copy import deepcopy
import numpy as np
from stable_baselines3.common.noise import ActionNoise, VectorizedActionNoise


class SeededNormalNoise(ActionNoise):
    def __init__(self, seed: int, sigma: float):
        super().__init__()
        self.sigma = sigma
        self.generator = np.random.default_rng(seed)

    def __call__(self):
        return np.asarray(self.generator.normal(0.0, self.sigma, size=(1,)), dtype=np.float32)

    def reset(self) -> None:
        # Independent successive episode noise; do not reseed the stream.
        return None


def make_noise(sigma: float, noise_seed: int, n_envs: int):
    if n_envs == 1:
        return SeededNormalNoise(noise_seed, sigma)
    master = np.random.SeedSequence(noise_seed)
    children = master.spawn(n_envs)
    wrapper = VectorizedActionNoise(SeededNormalNoise(noise_seed, sigma), n_envs)
    wrapper.noises = [SeededNormalNoise(int(c.generate_state(1)[0]), sigma) for c in children]
    return wrapper


def capture_noise_state(noise):
    if noise is None:
        return None
    if hasattr(noise, "generator"):
        return deepcopy(noise.generator.bit_generator.state)
    if hasattr(noise, "noises"):
        return [capture_noise_state(n) for n in noise.noises]
    raise TypeError("Non-seedable action noise attached to model")


def restore_noise_state(noise, state) -> None:
    if noise is None and state is None:
        return
    if hasattr(noise, "generator") and isinstance(state, dict):
        noise.generator.bit_generator.state = state
        return
    if hasattr(noise, "noises") and isinstance(state, list) and len(noise.noises) == len(state):
        for item, snap in zip(noise.noises, state, strict=True):
            restore_noise_state(item, snap)
        return
    raise TypeError("Noise stream checkpoint does not match SB3 noise object")
