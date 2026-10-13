"""Short CPU train/save/load/predict smoke for the four native SB3 algorithms.

This checks the installed SB3/PyTorch ABI only. Pendulum is synthetic and is not
an RLES environment or market-execution validation.
"""

from __future__ import annotations

from pathlib import Path
import tempfile
from typing import Any, cast

import gymnasium as gym
from stable_baselines3 import DDPG, PPO, SAC, TD3
from stable_baselines3.common.vec_env import DummyVecEnv
import torch


def make_env() -> gym.Env:
    return gym.make("Pendulum-v1")


def main() -> None:
    torch.set_num_threads(1)
    env = DummyVecEnv([make_env])
    configs: tuple[tuple[str, Any, dict[str, Any]], ...] = (
        ("PPO", PPO, {"n_steps": 64, "batch_size": 32, "n_epochs": 1}),
        ("SAC", SAC, {"learning_starts": 32, "buffer_size": 256, "batch_size": 32,
                      "train_freq": 1, "gradient_steps": 1}),
        ("TD3", TD3, {"learning_starts": 32, "buffer_size": 256, "batch_size": 32,
                       "train_freq": 1, "gradient_steps": 1}),
        ("DDPG", DDPG, {"learning_starts": 32, "buffer_size": 256, "batch_size": 32,
                        "train_freq": 1, "gradient_steps": 1}),
    )
    try:
        with tempfile.TemporaryDirectory(prefix="rles-q0-sb3-") as directory:
            for name, algorithm, params in configs:
                model = algorithm(
                    "MlpPolicy",
                    env,
                    seed=731,
                    device="cpu",
                    verbose=0,
                    policy_kwargs={"net_arch": [16, 16]},
                    **params,
                )
                model.learn(total_timesteps=128)
                artifact = Path(directory) / name
                model.save(str(artifact))
                restored = algorithm.load(str(artifact), env=env, device="cpu")
                observation = cast(Any, env.reset())
                action, _ = restored.predict(observation, deterministic=True)
                assert env.action_space.contains(action[0]), (name, action)
                print(f"PASS {name}: trained={model.num_timesteps}, save/load/predict=ok")
                del model, restored
    finally:
        env.close()


if __name__ == "__main__":
    main()
