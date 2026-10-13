from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import pytest
import torch
from stable_baselines3 import DDPG, PPO, SAC, TD3
from stable_baselines3.common.vec_env import DummyVecEnv

pytestmark = pytest.mark.algorithm_smoke


@pytest.mark.parametrize(
    ("name", "algorithm", "parameters"),
    (
        ("PPO", PPO, {"n_steps": 64, "batch_size": 32, "n_epochs": 1}),
        ("SAC", SAC, {"learning_starts": 32, "buffer_size": 256, "batch_size": 32,
                      "train_freq": 1, "gradient_steps": 1}),
        ("TD3", TD3, {"learning_starts": 32, "buffer_size": 256, "batch_size": 32,
                       "train_freq": 1, "gradient_steps": 1}),
        ("DDPG", DDPG, {"learning_starts": 32, "buffer_size": 256, "batch_size": 32,
                        "train_freq": 1, "gradient_steps": 1}),
    ),
)
def test_cpu_sb3_train_save_load_predict(name, algorithm, parameters, tmp_path: Path) -> None:
    torch.set_num_threads(1)
    env = DummyVecEnv([lambda: gym.make("Pendulum-v1")])
    try:
        model = algorithm(
            "MlpPolicy",
            env,
            seed=731,
            device="cpu",
            verbose=0,
            policy_kwargs={"net_arch": [16, 16]},
            **parameters,
        )
        model.learn(total_timesteps=128)
        artifact = tmp_path / name
        model.save(str(artifact))
        restored = algorithm.load(str(artifact), env=env, device="cpu")
        action, _ = restored.predict(env.reset(), deterministic=True)
        assert env.action_space.contains(action[0])
    finally:
        env.close()
