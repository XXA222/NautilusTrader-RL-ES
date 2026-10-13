"""Observation-only VecNormalize with no validation/test statistics updates."""
from __future__ import annotations

from copy import deepcopy
from typing import Any
import numpy as np

from .errors import ContractError


def attach_train(vec: Any, enabled: bool):
    if not enabled:
        return vec
    from stable_baselines3.common.vec_env import VecNormalize
    return VecNormalize(vec, training=True, norm_obs=True, norm_reward=False)


def export_stats(vec: Any) -> dict[str, Any] | None:
    from stable_baselines3.common.vec_env import VecNormalize
    if not isinstance(vec, VecNormalize):
        return None
    if vec.norm_reward or not vec.norm_obs:
        raise ContractError("Reward normalization forbidden; observation normalization required")
    # SB3's runtime field is RunningMeanStd or a dict for Dict spaces; the
    # following attribute guard intentionally rejects the latter.
    rms: Any = vec.obs_rms
    if not hasattr(rms, "mean"):
        raise ContractError("Dict observation normalization not supported")
    return {
        "schema": "rles.vecnorm.obs.v1",
        "mean": np.asarray(rms.mean, dtype=np.float64).tolist(),
        "var": np.asarray(rms.var, dtype=np.float64).tolist(),
        "count": float(rms.count),
        "epsilon": float(vec.epsilon),
        "clip_obs": float(vec.clip_obs),
    }


def apply_stats(vec: Any, stats: dict[str, Any] | None, *, training: bool):
    if stats is None:
        return vec
    if set(stats) != {"schema", "mean", "var", "count", "epsilon", "clip_obs"} or stats["schema"] != "rles.vecnorm.obs.v1":
        raise ContractError("Unknown observation normalizer snapshot")
    from stable_baselines3.common.vec_env import VecNormalize
    result = VecNormalize(vec, training=training, norm_obs=True, norm_reward=False,
                          epsilon=float(stats["epsilon"]), clip_obs=float(stats["clip_obs"]))
    result_rms: Any = result.obs_rms
    mean = np.asarray(stats["mean"], dtype=np.float64)
    var = np.asarray(stats["var"], dtype=np.float64)
    if not hasattr(result_rms, "mean") or not hasattr(result_rms, "var") or not hasattr(result_rms, "count"):
        raise ContractError("Dict observation normalization not supported")
    if mean.shape != result_rms.mean.shape or var.shape != result_rms.var.shape:
        raise ContractError("Normalization statistics schema shape mismatch")
    if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(var)) or np.any(var < 0):
        raise ContractError("Normalization statistics contain invalid numbers")
    count = float(stats["count"])
    if not np.isfinite(count) or count <= 0:
        raise ContractError("Invalid normalizer count")
    if (not np.isfinite(result.epsilon) or not np.isfinite(result.clip_obs)
            or result.epsilon <= 0 or result.clip_obs <= 0):
        raise ContractError("Invalid normalization epsilon/clip")
    result_rms.mean = mean.copy()
    result_rms.var = var.copy()
    result_rms.count = count
    if not training:
        result.training = False
    return result


def freeze_for_eval(train_vec: Any, eval_vec: Any):
    stats = export_stats(train_vec)
    return apply_stats(eval_vec, deepcopy(stats), training=False)


def assert_frozen(vec: Any) -> None:
    from stable_baselines3.common.vec_env import VecNormalize
    if isinstance(vec, VecNormalize) and (vec.training or vec.norm_reward):
        raise ContractError("Evaluation VecNormalize must be training=False, norm_reward=False")
