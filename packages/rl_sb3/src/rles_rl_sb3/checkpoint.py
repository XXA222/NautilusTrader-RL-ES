"""L1 stage-boundary checkpoint (not a mid-step/execution-engine snapshot)."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from decimal import Decimal, InvalidOperation
from hashlib import sha256
from pathlib import Path
from uuid import uuid4
import json
import os
import random
import shutil
import sys
import platform
from typing import Any

import numpy as np
import torch

from .config import RLConfig, canonical_json
from .errors import ArtifactError, ContractError, ReplayMismatch
from .noise import capture_noise_state, restore_noise_state
from .normalize import apply_stats, export_stats


def sha_file(path: Path) -> str:
    h = sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _json_dump(path: Path, payload: Any):
    path.write_bytes(canonical_json(payload))


def numeric_profile() -> dict[str, str]:
    """Resume is pinned to one approved numerical runtime, not device-portable."""
    import gymnasium
    return {
        "implementation": sys.implementation.name,
        "python": platform.python_version(),
        "platform": platform.system(),
        "machine": platform.machine(),
        "numpy": np.__version__,
        "gymnasium": gymnasium.__version__,
        "torch_cuda_runtime": str(torch.version.cuda),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
    }


def capture_rng() -> dict[str, Any]:
    n: Any = np.random.get_state()
    if not isinstance(n[1], np.ndarray):
        raise ArtifactError("unsupported NumPy RNG state")
    return {
        "python": repr(random.getstate()),   # Non-executable text; restored via ast.literal_eval.
        "numpy": [n[0], n[1].tolist(), int(n[2]), int(n[3]), float(n[4])],
        "torch_cpu": torch.random.get_rng_state().tolist(),
        "torch_cuda": [x.tolist() for x in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else [],
    }


def restore_rng(data: dict[str, Any]) -> None:
    import ast
    if set(data) != {"python", "numpy", "torch_cpu", "torch_cuda"}:
        raise ArtifactError("RNG snapshot schema mismatch")
    random.setstate(ast.literal_eval(data["python"]))
    n = data["numpy"]
    np.random.set_state((n[0], np.asarray(n[1], dtype=np.uint32), n[2], n[3], n[4]))
    torch.random.set_rng_state(torch.tensor(data["torch_cpu"], dtype=torch.uint8))
    if data["torch_cuda"]:
        if not torch.cuda.is_available() or len(data["torch_cuda"]) != torch.cuda.device_count():
            raise ArtifactError("CUDA RNG snapshot cannot restore on this device topology")
        torch.cuda.set_rng_state_all([torch.tensor(x, dtype=torch.uint8) for x in data["torch_cuda"]])


def _seed_schedule(vec, expected_seeds: tuple[int, ...]) -> list[dict[str, int]]:
    try:
        states = vec.get_attr("rles_seed_schedule_state")
    except (AttributeError, ValueError) as exc:
        raise ContractError("L3/L4 Env must expose its deterministic seed-schedule state") from exc
    if len(states) != len(expected_seeds):
        raise ContractError("environment seed-schedule count differs from SeedTree")
    checked = []
    for rank, (state, expected) in enumerate(zip(states, expected_seeds)):
        if (not isinstance(state, dict) or set(state) != {"root_seed", "next_episode"}
                or type(state["root_seed"]) is not int or state["root_seed"] != expected
                or type(state["next_episode"]) is not int or state["next_episode"] < 1):
            raise ContractError(
                f"invalid environment seed-schedule state at rank {rank}: "
                f"expected root {expected}, observed {state!r}"
            )
        checked.append({"root_seed": state["root_seed"], "next_episode": state["next_episode"]})
    return checked


def _episode_replays(vec, cfg: RLConfig, boundary: Boundary) -> list[dict[str, Any]]:
    try:
        values = vec.env_method("rles_export_replay")
    except (AttributeError, ValueError, TypeError) as exc:
        raise ContractError("L3 Env must expose its replay journal at a checkpoint boundary") from exc
    if not isinstance(values, (list, tuple)) or len(values) != cfg.num_envs:
        raise ContractError("L3 replay journal count differs from the frozen environment count")
    v1_fields = {"schema", "transition_hash", "actions", "boundary_token"}
    v2_fields = v1_fields | {"episode_index", "episode_start_offset"}
    checked: list[dict[str, Any]] = []
    for index, value in enumerate(values):
        if (not isinstance(value, dict)
                or (value.get("schema") == "rles.episode-replay.v1" and set(value) != v1_fields)
                or (value.get("schema") == "rles.episode-replay.v2" and set(value) != v2_fields)
                or value.get("schema") not in {"rles.episode-replay.v1", "rles.episode-replay.v2"}
                or value["transition_hash"] != cfg.fingerprint.hash
                or value["boundary_token"] != boundary.env_tokens[index]
                or not isinstance(value["actions"], list)):
            raise ContractError("L3 replay journal differs from its approved checkpoint boundary")
        if value["schema"] == "rles.episode-replay.v2" and (
            type(value["episode_index"]) is not int or value["episode_index"] < 0
            or type(value["episode_start_offset"]) is not int
            or value["episode_start_offset"] < 0
        ):
            raise ContractError("L3 replay journal has invalid rotating-window identity")
        for action in value["actions"]:
            if not isinstance(action, str):
                raise ContractError("L3 replay journal actions must be decimal strings")
            try:
                parsed = Decimal(action)
            except InvalidOperation as exc:
                raise ContractError("L3 replay journal contains an invalid action") from exc
            if not parsed.is_finite() or not Decimal(-1) <= parsed <= Decimal(1):
                raise ContractError("L3 replay journal action is outside the policy range")
        checked.append(value)
    return checked


def checkpoint_episode_replays(path: Path, cfg: RLConfig, *, trusted_source: bool) -> tuple[dict[str, Any], ...]:
    """Read L3 action journals only after verifying the containing trusted checkpoint."""
    manifest = checkpoint_verify(path, cfg, trusted_source=trusted_source)
    boundary = Boundary.parse(manifest["boundary"])
    boundary.check(cfg.num_envs)
    values = json.loads((Path(path) / "episode_replays.json").read_bytes())
    return tuple(_episode_replays_from_snapshot(values, cfg, boundary))


def _episode_replays_from_snapshot(values: Any, cfg: RLConfig,
                                   boundary: Boundary) -> list[dict[str, Any]]:
    class Snapshot:
        def env_method(self, _method: str) -> Any:
            return values

    return _episode_replays(Snapshot(), cfg, boundary)


@dataclass(frozen=True)
class Boundary:
    """Cross-line token: L3/L2 restore their own cursor/engine before RL resumes."""
    stage: str
    step_cursor: int
    env_tokens: tuple[str, ...]
    no_inflight_orders: bool
    no_pending_step: bool

    def check(self, n_envs: int) -> None:
        if not self.stage or type(self.step_cursor) is not int or self.step_cursor < 0:
            raise ContractError("Invalid stage/cursor")
        if len(self.env_tokens) != n_envs or any(not isinstance(t, str) or not t for t in self.env_tokens):
            raise ContractError("Each L3 env must provide an opaque restorable boundary token")
        if not self.no_inflight_orders or not self.no_pending_step:
            raise ContractError("L1 checkpoint requires a fully settled stage barrier")

    @classmethod
    def parse(cls, val: dict[str, Any]) -> Boundary:
        if set(val) != set(cls.__dataclass_fields__):
            raise ArtifactError("Unknown stage boundary fields")
        return cls(stage=val["stage"], step_cursor=val["step_cursor"],
                   env_tokens=tuple(val["env_tokens"]),
                   no_inflight_orders=val["no_inflight_orders"],
                   no_pending_step=val["no_pending_step"])


def checkpoint_save(model, vec, cfg: RLConfig, dest: Path, boundary: Boundary, *, attempt_id: str) -> Path:
    boundary.check(cfg.num_envs)
    if not attempt_id or "/" in attempt_id or "\\" in attempt_id or attempt_id in (".", ".."):
        raise ContractError("Invalid attempt_id")
    if model.num_timesteps <= 0 or model.num_timesteps % cfg.num_envs:
        raise ContractError("Checkpoint requires completed SB3 vector decision batches")
    if boundary.step_cursor * cfg.num_envs != model.num_timesteps:
        raise ContractError("Stage boundary cursor must equal cumulative vector decision count")
    if cfg.algorithm == "PPO" and model.num_timesteps % (
        cfg.hyperparameters.get("n_steps", 2048) * cfg.num_envs
    ):
        raise ContractError("PPO checkpoint must end at the exact native rollout boundary")
    last_obs = getattr(model, "_last_obs", None)
    last_starts = getattr(model, "_last_episode_starts", None)
    if last_obs is None or last_starts is None:
        raise ContractError("No uninterrupted SB3 continuation observation at this boundary")
    from .envs import boundary_raw_observations
    raw = boundary_raw_observations(vec, cfg.policy.observation_dim)
    last_obs = np.asarray(last_obs)
    last_starts = np.asarray(last_starts)
    if (last_obs.shape != raw.shape or last_obs.dtype != np.float32
            or last_starts.shape != (cfg.num_envs,) or last_starts.dtype != bool
            or not np.isfinite(last_obs).all()):
        raise ContractError("Malformed SB3 last-observation/episode-start state")
    expected = raw if not cfg.normalize_observation else vec.normalize_obs(raw)
    if not np.array_equal(last_obs, expected):
        raise ContractError("SB3 continuation observation differs from the frozen L3 boundary")
    dest = Path(dest)
    if dest.exists():
        raise ArtifactError("Refusing to overwrite a checkpoint")
    dest.parent.mkdir(parents=True, exist_ok=True)
    staging = dest.parent / (".staging-" + uuid4().hex)
    staging.mkdir(exist_ok=False)
    try:
        model.save(str(staging / "model.zip"))
        files = ["model.zip"]
        if cfg.algorithm != "PPO":
            if getattr(model, "replay_buffer", None) is None:
                raise ArtifactError("Off-policy replay buffer missing")
            model.save_replay_buffer(str(staging / "replay.pkl"))
            files.append("replay.pkl")
        # A pickle-free snapshot binds SB3's live continuation to L3's cursor.
        np.savez_compressed(staging / "continuation.npz", raw=raw,
                            last_obs=last_obs, episode_starts=last_starts)
        files.append("continuation.npz")
        _json_dump(staging / "normalization.json", export_stats(vec))
        _json_dump(staging / "rng.json", capture_rng())
        _json_dump(staging / "noise.json", capture_noise_state(getattr(model, "action_noise", None)))
        _json_dump(staging / "seed_schedule.json", _seed_schedule(vec, cfg.seeds.environments))
        _json_dump(staging / "episode_replays.json", _episode_replays(vec, cfg, boundary))
        files += ["normalization.json", "rng.json", "noise.json", "seed_schedule.json",
                  "episode_replays.json"]
        import stable_baselines3
        metadata = {
            "schema": "rles.l4.stage.v4", "attempt_id": attempt_id,
            "algorithm": cfg.algorithm, "config_hash": cfg.hash,
            "transition_hash": cfg.fingerprint.hash,
            "transition": asdict(cfg.fingerprint),
            "policy": asdict(cfg.policy),
            "seed_tree": asdict(cfg.seeds),
            "boundary": asdict(boundary),
            "sb3_version": stable_baselines3.__version__,
            "torch_version": str(torch.__version__),
            "numeric_profile": numeric_profile(),
            "num_timesteps": int(model.num_timesteps),
            "files": {f: sha_file(staging / f) for f in files},
        }
        _json_dump(staging / "manifest.json", metadata)
        # Atomic only for directories on the same filesystem; no overwrite ever.
        os.replace(staging, dest)
        (dest / "COMPLETED").write_text(sha_file(dest / "manifest.json"), encoding="ascii")
        return dest
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def checkpoint_verify(path: Path, cfg: RLConfig, *, trusted_source: bool) -> dict:
    """Hashes verify integrity, not authenticity; SB3/replay deserialization is unsafe."""
    if not trusted_source:
        raise ArtifactError("Explicit trusted_source=True required for SB3/replay deserialization")
    root = Path(path)
    if root.is_symlink() or not root.is_dir():
        raise ArtifactError("Checkpoint must be a real local directory, not a symlink")
    marker = root / "COMPLETED"
    manifest_path = root / "manifest.json"
    if not marker.is_file() or not manifest_path.is_file():
        raise ArtifactError("Missing checkpoint COMPLETED/manifest marker")
    if marker.read_text(encoding="ascii") != sha_file(manifest_path):
        raise ArtifactError("Checkpoint manifest digest mismatch")
    raw = json.loads(manifest_path.read_bytes())
    required = {"schema", "attempt_id", "algorithm", "config_hash", "transition_hash", "transition", "policy", "seed_tree", "boundary", "sb3_version", "torch_version", "numeric_profile", "num_timesteps", "files"}
    if set(raw) != required or raw["schema"] != "rles.l4.stage.v4":
        raise ArtifactError("Unknown or incomplete checkpoint schema")
    if raw["algorithm"] != cfg.algorithm or raw["config_hash"] != cfg.hash:
        raise ReplayMismatch("Checkpoint algorithm/recipe does not match")
    if raw["transition_hash"] != cfg.fingerprint.hash or raw["transition"] != asdict(cfg.fingerprint):
        raise ReplayMismatch("Replay environment/Execution/Risk/Action changed: new run required")
    import stable_baselines3
    if raw["sb3_version"] != stable_baselines3.__version__ or raw["torch_version"] != str(torch.__version__):
        raise ArtifactError("Checkpoint library versions differ; upgrade gate is required")
    if raw["numeric_profile"] != numeric_profile():
        raise ArtifactError("Resume numerical environment differs; G0 approval required")
    expected = {"model.zip", "normalization.json", "rng.json", "noise.json",
                "seed_schedule.json", "continuation.npz", "episode_replays.json"}
    if cfg.algorithm != "PPO":
        expected.add("replay.pkl")
    if set(raw["files"]) != expected:
        raise ArtifactError("Missing or unexpected checkpoint payload")
    if {p.name for p in root.iterdir()} != expected | {"manifest.json", "COMPLETED"}:
        raise ArtifactError("Unexpected files in checkpoint directory")
    for name, digest in raw["files"].items():
        file = root / name
        if file.is_symlink() or not file.is_file() or sha_file(file) != digest:
            raise ArtifactError(f"Checkpoint payload corruption: {name}")
    return raw


def checkpoint_load(path: Path, cfg: RLConfig, train_vec, *, trusted_source: bool, expected_boundary: Boundary):
    manifest = checkpoint_verify(path, cfg, trusted_source=trusted_source)
    boundary = Boundary.parse(manifest["boundary"])
    boundary.check(cfg.num_envs)
    if boundary != expected_boundary:
        raise ReplayMismatch("L3 boundary tokens/cursor not externally restored exactly")
    saved_replays = tuple(_episode_replays_from_snapshot(
        json.loads((Path(path) / "episode_replays.json").read_bytes()), cfg, boundary,
    ))
    restored_replays = tuple(_episode_replays(train_vec, cfg, boundary))
    if saved_replays != restored_replays:
        raise ReplayMismatch("Restored L3 action journals differ from the saved checkpoint")
    import stable_baselines3
    root = Path(path)
    stats = json.loads((root / "normalization.json").read_bytes())
    if (stats is None) != (not cfg.normalize_observation):
        raise ArtifactError("Normalizer presence differs from frozen config")
    vec = apply_stats(train_vec, stats, training=True)
    schedules = json.loads((root / "seed_schedule.json").read_bytes())
    if not isinstance(schedules, list) or len(schedules) != cfg.num_envs:
        raise ArtifactError("environment seed-schedule checkpoint count mismatch")
    checked_schedules = []
    for state, expected_seed in zip(schedules, cfg.seeds.environments):
        if (not isinstance(state, dict) or set(state) != {"root_seed", "next_episode"}
                or type(state["root_seed"]) is not int or state["root_seed"] != expected_seed
                or type(state["next_episode"]) is not int or state["next_episode"] < 1):
            raise ArtifactError("invalid environment seed-schedule checkpoint")
        checked_schedules.append(state)
    try:
        for index, state in enumerate(checked_schedules):
            vec.env_method("rles_restore_seed_schedule_state", state, indices=index)
    except (AttributeError, ValueError, TypeError) as exc:
        raise ArtifactError("L3/L4 Env cannot restore its seed-schedule state") from exc
    alg = getattr(stable_baselines3, cfg.algorithm)
    # SB3's default force_reset=True discards the saved last observation.
    # Disable it only after checking the explicit L3 observation/cursor proof.
    from .envs import boundary_raw_observations
    with np.load(root / "continuation.npz", allow_pickle=False) as stored:
        if set(stored.files) != {"raw", "last_obs", "episode_starts"}:
            raise ArtifactError("Unknown checkpoint continuation fields")
        raw = stored["raw"].copy()
        last_obs = stored["last_obs"].copy()
        starts = stored["episode_starts"].copy()
    shape = (cfg.num_envs, cfg.policy.observation_dim)
    if (raw.dtype != np.float32 or last_obs.dtype != np.float32 or
            raw.shape != shape or last_obs.shape != shape or
            starts.dtype != bool or starts.shape != (cfg.num_envs,) or
            not np.isfinite(raw).all() or not np.isfinite(last_obs).all()):
        raise ArtifactError("Continuation state shape/dtype/finite check failed")
    if not np.array_equal(raw, boundary_raw_observations(vec, cfg.policy.observation_dim)):
        raise ReplayMismatch("Restored L3 boundary observation differs from saved transition")
    expected_obs = raw if stats is None else vec.normalize_obs(raw)
    if not np.array_equal(last_obs, expected_obs):
        raise ReplayMismatch("SB3 continuation observation/normalization disagreement")
    model = alg.load(str(root / "model.zip"), env=vec, device=cfg.device,
                     print_system_info=False, force_reset=False)
    if (not np.array_equal(model._last_obs, last_obs) or
            not np.array_equal(model._last_episode_starts, starts)):
        raise ReplayMismatch("Saved SB3 internal continuation state differs from L3 boundary")
    if stats is not None:
        vec.old_obs = raw.copy()
        # SB3 reads the raw observation from VecNormalize when storing replay.
        if cfg.algorithm != "PPO":
            if not np.array_equal(model._last_original_obs, raw):
                raise ReplayMismatch("SB3 raw replay observation differs from L3 boundary")
    # Without VecNormalize, the verified `_last_obs` is already raw and is the
    # authoritative replay boundary. `_last_original_obs` is not a stable SB3
    # checkpoint field in this mode and must not be used as a second proof.
    if cfg.algorithm != "PPO":
        model.load_replay_buffer(str(root / "replay.pkl"), truncate_last_traj=False)
    noise_state = json.loads((root / "noise.json").read_bytes())
    # Rebuild a seedable noise object (SB3 zip may serialize it, but this makes it explicit).
    if cfg.algorithm in ("DDPG", "TD3"):
        from .noise import make_noise
        model.action_noise = make_noise(cfg.noise_sigma, cfg.seeds.noise, cfg.num_envs)
    restore_noise_state(getattr(model, "action_noise", None), noise_state)
    restore_rng(json.loads((root / "rng.json").read_bytes()))
    if (int(manifest["num_timesteps"]) != expected_boundary.step_cursor * cfg.num_envs
            or int(manifest["num_timesteps"]) <= 0):
        raise ReplayMismatch("Boundary cursor and sample count disagree")
    if int(model.num_timesteps) != int(manifest["num_timesteps"]):
        raise ArtifactError("Checkpoint SB3 step count mismatch")
    return model, vec
