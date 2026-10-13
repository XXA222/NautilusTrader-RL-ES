"""Algorithm-specific exact actor destinations, inference and transfer parity."""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from hashlib import sha256
from pathlib import Path
from collections.abc import Mapping
from typing import Any, cast
import json
import math
import os
import shutil
from uuid import uuid4

import numpy as np
import torch
from torch import nn

from .config import RLConfig, canonical_json, digest
from .errors import ActorActionParityMismatch, ArtifactError, ContractError, TransferBlocked
from .normalize import export_stats


@dataclass(frozen=True)
class LayerBinding:
    source: str
    weight_shape: tuple[int, ...]
    bias_shape: tuple[int, ...]
    dtype: str


def _actor_components(model, algorithm: str):
    if algorithm == "PPO":
        body = model.policy.mlp_extractor.policy_net
        head = model.policy.action_net
        if not isinstance(head, nn.Linear):
            raise TransferBlocked("PPO expected linear action_net (no gSDE/squashed policy)")
        sequence = list(body.children()) + [head]
        output = "clip"  # PPO deterministic mean is clipped, not tanh-squashed.
    elif algorithm == "SAC":
        actor = model.policy.actor
        if not isinstance(actor.mu, nn.Linear) or getattr(actor, "use_sde", False):
            raise TransferBlocked("SAC expected unsquashed linear mu with gSDE disabled")
        sequence = list(actor.latent_pi.children()) + [actor.mu]
        output = "tanh"  # SAC squashed Gaussian deterministic action.
    elif algorithm in ("TD3", "DDPG"):
        actor = model.policy.actor
        sequence = list(actor.mu.children())
        if not isinstance(sequence[-1], nn.Tanh):
            raise TransferBlocked(f"{algorithm} actor must end in tanh")
        output = "tanh"
    else:
        raise TransferBlocked("Unknown algorithm")
    for layer in sequence:
        if not isinstance(layer, (nn.Linear, nn.ReLU, nn.Tanh)):
            raise TransferBlocked(f"Unexpected actor layer type: {type(layer).__name__}")
    linear = [m for m in sequence if isinstance(m, nn.Linear)]
    if not linear:
        raise TransferBlocked("Empty actor MLP")
    # Ensure all actor layers are included; no ignored bias, BN, LN or hidden heads.
    return sequence, linear, output


def actor_mapping(model, cfg: RLConfig) -> dict:
    sequence, linear, output = _actor_components(model, cfg.algorithm)
    layers = [
        LayerBinding(source=f"layers.{i}", weight_shape=tuple(m.weight.shape),
                     bias_shape=tuple(m.bias.shape), dtype=str(m.weight.dtype).replace("torch.", ""))
        for i, m in enumerate(linear)
    ]
    activations = ["linear" if isinstance(m, nn.Linear) else "relu" if isinstance(m, nn.ReLU) else "tanh" for m in sequence]
    if cfg.algorithm == "PPO" and activations[-1] != "linear":
        raise TransferBlocked("PPO action head is not linear")
    if cfg.algorithm == "SAC" and activations[-1] != "linear":
        raise TransferBlocked("SAC mean head is not linear")
    structure = {
        "schema": "rles.l4.actor.mapping.v1", "algorithm": cfg.algorithm,
        "input_dim": cfg.policy.observation_dim, "output_dim": 1,
        "activation_order": activations, "output_transform": output,
        "layers": [asdict(x) for x in layers],
        "observation_schema": cfg.policy.observation_schema_hash,
        "action_schema": cfg.policy.action_schema_hash,
        "action_scale": [-1.0, 1.0], "precision": "float32",
    }
    structure["actor_signature"] = digest(structure)
    return structure


def extract_actor_weights(model, cfg: RLConfig) -> dict[str, np.ndarray]:
    _, linear, _ = _actor_components(model, cfg.algorithm)
    result: dict[str, np.ndarray] = {}
    for i, layer in enumerate(linear):
        result[f"layers.{i}.weight"] = layer.weight.detach().cpu().numpy().copy()
        result[f"layers.{i}.bias"] = layer.bias.detach().cpu().numpy().copy()
    return result


def _exact_validate_weights(model, cfg: RLConfig, weights: Mapping[str, np.ndarray]) -> list[nn.Linear]:
    _, layers, _ = _actor_components(model, cfg.algorithm)
    expected = {f"layers.{i}.{param}" for i in range(len(layers)) for param in ("weight", "bias")}
    if set(weights) != expected:
        raise TransferBlocked(f"Actor coverage mismatch missing={sorted(expected-set(weights))}, extra={sorted(set(weights)-expected)}")
    for i, layer in enumerate(layers):
        for name, target in (("weight", layer.weight), ("bias", layer.bias)):
            arr = weights[f"layers.{i}.{name}"]
            if not isinstance(arr, np.ndarray) or arr.dtype != np.dtype("float32") or arr.shape != tuple(target.shape) or not np.isfinite(arr).all():
                raise TransferBlocked(f"shape/dtype/finite mismatch layers.{i}.{name}")
    return layers


def _copy_weights(layers: list[nn.Linear], weights: Mapping[str, np.ndarray]):
    with torch.no_grad():
        for i, layer in enumerate(layers):
            for name in ("weight", "bias"):
                current = getattr(layer, name)
                source = torch.from_numpy(weights[f"layers.{i}.{name}"]).to(current.device)
                current.copy_(source)


def _sync_target(model, algorithm: str) -> None:
    if algorithm in ("TD3", "DDPG"):
        model.policy.actor_target.load_state_dict(model.policy.actor.state_dict(), strict=True)
    elif algorithm == "SAC":
        # SAC has Q targets but no actor target; leave fresh Q/targets native.
        pass


def _as_obs(obs: np.ndarray, cfg: RLConfig) -> np.ndarray:
    arr = np.asarray(obs)
    if arr.ndim != 2 or arr.shape[1] != cfg.policy.observation_dim or arr.dtype != np.float32 or not np.isfinite(arr).all():
        raise ContractError("fixed parity inputs must be finite float32 [batch, observation_dim] policy-domain observations")
    if arr.shape[0] == 0:
        raise ContractError("Empty parity observation set")
    return arr


@dataclass(frozen=True)
class ParityReport:
    samples: int
    max_abs_error: float
    mean_abs_error: float
    p50_abs_error: float
    p95_abs_error: float
    tolerance: float
    passed: bool
    method: str = "parity"
    expected_tensors: int = 0
    loaded_tensors: int = 0
    actor_signature: str = ""
    epochs_completed: int = 0


def action_parity(model, cfg: RLConfig, policy_observations: np.ndarray,
                  reference_actions: np.ndarray, *, tolerance: float = 1e-5) -> ParityReport:
    obs = _as_obs(policy_observations, cfg)
    refs = np.asarray(reference_actions)
    if refs.dtype != np.float32:
        raise TransferBlocked("Reference actions must be float32, without lossy coercion")
    if refs.shape != (obs.shape[0], 1) or not np.isfinite(refs).all() or (np.abs(refs) > 1.000001).any():
        raise TransferBlocked("reference action shape/range/finite mismatch")
    if not (0 < tolerance < 1 and math.isfinite(tolerance)):
        raise ContractError("Invalid parity tolerance")
    actions, _ = model.predict(obs, deterministic=True)
    actual = np.asarray(actions, dtype=np.float32).reshape((-1, 1))
    if actual.shape != refs.shape or not np.isfinite(actual).all():
        raise TransferBlocked("SB3 action shape/value mismatch")
    err = np.abs(actual.astype(np.float64) - refs.astype(np.float64)).reshape(-1)
    return ParityReport(samples=len(err), max_abs_error=float(np.max(err)),
                        mean_abs_error=float(np.mean(err)),
                        p50_abs_error=float(np.percentile(err, 50)),
                        p95_abs_error=float(np.percentile(err, 95)),
                        tolerance=tolerance, passed=bool(np.max(err) <= tolerance))


def transfer_actor(model, cfg: RLConfig, weights: Mapping[str, np.ndarray],
                   actor_signature: str, policy_observations: np.ndarray,
                   reference_actions: np.ndarray, *, tolerance: float = 1e-5,
                   source_actor_signature: str | None = None) -> ParityReport:
    """Pre-R1 E1 mapping with separate destination/source actor identities.

    ``actor_signature`` identifies this L4 algorithm-specific mapping. The
    optional source identity records the canonical L5 actor contract proven
    by the supplied weights and parity observations.
    """
    if int(model.num_timesteps) != 0:
        raise TransferBlocked("E1 transfer only permitted onto pristine R1 learner")
    if cfg.algorithm != "PPO" and getattr(model.replay_buffer, "pos", 0) != 0:
        raise TransferBlocked("Refuse actor transfer with existing replay")
    binding = actor_mapping(model, cfg)
    if actor_signature != binding["actor_signature"]:
        raise TransferBlocked("CanonicalActor signature mismatch; L5 ICR required")
    if (source_actor_signature is not None
            and (not isinstance(source_actor_signature, str) or len(source_actor_signature) != 64
                 or any(char not in "0123456789abcdef" for char in source_actor_signature))):
        raise ContractError("source_actor_signature must be lowercase SHA256")
    layers = _exact_validate_weights(model, cfg, weights)
    originals = extract_actor_weights(model, cfg)
    try:
        _copy_weights(layers, weights)
        _sync_target(model, cfg.algorithm)
        report = action_parity(model, cfg, policy_observations, reference_actions, tolerance=tolerance)
        if not report.passed:
            raise ActorActionParityMismatch(
                f"E1→R1 action parity exceeded tolerance after exact validated load: {report.max_abs_error}"
            )
        return replace(report, method="exact_weight_transfer",
                       expected_tensors=len(weights), loaded_tensors=len(weights),
                       actor_signature=source_actor_signature or actor_signature)
    except BaseException:
        _copy_weights(layers, originals)
        _sync_target(model, cfg.algorithm)
        raise


def behavioral_clone(model, cfg: RLConfig, observations: np.ndarray, actions: np.ndarray,
                     *, epochs: int, batch_size: int, learning_rate: float,
                     tolerance: float = 1e-3, early_stop: bool = False) -> ParityReport:
    """Distinct BC initialization path; not reported as transferred weights."""
    if model.num_timesteps != 0:
        raise TransferBlocked("Behavioral cloning must precede any RL updates")
    obs = _as_obs(observations, cfg)
    target = np.asarray(actions, dtype=np.float32)
    if target.dtype != np.float32 or target.shape != (len(obs), 1) or not np.isfinite(target).all() or (np.abs(target) > 1).any():
        raise TransferBlocked("Invalid supervised actor labels")
    if type(epochs) is not int or epochs <= 0 or type(batch_size) is not int or batch_size <= 0 or not 0 < learning_rate < 1:
        raise ContractError("Invalid distillation settings")
    if type(early_stop) is not bool:
        raise ContractError("distillation early_stop must be bool")
    layers = _exact_validate_weights(model, cfg, extract_actor_weights(model, cfg))
    original = extract_actor_weights(model, cfg)
    if cfg.algorithm == "PPO":
        network = model.policy.mlp_extractor.policy_net
        head = model.policy.action_net
        parameters = list(network.parameters()) + list(head.parameters())
    elif cfg.algorithm == "SAC":
        actor = model.policy.actor
        network, head = actor.latent_pi, actor.mu
        parameters = list(network.parameters()) + list(head.parameters())
    else:
        network, head = model.policy.actor.mu, None
        parameters = list(network.parameters())
    optimizer = torch.optim.Adam(parameters, lr=learning_rate)
    device = model.device
    # Local generator avoids introducing a new global RNG stream.
    rng = np.random.default_rng(cfg.seeds.algorithm)
    try:
        completed = 0
        for epoch in range(epochs):
            for ids in np.array_split(rng.permutation(len(obs)), max(1, math.ceil(len(obs) / batch_size))):
                inp = torch.as_tensor(obs[ids], device=device)
                ref = torch.as_tensor(target[ids], device=device)
                prediction = network(inp)
                if head is not None:
                    prediction = head(prediction)
                if cfg.algorithm in ("SAC", "TD3", "DDPG"):
                    prediction = torch.tanh(prediction)
                # PPO's deterministic inference clips its unsquashed actor
                # output. Train that raw output against the bounded teacher
                # action: clamping before the loss makes saturated outputs
                # have zero gradient and can permanently block Hybrid BC.
                loss = torch.nn.functional.mse_loss(prediction, ref)
                if not torch.isfinite(loss):
                    raise TransferBlocked("Non-finite distillation loss")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            completed = epoch + 1
            if early_stop:
                _sync_target(model, cfg.algorithm)
                if action_parity(model, cfg, obs, target, tolerance=tolerance).passed:
                    break
        _sync_target(model, cfg.algorithm)
        result = action_parity(model, cfg, obs, target, tolerance=tolerance)
        # BC quality is independently reported; the orchestration gate decides
        # whether this approximation is acceptable, never label it exact.
        return replace(result, method="behavioral_distillation", epochs_completed=completed)
    except BaseException:
        _copy_weights(layers, original)
        _sync_target(model, cfg.algorithm)
        raise


def _l0_data(model, cfg: RLConfig, normalization: dict | None) -> dict:
    mapping = actor_mapping(model, cfg)
    return {
        "schema": "rles.l4.inference.v1", "mapping": mapping,
        "fingerprint": asdict(cfg.fingerprint),
        "policy": asdict(cfg.policy),
        "normalization": normalization,
        "external_feature_scaler_sha256": cfg.fingerprint.feature_scaler,
        "action_domain": "target_net_exposure[-1,1]",
    }


def export_l0(model, cfg: RLConfig, vec, dest: Path, fixed_raw_obs: np.ndarray,
              *, tolerance: float = 1e-5) -> Path:
    """Actor-only weights and reproducible preprocess schema; not a trading artifact."""
    if dest.exists():
        raise ArtifactError("L0 destination already exists")
    stats = export_stats(vec)
    if (stats is None) != (not cfg.normalize_observation):
        raise ArtifactError("Inference normalizer contract missing")
    data = _l0_data(model, cfg, stats)
    weights = extract_actor_weights(model, cfg)
    dest.parent.mkdir(parents=True, exist_ok=True)
    temp = dest.parent / (".l0-" + uuid4().hex)
    temp.mkdir()
    try:
        np.savez_compressed(temp / "weights.npz", **cast(Any, weights))
        (temp / "actor.json").write_bytes(canonical_json(data))
        artifact = InferenceActor(temp, verify=False)
        obs = _as_obs(fixed_raw_obs, cfg)
        actual = artifact.predict(obs)
        model_input = obs if stats is None else _normalize_raw(obs, stats)
        expected, _ = model.predict(model_input, deterministic=True)
        err = np.max(np.abs(np.asarray(expected).reshape(-1, 1) - actual))
        if not np.isfinite(err) or err > tolerance:
            raise ArtifactError(f"L0 frozen actor inference parity failed, max_error={err}")
        files = {name: _sha(temp / name) for name in ("weights.npz", "actor.json")}
        (temp / "integrity.json").write_bytes(canonical_json(files))
        os.replace(temp, dest)
        (dest / "COMPLETED").write_text(_sha(dest / "integrity.json"), encoding="ascii")
        return dest
    finally:
        if temp.exists():
            shutil.rmtree(temp)


def _sha(p: Path) -> str:
    h = sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1024*1024), b""):
            h.update(b)
    return h.hexdigest()


def _normalize_raw(raw: np.ndarray, stats: dict) -> np.ndarray:
    mean = np.asarray(stats["mean"], dtype=np.float64)
    var = np.asarray(stats["var"], dtype=np.float64)
    if mean.shape != raw.shape[1:] or var.shape != raw.shape[1:]:
        raise ArtifactError("L0 feature normalization dimension mismatch")
    return np.clip((raw - mean) / np.sqrt(var + float(stats["epsilon"])),
                   -float(stats["clip_obs"]), float(stats["clip_obs"])).astype(np.float32)


class InferenceActor:
    """No pickle; runs a frozen SB3 MLP actor, not a separate RL implementation."""
    def __init__(self, location: Path, *, verify: bool = True):
        root = Path(location)
        if verify:
            if {p.name for p in root.iterdir()} != {"weights.npz", "actor.json", "integrity.json", "COMPLETED"}:
                raise ArtifactError("Malformed L0 artifact")
            if (root / "COMPLETED").read_text("ascii") != _sha(root / "integrity.json"):
                raise ArtifactError("L0 integrity marker mismatch")
            ledger = json.loads((root / "integrity.json").read_bytes())
            if ledger != {p: _sha(root / p) for p in ("weights.npz", "actor.json")}:
                raise ArtifactError("L0 file digests mismatch")
        data = json.loads((root / "actor.json").read_bytes())
        if data.get("schema") != "rles.l4.inference.v1" or data.get("mapping", {}).get("schema") != "rles.l4.actor.mapping.v1":
            raise ArtifactError("Unknown L0 schema")
        signature = data["mapping"].pop("actor_signature")
        if digest(data["mapping"]) != signature:
            raise ArtifactError("L0 actor signature mismatch")
        data["mapping"]["actor_signature"] = signature
        with np.load(root / "weights.npz", allow_pickle=False) as z:
            self.weights = {k: z[k].copy() for k in z.files}
        mapping = data["mapping"]
        expected = {f"layers.{i}.{w}" for i in range(len(mapping["layers"])) for w in ("weight", "bias")}
        if set(self.weights) != expected:
            raise ArtifactError("L0 weight coverage mismatch")
        for i, b in enumerate(mapping["layers"]):
            for n, shape in (("weight", b["weight_shape"]), ("bias", b["bias_shape"])):
                a = self.weights[f"layers.{i}.{n}"]
                if a.dtype != np.float32 or list(a.shape) != list(shape) or not np.isfinite(a).all():
                    raise ArtifactError("L0 parameter dtype/shape/value mismatch")
        self.mapping = mapping
        self.stats = data["normalization"]
        self.policy = data["policy"]
        self.external_feature_scaler_sha256 = data["external_feature_scaler_sha256"]
        if (self.external_feature_scaler_sha256 != data["fingerprint"]["feature_scaler"] or
                mapping["observation_schema"] != self.policy["observation_schema_hash"] or
                mapping["action_schema"] != self.policy["action_schema_hash"] or
                self.policy["observation_dim"] != mapping["input_dim"]):
            raise ArtifactError("L0 embedded policy/schema/scaler references disagree")

    def predict(self, raw_observation: np.ndarray) -> np.ndarray:
        x = np.asarray(raw_observation)
        if x.dtype != np.float32 or x.ndim != 2 or x.shape[1] != self.mapping["input_dim"] or not np.isfinite(x).all():
            raise ContractError("Expected finite float32 raw observations [batch, obs_dim]")
        if self.stats is not None:
            x = _normalize_raw(x, self.stats)
        elif not x.flags.writeable:
            x = x.copy()
        t = torch.from_numpy(x)
        linear_idx = 0
        with torch.no_grad():
            for activation in self.mapping["activation_order"]:
                if activation == "linear":
                    w = torch.from_numpy(self.weights[f"layers.{linear_idx}.weight"])
                    b = torch.from_numpy(self.weights[f"layers.{linear_idx}.bias"])
                    t = torch.nn.functional.linear(t, w, b)
                    linear_idx += 1
                elif activation == "relu":
                    t = torch.relu(t)
                elif activation == "tanh":
                    t = torch.tanh(t)
                else:
                    raise ArtifactError("Unknown actor activation")
            if self.mapping["output_transform"] == "clip":
                t = t.clamp(-1, 1)
            elif self.mapping["output_transform"] == "tanh":
                # SAC network has linear final head, whereas TD3/DDPG already
                # contain a terminal Tanh module. Do not squash twice.
                if self.mapping["activation_order"][-1] != "tanh":
                    t = torch.tanh(t)
            else:
                raise ArtifactError("Unknown action transform")
        result = t.detach().cpu().numpy().astype(np.float32)
        if result.shape != (len(x), 1) or not np.isfinite(result).all():
            raise ArtifactError("Non-finite L0 inference")
        return result
