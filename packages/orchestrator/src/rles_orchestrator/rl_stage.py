"""R1 composition of the independent L4 provider and E1 transfer contract."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict
from hashlib import sha256
import json
import math
from pathlib import Path
import tempfile

import numpy as np

from rles_es_evotorch.actor import ActorGenomeSpec, act, decode, genome_hash
from rles_rl_sb3.actor import InferenceActor
from rles_rl_sb3.config import RLConfig
from rles_rl_sb3.envs import EnvFactory
from rles_rl_sb3.errors import ActorActionParityMismatch
from rles_rl_sb3.provider import GateAuthorizer, RLProvider

from .transfer import (
    DistillationEvidence, ParameterLoadEvidence, ParityReport, TransferResult,
    adapt_l4_distillation, adapt_l4_load, transfer,
)
from .workflow import ProviderStageOutput, StageContext, transfer_evidence_hash


class RLStageExecutor:
    """Train R1, prove optional E1 transfer, and export its exact L4 actor."""

    def __init__(
        self, *, artifact_id: str, config: RLConfig, env_factory: EnvFactory,
        gate_authorizer: GateAuthorizer, train_steps: int, inference_dir: str | Path,
        fixed_observations: np.ndarray, actor_spec: ActorGenomeSpec | None = None,
        source_genome: np.ndarray | None = None,
        training_observations: np.ndarray | None = None,
        parity_tolerance: float = 1e-5,
        distill_epochs: int = 3000, distill_batch_size: int = 16,
        distill_early_stop: bool = True,
        distill_learning_rate: float = 0.01,
        checkpoint_dir: str | Path | None = None,
        checkpoint_interval_steps: int = 1024,
        resume_from: str | Path | None = None,
        resume_env_factory: Callable[[tuple[dict[str, object], ...]], EnvFactory] | None = None,
    ) -> None:
        if (not isinstance(artifact_id, str) or not artifact_id
                or not isinstance(config, RLConfig) or not callable(env_factory)
                or not callable(gate_authorizer) or type(train_steps) is not int
                or train_steps < 1 or not isinstance(fixed_observations, np.ndarray)
                or (actor_spec is None) != (source_genome is None)
                or (actor_spec is None) != (training_observations is None)
                or not isinstance(parity_tolerance, (int, float))
                or isinstance(parity_tolerance, bool) or not math.isfinite(parity_tolerance)
                or parity_tolerance < 0 or type(distill_epochs) is not int
                or distill_epochs < 1 or type(distill_early_stop) is not bool or type(distill_batch_size) is not int
                or distill_batch_size < 1 or isinstance(distill_learning_rate, bool)
                or not isinstance(distill_learning_rate, (int, float))
                or not math.isfinite(distill_learning_rate) or distill_learning_rate <= 0
                or (checkpoint_dir is not None and (type(checkpoint_interval_steps) is not int
                                                     or checkpoint_interval_steps < 1))
                or ((resume_from is None) != (resume_env_factory is None))):
            raise ValueError("R1 requires frozen L4 config, env, gate, and transfer inputs")
        fixed = np.asarray(fixed_observations)
        if (fixed.dtype != np.float32 or fixed.ndim != 2 or fixed.shape[0] < 1
                or fixed.shape[1] != config.policy.observation_dim or not np.isfinite(fixed).all()):
            raise ValueError("fixed R1 parity observations must match the raw L4 policy schema")
        self.artifact_id, self.config = artifact_id, config
        self.env_factory, self.gate_authorizer = env_factory, gate_authorizer
        self.train_steps, self.inference_dir = train_steps, Path(inference_dir)
        self.fixed_observations = fixed.copy()
        self.actor_spec = actor_spec
        self.source_genome = None if source_genome is None else np.asarray(source_genome).copy()
        self.training_observations = (None if training_observations is None
                                      else np.asarray(training_observations).copy())
        self.parity_tolerance = parity_tolerance
        self.distill_epochs, self.distill_batch_size = distill_epochs, distill_batch_size
        self.distill_early_stop = distill_early_stop
        self.distill_learning_rate = distill_learning_rate
        self.checkpoint_dir = None if checkpoint_dir is None else Path(checkpoint_dir)
        self.checkpoint_interval_steps = checkpoint_interval_steps
        self.resume_from = None if resume_from is None else Path(resume_from)
        self.resume_env_factory = resume_env_factory
        self._used = False

    def _train_with_checkpoints(self, provider: RLProvider, context: StageContext, *,
                                initial_steps: int = 0, resumed: bool = False):
        """Train in complete SB3 rollout blocks and save only after learn() settles."""
        from rles_rl_sb3.checkpoint import Boundary
        from rles_rl_sb3.envs import check_budget

        params = dict(self.config.hyperparameters)
        block = (int(params.get("n_steps", 2048)) if context.spec.algorithm == "PPO"
                 else int(params.get("train_freq", 1))) * self.config.num_envs
        # Preserve the provider's complete-budget validation before opening a
        # sequence of smaller, independently valid learning calls.
        check_budget(context.spec.algorithm, params, self.config.num_envs, self.train_steps)
        interval = max(block, ((self.checkpoint_interval_steps + block - 1) // block) * block)
        performances = []
        trained = 0
        remaining = self.train_steps - initial_steps
        while trained < remaining:
            chunk = min(interval, remaining - trained)
            performance = provider.train(chunk, resumed=resumed or trained > 0)
            performances.append(performance)
            trained += performance.samples
            sampled_total = initial_steps + trained
            if provider.vec is None:
                raise RuntimeError("R1 learner is unavailable at checkpoint boundary")
            tokens = tuple(provider.vec.get_attr("rles_boundary_token"))
            boundary = Boundary(
                "R1", sampled_total // self.config.num_envs, tokens,
                no_inflight_orders=True, no_pending_step=True,
            )
            if self.checkpoint_dir is not None:
                provider.save(
                    self.checkpoint_dir / f"step-{sampled_total:012d}", boundary,
                    attempt_id=context.spec.attempt_id,
                )
        return tuple(performances)

    def _transfer_record(self, context: StageContext, result: TransferResult | None = None) -> TransferResult:
        """Persist/reload pre-training E1 transfer evidence across R1 resume."""
        path = self.inference_dir.parent / "R1-transfer.json"
        if self.resume_from is not None:
            if not path.is_file():
                raise RuntimeError("A4 R1 resume is missing its durable E1 transfer evidence")
            payload = path.read_bytes()
            raw = json.loads(payload.decode("utf-8"))
            if not isinstance(raw, dict) or set(raw) != {"result", "evidence_hash"}:
                raise RuntimeError("durable E1 transfer evidence schema mismatch")
            value = raw["result"]
            parity_raw = value["parity"]
            parity = ParityReport(**parity_raw)
            load_raw = value["load_evidence"]
            load = None if load_raw is None else ParameterLoadEvidence(
                **{**load_raw, "missing_keys": tuple(load_raw["missing_keys"]),
                   "unexpected_keys": tuple(load_raw["unexpected_keys"])})
            distill_raw = value["distillation"]
            distill = None if distill_raw is None else DistillationEvidence(**distill_raw)
            restored = TransferResult(value["algorithm"], value["method"], parity, load, distill,
                                      value["training_split"])
            if (restored.algorithm != context.spec.algorithm
                    or raw["evidence_hash"] != transfer_evidence_hash(restored)):
                raise RuntimeError("durable E1 transfer evidence differs from the frozen RunSpec")
            return restored
        if result is None:
            raise ValueError("fresh E1 transfer result required for durable evidence")
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps({"result": asdict(result),
                              "evidence_hash": transfer_evidence_hash(result)},
                             sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
        temporary = path.with_name(path.name + ".tmp")
        with temporary.open("xb") as stream:
            stream.write(payload)
        temporary.replace(path)
        return result

    def _training_diagnostics_record(
        self, context: StageContext, performances: tuple[object, ...], *, initial_steps: int,
    ) -> str:
        """Persist deterministic per-checkpoint learning evidence for a completed R1 stage."""
        path = self.inference_dir.parent / "R1-learning.json"
        marker = path.with_suffix(path.suffix + ".sha256")
        if not performances:
            if not path.is_file() or not marker.is_file():
                raise RuntimeError("full-budget R1 resume is missing durable learning diagnostics")
            payload = path.read_bytes()
            identity = sha256(payload).hexdigest()
            if marker.read_text(encoding="ascii") != identity:
                raise RuntimeError("durable R1 learning diagnostics failed integrity verification")
            record = json.loads(payload.decode("utf-8"))
            if (not isinstance(record, dict)
                    or record.get("schema") not in {
                        "rles.r1-learning.v1", "rles.r1-learning.v2", "rles.r1-learning.v3",
                        "rles.r1-learning.v4", "rles.r1-learning.v5",
                    }
                    or record.get("config_hash") != self.config.hash
                    or record.get("transition_hash") != self.config.fingerprint.hash
                    or record.get("attempt_id") != context.spec.attempt_id
                    or record.get("final_steps") != self.train_steps):
                raise RuntimeError("durable R1 learning diagnostics differ from the frozen stage")
            return identity
        if path.exists() or marker.exists():
            raise RuntimeError("R1 learning diagnostics already exist for a fresh training result")
        blocks = []
        cursor = initial_steps
        for performance in performances:
            samples = getattr(performance, "samples", None)
            diagnostics = getattr(performance, "diagnostics", None)
            if (type(samples) is not int or samples < 1 or diagnostics is None
                    or getattr(diagnostics, "samples", None) != samples):
                raise RuntimeError("L4 training performance lacks native learning diagnostics")
            resources = {
                "cpu_rss_bytes": getattr(performance, "cpu_rss_bytes", None),
                "gpu_max_allocated_bytes": getattr(performance, "gpu_max_allocated_bytes", None),
                "gpu_max_reserved_bytes": getattr(performance, "gpu_max_reserved_bytes", None),
            }
            if any(value is not None and (type(value) is not int or value < 0)
                   for value in resources.values()):
                raise RuntimeError("L4 training performance has invalid resource measurements")
            block = {
                "step_start": cursor,
                "step_end": cursor + samples,
                "samples": samples,
                "updates": getattr(performance, "updates", None),
                "resource_measurements": {
                    "scope": "cpu_rss_at_learn_call_end; torch_cuda_allocator_peaks_since_learn_call_start",
                    **resources,
                },
                "diagnostics": asdict(diagnostics),
            }
            blocks.append(block)
            cursor += samples
        if cursor != self.train_steps:
            raise RuntimeError("R1 learning diagnostics do not cover the completed step budget")
        record = {
            "schema": "rles.r1-learning.v5",
            "attempt_id": context.spec.attempt_id,
            "algorithm": self.config.algorithm,
            "config_hash": self.config.hash,
            "transition_hash": self.config.fingerprint.hash,
            "initial_steps": initial_steps,
            "final_steps": cursor,
            "blocks": blocks,
        }
        payload = json.dumps(
            record, sort_keys=True, separators=(",", ":"), allow_nan=False,
        ).encode("utf-8")
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".tmp")
        with temporary.open("xb") as stream:
            stream.write(payload)
        temporary.replace(path)
        identity = sha256(payload).hexdigest()
        marker.write_text(identity, encoding="ascii")
        return identity

    def _preflight(self, context: StageContext) -> None:
        if (not isinstance(context, StageContext) or context.stage != "R1"
                or context.budget != context.spec.stage_budget.get("R1")
                or self.config.hash != context.spec.config_hash
                or self.config.algorithm != context.spec.algorithm
                or self.train_steps > context.budget.max_rl_steps
                or (self.inference_dir.exists() and self.resume_from is None)):
            raise ValueError("R1 stage config, budget, or inference destination changed")
        frozen = self.config.fingerprint
        spec = context.spec
        if (frozen.dataset != spec.data_hash or frozen.feature != spec.feature_hash
                or frozen.execution != spec.execution_hash
                or frozen.hard_risk != spec.hard_risk_hash
                or frozen.soft_risk != spec.initial_soft_risk_hash
                or frozen.normalization != spec.normalization_hash
                or frozen.observation != self.config.policy.observation_schema_hash):
            raise ValueError("L4 transition recipe differs from frozen R1 RunSpec")
        if context.previous is None:
            if self.actor_spec is not None:
                raise ValueError("pure R1 baseline cannot consume an E1 actor")
        else:
            if (context.previous.stage != "E1" or self.actor_spec is None
                    or self.source_genome is None or self.training_observations is None
                    or self.actor_spec.observation_names == ()
                    or len(self.actor_spec.observation_names) != self.config.policy.observation_dim
                    or genome_hash(self.actor_spec, self.source_genome) != context.previous.actor_hash):
                raise ValueError("R1 source actor differs from attested E1 result")
            train = self.training_observations
            if (train.dtype != np.float32 or train.ndim != 2 or train.shape[0] < 1
                    or train.shape[1] != self.config.policy.observation_dim
                    or not np.isfinite(train).all()):
                raise ValueError("Train-inner transfer observations violate R1 policy schema")

    def __call__(self, context: StageContext) -> ProviderStageOutput:
        if self._used:
            raise RuntimeError("R1 stage executor is single-use")
        self._preflight(context)
        self._used = True
        initial_steps = 0
        resumed_boundary = None
        env_factory = self.env_factory
        if self.resume_from is not None:
            if self.resume_env_factory is None:
                raise ValueError("application resume requires a replay environment factory")
            from rles_rl_sb3.checkpoint import Boundary, checkpoint_episode_replays, checkpoint_verify

            manifest = checkpoint_verify(self.resume_from, self.config, trusted_source=True)
            if manifest["attempt_id"] != context.spec.attempt_id:
                raise ValueError("R1 checkpoint attempt differs from the frozen RunSpec")
            initial_steps = manifest["num_timesteps"]
            if type(initial_steps) is not int or not 0 < initial_steps <= self.train_steps:
                raise ValueError("R1 resume checkpoint exceeds the frozen step budget")
            resumed_boundary = Boundary.parse(manifest["boundary"])
            replay_tapes = checkpoint_episode_replays(
                self.resume_from, self.config, trusted_source=True,
            )
            env_factory = self.resume_env_factory(replay_tapes)
        provider = RLProvider(self.config, env_factory, self.gate_authorizer)
        try:
            if self.resume_from is None:
                provider.create()
            else:
                if resumed_boundary is None:
                    raise RuntimeError("verified R1 resume boundary is missing")
                provider.load(self.resume_from, trusted_source=True,
                              restored_boundary=resumed_boundary)
            transfer_result = None
            if context.previous is not None:
                if self.resume_from is not None:
                    transfer_result = self._transfer_record(context)
                else:
                    source_spec, source = self.actor_spec, self.source_genome
                    train = self.training_observations
                    if source_spec is None or source is None or train is None:
                        raise RuntimeError("E1 transfer source disappeared")
                    calibrated_stats = None
                    if source_spec.version == 2 and self.config.normalize_observation:
                        calibrated_stats = provider.calibrate_observation_normalization(
                            train, training_split="train_inner",
                        )
                    teacher: Callable[[np.ndarray], np.ndarray] = lambda batch: act(source_spec, source, batch)
                    fixed = self.fixed_observations
                    labels = teacher(fixed)
                    destination_signature = provider.mapping()["actor_signature"]
                    source_weights = decode(source_spec, source)
                    if source_spec.input_scale is not None:
                        # Compose the E1 policy input scale into the raw-input
                        # first layer. L4 normalization and output-transform
                        # compatibility are still decided by exact parity/BC.
                        source_weights["layers.0.weight"] *= np.asarray(source_spec.input_scale, dtype=np.float32)
                    if calibrated_stats is not None:
                        raw_weights = source_weights["layers.0.weight"].copy()
                        source_weights["layers.0.bias"] += raw_weights @ np.asarray(calibrated_stats["mean"], dtype=np.float32)
                        source_weights["layers.0.weight"] *= np.sqrt(
                            np.asarray(calibrated_stats["var"], dtype=np.float32) + float(calibrated_stats["epsilon"]),
                        )
                    if (self.config.algorithm == "PPO" and source_spec.version == 2
                            and self.config.policy.activation == source_spec.activation == "tanh"
                            and self.config.policy.hidden_layers == (*source_spec.hidden_sizes, 1)):
                        # Standard SB3 Tanh hidden scalar reproduces the E1
                        # output Tanh. PPO's final affine head starts as the
                        # declared action scale, so its clipping is identity.
                        head = len(source_spec.hidden_sizes) + 1
                        source_weights[f"layers.{head}.weight"] = np.asarray([[source_spec.action_scale]], dtype=np.float32)
                        source_weights[f"layers.{head}.bias"] = np.zeros(1, dtype=np.float32)
                    load_exact = adapt_l4_load(
                        lambda: provider.transfer(
                            source_weights, destination_signature,
                            provider.policy_observations(fixed), labels,
                            source_actor_signature=context.spec.actor_signature,
                        ),
                        source_actor_signature=context.spec.actor_signature,
                        action_parity_mismatch=ActorActionParityMismatch,
                    )
                    distill = adapt_l4_distillation(
                        lambda observations, actions: provider.distill(
                            provider.policy_observations(observations), actions, epochs=self.distill_epochs,
                            batch_size=self.distill_batch_size,
                            learning_rate=self.distill_learning_rate,
                            tolerance=self.parity_tolerance,
                            early_stop=self.distill_early_stop,
                        ),
                        student=lambda batch: provider.predict(batch), epochs=self.distill_epochs,
                    )
                    if provider.model is None:
                        raise RuntimeError("L4 model missing before transfer")
                    initial_weights = {
                        key: value.detach().clone()
                        for key, value in provider.model.policy.state_dict().items()
                    }

                    def reset_student() -> None:
                        if provider.model is None:
                            raise RuntimeError("L4 model missing during transfer reset")
                        provider.model.policy.load_state_dict(initial_weights, strict=True)

                    transfer_result = transfer(
                        algorithm=context.spec.algorithm,
                        actor_signature=context.spec.actor_signature,
                        teacher=teacher, student=lambda batch: provider.predict(batch),
                        fixed_observations=fixed, training_observations=train,
                        load_exact=load_exact, reset_student=reset_student,
                        distill=distill, tolerance=self.parity_tolerance,
                    )
                    transfer_result = self._transfer_record(context, transfer_result)
            performances: tuple[object, ...]
            if initial_steps == self.train_steps:
                performances = ()
            elif self.checkpoint_dir is None:
                performances = (provider.train(self.train_steps),)
            else:
                performances = self._train_with_checkpoints(
                    provider, context, initial_steps=initial_steps,
                    resumed=self.resume_from is not None,
                )
            diagnostics_hash = self._training_diagnostics_record(
                context, performances, initial_steps=initial_steps,
            )
            if self.inference_dir.exists():
                if initial_steps != self.train_steps:
                    raise RuntimeError("existing R1 actor can only be reused from a full-budget checkpoint")
                with tempfile.TemporaryDirectory(prefix=".r1-verify-",
                                                 dir=self.inference_dir.parent) as temporary:
                    candidate = provider.export_inference(Path(temporary) / "actor",
                                                          self.fixed_observations)
                    candidate_actor = InferenceActor(candidate, verify=True)
                    inference = InferenceActor(self.inference_dir, verify=True)
                    candidate_hash = (candidate / "COMPLETED").read_text(encoding="ascii")
                    existing_hash = (self.inference_dir / "COMPLETED").read_text(encoding="ascii")
                    if candidate_hash != existing_hash:
                        raise RuntimeError("existing R1 actor differs from the restored full-budget checkpoint")
                    if not np.allclose(
                        candidate_actor.predict(self.fixed_observations),
                        provider.predict(self.fixed_observations), atol=1e-5, rtol=0,
                    ):
                        raise RuntimeError("restored R1 checkpoint and re-exported actor predictions differ")
                artifact = self.inference_dir
            else:
                artifact = provider.export_inference(self.inference_dir, self.fixed_observations)
                inference = InferenceActor(artifact, verify=True)
            if not np.allclose(
                inference.predict(self.fixed_observations),
                provider.predict(self.fixed_observations), atol=1e-5, rtol=0,
            ):
                raise RuntimeError("exported R1 actor differs from the trained L4 policy")
            actor_hash = (artifact / "COMPLETED").read_text(encoding="ascii")
            report_hash = transfer_evidence_hash(transfer_result) if transfer_result else None
            payload_hash = sha256(
                bytes.fromhex(actor_hash) + (bytes.fromhex(report_hash) if report_hash else b"")
                + bytes.fromhex(diagnostics_hash)
            ).hexdigest()
            metadata = {
                "rl_training_complete": "true", "inference_actor_hash": actor_hash,
                "training_diagnostics_hash": diagnostics_hash,
            }
            if report_hash is not None:
                metadata["transfer_report_hash"] = report_hash
            return ProviderStageOutput(
                artifact_id=self.artifact_id, payload_hash=payload_hash,
                actor_hash=actor_hash, soft_risk_hash=context.spec.initial_soft_risk_hash,
                metadata=metadata, rl_performances=performances,
                rl_prior_steps=initial_steps,
                transfer_evidence=transfer_result,
            )
        finally:
            provider.close()
