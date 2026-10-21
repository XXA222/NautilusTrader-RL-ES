from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pytest
import torch

from rles_rl_sb3.config import PolicySpec, RLConfig, SeedSpec, TransitionFingerprint
from rles_rl_sb3.checkpoint import Boundary, checkpoint_episode_replays
from rles_rl_sb3.errors import ContractError
from rles_rl_sb3.provider import RLProvider

pytestmark = pytest.mark.algorithm_smoke


def _small_pretraining_provider(*, normalize=False):
    fingerprints = TransitionFingerprint(**{name: f"{index + 1:064x}" for index, name in enumerate(
        TransitionFingerprint.__dataclass_fields__
    )})
    config = RLConfig(
        algorithm="PPO", policy=PolicySpec(2, hidden_layers=(2,), value_layers=(2,),
                                            observation_schema_hash="a" * 64, action_schema_hash="b" * 64),
        fingerprint=fingerprints, seeds=SeedSpec(algorithm=11, environments=(12,), noise=13, replay=14),
        hyperparameters={"n_steps": 2, "batch_size": 2, "n_epochs": 1},
        device="cpu", normalize_observation=normalize,
    )
    return RLProvider(config, lambda _rank, _seed, _phase: ProviderFixtureEnv(fingerprints.hash),
                      lambda _gates, _config: True)


def test_ppo_initial_exploration_and_flat_action_head_are_frozen_controls():
    fingerprints = TransitionFingerprint(**{
        name: f"{index + 1:064x}"
        for index, name in enumerate(TransitionFingerprint.__dataclass_fields__)
    })
    config = RLConfig(
        algorithm="PPO",
        policy=PolicySpec(2, hidden_layers=(2,), value_layers=(2,),
                          observation_schema_hash="a" * 64, action_schema_hash="b" * 64),
        fingerprint=fingerprints,
        seeds=SeedSpec(algorithm=11, environments=(12,), noise=13, replay=14),
        hyperparameters={
            "n_steps": 2, "batch_size": 2, "n_epochs": 1,
            "log_std_init": -2.0, "zero_init_action_head": True,
        },
        device="cpu",
    )
    provider = RLProvider(
        config, lambda _rank, _seed, _phase: ProviderFixtureEnv(fingerprints.hash),
        lambda _gates, _config: True,
    )
    try:
        provider.create()
        assert torch.count_nonzero(provider.model.policy.action_net.weight).item() == 0
        assert torch.count_nonzero(provider.model.policy.action_net.bias).item() == 0
        assert torch.equal(provider.model.policy.log_std, torch.full_like(provider.model.policy.log_std, -2.0))
        assert provider.predict(np.zeros((1, 2), dtype=np.float32)).item() == 0.0
        performance = provider.train(2)
        assert performance.cpu_rss_bytes is None or performance.cpu_rss_bytes > 0
        assert performance.gpu_max_allocated_bytes is None
        assert performance.gpu_max_reserved_bytes is None
        assert performance.diagnostics is not None
        assert performance.diagnostics.decision_start_ns == 1
        assert performance.diagnostics.decision_end_ns == 2
        assert performance.diagnostics.unique_decision_timestamps == 2
        assert performance.diagnostics.decision_timestamp_revisit_fraction == 0.0
        assert performance.diagnostics.soft_risk_intervention_fraction == 0.0
        assert performance.diagnostics.hard_risk_intervention_fraction == 0.0
        assert performance.diagnostics.execution_plan_reasons == (("target changed", 2),)
        metrics = dict(performance.diagnostics.optimizer_metrics)
        assert metrics["train/std"] == pytest.approx(np.exp(-2), rel=0.02)
    finally:
        provider.close()


@pytest.mark.parametrize(
    "parameters",
    (
        {"n_steps": 2, "batch_size": 2, "n_epochs": 1, "log_std_init": -21.0},
        {"n_steps": 2, "batch_size": 2, "n_epochs": 1, "log_std_init": float("nan")},
        {"n_steps": 2, "batch_size": 2, "n_epochs": 1, "zero_init_action_head": 1},
    ),
)
def test_ppo_rejects_invalid_exploration_initialization(parameters):
    fingerprints = TransitionFingerprint(**{
        name: f"{index + 1:064x}"
        for index, name in enumerate(TransitionFingerprint.__dataclass_fields__)
    })
    with pytest.raises(ContractError, match="log_std_init|zero_init_action_head"):
        RLConfig(
            algorithm="PPO",
            policy=PolicySpec(2, observation_schema_hash="a" * 64, action_schema_hash="b" * 64),
            fingerprint=fingerprints,
            seeds=SeedSpec(11, (12,), 13, 14),
            hyperparameters=parameters,
        )


def test_cuda_allocator_fraction_is_validated_and_fingerprinted():
    from dataclasses import asdict, replace
    from rles_rl_sb3.config import digest

    base = _small_pretraining_provider().config
    assert base.hash == digest({
        "algorithm": base.algorithm, "policy": asdict(base.policy),
        "fingerprint": asdict(base.fingerprint), "seeds": asdict(base.seeds),
        "hyperparameters": dict(base.hyperparameters), "device": base.device,
        "normalize_observation": base.normalize_observation,
        "normalize_reward": base.normalize_reward, "noise_sigma": base.noise_sigma,
        "version": base.version,
    })
    with pytest.raises(ContractError, match="requires CUDA device"):
        replace(base, cuda_allocator_fraction=0.5)
    cuda_config = replace(base, device="cuda", cuda_allocator_fraction=0.5)
    assert cuda_config.hash != replace(base, device="cuda").hash
    for invalid in (True, 0, 1.01, float("nan")):
        with pytest.raises(ContractError, match="cuda_allocator_fraction"):
            replace(base, device="cuda", cuda_allocator_fraction=invalid)

    provider = RLProvider(cuda_config, lambda *_args: None, lambda *_args: True)
    calls = []
    fake_torch = type("Torch", (), {"cuda": type("Cuda", (), {
        "is_available": staticmethod(lambda: True),
        "current_device": staticmethod(lambda: 3),
        "set_per_process_memory_fraction": staticmethod(
            lambda fraction, device: calls.append((fraction, device))),
    })()})
    provider._configure_cuda_allocator(fake_torch)
    assert calls == [(0.5, 3)]


@pytest.mark.gpu
def test_cuda_allocator_fraction_runs_native_provider_training():
    from dataclasses import replace

    if not torch.cuda.is_available():
        pytest.skip("CUDA learner unavailable")
    template = _small_pretraining_provider()
    device = torch.cuda.current_device()
    previous_fraction = torch.cuda.get_per_process_memory_fraction(device)
    provider = RLProvider(
        replace(template.config, device="cuda", cuda_allocator_fraction=0.9),
        template.env_factory,
        template.gate_authorizer,
    )
    try:
        provider.create()
        assert torch.cuda.get_per_process_memory_fraction(device) == pytest.approx(0.9)
        performance = provider.train(2)
        assert performance.samples == 2
        torch.cuda.synchronize(device)
        assert performance.gpu_max_allocated_bytes is not None
        assert performance.gpu_max_reserved_bytes is not None
        assert performance.gpu_max_reserved_bytes >= performance.gpu_max_allocated_bytes
    finally:
        provider.close()
        torch.cuda.empty_cache()
        torch.cuda.set_per_process_memory_fraction(previous_fraction, device)


class ProviderFixtureEnv(gym.Env):
    """Synthetic boundary provider; tests L4 mechanics, not L3/engine signoff."""

    def __init__(self, transition_hash: str, restored_state=None) -> None:
        super().__init__()
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(2,), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
        self._transition_hash = transition_hash
        self._restored_state = restored_state
        self._step = 0 if restored_state is None else restored_state[0]
        self._observation = (np.zeros(2, dtype=np.float32) if restored_state is None
                             else restored_state[1].copy())
        self._actions = [] if restored_state is None else list(restored_state[2])
        self._reset_complete = False

    @property
    def rles_transition_hash(self) -> str:
        return self._transition_hash

    @property
    def rles_boundary_token(self) -> str:
        if not self._reset_complete:
            raise RuntimeError("boundary is available only after the Env has restored")
        return f"fixture-boundary-{self._step}"

    @property
    def rles_boundary_observation(self) -> np.ndarray:
        if not self._reset_complete:
            raise RuntimeError("boundary is available only after the Env has restored")
        return self._observation.copy()

    def rles_export_replay(self) -> dict:
        if not self._reset_complete:
            raise RuntimeError("replay is available only after the Env has restored")
        return {"schema": "rles.episode-replay.v1", "transition_hash": self._transition_hash,
                "actions": list(self._actions), "boundary_token": self.rles_boundary_token}

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._reset_complete = True
        if self._restored_state is None:
            self._step = 0
            self._observation = np.zeros(2, dtype=np.float32)
        return self._observation.copy(), {}

    def step(self, action: np.ndarray):
        self._step += 1
        requested = float(action[0])
        reward = -(requested ** 2)
        self._actions.append(str(requested))
        self._observation = np.asarray([self._step, requested], dtype=np.float32)
        return self._observation.copy(), reward, False, False, {
            "r_econ": reward,
            "r_shape": 0.0,
            "r_train": reward,
            "requested_action": requested,
            "policy_executed_action_pre_terminal": requested,
            "risk_guard_intervention": 0.0,
            "risk_soft_adjustment": 0.0,
            "risk_hard_adjustment": 0.0,
            "execution_plan_action": "SUBMIT",
            "execution_plan_reason": "target changed",
            "decision_ts_ns": self._step,
            "episode_window": {
                "start_ns": 1, "end_ns": 1_000_000,
                "episode_index": 0, "start_offset": 0, "decision_steps": 999_999,
            },
            "market_features": {"synthetic_step": float(self._step)},
            "net_exposure": requested,
        }


@pytest.mark.parametrize("early_stop", [False, True])
def test_bc_records_actual_epochs_and_stops_only_after_full_parity(early_stop):
    from rles_orchestrator.transfer import adapt_l4_distillation

    torch.set_num_threads(1)
    provider = _small_pretraining_provider()
    try:
        provider.create()
        with torch.no_grad():
            provider.model.policy.action_net.weight.zero_()
            provider.model.policy.action_net.bias.fill_(0.02)
        reports = []

        def clone(observations, labels):
            report = provider.distill(observations, labels, epochs=80, batch_size=8,
                                      learning_rate=0.01, tolerance=0.001, early_stop=early_stop)
            reports.append(report)
            return report

        distill = adapt_l4_distillation(clone, student=provider.predict, epochs=80)
        evidence = distill(np.zeros((8, 2), dtype=np.float32), np.zeros((8, 1), dtype=np.float32))
        assert reports[0].passed and reports[0].max_abs_error <= 0.001
        assert evidence.epochs == reports[0].epochs_completed
        assert (1 <= evidence.epochs < 80) if early_stop else evidence.epochs == 80
        assert provider.model.num_timesteps == 0
    finally:
        provider.close()


def test_train_only_normalizer_calibration_preserves_pretraining_account_variation():
    from rles_rl_sb3.errors import ContractError

    provider = _small_pretraining_provider(normalize=True)
    try:
        provider.create()
        raw = np.asarray([[9000, -1], [10000, 1], [11000, 3]], dtype=np.float32)
        assert len(np.unique(provider.policy_observations(raw)[:, 0])) == 1
        with pytest.raises(ContractError, match="Train-inner"):
            provider.calibrate_observation_normalization(raw, training_split="validation")
        stats = provider.calibrate_observation_normalization(raw, training_split="train_inner")
        assert stats["count"] == pytest.approx(3.0001)
        assert len(np.unique(provider.policy_observations(raw)[:, 0])) == 3
        with pytest.raises(ContractError, match="fresh pre-training"):
            provider.calibrate_observation_normalization(raw, training_split="train_inner")
        provider.train(2)
        with pytest.raises(ContractError, match="fresh pre-training"):
            provider.calibrate_observation_normalization(raw, training_split="train_inner")
    finally:
        provider.close()


def test_calibrated_train_outliers_are_not_aliased_by_normalizer_clipping():
    provider = _small_pretraining_provider(normalize=True)
    try:
        provider.create()
        raw = np.zeros((1001, 2), dtype=np.float32)
        raw[-1] = [100000, 100]
        stats = provider.calibrate_observation_normalization(raw, training_split="train_inner")
        assert stats["clip_obs"] > 30
        expected = (raw - np.asarray(stats["mean"])) / np.sqrt(np.asarray(stats["var"]) + stats["epsilon"])
        assert np.allclose(provider.policy_observations(raw), expected, atol=1e-5)
    finally:
        provider.close()


@pytest.mark.parametrize("algorithm", ("PPO", "SAC", "TD3", "DDPG"))
def test_l4_provider_create_and_train_uses_transition_guard(algorithm: str) -> None:
    torch.set_num_threads(1)
    fingerprints = TransitionFingerprint(**{name: f"{index + 1:064x}" for index, name in enumerate(
        TransitionFingerprint.__dataclass_fields__
    )})
    policy = PolicySpec(
        observation_dim=2,
        observation_schema_hash="a" * 64,
        action_schema_hash="b" * 64,
    )
    hyperparameters = (
        {"n_steps": 2, "batch_size": 2, "n_epochs": 1}
        if algorithm == "PPO" else
        {"learning_starts": 2, "buffer_size": 32, "batch_size": 2,
         "train_freq": 1, "gradient_steps": 1}
    )
    config = RLConfig(
        algorithm=algorithm,
        policy=policy,
        fingerprint=fingerprints,
        seeds=SeedSpec(algorithm=11, environments=(12,), noise=13, replay=14),
        hyperparameters=hyperparameters,
        device="cpu",
    )
    checks: list[tuple[str, ...]] = []
    provider = RLProvider(
        config,
        lambda _rank, _seed, _phase: ProviderFixtureEnv(fingerprints.hash),
        lambda gates, _config: checks.append(gates) is None or True,
    )
    try:
        provider.create()
        performance = provider.train(4)
        assert performance.samples == 4
        assert provider.model is not None and provider.model.num_timesteps == 4
        assert checks and all("L3-021" in gates for gates in checks)
    finally:
        provider.close()


@pytest.mark.parametrize("algorithm", ("SAC", "TD3", "DDPG"))
@pytest.mark.parametrize("scaled", [False, True])
def test_tanh_actor_transfer_separates_source_and_l4_mapping_signatures(algorithm: str, scaled: bool) -> None:
    from rles_es_evotorch.actor import ActorGenomeSpec, act, decode, encode
    from rles_orchestrator.transfer import adapt_l4_load, transfer
    from rles_rl_sb3.actor import extract_actor_weights
    from rles_rl_sb3.errors import ActorActionParityMismatch

    torch.set_num_threads(1)
    fingerprints = TransitionFingerprint(**{name: f"{index + 1:064x}" for index, name in enumerate(
        TransitionFingerprint.__dataclass_fields__
    )})
    policy = PolicySpec(
        observation_dim=2, hidden_layers=(2,), value_layers=(2,), activation="tanh",
        observation_schema_hash="a" * 64, action_schema_hash="b" * 64,
    )
    config = RLConfig(
        algorithm=algorithm, policy=policy, fingerprint=fingerprints,
        seeds=SeedSpec(algorithm=31, environments=(32,), noise=33, replay=34),
        hyperparameters={"learning_starts": 2, "buffer_size": 32, "batch_size": 2,
                         "train_freq": 1, "gradient_steps": 1}, device="cpu", normalize_observation=scaled,
    )
    provider = RLProvider(
        config, lambda _rank, _seed, _phase: ProviderFixtureEnv(fingerprints.hash),
        lambda _gates, _config: True,
    )
    try:
        provider.create()
        assert provider.model is not None
        source_spec = ActorGenomeSpec(("feature.x", "feature.y"), (2,), version=2 if scaled else 1,
                                      input_scale=(0.0001, 1.0) if scaled else None)
        source_genome = encode(source_spec, extract_actor_weights(provider.model, config))
        source_weights = decode(source_spec, source_genome)
        observations = np.asarray(
            [[-1.0, 0.5], [0.25, -0.75], [0.0, 0.0], [0.8, 0.9]], dtype=np.float32,
        )
        if scaled:
            observations[:, 0] = 10000 + observations[:, 0] * 2000
            stats = provider.calibrate_observation_normalization(observations, training_split="train_inner")
            raw_weights = source_weights["layers.0.weight"] * np.asarray(source_spec.input_scale, dtype=np.float32)
            source_weights["layers.0.bias"] += raw_weights @ np.asarray(stats["mean"], dtype=np.float32)
            source_weights["layers.0.weight"] = raw_weights * np.sqrt(
                np.asarray(stats["var"], dtype=np.float32) + float(stats["epsilon"]))
        actions = act(source_spec, source_genome, observations)
        destination_signature = provider.mapping()["actor_signature"]

        load_exact = adapt_l4_load(
            lambda: provider.transfer(
                source_weights, destination_signature, provider.policy_observations(observations), actions,
                source_actor_signature=source_spec.signature,
            ),
            source_actor_signature=source_spec.signature,
            action_parity_mismatch=ActorActionParityMismatch,
        )

        transfer_result = transfer(
            algorithm=algorithm, actor_signature=source_spec.signature,
            teacher=lambda batch: act(source_spec, source_genome, batch),
            student=lambda batch: provider.predict(batch),
            fixed_observations=observations, training_observations=observations,
            load_exact=load_exact,
            reset_student=lambda: pytest.fail("parity-valid exact transfer must not reset"),
            distill=lambda _obs, _labels: pytest.fail("parity-valid transfer must not distill"),
            tolerance=1e-5,
        )

        assert transfer_result.method == "DIRECT"
        assert transfer_result.parity.passed
        assert transfer_result.parity.actor_signature == source_spec.signature
        assert destination_signature != source_spec.signature
    finally:
        provider.close()


def test_l4_distillation_failure_reports_measured_parity_error() -> None:
    from types import SimpleNamespace
    from rles_orchestrator.transfer import adapt_l4_distillation

    observations = np.zeros((3, 2), dtype=np.float32)
    labels = np.zeros((3, 1), dtype=np.float32)
    failed = SimpleNamespace(method="behavioral_distillation", samples=3, passed=False,
                             max_abs_error=0.2, tolerance=0.01)
    distill = adapt_l4_distillation(
        lambda _observations, _labels: failed,
        student=lambda batch: np.zeros((len(batch), 1), dtype=np.float32), epochs=1,
    )
    with pytest.raises(RuntimeError, match="parity failed: max_abs_error=0.2, tolerance=0.01"):
        distill(observations, labels)


def test_ppo_clip_tanh_difference_uses_l5_train_inner_distillation_fallback() -> None:
    from rles_es_evotorch.actor import ActorGenomeSpec, act, decode, encode
    from rles_orchestrator.transfer import adapt_l4_distillation, adapt_l4_load, transfer
    from rles_rl_sb3.errors import ActorActionParityMismatch

    torch.set_num_threads(1)
    fingerprints = TransitionFingerprint(**{name: f"{index + 1:064x}" for index, name in enumerate(
        TransitionFingerprint.__dataclass_fields__
    )})
    policy = PolicySpec(
        observation_dim=2, hidden_layers=(2,), value_layers=(2,), activation="tanh",
        observation_schema_hash="a" * 64, action_schema_hash="b" * 64,
    )
    config = RLConfig(
        algorithm="PPO", policy=policy, fingerprint=fingerprints,
        seeds=SeedSpec(algorithm=41, environments=(42,), noise=43, replay=44),
        hyperparameters={"n_steps": 2, "batch_size": 2, "n_epochs": 1}, device="cpu",
    )
    provider = RLProvider(
        config, lambda _rank, _seed, _phase: ProviderFixtureEnv(fingerprints.hash),
        lambda _gates, _config: True,
    )
    try:
        provider.create()
        assert provider.model is not None
        # Force PPO's unsquashed action head into a region where inference
        # clipping has zero gradient; BC must still recover teacher actions.
        with torch.no_grad():
            provider.model.policy.action_net.weight.zero_()
            provider.model.policy.action_net.bias.fill_(8.0)
        initial_policy = {
            key: value.detach().clone()
            for key, value in provider.model.policy.state_dict().items()
        }
        source_spec = ActorGenomeSpec(("feature.x", "feature.y"), (2,))
        source_parameters = {
            name: np.zeros(shape, dtype=np.float32) for name, shape in source_spec.shapes
        }
        source_parameters["layers.1.bias"][0] = np.float32(0.8)
        source_genome = encode(source_spec, source_parameters)
        source_weights = decode(source_spec, source_genome)
        rng = np.random.default_rng(45)
        observations = rng.normal(0, 0.5, size=(80, 2)).astype(np.float32)
        training_observations, fixed_observations = observations[:64], observations[64:]
        teacher = lambda batch: act(source_spec, source_genome, batch)
        labels = teacher(fixed_observations)
        destination_signature = provider.mapping()["actor_signature"]

        load_exact = adapt_l4_load(
            lambda: provider.transfer(
                source_weights, destination_signature, fixed_observations, labels,
                source_actor_signature=source_spec.signature,
            ),
            source_actor_signature=source_spec.signature,
            action_parity_mismatch=ActorActionParityMismatch,
        )
        epochs = 500
        distill = adapt_l4_distillation(
            lambda train, targets: provider.distill(
                train, targets, epochs=epochs, batch_size=16,
                learning_rate=0.01, tolerance=0.08,
            ),
            student=lambda batch: provider.predict(batch), epochs=epochs,
        )

        def reset_student() -> None:
            assert provider.model is not None
            for key, value in provider.model.policy.state_dict().items():
                assert torch.equal(value, initial_policy[key]), key

        result = transfer(
            algorithm="PPO", actor_signature=source_spec.signature,
            teacher=teacher, student=lambda batch: provider.predict(batch),
            fixed_observations=fixed_observations, training_observations=training_observations,
            load_exact=load_exact, reset_student=reset_student, distill=distill,
            tolerance=0.08,
        )

        assert result.method == "DISTILLATION_NOT_WEIGHT_INHERITANCE"
        assert result.parity.passed
        assert result.parity.actor_signature == source_spec.signature
        assert result.distillation is not None
        assert result.distillation.training_samples == len(training_observations)
    finally:
        provider.close()


@pytest.mark.parametrize("algorithm", ("PPO", "SAC", "TD3", "DDPG"))
@pytest.mark.parametrize("normalize_observation", (False, True))
@pytest.mark.artifact_restore
def test_l4_stage_checkpoint_resumes_with_identical_cpu_weights(
    tmp_path, algorithm: str, normalize_observation: bool
) -> None:
    torch.set_num_threads(1)
    fingerprints = TransitionFingerprint(**{name: f"{index + 1:064x}" for index, name in enumerate(
        TransitionFingerprint.__dataclass_fields__
    )})
    policy = PolicySpec(2, observation_schema_hash="a" * 64, action_schema_hash="b" * 64)
    hyperparameters = (
        {"n_steps": 2, "batch_size": 2, "n_epochs": 1}
        if algorithm == "PPO" else
        {"learning_starts": 2, "buffer_size": 32, "batch_size": 2,
         "train_freq": 1, "gradient_steps": 1}
    )
    config = RLConfig(
        algorithm=algorithm, policy=policy, fingerprint=fingerprints,
        seeds=SeedSpec(algorithm=21, environments=(22,), noise=23, replay=24),
        hyperparameters=hyperparameters, device="cpu",
        normalize_observation=normalize_observation,
    )
    authorizer = lambda _gates, _config: True
    staged = RLProvider(config, lambda _rank, _seed, _phase: ProviderFixtureEnv(fingerprints.hash), authorizer)
    try:
        staged.create()
        staged.train(4)
        assert staged.vec is not None and staged.model is not None
        from rles_rl_sb3.envs import boundary_raw_observations

        token = staged.vec.get_attr("rles_boundary_token")[0]
        observation = boundary_raw_observations(staged.vec, 2)[0]
        saved_env_step = int(observation[0])
        saved_replay_actions = staged.vec.env_method("rles_export_replay")[0]["actions"]
        staged.vec.env_method("rles_restore_seed_schedule_state",
                              {"root_seed": 22, "next_episode": 7})
        boundary = Boundary("R1", 4, (token,), True, True)
        checkpoint = staged.save(tmp_path / "stage", boundary, attempt_id="attempt-1")
        assert checkpoint_episode_replays(checkpoint, config, trusted_source=True) == ({
            "schema": "rles.episode-replay.v1", "transition_hash": fingerprints.hash,
            "actions": saved_replay_actions, "boundary_token": token,
        },)
    finally:
        staged.close()

    restored_state = (saved_env_step, observation, saved_replay_actions)
    resumed = RLProvider(
        config,
        lambda _rank, _seed, _phase: ProviderFixtureEnv(fingerprints.hash, restored_state),
        authorizer,
    )
    try:
        resumed.load(checkpoint, trusted_source=True, restored_boundary=boundary)
        assert resumed.vec is not None
        assert resumed.vec.get_attr("rles_seed_schedule_state") == [
            {"root_seed": 22, "next_episode": 7},
        ]
        resumed.train(2, resumed=True)
        assert resumed.model is not None and resumed.model.num_timesteps == 6
        continuous = RLProvider(
            config, lambda _rank, _seed, _phase: ProviderFixtureEnv(fingerprints.hash), authorizer
        )
        try:
            continuous.create()
            continuous.train(6)
            assert continuous.model is not None
            for key, value in continuous.model.policy.state_dict().items():
                assert torch.equal(value, resumed.model.policy.state_dict()[key]), key
        finally:
            continuous.close()
    finally:
        resumed.close()
