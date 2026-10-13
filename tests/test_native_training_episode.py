"""Synthetic market, real native engine, actual independent L3/L4/L5 modules."""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from hashlib import sha256
from types import SimpleNamespace

import numpy as np
import pytest

from nautilus_trader.model import AccountId
from rles_contracts.dto import ExecutionPlan
from rles_data.core import Availability, CuratedStore, Event, FeatureField, FeatureSpec
from rles_data.features import make_feature_row
from rles_es_evotorch.actor import ActorGenomeSpec, act, encode, genome_hash
from rles_es_evotorch.evaluation import (
    EvaluationBudget, GenerationLaneEvaluator, WorkerLaneDispatcher,
)
from rles_es_evotorch.fitness import FitnessSpec
from rles_es_evotorch.soft_risk import SoftRiskDimension, SoftRiskSpace
from rles_execution import InstrumentRules, PendingOrder, to_contract_plan
from rles_gym_bridge import (
    BarrierEpisode, BarrierFrame, EpisodeReplay, NautilusBarrierEnv, RiskActionReview,
    ObservationBuilder, ObservationSchema, RewardConfig, visible_market_features,
)
from rles_nautilus_adapter import EngineSession, NativeFactsReader, NativePlanSubmitter
from rles_risk import HardRiskConfig, HardRiskGuard, SoftRiskParams, SoftRiskTransform

from test_execution_contract_bridge import _planner
from test_nautilus_adapter_smoke import _builder, _instrument, _order_batch

pytestmark = pytest.mark.micro_golden

_REWARD = RewardConfig(lambda_dd=0.5, lambda_guard=0.2)
_SCHEMA = ObservationSchema("0.4.0", (
    "market.close", "account.equity", "account.peak_equity", "account.drawdown",
    "risk.guard_intervention", "execution.prev_executed_target",
), ("USD", "USD", "USD", "ratio", "ratio", "ratio"))


def _quantity(snapshot) -> Decimal:
    return sum((Decimal(row["quantity"]) * (-1 if row["side"] == "SHORT" else 1)
                for row in snapshot.positions), Decimal(0))


def _episode(transition: str, replay: EpisodeReplay | None = None,
             soft_params: SoftRiskParams | None = None) -> BarrierEpisode:
    planner = _planner()
    rules = InstrumentRules(str(_instrument().id), Decimal("0.0001"),
                             Decimal("0.0001"), Decimal("0.01"), Decimal(1))
    submitter = NativePlanSubmitter(instrument_id=rules.instrument_id, quantity_unit="BTC",
                                   quantity_increment=rules.qty_step,
                                   execution_spec_hash=planner.config.fingerprint(), latency_model_id="base")
    session = EngineSession(
        build_native=_builder(submitter), submit_plan=submitter,
        decode_event=lambda event: event.native_bar,
        facts_reader=NativeFactsReader(account_id=AccountId.from_str("SIM-001"),
                                       instrument_id=_instrument().id, currency="USD"),
        contract_snapshot_factory=lambda **facts: SimpleNamespace(**facts),
        certified_barrier_protocol=EngineSession.BARRIER_PROTOCOL,
    )
    soft = SoftRiskTransform(soft_params or SoftRiskParams(Decimal("0.8"), Decimal(1), Decimal(0), Decimal(0)))
    hard = HardRiskGuard(HardRiskConfig(
        Decimal("50"), Decimal(1), Decimal(0), Decimal(0), Decimal(1),
        require_mark=False, require_funding=False,  # Synthetic spot fixture, no perpetual capability claim.
    ))

    def plan(target: Decimal, snapshot, *, terminal: bool = False) -> ExecutionPlan:
        pending = tuple(PendingOrder(row["client_order_id"], Decimal(row["leaves_qty"]) *
                                     (-1 if row["side"] == "SELL" else 1), row["status"])
                        for row in snapshot.orders)
        decision = planner.plan(
            decision_id=f"{'terminal' if terminal else 'decision'}-{snapshot.decision_ts_ns}",
            instrument=rules, target_exposure=target, risk_budget_notional=Decimal(100),
            equity=snapshot.equity, mark_price=Decimal(100), current_qty=_quantity(snapshot),
            pending=pending, terminal=terminal,
        )
        return to_contract_plan(decision, ExecutionPlan, latency_model_id="base", quantity_unit="BTC")

    def action_to_plan(action: Decimal, snapshot, drawdown: Decimal):
        softened = soft.apply(action, current_exposure=_quantity(snapshot),
                              drawdown=drawdown, funding_abs=None)
        free_margin = Decimal(snapshot.account_events[-1]["balances"][0]["free"])
        capped = hard.apply(
            softened.adjusted, budget_notional=Decimal(100), equity=snapshot.equity,
            free_margin=free_margin, current_notional=_quantity(snapshot) * 100,
            mark_available=False, funding_available=False, liquidation_capability=False,
            data_quality_grade="DQ-C",
        )
        return RiskActionReview(
            plan(capped.adjusted, snapshot), abs(action - capped.adjusted),
            softened.intervened_fraction, capped.intervened_fraction, True,
            softened.adjusted, capped.adjusted,
        )

    feature_spec = FeatureSpec("0.4.0", (FeatureField("close", "float64", "USD", "ohlcv_1m", 1, 1, "close"),))
    events = tuple(Event(
        f"episode-bar-{number}", "ohlcv_1m", number * 60_000_000_000, number * 60_000_000_000,
        number * 60_000_000_000, Availability.NATIVE, "a" * 64,
        {"instrument": rules.instrument_id, "open": "100", "high": "100", "low": "100", "close": "100", "volume": "100"},
        "synthetic-native-training-fixture",
    ) for number in range(1, 11))
    store = CuratedStore(events)
    frames = tuple(BarrierFrame(
        number * 60_000_000_000, (_order_batch(number),),
        visible_market_features(
            make_feature_row(store, feature_spec, number * 60_000_000_000, instrument=rules.instrument_id),
            feature_spec, _SCHEMA, instrument=rules.instrument_id, decision_ts_ns=number * 60_000_000_000,
        ), {},
    ) for number in range(1, 11))
    return BarrierEpisode(
        native_session=session, native_identity={"dataset_hash": "a" * 64, "fold_hash": "b" * 64, "config_hash": "c" * 64},
        transition_hash=transition, frames=frames, scoring_frames=8,
        plan_action=action_to_plan, plan_terminal=lambda snapshot: plan(Decimal(0), snapshot, terminal=True),
        executed_exposure=_quantity, reward_config=_REWARD, replay=replay,
    )


def _env(transition: str = "d" * 64, *, replay: EpisodeReplay | None = None,
         max_steps: int = 6, episodes: list | None = None,
         soft_params: SoftRiskParams | None = None) -> NautilusBarrierEnv:
    pending_replay = replay
    def factory(*, seed, options):
        nonlocal pending_replay
        episode = _episode(transition, pending_replay, soft_params)
        pending_replay = None  # Resume once; subsequent done/reset starts a fresh episode.
        if episodes is not None:
            episodes.append(episode)
        return episode
    return NautilusBarrierEnv(episode_factory=factory, observation_builder=ObservationBuilder(_SCHEMA),
                              reward_config=_REWARD, max_steps=max_steps, transition_hash=transition)


_ES_ACTOR_SPEC = ActorGenomeSpec(_SCHEMA.names, (2,))


def make_native_es_session(config: dict):
    """Spawn-importable real-Nautilus evaluation factory for L6 integration tests."""
    if config.get("candidate_kind") not in {"actor", "soft_risk"}:
        raise ValueError("invalid candidate kind")
    return _NativeESSession(
        config["candidate_kind"], frozen_actor_genome=config.get("frozen_actor_genome"),
    )


def make_reviewed_native_env_fixture(*, env_config: dict, candidate_context: dict,
                                     soft_risk_multipliers: dict) -> NautilusBarrierEnv:
    """Synthetic-data L3 factory injected into the installable ES worker adapter."""
    if (candidate_context.get("split") != "train_inner"
            or env_config.get("episode_steps") != 2
            or not isinstance(env_config.get("transition_hash"), str)
            or len(env_config["transition_hash"]) != 64
            or any(char not in "0123456789abcdef" for char in env_config["transition_hash"])):
        raise ValueError("native fixture recipe changed")
    soft = SoftRiskParams(Decimal("0.8"), Decimal(1), Decimal("0.2"), Decimal(0))
    if soft_risk_multipliers:
        soft = soft.with_multipliers(soft_risk_multipliers)
    return _env(transition=env_config["transition_hash"], max_steps=2,
                soft_params=soft)


def _installed_worker_config(candidate_kind: str) -> dict[str, object]:
    config: dict[str, object] = {
        "candidate_kind": candidate_kind,
        "env_factory_ref": f"{__name__}:make_reviewed_native_env_fixture",
        "env_config": {"transition_hash": "d" * 64, "episode_steps": 2},
        "actor_spec": {
            "observation_names": list(_ES_ACTOR_SPEC.observation_names),
            "hidden_sizes": list(_ES_ACTOR_SPEC.hidden_sizes),
        },
        "actor_signature": _ES_ACTOR_SPEC.signature,
        "transition_hash": "d" * 64,
        "episode_steps": 2,
    }
    if candidate_kind == "soft_risk":
        config["risk_dimensions"] = [vars(item) for item in _ES_TEST_RISK_SPACE.dimensions]
        config["frozen_actor_genome"] = [float(value) for value in _ES_FROZEN_R1_GENOME]
        config["frozen_actor_hash"] = genome_hash(_ES_ACTOR_SPEC, _ES_FROZEN_R1_GENOME)
    return config


class _NativeESSession:
    def __init__(self, candidate_kind: str, *, frozen_actor_genome=None):
        self.candidate_kind = candidate_kind
        if frozen_actor_genome is None:
            self.frozen_actor_genome = _ES_FROZEN_R1_GENOME.copy()
        else:
            self.frozen_actor_genome = np.asarray(frozen_actor_genome, dtype=np.float32)
            if (self.frozen_actor_genome.shape != (_ES_ACTOR_SPEC.size,)
                    or not np.isfinite(self.frozen_actor_genome).all()):
                raise ValueError("worker requires a valid frozen R1 actor genome")

    def evaluate(self, *, rles_context: dict) -> dict:
        if (rles_context.get("version") != 1 or rles_context.get("split") != "train_inner"
                or not isinstance(rles_context.get("account_isolation_id"), str)):
            raise ValueError("invalid L5 candidate context")
        actor_genome = np.asarray(rles_context["genome"], dtype=np.float32)
        if self.candidate_kind == "actor":
            if (rles_context["candidate_kind"] != "actor"
                    or rles_context["candidate_schema_hash"] != _ES_ACTOR_SPEC.signature):
                raise ValueError("E1 actor schema mismatch")
            soft = SoftRiskParams(Decimal("0.8"), Decimal(1), Decimal("0.2"), Decimal(0))
        else:
            if (rles_context["candidate_kind"] != "soft_risk"
                    or rles_context["actor_signature"] != _ES_ACTOR_SPEC.signature):
                raise ValueError("E2 must preserve the frozen R1 actor")
            risk_space = _ES_TEST_RISK_SPACE
            if rles_context["candidate_schema_hash"] != risk_space.signature:
                raise ValueError("E2 SoftRisk schema mismatch")
            risk_values = risk_space.decode(actor_genome)
            soft = SoftRiskParams(Decimal("0.8"), Decimal(1), Decimal("0.2"), Decimal(0))
            soft = soft.with_multipliers(risk_values)
            actor_genome = self.frozen_actor_genome
        env = _env(max_steps=2, soft_params=soft)
        initial_equity = None
        step_log_returns = []
        drawdowns = []
        constraint_violations = 0.0
        try:
            observation, reset_info = env.reset(seed=int(rles_context["seed"]))
            initial_equity = float(reset_info["initial_equity"])
            last_info = {}
            for _ in range(2):
                action = (act(_ES_ACTOR_SPEC, actor_genome, observation[None, :])[0]
                          if self.candidate_kind == "actor" else
                          act(_ES_ACTOR_SPEC, self.frozen_actor_genome, observation[None, :])[0])
                observation, _reward, terminated, truncated, last_info = env.step(action)
                step_log_returns.append(float(last_info["r_econ"]))
                drawdowns.append(float(last_info["drawdown"]))
                constraint_violations += float(last_info["risk_guard_intervention"])
                if terminated:
                    raise RuntimeError("native candidate episode terminated unexpectedly")
            if not truncated or not last_info.get("terminal_settled"):
                raise RuntimeError("native candidate episode lacks terminal settlement")
            evidence_hash = sha256(repr((
                rles_context["request_hash"], rles_context["account_isolation_id"],
                initial_equity, last_info["native_equity"], step_log_returns,
                drawdowns, constraint_violations, last_info["evidence_hash"],
            )).encode()).hexdigest()
            return {
                "initial_equity": initial_equity,
                "final_equity": float(last_info["native_equity"]),
                "step_log_returns": step_log_returns,
                "drawdowns": drawdowns,
                "constraint_violations": constraint_violations,
                "evidence_hash": evidence_hash,
                "request_hash": rles_context["request_hash"],
                "account_isolation_id": rles_context["account_isolation_id"],
                "native_equity": last_info["native_equity"],
            }
        finally:
            env.close()


_ES_TEST_RISK_SPACE = SoftRiskSpace((
    SoftRiskDimension("exposure_multiplier", 0.1, 1.0),
    SoftRiskDimension("drawdown_throttle", 0.2, 1.0),
    SoftRiskDimension("volatility_throttle", 0.1, 1.0),
))
_frozen_state = {name: np.zeros(shape, dtype=np.float32)
                 for name, shape in _ES_ACTOR_SPEC.shapes}
_frozen_state["layers.1.bias"][0] = np.float32(0.5)
_ES_FROZEN_R1_GENOME = encode(_ES_ACTOR_SPEC, _frozen_state)


@dataclass(frozen=True)
class _ParallelNativeEnvFactory:
    transition: str

    def __call__(self, rank, _seed, _phase):
        # Stagger terminal barriers so autoreset touches only the completed
        # native worker while the other account continues stepping.
        return _env(self.transition, max_steps=rank + 2)


def test_native_gym_pipeline_uses_risk_capped_fills_and_terminal_costs() -> None:
    env = _env(max_steps=1)
    try:
        assert env.rles_transition_hash == "d" * 64  # Available to SB3 before reset.
        env.reset(seed=1)
        observation, reward, terminated, truncated, info = env.step(np.array([1], dtype=np.float32))
        assert not terminated and truncated and info["terminal_settled"]
        assert info["executed_action"] == 0.5  # Hard cap, rather than the raw action 1.
        assert info["native_equity"] == "99999.90"  # Both actual native commissions.
        assert observation[-1] == 0  # Forced terminal flatten.
        assert reward == pytest.approx(np.log(99999.9 / 100000) - 0.2 * 0.5 - 0.5 * 1e-6)
        assert info["decision_ts_ns"] == 180_000_000_000
    finally:
        env.close()


@pytest.mark.artifact_restore
def test_native_episode_replay_restores_account_observation_reward_and_step_budget() -> None:
    episodes: list = []
    continuous = _env(episodes=episodes)
    restored = None
    try:
        continuous.reset(seed=1)
        continuous.step(np.array([1], dtype=np.float32))
        before = continuous.step(np.array([0.2], dtype=np.float32))
        replay = episodes[0].export_replay()
        expected_results = [continuous.step(np.array([value], dtype=np.float32))
                            for value in (0.0, -1.0, -0.2, 0.0)]
        continuous.close()
        restored = _env(replay=replay)
        observation, _info = restored.reset(seed=1)
        assert np.array_equal(observation, before[0])
        assert restored.rles_boundary_token == replay.boundary_token
        for value, expected in zip((0.0, -1.0, -0.2, 0.0), expected_results):
            action = np.array([value], dtype=np.float32)
            actual = restored.step(action)
            assert np.array_equal(expected[0], actual[0])
            assert expected[1:4] == actual[1:4]
            assert expected[4] == actual[4]
        assert actual[3] is True  # Restoring does not reset the max_steps budget.
    finally:
        continuous.close()
        if restored is not None:
            restored.close()


def test_native_sessions_reject_overlap_before_corrupting_first_account() -> None:
    from rles_nautilus_adapter import EngineBoundaryError

    first, second = _env(), _env()
    try:
        first.reset(seed=1)
        with pytest.raises(EngineBoundaryError, match="one live native EngineSession per process"):
            second.reset(seed=2)
        _observation, _reward, _terminated, _truncated, info = first.step(np.array([1], dtype=np.float32))
        assert info["executed_action"] == 0.5
        assert info["native_equity"] == "99999.95"
    finally:
        first.close()
        second.close()


@pytest.mark.parametrize("algorithm", ("PPO", "SAC", "TD3", "DDPG"))
def test_independent_rl_provider_trains_on_actual_native_episode(algorithm: str) -> None:
    import torch
    from rles_rl_sb3.config import PolicySpec, RLConfig, SeedSpec, TransitionFingerprint
    from rles_rl_sb3.provider import RLProvider

    torch.set_num_threads(1)
    fingerprint = TransitionFingerprint(**{name: f"{index + 1:064x}" for index, name in enumerate(TransitionFingerprint.__dataclass_fields__)})
    config = RLConfig(
        algorithm=algorithm, policy=PolicySpec(observation_dim=6, observation_schema_hash=_SCHEMA.hash,
                                              action_schema_hash="a" * 64, hidden_layers=(8,), value_layers=(8,)),
        fingerprint=fingerprint, seeds=SeedSpec(11, (12,), 13, 14), device="cpu",
        hyperparameters=({"n_steps": 2, "batch_size": 2, "n_epochs": 1} if algorithm == "PPO" else
                         {"learning_starts": 2, "buffer_size": 16, "batch_size": 2, "train_freq": 1, "gradient_steps": 1}),
    )
    # Test-only authorization: this is synthetic integration, not Q sign-off.
    provider = RLProvider(config, lambda _rank, _seed, _phase: _env(fingerprint.hash), lambda _gates, _cfg: True)
    try:
        provider.create()
        performance = provider.train(4)
        assert performance.samples == provider.model.num_timesteps == 4
        from rles_orchestrator.workflow import StageUsage
        assert StageUsage.from_rl_performances((performance,)) == StageUsage(rl_steps=4)
        assert provider.vec.get_attr("rles_transition_hash") == [fingerprint.hash]
    finally:
        provider.close()


def test_sb3_vecenv_trains_multiple_isolated_native_engines_with_staggered_resets() -> None:
    import torch
    from rles_rl_sb3.config import PolicySpec, RLConfig, SeedSpec, TransitionFingerprint
    from rles_rl_sb3.provider import RLProvider

    torch.set_num_threads(1)
    fingerprint = TransitionFingerprint(**{name: f"{index + 1:064x}" for index, name in enumerate(TransitionFingerprint.__dataclass_fields__)})
    config = RLConfig(
        algorithm="PPO", policy=PolicySpec(observation_dim=6, observation_schema_hash=_SCHEMA.hash,
                                           action_schema_hash="a" * 64, hidden_layers=(8,), value_layers=(8,)),
        fingerprint=fingerprint, seeds=SeedSpec(21, (22, 23), 24, 25), device="cpu",
        hyperparameters={"n_steps": 2, "batch_size": 4, "n_epochs": 1},
        normalize_observation=True,
    )
    provider = RLProvider(config, _ParallelNativeEnvFactory(fingerprint.hash), lambda _gates, _cfg: True)
    try:
        provider.create()
        performance = provider.train(8)
        assert performance.samples == provider.model.num_timesteps == 8
        assert provider.vec.num_envs == 2
        assert provider.vec.get_attr("rles_transition_hash") == [fingerprint.hash] * 2
        assert all(provider.vec.get_attr("rles_boundary_token"))
        assert float(provider.vec.obs_rms.count) > provider.vec.num_envs
    finally:
        provider.close()


def test_independent_es_actor_evaluates_on_same_native_episode() -> None:
    from rles_es_evotorch import ActorGenomeSpec, act, encode

    spec = ActorGenomeSpec(_SCHEMA.names, (2,))
    parameters = {name: np.zeros(shape, dtype=np.float32) for name, shape in spec.shapes}
    parameters["layers.1.bias"][0] = 1
    genome = encode(spec, parameters)
    env = _env(max_steps=2)
    try:
        observation, _info = env.reset(seed=1)
        for _ in range(2):
            action = act(spec, genome, observation[None, :])[0]
            observation, reward, _terminated, truncated, info = env.step(action)
            assert np.isfinite(reward)
        assert truncated and info["terminal_settled"]
        assert info["executed_action"] == 0.5
        assert info["native_equity"] == "99999.90"
    finally:
        env.close()


@pytest.mark.parametrize("algorithm", ("PPO", "SAC"))
@pytest.mark.artifact_restore
def test_native_rl_stage_restore_matches_continuous_weights(tmp_path, algorithm: str) -> None:
    import torch
    from rles_rl_sb3.checkpoint import Boundary
    from rles_rl_sb3.config import PolicySpec, RLConfig, SeedSpec, TransitionFingerprint
    from rles_rl_sb3.provider import RLProvider

    torch.set_num_threads(1)
    fingerprint = TransitionFingerprint(**{name: f"{index + 1:064x}" for index, name in enumerate(TransitionFingerprint.__dataclass_fields__)})
    config = RLConfig(
        algorithm=algorithm,
        policy=PolicySpec(observation_dim=6, observation_schema_hash=_SCHEMA.hash, action_schema_hash="a" * 64,
                          hidden_layers=(8,), value_layers=(8,)),
        fingerprint=fingerprint, seeds=SeedSpec(31, (32,), 33, 34), device="cpu",
        hyperparameters=({"n_steps": 2, "batch_size": 2, "n_epochs": 1} if algorithm == "PPO" else
                         {"learning_starts": 2, "buffer_size": 16, "batch_size": 2, "train_freq": 1, "gradient_steps": 1}),
    )
    episodes: list = []
    authorizer = lambda _gates, _cfg: True
    staged = RLProvider(config, lambda _rank, _seed, _phase: _env(fingerprint.hash, episodes=episodes), authorizer)
    try:
        staged.create()
        first_performance = staged.train(4)
        replay = episodes[0].export_replay()
        boundary = Boundary("R1", 4, (replay.boundary_token,), True, True)
        checkpoint = staged.save(tmp_path / "stage", boundary, attempt_id="native-stage")
    finally:
        staged.close()

    def restore_factory(_rank, seed, _phase):
        return _env(fingerprint.hash, replay=replay)

    resumed = RLProvider(config, restore_factory, authorizer)
    try:
        resumed.load(checkpoint, trusted_source=True, restored_boundary=boundary)
        resumed_performance = resumed.train(2, resumed=True)
        from rles_orchestrator.workflow import StageUsage
        assert StageUsage.from_rl_performances((first_performance, resumed_performance)) == StageUsage(
            rl_steps=6,
        )
        resumed_weights = {key: value.clone() for key, value in resumed.model.policy.state_dict().items()}
    finally:
        resumed.close()
    continuous = RLProvider(config, lambda _rank, _seed, _phase: _env(fingerprint.hash), authorizer)
    try:
        continuous.create()
        continuous.train(6)
        for key, value in continuous.model.policy.state_dict().items():
            assert torch.equal(value, resumed_weights[key]), key
    finally:
        continuous.close()


@pytest.mark.parametrize("algorithm", ("pgpe", "cmaes"))
def test_independent_es_provider_evolves_using_native_episode_fitness(algorithm: str) -> None:
    from rles_es_evotorch import (
        ActorGenomeSpec, EvoTorchProvider, EvolutionConfig, SoftRiskDimension, SoftRiskSpace, act,
    )

    actor = ActorGenomeSpec(_SCHEMA.names, (2,))
    baseline = SoftRiskParams(Decimal("0.8"), Decimal(1), Decimal("0.2"), Decimal(0))
    results = []

    def fitness(candidate, *, split):
        assert split == "train_inner"
        soft = baseline.with_multipliers(candidate) if algorithm == "cmaes" else baseline
        env = _env(max_steps=2, soft_params=soft)
        total_reward = 0.0
        try:
            observation, _info = env.reset(seed=10)
            for _ in range(2):
                action = (act(actor, candidate, observation[None, :])[0] if algorithm == "pgpe"
                          else np.array([1], dtype=np.float32))
                observation, reward, _terminated, truncated, info = env.step(action)
                total_reward += reward
                assert abs(info["executed_action"]) <= 0.5  # Immutable hard guard remains effective.
            assert truncated and info["terminal_settled"]
            results.append(info["native_equity"])
            return total_reward
        finally:
            env.close()

    space = (SoftRiskSpace(tuple(SoftRiskDimension(name, 0.1, 0.9) for name in
                                ("exposure_multiplier", "drawdown_throttle", "volatility_throttle")))
             if algorithm == "cmaes" else None)
    provider = EvoTorchProvider(EvolutionConfig(algorithm, 4, 1, 42, max_evaluations=4), fitness, risk_space=space)
    result = provider.evolve(dimension=actor.size if space is None else 3)
    assert result.generations == 1 and result.evaluations == len(results) == 4
    from rles_orchestrator.workflow import StageUsage
    assert StageUsage.from_evolution_result(result, population_size=provider.config.population_size) == StageUsage(
        population=4, generations=1, evaluations=4,
    )
    assert np.isfinite(result.best_fitness)


@pytest.mark.parametrize("worker_adapter", ["fixture", "installed"])
@pytest.mark.parametrize("candidate_kind", ["actor", "soft_risk"])
def test_l5_e1_e2_candidates_run_real_native_engines_in_l6_spawn_workers(
        tmp_path, candidate_kind: str, worker_adapter: str) -> None:
    """Exercise EvoTorch -> L5 context -> L6 spawn -> native Engine -> evidence."""
    from rles_es_evotorch.provider import EvoTorchProvider, EvolutionConfig
    from rles_compute_runtime.lanes import ESEvalLane
    from rles_compute_runtime.resources import ResourceQuota
    from rles_compute_runtime.workers import CPUWorkerPool

    actor_signature = _ES_ACTOR_SPEC.signature
    candidate_spec = _ES_ACTOR_SPEC if candidate_kind == "actor" else _ES_TEST_RISK_SPACE
    algorithm = "pgpe" if candidate_kind == "actor" else "cmaes"
    quota = ResourceQuota(tmp_path / f"{candidate_kind}-quota.sqlite",
                          cpu_cores=2, ram_bytes=4096, gpus=0)
    factory_ref = f"{__name__}:make_native_es_session"
    factory_config: dict[str, object] = {"candidate_kind": candidate_kind}
    if worker_adapter == "installed":
        factory_ref = "rles_orchestrator.native_es_worker:make_native_es_session"
        factory_config = _installed_worker_config(candidate_kind)
    pool = CPUWorkerPool(
        workers=2, factory_ref=factory_ref,
        factory_config=factory_config, mode="es", quota=quota,
        lease_owner=f"native-{candidate_kind}", ram_bytes_per_worker=1024,
        cpu_cores_per_worker=1, startup_timeout_s=30, torch_threads=1,
    )
    with pool:
        lane = ESEvalLane(pool, timeout_s=30)
        dispatcher = WorkerLaneDispatcher(
            lane, lambda _request: {},
            payload_builder_id=sha256(b"native-test-context-only-payload-v1").hexdigest(),
            fitness_spec=FitnessSpec(), tail_fraction=0.25,
        )
        budget = EvaluationBudget(4)
        generation_fitness = GenerationLaneEvaluator(
            run_id=f"native-{candidate_kind}", attempt_id="spawn-attempt", fold_id="native-fold",
            root_seed=1842, episode_ids=("native-episode",), candidate_spec=candidate_spec,
            actor_signature=actor_signature if candidate_kind == "soft_risk" else None,
            dispatch=dispatcher, dispatch_id=dispatcher.signature, budget=budget,
        )
        provider = EvoTorchProvider(
            EvolutionConfig(algorithm, 4, 1, 1842, max_evaluations=4),
            lambda _candidate, *, split: (_ for _ in ()).throw(AssertionError(split)),
            risk_space=_ES_TEST_RISK_SPACE if candidate_kind == "soft_risk" else None,
            batch_fitness=generation_fitness, batch_fitness_id=generation_fitness.signature,
        )
        result = provider.evolve(
            dimension=_ES_ACTOR_SPEC.size if candidate_kind == "actor" else len(_ES_TEST_RISK_SPACE.dimensions)
        )
        assert result.generations == 1 and result.evaluations == 4
        assert np.isfinite(result.best_fitness)
        assert budget.consumed == 4
    assert quota.active() == []


def test_installed_native_es_worker_rejects_forged_request_before_engine_creation() -> None:
    from rles_es_evotorch.evaluation import CandidateRequest
    from rles_orchestrator.native_es_worker import make_native_es_session as make_installed_session

    config = _installed_worker_config("actor")
    config["worker_index"] = 0
    session = make_installed_session(config)
    request = CandidateRequest(
        run_id="native-forged", attempt_id="attempt-1", generation=0,
        candidate_id="candidate-0", episode_id="episode-0", seed=71,
        genome=(0.0,) * _ES_ACTOR_SPEC.size,
        actor_signature=_ES_ACTOR_SPEC.signature, candidate_kind="actor",
        candidate_schema_hash=_ES_ACTOR_SPEC.signature,
        account_isolation_id="isolated-account-0",
    )
    context = {
        "version": 1, "request_hash": "0" * 64,
        "run_id": request.run_id, "attempt_id": request.attempt_id,
        "generation": request.generation, "candidate_id": request.candidate_id,
        "episode_id": request.episode_id, "seed": request.seed,
        "genome": list(request.genome), "actor_signature": request.actor_signature,
        "candidate_kind": request.candidate_kind,
        "candidate_schema_hash": request.candidate_schema_hash,
        "account_isolation_id": request.account_isolation_id, "split": request.split,
    }
    with pytest.raises(ValueError, match="request identity differs"):
        session.evaluate(rles_context=context)

    changed_r1 = _installed_worker_config("soft_risk")
    changed_r1["worker_index"] = 0
    changed_r1["frozen_actor_hash"] = "0" * 64
    with pytest.raises(ValueError, match="frozen R1 actor weights differ"):
        make_installed_session(changed_r1)


@pytest.mark.artifact_restore
def test_native_engine_es_checkpoint_resume_uses_l6_reservations_and_receipt(tmp_path) -> None:
    """Restore a real Native Engine E1 generation in a fresh spawn pool and attest it."""
    import random
    from pathlib import Path

    import torch

    from rles_es_evotorch.provider import EvoTorchProvider, EvolutionConfig
    from rles_es_evotorch.evaluation import CandidateRequest, scenario_seed
    from rles_compute_runtime.lanes import ESEvalLane
    from rles_compute_runtime.resources import ResourceQuota
    from rles_compute_runtime.workers import CPUWorkerPool
    from rles_orchestrator.es_stage import ESStageExecutor
    from rles_orchestrator.workflow import (
        Orchestrator, ProviderBackedStageRunner, Route, RunSpec, StageBudget, StageContext,
    )
    from rles_research.ledger import ResearchLedger, SelectionLedger

    torch.set_num_threads(1)
    actor_spec = _ES_ACTOR_SPEC
    hashes = {name: sha256(f"native-resume-{name}".encode()).hexdigest()
              for name in ("data", "feature", "config", "hard", "execution", "normalization", "soft")}
    budget = StageBudget(
        max_population=4, max_generations=2, max_evaluations=8, max_lane_requests=8,
    )
    spec = RunSpec(
        run_id="native-resume-run", attempt_id="native-resume-attempt", route=Route.A1,
        algorithm="PPO", seed=1842, data_hash=hashes["data"], feature_hash=hashes["feature"],
        config_hash=hashes["config"], hard_risk_hash=hashes["hard"],
        execution_hash=hashes["execution"], normalization_hash=hashes["normalization"],
        soft_risk_schema_hash=_ES_TEST_RISK_SPACE.signature,
        initial_soft_risk_hash=hashes["soft"], actor_signature=actor_spec.signature,
        stage_budget={"E1": budget},
    )
    research = ResearchLedger(tmp_path / "native-resume-ledger.sqlite")
    research.create("native-resume-protocol", spec.run_id, validation_budget=1,
                    hpo_budget=0, es_evaluation_budget=8)
    selection = SelectionLedger(research, "native-resume-protocol")
    payload_builder_id = sha256(b"native-resume-empty-payload-v1").hexdigest()

    def make_provider(owner: str, *, durable: bool):
        quota = ResourceQuota(
            tmp_path / f"{owner}-quota.sqlite", cpu_cores=2, ram_bytes=4096, gpus=0,
        )
        pool = CPUWorkerPool(
            workers=2, factory_ref="rles_orchestrator.native_es_worker:make_native_es_session",
            factory_config=_installed_worker_config("actor"), mode="es", quota=quota,
            lease_owner=owner, ram_bytes_per_worker=1024, cpu_cores_per_worker=1,
            startup_timeout_s=30, torch_threads=1,
        )
        dispatcher = WorkerLaneDispatcher(
            ESEvalLane(pool, timeout_s=30), lambda _request: {},
            payload_builder_id=payload_builder_id, fitness_spec=FitnessSpec(), tail_fraction=0.25,
        )
        if durable:
            evaluation_budget = EvaluationBudget(
                budget.max_lane_requests,
                reserve_requests=lambda requests: selection.reserve_lane_requests(
                    spec, stage="E1", requests=requests,
                ),
                verify_requests=lambda keys: selection.verify_lane_requests(
                    spec, stage="E1", request_keys=keys,
                ),
                reservation_id=selection.reservation_scope(spec, stage="E1"),
            )
        else:
            evaluation_budget = EvaluationBudget(budget.max_lane_requests)
        generation_fitness = GenerationLaneEvaluator(
            run_id=spec.run_id, attempt_id=spec.attempt_id, fold_id="native-resume-fold",
            root_seed=spec.seed, episode_ids=("native-episode",), candidate_spec=actor_spec,
            dispatch=dispatcher, dispatch_id=dispatcher.signature, budget=evaluation_budget,
        )
        provider = EvoTorchProvider(
            EvolutionConfig("pgpe", 4, 2, spec.seed,
                            max_evaluations=budget.max_evaluations),
            lambda _candidate, *, split: (_ for _ in ()).throw(AssertionError(split)),
            batch_fitness=generation_fitness, batch_fitness_id=generation_fitness.signature,
        )
        return pool, quota, provider, generation_fitness, evaluation_budget

    def seed_globals() -> None:
        random.seed(1842)
        np.random.seed(1842)
        torch.manual_seed(1842)

    seed_globals()
    continuous_pool, continuous_quota, continuous, continuous_eval, continuous_budget = make_provider(
        "native-continuous", durable=False,
    )
    with continuous_pool:
        expected = continuous.evolve(dimension=actor_spec.size)
        assert continuous_budget.consumed == 8
    assert continuous_quota.active() == []

    seed_globals()
    first_pool, first_quota, interrupted, _, first_budget = make_provider(
        "native-interrupted", durable=True,
    )
    with first_pool:
        partial = interrupted.evolve(dimension=actor_spec.size, generations=1)
        assert partial.evaluations == 4 and first_budget.consumed == 4
    assert first_quota.active() == []
    checkpoint = interrupted.save(
        str(tmp_path / "native-e1-checkpoint"), attempt_id=spec.attempt_id,
        actor_signature=actor_spec.signature, dataset_hash=spec.data_hash,
        config_hash=spec.config_hash,
    )

    seed_globals()
    restored_quota = ResourceQuota(
        tmp_path / "native-restored-quota.sqlite", cpu_cores=2, ram_bytes=4096, gpus=0,
    )
    runtime_kwargs = dict(
        stage="E1", artifact_id="native-resumed-e1", candidate_spec=actor_spec,
        evolution=EvolutionConfig("pgpe", 4, 2, spec.seed,
                                  max_evaluations=budget.max_evaluations),
        selection=selection, quota=restored_quota,
        factory_ref="rles_orchestrator.native_es_worker:make_native_es_session",
        factory_config=_installed_worker_config("actor"), payload_builder=lambda _request: {},
        payload_builder_id=payload_builder_id, fitness_spec=FitnessSpec(), tail_fraction=0.25,
        fold_id="native-resume-fold", episode_ids=("native-episode",),
        checkpoint_dir=Path(checkpoint).parent, workers=2, lease_owner="native-restored",
        ram_bytes_per_worker=1024, startup_timeout_s=30, timeout_s=30,
    )
    # Model a process failure after the next generation's requests were durably
    # reserved but before its optimizer checkpoint was committed. A retry from
    # generation one must fail before spawning workers or taking a quota lease.
    partial_research = ResearchLedger(tmp_path / "native-partial-ledger.sqlite")
    partial_research.create("native-partial-protocol", spec.run_id, validation_budget=1,
                            hpo_budget=0, es_evaluation_budget=8)
    partial_selection = SelectionLedger(partial_research, "native-partial-protocol")
    partial_request = CandidateRequest(
        run_id=spec.run_id, attempt_id=spec.attempt_id, generation=1,
        candidate_id="g00000001-c00000000", episode_id="native-episode",
        seed=scenario_seed(spec.seed, "native-resume-fold", 1, "native-episode"),
        genome=tuple(0.0 for _ in range(actor_spec.size)), actor_signature=actor_spec.signature,
        candidate_kind="actor", candidate_schema_hash=actor_spec.signature,
        account_isolation_id="acct-partial-generation-probe",
    )
    partial_selection.reserve_lane_requests(spec, stage="E1", requests=(partial_request,))
    partial_quota = ResourceQuota(
        tmp_path / "native-partial-quota.sqlite", cpu_cores=2, ram_bytes=4096, gpus=0,
    )
    partial_executor = ESStageExecutor(
        **(runtime_kwargs | {"selection": partial_selection, "quota": partial_quota}), resume=True,
    )
    with pytest.raises(RuntimeError, match="incomplete generation cannot resume"):
        partial_executor._preflight(StageContext(spec, "E1", budget, None))
    assert partial_quota.active() == []

    with pytest.raises(RuntimeError, match="already started"):
        ESStageExecutor(**runtime_kwargs)(StageContext(spec, "E1", budget, None))
    assert restored_quota.active() == []
    executor = ESStageExecutor(**runtime_kwargs, resume=True)
    outputs = []

    def execute(context):
        output = executor(context)
        outputs.append(output)
        return output

    lineage = Orchestrator(
        {"E1": ProviderBackedStageRunner("E1", execute, selection)},
        attest_usage=selection,
    ).run(spec)
    assert len(outputs) == 1
    actual = outputs[0].evolution_result
    assert actual == expected
    assert outputs[0].lane_requests == 8
    assert restored_quota.active() == []
    assert outputs[0].metadata["lane_evidence_hash"] == continuous_eval.evidence_hash
    assert research.evidence_summary("native-resume-protocol")["es_lane_request_count"] == 8
    assert (Path(checkpoint).parent / "generation-00000002" / spec.attempt_id).is_dir()
    completed = ESStageExecutor(**runtime_kwargs, resume=True)(StageContext(spec, "E1", budget, None))
    assert completed == outputs[0]
    assert restored_quota.active() == []
    assert research.evidence_summary("native-resume-protocol")["es_lane_request_count"] == 8
    artifact = lineage[0]
    assert artifact.usage.lane_requests == 8 and artifact.usage.evaluations == 8
    assert artifact.payload_hash == outputs[0].payload_hash
    assert len(research.evidence_summary("native-resume-protocol")["stage_receipts"]) == 1


@pytest.mark.parametrize("normalize", (False, True))
def test_a4_provider_backed_workflow_runs_real_es_rl_es_and_l6_ledger(tmp_path, normalize, monkeypatch) -> None:
    """Run one synthetic A4 cycle through real Providers, Native Engine, and SQLite L6."""
    import json

    import torch

    from rles_es_evotorch.actor import ActorGenomeSpec
    from rles_es_evotorch.fitness import FitnessSpec
    from rles_es_evotorch.provider import EvolutionConfig
    from rles_compute_runtime.resources import ResourceQuota
    from rles_orchestrator.es_stage import ESStageExecutor
    from rles_orchestrator.rl_stage import RLStageExecutor
    from rles_orchestrator.workflow import (
        Orchestrator, ProviderBackedStageRunner, ProviderStageOutput, Route,
        RunSpec, StageBudget, StageContext, lineage_hash,
    )
    from rles_research.ledger import ResearchLedger, SelectionLedger
    from rles_rl_sb3.config import PolicySpec, RLConfig, SeedSpec, TransitionFingerprint

    if normalize:
        from rles_rl_sb3.provider import RLProvider

        class ProviderWithFrozenTransferStats(RLProvider):
            def create(self):
                super().create()
                self.vec.obs_rms.mean.fill(0.25)
                self.vec.obs_rms.var.fill(0.5)
                self.vec.obs_rms.count = 100
                return self

        monkeypatch.setattr("rles_orchestrator.rl_stage.RLProvider", ProviderWithFrozenTransferStats)

    torch.set_num_threads(1)
    actor_spec = ActorGenomeSpec(_SCHEMA.names, (2,))
    risk_space = _ES_TEST_RISK_SPACE
    hashes = {name: sha256(f"a4-{name}".encode()).hexdigest()
              for name in ("data", "hard", "execution", "normalization", "soft-initial")}
    fingerprint_values = {
        name: f"{index + 101:064x}"
        for index, name in enumerate(TransitionFingerprint.__dataclass_fields__)
    }
    fingerprint_values.update({
        "dataset": hashes["data"], "feature": _SCHEMA.hash,
        "observation": _SCHEMA.hash, "execution": hashes["execution"],
        "hard_risk": hashes["hard"], "soft_risk": hashes["soft-initial"],
        "normalization": hashes["normalization"],
    })
    fingerprint = TransitionFingerprint(**fingerprint_values)
    rl_config = RLConfig(
        algorithm="PPO",
        policy=PolicySpec(
            observation_dim=len(_SCHEMA.names), hidden_layers=(2,), value_layers=(2,),
            activation="tanh", observation_schema_hash=_SCHEMA.hash,
            action_schema_hash=sha256(b"rles-target-action-v1").hexdigest(),
        ),
        fingerprint=fingerprint, seeds=SeedSpec(71, (72,), 73, 74), device="cpu",
        hyperparameters={"n_steps": 2, "batch_size": 2, "n_epochs": 1},
        normalize_observation=normalize,
    )
    budgets = {
        "E1": StageBudget(max_population=4, max_generations=1, max_evaluations=4,
                          max_lane_requests=8),
        "R1": StageBudget(max_rl_steps=4),
        "E2": StageBudget(max_population=4, max_generations=1, max_evaluations=4,
                          max_lane_requests=8),
    }
    base_spec = RunSpec(
        run_id="native-a4-run", attempt_id="native-a4-attempt", route=Route.A4,
        algorithm="PPO", seed=71, data_hash=hashes["data"],
        feature_hash=_SCHEMA.hash, config_hash=rl_config.hash,
        hard_risk_hash=hashes["hard"], execution_hash=hashes["execution"],
        normalization_hash=hashes["normalization"],
        soft_risk_schema_hash=risk_space.signature,
        initial_soft_risk_hash=hashes["soft-initial"],
        actor_signature=actor_spec.signature, stage_budget=budgets,
    )
    research = ResearchLedger(tmp_path / "a4-research.sqlite")
    research.create("native-a4-protocol", base_spec.run_id, validation_budget=1,
                    hpo_budget=8, es_evaluation_budget=16)
    selection = SelectionLedger(research, "native-a4-protocol")
    state: dict[str, object] = {}

    def es_executor(stage: str, candidate_spec, factory_config: dict[str, object]) -> ESStageExecutor:
        factory_config["actor_signature"] = base_spec.actor_signature
        factory_config["transition_hash"] = fingerprint.hash
        factory_config["env_config"]["transition_hash"] = fingerprint.hash
        quota = ResourceQuota(tmp_path / f"{stage.lower()}-lane-quota.sqlite",
                              cpu_cores=1, ram_bytes=2048, gpus=0)
        state[f"{stage.lower()}_quota"] = quota
        return ESStageExecutor(
            stage=stage, artifact_id=f"native-a4-{stage.lower()}", candidate_spec=candidate_spec,
            evolution=EvolutionConfig("pgpe" if stage == "E1" else "cmaes", 4, 1,
                                      base_spec.seed, max_evaluations=4),
            selection=selection, quota=quota,
            factory_ref="rles_orchestrator.native_es_worker:make_native_es_session",
            factory_config=factory_config, payload_builder=lambda _request: {},
            payload_builder_id=sha256(f"a4-{stage}-native-payload-v1".encode()).hexdigest(),
            fitness_spec=FitnessSpec(), tail_fraction=0.25,
            fold_id="a4-train-inner",
            episode_ids=(f"{stage.lower()}-episode-a", f"{stage.lower()}-episode-b"),
            checkpoint_dir=tmp_path / f"{stage.lower()}-checkpoint",
            workers=1, lease_owner=f"{stage.lower()}-a4", ram_bytes_per_worker=1024,
            startup_timeout_s=30, timeout_s=30,
        )

    def execute_e1(context: StageContext) -> ProviderStageOutput:
        assert context.stage == "E1"
        output = es_executor("E1", actor_spec, _installed_worker_config("actor"))(context)
        result = output.evolution_result
        assert result is not None
        genome = np.asarray(result.best_search_vector, dtype=np.float32)
        state["e1_genome"] = genome
        assert state["e1_quota"].active() == []
        return output

    def execute_r1(context: StageContext) -> ProviderStageOutput:
        assert context.stage == "R1" and context.previous is not None
        genome = state["e1_genome"]
        assert isinstance(genome, np.ndarray)
        rng = np.random.default_rng(75)
        train_observations = rng.normal(0, 0.35, size=(64, len(_SCHEMA.names))).astype(np.float32)
        fixed_observations = rng.normal(0, 0.35, size=(16, len(_SCHEMA.names))).astype(np.float32)
        executor = RLStageExecutor(
            artifact_id="native-a4-r1", config=rl_config,
            env_factory=lambda _rank, _seed, _phase: _env(fingerprint.hash, max_steps=2),
            gate_authorizer=lambda _gates, _config: True,
            train_steps=4, inference_dir=tmp_path / "a4-r1-inference",
            fixed_observations=fixed_observations, actor_spec=actor_spec,
            source_genome=genome, training_observations=train_observations,
            parity_tolerance=0.12, distill_epochs=120, distill_batch_size=16,
            distill_learning_rate=0.01,
        )
        output = executor(context)
        state["r1_inference_artifact"] = str(executor.inference_dir)
        state["trained_actor_hash"] = output.actor_hash
        return output

    def execute_e2(context: StageContext) -> ProviderStageOutput:
        assert context.stage == "E2" and context.previous is not None
        stale_canonical = _installed_worker_config("soft_risk")
        stale_canonical["frozen_actor_hash"] = context.previous.actor_hash
        with pytest.raises(ValueError, match="exported L4 inference actor"):
            es_executor("E2", risk_space, stale_canonical)(context)
        assert state["e2_quota"].active() == []
        factory_config = _installed_worker_config("soft_risk")
        factory_config.pop("frozen_actor_genome")
        factory_config["actor_inference"] = "l4_inference"
        factory_config["frozen_actor_artifact"] = state["r1_inference_artifact"]
        factory_config["frozen_actor_hash"] = state["trained_actor_hash"]
        missing_artifact = dict(factory_config, frozen_actor_artifact=str(tmp_path / "missing-r1"))
        with pytest.raises(FileNotFoundError):
            es_executor("E2", risk_space, missing_artifact)(context)
        assert state["e2_quota"].active() == []
        assert research.evidence_summary("native-a4-protocol")["es_lane_request_count"] == 8
        output = es_executor("E2", risk_space, factory_config)(context)
        assert state["e2_quota"].active() == []
        return output

    runners = {
        "E1": ProviderBackedStageRunner("E1", execute_e1, selection),
        "R1": ProviderBackedStageRunner("R1", execute_r1, selection),
        "E2": ProviderBackedStageRunner("E2", execute_e2, selection),
    }
    orchestrator = Orchestrator(runners, attest_usage=selection)
    lineage = orchestrator.run(base_spec)
    assert tuple(item.stage for item in lineage) == ("E1", "R1", "E2")
    assert all((tmp_path / f"{stage.lower()}-checkpoint" / "generation-00000001"
                / base_spec.attempt_id).is_dir() for stage in ("E1", "E2"))
    assert lineage[1].transfer_evidence is not None
    assert lineage[1].transfer_evidence.method == "DISTILLATION_NOT_WEIGHT_INHERITANCE"
    assert lineage_hash(lineage)
    assert all(len(lineage[index].metadata["lane_evidence_hash"]) == 64 for index in (0, 2))
    e1_payload = sha256(
        state["e1_genome"].tobytes(order="C")
        + bytes.fromhex(lineage[0].metadata["lane_evidence_hash"]),
    ).hexdigest()
    e2_payload = sha256(
        bytes.fromhex(lineage[2].soft_risk_hash)
        + bytes.fromhex(lineage[2].metadata["lane_evidence_hash"]),
    ).hexdigest()
    assert lineage[0].payload_hash == e1_payload
    assert lineage[2].payload_hash == e2_payload
    receipts = research.evidence_summary("native-a4-protocol")["stage_receipts"]
    assert len(receipts) == 3
    assert {item["stage"] for item in receipts} == {"E1", "R1", "E2"}
    receipt_payloads = {item["stage"]: item["payload_hash"] for item in receipts}
    assert receipt_payloads["E1"] == lineage[0].payload_hash
    assert receipt_payloads["E2"] == lineage[2].payload_hash
    usage_by_stage = {item["stage"]: json.loads(item["usage_json"]) for item in receipts}
    assert usage_by_stage["E1"]["lane_requests"] == 8
    assert usage_by_stage["E2"]["lane_requests"] == 8
    assert usage_by_stage["E1"]["evaluations"] == 4
    assert usage_by_stage["E2"]["evaluations"] == 4
    assert research.evidence_summary("native-a4-protocol")["es_lane_request_count"] == 16
    assert research.verify_audit_chain()
    assert all(item.usage.check(item.stage, base_spec.stage_budget[item.stage]) is None for item in lineage)


@pytest.mark.parametrize("current,target", [("2", "0.3"), ("-2", "-0.3")])
def test_soft_risk_allows_reducing_native_exposure_after_market_drift(current: str, target: str) -> None:
    baseline = SoftRiskParams(Decimal("0.8"), Decimal("0.1"), Decimal("0.2"), Decimal(0))
    reduced = baseline.with_multipliers({"exposure_multiplier": 0.7, "drawdown_throttle": 0.3,
                                        "volatility_throttle": 0.8})
    base_decision = SoftRiskTransform(baseline).apply(Decimal(target), current_exposure=Decimal(current),
                                                    drawdown=Decimal("0.2"), funding_abs=None)
    decision = SoftRiskTransform(reduced).apply(Decimal(target), current_exposure=Decimal(current),
                                               drawdown=Decimal("0.2"), funding_abs=None)
    assert abs(decision.adjusted) <= abs(base_decision.adjusted) < abs(Decimal(current))
