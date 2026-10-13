from __future__ import annotations

import random
from hashlib import sha256

import numpy as np
import pytest

from rles_es_evotorch.actor import ActorGenomeSpec
from rles_es_evotorch.evaluation import (
    CandidateRequest, CandidateResult, EvaluationBudget, GenerationLaneEvaluator,
)
from rles_es_evotorch.provider import EvolutionConfig, EvoTorchProvider
from rles_es_evotorch.soft_risk import SoftRiskDimension, SoftRiskSpace

pytestmark = [pytest.mark.algorithm_smoke, pytest.mark.artifact_restore]

_ACTOR = "a" * 64
_DATASET = "b" * 64
_CONFIG = "c" * 64


def _provider(algorithm: str, seen: list[tuple[float, ...]], *, seed: int = 712) -> EvoTorchProvider:
    risk_space = (SoftRiskSpace((SoftRiskDimension("exposure_multiplier", 0.1, 1.0),))
                  if algorithm == "cmaes" else None)

    def fitness(candidate: np.ndarray | dict[str, float], *, split: str) -> float:
        import torch
        assert split == "train_inner"
        values = np.asarray(list(candidate.values()) if isinstance(candidate, dict) else candidate,
                            dtype=np.float64)
        seen.append(tuple(float(value) for value in values))
        stochastic_probe = random.random() + float(np.random.random()) + float(torch.rand(()))
        return -float(np.square(values - 0.2).sum()) + stochastic_probe * 1e-4

    config = EvolutionConfig(algorithm, 4, 4, seed, max_evaluations=16)
    return EvoTorchProvider(config, fitness, risk_space=risk_space)


@pytest.mark.parametrize("algorithm", ["pgpe", "cmaes"])
def test_evolution_checkpoint_resume_matches_uninterrupted_candidate_stream(
        tmp_path, algorithm: str) -> None:
    import torch

    def seed_globals() -> None:
        random.seed(9182)
        np.random.seed(9182)
        torch.manual_seed(9182)

    seed_globals()
    continuous_seen: list[tuple[float, ...]] = []
    continuous = _provider(algorithm, continuous_seen)
    continuous_result = continuous.evolve(dimension=1)

    seed_globals()
    resumed_seen: list[tuple[float, ...]] = []
    interrupted = _provider(algorithm, resumed_seen)
    partial_result = interrupted.evolve(dimension=1, generations=2)
    assert partial_result.generations == 2
    checkpoint_path = interrupted.save(
        str(tmp_path / f"{algorithm}-checkpoint"), attempt_id=f"{algorithm}-attempt",
        actor_signature=_ACTOR, dataset_hash=_DATASET, config_hash=_CONFIG,
    )
    saved_generator = interrupted._searcher.problem._generator.get_state().clone()
    saved_population = interrupted._searcher._population.access_values(keep_evals=True).clone()
    saved_evaluations = interrupted._searcher._population.access_evals().clone()
    saved_distribution = ({key: value.clone() if isinstance(value, torch.Tensor) else value
                           for key, value in interrupted._searcher._distribution._Distribution__parameters.items()}
                          if algorithm == "pgpe" else None)
    saved_velocity = (interrupted._searcher._optimizer._velocity.clone()
                      if algorithm == "pgpe" else None)
    expected_continuation = interrupted.continue_evolution()

    expected_seen = list(resumed_seen)
    partial_count = partial_result.evaluations
    restored_seen: list[tuple[float, ...]] = []
    restored = _provider(algorithm, restored_seen)
    assert restored.resume(
        checkpoint_path.rsplit("\\", 1)[0], attempt_id=f"{algorithm}-attempt",
        actor_signature=_ACTOR, dataset_hash=_DATASET, config_hash=_CONFIG,
    ) == 0
    assert torch.equal(restored._searcher.problem._generator.get_state(), saved_generator)
    assert torch.equal(restored._searcher._population.access_values(keep_evals=True), saved_population)
    assert torch.equal(restored._searcher._population.access_evals(), saved_evaluations)
    if algorithm == "pgpe":
        for key, value in saved_distribution.items():
            actual = restored._searcher._distribution._Distribution__parameters[key]
            assert torch.equal(actual, value) if isinstance(value, torch.Tensor) else actual == value
        assert torch.equal(restored._searcher._optimizer._velocity, saved_velocity)
    actual_continuation = restored.continue_evolution()

    assert expected_seen == continuous_seen[:len(expected_seen)]
    assert restored_seen == continuous_seen[partial_count:len(expected_seen)]
    assert actual_continuation == expected_continuation == continuous_result


def test_evolution_checkpoint_resume_rejects_fingerprint_drift(tmp_path) -> None:
    seen: list[tuple[float, ...]] = []
    provider = _provider("pgpe", seen)
    provider.evolve(dimension=2, generations=1)
    path = provider.save(str(tmp_path / "checkpoint"), attempt_id="fingerprint-attempt",
                         actor_signature=_ACTOR, dataset_hash=_DATASET, config_hash=_CONFIG)
    restored = _provider("pgpe", [])
    with pytest.raises(ValueError, match="identity"):
        restored.resume(path.rsplit("\\", 1)[0], attempt_id="fingerprint-attempt",
                        actor_signature=_ACTOR, dataset_hash="d" * 64, config_hash=_CONFIG)
    assert restored._searcher is None
    assert restored._failed is True
    with pytest.raises(RuntimeError, match="uninitialized"):
        restored.continue_evolution()

    changed_config = _provider("pgpe", [], seed=713)
    with pytest.raises(ValueError, match="configuration"):
        changed_config.resume(path.rsplit("\\", 1)[0], attempt_id="fingerprint-attempt",
                             actor_signature=_ACTOR, dataset_hash=_DATASET, config_hash=_CONFIG)


def test_batch_evaluation_budget_and_request_keys_resume_with_optimizer(tmp_path) -> None:
    import torch
    from types import SimpleNamespace
    from rles_research.ledger import ResearchLedger, ResearchLockError, SelectionLedger

    spec = ActorGenomeSpec(("feature",), (1,))
    ledger = ResearchLedger(tmp_path / "selection.sqlite")
    ledger.create("resume-protocol", "checkpoint-run", validation_budget=1,
                  hpo_budget=0, es_evaluation_budget=16)
    selection = SelectionLedger(ledger, "resume-protocol")

    reservation_spec = SimpleNamespace(
        run_id="checkpoint-run", attempt_id="checkpoint-attempt",
        stage_budget={"E1": SimpleNamespace(max_lane_requests=16)},
    )

    def dispatch(requests: tuple[CandidateRequest, ...]) -> tuple[CandidateResult, ...]:
        return tuple(CandidateResult(
            run_id=request.run_id, attempt_id=request.attempt_id,
            generation=request.generation, candidate_id=request.candidate_id,
            episode_id=request.episode_id, seed=request.seed,
            request_hash=request.request_hash,
            account_isolation_id=request.account_isolation_id,
            fitness=sum(request.genome),
            evidence_hash=sha256(request.request_hash.encode()).hexdigest(),
        ) for request in requests)

    dispatch_id = sha256(b"checkpoint-budget-dispatch-v1").hexdigest()

    def create_pair(durable_selection=None):
        if durable_selection is None:
            budget = EvaluationBudget(16)
        else:
            budget = EvaluationBudget(
                16,
                reserve_requests=lambda requests: durable_selection.reserve_lane_requests(
                    reservation_spec, stage="E1", requests=requests,
                ),
                verify_requests=lambda keys: durable_selection.verify_lane_requests(
                    reservation_spec, stage="E1", request_keys=keys,
                ),
                reservation_id=durable_selection.reservation_scope(
                    reservation_spec, stage="E1",
                ),
            )
        evaluator = GenerationLaneEvaluator(
            run_id="checkpoint-run", attempt_id="checkpoint-attempt", fold_id="fold-0",
            root_seed=912, episode_ids=("episode-a", "episode-b"), candidate_spec=spec,
            dispatch=dispatch, budget=budget, dispatch_id=dispatch_id,
        )
        scalar_fitness = lambda _candidate, *, split: (_ for _ in ()).throw(AssertionError(split))
        provider = EvoTorchProvider(
            EvolutionConfig("pgpe", 4, 2, 691, max_evaluations=8), scalar_fitness,
            batch_fitness=evaluator, batch_fitness_id=evaluator.signature,
        )
        return provider, evaluator, budget

    def seed_globals() -> None:
        random.seed(381)
        np.random.seed(381)
        torch.manual_seed(381)

    seed_globals()
    continuous, continuous_evaluator, continuous_budget = create_pair()
    continuous_result = continuous.evolve(dimension=spec.size)
    assert continuous_budget.consumed == 16

    seed_globals()
    interrupted, interrupted_evaluator, interrupted_budget = create_pair(selection)
    partial = interrupted.evolve(dimension=spec.size, generations=1)
    assert partial.evaluations == 4 and interrupted_budget.consumed == 8
    assert ledger.evidence_summary("resume-protocol")["es_lane_request_count"] == 8
    assert interrupted_evaluator.evidence_hash
    path = interrupted.save(str(tmp_path / "batch-checkpoint"), attempt_id="checkpoint-attempt",
                            actor_signature=spec.signature, dataset_hash=_DATASET, config_hash=_CONFIG)

    # Same frozen scope in a fresh but empty ledger must fail before optimization resumes.
    empty_ledger = ResearchLedger(tmp_path / "empty-selection.sqlite")
    empty_ledger.create("resume-protocol", "checkpoint-run", validation_budget=1,
                        hpo_budget=0, es_evaluation_budget=16)
    empty_selection = SelectionLedger(empty_ledger, "resume-protocol")
    invalid_restore, _, _ = create_pair(empty_selection)
    with pytest.raises(ResearchLockError, match="differs from durable"):
        invalid_restore.resume(
            path.rsplit("\\", 1)[0], attempt_id="checkpoint-attempt",
            actor_signature=spec.signature, dataset_hash=_DATASET, config_hash=_CONFIG,
        )

    seed_globals()
    restored, restored_evaluator, restored_budget = create_pair(selection)
    restored.resume(path.rsplit("\\", 1)[0], attempt_id="checkpoint-attempt",
                    actor_signature=spec.signature, dataset_hash=_DATASET, config_hash=_CONFIG)
    assert restored_budget.consumed == 8
    assert restored_evaluator.evidence_hash == interrupted_evaluator.evidence_hash
    duplicate_population = np.zeros((4, spec.size), dtype=np.float32)
    with pytest.raises(RuntimeError, match="already dispatched"):
        restored_evaluator(duplicate_population, generation=0)
    assert restored_budget.consumed == 8
    result = restored.continue_evolution()
    assert result == continuous_result
    assert restored_budget.consumed == 16
    assert ledger.evidence_summary("resume-protocol")["es_lane_request_count"] == 16
    assert restored_evaluator.signature == interrupted_evaluator.signature
    assert restored_evaluator.evidence_hash == continuous_evaluator.evidence_hash
