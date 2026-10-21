"""Actual spawn/IPC tests with synthetic sessions, not native-engine goldens."""
from __future__ import annotations

import os
from pathlib import Path
import threading
import time
from uuid import uuid4
from hashlib import sha256

import numpy as np
import pytest

from rles_compute_runtime.errors import ComputeProtocolError, DeviceUnavailable, WorkerFailure
from rles_compute_runtime.lanes import ESEvalLane, RLTrainLane
from rles_compute_runtime.resources import ResourceQuota
from rles_compute_runtime.workers import CPUWorkerPool
from rles_es_evotorch.actor import ActorGenomeSpec
from rles_es_evotorch.evaluation import (
    CandidateRequest, CandidateResult, EvaluationBudget, GenerationLaneEvaluator,
    WorkerLaneDispatcher, dispatch_generation, scenario_seed,
)
from rles_es_evotorch.fitness import FitnessSpec
from rles_es_evotorch.provider import EvoTorchProvider, EvolutionConfig
from rles_es_evotorch.soft_risk import SoftRiskDimension, SoftRiskSpace

pytestmark = pytest.mark.micro_golden


class SyntheticSession:
    def __init__(self, config: dict):
        self.worker = config["worker_index"]
        self.session_id = uuid4().hex
        self.counter = 0

    def reset(self, *, seed: int) -> dict:
        self.counter = 0
        return {"pid": os.getpid(), "worker": self.worker, "seed": seed,
                "cuda_visible": os.environ.get("CUDA_VISIBLE_DEVICES")}

    def step(self, *, action: int = 0, fail: bool = False, sleep_s: float = 0) -> dict:
        if fail:
            raise RuntimeError("synthetic worker failure")
        if sleep_s:
            time.sleep(sleep_s)
        self.counter += action
        return {"counter": self.counter, "worker": self.worker}

    def evaluate(self, *, value: int, genome: list[float] | None = None,
                 rles_context: dict | None = None, fail: bool = False) -> dict:
        if fail:
            raise RuntimeError("synthetic worker evaluation failure")
        score = float(value) + sum(float(item) for item in (genome or []))
        evidence = sha256(f"{self.session_id}:{score}:{genome}".encode()).hexdigest()
        output = {"value": score, "fitness": score, "evidence_hash": evidence,
                  "session_id": self.session_id, "pid": os.getpid()}
        if rles_context is not None:
            initial_equity = 100_000.0
            final_equity = initial_equity * (1 + score / 10_000)
            output["evidence_hash"] = sha256(
                f"{evidence}:{rles_context['request_hash']}:{rles_context['account_isolation_id']}".encode()
            ).hexdigest()
            output["request_hash"] = rles_context["request_hash"]
            output["account_isolation_id"] = rles_context["account_isolation_id"]
            output.update({
                "initial_equity": initial_equity, "final_equity": final_equity,
                "step_log_returns": [score / 10_000], "drawdowns": [0.0],
                "risk_intervention_count": 0, "risk_intervention_steps": 1,
                "risk_intervention_rate": 0.0,
                "insolvency": False,
                "margin_facts_known": True,
                "maintenance_margin_breach": False,
                "native_liquidation_detected": False,
            })
        return output

    def close(self) -> None:
        pass


def make_session(config: dict) -> SyntheticSession:
    return SyntheticSession(config)


def _pool(tmp_path: Path, mode: str = "rl", **overrides) -> tuple[CPUWorkerPool, ResourceQuota]:
    quota = ResourceQuota(tmp_path / "quota.sqlite", cpu_cores=2, ram_bytes=4096, gpus=0)
    settings = dict(workers=2, factory_ref=f"{__name__}:make_session", factory_config={},
                    mode=mode, quota=quota, lease_owner="attempt-1", ram_bytes_per_worker=1024,
                    cpu_cores_per_worker=1, startup_timeout_s=15, torch_threads=1)
    settings.update(overrides)
    return CPUWorkerPool(**settings), quota


def _key(worker: int, *, step: int = 0, candidate: str = "none") -> dict:
    return {"run_id": "run-1", "attempt_id": "attempt-1", "env_id": worker,
            "episode_id": 1, "step_id": step, "candidate_id": candidate,
            "model_id": "model-1", "schema_version": "0.4.0", "dtype": "float32", "shape": [1]}


@pytest.mark.parametrize(("field", "value"), [
    ("workers", True), ("workers", 1.5), ("torch_threads", True),
    ("ram_bytes_per_worker", 1.5), ("cpu_cores_per_worker", True),
    ("enforce_memory_limit", "true"),
    ("startup_timeout_s", float("nan")), ("startup_timeout_s", float("inf")),
    ("startup_timeout_s", threading.TIMEOUT_MAX * 2),
])
def test_pool_rejects_invalid_resources_before_acquiring_quota(tmp_path: Path, field: str, value: object) -> None:
    with pytest.raises(ValueError):
        _pool(tmp_path, **{field: value})
    quota = ResourceQuota(tmp_path / "quota.sqlite", cpu_cores=2, ram_bytes=4096, gpus=0)
    assert quota.active() == []


def test_required_windows_limit_never_silently_downgrades_on_other_platform(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import rles_compute_runtime.workers as worker_module

    monkeypatch.setattr(worker_module, "os", SimpleNamespace(name="unsupported"))
    with pytest.raises(DeviceUnavailable, match="unsupported platform"):
        _pool(tmp_path, enforce_memory_limit=True)
    quota = ResourceQuota(tmp_path / "quota.sqlite", cpu_cores=2, ram_bytes=4096, gpus=0)
    assert quota.active() == []


def test_spawn_rl_workers_keep_state_and_barrier_identity(tmp_path: Path) -> None:
    pool, quota = _pool(tmp_path)
    with pool:
        assert len(quota.active()) == 1
        for timeout in (float("nan"), float("inf"), True, -1, 10 ** 1000):
            with pytest.raises(WorkerFailure, match="invalid timeout"):
                pool.dispatch([], timeout_s=timeout)
        lane = RLTrainLane(pool, timeout_s=5)
        reset = lane.barrier([_key(0), _key(1)], [{"seed": 1}, {"seed": 2}], operation="reset")
        assert len({result["pid"] for result in reset}) == 2
        assert all(result["pid"] != os.getpid() and result["cuda_visible"] == "-1" for result in reset)
        first = lane.barrier([_key(0), _key(1)], [{"action": 2}, {"action": 5}])
        assert [result["counter"] for result in first] == [2, 5]
        with pytest.raises(ComputeProtocolError, match="noncontiguous step"):
            lane.barrier([_key(0), _key(1)], [{"action": 99}, {"action": 99}])
        second = lane.barrier([_key(0, step=1), _key(1, step=1)], [{"action": 3}, {"action": 4}])
        assert [result["counter"] for result in second] == [5, 9]
    assert quota.active() == []
    assert all(not process.is_alive() for process in pool._processes)


def test_es_candidates_receive_fresh_sessions_and_attributed_results(tmp_path: Path) -> None:
    pool, quota = _pool(tmp_path, "es")
    with pool:
        lane = ESEvalLane(pool, timeout_s=5)
        result = lane.evaluate([(_key(0, candidate=f"c-{i}"), {"value": i}) for i in range(5)])
        assert {key: value["value"] for key, value in result.items()} == {f"c-{i}": i for i in range(5)}
        assert len({value["session_id"] for value in result.values()}) == 5
    assert quota.active() == []


def test_es_receives_completed_result_when_process_status_precedes_queue_delivery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    pool, quota = _pool(tmp_path, "es", workers=1)
    with pool:
        process = pool._processes[0]
        alive_checks = 0
        original_is_alive = process.is_alive

        def stale_process_status() -> bool:
            nonlocal alive_checks
            alive_checks += 1
            return original_is_alive() if alive_checks == 1 else False

        with monkeypatch.context() as patch:
            patch.setattr(process, "is_alive", stale_process_status)
            patch.setattr(pool._outboxes[0], "empty", lambda: True)
            result = ESEvalLane(pool, timeout_s=5).evaluate(
                [(_key(0, candidate="completed"), {"value": 7})],
            )
        assert result["completed"]["value"] == 7
    assert quota.active() == []


@pytest.mark.parametrize("candidate_kind", ["actor", "soft_risk"])
def test_l5_candidate_matrix_runs_through_l6_es_lane(tmp_path: Path, candidate_kind: str) -> None:
    """Exercise E1 Actor and E2 SoftRisk request paths over real L6 spawn/IPC scheduling."""
    if candidate_kind == "actor":
        candidate_spec = ActorGenomeSpec(("price_return",), (2,))
        actor_signature = candidate_spec.signature
    else:
        candidate_spec = SoftRiskSpace((
            SoftRiskDimension("exposure_multiplier", 0.1, 1.0),
            SoftRiskDimension("drawdown_throttle", 0.2, 1.0),
        ))
        actor_signature = "d" * 64  # frozen R1 actor; the searched space is SoftRisk-only
    candidate_schema_hash = candidate_spec.signature
    candidate_size = candidate_spec.size if isinstance(candidate_spec, ActorGenomeSpec) else len(candidate_spec.dimensions)
    candidates = {"candidate-a": ((0.5,) * candidate_size if candidate_kind == "soft_risk"
                                   else (0.0,) * candidate_size),
                  "candidate-b": ((0.75,) * candidate_size if candidate_kind == "soft_risk"
                                   else (0.25,) * candidate_size)}
    episodes = ("episode-0", "episode-1")
    requests = tuple(
        CandidateRequest(
            run_id="hybrid-run", attempt_id="attempt-1", generation=3,
            candidate_id=candidate_id, episode_id=episode_id,
            seed=scenario_seed(77, "fold-2", 3, episode_id), genome=genome,
            actor_signature=actor_signature, candidate_kind=candidate_kind,
            candidate_schema_hash=candidate_schema_hash,
            account_isolation_id=f"account-{candidate_id}-{episode_id}",
        )
        for candidate_id, genome in candidates.items()
        for episode_id in episodes
    )
    pool, quota = _pool(tmp_path, "es")
    with pool:
        lane = ESEvalLane(pool, timeout_s=5)

        def dispatch(batch: tuple[CandidateRequest, ...]) -> tuple[CandidateResult, ...]:
            lane_requests = []
            lane_ids = {}
            for request in batch:
                lane_id = f"{request.candidate_id}|{request.episode_id}"
                lane_ids[lane_id] = request
                candidate_number = 1 if request.candidate_id == "candidate-a" else 2
                episode_number = int(request.episode_id.rsplit("-", 1)[1])
                key = {
                    "run_id": request.run_id,
                    "attempt_id": request.attempt_id,
                    "env_id": 0,
                    "episode_id": request.episode_id,
                    "step_id": request.generation,
                    "candidate_id": lane_id,
                    "model_id": request.actor_signature,
                    "schema_version": request.candidate_schema_hash,
                    "dtype": "float32",
                    "shape": [len(request.genome)],
                }
                lane_requests.append((key, {
                    "value": candidate_number + episode_number,
                    "genome": list(request.genome),
                }))
            outputs = lane.evaluate(lane_requests)
            return tuple(
                CandidateResult(
                    run_id=request.run_id, attempt_id=request.attempt_id,
                    generation=request.generation, candidate_id=request.candidate_id,
                    episode_id=request.episode_id, seed=request.seed,
                    request_hash=request.request_hash,
                    account_isolation_id=request.account_isolation_id,
                    fitness=float(outputs[lane_id]["value"]),
                    evidence_hash=sha256(
                        f"{outputs[lane_id]['pid']}:{outputs[lane_id]['session_id']}".encode()
                    ).hexdigest(),
                )
                for lane_id, request in lane_ids.items()
            )

        aggregates = dispatch_generation(
            requests, dispatch=dispatch, budget=EvaluationBudget(4),
            candidate_spec=candidate_spec,
            actor_signature=actor_signature if candidate_kind == "soft_risk" else None,
        )
        assert [(x.candidate_id, x.episodes, x.mean_fitness) for x in aggregates] == [
            ("candidate-a", 2, 1.5 + (0.5 if candidate_kind == "soft_risk" else 0.0) * candidate_size),
            ("candidate-b", 2, 2.5 + (0.75 if candidate_kind == "soft_risk" else 0.25) * candidate_size),
        ]
        assert all(x.candidate_kind == candidate_kind
                   and x.candidate_schema_hash == candidate_schema_hash for x in aggregates)
    assert quota.active() == []


@pytest.mark.artifact_restore
def test_optimizer_resume_restarts_spawn_lane_and_reconciles_l6_reservations(tmp_path: Path) -> None:
    """Resume PGPE through a new OS worker pool and the same durable L6 ledger."""
    import random
    from types import SimpleNamespace

    import torch

    from rles_research.ledger import ResearchLedger, ResearchLockError, SelectionLedger

    actor_spec = ActorGenomeSpec(("price_return",), (2,))
    ledger = ResearchLedger(tmp_path / "selection.sqlite")
    ledger.create("spawn-resume", "spawn-resume-run", validation_budget=1,
                  hpo_budget=0, es_evaluation_budget=16)
    selection = SelectionLedger(ledger, "spawn-resume")
    frozen = SimpleNamespace(
        run_id="spawn-resume-run", attempt_id="spawn-resume-attempt",
        stage_budget={"E1": SimpleNamespace(max_lane_requests=16)},
    )
    fitness = FitnessSpec()
    builder_id = sha256(b"spawn-resume-payload-v1").hexdigest()

    def make_provider(lane: ESEvalLane, *, durable: bool):
        dispatcher = WorkerLaneDispatcher(
            lane, lambda request: {"value": sum(request.genome),
                                   "genome": list(request.genome)},
            payload_builder_id=builder_id, fitness_spec=fitness, tail_fraction=0.5,
        )
        if durable:
            budget = EvaluationBudget(
                16,
                reserve_requests=lambda requests: selection.reserve_lane_requests(
                    frozen, stage="E1", requests=requests,
                ),
                verify_requests=lambda keys: selection.verify_lane_requests(
                    frozen, stage="E1", request_keys=keys,
                ),
                reservation_id=selection.reservation_scope(frozen, stage="E1"),
            )
        else:
            budget = EvaluationBudget(16)
        evaluator = GenerationLaneEvaluator(
            run_id=frozen.run_id, attempt_id=frozen.attempt_id, fold_id="fold-spawn",
            root_seed=371, episode_ids=("episode-a", "episode-b"),
            candidate_spec=actor_spec, dispatch=dispatcher, budget=budget,
            dispatch_id=dispatcher.signature,
        )
        scalar = lambda _candidate, *, split: (_ for _ in ()).throw(AssertionError(split))
        provider = EvoTorchProvider(
            EvolutionConfig("pgpe", 4, 2, 719, max_evaluations=8), scalar,
            batch_fitness=evaluator, batch_fitness_id=evaluator.signature,
        )
        return provider, evaluator, budget

    def seed_globals() -> None:
        random.seed(719)
        np.random.seed(719)
        torch.manual_seed(719)

    seed_globals()
    continuous_pool, continuous_quota = _pool(tmp_path, "es", lease_owner="continuous")
    with continuous_pool:
        continuous, _, continuous_budget = make_provider(ESEvalLane(continuous_pool, timeout_s=10),
                                                          durable=False)
        expected = continuous.evolve(dimension=actor_spec.size)
        assert continuous_budget.consumed == 16
    assert continuous_quota.active() == []

    seed_globals()
    interrupted_pool, interrupted_quota = _pool(tmp_path, "es", lease_owner="interrupted")
    with interrupted_pool:
        interrupted, _, budget = make_provider(ESEvalLane(interrupted_pool, timeout_s=10),
                                                durable=True)
        partial = interrupted.evolve(dimension=actor_spec.size, generations=1)
        assert partial.evaluations == 4 and budget.consumed == 8
    assert interrupted_quota.active() == []
    checkpoint = interrupted.save(
        str(tmp_path / "spawn-resume-checkpoint"), attempt_id=frozen.attempt_id,
        actor_signature=actor_spec.signature, dataset_hash="b" * 64, config_hash="c" * 64,
    )
    assert ledger.evidence_summary("spawn-resume")["es_lane_request_count"] == 8

    # This new OS worker pool has no matching durable request ledger. Restore
    # must fail during preflight, before a new candidate reaches a worker.
    empty_ledger = ResearchLedger(tmp_path / "empty-selection.sqlite")
    empty_ledger.create("spawn-resume", "spawn-resume-run", validation_budget=1,
                        hpo_budget=0, es_evaluation_budget=16)
    empty_selection = SelectionLedger(empty_ledger, "spawn-resume")
    wrong_pool, _ = _pool(tmp_path, "es", lease_owner="wrong-ledger")
    wrong_dispatcher = WorkerLaneDispatcher(
        ESEvalLane(wrong_pool, timeout_s=10),
        lambda request: {"value": sum(request.genome), "genome": list(request.genome)},
        payload_builder_id=builder_id, fitness_spec=fitness, tail_fraction=0.5,
    )
    wrong_budget = EvaluationBudget(
        16,
        reserve_requests=lambda requests: empty_selection.reserve_lane_requests(
            frozen, stage="E1", requests=requests,
        ),
        verify_requests=lambda keys: empty_selection.verify_lane_requests(
            frozen, stage="E1", request_keys=keys,
        ),
        reservation_id=empty_selection.reservation_scope(frozen, stage="E1"),
    )
    wrong_evaluator = GenerationLaneEvaluator(
        run_id=frozen.run_id, attempt_id=frozen.attempt_id, fold_id="fold-spawn",
        root_seed=371, episode_ids=("episode-a", "episode-b"),
        candidate_spec=actor_spec, dispatch=wrong_dispatcher, budget=wrong_budget,
        dispatch_id=wrong_dispatcher.signature,
    )
    wrong_provider = EvoTorchProvider(
        EvolutionConfig("pgpe", 4, 2, 719, max_evaluations=8),
        lambda _candidate, *, split: (_ for _ in ()).throw(AssertionError(split)),
        batch_fitness=wrong_evaluator, batch_fitness_id=wrong_evaluator.signature,
    )
    with pytest.raises(ResearchLockError, match="differs from durable"):
        wrong_provider.resume(
            str(Path(checkpoint).parent), attempt_id=frozen.attempt_id,
            actor_signature=actor_spec.signature, dataset_hash="b" * 64, config_hash="c" * 64,
        )
    wrong_pool.close()

    # A worker recipe change must invalidate the optimizer checkpoint even
    # when the payload builder and durable reservation ledger are unchanged.
    changed_pool, _ = _pool(
        tmp_path, "es", lease_owner="changed-worker-recipe",
        factory_config={"worker_recipe": "changed"},
    )
    changed, _, _ = make_provider(ESEvalLane(changed_pool, timeout_s=10), durable=True)
    with pytest.raises(ValueError, match="checkpoint optimizer configuration mismatch"):
        changed.resume(
            str(Path(checkpoint).parent), attempt_id=frozen.attempt_id,
            actor_signature=actor_spec.signature, dataset_hash="b" * 64, config_hash="c" * 64,
        )
    changed_pool.close()

    resumed_pool, resumed_quota = _pool(tmp_path, "es", lease_owner="resumed")
    with resumed_pool:
        resumed, evaluator, restored_budget = make_provider(
            ESEvalLane(resumed_pool, timeout_s=10), durable=True,
        )
        resumed.resume(
            str(Path(checkpoint).parent), attempt_id=frozen.attempt_id,
            actor_signature=actor_spec.signature, dataset_hash="b" * 64, config_hash="c" * 64,
        )
        assert restored_budget.consumed == 8
        actual = resumed.continue_evolution()
        assert actual == expected
        assert restored_budget.consumed == 16
        assert evaluator.evidence_hash
    assert resumed_quota.active() == []
    assert ledger.evidence_summary("spawn-resume")["es_lane_request_count"] == 16


def test_spawn_worker_failure_keeps_l6_es_reservations_and_blocks_new_attempt(tmp_path: Path) -> None:
    from types import SimpleNamespace

    from rles_research.ledger import ResearchLedger, ResearchLockError, SelectionLedger

    candidate_spec = ActorGenomeSpec(("feature",), (1,))
    ledger = ResearchLedger(tmp_path / "failed-worker-ledger.sqlite")
    ledger.create("failed-worker", "failed-worker-run", validation_budget=1,
                  hpo_budget=0, es_evaluation_budget=4)
    selection = SelectionLedger(ledger, "failed-worker")

    def make_requests(attempt_id: str) -> tuple[CandidateRequest, ...]:
        return tuple(CandidateRequest(
            run_id="failed-worker-run", attempt_id=attempt_id, generation=0,
            candidate_id=candidate, episode_id=episode, seed=29,
            genome=tuple((0.1 if candidate == "candidate-fails" else 0.2)
                         for _ in range(candidate_spec.size)),
            actor_signature=candidate_spec.signature, candidate_kind="actor",
            candidate_schema_hash=candidate_spec.signature,
            account_isolation_id=f"{attempt_id}-{candidate}-{episode}",
        ) for candidate in ("candidate-fails", "candidate-ok")
          for episode in ("episode-a", "episode-b"))

    def make_budget(attempt_id: str) -> EvaluationBudget:
        frozen = SimpleNamespace(
            run_id="failed-worker-run", attempt_id=attempt_id,
            stage_budget={"E1": SimpleNamespace(max_lane_requests=4)},
        )
        return EvaluationBudget(
            4,
            reserve_requests=lambda batch: selection.reserve_lane_requests(
                frozen, stage="E1", requests=batch,
            ),
            verify_requests=lambda keys: selection.verify_lane_requests(
                frozen, stage="E1", request_keys=keys,
            ),
            reservation_id=selection.reservation_scope(frozen, stage="E1"),
        )

    pool, quota = _pool(tmp_path, "es", workers=2, lease_owner="failed-worker")
    budget = make_budget("failed-attempt")
    dispatcher = WorkerLaneDispatcher(
        ESEvalLane(pool, timeout_s=5),
        lambda request: {
            "value": sum(request.genome), "genome": list(request.genome),
            "fail": request.candidate_id == "candidate-fails",
        },
        payload_builder_id=sha256(b"worker-failure-payload-v1").hexdigest(),
        fitness_spec=FitnessSpec(), tail_fraction=0.5,
    )
    with pytest.raises((WorkerFailure, ComputeProtocolError)):
        with pool:
            dispatch_generation(
                make_requests("failed-attempt"), dispatch=dispatcher, budget=budget,
                candidate_spec=candidate_spec,
            )
    assert budget.consumed == 4
    assert quota.active() == []
    assert not any(process.is_alive() for process in pool._processes)
    assert ledger.evidence_summary("failed-worker")["es_lane_request_count"] == 4

    retry_budget = make_budget("retry-attempt")
    with pytest.raises(ResearchLockError, match="evaluation budget exhausted"):
        retry_budget.reserve(make_requests("retry-attempt"))
    assert retry_budget.consumed == 0
    assert ledger.evidence_summary("failed-worker")["es_lane_request_count"] == 4


def test_e2_candidate_schema_cannot_replace_frozen_r1_actor() -> None:
    from dataclasses import replace

    risk_space = SoftRiskSpace((SoftRiskDimension("exposure_multiplier", 0.1, 1.0),))
    request = CandidateRequest(
        run_id="run-e2", attempt_id="attempt-e2", generation=0,
        candidate_id="risk-candidate", episode_id="scenario-0", seed=5,
        genome=(0.5,), actor_signature="e" * 64, candidate_kind="soft_risk",
        candidate_schema_hash=risk_space.signature, account_isolation_id="isolated-account",
    )
    invoked = False

    def dispatch(_batch):
        nonlocal invoked
        invoked = True
        return ()

    with pytest.raises(ValueError, match="frozen actor/risk schema"):
        dispatch_generation(
            (request,), dispatch=dispatch, budget=EvaluationBudget(1),
            candidate_spec=risk_space, actor_signature="d" * 64,
        )
    assert not invoked
    out_of_range = replace(request, genome=(0.0,))
    with pytest.raises(ValueError, match="exceeds hard-allowed search domain"):
        dispatch_generation(
            (out_of_range,), dispatch=dispatch, budget=EvaluationBudget(1),
            candidate_spec=risk_space, actor_signature="e" * 64,
        )
    assert not invoked
    budget = EvaluationBudget(2)
    assert budget.reserve((request,)) == 1
    changed_content_same_identity = replace(request, genome=(0.75,))
    with pytest.raises(RuntimeError, match="already dispatched"):
        budget.reserve((changed_content_same_identity,))
    assert budget.consumed == 1


def test_worker_dispatcher_transmits_and_reconciles_account_context() -> None:
    spec = ActorGenomeSpec(("price_return",), (1,))
    request = CandidateRequest(
        run_id="context-run", attempt_id="context-attempt", generation=2,
        candidate_id="context-candidate", episode_id="context-episode", seed=73,
        genome=(0.25,) * spec.size, actor_signature=spec.signature,
        candidate_kind="actor", candidate_schema_hash=spec.signature,
        account_isolation_id="unique-account-context",
    )

    class EchoLane:
        runtime_signature = "f" * 64

        def __init__(self, *, wrong_account: bool, final_equity: float = 101.0):
            self.wrong_account = wrong_account
            self.final_equity = final_equity
            self.payload = None

        def evaluate(self, candidates):
            key, self.payload = candidates[0]
            context = self.payload["rles_context"]
            return {key["candidate_id"]: {
                "initial_equity": 100.0, "final_equity": self.final_equity,
                "step_log_returns": ([0.01] if self.final_equity > 0 else [-1.0]),
                "drawdowns": ([0.0] if self.final_equity > 0 else [1.0]),
                "risk_intervention_count": 0, "risk_intervention_steps": 1,
                "risk_intervention_rate": 0.0, "insolvency": self.final_equity <= 0,
                "margin_facts_known": True, "maintenance_margin_breach": False,
                "native_liquidation_detected": False,
                "evidence_hash": "a" * 64,
                "request_hash": context["request_hash"],
                "account_isolation_id": ("other-account" if self.wrong_account
                                          else context["account_isolation_id"]),
            }}

    payload_id = sha256(b"echo-payload-v1").hexdigest()
    lane = EchoLane(wrong_account=False)
    result = WorkerLaneDispatcher(lane, lambda _request: {"native_arg": 1},
                                  payload_builder_id=payload_id, fitness_spec=FitnessSpec(),
                                  tail_fraction=0.1)((request,))
    context = lane.payload["rles_context"]
    assert context["seed"] == request.seed
    assert context["genome"] == list(request.genome)
    assert context["account_isolation_id"] == request.account_isolation_id
    assert result[0].request_hash == request.request_hash

    from rles_es_evotorch.fitness import (ConstraintPriorityFitnessSpec,
                                          FeasibilityFirstFitnessSpec)

    insolvent_lane = EchoLane(wrong_account=False, final_equity=0.0)
    constraint_dispatcher = WorkerLaneDispatcher(
        insolvent_lane, lambda _request: {"native_arg": 1},
        payload_builder_id=payload_id, fitness_spec=ConstraintPriorityFitnessSpec(), tail_fraction=0.1,
    )
    rejected = constraint_dispatcher((request,))[0]
    assert rejected.fitness == -2.0
    assert rejected.feasible is False
    assert rejected.rejection_reason == "episode_insolvency_detected"
    assert rejected.evidence_hash != "a" * 64
    v2_dispatcher = WorkerLaneDispatcher(
        EchoLane(wrong_account=False, final_equity=0.0), lambda _request: {"native_arg": 1},
        payload_builder_id=payload_id, fitness_spec=FeasibilityFirstFitnessSpec(), tail_fraction=0.1,
    )
    assert v2_dispatcher.signature != constraint_dispatcher.signature
    assert v2_dispatcher((request,))[0].rejection_reason == "episode_insolvency_detected"

    bad_lane = EchoLane(wrong_account=True)
    bad_dispatcher = WorkerLaneDispatcher(bad_lane, lambda _request: {"native_arg": 1},
                                          payload_builder_id=payload_id, fitness_spec=FitnessSpec(),
                                          tail_fraction=0.1)
    with pytest.raises(ValueError, match="does not echo the dispatched request"):
        bad_dispatcher((request,))

    drift_lane = EchoLane(wrong_account=False)
    drift_dispatcher = WorkerLaneDispatcher(
        drift_lane, lambda _request: {"native_arg": 1},
        payload_builder_id=payload_id, fitness_spec=FitnessSpec(), tail_fraction=0.1,
    )
    drift_lane.runtime_signature = "e" * 64
    with pytest.raises(RuntimeError, match="lane recipe changed"):
        drift_dispatcher((request,))


def test_e1_generation_keeps_genome_schema_separate_from_r1_actor_signature() -> None:
    spec = ActorGenomeSpec(("price_return",), (1,))
    r1_actor_signature = sha256(b"sb3-td3-native-actor-mapping-v1").hexdigest()
    requests_seen: list[CandidateRequest] = []

    def dispatch(requests: tuple[CandidateRequest, ...]) -> tuple[CandidateResult, ...]:
        requests_seen.extend(requests)
        return tuple(CandidateResult(
            request.run_id, request.attempt_id, request.generation,
            request.candidate_id, request.episode_id, request.seed,
            request.request_hash, request.account_isolation_id, 1.0, "a" * 64,
        ) for request in requests)

    evaluator = GenerationLaneEvaluator(
        run_id="e1-run", attempt_id="e1-attempt", fold_id="fold-0", root_seed=19,
        episode_ids=("train-inner-0",), candidate_spec=spec,
        dispatch=dispatch, budget=EvaluationBudget(4),
        dispatch_id=sha256(b"e1-dispatch-v1").hexdigest(),
        actor_signature=r1_actor_signature,
    )
    scores = evaluator(np.zeros((4, spec.size), dtype=np.float32), generation=0)

    assert scores.shape == (4,)
    assert np.array_equal(scores, np.ones(4, dtype=np.float32))
    assert len(requests_seen) == 4
    assert {request.actor_signature for request in requests_seen} == {r1_actor_signature}
    assert {request.candidate_schema_hash for request in requests_seen} == {spec.signature}
    assert evaluator.signature != spec.signature


def test_e2_neutral_baseline_is_paired_budgeted_and_checkpointed() -> None:
    spec = SoftRiskSpace((
        SoftRiskDimension("exposure_multiplier", 0.1, 1.0),
        SoftRiskDimension("drawdown_throttle", 0.2, 1.0),
    ))
    requests_seen: list[CandidateRequest] = []

    def dispatch(requests: tuple[CandidateRequest, ...]) -> tuple[CandidateResult, ...]:
        requests_seen.extend(requests)
        return tuple(CandidateResult(
            request.run_id, request.attempt_id, request.generation,
            request.candidate_id, request.episode_id, request.seed,
            request.request_hash, request.account_isolation_id,
            10.0 if request.genome == (1.0, 1.0) else 1.0, "b" * 64,
        ) for request in requests)

    def create_evaluator() -> tuple[GenerationLaneEvaluator, EvaluationBudget]:
        budget = EvaluationBudget(10)
        evaluator = GenerationLaneEvaluator(
            run_id="e2-neutral-run", attempt_id="e2-neutral-attempt", fold_id="train-inner",
            root_seed=19, episode_ids=("window-a", "window-b"), candidate_spec=spec,
            dispatch=dispatch, budget=budget, dispatch_id=sha256(b"e2-neutral-v1").hexdigest(),
            actor_signature="f" * 64, include_neutral_baseline=True,
        )
        return evaluator, budget

    evaluator, budget = create_evaluator()
    scores = evaluator(np.full((4, 2), 0.5, dtype=np.float32), generation=0)
    assert np.array_equal(scores, np.ones(4, dtype=np.float32))
    assert budget.consumed == 10
    assert len(requests_seen) == 10
    assert len({request.seed for request in requests_seen if request.episode_id == "window-a"}) == 1
    assert sum(request.genome == (1.0, 1.0) for request in requests_seen) == 2
    baseline = evaluator.best_neutral_baseline
    assert baseline is not None and baseline[:2] == (10.0, 0) and len(baseline[2]) == 64

    restored, restored_budget = create_evaluator()
    restored.restore_checkpoint_state(evaluator.checkpoint_state())
    assert restored_budget.consumed == 10
    assert restored.best_neutral_baseline == evaluator.best_neutral_baseline
    assert restored.evidence_hash == evaluator.evidence_hash


def test_constraint_rejections_are_reported_and_checkpointed_with_e2_neutral_protocol() -> None:
    spec = SoftRiskSpace((
        SoftRiskDimension("exposure_multiplier", 0.1, 1.0),
        SoftRiskDimension("drawdown_throttle", 0.2, 1.0),
    ))

    def dispatch(requests: tuple[CandidateRequest, ...]) -> tuple[CandidateResult, ...]:
        output = []
        for request in requests:
            insolvent = request.candidate_id.endswith("c00000000") and request.episode_id == "window-a"
            output.append(CandidateResult(
                request.run_id, request.attempt_id, request.generation,
                request.candidate_id, request.episode_id, request.seed,
                request.request_hash, request.account_isolation_id,
                -2.0 if insolvent else 0.5, "c" * 64,
                feasible=not insolvent,
                rejection_reason="episode_insolvency_detected" if insolvent else None,
            ))
        return tuple(output)

    def create_evaluator() -> tuple[GenerationLaneEvaluator, EvaluationBudget]:
        budget = EvaluationBudget(6)
        evaluator = GenerationLaneEvaluator(
            run_id="constraint-run", attempt_id="constraint-attempt", fold_id="train-inner",
            root_seed=19, episode_ids=("window-a", "window-b"), candidate_spec=spec,
            dispatch=dispatch, budget=budget, dispatch_id=sha256(b"constraint-v1").hexdigest(),
            actor_signature="f" * 64, include_neutral_baseline=True,
            include_constraint_evidence=True,
        )
        return evaluator, budget

    evaluator, budget = create_evaluator()
    scores = evaluator(np.full((2, 2), 0.5, dtype=np.float32), generation=0)
    assert scores[1] > scores[0]
    assert budget.consumed == 6
    evidence = evaluator.constraint_evidence
    first = evidence[0]["candidates"]
    assert isinstance(first, tuple)
    rejected = next(row for row in first if row["candidate_id"] == "g00000000-c00000000")
    assert isinstance(rejected["mean_fitness"], float)
    assert rejected["infeasible_episodes"] == 1
    assert rejected["rejection_reasons"] == ("episode_insolvency_detected",)

    restored, restored_budget = create_evaluator()
    restored.restore_checkpoint_state(evaluator.checkpoint_state())
    assert restored_budget.consumed == 6
    assert restored.constraint_evidence == evaluator.constraint_evidence


def test_v2_candidate_feasibility_is_lexicographic_across_scenarios() -> None:
    spec = SoftRiskSpace((
        SoftRiskDimension("exposure_multiplier", 0.1, 1.0),
        SoftRiskDimension("drawdown_throttle", 0.2, 1.0),
    ))

    def dispatch(requests: tuple[CandidateRequest, ...]) -> tuple[CandidateResult, ...]:
        results = []
        for request in requests:
            rejected = (request.candidate_id.endswith("c00000000")
                        and request.episode_id == "window-a")
            score = -2.0 if rejected else (0.9 if request.candidate_id.endswith("c00000000") else -0.5)
            results.append(CandidateResult(
                request.run_id, request.attempt_id, request.generation,
                request.candidate_id, request.episode_id, request.seed,
                request.request_hash, request.account_isolation_id, score, "e" * 64,
                feasible=not rejected,
                rejection_reason="native_liquidation_detected" if rejected else None,
            ))
        return tuple(results)

    evaluator = GenerationLaneEvaluator(
        run_id="constraint-v2", attempt_id="attempt", fold_id="train-inner",
        root_seed=7, episode_ids=("window-a", "window-b", "window-c", "window-d"),
        candidate_spec=spec, dispatch=dispatch, budget=EvaluationBudget(8),
        dispatch_id=sha256(b"constraint-v2").hexdigest(), actor_signature="f" * 64,
        include_constraint_evidence=True, feasibility_first=True,
    )
    scores = evaluator(np.full((2, 2), 0.5, dtype=np.float32), generation=0)
    assert scores[0] == -2.0
    assert scores[1] == pytest.approx(-0.5)
    rows = evaluator.constraint_evidence[0]["candidates"]
    rejected = next(row for row in rows if row["candidate_id"] == "g00000000-c00000000")
    assert rejected["mean_fitness"] == pytest.approx(0.175)
    assert rejected["selection_fitness"] == -2.0
    assert rejected["infeasible_episodes"] == 1


@pytest.mark.parametrize("feasible_fitness", (-2.5, 1.0))
def test_v2_dispatch_rejects_feasible_fitness_outside_strict_selection_bounds(
        feasible_fitness: float) -> None:
    candidate_spec = SoftRiskSpace((SoftRiskDimension("exposure_multiplier", 0.1, 1.0),))
    requests = tuple(
        CandidateRequest(
            run_id="bounded-selection", attempt_id="attempt", generation=0,
            candidate_id=candidate_id, episode_id="scenario-0", seed=11,
            genome=(0.5,), actor_signature="f" * 64, candidate_kind="soft_risk",
            candidate_schema_hash=candidate_spec.signature,
            account_isolation_id=f"account-{candidate_id}",
        )
        for candidate_id in ("feasible-candidate", "liquidated-candidate")
    )

    def dispatch(batch: tuple[CandidateRequest, ...]) -> tuple[CandidateResult, ...]:
        return tuple(
            CandidateResult(
                run_id=request.run_id, attempt_id=request.attempt_id,
                generation=request.generation, candidate_id=request.candidate_id,
                episode_id=request.episode_id, seed=request.seed,
                request_hash=request.request_hash,
                account_isolation_id=request.account_isolation_id,
                fitness=feasible_fitness if request.candidate_id == "feasible-candidate" else -2.0,
                evidence_hash="e" * 64,
                feasible=request.candidate_id == "feasible-candidate",
                rejection_reason=(None if request.candidate_id == "feasible-candidate"
                                  else "native_liquidation_detected"),
            )
            for request in batch
        )

    with pytest.raises(ValueError, match="strictly bounded in \\(-1, 1\\)"):
        dispatch_generation(
            requests, dispatch=dispatch, budget=EvaluationBudget(2),
            candidate_spec=candidate_spec, actor_signature="f" * 64,
            feasibility_first=True,
        )


@pytest.mark.parametrize("candidate_kind", ["actor", "soft_risk"])
def test_evotorch_generation_dispatches_full_matrix_through_spawn_lane(
        tmp_path: Path, candidate_kind: str) -> None:
    if candidate_kind == "actor":
        candidate_spec = ActorGenomeSpec(("price_return",), (2,))
        actor_signature = candidate_spec.signature
        algorithm = "pgpe"
    else:
        candidate_spec = SoftRiskSpace((
            SoftRiskDimension("exposure_multiplier", 0.1, 1.0),
            SoftRiskDimension("drawdown_throttle", 0.2, 1.0),
        ))
        actor_signature = "f" * 64
        algorithm = "cmaes"

    pool, quota = _pool(tmp_path, "es")
    with pool:
        lane = ESEvalLane(pool, timeout_s=10)
        dispatcher = WorkerLaneDispatcher(
            lane,
            lambda request: {"value": int(request.episode_id[-1]),
                             "genome": list(request.genome)},
            payload_builder_id=sha256(b"synthetic-evaluate-payload-v1").hexdigest(),
            fitness_spec=FitnessSpec(), tail_fraction=0.1,
        )
        budget = EvaluationBudget(16)
        generation_fitness = GenerationLaneEvaluator(
            run_id=f"integration-{candidate_kind}", attempt_id="attempt-1", fold_id="fold-1",
            root_seed=712, episode_ids=("episode-0", "episode-1"),
            candidate_spec=candidate_spec,
            actor_signature=actor_signature if candidate_kind == "soft_risk" else None,
            dispatch=dispatcher, budget=budget, dispatch_id=dispatcher.signature,
        )
        scalar_fitness = lambda _candidate, *, split: 0.0
        provider = EvoTorchProvider(
            EvolutionConfig(algorithm, 4, 2, 41, max_evaluations=8), scalar_fitness,
            risk_space=candidate_spec if isinstance(candidate_spec, SoftRiskSpace) else None,
            batch_fitness=generation_fitness,
            batch_fitness_id=generation_fitness.signature,
        )
        dimension = (candidate_spec.size if isinstance(candidate_spec, ActorGenomeSpec) else 2)
        result = provider.evolve(dimension=dimension)
        assert result.generations == 2
        assert result.evaluations == 8
        assert np.isfinite(result.best_fitness)
        assert budget.consumed == 16
    assert quota.active() == []


@pytest.mark.parametrize("payload", [{"fail": True}, {"sleep_s": 1}])
def test_worker_failure_or_timeout_closes_attempt_and_releases_quota(tmp_path: Path, payload: dict) -> None:
    pool, quota = _pool(tmp_path)
    pool.start()
    try:
        with pytest.raises((ComputeProtocolError, WorkerFailure)):
            pool.dispatch([(0, _key(0), "step", payload)], timeout_s=0.05 if "sleep_s" in payload else 5)
        assert quota.active() == []
        assert all(not process.is_alive() for process in pool._processes)
        with pytest.raises(WorkerFailure, match="pool not running"):
            pool.dispatch([(0, _key(0), "step", {})], timeout_s=1)
    finally:
        pool.close()


def test_same_thread_reentrant_dispatch_is_rejected(tmp_path: Path, monkeypatch) -> None:
    pool, _ = _pool(tmp_path)
    monkeypatch.setattr(pool, "_dispatch_barrier", lambda *args, **kwargs: pool.dispatch([], timeout_s=1))
    try:
        with pytest.raises(WorkerFailure, match="active dispatch"):
            pool.dispatch([], timeout_s=1)
        assert not pool._dispatching
    finally:
        pool.close()
