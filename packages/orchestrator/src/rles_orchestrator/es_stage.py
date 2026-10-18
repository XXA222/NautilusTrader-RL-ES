"""Installable E1/E2 composition of EvoTorch, L6 spawn lanes, and receipts.

This module belongs to the workflow boundary. The independent RL and ES
providers do not import each other or the NautilusTrader implementation.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from hashlib import sha256
import json
from pathlib import Path
import random
from dataclasses import replace

import numpy as np

from rles_compute_runtime.lanes import ESEvalLane
from rles_compute_runtime.resources import ResourceQuota
from rles_compute_runtime.workers import CPUWorkerPool
from rles_es_evotorch.actor import ActorGenomeSpec, genome_hash
from rles_es_evotorch.checkpoint import ESCheckpointStore
from rles_es_evotorch.evaluation import (
    CandidateRequest, EvaluationBudget, GenerationLaneEvaluator, WorkerLaneDispatcher,
)
from rles_es_evotorch.fitness import (ConstraintPriorityFitnessSpec,
                                      FeasibilityFirstFitnessSpec, FitnessSpec)
from rles_es_evotorch.provider import EvoTorchProvider, EvolutionConfig, EvolutionResult
from rles_es_evotorch.soft_risk import SoftRiskSpace
from rles_research.ledger import SelectionLedger

from .workflow import ProviderStageOutput, StageContext


def _digest(value: object) -> str:
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                             allow_nan=False).encode()).hexdigest()


def _select_e2_candidate(
    result: EvolutionResult, candidate_spec: SoftRiskSpace,
    baseline: tuple[float, int, str],
) -> EvolutionResult:
    """Compare the exact-neutral incumbent under the same generation protocol."""
    baseline_score, _generation, _evidence = baseline
    evaluations = result.evaluations + result.generations
    if baseline_score >= result.best_fitness:
        neutral = candidate_spec.decode(np.ones(len(candidate_spec.dimensions), dtype=np.float32))
        return replace(
            result, evaluations=evaluations, best_fitness=baseline_score,
            risk_parameters=tuple(sorted(neutral.items())),
            selected_candidate_kind="neutral_r1",
        )
    return replace(result, evaluations=evaluations)


class ESStageExecutor:
    """Run one frozen ES stage, checkpoint each generation, and return facts.

    Wrap this callable with ``ProviderBackedStageRunner``. That runner and the
    Orchestrator verify usage/lineage and persist the L6 stage receipt.
    """

    def __init__(
        self, *, stage: str, artifact_id: str,
        candidate_spec: ActorGenomeSpec | SoftRiskSpace,
        evolution: EvolutionConfig, selection: SelectionLedger,
        quota: ResourceQuota, factory_ref: str, factory_config: Mapping[str, object],
        payload_builder: Callable[[CandidateRequest], Mapping[str, object]],
        payload_builder_id: str,
        fitness_spec: FitnessSpec | ConstraintPriorityFitnessSpec, tail_fraction: float,
        fold_id: str, episode_ids: tuple[str, ...], checkpoint_dir: str | Path,
        workers: int, lease_owner: str, ram_bytes_per_worker: int,
        cpu_cores_per_worker: int = 1, startup_timeout_s: float = 30,
        timeout_s: float = 30, torch_threads: int = 1,
        initial_bounds: tuple[float, float] = (-0.1, 0.1),
        center: np.ndarray | None = None, resume: bool = False,
        enforce_memory_limit: bool = False,
        include_neutral_baseline: bool = False,
        include_constraint_evidence: bool = False,
    ) -> None:
        if (stage not in {"E1", "E2"} or not isinstance(artifact_id, str) or not artifact_id
                or not isinstance(candidate_spec, (ActorGenomeSpec, SoftRiskSpace))
                or not isinstance(evolution, EvolutionConfig)
                or not isinstance(selection, SelectionLedger)
                or not isinstance(quota, ResourceQuota)
                or not isinstance(factory_ref, str) or not factory_ref
                or not callable(payload_builder)
                or not isinstance(fitness_spec, (FitnessSpec, ConstraintPriorityFitnessSpec))
                or not isinstance(fold_id, str) or not fold_id
                or not isinstance(episode_ids, tuple) or not episode_ids
                or any(not isinstance(item, str) or not item for item in episode_ids)
                or len(set(episode_ids)) != len(episode_ids)
                or type(resume) is not bool or type(enforce_memory_limit) is not bool
                or type(include_neutral_baseline) is not bool
                or type(include_constraint_evidence) is not bool):
            raise ValueError("frozen E1/E2 stage plan and installed dependencies required")
        if evolution.device != "cpu":
            raise RuntimeError("native ES stage checkpoint currently requires CPU EvoTorch")
        # The L6 pool validates resource sizes and startup settings. Clone the
        # JSON recipe here so callers cannot mutate it between stages.
        recipe = json.loads(json.dumps(dict(factory_config), sort_keys=True,
                                       separators=(",", ":"), allow_nan=False))
        if not isinstance(recipe, dict):
            raise ValueError("worker factory configuration must be a JSON object")
        self.stage, self.artifact_id = stage, artifact_id
        self.candidate_spec, self.evolution = candidate_spec, evolution
        self.selection, self.quota = selection, quota
        self.factory_ref, self.factory_config = factory_ref, recipe
        self.payload_builder, self.payload_builder_id = payload_builder, payload_builder_id
        self.fitness_spec, self.tail_fraction = fitness_spec, tail_fraction
        self.fold_id, self.episode_ids = fold_id, episode_ids
        self.checkpoint_dir = Path(checkpoint_dir)
        self.workers, self.lease_owner = workers, lease_owner
        self.ram_bytes_per_worker = ram_bytes_per_worker
        self.cpu_cores_per_worker, self.startup_timeout_s = cpu_cores_per_worker, startup_timeout_s
        self.timeout_s, self.torch_threads = timeout_s, torch_threads
        self.initial_bounds = initial_bounds
        self.center = None if center is None else np.asarray(center).copy()
        self.resume = resume
        self.enforce_memory_limit = enforce_memory_limit
        self.include_neutral_baseline = include_neutral_baseline
        self.include_constraint_evidence = (
            include_constraint_evidence or isinstance(fitness_spec, ConstraintPriorityFitnessSpec)
        )
        self.worker_resource_evidence: list[dict] = []
        self._used = False

    def _checkpoint_root(self, generation: int) -> Path:
        return self.checkpoint_dir / f"generation-{generation:08d}"

    def _latest_checkpoint_root(self, attempt_id: str) -> Path:
        candidates = sorted(
            (path for path in self.checkpoint_dir.glob("generation-[0-9]*")
             if path.is_dir() and path.name[11:].isdigit()),
            reverse=True,
        )
        for path in candidates:
            if (path / attempt_id).is_dir():
                saved = ESCheckpointStore(path).load(attempt_id)
                if saved.generation != int(path.name[11:]):
                    raise ValueError("ES generation directory disagrees with checkpoint")
                return path
        # Accept an earlier standalone checkpoint when migrating to the
        # generation-directory layout. Its next snapshot uses a new root.
        if (self.checkpoint_dir / attempt_id).is_dir():
            return self.checkpoint_dir
        raise FileNotFoundError("no completed ES generation checkpoint for this attempt")

    def _preflight(self, context: StageContext) -> None:
        if self.evolution.device != "cpu":
            raise ValueError("ES stage checkpoint/resume currently supports CPU only; rejected before evaluation")
        if (not isinstance(context, StageContext) or context.stage != self.stage
                or context.budget != context.spec.stage_budget.get(self.stage)):
            raise ValueError("ES stage context differs from frozen run")
        spec, cap = context.spec, context.budget
        expected_algorithm = "pgpe" if self.stage == "E1" else "cmaes"
        expected_kind = "actor" if self.stage == "E1" else "soft_risk"
        auxiliary_candidates = int(self.stage == "E2" and self.include_neutral_baseline)
        expected_evaluations = (self.evolution.population_size + auxiliary_candidates) * self.evolution.generations
        expected_lane_requests = expected_evaluations * len(self.episode_ids)
        if (self.evolution.algorithm != expected_algorithm or self.evolution.seed != spec.seed
                or self.evolution.population_size > cap.max_population
                or self.evolution.generations > cap.max_generations
                or self.evolution.max_evaluations > cap.max_evaluations
                or expected_evaluations > cap.max_evaluations
                or expected_lane_requests > cap.max_lane_requests
                or self.factory_config.get("candidate_kind") != expected_kind
                or self.factory_config.get("actor_signature") != spec.actor_signature):
            raise ValueError("ES stage optimizer, worker recipe, or lane use exceeds frozen RunSpec")
        if self.stage == "E1":
            if not isinstance(self.candidate_spec, ActorGenomeSpec) or context.previous is not None:
                raise ValueError("E1 requires ActorGenome search at route start")
        else:
            if (not isinstance(self.candidate_spec, SoftRiskSpace)
                    or self.candidate_spec.signature != spec.soft_risk_schema_hash
                    or context.previous is None or context.previous.stage != "R1"
                    or self.factory_config.get("frozen_actor_hash") != context.previous.actor_hash):
                raise ValueError("E2 must preserve the attested R1 actor and SoftRisk search space")
            if self.include_neutral_baseline:
                self.candidate_spec.decode(np.ones(len(self.candidate_spec.dimensions), dtype=np.float32))
            if (self.factory_ref == "rles_orchestrator.native_es_worker:make_native_es_session"
                    and self.factory_config.get("actor_inference") != "l4_inference"):
                raise ValueError("E2 must evaluate the exported L4 inference actor")
            if self.factory_ref == "rles_orchestrator.native_es_worker:make_native_es_session":
                artifact = self.factory_config.get("frozen_actor_artifact")
                if not isinstance(artifact, str) or not Path(artifact).is_absolute():
                    raise ValueError("E2 requires an absolute R1 inference artifact path")
                from rles_rl_sb3.actor import InferenceActor

                InferenceActor(Path(artifact), verify=True)
                if (Path(artifact) / "COMPLETED").read_text(encoding="ascii") != context.previous.actor_hash:
                    raise ValueError("E2 inference artifact differs from attested R1 actor")
        if not self.resume:
            existing = self.selection.ledger.es_lane_request_count(
                self.selection.protocol_id, spec.run_id, spec.attempt_id, self.stage,
            )
            snapshots = (self.checkpoint_dir / spec.attempt_id).exists() or any(
                (path / spec.attempt_id).exists()
                for path in self.checkpoint_dir.glob("generation-[0-9]*") if path.is_dir()
            )
            if existing or snapshots:
                raise RuntimeError("ES attempt already started; resume from its completed checkpoint")
        else:
            # A generation is checkpointed only after every candidate×episode
            # request and optimizer update completes. Reservations are durable
            # before dispatch, so a crash inside the next generation leaves
            # extra ledger requests beyond the latest checkpoint. Reject that
            # state here, before starting worker processes or taking leases;
            # retrying those requests would violate the one-charge policy.
            checkpoint_root = self._latest_checkpoint_root(spec.attempt_id)
            checkpoint = ESCheckpointStore(checkpoint_root).load(spec.attempt_id)
            checkpoint_requests = (checkpoint.evaluations +
                                   (checkpoint.generation if auxiliary_candidates else 0)) * len(self.episode_ids)
            reserved_requests = self.selection.ledger.es_lane_request_count(
                self.selection.protocol_id, spec.run_id, spec.attempt_id, self.stage,
            )
            if reserved_requests != checkpoint_requests:
                raise RuntimeError(
                    "ES checkpoint is behind durable lane reservations; incomplete generation "
                    "cannot resume in this attempt"
                )

    def __call__(self, context: StageContext) -> ProviderStageOutput:
        if self._used:
            raise RuntimeError("ES stage executor is single-use")
        self._preflight(context)
        self._used = True
        spec, cap = context.spec, context.budget
        pool = CPUWorkerPool(
            workers=self.workers, factory_ref=self.factory_ref,
            factory_config=self.factory_config, mode="es", quota=self.quota,
            lease_owner=self.lease_owner, ram_bytes_per_worker=self.ram_bytes_per_worker,
            cpu_cores_per_worker=self.cpu_cores_per_worker,
            startup_timeout_s=self.startup_timeout_s, torch_threads=self.torch_threads,
            enforce_memory_limit=self.enforce_memory_limit,
        )
        lane = ESEvalLane(pool, timeout_s=self.timeout_s)
        dispatcher = WorkerLaneDispatcher(
            lane, self.payload_builder, payload_builder_id=self.payload_builder_id,
            fitness_spec=self.fitness_spec, tail_fraction=self.tail_fraction,
        )

        def reserve_requests(requests: tuple[CandidateRequest, ...]) -> None:
            self.selection.reserve_lane_requests(spec, stage=self.stage, requests=requests)

        def verify_requests(keys: tuple[tuple[str, str, int, str, str, int], ...]) -> None:
            self.selection.verify_lane_requests(spec, stage=self.stage, request_keys=keys)

        lane_budget = EvaluationBudget(
            cap.max_lane_requests,
            reserve_requests=reserve_requests,
            verify_requests=verify_requests,
            reservation_id=self.selection.reservation_scope(spec, stage=self.stage),
        )
        evaluator = GenerationLaneEvaluator(
            run_id=spec.run_id, attempt_id=spec.attempt_id, fold_id=self.fold_id,
            root_seed=spec.seed, episode_ids=self.episode_ids,
            candidate_spec=self.candidate_spec, actor_signature=spec.actor_signature,
            dispatch=dispatcher, dispatch_id=dispatcher.signature, budget=lane_budget,
            include_neutral_baseline=self.include_neutral_baseline,
            include_constraint_evidence=self.include_constraint_evidence,
            feasibility_first=isinstance(self.fitness_spec, FeasibilityFirstFitnessSpec),
        )
        provider = EvoTorchProvider(
            self.evolution,
            risk_space=self.candidate_spec if isinstance(self.candidate_spec, SoftRiskSpace) else None,
            batch_fitness=evaluator, batch_fitness_id=evaluator.signature,
        )
        import torch
        torch_rng, numpy_rng, python_rng = torch.get_rng_state().clone(), np.random.get_state(), random.getstate()
        try:
            if self.resume:
                provider.resume(
                    self._latest_checkpoint_root(spec.attempt_id), attempt_id=spec.attempt_id,
                    actor_signature=spec.actor_signature, dataset_hash=spec.data_hash,
                    config_hash=spec.config_hash,
                )
            else:
                random.seed(spec.seed)
                np.random.seed(spec.seed)
                torch.manual_seed(spec.seed)
            dimension = (self.candidate_spec.size if isinstance(self.candidate_spec, ActorGenomeSpec)
                         else len(self.candidate_spec.dimensions))
            completed = provider.result().generations if self.resume else 0
            if completed < self.evolution.generations:
                with pool:
                    if completed == 0:
                        provider.evolve(dimension=dimension, initial_bounds=self.initial_bounds,
                                        center=self.center, generations=1)
                        completed = 1
                        provider.save(
                            self._checkpoint_root(completed), attempt_id=spec.attempt_id,
                            actor_signature=spec.actor_signature, dataset_hash=spec.data_hash,
                            config_hash=spec.config_hash,
                        )
                    while completed < self.evolution.generations:
                        provider.continue_evolution(generations=1)
                        completed += 1
                        provider.save(
                            self._checkpoint_root(completed), attempt_id=spec.attempt_id,
                            actor_signature=spec.actor_signature, dataset_hash=spec.data_hash,
                            config_hash=spec.config_hash,
                        )
            result = provider.result()
            if self.stage == "E2" and self.include_neutral_baseline:
                if not isinstance(self.candidate_spec, SoftRiskSpace):
                    raise RuntimeError("E2 neutral candidate lost its SoftRisk schema")
                baseline = evaluator.best_neutral_baseline
                if baseline is None:
                    raise RuntimeError("E2 completed without a scored exact-neutral R1 candidate")
                result = _select_e2_candidate(result, self.candidate_spec, baseline)
        finally:
            try:
                pool.close()
            finally:
                try:
                    self.worker_resource_evidence = pool.resource_evidence()
                finally:
                    torch.set_rng_state(torch_rng)
                    np.random.set_state(numpy_rng)
                    random.setstate(python_rng)
        expected_requests = result.evaluations * len(self.episode_ids)
        if lane_budget.consumed != expected_requests:
            raise RuntimeError("completed ES stage lane requests differ from optimizer evaluations")
        evidence = evaluator.evidence_hash
        if self.stage == "E1":
            actor_spec = self.candidate_spec
            if not isinstance(actor_spec, ActorGenomeSpec):
                raise RuntimeError("E1 actor schema disappeared")
            genome = np.asarray(result.best_search_vector, dtype=np.float32)
            actor_hash = genome_hash(actor_spec, genome)
            soft_risk_hash = spec.initial_soft_risk_hash
            payload_hash = sha256(genome.tobytes(order="C") + bytes.fromhex(evidence)).hexdigest()
            metadata = {"actor_signature": spec.actor_signature,
                        "scenario_fold_id": self.fold_id, "lane_evidence_hash": evidence}
        else:
            if result.risk_parameters is None or context.previous is None:
                raise RuntimeError("E2 completed without bounded soft-risk parameters")
            actor_hash = context.previous.actor_hash
            soft_risk_hash = _digest(result.risk_parameters)
            payload_hash = sha256(bytes.fromhex(soft_risk_hash) + bytes.fromhex(evidence)).hexdigest()
            metadata = {"soft_risk_bounds_gate": "PASS",
                        "soft_risk_schema_hash": spec.soft_risk_schema_hash,
                        "selected_candidate_kind": result.selected_candidate_kind,
                        "selected_fitness": repr(result.best_fitness),
                        "scenario_fold_id": self.fold_id,
                        "neutral_baseline_protocol": (
                            "paired_generation_neutral_r1_v1" if self.include_neutral_baseline else "disabled"
                        ),
                        "lane_evidence_hash": evidence}
        if self.include_constraint_evidence:
            metadata["fitness_protocol"] = (
                self.fitness_spec.protocol if isinstance(self.fitness_spec, ConstraintPriorityFitnessSpec)
                else "legacy_weighted_v1"
            )
            metadata["fitness_spec_json"] = json.dumps(
                vars(self.fitness_spec), sort_keys=True, separators=(",", ":"), allow_nan=False,
            )
            metadata["constraint_evidence_json"] = json.dumps(
                evaluator.constraint_evidence, sort_keys=True, separators=(",", ":"), allow_nan=False,
            )
        return ProviderStageOutput(
            artifact_id=self.artifact_id, payload_hash=payload_hash,
            actor_hash=actor_hash, soft_risk_hash=soft_risk_hash, metadata=metadata,
            evolution_result=result, population_size=self.evolution.population_size,
            lane_requests=lane_budget.consumed,
        )
