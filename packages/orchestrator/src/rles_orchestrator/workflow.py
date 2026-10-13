"""A0-A4 single-cycle Hybrid run machine with strict stage freeze and budgets.

No algorithm implementation lives in this module; the L4/L6 stage runner and
SelectionLedger attestor are mandatory injected dependencies at run time.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from enum import Enum
from hashlib import sha256
import json
from types import MappingProxyType
from typing import Callable, Iterable, Mapping, Protocol

from .transfer import TransferResult

_HEX = frozenset("0123456789abcdef")
_ALGOS = frozenset({"PPO", "SAC", "TD3", "DDPG"})


def _is_hash(v: object) -> bool:
    return isinstance(v, str) and len(v) == 64 and all(c in _HEX for c in v)


def _canonical(v: object) -> bytes:
    return json.dumps(v, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


class Route(str, Enum):
    A0 = "A0"  # R1
    A1 = "A1"  # E1
    A2 = "A2"  # R1 E2
    A3 = "A3"  # E1 R1
    A4 = "A4"  # E1 R1 E2


STAGES: dict[Route, tuple[str, ...]] = {
    Route.A0: ("R1",), Route.A1: ("E1",), Route.A2: ("R1", "E2"),
    Route.A3: ("E1", "R1"), Route.A4: ("E1", "R1", "E2"),
}


@dataclass(frozen=True)
class StageBudget:
    max_population: int = 0
    max_generations: int = 0
    max_evaluations: int = 0
    max_rl_steps: int = 0
    max_validation_queries: int = 0
    # L6 candidate×episode worker requests are distinct from ES objective calls.
    max_lane_requests: int = 0

    def __post_init__(self) -> None:
        if any(type(v) is not int or v < 0 for v in vars(self).values()):
            raise ValueError("stage budget must contain nonnegative integer caps")
        if self.max_validation_queries != 0:
            raise ValueError("train-stage validation queries must be zero")

    def validate_for(self, stage: str) -> None:
        if stage in {"E1", "E2"}:
            if (self.max_population < 4 or self.max_generations < 1
                    or self.max_evaluations < self.max_population * self.max_generations
                    or self.max_rl_steps != 0):
                raise ValueError("ES budget requires adequate population/generations/evaluations")
        elif stage == "R1":
            if (self.max_rl_steps < 1 or any((self.max_population, self.max_generations,
                                              self.max_evaluations, self.max_lane_requests))):
                raise ValueError("R1 budget requires only positive RL steps")
        else:
            raise ValueError("unknown stage")


@dataclass(frozen=True)
class StageUsage:
    population: int = 0
    generations: int = 0
    evaluations: int = 0
    rl_steps: int = 0
    validation_queries: int = 0
    lane_requests: int = 0

    def __post_init__(self) -> None:
        if any(type(v) is not int or v < 0 for v in vars(self).values()):
            raise ValueError("usage must be nonnegative integer counters")

    @classmethod
    def from_rl_performance(cls, performance: object) -> StageUsage:
        """Derive R1 usage from the L4 Performance returned by ``train``.

        This structural adapter keeps L5 independent of the L4 package while
        using its measured SB3 timestep delta instead of a caller-supplied
        counter. Use ``from_rl_performances`` when a stage resumed training in
        multiple calls, because each L4 result reports that call's exact delta.
        """
        samples = getattr(performance, "samples", None)
        if type(samples) is not int or samples < 1:
            raise ValueError("L4 Performance must report a positive integer sample count")
        return cls(rl_steps=samples)

    @classmethod
    def from_rl_performances(cls, performances: Iterable[object]) -> StageUsage:
        """Sum exact L4 timestep deltas over all training/resume calls."""
        total = 0
        count = 0
        for performance in performances:
            samples = getattr(performance, "samples", None)
            if type(samples) is not int or samples < 1:
                raise ValueError("each L4 Performance must report a positive integer sample count")
            total += samples
            count += 1
        if count == 0:
            raise ValueError("at least one L4 Performance result is required")
        return cls(rl_steps=total)

    @classmethod
    def from_evolution_result(
        cls, result: object, *, population_size: int, lane_requests: int = 0,
    ) -> StageUsage:
        """Derive ES usage from L5 EvolutionResult and frozen population size."""
        generations = getattr(result, "generations", None)
        evaluations = getattr(result, "evaluations", None)
        if (type(population_size) is not int or population_size < 4
                or type(generations) is not int or generations < 1
                or type(evaluations) is not int or evaluations < population_size * generations):
            raise ValueError("L5 EvolutionResult counters do not match the frozen population")
        if type(lane_requests) is not int or lane_requests < 0:
            raise ValueError("lane request usage must be a nonnegative integer")
        return cls(population=population_size, generations=generations,
                   evaluations=evaluations, lane_requests=lane_requests)

    def check(self, stage: str, cap: StageBudget) -> None:
        if (self.population > cap.max_population or self.generations > cap.max_generations
                or self.evaluations > cap.max_evaluations or self.rl_steps > cap.max_rl_steps
                or self.validation_queries > cap.max_validation_queries
                or self.lane_requests > cap.max_lane_requests):
            raise RuntimeError("stage usage exceeds its frozen caps")
        if self.validation_queries:
            raise RuntimeError("stage fit must not query Validation")
        if stage in {"E1", "E2"}:
            if (self.population < 4 or self.generations < 1
                    or self.evaluations < self.population * self.generations or self.rl_steps):
                raise RuntimeError("ES stage usage incomplete or inconsistent")
            if (cap.max_lane_requests > 0) != (self.lane_requests > 0):
                raise RuntimeError("ES lane request usage must match its frozen lane cap")
            if self.lane_requests and self.lane_requests < self.evaluations:
                raise RuntimeError("ES lane requests cannot be fewer than candidate evaluations")
        elif stage == "R1":
            if self.rl_steps < 1 or any((self.population, self.generations, self.evaluations,
                                         self.lane_requests)):
                raise RuntimeError("RL stage usage invalid")
        else:
            raise RuntimeError("unrecognised stage")


@dataclass(frozen=True)
class RunSpec:
    run_id: str
    attempt_id: str
    route: Route
    algorithm: str
    seed: int
    data_hash: str
    feature_hash: str
    config_hash: str
    hard_risk_hash: str
    execution_hash: str
    normalization_hash: str
    soft_risk_schema_hash: str
    initial_soft_risk_hash: str
    actor_signature: str
    stage_budget: Mapping[str, StageBudget]

    def __post_init__(self) -> None:
        if (not isinstance(self.run_id, str) or not self.run_id or not isinstance(self.attempt_id, str)
                or not self.attempt_id or self.algorithm not in _ALGOS or not isinstance(self.route, Route)
                or type(self.seed) is not int or self.seed < 0):
            raise ValueError("invalid run identity/algorithm/route/seed")
        for key in ("data_hash", "feature_hash", "config_hash", "hard_risk_hash",
                    "execution_hash", "normalization_hash", "soft_risk_schema_hash", "initial_soft_risk_hash", "actor_signature"):
            if not _is_hash(getattr(self, key)):
                raise ValueError(f"missing or invalid frozen hash: {key}")
        budgets = dict(self.stage_budget)
        if set(budgets) != set(STAGES[self.route]):
            raise ValueError("stage budgets must exactly match the chosen route")
        for stage, budget in budgets.items():
            if not isinstance(budget, StageBudget):
                raise TypeError("StageBudget required")
            budget.validate_for(stage)
        object.__setattr__(self, "stage_budget", MappingProxyType(budgets))

    @property
    def fingerprint(self) -> str:
        values = {k: getattr(self, k) for k in (
            "run_id", "attempt_id", "route", "algorithm", "seed", "data_hash", "feature_hash",
            "config_hash", "hard_risk_hash", "execution_hash", "normalization_hash",
            "soft_risk_schema_hash", "initial_soft_risk_hash", "actor_signature",
        )}
        values["stage_budget"] = {stage: vars(b) for stage, b in sorted(self.stage_budget.items())}
        return sha256(_canonical(values)).hexdigest()


class ReceiptHashProvider(Protocol):
    def receipt_hash_for(self, spec: object, *, stage: str, artifact_id: str,
                         payload_hash: str, usage: object) -> str: ...


@dataclass(frozen=True)
class StageArtifact:
    stage: str
    artifact_id: str
    payload_hash: str
    run_fingerprint: str
    parent_id: str | None
    actor_hash: str
    soft_risk_hash: str
    hard_risk_hash: str
    data_hash: str
    execution_hash: str
    feature_hash: str
    normalization_hash: str
    usage: StageUsage
    ledger_receipt_hash: str
    metadata: Mapping[str, str]
    transfer_evidence: TransferResult | None = None

    @classmethod
    def for_stage(cls, spec: RunSpec, stage: str, *, artifact_id: str, payload_hash: str,
                  actor_hash: str, soft_risk_hash: str, usage: StageUsage,
                  receipt_provider: ReceiptHashProvider, metadata: Mapping[str, str],
                  previous: StageArtifact | None = None,
                  transfer_evidence: TransferResult | None = None) -> StageArtifact:
        """Build a stage artifact from frozen run identity and its measured usage.

        The L6 receipt hash is computed before construction so the immutable
        artifact carries the exact identity and usage which the L6 attestor
        will persist after Orchestrator validates the completed stage.
        """
        if not isinstance(spec, RunSpec) or stage not in STAGES[spec.route]:
            raise ValueError("stage is not part of the frozen RunSpec route")
        if previous is not None and previous.run_fingerprint != spec.fingerprint:
            raise ValueError("previous stage belongs to a different frozen run")
        receipt = receipt_provider.receipt_hash_for(
            spec, stage=stage, artifact_id=artifact_id, payload_hash=payload_hash, usage=usage,
        )
        return cls(
            stage=stage, artifact_id=artifact_id, payload_hash=payload_hash,
            run_fingerprint=spec.fingerprint,
            parent_id=previous.artifact_id if previous is not None else None,
            actor_hash=actor_hash, soft_risk_hash=soft_risk_hash,
            hard_risk_hash=spec.hard_risk_hash, data_hash=spec.data_hash,
            execution_hash=spec.execution_hash, feature_hash=spec.feature_hash,
            normalization_hash=spec.normalization_hash, usage=usage,
            ledger_receipt_hash=receipt, metadata=metadata,
            transfer_evidence=transfer_evidence,
        )

    def __post_init__(self) -> None:
        if self.stage not in {"E1", "R1", "E2"} or not self.artifact_id:
            raise ValueError("invalid stage artifact")
        for key in ("payload_hash", "run_fingerprint", "actor_hash", "soft_risk_hash",
                    "hard_risk_hash", "data_hash", "execution_hash", "feature_hash",
                    "normalization_hash", "ledger_receipt_hash"):
            if not _is_hash(getattr(self, key)):
                raise ValueError(f"invalid stage artifact fingerprint: {key}")
        if self.parent_id is not None and (not isinstance(self.parent_id, str) or not self.parent_id):
            raise ValueError("invalid stage ancestry")
        if not isinstance(self.usage, StageUsage):
            raise TypeError("stage usage must be StageUsage")
        if self.transfer_evidence is not None and not isinstance(self.transfer_evidence, TransferResult):
            raise TypeError("typed TransferResult evidence required")
        metadata = dict(self.metadata)
        if any(not isinstance(k, str) or not k or not isinstance(v, str) for k, v in metadata.items()):
            raise ValueError("invalid immutable metadata")
        object.__setattr__(self, "metadata", MappingProxyType(metadata))


@dataclass(frozen=True)
class StageContext:
    spec: RunSpec
    stage: str
    budget: StageBudget
    previous: StageArtifact | None


StageRunner = Callable[[StageContext], StageArtifact]
UsageAttestor = Callable[[RunSpec, StageArtifact], str]


@dataclass(frozen=True)
class ProviderStageOutput:
    """Artifact facts plus native Provider results returned by one stage run."""
    artifact_id: str
    payload_hash: str
    actor_hash: str
    soft_risk_hash: str
    metadata: Mapping[str, str]
    rl_performances: tuple[object, ...] = ()
    evolution_result: object | None = None
    population_size: int | None = None
    transfer_evidence: TransferResult | None = None
    lane_requests: int = 0
    rl_prior_steps: int = 0

    def __post_init__(self) -> None:
        if (not isinstance(self.artifact_id, str) or not self.artifact_id.strip()
                or not isinstance(self.rl_performances, tuple)
                or type(self.lane_requests) is not int or self.lane_requests < 0
                or type(self.rl_prior_steps) is not int or self.rl_prior_steps < 0
                or not isinstance(self.metadata, Mapping)
                or any(not isinstance(key, str) or not isinstance(value, str)
                       for key, value in self.metadata.items())):
            raise ValueError("stage output requires artifact identity and immutable performance tuple")
        lane_evidence_hash = self.metadata.get("lane_evidence_hash")
        if lane_evidence_hash is not None and not _is_hash(lane_evidence_hash):
            raise ValueError("lane evidence hash must be a lowercase SHA256")
        is_rl = bool(self.rl_performances) or self.rl_prior_steps > 0
        is_es = self.evolution_result is not None or self.population_size is not None
        if is_rl == is_es:
            raise ValueError("stage output must contain exactly one RL or ES Provider result")
        if is_rl and (self.evolution_result is not None or self.population_size is not None):
            raise ValueError("RL stage output cannot include ES result fields")
        if is_rl and (self.lane_requests or lane_evidence_hash is not None):
            raise ValueError("RL stage output cannot include ES lane usage or evidence")
        if is_es and self.rl_prior_steps:
            raise ValueError("ES stage output cannot include prior RL step usage")
        if is_es and (self.evolution_result is None or type(self.population_size) is not int):
            raise ValueError("ES stage output requires an EvolutionResult and frozen population size")
        if is_es and ((self.lane_requests > 0) != (lane_evidence_hash is not None)):
            raise ValueError("ES lane request usage and lane evidence hash must be provided together")


ProviderStageExecutor = Callable[[StageContext], ProviderStageOutput]


class ProviderBackedStageRunner:
    """Convert native L4/L5 Provider results into a receipt-bound StageArtifact."""

    def __init__(self, stage: str, execute: ProviderStageExecutor,
                 receipt_provider: ReceiptHashProvider):
        if stage not in {"E1", "R1", "E2"} or not callable(execute):
            raise ValueError("valid stage and Provider executor required")
        if not callable(getattr(receipt_provider, "receipt_hash_for", None)):
            raise TypeError("L6 receipt hash provider required")
        self.stage = stage
        self.execute = execute
        self.receipt_provider = receipt_provider

    def __call__(self, context: StageContext) -> StageArtifact:
        if context.stage != self.stage:
            raise RuntimeError("Provider stage runner invoked for a different route stage")
        output = self.execute(context)
        if not isinstance(output, ProviderStageOutput):
            raise TypeError("Provider executor must return ProviderStageOutput")
        if self.stage == "R1":
            if (not output.rl_performances and output.rl_prior_steps == 0) or output.evolution_result is not None:
                raise RuntimeError("R1 requires measured or checkpoint-attested L4 timestep usage")
            measured = (StageUsage.from_rl_performances(output.rl_performances)
                        if output.rl_performances else StageUsage())
            usage = StageUsage(rl_steps=output.rl_prior_steps + measured.rl_steps)
        else:
            if output.evolution_result is None or output.population_size is None:
                raise RuntimeError("E1/E2 requires an L5 EvolutionResult and frozen population size")
            expected_es_algorithm = "pgpe" if self.stage == "E1" else "cmaes"
            if getattr(output.evolution_result, "algorithm", None) != expected_es_algorithm:
                raise RuntimeError(f"{self.stage} requires the frozen {expected_es_algorithm} searcher")
            usage = StageUsage.from_evolution_result(
                output.evolution_result, population_size=output.population_size,
                lane_requests=output.lane_requests,
            )
        usage.check(context.stage, context.budget)
        return StageArtifact.for_stage(
            context.spec, context.stage, artifact_id=output.artifact_id,
            payload_hash=output.payload_hash, actor_hash=output.actor_hash,
            soft_risk_hash=output.soft_risk_hash, usage=usage,
            receipt_provider=self.receipt_provider, metadata=output.metadata,
            previous=context.previous, transfer_evidence=output.transfer_evidence,
        )


class Orchestrator:
    """Only one pass over the route; all stages are checked before closing."""
    def __init__(self, runners: Mapping[str, StageRunner], *, attest_usage: UsageAttestor):
        if not callable(attest_usage):
            raise TypeError("L6 ledger usage attestation must be injected")
        if any(stage not in {"E1", "R1", "E2"} or not callable(fn) for stage, fn in runners.items()):
            raise ValueError("invalid stage runner registry")
        self._runners = dict(runners)
        self._attest_usage = attest_usage
        self.state = "CREATED"
        self.lineage: tuple[StageArtifact, ...] = ()

    def run(self, spec: RunSpec) -> tuple[StageArtifact, ...]:
        if self.state != "CREATED":
            raise RuntimeError("orchestrator is single-use; no implicit retraining loops")
        if not isinstance(spec, RunSpec):
            raise TypeError("RunSpec required")
        stages = STAGES[spec.route]
        if any(stage not in self._runners for stage in stages):
            raise RuntimeError("no stage fallback permitted")
        self.state = "READY"
        try:
            self.state = "RUNNING"
            for stage in stages:
                before = self.lineage[-1] if self.lineage else None
                context = StageContext(spec, stage, spec.stage_budget[stage], before)
                artifact = self._runners[stage](context)
                self._verify(spec, context, artifact)
                receipt = self._attest_usage(spec, artifact)
                if not _is_hash(receipt) or receipt != artifact.ledger_receipt_hash:
                    raise RuntimeError("L6 SelectionLedger usage receipt mismatch")
                self.lineage = (*self.lineage, artifact)
            self.state = "DRAINING"
            self.state = "CLOSED"
            return self.lineage
        except Exception:
            self.state = "FAILED"
            raise

    def _verify(self, spec: RunSpec, ctx: StageContext, artifact: StageArtifact) -> None:
        if not isinstance(artifact, StageArtifact) or artifact.stage != ctx.stage:
            raise RuntimeError("stage did not return the requested StageArtifact")
        if artifact.run_fingerprint != spec.fingerprint:
            raise RuntimeError("stage mutated run identity / budget / config")
        for field, expected in (("hard_risk_hash", spec.hard_risk_hash),
                                ("data_hash", spec.data_hash),
                                ("execution_hash", spec.execution_hash),
                                ("feature_hash", spec.feature_hash),
                                ("normalization_hash", spec.normalization_hash)):
            if getattr(artifact, field) != expected:
                raise RuntimeError(f"stage hot-swapped a frozen {field}")
        if artifact.parent_id != (ctx.previous.artifact_id if ctx.previous else None):
            raise RuntimeError("broken stage lineage parent")
        if any(a.artifact_id == artifact.artifact_id for a in self.lineage):
            raise RuntimeError("repeated stage artifact id")
        artifact.usage.check(ctx.stage, ctx.budget)
        if ctx.stage in {"E1", "R1"}:
            if artifact.soft_risk_hash != spec.initial_soft_risk_hash:
                raise RuntimeError("E1/R1 modified frozen initial SoftRisk configuration")
        if ctx.stage == "R1":
            if ctx.previous is not None:
                if ctx.previous.stage != "E1":
                    raise RuntimeError("R1 stage predecessor must be E1")
                report = artifact.transfer_evidence
                if (report is None or report.algorithm != spec.algorithm or not report.parity.passed
                        or report.parity.actor_signature != spec.actor_signature
                        or report.training_split != "train_inner"):
                    raise RuntimeError("typed E1->R1 transfer parity evidence missing")
                if report.method == "DIRECT":
                    if report.load_evidence is None or not report.load_evidence.exact or report.distillation is not None:
                        raise RuntimeError("DIRECT transfer lacks full parameter coverage")
                elif report.method == "DISTILLATION_NOT_WEIGHT_INHERITANCE":
                    if report.distillation is None or report.distillation.training_split != "train_inner":
                        raise RuntimeError("distillation imitation evidence missing")
                else:
                    raise RuntimeError("unknown E1->R1 transfer method")
                if artifact.metadata.get("transfer_report_hash") != transfer_evidence_hash(report):
                    raise RuntimeError("transfer evidence and artifact hash disagree")
            if ctx.previous is None and artifact.transfer_evidence is not None:
                raise RuntimeError("pure RL baseline must not attach fictitious E1 transfer evidence")
            if artifact.metadata.get("rl_training_complete") != "true":
                raise RuntimeError("R1 training completion not attested")
        if ctx.stage == "E2":
            if ctx.previous is None or ctx.previous.stage != "R1":
                raise RuntimeError("E2 must follow R1")
            if artifact.actor_hash != ctx.previous.actor_hash:
                raise RuntimeError("E2 modified frozen R1 actor")
            if artifact.metadata.get("soft_risk_bounds_gate") != "PASS":
                raise RuntimeError("E2 lacks SoftRisk bounds evidence")
            if artifact.metadata.get("soft_risk_schema_hash") != spec.soft_risk_schema_hash:
                raise RuntimeError("E2 soft-risk search schema changed")
        if ctx.stage == "E1":
            if ctx.previous is not None:
                raise RuntimeError("E1 can only occur once at start")
            if artifact.metadata.get("actor_signature") != spec.actor_signature:
                raise RuntimeError("E1 actor signature incorrect")


def lineage_hash(artifacts: tuple[StageArtifact, ...]) -> str:
    if not artifacts or any(not isinstance(a, StageArtifact) for a in artifacts):
        raise ValueError("nonempty verified stage lineage required")
    for prev, current in zip(artifacts, artifacts[1:]):
        if current.parent_id != prev.artifact_id or current.run_fingerprint != prev.run_fingerprint:
            raise ValueError("invalid stage ancestry or run fingerprint")
    entries = []
    for a in artifacts:
        entries.append({
            "stage": a.stage, "artifact_id": a.artifact_id, "payload_hash": a.payload_hash,
            "run_fingerprint": a.run_fingerprint, "parent_id": a.parent_id,
            "actor_hash": a.actor_hash, "soft_risk_hash": a.soft_risk_hash,
            "hard_risk_hash": a.hard_risk_hash, "data_hash": a.data_hash,
            "execution_hash": a.execution_hash, "feature_hash": a.feature_hash,
            "normalization_hash": a.normalization_hash,
            "ledger_receipt_hash": a.ledger_receipt_hash,
            "usage": vars(a.usage), "metadata": dict(a.metadata),
            "transfer_evidence": asdict(a.transfer_evidence) if a.transfer_evidence else None,
        })
    return sha256(_canonical(entries)).hexdigest()


def transfer_evidence_hash(report: TransferResult) -> str:
    if not isinstance(report, TransferResult):
        raise TypeError("typed TransferResult required")
    return sha256(_canonical(asdict(report))).hexdigest()
