"""ESEvalLane V2 candidate envelopes, deterministic CRN, strict reconciliation.

All candidates are Train-inner only. The external ComputeProvider creates
independent engine/account instances; the IDs here make isolation auditable.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math
from threading import Lock
from typing import Callable, Iterable, Mapping, Protocol, Sequence

import numpy as np
from .actor import ActorGenomeSpec
from .fitness import FitnessSpec
from .soft_risk import SoftRiskSpace


_HEX = frozenset("0123456789abcdef")


def _sha_json(value: object) -> str:
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                             allow_nan=False).encode("utf-8")).hexdigest()


def _check_hash(value: object) -> None:
    if not isinstance(value, str) or len(value) != 64 or any(x not in _HEX for x in value):
        raise ValueError("expected lowercase SHA256")


@dataclass(frozen=True)
class CandidateRequest:
    run_id: str
    attempt_id: str
    generation: int
    candidate_id: str
    episode_id: str
    seed: int
    genome: tuple[float, ...]
    actor_signature: str
    candidate_kind: str
    candidate_schema_hash: str
    account_isolation_id: str
    split: str = "train_inner"

    def __post_init__(self) -> None:
        if any(not isinstance(v, str) or not v.strip() for v in (
            self.run_id, self.attempt_id, self.candidate_id, self.episode_id, self.account_isolation_id
        )):
            raise ValueError("empty evaluation identity")
        if (type(self.generation) is not int or self.generation < 0
                or type(self.seed) is not int or not 0 <= self.seed < 2 ** 32):
            raise ValueError("invalid evaluation seed or generation")
        if self.split != "train_inner":
            raise PermissionError("ES requests must use Train-inner")
        _check_hash(self.actor_signature)
        if not isinstance(self.candidate_kind, str) or self.candidate_kind not in {"actor", "soft_risk"}:
            raise ValueError("unsupported candidate parameter kind")
        _check_hash(self.candidate_schema_hash)
        if not isinstance(self.genome, tuple) or not self.genome or any(
            isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
            for v in self.genome
        ):
            raise ValueError("genome must be finite nonempty numeric tuple")

    @property
    def key(self) -> tuple[str, str, int, str, str, int]:
        return (self.run_id, self.attempt_id, self.generation,
                self.candidate_id, self.episode_id, self.seed)

    @property
    def request_hash(self) -> str:
        return _sha_json((self.key, self.actor_signature, self.candidate_kind,
                          self.candidate_schema_hash, self.account_isolation_id,
                          self.genome, self.split))


@dataclass(frozen=True)
class CandidateResult:
    run_id: str
    attempt_id: str
    generation: int
    candidate_id: str
    episode_id: str
    seed: int
    request_hash: str
    account_isolation_id: str
    fitness: float
    evidence_hash: str
    split: str = "train_inner"

    def __post_init__(self) -> None:
        if any(not isinstance(v, str) or not v.strip() for v in (
            self.run_id, self.attempt_id, self.candidate_id, self.episode_id, self.account_isolation_id
        )):
            raise ValueError("empty candidate result identity")
        if (type(self.generation) is not int or self.generation < 0
                or type(self.seed) is not int or not 0 <= self.seed < 2 ** 32):
            raise ValueError("invalid evaluation result seed/generation")
        if self.split != "train_inner":
            raise PermissionError("ES result cannot originate from Validation/Test")
        if isinstance(self.fitness, bool) or not isinstance(self.fitness, (int, float)) or not math.isfinite(self.fitness):
            raise ValueError("nonfinite result fitness")
        _check_hash(self.request_hash)
        _check_hash(self.evidence_hash)

    @property
    def key(self) -> tuple[str, str, int, str, str, int]:
        return (self.run_id, self.attempt_id, self.generation,
                self.candidate_id, self.episode_id, self.seed)


def scenario_seed(root_seed: int, fold_id: str, generation: int, episode_id: str) -> int:
    """Common scenario for all candidates, but never a common account."""
    if (type(root_seed) is not int or root_seed < 0 or type(generation) is not int or generation < 0
            or not fold_id or not episode_id):
        raise ValueError("invalid CRN seed schedule")
    digest = sha256(json.dumps((root_seed, fold_id, generation, episode_id),
                               separators=(",", ":")).encode()).digest()
    return int.from_bytes(digest[:4], "big")


def reconcile(requests: Iterable[CandidateRequest], results: Iterable[CandidateResult]) -> tuple[CandidateResult, ...]:
    expected: dict[tuple, CandidateRequest] = {}
    accounts: set[str] = set()
    for req in requests:
        if not isinstance(req, CandidateRequest) or req.key in expected:
            raise ValueError("duplicate or invalid candidate request")
        if req.account_isolation_id in accounts:
            raise ValueError("engine/account reused across candidate evaluations")
        expected[req.key] = req
        accounts.add(req.account_isolation_id)
    if not expected:
        raise ValueError("no candidate requests")
    actual: dict[tuple, CandidateResult] = {}
    for result in results:
        if not isinstance(result, CandidateResult) or result.key not in expected or result.key in actual:
            raise ValueError("duplicate, foreign, or invalid evaluation result")
        request = expected[result.key]
        if (result.request_hash != request.request_hash
                or result.account_isolation_id != request.account_isolation_id):
            raise ValueError("result does not match request content / isolated account")
        actual[result.key] = result
    if actual.keys() != expected.keys():
        raise ValueError("incomplete candidate evaluation batch")
    return tuple(actual[k] for k in sorted(expected))


class EvaluationBudget:
    """Local fail-closed dispatch cap; NOT a substitute for L6 SelectionLedger."""
    def __init__(self, maximum: int, *,
                 reserve_requests: Callable[[tuple[CandidateRequest, ...]], None] | None = None,
                 verify_requests: Callable[[tuple[tuple[str, str, int, str, str, int], ...]], None] | None = None,
                 reservation_id: str | None = None):
        if type(maximum) is not int or maximum < 1:
            raise ValueError("positive evaluation budget required")
        if reserve_requests is not None and not callable(reserve_requests):
            raise TypeError("reserve_requests must be a callable or None")
        if verify_requests is not None and not callable(verify_requests):
            raise TypeError("verify_requests must be a callable or None")
        if ((reserve_requests is None) != (verify_requests is None)
                or (reserve_requests is None) != (reservation_id is None)
                or (reservation_id is not None and
                    (not isinstance(reservation_id, str) or len(reservation_id) != 64
                     or any(char not in _HEX for char in reservation_id)))):
            raise ValueError("durable reservation callback and SHA256 scope ID must be paired")
        self.maximum = maximum
        self._reserve_requests = reserve_requests
        self._verify_requests = verify_requests
        self.reservation_id = reservation_id
        self._dispatched: set[tuple[str, str, int, str, str, int]] = set()
        self._lock = Lock()

    @property
    def consumed(self) -> int:
        with self._lock:
            return len(self._dispatched)

    def checkpoint_state(self) -> dict[str, object]:
        with self._lock:
            return {"maximum": self.maximum, "dispatched": tuple(sorted(self._dispatched)),
                    "reservation_id": self.reservation_id}

    def restore_checkpoint_state(self, state: object) -> None:
        if (not isinstance(state, dict)
                or set(state) != {"maximum", "dispatched", "reservation_id"}
                or state["maximum"] != self.maximum or type(state["maximum"]) is not int
                or not isinstance(state["dispatched"], tuple)
                or state["reservation_id"] != self.reservation_id):
            raise ValueError("evaluation budget checkpoint identity/schema mismatch")
        keys = state["dispatched"]
        if len(keys) > self.maximum:
            raise ValueError("invalid dispatched request keys in evaluation budget checkpoint")
        for key in keys:
            if (not isinstance(key, tuple) or len(key) != 6
                    or any(not isinstance(key[index], str) or not key[index].strip()
                           for index in (0, 1, 3, 4))
                    or type(key[2]) is not int or key[2] < 0
                    or type(key[5]) is not int or not 0 <= key[5] < 2 ** 32):
                raise ValueError("malformed logical request identity in budget checkpoint")
        if len(set(keys)) != len(keys):
            raise ValueError("duplicate logical request identity in budget checkpoint")
        with self._lock:
            if self._dispatched:
                raise RuntimeError("evaluation budget restore requires a fresh budget")
            if self._verify_requests is not None:
                self._verify_requests(tuple(sorted(keys)))
            self._dispatched = set(keys)

    def reserve(self, requests: Iterable[CandidateRequest]) -> int:
        batch = list(requests)
        if not batch or any(not isinstance(r, CandidateRequest) for r in batch):
            raise ValueError("evaluation reservation requires valid requests")
        keys = [r.key for r in batch]
        if len(set(keys)) != len(keys):
            raise ValueError("duplicate candidate identity within batch")
        with self._lock:
            if any(key in self._dispatched for key in keys):
                raise RuntimeError("candidate evaluation already dispatched")
            if len(self._dispatched) + len(keys) > self.maximum:
                raise RuntimeError("candidate evaluation budget exceeded")
            # Persist the logical requests before any worker can execute. A
            # failed process/dispatch therefore consumes the protocol budget
            # and cannot be retried under a fresh local EvaluationBudget.
            if self._reserve_requests is not None:
                self._reserve_requests(tuple(batch))
            self._dispatched.update(keys)
            return len(self._dispatched)


@dataclass(frozen=True)
class CandidateAggregate:
    candidate_id: str
    generation: int
    episodes: int
    mean_fitness: float
    evidence_hash: str
    candidate_kind: str
    candidate_schema_hash: str


def dispatch_generation(
    requests: Iterable[CandidateRequest],
    *, dispatch: "Callable[[tuple[CandidateRequest, ...]], Iterable[CandidateResult]]",
    budget: EvaluationBudget,
    candidate_spec: ActorGenomeSpec | SoftRiskSpace,
    actor_signature: str | None = None,
) -> tuple[CandidateAggregate, ...]:
    """Candidate-level lane adapter; external L6 owns asynchronous execution.

    Rejects unbalanced candidate/episode matrices and any missing or foreign
    response. Dispatch reserves every attempted evaluation (no free retries).
    """
    batch = tuple(requests)
    if (not batch or not callable(dispatch) or not isinstance(budget, EvaluationBudget)
            or not isinstance(candidate_spec, (ActorGenomeSpec, SoftRiskSpace))):
        raise ValueError("invalid ES compute lane dependencies")
    if isinstance(candidate_spec, ActorGenomeSpec):
        expected_kind = "actor"
        expected_signature = candidate_spec.signature
        # The search vector schema and the actor contract consumed by R1 are
        # distinct identities.  E1 pins the latter in actor_signature while
        # candidate_schema_hash describes the L5 genome layout.
        expected_actor_signature = (expected_signature if actor_signature is None
                                    else actor_signature)
        _check_hash(expected_actor_signature)
        expected_size = candidate_spec.size
    else:
        expected_kind = "soft_risk"
        expected_signature = candidate_spec.signature
        if actor_signature is None:
            raise ValueError("E2 SoftRisk evaluation must pin the frozen R1 actor signature")
        _check_hash(actor_signature)
        expected_actor_signature = actor_signature
        expected_size = len(candidate_spec.dimensions)
    if len({r.key for r in batch}) != len(batch):
        raise ValueError("duplicate candidate/episode request identity")
    if len({r.account_isolation_id for r in batch}) != len(batch):
        raise ValueError("ComputeProvider must isolate every candidate account")
    if any(r.actor_signature != expected_actor_signature
           or r.candidate_kind != expected_kind
           or r.candidate_schema_hash != expected_signature
           or len(r.genome) != expected_size
           or not np.isfinite(np.asarray(r.genome, dtype=np.float32)).all() for r in batch):
        raise ValueError("ES candidate genome violates frozen actor/risk schema")
    if isinstance(candidate_spec, SoftRiskSpace):
        for request in batch:
            candidate_spec.decode(np.asarray(request.genome, dtype=np.float32))
    first = batch[0]
    if any((r.run_id, r.attempt_id, r.generation, r.actor_signature, r.split) !=
           (first.run_id, first.attempt_id, first.generation, first.actor_signature, first.split)
           for r in batch):
        raise ValueError("mixed run/generation/signature in candidate batch")
    candidate_ids = {r.candidate_id for r in batch}
    episodes = {r.episode_id for r in batch}
    if len(batch) != len(candidate_ids) * len(episodes):
        raise ValueError("candidate/episode matrix incomplete or duplicated")
    expected_seeds: dict[str, int] = {}
    expected_genomes: dict[str, tuple[float, ...]] = {}
    for r in batch:
        if r.episode_id in expected_seeds and expected_seeds[r.episode_id] != r.seed:
            raise ValueError("CRN: candidates must share the same scenario seed")
        expected_seeds[r.episode_id] = r.seed
        if r.candidate_id in expected_genomes and expected_genomes[r.candidate_id] != r.genome:
            raise ValueError("candidate genome changed across scenarios")
        expected_genomes[r.candidate_id] = r.genome
    budget.reserve(batch)
    raw_results = tuple(dispatch(batch))
    ordered = reconcile(batch, raw_results)
    output = []
    for candidate_id in sorted(candidate_ids):
        group = [r for r in ordered if r.candidate_id == candidate_id]
        score = math.fsum(r.fitness for r in group) / len(group)
        if not math.isfinite(score):
            raise OverflowError("nonfinite across-scenario aggregate fitness")
        evidence = _sha_json(tuple((r.episode_id, r.request_hash, r.evidence_hash, r.fitness)
                                   for r in group))
        output.append(CandidateAggregate(candidate_id, first.generation, len(group), score,
                                         evidence, expected_kind, expected_signature))
    return tuple(output)


class CandidateBatchLane(Protocol):
    @property
    def runtime_signature(self) -> str: ...

    def evaluate(self, candidates: Sequence[tuple[Mapping[str, object], Mapping[str, object]]]) -> Mapping[str, object]: ...


class GenerationLaneEvaluator:
    """Turn one EvoTorch population into a complete, auditable CRN matrix."""

    def __init__(self, *, run_id: str, attempt_id: str, fold_id: str, root_seed: int,
                 episode_ids: tuple[str, ...], candidate_spec: ActorGenomeSpec | SoftRiskSpace,
                 dispatch: Callable[[tuple[CandidateRequest, ...]], Iterable[CandidateResult]],
                 budget: EvaluationBudget, dispatch_id: str,
                 actor_signature: str | None = None):
        if (not all(isinstance(value, str) and value.strip() for value in
                    (run_id, attempt_id, fold_id))
                or type(root_seed) is not int or root_seed < 0
                or not isinstance(episode_ids, tuple) or not episode_ids
                or any(not isinstance(value, str) or not value.strip() for value in episode_ids)
                or len(set(episode_ids)) != len(episode_ids)
                or not callable(dispatch) or not isinstance(budget, EvaluationBudget)):
            raise ValueError("invalid generation lane identity, scenarios, or dispatch")
        _check_hash(dispatch_id)
        if isinstance(candidate_spec, ActorGenomeSpec):
            if actor_signature is None:
                actor_signature = candidate_spec.signature
            else:
                _check_hash(actor_signature)
        elif isinstance(candidate_spec, SoftRiskSpace):
            if actor_signature is None:
                raise ValueError("E2 requires the frozen R1 actor signature")
            _check_hash(actor_signature)
        else:
            raise TypeError("unsupported candidate schema")
        self.run_id, self.attempt_id, self.fold_id = run_id, attempt_id, fold_id
        self.root_seed, self.episode_ids = root_seed, episode_ids
        self.candidate_spec, self.dispatch, self.budget = candidate_spec, dispatch, budget
        self.actor_signature = actor_signature
        self.dispatch_id = dispatch_id
        self._generation_evidence: list[tuple[int, int, int, str]] = []

    @property
    def signature(self) -> str:
        return _sha_json({
            "schema": 3,
            "run_id": self.run_id, "attempt_id": self.attempt_id, "fold_id": self.fold_id,
            "root_seed": self.root_seed, "episode_ids": self.episode_ids,
            "candidate_schema": self.candidate_spec.signature,
            "actor_signature": self.actor_signature, "dispatch_id": self.dispatch_id,
            "budget_maximum": self.budget.maximum,
            "budget_reservation_id": self.budget.reservation_id,
        })

    def checkpoint_state(self) -> dict[str, object]:
        return {
            "schema": 3,
            "signature": self.signature,
            "budget": self.budget.checkpoint_state(),
            "generation_evidence": tuple(self._generation_evidence),
        }

    def restore_checkpoint_state(self, state: object) -> None:
        if (not isinstance(state, dict)
                or set(state) != {"schema", "signature", "budget", "generation_evidence"}
                or state["schema"] != 3 or type(state["schema"]) is not int
                or state["signature"] != self.signature
                or not isinstance(state["generation_evidence"], tuple)):
            raise ValueError("generation evaluator checkpoint identity mismatch")
        evidence = state["generation_evidence"]
        previous_generation = -1
        request_total = 0
        for record in evidence:
            if (not isinstance(record, tuple) or len(record) != 4
                    or type(record[0]) is not int or record[0] != previous_generation + 1
                    or type(record[1]) is not int or record[1] < 1
                    or type(record[2]) is not int
                    or record[2] != record[1] * len(self.episode_ids)
                    or not isinstance(record[3], str) or len(record[3]) != 64
                    or any(char not in "0123456789abcdef" for char in record[3])):
                raise ValueError("malformed generation evaluation evidence checkpoint")
            previous_generation = record[0]
            request_total += record[2]
        budget_state = state["budget"]
        if (not isinstance(budget_state, dict)
                or not isinstance(budget_state.get("dispatched"), tuple)
                or len(budget_state["dispatched"]) != request_total):
            raise ValueError("generation evidence and reserved lane requests disagree")
        self.budget.restore_checkpoint_state(state["budget"])
        self._generation_evidence = list(evidence)

    @property
    def evidence_hash(self) -> str:
        """Digest all generation-level candidate and worker evidence in this attempt."""
        return _sha_json(tuple(self._generation_evidence))

    def __call__(self, candidates: np.ndarray, *, generation: int) -> np.ndarray:
        matrix = np.asarray(candidates)
        if (matrix.dtype != np.float32 or matrix.ndim != 2 or matrix.shape[0] < 1
                or not np.isfinite(matrix).all() or type(generation) is not int or generation < 0):
            raise ValueError("generation candidates must be a finite float32 matrix")
        expected_size = (self.candidate_spec.size if isinstance(self.candidate_spec, ActorGenomeSpec)
                         else len(self.candidate_spec.dimensions))
        if matrix.shape[1] != expected_size:
            raise ValueError("generation candidate dimension violates frozen schema")
        genomes = {f"g{generation:08d}-c{index:08d}": tuple(float(x) for x in row)
                   for index, row in enumerate(matrix)}
        requests = tuple(
            CandidateRequest(
                run_id=self.run_id, attempt_id=self.attempt_id, generation=generation,
                candidate_id=candidate_id, episode_id=episode_id,
                seed=scenario_seed(self.root_seed, self.fold_id, generation, episode_id),
                genome=genome, actor_signature=self.actor_signature or "",
                candidate_kind="actor" if isinstance(self.candidate_spec, ActorGenomeSpec) else "soft_risk",
                candidate_schema_hash=self.candidate_spec.signature,
                account_isolation_id="acct-" + _sha_json((self.run_id, self.attempt_id, generation,
                                                           candidate_id, episode_id)),
            )
            for candidate_id, genome in genomes.items()
            for episode_id in self.episode_ids
        )
        aggregates = dispatch_generation(
            requests, dispatch=self.dispatch, budget=self.budget,
            candidate_spec=self.candidate_spec,
            actor_signature=self.actor_signature,
        )
        scores = np.asarray([aggregate.mean_fitness for aggregate in aggregates], dtype=np.float32)
        if scores.shape != (matrix.shape[0],) or not np.isfinite(scores).all():
            raise ValueError("candidate aggregates do not match population")
        if generation != len(self._generation_evidence):
            raise RuntimeError("lane generation evidence must be recorded once in order")
        generation_hash = _sha_json(tuple(
            (aggregate.candidate_id, aggregate.episodes, aggregate.mean_fitness,
             aggregate.evidence_hash, aggregate.candidate_kind,
             aggregate.candidate_schema_hash)
            for aggregate in aggregates
        ))
        self._generation_evidence.append((generation, len(aggregates), len(requests), generation_hash))
        return scores


class WorkerLaneDispatcher:
    """L5 adapter for an injected L6 lane, without importing the L6 package.

    The payload builder is the only application-specific seam: it converts the
    frozen candidate request into the installed L3 evaluation function's JSON
    arguments. Worker output must contain finite ``fitness`` and a SHA256
    ``evidence_hash`` produced by that evaluation.
    """

    def __init__(self, lane: CandidateBatchLane,
                 payload_builder: Callable[[CandidateRequest], Mapping[str, object]],
                 *, payload_builder_id: str, fitness_spec: FitnessSpec,
                 tail_fraction: float):
        if not callable(getattr(lane, "evaluate", None)) or not callable(payload_builder):
            raise TypeError("worker lane and JSON payload builder are required")
        runtime_signature = getattr(lane, "runtime_signature", None)
        _check_hash(runtime_signature)
        _check_hash(payload_builder_id)
        if not isinstance(fitness_spec, FitnessSpec):
            raise TypeError("explicit frozen FitnessSpec is required")
        if (isinstance(tail_fraction, bool) or not isinstance(tail_fraction, (int, float))
                or not math.isfinite(tail_fraction) or not 0 < tail_fraction <= 1):
            raise ValueError("tail fraction must be finite and in (0,1]")
        self.lane, self.payload_builder = lane, payload_builder
        self._lane_runtime_signature = runtime_signature
        self.payload_builder_id = payload_builder_id
        self.fitness_spec, self.tail_fraction = fitness_spec, float(tail_fraction)

    @property
    def signature(self) -> str:
        return _sha_json(("worker-lane-dispatch-v3", self._lane_runtime_signature,
                          self.payload_builder_id,
                          tuple(vars(self.fitness_spec).items()), self.tail_fraction))

    def __call__(self, requests: tuple[CandidateRequest, ...]) -> tuple[CandidateResult, ...]:
        if self.lane.runtime_signature != self._lane_runtime_signature:
            raise RuntimeError("ES worker lane recipe changed after dispatcher construction")
        lane_requests = []
        lane_to_request: dict[str, CandidateRequest] = {}
        for request in requests:
            lane_id = "eval-" + _sha_json(request.key)
            lane_to_request[lane_id] = request
            key = {
                "run_id": request.run_id, "attempt_id": request.attempt_id,
                "env_id": 0, "episode_id": request.episode_id,
                "step_id": request.generation, "candidate_id": lane_id,
                "model_id": request.actor_signature,
                "schema_version": request.candidate_schema_hash,
                "dtype": "float32", "shape": [len(request.genome)],
            }
            payload = self.payload_builder(request)
            if not isinstance(payload, Mapping):
                raise TypeError("worker payload builder must return a JSON mapping")
            if "rles_context" in payload:
                raise ValueError("payload builder cannot override reserved rles_context")
            worker_payload = dict(payload)
            worker_payload["rles_context"] = {
                "version": 1, "request_hash": request.request_hash,
                "run_id": request.run_id, "attempt_id": request.attempt_id,
                "generation": request.generation, "candidate_id": request.candidate_id,
                "episode_id": request.episode_id, "seed": request.seed,
                "genome": list(request.genome), "actor_signature": request.actor_signature,
                "candidate_kind": request.candidate_kind,
                "candidate_schema_hash": request.candidate_schema_hash,
                "account_isolation_id": request.account_isolation_id,
                "split": request.split,
            }
            lane_requests.append((key, worker_payload))
        output = self.lane.evaluate(lane_requests)
        if not isinstance(output, Mapping) or set(output) != set(lane_to_request):
            raise ValueError("L6 response candidate set mismatch")
        results = []
        for lane_id, request in lane_to_request.items():
            value = output[lane_id]
            required = {"initial_equity", "final_equity", "step_log_returns", "drawdowns",
                        "constraint_violations", "evidence_hash", "request_hash",
                        "account_isolation_id"}
            if not isinstance(value, Mapping) or not required.issubset(value):
                raise ValueError("L6 output needs native episode facts, evidence, and echoed request/account identity")
            if (value["request_hash"] != request.request_hash
                    or value["account_isolation_id"] != request.account_isolation_id):
                raise ValueError("L6 output does not echo the dispatched request and isolated account")
            raw_returns, raw_drawdowns = value["step_log_returns"], value["drawdowns"]
            if not isinstance(raw_returns, list) or not isinstance(raw_drawdowns, list):
                raise ValueError("L6 episode facts must use JSON arrays")
            fitness = self.fitness_spec.score_episode(
                split=request.split,
                initial_equity=value["initial_equity"], final_equity=value["final_equity"],
                step_log_returns=raw_returns, drawdowns=raw_drawdowns,
                constraint_violations=value["constraint_violations"],
                tail_fraction=self.tail_fraction,
            )
            results.append(CandidateResult(
                run_id=request.run_id, attempt_id=request.attempt_id,
                generation=request.generation, candidate_id=request.candidate_id,
                episode_id=request.episode_id, seed=request.seed,
                request_hash=request.request_hash,
                account_isolation_id=request.account_isolation_id,
                fitness=fitness, evidence_hash=value["evidence_hash"],
            ))
        return tuple(results)
