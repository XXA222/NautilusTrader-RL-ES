"""Dependency-inverted runtime protocols. Concrete implementations live in L2-L6."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from typing import Protocol, runtime_checkable

from .dto import (
    AccountView, ActorGenomeSpec, ComputeLease, ComputeRequest, ComputeResponse,
    EngineAdvanceResult, EngineFingerprint, EpisodeSpec, ESConfig, ESState,
    ExecutionPlan, ExecutionSnapshot, InstrumentSpec, MarketEvent, ModelRef,
    PolicyAction, RiskDecision, RiskState, RLConfig, StepOutcome, TargetExposure, Observation,
    TerminalEvidence,
)


@runtime_checkable
class EngineSession(Protocol):
    def open(self, episode: EpisodeSpec) -> EngineFingerprint: ...
    def advance(self, plan: ExecutionPlan) -> EngineAdvanceResult: ...
    def snapshot(self) -> ExecutionSnapshot: ...
    def drain_and_close(self, terminal_policy: str) -> TerminalEvidence: ...


@runtime_checkable
class DataProvider(Protocol):
    def stream(self, episode: EpisodeSpec) -> Iterator[MarketEvent]: ...
    def dataset_hash(self) -> str: ...


@runtime_checkable
class ExecutionPlanner(Protocol):
    def plan(self, action: PolicyAction, account: AccountView, spec: InstrumentSpec) -> ExecutionPlan: ...


@runtime_checkable
class HardRiskGuard(Protocol):
    def apply(self, target: TargetExposure, state: RiskState) -> RiskDecision: ...


@runtime_checkable
class SoftRiskTransform(Protocol):
    def apply(self, target: TargetExposure, state: RiskState) -> TargetExposure: ...


@runtime_checkable
class VecEnvContract(Protocol):
    @property
    def num_envs(self) -> int: ...
    def reset(self) -> tuple[tuple[Observation, ...], tuple[Mapping[str, object], ...]]: ...
    def step(self, actions: tuple[PolicyAction, ...]) -> tuple[StepOutcome, ...]: ...
    def close(self) -> None: ...


@runtime_checkable
class RLProvider(Protocol):
    def train(self, vec_env: VecEnvContract, cfg: RLConfig, compute: ComputeLease) -> ModelRef: ...
    def load(self, ref: ModelRef) -> object: ...


@runtime_checkable
class FitnessEvaluator(Protocol):
    def evaluate(self, actor_hashes: Iterable[str]) -> Mapping[str, float]: ...


@runtime_checkable
class ESProvider(Protocol):
    def evolve(self, genome: ActorGenomeSpec, evaluator: FitnessEvaluator, cfg: ESConfig) -> ESState: ...
    def resume(self, state: ESState) -> ESState: ...


@runtime_checkable
class ComputeProvider(Protocol):
    def submit(self, request: ComputeRequest) -> None: ...
    def receive(self, request: ComputeRequest) -> ComputeResponse: ...
    def close(self) -> None: ...


@runtime_checkable
class ProviderFactory(Protocol):
    """The entry point symbol is a factory descriptor, never a concrete model import."""

    kind: str
    contract_version: str
    def create(self, settings: Mapping[str, object]) -> object: ...
