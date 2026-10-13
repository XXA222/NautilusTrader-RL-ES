"""Engine-neutral, frozen value carriers only. All quantities are explicit Decimal.

No order calculation, matching, risk mathematics or third-party-engine object occurs here.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum

from .frozen import FrozenFields
from .schema import CONTRACT_VERSION, require_current_version
from .validation import exact_decimal_multiple, require_decimal, require_float, require_hash, require_name, require_ns


class DataQualityGrade(StrEnum):
    A = "DQ-A"
    B = "DQ-B"
    C = "DQ-C"
    D = "DQ-D"


class TerminalPolicy(StrEnum):
    FORCE_FLATTEN_WITH_COST = "force_flatten_with_cost"
    MARK_TO_MARKET = "mark_to_market"


class OrderType(StrEnum):
    MARKET_LIKE = "market_like"
    LIMIT = "limit"


class PendingOrderPolicy(StrEnum):
    NONE = "none"
    CANCEL_REPLACE = "cancel_replace"
    WAIT = "wait"
    FAIL = "fail"


class TimeInForce(StrEnum):
    GTC = "GTC"
    IOC = "IOC"
    FOK = "FOK"


@dataclass(frozen=True, slots=True)
class MarketEvent:
    event_id: str
    ts_event_ns: int
    ts_init_ns: int
    available_at_ns: int
    source_hash: str
    quality_grade: DataQualityGrade
    event_type: str
    payload: FrozenFields
    schema_version: str = CONTRACT_VERSION

    def __post_init__(self) -> None:
        require_current_version(self.schema_version)
        require_name(self.event_id, "event_id")
        require_name(self.event_type, "event_type")
        for field in ("ts_event_ns", "ts_init_ns", "available_at_ns"):
            require_ns(getattr(self, field), field)
        if self.available_at_ns < max(self.ts_event_ns, self.ts_init_ns):
            raise ValueError("available_at_ns cannot precede event or initialization time")
        require_hash(self.source_hash, "source_hash")
        if type(self.quality_grade) is not DataQualityGrade:
            raise TypeError("quality_grade must be DataQualityGrade")
        if type(self.payload) is not FrozenFields:
            raise TypeError("payload must be FrozenFields")

    def require_available(self, decision_ts_ns: int) -> None:
        require_ns(decision_ts_ns, "decision_ts_ns")
        if self.available_at_ns > decision_ts_ns:
            raise ValueError("future event unavailable at decision barrier")
        if self.quality_grade is DataQualityGrade.D:
            raise ValueError("DQ-D events are unusable for training/evaluation")


@dataclass(frozen=True, slots=True)
class Observation:
    decision_ts_ns: int
    field_names: tuple[str, ...]
    field_units: tuple[str, ...]
    vector: tuple[float, ...]
    account_state: FrozenFields
    risk_state: FrozenFields
    execution_state: FrozenFields
    prev_action: Decimal
    available_through_ns: int
    schema_version: str = CONTRACT_VERSION

    def __post_init__(self) -> None:
        require_current_version(self.schema_version)
        require_ns(self.decision_ts_ns, "decision_ts_ns")
        require_ns(self.available_through_ns, "available_through_ns")
        if self.available_through_ns > self.decision_ts_ns:
            raise ValueError("observation contains future-available data")
        if any(type(x) is not tuple for x in (self.field_names, self.field_units, self.vector)):
            raise TypeError("feature name/unit/vector must be tuples")
        if len(self.field_names) != len(self.field_units) or len(self.vector) != len(self.field_names):
            raise ValueError("feature shape/name/unit mismatch")
        if not self.field_names or len(set(self.field_names)) != len(self.field_names):
            raise ValueError("feature names must be nonempty and unique")
        for name, unit in zip(self.field_names, self.field_units, strict=True):
            require_name(name, "feature name")
            require_name(unit, "feature unit")
        for idx, x in enumerate(self.vector):
            require_float(x, f"vector[{idx}]")
        for field in ("account_state", "risk_state", "execution_state"):
            if type(getattr(self, field)) is not FrozenFields:
                raise TypeError(f"{field} must be FrozenFields")
        require_decimal(self.prev_action, "prev_action")


@dataclass(frozen=True, slots=True)
class PolicyAction:
    decision_id: str
    policy_id: str
    raw_action: Decimal
    logprob_ref: str | None = None
    schema_version: str = CONTRACT_VERSION

    def __post_init__(self) -> None:
        require_current_version(self.schema_version)
        require_name(self.decision_id, "decision_id")
        require_name(self.policy_id, "policy_id")
        require_decimal(self.raw_action, "raw_action")
        if not Decimal("-1") <= self.raw_action <= Decimal("1"):
            raise ValueError("Box([-1,1]) raw action outside bounds")
        if self.logprob_ref is not None:
            require_name(self.logprob_ref, "logprob_ref")


# L1 work order calls it ActionIntent; the v0.4.0 technical baseline calls it PolicyAction.
# One wire contract and one class identity eliminate divergent consumers.
ActionIntent = PolicyAction


@dataclass(frozen=True, slots=True)
class TargetExposure:
    decision_id: str
    requested: Decimal
    soft_risk_adjusted: Decimal
    hard_risk_capped: Decimal
    unit: str
    schema_version: str = CONTRACT_VERSION

    def __post_init__(self) -> None:
        require_current_version(self.schema_version)
        require_name(self.decision_id, "decision_id")
        require_name(self.unit, "unit")
        for k in ("requested", "soft_risk_adjusted", "hard_risk_capped"):
            require_decimal(getattr(self, k), k)
        # Does not perform risk clipping: semantics belong exclusively to rles-risk.


@dataclass(frozen=True, slots=True)
class ExecutionPlan:
    decision_id: str
    plan_id: str
    delta_qty: Decimal
    quantity_unit: str
    quantity_increment: Decimal
    order_type: OrderType
    tif: TimeInForce
    cancel_policy: PendingOrderPolicy
    latency_model_id: str
    instrument_id: str
    execution_spec_hash: str
    idempotency_key: str
    action: str
    reason: str
    cancel_order_ids: tuple[str, ...] = ()
    requires_cancel_ack: bool = False
    schema_version: str = CONTRACT_VERSION

    def __post_init__(self) -> None:
        require_current_version(self.schema_version)
        for key in ("decision_id", "plan_id", "quantity_unit", "latency_model_id",
                    "instrument_id", "idempotency_key"):
            require_name(getattr(self, key), key)
        if (type(self.reason) is not str or not self.reason or len(self.reason) > 256 or
                any(ord(char) < 32 for char in self.reason)):
            raise ValueError("reason must be bounded nonempty text")
        require_hash(self.execution_spec_hash, "execution_spec_hash")
        require_decimal(self.delta_qty, "delta_qty")
        require_decimal(self.quantity_increment, "quantity_increment")
        if self.quantity_increment <= 0:
            raise ValueError("quantity_increment must be positive")
        if not exact_decimal_multiple(self.delta_qty, self.quantity_increment):
            raise ValueError("delta_qty must be quantized to instrument increment")
        if type(self.order_type) is not OrderType or type(self.tif) is not TimeInForce:
            raise TypeError("order_type / tif must be enum")
        if type(self.cancel_policy) is not PendingOrderPolicy:
            raise TypeError("cancel_policy must be enum")
        if self.action not in {"SUBMIT", "CANCEL", "WAIT", "NOOP"}:
            raise ValueError("unknown execution action")
        if (type(self.cancel_order_ids) is not tuple or
                any(type(order_id) is not str or not order_id for order_id in self.cancel_order_ids) or
                len(set(self.cancel_order_ids)) != len(self.cancel_order_ids) or
                type(self.requires_cancel_ack) is not bool):
            raise ValueError("invalid cancel/ACK identity")
        if self.action == "SUBMIT":
            if (self.delta_qty == 0 or self.cancel_order_ids or self.requires_cancel_ack or
                    self.cancel_policy is not PendingOrderPolicy.NONE):
                raise ValueError("submit cannot also cancel or wait")
        elif self.action == "CANCEL":
            if (self.delta_qty != 0 or not self.cancel_order_ids or not self.requires_cancel_ack or
                    self.cancel_policy is not PendingOrderPolicy.CANCEL_REPLACE):
                raise ValueError("cancel requires distinct live IDs and ACK")
        elif (self.delta_qty != 0 or self.cancel_order_ids or self.requires_cancel_ack or
              self.cancel_policy is not (PendingOrderPolicy.WAIT if self.action == "WAIT"
                                         else PendingOrderPolicy.NONE)):
            raise ValueError("non-trading plan contains order side effects")


@dataclass(frozen=True, slots=True)
class ExecutionSnapshot:
    snapshot_id: str
    decision_ts_ns: int
    orders: tuple[FrozenFields, ...]
    fills: tuple[FrozenFields, ...]
    fees: Decimal
    funding: Decimal
    positions: tuple[FrozenFields, ...]
    margin: Decimal
    equity: Decimal
    mark: Decimal
    currency: str
    fact_source: str
    schema_version: str = CONTRACT_VERSION

    def __post_init__(self) -> None:
        require_current_version(self.schema_version)
        require_name(self.snapshot_id, "snapshot_id")
        require_name(self.currency, "currency")
        require_name(self.fact_source, "fact_source")
        require_ns(self.decision_ts_ns, "decision_ts_ns")
        for k in ("fees", "funding", "margin", "equity", "mark"):
            require_decimal(getattr(self, k), k)
        for k in ("orders", "fills", "positions"):
            value = getattr(self, k)
            if type(value) is not tuple or any(type(row) is not FrozenFields for row in value):
                raise TypeError(f"{k} must be tuple[FrozenFields,...]")
        # Only the adapter can assert fact_source authority; DTO records its provenance.


@dataclass(frozen=True, slots=True)
class StepOutcome:
    obs_next: Observation
    reward: Decimal
    terminated: bool
    truncated: bool
    executed_action: TargetExposure
    evidence_hash: str
    barrier_ts_ns: int
    schema_version: str = CONTRACT_VERSION

    def __post_init__(self) -> None:
        require_current_version(self.schema_version)
        if type(self.obs_next) is not Observation or type(self.executed_action) is not TargetExposure:
            raise TypeError("step outcome expects DTOs")
        require_decimal(self.reward, "reward")
        require_hash(self.evidence_hash, "evidence_hash")
        require_ns(self.barrier_ts_ns, "barrier_ts_ns")
        if type(self.terminated) is not bool or type(self.truncated) is not bool:
            raise TypeError("terminated/truncated must be bool")
        if self.terminated and self.truncated:
            raise ValueError("terminated and truncated are mutually exclusive")
        if self.obs_next.decision_ts_ns != self.barrier_ts_ns:
            raise ValueError("next observation must be at completed event barrier")


@dataclass(frozen=True, slots=True)
class ComputeRequest:
    run_id: str
    env_id: str
    episode_id: str
    step_id: int
    model_id: str
    observation_schema_hash: str
    dtype: str
    shape: tuple[int, ...]
    payload_hash: str
    schema_version: str = CONTRACT_VERSION

    def __post_init__(self) -> None:
        require_current_version(self.schema_version)
        for k in ("run_id", "env_id", "episode_id", "model_id", "dtype"):
            require_name(getattr(self, k), k)
        if type(self.step_id) is not int or self.step_id < 0:
            raise ValueError("step_id must be a nonnegative integer")
        for k in ("observation_schema_hash", "payload_hash"):
            require_hash(getattr(self, k), k)
        if type(self.shape) is not tuple or not self.shape or any(type(x) is not int or x <= 0 for x in self.shape):
            raise ValueError("shape must be tuple of positive ints")

    @property
    def correlation_key(self) -> tuple[str, str, str, int, str, str, str, tuple[int, ...]]:
        return (self.run_id, self.env_id, self.episode_id, self.step_id,
                self.model_id, self.observation_schema_hash, self.dtype, self.shape)


@dataclass(frozen=True, slots=True)
class ComputeResponse:
    request: ComputeRequest
    result_hash: str
    result_ref: str
    schema_version: str = CONTRACT_VERSION

    def __post_init__(self) -> None:
        require_current_version(self.schema_version)
        if type(self.request) is not ComputeRequest:
            raise TypeError("response.request must be original ComputeRequest")
        require_hash(self.result_hash, "result_hash")
        require_name(self.result_ref, "result_ref")

    def require_match(self, original: ComputeRequest) -> None:
        if self.request != original:
            raise ValueError("compute response request/correlation mismatch")


@dataclass(frozen=True, slots=True)
class EpisodeSpec:
    episode_id: str
    fold_id: str
    score_start_ns: int
    score_end_ns: int
    terminal_policy: TerminalPolicy
    dataset_hash: str
    schema_version: str = CONTRACT_VERSION

    def __post_init__(self) -> None:
        require_current_version(self.schema_version)
        require_name(self.episode_id, "episode_id")
        require_name(self.fold_id, "fold_id")
        require_ns(self.score_start_ns, "score_start_ns")
        require_ns(self.score_end_ns, "score_end_ns")
        if self.score_end_ns <= self.score_start_ns:
            raise ValueError("score window must be nonempty")
        if type(self.terminal_policy) is not TerminalPolicy:
            raise TypeError("terminal policy must be enum")
        require_hash(self.dataset_hash, "dataset_hash")


@dataclass(frozen=True, slots=True)
class EngineFingerprint:
    adapter_hash: str
    engine_hash: str
    schema_version: str = CONTRACT_VERSION

    def __post_init__(self) -> None:
        require_current_version(self.schema_version)
        require_hash(self.adapter_hash, "adapter_hash")
        require_hash(self.engine_hash, "engine_hash")


@dataclass(frozen=True, slots=True)
class EngineAdvanceResult:
    barrier_ts_ns: int
    event_trace_hash: str
    schema_version: str = CONTRACT_VERSION

    def __post_init__(self) -> None:
        require_current_version(self.schema_version)
        require_ns(self.barrier_ts_ns, "barrier_ts_ns")
        require_hash(self.event_trace_hash, "event_trace_hash")


@dataclass(frozen=True, slots=True)
class TerminalEvidence:
    policy: TerminalPolicy
    settlement_hash: str
    schema_version: str = CONTRACT_VERSION

    def __post_init__(self) -> None:
        require_current_version(self.schema_version)
        if type(self.policy) is not TerminalPolicy:
            raise TypeError("invalid terminal policy")
        require_hash(self.settlement_hash, "settlement_hash")


@dataclass(frozen=True, slots=True)
class InstrumentSpec:
    instrument_id: str
    quantity_increment: Decimal
    quantity_unit: str
    price_increment: Decimal
    price_unit: str
    spec_hash: str
    schema_version: str = CONTRACT_VERSION

    def __post_init__(self) -> None:
        require_current_version(self.schema_version)
        for k in ("instrument_id", "quantity_unit", "price_unit"):
            require_name(getattr(self, k), k)
        for k in ("quantity_increment", "price_increment"):
            v = getattr(self, k)
            require_decimal(v, k)
            if v <= 0:
                raise ValueError(f"{k} must be positive")
        require_hash(self.spec_hash, "spec_hash")


@dataclass(frozen=True, slots=True)
class AccountView:
    snapshot: ExecutionSnapshot
    schema_version: str = CONTRACT_VERSION

    def __post_init__(self) -> None:
        require_current_version(self.schema_version)
        if type(self.snapshot) is not ExecutionSnapshot:
            raise TypeError("AccountView must contain ExecutionSnapshot")


@dataclass(frozen=True, slots=True)
class RiskState:
    account: AccountView
    risk_features: FrozenFields
    schema_version: str = CONTRACT_VERSION

    def __post_init__(self) -> None:
        require_current_version(self.schema_version)
        if type(self.account) is not AccountView or type(self.risk_features) is not FrozenFields:
            raise TypeError("RiskState requires AccountView and FrozenFields")


@dataclass(frozen=True, slots=True)
class RiskDecision:
    capped: TargetExposure
    reason_code: str
    schema_version: str = CONTRACT_VERSION

    def __post_init__(self) -> None:
        require_current_version(self.schema_version)
        if type(self.capped) is not TargetExposure:
            raise TypeError("RiskDecision.capped must be TargetExposure")
        require_name(self.reason_code, "reason_code")


@dataclass(frozen=True, slots=True)
class ActorGenomeSpec:
    actor_signature_hash: str
    observation_schema_hash: str
    action_schema_hash: str
    parameter_count: int
    parameter_order_hash: str
    schema_version: str = CONTRACT_VERSION

    def __post_init__(self) -> None:
        require_current_version(self.schema_version)
        for k in ("actor_signature_hash", "observation_schema_hash", "action_schema_hash", "parameter_order_hash"):
            require_hash(getattr(self, k), k)
        if type(self.parameter_count) is not int or self.parameter_count <= 0:
            raise ValueError("parameter_count must be a positive integer")


@dataclass(frozen=True, slots=True)
class ModelRef:
    model_id: str
    artifact_hash: str
    actor_signature_hash: str
    schema_version: str = CONTRACT_VERSION

    def __post_init__(self) -> None:
        require_current_version(self.schema_version)
        require_name(self.model_id, "model_id")
        require_hash(self.artifact_hash, "artifact_hash")
        require_hash(self.actor_signature_hash, "actor_signature_hash")


@dataclass(frozen=True, slots=True)
class ESState:
    state_id: str
    artifact_hash: str
    schema_version: str = CONTRACT_VERSION

    def __post_init__(self) -> None:
        require_current_version(self.schema_version)
        require_name(self.state_id, "state_id")
        require_hash(self.artifact_hash, "artifact_hash")


@dataclass(frozen=True, slots=True)
class ComputeLease:
    lease_id: str
    device: str
    schema_version: str = CONTRACT_VERSION

    def __post_init__(self) -> None:
        require_current_version(self.schema_version)
        require_name(self.lease_id, "lease_id")
        require_name(self.device, "device")


@dataclass(frozen=True, slots=True)
class RLConfig:
    algorithm: str
    parameters: FrozenFields
    schema_version: str = CONTRACT_VERSION

    def __post_init__(self) -> None:
        require_current_version(self.schema_version)
        if self.algorithm not in ("ppo", "sac", "td3", "ddpg"):
            raise ValueError("algorithm must be PPO/SAC/TD3/DDPG")
        if type(self.parameters) is not FrozenFields:
            raise TypeError("parameters must be FrozenFields")


@dataclass(frozen=True, slots=True)
class ESConfig:
    algorithm: str
    parameters: FrozenFields
    schema_version: str = CONTRACT_VERSION

    def __post_init__(self) -> None:
        require_current_version(self.schema_version)
        if self.algorithm not in ("pgpe", "cma_es"):
            raise ValueError("algorithm must be PGPE/CMA-ES")
        if type(self.parameters) is not FrozenFields:
            raise TypeError("parameters must be FrozenFields")
