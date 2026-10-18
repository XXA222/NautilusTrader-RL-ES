"""Deterministic exposure-to-order planning (no matching or accounting)."""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_DOWN
from hashlib import sha256
from typing import Any, Callable

from .spec import ResolvedExecutionConfig, dec


class PlanningBlocked(RuntimeError):
    """Fail-closed when a valid, evidence-backed order plan is impossible."""


@dataclass(frozen=True, slots=True)
class InstrumentRules:
    instrument_id: str
    qty_step: Decimal
    min_qty: Decimal
    min_notional: Decimal
    contract_multiplier: Decimal
    reduce_only_min_notional_exempt: bool = False

    def __post_init__(self) -> None:
        if not self.instrument_id or not self.instrument_id.strip():
            raise ValueError("instrument_id required")
        for name in ("qty_step", "min_qty", "min_notional", "contract_multiplier"):
            value = dec(getattr(self, name))
            if value < 0 or (name in {"qty_step", "contract_multiplier"} and value == 0):
                raise ValueError(f"invalid instrument {name}")
            object.__setattr__(self, name, value)
        if self.min_qty and self.min_qty % self.qty_step:
            raise ValueError("minimum quantity must align to quantity step")
        if type(self.reduce_only_min_notional_exempt) is not bool:
            raise TypeError("reduce-only minimum-notional capability must be bool")


@dataclass(frozen=True, slots=True)
class PendingOrder:
    order_id: str
    signed_leaves_qty: Decimal
    status: str

    def __post_init__(self) -> None:
        if not self.order_id or self.status not in {
            "SUBMITTED", "ACCEPTED", "PARTIALLY_FILLED", "PENDING_CANCEL", "PENDING_UPDATE",
        }:
            raise ValueError("pending order snapshot must have a live state and ID")
        object.__setattr__(self, "signed_leaves_qty", dec(self.signed_leaves_qty))


@dataclass(frozen=True, slots=True)
class PlanDecision:
    """L3-local internal result; L1 owns the serialized ExecutionPlan."""
    decision_id: str
    idempotency_key: str
    instrument_id: str
    target_exposure: Decimal
    target_qty: Decimal
    delta_qty: Decimal
    quantity_increment: Decimal
    cancel_order_ids: tuple[str, ...]
    requires_cancel_ack: bool
    action: str
    reason: str
    execution_spec_hash: str


class TargetQuantityPlanner:
    def __init__(self, config: ResolvedExecutionConfig) -> None:
        if not isinstance(config, ResolvedExecutionConfig):
            raise TypeError("resolved execution configuration required")
        self.config = config

    @staticmethod
    def _truncate_toward_zero(qty: Decimal, step: Decimal) -> Decimal:
        return (abs(qty) / step).to_integral_value(rounding=ROUND_DOWN) * step * (-1 if qty < 0 else 1)

    def plan(
        self, *, decision_id: str, instrument: InstrumentRules,
        target_exposure: Decimal, risk_budget_notional: Decimal,
        equity: Decimal, mark_price: Decimal, current_qty: Decimal,
        pending: tuple[PendingOrder, ...], terminal: bool = False, force_reduce: bool = False,
    ) -> PlanDecision:
        if not isinstance(decision_id, str) or not decision_id:
            raise ValueError("decision ID required")
        if not isinstance(instrument, InstrumentRules) or not isinstance(pending, tuple):
            raise TypeError("frozen instrument rules and pending order tuple required")
        if type(terminal) is not bool:
            raise TypeError("terminal must be bool")
        if type(force_reduce) is not bool:
            raise TypeError("force_reduce must be bool")
        target_exposure = dec(target_exposure)
        if not -1 <= target_exposure <= 1:
            raise PlanningBlocked("hard-capped exposure must be within [-1,1]")
        if terminal and target_exposure != 0:
            raise PlanningBlocked("terminal plan may only target flat exposure")
        budget = dec(risk_budget_notional, positive=True)
        account_equity = dec(equity)
        if account_equity < 0 or (account_equity == 0 and not (terminal or force_reduce)):
            raise PlanningBlocked("non-positive equity only permits mandatory reduction")
        mark = dec(mark_price, positive=True)
        current = dec(current_qty)
        if current % instrument.qty_step:
            raise PlanningBlocked("native current quantity incompatible with historical step rule")
        ids = [p.order_id for p in pending]
        if any(not isinstance(p, PendingOrder) for p in pending) or len(ids) != len(set(ids)):
            raise PlanningBlocked("duplicate or invalid native open order")
        target_qty = self._truncate_toward_zero(
            target_exposure * budget / (mark * instrument.contract_multiplier), instrument.qty_step,
        )
        # Do not create a target that can never be reduced or flattened under
        # the instrument's own minimums. A minimum-sized prior position could
        # otherwise be partially reduced into unexecutable dust, making the
        # terminal flatten fail later in the same episode.
        target_notional = abs(target_qty * mark * instrument.contract_multiplier)
        if target_qty != 0 and (abs(target_qty) < instrument.min_qty
                                or target_notional < instrument.min_notional):
            target_qty = Decimal(0)
        delta = target_qty - current
        policy_flat = self.config.zero_target_policy == "close" and target_exposure == 0
        if force_reduce and (target_qty * current < 0 or abs(target_qty) >= abs(current)):
            raise PlanningBlocked("forced hard-risk action must strictly reduce the existing position")
        key = "action-" + sha256(
            f"{decision_id}|{instrument.instrument_id}|{self.config.fingerprint()}|terminal={terminal}".encode()
        ).hexdigest()
        def outcome(action: str, reason: str, *, quantity: Decimal = Decimal(0),
                    cancels: tuple[str, ...] = (), ack: bool = False) -> PlanDecision:
            return PlanDecision(
                decision_id=decision_id, idempotency_key=key,
                instrument_id=instrument.instrument_id, target_exposure=target_exposure,
                target_qty=target_qty, delta_qty=quantity, cancel_order_ids=cancels,
                quantity_increment=instrument.qty_step,
                requires_cancel_ack=ack, action=action, reason=reason,
                execution_spec_hash=self.config.fingerprint(),
            )

        if pending:
            if self.config.pending_order_policy == "fail" and not (terminal or force_reduce or policy_flat):
                raise PlanningBlocked("open native orders present")
            if self.config.pending_order_policy == "wait" and not (terminal or force_reduce or policy_flat):
                return outcome("WAIT", "open orders must resolve before new target")
            if any(p.status == "PENDING_CANCEL" for p in pending):
                return outcome("WAIT", "await cancel acknowledgments before resnapshot")
            return outcome("CANCEL", "cancel then resnapshot and replan after ACK",
                           cancels=tuple(sorted(ids)), ack=True)

        if terminal and current == 0:
            return outcome("NOOP", "native account already flat")
        if not (terminal or force_reduce or policy_flat):
            exposure_now = current * mark * instrument.contract_multiplier / budget
            if abs(target_exposure - exposure_now) < self.config.deadband:
                return outcome("NOOP", "deadband")
        if delta == 0:
            return outcome("NOOP", "quantity rounded to no change")
        # Binance USDⓈ-M permits Reduce-Only orders below the symbol min-notional.
        # The exception is opt-in per dated InstrumentRules; quantity and step
        # constraints still apply, and the native submitter marks mandatory reductions
        # reduce-only. Other venues remain fail-closed without explicit evidence.
        notional = abs(delta * mark * instrument.contract_multiplier)
        below_min_qty = abs(delta) < instrument.min_qty
        below_min_notional = notional < instrument.min_notional
        reduce_only_close = (terminal or force_reduce or policy_flat) and instrument.reduce_only_min_notional_exempt
        if below_min_qty or (below_min_notional and not reduce_only_close):
            if terminal:
                raise PlanningBlocked("terminal close violates historical min quantity/notional")
            if force_reduce:
                raise PlanningBlocked("hard-risk reduction violates historical min quantity/notional")
            if policy_flat:
                return outcome("NOOP", "policy flat blocked by historical minimum quantity/notional")
            return outcome("NOOP", "below minimum executable quantity/notional")
        if delta % instrument.qty_step:
            raise PlanningBlocked("computed quantity violates instrument precision")
        reason = ("terminal_close" if terminal else "hard_risk_reduce" if force_reduce else
                  "policy_flat" if policy_flat else "rebalance")
        return outcome("SUBMIT", reason, quantity=delta)


def to_contract_plan(decision: PlanDecision, factory: Callable[..., Any], *,
                     latency_model_id: str, quantity_unit: str) -> Any:
    """Explicitly adapt an L3 decision to a reviewed L1 constructor; no shadow DTO."""
    if (not isinstance(decision, PlanDecision) or not callable(factory) or
            not latency_model_id or not quantity_unit):
        raise TypeError("reviewed L1 factory, plan, and latency identity required")
    from rles_contracts.dto import ExecutionPlan, OrderType, PendingOrderPolicy, TimeInForce

    policy = {
        "SUBMIT": PendingOrderPolicy.NONE,
        "CANCEL": PendingOrderPolicy.CANCEL_REPLACE,
        "WAIT": PendingOrderPolicy.WAIT,
        "NOOP": PendingOrderPolicy.NONE,
    }[decision.action]
    plan = factory(
        decision_id=decision.decision_id, plan_id=decision.idempotency_key,
        idempotency_key=decision.idempotency_key,
        instrument_id=decision.instrument_id, quantity_unit=quantity_unit,
        quantity_increment=decision.quantity_increment, delta_qty=decision.delta_qty,
        order_type=OrderType.MARKET_LIKE, tif=TimeInForce.GTC,
        cancel_policy=policy, cancel_order_ids=decision.cancel_order_ids,
        latency_model_id=latency_model_id, execution_spec_hash=decision.execution_spec_hash,
        action=decision.action, requires_cancel_ack=decision.requires_cancel_ack,
        reason=decision.reason,
    )
    if type(plan) is not ExecutionPlan:
        raise TypeError("execution factory must return the reviewed L1 ExecutionPlan")
    return plan
