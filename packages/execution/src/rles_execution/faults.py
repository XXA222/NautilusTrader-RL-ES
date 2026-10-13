"""Execution scenario fault gates. Never synthesize missing prices or account PnL."""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from .spec import dec


class MarketFault(RuntimeError):
    """A fail-closed historical market/execution fault."""


@dataclass(frozen=True, slots=True)
class FaultPolicy:
    outage_action: str
    shock_action: str
    max_price_jump_fraction: Decimal

    def __post_init__(self) -> None:
        if self.outage_action not in {"fail", "halt", "skip"}:
            raise ValueError("unknown outage action")
        if self.shock_action not in {"fail", "halt"}:
            raise ValueError("shock cannot be silently skipped")
        jump = dec(self.max_price_jump_fraction)
        if not 0 < jump <= 1:
            raise ValueError("price jump threshold must be in (0,1]")
        object.__setattr__(self, "max_price_jump_fraction", jump)


@dataclass(frozen=True, slots=True)
class FaultDecision:
    status: str
    reason: str
    severity: str


class ScenarioFaultGuard:
    """Pure fault decisions from independently proven L2-visible status."""
    def __init__(self, policy: FaultPolicy) -> None:
        if not isinstance(policy, FaultPolicy):
            raise TypeError("fault policy required")
        self.policy = policy

    def outage(self, *, missing: bool, open_positions: bool, pending_orders: bool) -> FaultDecision:
        if any(type(v) is not bool for v in (missing, open_positions, pending_orders)):
            raise TypeError("fault status must be explicit booleans")
        if not missing:
            return FaultDecision("CONTINUE", "complete native data", "none")
        action = self.policy.outage_action
        if action == "fail":
            raise MarketFault("L2 reported critical market outage")
        if action == "skip" and (open_positions or pending_orders):
            raise MarketFault("cannot skip outage while native exposure or orders exist")
        return FaultDecision("SKIP" if action == "skip" else "HALT",
                             "missing market interval; caller must not forward-fill", "critical")

    def price_shock(self, *, prior_price: Decimal, visible_price: Decimal) -> FaultDecision:
        before, now = dec(prior_price, positive=True), dec(visible_price, positive=True)
        jump = abs(now / before - 1)
        if jump <= self.policy.max_price_jump_fraction:
            return FaultDecision("CONTINUE", "visible price within shock threshold", "none")
        if self.policy.shock_action == "fail":
            raise MarketFault(f"visible shock ratio {jump} exceeds risk threshold")
        return FaultDecision("HALT", "visible shock exceeds risk threshold", "critical")
