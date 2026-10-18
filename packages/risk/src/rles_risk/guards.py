"""Physically separate hard risk limits and E2-searchable soft parameters."""
from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping
from decimal import Decimal
from hashlib import sha256
import json


def _d(value: object, *, positive: bool = False) -> Decimal:
    if type(value) is bool or isinstance(value, float):
        raise TypeError("risk values must use Decimal, integer, or decimal string")
    out = Decimal(str(value))
    if not out.is_finite() or (positive and out <= 0):
        raise ValueError("non-finite or non-positive risk value")
    return out


class RiskHalt(RuntimeError):
    """Risk assessment unavailable or unsafe; no new opening orders allowed."""


@dataclass(frozen=True, slots=True)
class HardRiskConfig:
    max_notional: Decimal
    max_leverage: Decimal
    margin_buffer_fraction: Decimal
    min_equity: Decimal
    incomplete_data_max_leverage: Decimal
    require_mark: bool = True
    require_funding: bool = True

    def __post_init__(self) -> None:
        for n in ("max_notional", "max_leverage", "incomplete_data_max_leverage"):
            object.__setattr__(self, n, _d(getattr(self, n), positive=True))
        for n in ("margin_buffer_fraction", "min_equity"):
            object.__setattr__(self, n, _d(getattr(self, n)))
        if not 0 <= self.margin_buffer_fraction < 1 or self.min_equity < 0:
            raise ValueError("invalid hard equity/margin guard")
        if self.incomplete_data_max_leverage > self.max_leverage:
            raise ValueError("missing-data limit cannot increase leverage")
        if type(self.require_mark) is not bool or type(self.require_funding) is not bool:
            raise TypeError("hard capability flags must be bool")

    def signature(self) -> str:
        from dataclasses import asdict
        return sha256(json.dumps(asdict(self), sort_keys=True, default=str).encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class SoftRiskParams:
    target_scale: Decimal
    max_turnover_fraction: Decimal
    drawdown_throttle: Decimal
    funding_sensitivity: Decimal

    def __post_init__(self) -> None:
        for n in ("target_scale", "max_turnover_fraction", "drawdown_throttle", "funding_sensitivity"):
            value = _d(getattr(self, n))
            object.__setattr__(self, n, value)
            if not 0 <= value <= 1:
                raise ValueError(f"soft risk {n} must be in [0,1]")

    @classmethod
    def from_search_dict(cls, values: dict) -> SoftRiskParams:
        keys = set(cls.__dataclass_fields__)
        if set(values) != keys:
            raise ValueError("E2 may only change the four declared SoftRiskParams keys")
        return cls(**{name: _d(value) for name, value in values.items()})

    def with_multipliers(self, values: Mapping[str, float]) -> SoftRiskParams:
        """Apply L5's neutral multiplier mapping without exposing hard limits.

        New E2 searches use independent exposure and drawdown multipliers.
        The three-key form remains readable for existing checkpoint schemas;
        it must not be used when constructing a new search space.
        """
        if set(values) not in ({"exposure_multiplier", "drawdown_throttle"},
                               {"exposure_multiplier", "drawdown_throttle", "volatility_throttle"}):
            raise ValueError("exposure/drawdown multiplier keys required")
        physical = {}
        for name, value in values.items():
            if type(value) not in (int, float, Decimal):
                raise TypeError("soft multiplier must be a finite numeric search value")
            number = _d(str(value))  # Explicit conversion at the float search boundary.
            if not 0 <= number <= 1:
                raise ValueError("soft multiplier must only reduce exposure")
            physical[name] = number
        return SoftRiskParams(
            target_scale=self.target_scale * physical["exposure_multiplier"]
            * physical.get("volatility_throttle", Decimal(1)),
            max_turnover_fraction=self.max_turnover_fraction,
            drawdown_throttle=1 - (1 - self.drawdown_throttle) * physical["drawdown_throttle"],
            funding_sensitivity=self.funding_sensitivity,
        )


@dataclass(frozen=True, slots=True)
class RiskDecisionInternal:
    requested: Decimal
    adjusted: Decimal
    reason: str
    intervened_fraction: Decimal
    force_reduce: bool = False


class SoftRiskTransform:
    def __init__(self, params: SoftRiskParams) -> None:
        if not isinstance(params, SoftRiskParams):
            raise TypeError("soft layer accepts SoftRiskParams only")
        self.params = params

    def apply(self, target: Decimal, *, current_exposure: Decimal, drawdown: Decimal,
              funding_abs: Decimal | None) -> RiskDecisionInternal:
        target, current, dd = _d(target), _d(current_exposure), _d(drawdown)
        # Mark-to-market drift can take the current exposure beyond the action
        # box. Preserve that native fact so reducing it remains possible.
        if not -1 <= target <= 1 or not 0 <= dd <= 1:
            raise ValueError("normalized risk state out of bounds")
        p = self.params
        if p.funding_sensitivity and funding_abs is None:
            raise RiskHalt("funding unavailable for configured soft risk rule")
        funding_factor = Decimal(1) if funding_abs is None else max(Decimal(0), 1 - p.funding_sensitivity * min(Decimal(1), abs(_d(funding_abs))))
        throttle = max(Decimal(0), 1 - p.drawdown_throttle * dd)
        wanted = target * p.target_scale * throttle * funding_factor
        # Capped turnover never blocks risk reduction: only further shrink opening exposure.
        if abs(wanted) > abs(current) and wanted * current >= 0:
            wanted = min(abs(wanted), abs(current) + p.max_turnover_fraction) * (1 if wanted >= 0 else -1)
        elif current * wanted < 0:
            # Flip across zero: turnover limit applies to new opposite-side exposure.
            wanted = min(abs(wanted), max(Decimal(0), p.max_turnover_fraction - abs(current))) * (1 if wanted >= 0 else -1)
        if abs(wanted) > abs(target):
            raise AssertionError("soft risk expanded exposure")
        return RiskDecisionInternal(target, wanted, "soft_limits", abs(target - wanted))


class HardRiskGuard:
    def __init__(self, config: HardRiskConfig) -> None:
        if not isinstance(config, HardRiskConfig):
            raise TypeError("hard guard accepts immutable HardRiskConfig only")
        self.config = config

    def apply(self, requested: Decimal, *, budget_notional: Decimal, equity: Decimal,
              free_margin: Decimal, current_notional: Decimal,
              mark_available: bool, funding_available: bool,
              liquidation_capability: bool, data_quality_grade: str) -> RiskDecisionInternal:
        target, budget = _d(requested), _d(budget_notional, positive=True)
        equity, free, current = _d(equity), _d(free_margin), _d(current_notional)
        if any(type(x) is not bool for x in (mark_available, funding_available, liquidation_capability)):
            raise RiskHalt("capability assertions must be explicit booleans")
        if abs(target) > 1 or equity < 0:
            raise RiskHalt("invalid risk or account facts")
        # Native free margin may genuinely be negative near liquidation.
        # Preserve the ability to reduce exposure, while forbidding new exposure.
        opening_margin = max(Decimal(0), free)
        cfg = self.config
        if data_quality_grade not in {"DQ-A", "DQ-B", "DQ-C", "DQ-D"}:
            raise RiskHalt("data quality grade must be a documented L2 proof")
        if data_quality_grade == "DQ-D":
            raise RiskHalt("DQ-D data cannot be used for training or evaluation")
        if cfg.require_mark and not mark_available:
            raise RiskHalt("required mark price is unavailable")
        if cfg.require_funding and not funding_available:
            raise RiskHalt("funding boundary capability unavailable")
        if equity <= cfg.min_equity:
            if target:
                return RiskDecisionInternal(target, Decimal(0), "equity_halt_flatten", abs(target), current != 0)
            return RiskDecisionInternal(target, target, "equity_halt", Decimal(0), current != 0)
        # DQ-C cannot substantiate margin/liquidation safety; never trust
        # a Boolean liquidation flag to override incomplete source data.
        verified = liquidation_capability and data_quality_grade in {"DQ-A", "DQ-B"}
        max_leverage = cfg.max_leverage if verified else min(cfg.max_leverage, cfg.incomplete_data_max_leverage)
        cap = min(cfg.max_notional, equity * max_leverage)
        if not verified and cap > equity * cfg.incomplete_data_max_leverage:
            raise AssertionError("capability downgrade was ineffective")
        adjusted = max(-cap / budget, min(cap / budget, target))
        # If target enlarges abs position, enforce free-margin buffer. Reduction-to-flat is always allowed.
        if abs(adjusted * budget) > abs(current) and adjusted * current >= 0:
            spendable = max(Decimal(0), opening_margin - equity * cfg.margin_buffer_fraction)
            additional = spendable * max_leverage
            opening_cap = max(Decimal(0), abs(current)) + additional
            adjusted = max(-opening_cap / budget, min(opening_cap / budget, adjusted))
        if adjusted * current < 0 and abs(adjusted * budget) > 0:
            # Reversals require cancel/flatten confirmation before opening a new opposite position.
            adjusted = Decimal(0)
        adjusted = max(Decimal(-1), min(Decimal(1), adjusted))
        if abs(adjusted) > abs(target):
            raise AssertionError("hard risk expanded target")
        force_reduce = (
            abs(adjusted * budget) < abs(current) and adjusted * current >= 0
            and (adjusted != target or abs(current) > cap)
        )
        return RiskDecisionInternal(target, adjusted, "hard_cap" if adjusted != target else "approved",
                                    abs(target-adjusted), force_reduce)
