"""Finite constraint-first aggregate over Train-inner facts only."""
from __future__ import annotations
from dataclasses import dataclass
import math
from collections.abc import Sequence


@dataclass(frozen=True)
class FitnessSpec:
    return_weight: float = 1.0
    tail_weight: float = 1.0
    drawdown_weight: float = 1.0
    risk_intervention_weight: float = 0.05

    def __post_init__(self) -> None:
        for name, value in vars(self).items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.return_weight <= 0:
            raise ValueError("return_weight must be positive")

    def score(self, *, split: str, net_return: float, tail_loss: float,
              max_drawdown: float, risk_intervention_rate: float) -> float:
        if split != "train_inner":
            raise PermissionError("population fitness may consume only Train-inner")
        vals = (net_return, tail_loss, max_drawdown, risk_intervention_rate)
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in vals):
            raise ValueError("fitness inputs must be finite real numbers")
        if any(v < 0 for v in vals[1:]) or max_drawdown > 1 or risk_intervention_rate > 1:
            raise ValueError("loss, drawdown and risk intervention rate must be within valid ranges")
        score = (self.return_weight * net_return - self.tail_weight * tail_loss
                 - self.drawdown_weight * max_drawdown
                 - self.risk_intervention_weight * risk_intervention_rate)
        if not math.isfinite(score):
            raise OverflowError("nonfinite aggregate fitness")
        return score

    def score_episode(self, *, split: str, initial_equity: float, final_equity: float,
                      step_log_returns: Sequence[float], drawdowns: Sequence[float],
                      risk_intervention_rate: float, tail_fraction: float) -> float:
        """Score native episode facts without using shaped Gym reward.

        Tail loss is the mean of the worst ``ceil(tail_fraction * steps)``
        nonnegative losses derived from unshaped per-step log returns.
        Risk-intervention rate is the fraction of decisions where HardRisk
        adjusted the requested action. It is a bounded diagnostic penalty, not
        a count of infeasible or unsafe episodes.
        """
        if (isinstance(initial_equity, bool) or not isinstance(initial_equity, (int, float))
                or not math.isfinite(initial_equity) or initial_equity <= 0
                or isinstance(final_equity, bool) or not isinstance(final_equity, (int, float))
                or not math.isfinite(final_equity) or final_equity < 0):
            raise ValueError("episode equity facts must be finite with positive initial equity")
        if (not isinstance(step_log_returns, Sequence) or not step_log_returns
                or any(isinstance(value, bool) or not isinstance(value, (int, float))
                       or not math.isfinite(value) for value in step_log_returns)):
            raise ValueError("nonempty finite unshaped step log returns required")
        if (not isinstance(drawdowns, Sequence) or len(drawdowns) != len(step_log_returns)
                or any(isinstance(value, bool) or not isinstance(value, (int, float))
                       or not math.isfinite(value) or not 0 <= value <= 1 for value in drawdowns)):
            raise ValueError("one finite [0,1] drawdown fact is required per step")
        if (isinstance(tail_fraction, bool) or not isinstance(tail_fraction, (int, float))
                or not math.isfinite(tail_fraction) or not 0 < tail_fraction <= 1):
            raise ValueError("tail fraction must be finite and in (0,1]")
        losses = sorted((max(0.0, -float(value)) for value in step_log_returns), reverse=True)
        tail_count = max(1, math.ceil(float(tail_fraction) * len(losses)))
        tail_loss = math.fsum(losses[:tail_count]) / tail_count
        net_return = float(final_equity) / float(initial_equity) - 1.0
        if not math.isfinite(net_return):
            raise OverflowError("nonfinite episode net return")
        return self.score(
            split=split, net_return=net_return, tail_loss=tail_loss,
            max_drawdown=max(float(value) for value in drawdowns),
            risk_intervention_rate=risk_intervention_rate,
        )


@dataclass(frozen=True)
class FitnessDecision:
    """Auditable result of constraint-first candidate scoring."""

    score: float
    feasible: bool
    rejection_reason: str | None


@dataclass(frozen=True)
class ConstraintPriorityFitnessSpec:
    """Versioned, bounded economic ranking after terminal solvency screening.

    The score range is (-1, 1) for feasible episodes and exactly -2 for
    terminal insolvency. This keeps feasibility lexicographically ahead of
    any finite economic outcome without treating HardRisk interventions as
    violations or allowing an unbounded return to overwhelm the constraint.
    """

    return_weight: float = 1.0
    tail_weight: float = 1.0
    drawdown_weight: float = 1.0
    protocol: str = "constraint_priority_train_inner_v1"

    def __post_init__(self) -> None:
        for name in ("return_weight", "tail_weight", "drawdown_weight"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.return_weight <= 0 or self.protocol != "constraint_priority_train_inner_v1":
            raise ValueError("constraint-priority protocol and positive return weight are required")

    def evaluate_episode(self, *, split: str, initial_equity: float, final_equity: float,
                         step_log_returns: Sequence[float], drawdowns: Sequence[float],
                         risk_intervention_rate: float, tail_fraction: float,
                         insolvency: bool | None = None,
                         maintenance_margin_breach: bool | None = None,
                         native_liquidation_detected: bool | None = None) -> FitnessDecision:
        # Reuse the legacy input validation and exact native-fact aggregation,
        # while explicitly removing intervention count from the objective.
        if (isinstance(final_equity, bool) or not isinstance(final_equity, (int, float))
                or not math.isfinite(final_equity)):
            raise ValueError("episode final equity must be a finite real number")
        # A negative terminal account balance is a valid bankruptcy fact in a
        # futures account. The legacy economic scorer requires nonnegative
        # equity, so clamp only its unused score input; feasibility below still
        # uses the original value and rejects every terminal balance <= 0.
        economic_final_equity = max(float(final_equity), 0.0)
        economics = FitnessSpec(
            return_weight=self.return_weight,
            tail_weight=self.tail_weight,
            drawdown_weight=self.drawdown_weight,
            risk_intervention_weight=0.0,
        ).score_episode(
            split=split, initial_equity=initial_equity, final_equity=economic_final_equity,
            step_log_returns=step_log_returns, drawdowns=drawdowns,
            risk_intervention_rate=risk_intervention_rate, tail_fraction=tail_fraction,
        )
        constraint_facts = (insolvency, maintenance_margin_breach, native_liquidation_detected)
        if any(value is not None and type(value) is not bool for value in constraint_facts):
            raise ValueError("native feasibility facts must be Boolean when provided")
        reasons = []
        if final_equity <= 0 or insolvency is True:
            reasons.append("episode_insolvency_detected")
        if maintenance_margin_breach is True:
            reasons.append("maintenance_margin_breach_detected")
        if native_liquidation_detected is True:
            reasons.append("native_liquidation_detected")
        if reasons:
            return FitnessDecision(score=-2.0, feasible=False,
                                   rejection_reason="+".join(reasons))
        bounded = math.tanh(economics)
        if not math.isfinite(bounded):
            raise OverflowError("nonfinite bounded economic fitness")
        if self.protocol == "constraint_priority_train_inner_v2":
            bounded = min(max(bounded, math.nextafter(-1.0, 0.0)),
                          math.nextafter(1.0, 0.0))
        return FitnessDecision(score=bounded, feasible=True, rejection_reason=None)

    def score_episode(self, **facts: object) -> float:
        """Score-compatible wrapper for callers that do not need the reason."""
        decision = self.evaluate_episode(**facts)  # type: ignore[arg-type]
        return decision.score


@dataclass(frozen=True)
class FeasibilityFirstFitnessSpec(ConstraintPriorityFitnessSpec):
    """Candidate-level feasibility ordering; incompatible with v1 checkpoints."""

    protocol: str = "constraint_priority_train_inner_v2"

    def __post_init__(self) -> None:
        for name in ("return_weight", "tail_weight", "drawdown_weight"):
            value = getattr(self, name)
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or value < 0):
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.return_weight <= 0 or self.protocol != "constraint_priority_train_inner_v2":
            raise ValueError("feasibility-first protocol and positive return weight are required")
