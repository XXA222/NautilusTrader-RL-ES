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
    constraint_weight: float = 1.0

    def __post_init__(self) -> None:
        for name, value in vars(self).items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.return_weight <= 0:
            raise ValueError("return_weight must be positive")

    def score(self, *, split: str, net_return: float, tail_loss: float,
              max_drawdown: float, constraint_violations: float) -> float:
        if split != "train_inner":
            raise PermissionError("population fitness may consume only Train-inner")
        vals = (net_return, tail_loss, max_drawdown, constraint_violations)
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in vals):
            raise ValueError("fitness inputs must be finite real numbers")
        if any(v < 0 for v in vals[1:]) or max_drawdown > 1:
            raise ValueError("negative loss, drawdown outside [0,1], or invalid constraints")
        score = (self.return_weight * net_return - self.tail_weight * tail_loss
                 - self.drawdown_weight * max_drawdown - self.constraint_weight * constraint_violations)
        if not math.isfinite(score):
            raise OverflowError("nonfinite aggregate fitness")
        return score

    def score_episode(self, *, split: str, initial_equity: float, final_equity: float,
                      step_log_returns: Sequence[float], drawdowns: Sequence[float],
                      constraint_violations: float, tail_fraction: float) -> float:
        """Score native episode facts without using shaped Gym reward.

        Tail loss is the mean of the worst ``ceil(tail_fraction * steps)``
        nonnegative losses derived from unshaped per-step log returns.
        Constraint violations are supplied by the frozen risk/execution adapter.
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
            constraint_violations=constraint_violations,
        )
