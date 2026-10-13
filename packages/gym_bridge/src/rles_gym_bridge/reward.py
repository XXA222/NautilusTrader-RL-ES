"""Equity-only economic reward and explicit drawdown/guard shaping."""
from __future__ import annotations

from dataclasses import dataclass
from math import log, isfinite


@dataclass(frozen=True, slots=True)
class RewardConfig:
    lambda_dd: float = 0.0
    lambda_guard: float = 0.0
    insolvent_penalty: float = -10.0

    def __post_init__(self) -> None:
        if any(not isfinite(x) for x in (self.lambda_dd, self.lambda_guard, self.insolvent_penalty)):
            raise ValueError("reward config must be finite")
        if self.lambda_dd < 0 or self.lambda_guard < 0 or self.insolvent_penalty >= 0:
            raise ValueError("invalid reward costs")


@dataclass(frozen=True, slots=True)
class RewardBreakdown:
    r_econ: float
    r_shape: float
    r_total: float
    drawdown: float
    peak_equity: float
    insolvency: bool


@dataclass(frozen=True, slots=True)
class RewardState:
    previous_equity: float
    peak_equity: float
    drawdown: float


class RewardCalculator:
    """State is reward-related observation memory, NOT an account ledger."""
    def __init__(self, config: RewardConfig) -> None:
        self.config = config
        self._prev_equity: float | None = None
        self._peak: float | None = None
        self._prev_dd = 0.0

    @property
    def peak_equity(self) -> float:
        if self._peak is None:
            raise RuntimeError("reward not initialized")
        return self._peak

    @property
    def drawdown(self) -> float:
        return self._prev_dd

    def export_state(self) -> RewardState:
        if self._prev_equity is None or self._peak is None:
            raise RuntimeError("reward not initialized")
        return RewardState(self._prev_equity, self._peak, self._prev_dd)

    def restore(self, state: RewardState) -> None:
        if (type(state) is not RewardState or
                any(not isfinite(v) for v in (state.previous_equity, state.peak_equity, state.drawdown)) or
                not 0 < state.previous_equity <= state.peak_equity or
                not 0 <= state.drawdown <= 1 or
                abs(state.drawdown - (1 - state.previous_equity / state.peak_equity)) > 1e-12):
            raise ValueError("invalid live reward restore state")
        self._prev_equity, self._peak, self._prev_dd = state.previous_equity, state.peak_equity, state.drawdown

    def reset(self, equity: float) -> None:
        if not isfinite(equity) or equity <= 0:
            raise ValueError("initial native equity must be positive and finite")
        self._prev_equity = self._peak = float(equity)
        self._prev_dd = 0.0

    def calculate(self, *, native_equity: float, guard_intervention: float) -> RewardBreakdown:
        if self._prev_equity is None or self._peak is None:
            raise RuntimeError("reset reward at first native snapshot")
        if not isfinite(native_equity) or not isfinite(guard_intervention) or not 0 <= guard_intervention <= 2:
            raise ValueError("invalid native equity or guard input")
        if native_equity <= 0:
            dd_growth = max(0.0, 1.0 - self._prev_dd)
            self._prev_equity = float(native_equity)
            self._prev_dd = 1.0
            shape = -self.config.lambda_dd * dd_growth - self.config.lambda_guard * guard_intervention
            return RewardBreakdown(self.config.insolvent_penalty, shape,
                                   self.config.insolvent_penalty + shape, 1.0, self._peak, True)
        # Log differences avoid overflow/underflow in extreme native-equity ratios.
        if self._prev_equity <= 0:
            raise RuntimeError("insolvent episode cannot continue")
        econ = log(native_equity) - log(self._prev_equity)
        new_peak = max(self._peak, native_equity)
        dd = max(0.0, 1.0 - native_equity / new_peak)
        dd_growth = max(0.0, dd - self._prev_dd)
        shape = -self.config.lambda_dd * dd_growth - self.config.lambda_guard * guard_intervention
        self._prev_dd, self._prev_equity, self._peak = dd, native_equity, new_peak
        if not isfinite(econ) or not isfinite(shape) or not isfinite(econ + shape):
            raise ValueError("reward non-finite")
        return RewardBreakdown(econ, shape, econ + shape, dd, new_peak, False)
