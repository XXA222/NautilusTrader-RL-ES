"""E2 permitted soft-risk parameter space; never accepts HardRiskConfig.

CMA-ES works in an unbounded latent space. A deterministic tanh mapping
keeps every objective evaluation inside strictly allowed soft-risk ranges.
This is not an evolutionary update algorithm.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math
from typing import Sequence

import numpy as np

_ALLOWED = frozenset({"exposure_multiplier", "drawdown_throttle", "volatility_throttle"})


@dataclass(frozen=True)
class SoftRiskDimension:
    name: str
    lower: float
    upper: float

    def __post_init__(self) -> None:
        if self.name not in _ALLOWED or not all(
            isinstance(x, (float, int)) and not isinstance(x, bool) and math.isfinite(x)
            for x in (self.lower, self.upper)
        ):
            raise ValueError("dimension is not an allowed finite SoftRisk multiplier")
        if not (0 <= self.lower < self.upper <= 1):
            raise ValueError("SoftRisk multipliers must only reduce exposure")


@dataclass(frozen=True)
class SoftRiskSpace:
    dimensions: tuple[SoftRiskDimension, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.dimensions, tuple) or not self.dimensions:
            raise ValueError("nonempty soft-risk dimension tuple required")
        names = [d.name for d in self.dimensions]
        if any(not isinstance(d, SoftRiskDimension) for d in self.dimensions) or len(set(names)) != len(names):
            raise ValueError("unique SoftRiskDimension members required")

    @property
    def signature(self) -> str:
        data = [(d.name, d.lower, d.upper) for d in self.dimensions]
        return sha256(json.dumps(data, separators=(",", ":"), allow_nan=False).encode()).hexdigest()

    def decode(self, vector: Sequence[float] | np.ndarray) -> dict[str, float]:
        """Validate *physical* multipliers, with zero implicit clipping."""
        a = np.asarray(vector)
        if a.shape != (len(self.dimensions),) or a.dtype.kind not in "fi" or not np.isfinite(a).all():
            raise ValueError("invalid SoftRisk vector shape, dtype, or finiteness")
        out = {}
        for d, value in zip(self.dimensions, a, strict=True):
            f = float(value)
            if not (d.lower <= f <= d.upper):
                raise ValueError(f"SoftRisk multiplier exceeds hard-allowed search domain: {d.name}")
            out[d.name] = f
        return out

    def from_latent(self, latent: Sequence[float] | np.ndarray) -> dict[str, float]:
        """Smooth R^n -> constrained multipliers, even for a CMA-ES without bounds."""
        x = np.asarray(latent)
        if x.shape != (len(self.dimensions),) or x.dtype.kind not in "fi" or not np.isfinite(x).all():
            raise ValueError("latent SoftRisk vector must be finite and dimensionally exact")
        values = {}
        for d, raw in zip(self.dimensions, x, strict=True):
            mid = (d.lower + d.upper) / 2.0
            half = (d.upper - d.lower) / 2.0
            # Constrained transform, never a replacement for the L3 HardRiskGuard.
            value = mid + half * math.tanh(float(raw))
            values[d.name] = min(d.upper, max(d.lower, value))
        return values
