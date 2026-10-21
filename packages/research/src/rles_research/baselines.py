"""B0–B3 target-only baseline policies; L3/L5 execution and HardRisk stay external."""
from __future__ import annotations

import random
from dataclasses import dataclass
from .numeric import finite_real


@dataclass(frozen=True)
class BaselineSpec:
    name: str
    seed: int
    target_long: float = 1.0
    momentum_lookback: int = 20

    def __post_init__(self):
        if self.name not in {'B0_no_trade','B1_always_long','B2_simple_momentum','B3_random_action',
                             'B4_always_short', 'B5_simple_reversal'}:
            raise ValueError('unknown baseline')
        if (type(self.seed) is not int or type(self.momentum_lookback) is not int
            or not finite_real(self.target_long)
            or not 0 <= self.target_long <= 1 or self.momentum_lookback < 2):
            raise ValueError('invalid preregistered baseline parameters')


class BaselinePolicy:
    """Return raw target only; caller MUST execute through same HardRisk and Execution."""

    def __init__(self,spec: BaselineSpec):
        self.spec=spec
        self._rng=random.Random(spec.seed)

    def action(self, closes: list[float]) -> float:
        if any(not finite_real(x) or x<=0 for x in closes):
            raise ValueError('invalid historic close')
        name=self.spec.name
        if name=='B0_no_trade':return 0.0
        if name=='B1_always_long':return self.spec.target_long
        if name=='B4_always_short':return -self.spec.target_long
        if name=='B3_random_action':return self._rng.uniform(-1,1)
        n=self.spec.momentum_lookback
        if len(closes)<n+1:return 0.0
        direction = 1 if closes[-1]>closes[-n-1] else -1 if closes[-1]<closes[-n-1] else 0
        return direction * self.spec.target_long * (-1 if name == 'B5_simple_reversal' else 1)
