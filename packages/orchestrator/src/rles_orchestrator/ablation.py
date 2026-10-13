"""A0-A4 research ablation comparability guard, prior to L6 scoring."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping
from types import MappingProxyType

from .workflow import Route, RunSpec, STAGES


@dataclass(frozen=True)
class AblationPlan:
    """Demand same market, execution, actor, soft-risk schema, seed, budgets.

    Config hashes may differ by route (because the route is part of config),
    but shared experiment semantics must be identical, verified externally by L1.
    """
    runs: Mapping[Route, RunSpec]

    def __post_init__(self) -> None:
        records = dict(self.runs)
        if set(records) != set(Route):
            raise ValueError("ablation requires exactly A0 through A4")
        if any(not isinstance(v, RunSpec) or v.route != route for route, v in records.items()):
            raise ValueError("ablation route/spec mismatch")
        ids = {(v.run_id, v.attempt_id) for v in records.values()}
        if len(ids) != 5 or len({v.run_id for v in records.values()}) != 5 or len({v.attempt_id for v in records.values()}) != 5:
            raise ValueError("ablation cannot reuse run/attempt identities")
        baseline = records[Route.A0]
        frozen = ("algorithm", "seed", "data_hash", "feature_hash", "hard_risk_hash",
                  "execution_hash", "normalization_hash", "soft_risk_schema_hash", "initial_soft_risk_hash",
                  "actor_signature")
        for route, spec in records.items():
            for name in frozen:
                if getattr(spec, name) != getattr(baseline, name):
                    raise ValueError(f"noncomparable A0-A4 input: {name} ({route})")
        for stage in ("E1", "R1", "E2"):
            budgets = [s.stage_budget[stage] for s in records.values() if stage in STAGES[s.route]]
            if budgets and any(b != budgets[0] for b in budgets[1:]):
                raise ValueError(f"inconsistent shared {stage} compute budget across ablations")
        object.__setattr__(self, "runs", MappingProxyType(records))
