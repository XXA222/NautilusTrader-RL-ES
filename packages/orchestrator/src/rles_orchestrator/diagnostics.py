"""Streaming, descriptive attribution of policy targets by native plan outcome."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class _Group:
    count: int = 0
    risk_count: int = 0
    totals: dict[str, float] = field(default_factory=dict)

    def add(self, values: dict[str, float]) -> None:
        for name, value in values.items():
            self.totals[name] = self.totals.get(name, 0.0) + value


class ExecutionAttribution:
    """Keep observed plan reasons separate; absent risk targets remain unknown.

    A SUBMIT is an order plan, not proof of a fill. The final gap includes
    quantity rounding, pending orders, fills and mark movement. These summaries
    describe observations and do not establish a counterfactual causal effect.
    """

    def __init__(self) -> None:
        self.count = 0
        self.missing_plan_count = 0
        self.groups: dict[tuple[str, str], _Group] = {}

    def observe(self, *, requested: float, executed: float,
                soft: float | None, risk: float | None,
                action: str | None, reason: str | None) -> None:
        self.count += 1
        if action is None or reason is None:
            self.missing_plan_count += 1
            return
        group = self.groups.setdefault((action, reason), _Group())
        group.count += 1
        group.add({
            "requested_mean": requested,
            "requested_abs_mean": abs(requested),
            "executed_mean": executed,
            "executed_abs_mean": abs(executed),
            "requested_short_fraction": float(requested < -1e-6),
            "requested_long_fraction": float(requested > 1e-6),
            "executed_nonzero_fraction": float(abs(executed) > 1e-6),
            "requested_to_executed_gap_abs_mean": abs(requested - executed),
        })
        if soft is not None and risk is not None:
            group.risk_count += 1
            group.add({
                "soft_target_abs_mean": abs(soft),
                "risk_target_abs_mean": abs(risk),
                "requested_to_soft_gap_abs_mean": abs(requested - soft),
                "soft_to_risk_gap_abs_mean": abs(soft - risk),
                "risk_to_executed_gap_abs_mean": abs(risk - executed),
            })

    def report(self) -> dict:
        groups = []
        risk_fields = {
            "soft_target_abs_mean", "risk_target_abs_mean", "requested_to_soft_gap_abs_mean",
            "soft_to_risk_gap_abs_mean", "risk_to_executed_gap_abs_mean",
        }
        for (action, reason), group in sorted(self.groups.items()):
            groups.append({
                "action": action, "reason": reason, "count": group.count,
                "fraction": group.count / self.count,
                **{name: value / group.count for name, value in group.totals.items()
                   if name not in risk_fields},
                "risk_targets": {
                    "count": group.risk_count,
                    "components_known": group.risk_count == group.count,
                    **{name: group.totals[name] / group.risk_count if group.risk_count else None
                       for name in sorted(risk_fields)},
                },
            })
        return {
            "schema": "rles.execution-attribution.v1", "count": self.count,
            "missing_plan_count": self.missing_plan_count,
            "components_known": self.count > 0 and self.missing_plan_count == 0
            and all(group.risk_count == group.count for group in self.groups.values()),
            "groups": groups,
            "interpretation": "descriptive; SUBMIT is not a fill; final gap includes mark movement",
        }
