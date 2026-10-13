"""Fail-closed data outage policies and time-versioned instrument rules."""
from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from .core import CuratedStore, DataError, DataQualityReport, Event, Grade, require_ns


class GapAction(StrEnum):
    FAIL = "fail"  # reject input/experiment configuration before starting
    SKIP = "skip"  # omit this noncritical decision from directional-only research
    HALT = "halt"  # stop a running episode, never synthesize price/funding/rules


@dataclass(frozen=True)
class GapDecision:
    action: GapAction
    reason: str
    permitted_purpose: str | None = None


def decide_gap(report: DataQualityReport, purpose: str, *, runtime: bool = False) -> GapDecision:
    """An explicit fail/skip/halt table; absence is never mapped to 0."""
    if purpose not in ("directional", "rl_es", "leveraged", "liquidation", "settlement"):
        raise DataError("unrecognized study purpose")
    fatal = report.grade == Grade.D or report.conflict_count > 0
    if fatal:
        return GapDecision(GapAction.HALT if runtime else GapAction.FAIL,
                           "DQ-D, critical missing market fact or conflicting rules")
    if report.grade == Grade.C:
        if purpose == "directional":
            return GapDecision(GapAction.SKIP, "no reliable mark/index/historical risk rule",
                               "directional")
        return GapDecision(GapAction.HALT if runtime else GapAction.FAIL,
                           "DQ-C blocks leveraged/settlement/full research")
    if "partial_capability_assessment" in report.findings and purpose != "directional":
        return GapDecision(GapAction.FAIL, "partial DQ cannot authorize full research")
    return GapDecision(GapAction.SKIP, "no critical gap; normal continuation permitted", purpose)


class RuleTimeline:
    """Rule as-of query; blocks overlapping conflicting intervals.

    Effective-from is inclusive and valid-until is exclusive. If valid-until is
    absent, a later available effective version supersedes the earlier version.
    """
    def __init__(self, store: CuratedStore, instrument: str):
        self.store = store
        self.instrument = instrument
        self.rules = tuple(sorted((e for e in store.events if e.kind == "instrument_rule"
                                   and e.payload.get("instrument") == instrument),
                                  key=lambda e: (e.payload["effective_ns"], e.event_id)))
        if not self.rules:
            raise DataError("instrument rule timeline empty")
        for i, previous in enumerate(self.rules):
            for next_rule in self.rules[i + 1:]:
                if next_rule.payload["effective_ns"] == previous.payload["effective_ns"] and next_rule.payload != previous.payload:
                    raise DataError("conflicting rules with same effective timestamp")
                cutoff = previous.payload.get("valid_until_ns")
                if (cutoff is not None and next_rule.payload["effective_ns"] < cutoff and
                        next_rule.payload != previous.payload):
                    raise DataError("overlapping explicit instrument rule intervals")

    def at(self, decision_ns: int) -> Event:
        require_ns(decision_ns)
        return self.store.latest_rule(decision_ns, self.instrument)

    def fingerprint(self, decision_ns: int) -> str:
        return self.at(decision_ns).hash
