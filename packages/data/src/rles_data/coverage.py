"""Source-declared temporal coverage, separate from mere event presence."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, TYPE_CHECKING

from .core import DataError, FrozenDict, require_ns

if TYPE_CHECKING:
    from .core import CuratedStore


@dataclass(frozen=True)
class CoveragePolicy:
    """Explicit freshness and funding schedule for this audited interval.

    Funding intervals must come from the source (they need not be eight hours).
    An empty settlement tuple is valid only for an interval with none due.
    Provenance identifies the source of those schedule/freshness assumptions.
    """
    maximum_age_ns: Mapping[str, int]
    settlement_times_ns: Mapping[str, tuple[int, ...]]
    provenance: str
    settlement_max_delay_ns: int = 0
    settlement_timestamp_tolerance_ns: int = 0
    known_schedule_times_ns: Mapping[str, tuple[int, ...]] | None = None

    def __post_init__(self) -> None:
        if type(self.provenance) is not str or not self.provenance.strip():
            raise DataError("coverage policy requires source provenance")
        if not isinstance(self.maximum_age_ns, Mapping) or set(self.maximum_age_ns) != {"mark", "index", "funding_update"}:
            raise DataError("explicit mark/index/funding freshness limits required")
        if any(type(v) is not int or v <= 0 for v in self.maximum_age_ns.values()):
            raise DataError("freshness limits must be positive integer nanoseconds")
        if not isinstance(self.settlement_times_ns, Mapping) or not self.settlement_times_ns:
            raise DataError("settlement schedule must explicitly name every instrument")
        schedule = {}
        for symbol, times in self.settlement_times_ns.items():
            if type(symbol) is not str or not symbol or not isinstance(times, (list, tuple)):
                raise DataError("invalid instrument settlement schedule")
            frozen = tuple(require_ns(t) for t in times)
            if frozen != tuple(sorted(set(frozen))):
                raise DataError("settlement schedule must be ordered and unique")
            schedule[symbol] = frozen
        require_ns(self.settlement_max_delay_ns)
        require_ns(self.settlement_timestamp_tolerance_ns)
        if any(
            right - left <= 2 * self.settlement_timestamp_tolerance_ns
            for times in schedule.values()
            for left, right in zip(times, times[1:])
        ):
            raise DataError("settlement timestamp tolerance overlaps scheduled settlements")
        known = None
        if self.known_schedule_times_ns is not None:
            if (not isinstance(self.known_schedule_times_ns, Mapping)
                    or set(self.known_schedule_times_ns) != set(schedule)):
                raise DataError("known funding calendar must name every audited instrument")
            known = {}
            for symbol, times in self.known_schedule_times_ns.items():
                if not isinstance(times, (list, tuple)):
                    raise DataError("known funding calendar timestamps must be sequences")
                frozen = tuple(require_ns(t) for t in times)
                if frozen != tuple(sorted(set(frozen))):
                    raise DataError("known funding calendar must be ordered and unique")
                expected = schedule[symbol]
                if any(not any(abs(actual - expected_ts) <= self.settlement_timestamp_tolerance_ns
                               for actual in frozen) for expected_ts in expected):
                    raise DataError("known funding calendar omits audited settlement timestamps")
                known[symbol] = frozen
        object.__setattr__(self, "maximum_age_ns", FrozenDict(self.maximum_age_ns))
        object.__setattr__(self, "settlement_times_ns", FrozenDict(schedule))
        if known is not None:
            object.__setattr__(self, "known_schedule_times_ns", FrozenDict(known))


def assess_coverage(store: CuratedStore, policy: CoveragePolicy, *,
                    start_ns: int, end_ns: int, instruments: tuple[str, ...]) -> tuple[set[str], tuple[str, ...]]:
    if set(policy.settlement_times_ns) != set(instruments):
        raise DataError("coverage schedule instruments differ from audited scope")
    findings: set[str] = set()
    verified = set(policy.maximum_age_ns) | {"instrument_rule", "funding_settlement"}
    for symbol in instruments:
        events = [e for e in store.events if e.payload["instrument"] == symbol]
        for kind, limit in policy.maximum_age_ns.items():
            updates = sorted((e for e in events if e.kind == kind and e.available_at_ns is not None
                              and e.available_at_ns < end_ns),
                             key=lambda e: (e.available_at_ns, e.ts_event_ns))
            cursor = start_ns
            last_event_ns: int | None = None
            for event in updates:
                visible_at = event.available_at_ns
                assert visible_at is not None
                if visible_at > cursor and (last_event_ns is None or visible_at - 1 - last_event_ns > limit):
                    findings.add(f"{kind}_coverage_gap")
                    verified.discard(kind)
                cursor = max(cursor, visible_at)
                last_event_ns = max(last_event_ns or 0, event.ts_event_ns)
            if last_event_ns is None or end_ns - 1 - last_event_ns > limit:
                findings.add(f"{kind}_coverage_gap")
                verified.discard(kind)
        # Check every possible rule transition/expiry, including late arrivals.
        boundaries = {start_ns, end_ns - 1}
        for event in events:
            if event.kind == "instrument_rule":
                for boundary in (event.available_at_ns, event.payload["effective_ns"],
                                 event.payload.get("valid_until_ns")):
                    if boundary is not None and start_ns <= boundary < end_ns:
                        boundaries.add(boundary)
        for timestamp in boundaries:
            try:
                store.latest_rule(timestamp, symbol)
            except DataError:
                findings.add("instrument_rule_coverage_gap")
                verified.discard("instrument_rule")
                break
        expected = policy.settlement_times_ns[symbol]
        if any(not start_ns <= t < end_ns for t in expected):
            raise DataError("expected funding settlement outside audited interval")
        settlements = sorted(
            (e for e in events if e.kind == "funding_settlement"
             and start_ns <= e.ts_event_ns < end_ns),
            key=lambda event: event.ts_event_ns,
        )
        actual = tuple(e.ts_event_ns for e in settlements)
        schedule_matches = (
            len(actual) == len(expected)
            and all(abs(observed - scheduled) <= policy.settlement_timestamp_tolerance_ns
                    for observed, scheduled in zip(actual, expected))
        )
        if not schedule_matches or any(e.available_at_ns is None or
                                      e.available_at_ns - e.ts_event_ns > policy.settlement_max_delay_ns
                                      for e in settlements):
            findings.add("funding_settlement_coverage_gap")
            verified.discard("funding_settlement")
    return findings, tuple(sorted(verified))
