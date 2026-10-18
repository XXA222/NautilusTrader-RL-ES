"""Fail-closed B0 checks for time alignment, availability, gaps and folds."""
from __future__ import annotations

from dataclasses import dataclass

from rles_data.core import DataError, MINUTE_NS

from .contracts import MarketReplay


@dataclass(frozen=True)
class DataAudit:
    instrument: str
    bars: int
    start_ns: int
    end_ns: int
    missing_minutes: int
    duplicate_minutes: int
    availability_violations: int
    dataset_hash: str
    market_model_eligible: bool
    perpetual_economics_eligible: bool
    blockers: tuple[str, ...]


def audit_replay(replay: MarketReplay, *, has_instrument_rule: bool = False) -> DataAudit:
    if type(has_instrument_rule) is not bool:
        raise DataError("instrument rule evidence flag must be Boolean")
    frames = replay.frames
    missing = sum(max(0, (right.ts_close_ns - left.ts_close_ns) // MINUTE_NS - 1)
                  for left, right in zip(frames, frames[1:]))
    duplicates = sum(left.ts_close_ns == right.ts_close_ns for left, right in zip(frames, frames[1:]))
    availability_violations = sum(frame.available_at_ns < frame.ts_close_ns for frame in frames)
    blockers = []
    if missing:
        blockers.append("one_minute_gaps")
    if duplicates:
        blockers.append("duplicate_minute_timestamps")
    if availability_violations:
        blockers.append("availability_before_bar_close")
    if not has_instrument_rule:
        blockers.append("instrument_rule_missing_perpetual_economics_blocked")
    return DataAudit(
        instrument=frames[0].instrument,
        bars=len(frames),
        start_ns=frames[0].ts_close_ns,
        end_ns=frames[-1].ts_close_ns,
        missing_minutes=missing,
        duplicate_minutes=duplicates,
        availability_violations=availability_violations,
        dataset_hash=replay.dataset_hash,
        market_model_eligible=not bool(missing or duplicates or availability_violations),
        perpetual_economics_eligible=not bool(missing or duplicates or availability_violations or not has_instrument_rule),
        blockers=tuple(blockers),
    )
