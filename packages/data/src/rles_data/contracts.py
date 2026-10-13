"""Explicit L2-to-L1 event boundary; no event clock or quality is invented."""
from __future__ import annotations

from rles_contracts.dto import DataQualityGrade, MarketEvent
from rles_contracts.frozen import FrozenFields

from .core import DataError, Event


def to_market_event(event: Event, *, quality_grade: DataQualityGrade) -> MarketEvent:
    """Map an L2 event into the frozen L1 carrier without changing its facts.

    Quality is supplied by the caller from an evaluated dataset/report context;
    it is never guessed from an individual row. Events lacking an availability
    proof or carrying nested payloads cannot cross this scalar metadata DTO.
    """
    if not isinstance(event, Event):
        raise TypeError("event must be an L2 Event")
    if not isinstance(quality_grade, DataQualityGrade):
        raise TypeError("quality_grade must be an explicit L1 DataQualityGrade")
    if event.available_at_ns is None:
        raise DataError("event has no availability proof and cannot become an L1 MarketEvent")
    if quality_grade is DataQualityGrade.D:
        raise DataError("DQ-D event cannot cross into the usable L1 event stream")
    payload = dict(event.payload)
    try:
        frozen = FrozenFields.from_mapping(payload)
    except (TypeError, ValueError) as exc:
        raise DataError("L1 MarketEvent payload must contain scalar metadata only") from exc
    try:
        return MarketEvent(
            event_id=event.event_id,
            ts_event_ns=event.ts_event_ns,
            ts_init_ns=event.ts_init_ns,
            available_at_ns=event.available_at_ns,
            source_hash=event.source_hash,
            quality_grade=quality_grade,
            event_type=event.kind,
            payload=frozen,
        )
    except (TypeError, ValueError) as exc:
        raise DataError("L2 event violates the frozen L1 MarketEvent contract") from exc
