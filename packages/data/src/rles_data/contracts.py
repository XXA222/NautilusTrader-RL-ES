"""Explicit L2-to-L1 event boundary; no event clock or quality is invented."""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import re
from typing import Any, Mapping

from rles_contracts.dto import DataQualityGrade, MarketEvent
from rles_contracts.frozen import FrozenFields

from .core import DataError, Event


@dataclass(frozen=True, slots=True)
class InstrumentContractSpec:
    """Currency and contract units frozen alongside dated instrument rules."""

    instrument_id: str
    raw_symbol: str
    base_currency: str
    quote_currency: str
    settlement_currency: str
    is_inverse: bool
    multiplier: Decimal

    def __post_init__(self) -> None:
        if not isinstance(self.instrument_id, str) or not self.instrument_id.strip():
            raise DataError("contract spec instrument ID required")
        for name in ("raw_symbol", "base_currency", "quote_currency", "settlement_currency"):
            value = getattr(self, name)
            if not isinstance(value, str) or re.fullmatch(r"[A-Z0-9][A-Z0-9._/-]{0,31}", value) is None:
                raise DataError(f"invalid contract spec {name}")
        if type(self.is_inverse) is not bool:
            raise DataError("contract spec is_inverse must be bool")
        try:
            multiplier = Decimal(str(self.multiplier))
        except (InvalidOperation, ValueError) as exc:
            raise DataError("invalid contract multiplier") from exc
        if not multiplier.is_finite() or multiplier <= 0:
            raise DataError("contract multiplier must be positive and finite")
        if not self.is_inverse and self.settlement_currency != self.quote_currency:
            raise DataError("linear contract settlement currency must equal quote currency")
        object.__setattr__(self, "multiplier", multiplier)

    @classmethod
    def from_rule_payload(cls, rule: Mapping[str, Any], *, instrument_id: str) -> InstrumentContractSpec:
        contract = rule.get("contract_spec")
        if contract is None:
            # Backward-compatibility for the repository's frozen ETH golden rule.
            # New instruments must carry an explicit contract_spec in their rule event.
            if instrument_id != "ETHUSDT-PERP.SIM":
                raise DataError("instrument rule is missing explicit contract_spec metadata")
            contract = {
                "raw_symbol": "ETHUSDT", "base_currency": "ETH", "quote_currency": "USDT",
                "settlement_currency": "USDT", "is_inverse": False, "multiplier": "1",
            }
        if not isinstance(contract, Mapping) or set(contract) != {
            "raw_symbol", "base_currency", "quote_currency", "settlement_currency", "is_inverse", "multiplier",
        }:
            raise DataError("invalid contract_spec keys")
        if rule.get("instrument") != instrument_id:
            raise DataError("contract spec rule identity mismatch")
        return cls(
            instrument_id=instrument_id,
            raw_symbol=contract["raw_symbol"],
            base_currency=contract["base_currency"],
            quote_currency=contract["quote_currency"],
            settlement_currency=contract["settlement_currency"],
            is_inverse=contract["is_inverse"],
            multiplier=contract["multiplier"],
        )


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
