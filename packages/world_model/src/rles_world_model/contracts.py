"""Small immutable contracts for causal market-only data."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

from rles_data.core import DataError, Event, MINUTE_NS, require_hash, require_ns


@dataclass(frozen=True)
class MarketFrame:
    """One closed 1m candle with explicit event and availability times."""

    ts_close_ns: int
    available_at_ns: int
    instrument: str
    open: float
    high: float
    low: float
    close: float
    volume: float

    def __post_init__(self) -> None:
        require_ns(self.ts_close_ns)
        require_ns(self.available_at_ns)
        if self.ts_close_ns % MINUTE_NS:
            raise DataError("market frame close timestamp must align to a 1m boundary")
        if self.available_at_ns < self.ts_close_ns:
            raise DataError("closed candle cannot be available before its close")
        if not isinstance(self.instrument, str) or not self.instrument.strip():
            raise DataError("market frame instrument is required")
        values = (self.open, self.high, self.low, self.close, self.volume)
        if any(type(value) not in (int, float) or not math.isfinite(value) for value in values):
            raise DataError("OHLCV values must be finite numbers")
        if min(self.open, self.high, self.low, self.close) <= 0 or self.volume < 0:
            raise DataError("prices must be positive and volume nonnegative")
        if self.low > min(self.open, self.close) or self.high < max(self.open, self.close):
            raise DataError("market frame has invalid OHLC bounds")

    @classmethod
    def from_event(cls, event: Event) -> MarketFrame:
        if event.kind != "ohlcv_1m" or event.available_at_ns is None:
            raise DataError("MarketFrame requires an available ohlcv_1m event")
        payload = event.payload
        return cls(event.ts_event_ns, event.available_at_ns, str(payload["instrument"]),
                   float(payload["open"]), float(payload["high"]), float(payload["low"]),
                   float(payload["close"]), float(payload["volume"]))


@dataclass(frozen=True)
class MarketReplay:
    """A single-instrument, gap-free sequence of causal 1m candles."""

    frames: tuple[MarketFrame, ...]
    dataset_hash: str

    def __post_init__(self) -> None:
        require_hash(self.dataset_hash)
        if (type(self.frames) is not tuple or not self.frames
                or any(not isinstance(frame, MarketFrame) for frame in self.frames)):
            raise DataError("market replay requires at least one validated frame")
        instrument = self.frames[0].instrument
        for previous, current in zip(self.frames, self.frames[1:]):
            if current.instrument != instrument:
                raise DataError("market replay cannot mix instruments")
            if current.ts_close_ns < previous.ts_close_ns:
                raise DataError("market replay candles must be ordered by close time")
            if current.available_at_ns < previous.available_at_ns:
                raise DataError("market availability regresses within replay")

    @property
    def is_contiguous(self) -> bool:
        return all(current.ts_close_ns - previous.ts_close_ns == MINUTE_NS
                   for previous, current in zip(self.frames, self.frames[1:]))

    @classmethod
    def from_events(cls, events: Iterable[Event], *, dataset_hash: str,
                    instrument: str) -> MarketReplay:
        require_hash(dataset_hash)
        if not instrument:
            raise DataError("instrument is required")
        selected = [MarketFrame.from_event(event) for event in events
                    if event.kind == "ohlcv_1m" and event.payload.get("instrument") == instrument]
        selected.sort(key=lambda frame: frame.ts_close_ns)
        if not selected:
            raise DataError("no available 1m bars in selected source window")
        return cls(tuple(selected), dataset_hash)
