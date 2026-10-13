"""L1 MarketEvent to official Nautilus market-data conversion.

The decoder carries L2 clocks/provenance unchanged and delegates all fills,
funding, valuation, and account updates to the pinned native engine.
"""
from __future__ import annotations

from decimal import Decimal, InvalidOperation
from typing import Any

from rles_contracts.dto import MarketEvent

from .session import EngineBoundaryError


class NativeMarketEventDecoder:
    """Decode supported L1 events using one frozen native instrument/bar type."""

    def __init__(self, *, instrument: Any, bar_type: Any) -> None:
        from nautilus_trader.model import BarType

        if (not isinstance(bar_type, BarType) or
                any(not hasattr(instrument, name) for name in ("id", "make_price", "make_qty"))):
            raise TypeError("official native instrument API and BarType are required")
        if bar_type.instrument_id != instrument.id:
            raise ValueError("native BarType instrument differs from the frozen Instrument")
        self.instrument = instrument
        self.bar_type = bar_type
        self._scheduled_rates: dict[tuple[str, int], Decimal] = {}

    @staticmethod
    def _decimal(payload: dict[str, Any], name: str) -> Decimal:
        value = payload.get(name)
        if isinstance(value, bool) or not isinstance(value, (str, int, float, Decimal)):
            raise EngineBoundaryError(f"native event requires numeric {name}")
        try:
            result = Decimal(str(value))
        except InvalidOperation as exc:
            raise EngineBoundaryError(f"native event has invalid {name}") from exc
        if not result.is_finite():
            raise EngineBoundaryError(f"native event has non-finite {name}")
        return result

    def __call__(self, event: MarketEvent) -> Any:
        from nautilus_trader.model import (
            Bar, FundingRateUpdate, IndexPriceUpdate, MarkPriceUpdate, QuoteTick,
        )

        if type(event) is not MarketEvent:
            raise TypeError("official L1 MarketEvent required")
        try:
            event.require_available(event.ts_init_ns)
        except (TypeError, ValueError) as exc:
            raise EngineBoundaryError("L1 MarketEvent is unavailable or has unusable quality") from exc
        payload = dict(event.payload.entries)
        if payload.get("instrument") != str(self.instrument.id):
            raise EngineBoundaryError("MarketEvent instrument differs from native adapter scope")
        ts_event, ts_init = event.ts_event_ns, event.ts_init_ns
        kind = event.event_type
        if kind == "ohlcv_1m":
            open_ = self._decimal(payload, "open")
            high = self._decimal(payload, "high")
            low = self._decimal(payload, "low")
            close = self._decimal(payload, "close")
            volume = self._decimal(payload, "volume")
            if (low > min(open_, close) or high < max(open_, close) or low > high
                    or volume < 0):
                raise EngineBoundaryError("invalid OHLCV market event")
            try:
                return Bar(
                    bar_type=self.bar_type, open=self.instrument.make_price(open_),
                    high=self.instrument.make_price(high), low=self.instrument.make_price(low),
                    close=self.instrument.make_price(close), volume=self.instrument.make_qty(volume),
                    ts_event=ts_event, ts_init=ts_init,
                )
            except (TypeError, ValueError, OverflowError) as exc:
                raise EngineBoundaryError("OHLCV values exceed native instrument precision") from exc
        if kind in {"mark", "index"}:
            price = self._decimal(payload, "price")
            if price <= 0:
                raise EngineBoundaryError("native mark/index price must be positive")
            update = MarkPriceUpdate if kind == "mark" else IndexPriceUpdate
            try:
                return update(self.instrument.id, self.instrument.make_price(price), ts_event, ts_init)
            except (TypeError, ValueError, OverflowError) as exc:
                raise EngineBoundaryError("mark/index value exceeds native instrument precision") from exc
        if kind == "best_quote":
            bid = self._decimal(payload, "bid_price")
            ask = self._decimal(payload, "ask_price")
            bid_qty = self._decimal(payload, "bid_qty")
            ask_qty = self._decimal(payload, "ask_qty")
            if bid <= 0 or ask < bid or bid_qty <= 0 or ask_qty <= 0:
                raise EngineBoundaryError("native best quote must be uncrossed with positive displayed sizes")
            try:
                native_bid = self.instrument.make_price(bid)
                native_ask = self.instrument.make_price(ask)
                native_bid_qty = self.instrument.make_qty(bid_qty)
                native_ask_qty = self.instrument.make_qty(ask_qty)
                if (native_bid.as_decimal() != bid or native_ask.as_decimal() != ask
                        or native_bid_qty.as_decimal() != bid_qty
                        or native_ask_qty.as_decimal() != ask_qty):
                    raise EngineBoundaryError("best quote exceeds frozen native precision")
                return QuoteTick(self.instrument.id, native_bid, native_ask,
                                 native_bid_qty, native_ask_qty, ts_event, ts_init)
            except (TypeError, ValueError, OverflowError) as exc:
                raise EngineBoundaryError("best quote violates native instrument precision") from exc
        if kind in {"funding_update", "funding_settlement"}:
            rate = self._decimal(payload, "rate")
            if kind == "funding_update":
                boundary = payload.get("next_funding_ns")
                interval = payload.get("interval_minutes")
                if boundary is not None and (type(boundary) is not int or boundary <= ts_event):
                    raise EngineBoundaryError("funding update next_funding_ns must follow the update")
                if interval is not None and (type(interval) is not int or interval <= 0):
                    raise EngineBoundaryError("funding interval_minutes must be positive integer")
            else:
                boundary = payload.get("settlement_ns")
                interval = payload.get("interval_minutes")
                if type(boundary) is not int or boundary != ts_event:
                    raise EngineBoundaryError("funding settlement timestamp must equal event time")
                if interval is not None and (type(interval) is not int or interval <= 0):
                    raise EngineBoundaryError("funding interval_minutes must be positive integer")
                # Settled archive rows are only available after their source
                # boundary. Schedule the native account adjustment just after
                # the proven availability clock; replaying it at the original
                # boundary would make an ex-post rate available before it was
                # published. Keep settlement_ns in the L1/L2 source fact.
                available_at = event.available_at_ns
                if available_at is not None and available_at > boundary:
                    if available_at >= 2**63 - 1:
                        raise EngineBoundaryError("funding availability timestamp cannot be advanced")
                    boundary = available_at + 1
            scheduled_key = None
            if boundary is not None:
                scheduled_key = (str(self.instrument.id), boundary)
                previous = self._scheduled_rates.get(scheduled_key)
                if previous is not None and previous != rate:
                    raise EngineBoundaryError("conflicting funding rates for one native settlement boundary")
            try:
                result = FundingRateUpdate(
                    self.instrument.id, rate, ts_event, ts_init,
                    interval=interval, next_funding_ns=boundary,
                )
            except (TypeError, ValueError, OverflowError) as exc:
                raise EngineBoundaryError("funding rate fields violate native event constraints") from exc
            if scheduled_key is not None:
                self._scheduled_rates[scheduled_key] = rate
            return result
        if kind == "instrument_rule":
            raise EngineBoundaryError("instrument_rule belongs to frozen execution config, not market replay")
        raise EngineBoundaryError(f"unsupported native MarketEvent type: {kind}")
