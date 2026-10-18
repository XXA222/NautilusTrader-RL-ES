"""Strict source event schema validation, shared by DTOs and Parquet ingestion."""
from __future__ import annotations
from typing import Any
from .core import DataError, KINDS, MINUTE_NS, require_ns

def _number(payload: dict[str, Any], key: str, *, allow_zero: bool = False,
            signed: bool = False) -> float:
    import math
    if key not in payload or isinstance(payload[key], bool) or not isinstance(payload[key], (str, int, float)):
        raise DataError(f"{key} absent or not numeric")
    try:
        value = float(payload[key])
    except (ValueError, OverflowError) as exc:
        raise DataError(f"{key} invalid") from exc
    if not math.isfinite(value) or (not signed and (value < 0 if allow_zero else value <= 0)):
        raise DataError(f"{key} out of range")
    return value


def validate_payload(kind: str, p: dict[str, Any], ts: int) -> None:
    """Validate exchange-normalized payload without inventing default fees/funding."""
    if kind not in KINDS or not isinstance(p, dict):
        raise DataError("invalid event kind or JSON payload")
    if not p.get("instrument") or not isinstance(p["instrument"], str):
        raise DataError("instrument must be explicit in each event")
    if kind == "ohlcv_1m":
        if ts % MINUTE_NS:
            raise DataError("1m candle event timestamp must denote aligned *close*")
        o = _number(p, "open")
        h = _number(p, "high")
        l = _number(p, "low")
        c = _number(p, "close")
        _number(p, "volume", allow_zero=True)
        if l > min(o, c) or h < max(o, c) or l > h:
            raise DataError("invalid OHLC high/low relation")
    elif kind in ("mark", "index"):
        _number(p, "price")
    elif kind == "best_quote":
        bid = _number(p, "bid_price")
        ask = _number(p, "ask_price")
        _number(p, "bid_qty")
        _number(p, "ask_qty")
        if ask < bid:
            raise DataError("crossed best quote")
        for key in ("update_id", "transaction_time_ns"):
            if type(p.get(key)) is not int or p[key] < 0:
                raise DataError(f"{key} must be a nonnegative integer")
    elif kind == "trade_flow_1s":
        for key in ("aggressor_buy_qty", "aggressor_sell_qty",
                    "aggressor_buy_notional", "aggressor_sell_notional",
                    "first_price", "last_price", "high_price", "low_price"):
            _number(p, key, allow_zero=key.startswith("aggressor_"))
        if (type(p.get("trade_count")) is not int or p["trade_count"] <= 0
                or type(p.get("first_trade_id")) is not int or p["first_trade_id"] < 0
                or type(p.get("last_trade_id")) is not int or p["last_trade_id"] < p["first_trade_id"]):
            raise DataError("trade_flow_1s requires positive count and ordered nonnegative trade IDs")
        if _number(p, "high_price") < max(_number(p, "first_price"), _number(p, "last_price")):
            raise DataError("trade_flow_1s high price is below an observed endpoint")
        if _number(p, "low_price") > min(_number(p, "first_price"), _number(p, "last_price")):
            raise DataError("trade_flow_1s low price is above an observed endpoint")
    elif kind in ("funding_update", "funding_settlement"):
        _number(p, "rate", signed=True)
        if kind == "funding_settlement":
            settlement_value = p.get("settlement_ns")
            if type(settlement_value) is not int:
                raise DataError("settlement_ns must be an integer timestamp")
            settlement = require_ns(settlement_value)
            if settlement != ts:
                raise DataError("funding settlement event time must equal settlement time")
    elif kind == "instrument_rule":
        if ("reduce_only_min_notional_exempt" in p
                and type(p["reduce_only_min_notional_exempt"]) is not bool):
            raise DataError("reduce_only_min_notional_exempt must be bool when provided")
        for key in ("tick_size", "step_size", "min_notional", "min_qty"):
            _number(p, key)
        for key in ("maker_fee", "taker_fee", "initial_margin_rate", "maintenance_margin_rate"):
            _number(p, key, allow_zero=True)
        if _number(p, "initial_margin_rate", allow_zero=True) < _number(p, "maintenance_margin_rate", allow_zero=True):
            raise DataError("initial margin rate below maintenance rate")
        effective_value = p.get("effective_ns")
        if type(effective_value) is not int:
            raise DataError("effective_ns must be an integer timestamp")
        effective = require_ns(effective_value)
        valid_until = p.get("valid_until_ns")
        if valid_until is not None:
            if type(valid_until) is not int or require_ns(valid_until) <= effective:
                raise DataError("invalid rule validity interval")
        if not isinstance(p.get("rule_version"), str) or not p["rule_version"]:
            raise DataError("explicit rule_version required")


