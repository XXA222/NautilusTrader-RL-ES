"""Closed-interval multi-timeframe features and immutable train-only normalization."""
from __future__ import annotations

from dataclasses import dataclass
import math
import struct
from typing import Iterable

from .core import (CuratedStore, DataError, FeatureField, FeatureSpec,
                   MINUTE_NS, Window, digest, require_ns)

TIMEFRAMES = (5, 15, 60, 240, 480, 1440)


def _instrument(store: CuratedStore, instrument: str | None) -> str:
    symbols = store.instruments
    if instrument is None:
        if len(symbols) != 1 or None in symbols:
            raise DataError("instrument must be explicitly selected for multi-symbol snapshots")
        symbol = next(iter(symbols))
        if not isinstance(symbol, str):
            raise DataError("instrument must be explicit in each event")
        return symbol
    if instrument not in symbols:
        raise DataError(f"instrument not found: {instrument}")
    return instrument


def _bars_1m(store: CuratedStore, decision_ns: int, instrument: str,
             *, start_event_ns: int | None = None) -> tuple[dict, ...]:
    source = (store.visible(decision_ns, "ohlcv_1m") if start_event_ns is None
              else store.visible_event_window(decision_ns, start_event_ns, "ohlcv_1m"))
    bars = sorted((e for e in source
                   if e.payload.get("instrument") == instrument), key=lambda e: e.ts_event_ns)
    closes = set()
    result = []
    for e in bars:
        if e.ts_event_ns in closes:
            raise DataError("non-unique 1m close timestamp in feature history")
        closes.add(e.ts_event_ns)
        result.append({"close_ns": e.ts_event_ns, "available_at_ns": e.available_at_ns,
                       "open": float(e.payload["open"]), "high": float(e.payload["high"]),
                       "low": float(e.payload["low"]), "close": float(e.payload["close"]),
                       "volume": float(e.payload["volume"])})
    return tuple(result)


def aggregate_closed(store: CuratedStore, decision_ns: int, timeframe_minutes: int,
                     *, instrument: str | None = None,
                     lookback_buckets: int | None = None) -> tuple[dict, ...]:
    """Bars are UTC epoch-aligned, close-labelled and include all required 1m bars.

    A partial current bucket is never exposed, even when intermediate 1m are visible.
    A completed bucket is delayed until its last constituent becomes available.
    """
    require_ns(decision_ns)
    if timeframe_minutes not in TIMEFRAMES:
        raise DataError("unsupported aggregation interval")
    if lookback_buckets is not None and (type(lookback_buckets) is not int or lookback_buckets <= 0):
        raise DataError("lookback_buckets must be a positive integer")
    symbol = _instrument(store, instrument)
    span = timeframe_minutes * MINUTE_NS
    buckets: dict[int, list[dict]] = {}
    start_event_ns = None
    if lookback_buckets is not None:
        last_complete_bucket = (decision_ns // span) * span
        start_event_ns = max(0, last_complete_bucket - lookback_buckets * span + MINUTE_NS)
    for bar in _bars_1m(store, decision_ns, symbol, start_event_ns=start_event_ns):
        timestamp = bar["close_ns"]
        bucket_close = ((timestamp - 1) // span + 1) * span
        if bucket_close <= decision_ns:
            buckets.setdefault(bucket_close, []).append(bar)
    output = []
    for bucket_close, segment in sorted(buckets.items()):
        if len(segment) != timeframe_minutes:
            continue
        if (segment[0]["close_ns"] != bucket_close - span + MINUTE_NS or
                segment[-1]["close_ns"] != bucket_close or
                any(b["close_ns"] - a["close_ns"] != MINUTE_NS for a, b in zip(segment, segment[1:]))):
            continue
        availability = max(b["available_at_ns"] for b in segment)
        if availability > decision_ns:
            continue
        output.append({"close_ns": bucket_close, "available_at_ns": availability,
                       "open": segment[0]["open"], "high": max(b["high"] for b in segment),
                       "low": min(b["low"] for b in segment),
                       "close": segment[-1]["close"],
                       "volume": math.fsum(b["volume"] for b in segment)})
    return tuple(output)


def _recipe(field: FeatureField) -> str:
    if field.recipe is not None:
        return field.recipe
    for prefix, recipe in (("return_", "return"), ("volume_mean_", "volume_mean"),
                           ("close_", "close"), ("volatility_", "volatility")):
        if field.name.startswith(prefix):
            return recipe
    raise DataError(f"feature {field.name} requires explicit recipe")


def _assert_contiguous(bars: tuple[dict, ...] | list[dict], timeframe: int) -> None:
    gap = timeframe * MINUTE_NS
    if any(b["close_ns"] - a["close_ns"] != gap for a, b in zip(bars, bars[1:])):
        raise DataError("feature lookback crosses an OHLCV gap")


@dataclass(frozen=True)
class FeatureRow:
    schema_hash: str
    decision_ts_ns: int
    available_at_ns: int
    instrument: str
    values: tuple[float, ...]

    def __post_init__(self) -> None:
        from .core import require_hash

        require_hash(self.schema_hash)
        require_ns(self.decision_ts_ns)
        require_ns(self.available_at_ns)
        if (type(self.instrument) is not str or not self.instrument or type(self.values) is not tuple or
                any(type(value) not in (float, int) or not math.isfinite(value) for value in self.values) or
                self.available_at_ns > self.decision_ts_ns):
            raise DataError("future or non-finite feature observation")

    @property
    def hash(self) -> str:
        return digest(self)


def make_feature_row(store: CuratedStore, spec: FeatureSpec, decision_ns: int,
                     *, instrument: str | None = None) -> FeatureRow:
    require_ns(decision_ns)
    symbol = _instrument(store, instrument)
    values: list[float] = []
    availability: list[int] = []
    series_cache: dict[tuple[int, int], tuple[dict, ...]] = {}
    for field in spec.fields:
        recipe = _recipe(field)
        if field.source_kind == "ohlcv_1m":
            if recipe not in ("return", "volume_mean", "close", "volatility", "trend_efficiency"):
                raise DataError("invalid candle recipe")
            n = field.lookback_bars + (1 if recipe in {"return", "trend_efficiency"} else 0)
            cache_key = (field.timeframe_minutes, n)
            if cache_key not in series_cache:
                if field.timeframe_minutes == 1:
                    start_event_ns = max(0, decision_ns - (n + 2) * MINUTE_NS)
                    series_cache[cache_key] = _bars_1m(
                        store, decision_ns, symbol, start_event_ns=start_event_ns)
                else:
                    series_cache[cache_key] = aggregate_closed(
                        store, decision_ns, field.timeframe_minutes,
                        instrument=symbol, lookback_buckets=n)
            bars = series_cache[cache_key]
            if len(bars) < n:
                raise DataError("not enough closed bars for feature")
            subset = bars[-n:]
            expected_close = (decision_ns // (field.timeframe_minutes * MINUTE_NS)) * (field.timeframe_minutes * MINUTE_NS)
            allowed_close_lag = field.max_age_ns if field.max_age_ns is not None else 0
            if expected_close - subset[-1]["close_ns"] > allowed_close_lag:
                raise DataError("latest expected closed candle missing/stale")
            _assert_contiguous(subset, field.timeframe_minutes)
            if recipe == "return":
                value = subset[-1]["close"] / subset[0]["close"] - 1.0
            elif recipe == "trend_efficiency":
                returns = [math.log(b["close"] / a["close"])
                           for a, b in zip(subset, subset[1:])]
                path = math.fsum(abs(value) for value in returns)
                value = 0.0 if path == 0 else max(-1.0, min(1.0, math.fsum(returns) / path))
            elif recipe == "close":
                value = subset[-1]["close"]
            elif recipe == "volume_mean":
                value = math.fsum(b["volume"] for b in subset) / len(subset)
            else:
                if len(subset) < 2:
                    raise DataError("volatility requires at least two bars")
                returns = [math.log(b["close"] / a["close"]) for a, b in zip(subset, subset[1:])]
                mean = math.fsum(returns) / len(returns)
                value = math.sqrt(math.fsum((x - mean)**2 for x in returns) / len(returns))
            availability.append(max(b["available_at_ns"] for b in subset))
        else:
            if field.timeframe_minutes != 1 or field.lookback_bars != 1 or field.max_age_ns is None:
                raise DataError("event feature requires 1m, lookback=1 and explicit max_age_ns")
            if recipe not in ("latest", "basis"):
                raise DataError("invalid event feature recipe")
            event_start_ns = max(0, decision_ns - field.max_age_ns)
            matching = [e for e in store.visible_event_window(
                decision_ns, event_start_ns, field.source_kind)
                        if e.payload.get("instrument") == symbol]
            if not matching:
                raise DataError(f"{field.source_kind} event unavailable")
            event = max(matching, key=lambda e: (e.ts_event_ns, e.available_at_ns, e.event_id))
            if decision_ns - event.ts_event_ns > field.max_age_ns:
                raise DataError(f"{field.source_kind} feature stale")
            if recipe == "basis":
                if field.source_kind != "mark":
                    raise DataError("basis feature must be sourced from mark")
                indices = [e for e in store.visible_event_window(
                    decision_ns, event_start_ns, "index")
                           if e.payload.get("instrument") == symbol
                           and decision_ns - e.ts_event_ns <= field.max_age_ns]
                if not indices:
                    raise DataError("index price not available for basis")
                index = max(indices, key=lambda e: (e.ts_event_ns, e.available_at_ns, e.event_id))
                value = float(event.payload["price"]) / float(index.payload["price"]) - 1.0
                event_available = event.available_at_ns
                index_available = index.available_at_ns
                if event_available is None or index_available is None:
                    raise DataError("visible price feature has no availability timestamp")
                availability.append(max(event_available, index_available))
            elif field.source_kind in ("mark", "index"):
                value = float(event.payload["price"])
                if event.available_at_ns is None:
                    raise DataError("visible price feature has no availability timestamp")
                availability.append(event.available_at_ns)
            elif field.source_kind in ("funding_update", "funding_settlement"):
                value = float(event.payload["rate"])
                if event.available_at_ns is None:
                    raise DataError("visible funding feature has no availability timestamp")
                availability.append(event.available_at_ns)
            else:
                raise DataError("rules are not scalar features: use latest_rule")
        if not math.isfinite(value):
            raise DataError("non-finite derived feature")
        if field.dtype == "float32":
            try:
                value = struct.unpack("!f", struct.pack("!f", value))[0]
            except (OverflowError, struct.error) as exc:
                raise DataError("float32 feature overflow") from exc
            if not math.isfinite(value):
                raise DataError("non-finite float32 feature")
        values.append(value)
    return FeatureRow(spec.hash, decision_ns, max(availability), symbol, tuple(values))


def make_features(store: CuratedStore, spec: FeatureSpec, decision_ns: int,
                  *, instrument: str | None = None) -> tuple[float, ...]:
    """Backwards compatible bare vector; use make_feature_row for availability/hash."""
    return make_feature_row(store, spec, decision_ns, instrument=instrument).values


@dataclass(frozen=True)
class Scaler:
    feature_hash: str
    mean: tuple[float, ...]
    std: tuple[float, ...]
    train_window_hash: str
    sample_count: int
    training_only: bool = True

    def __post_init__(self) -> None:
        from .core import require_hash

        require_hash(self.feature_hash)
        require_hash(self.train_window_hash)
        if type(self.mean) is not tuple or type(self.std) is not tuple:
            raise DataError("scaler statistics must be immutable tuples")
        if (not self.mean or len(self.mean) != len(self.std) or
                type(self.sample_count) is not int or self.sample_count < 1):
            raise DataError("invalid scaler shape/count")
        if (type(self.training_only) is not bool or not self.training_only or
                any(type(s) not in (int, float) or s <= 0 or not math.isfinite(s) for s in self.std)):
            raise DataError("invalid/frozen scaler variance")
        if any(type(x) not in (int, float) or not math.isfinite(x) for x in self.mean):
            raise DataError("invalid scaler means")

    @property
    def hash(self) -> str:
        return digest(self)

    def transform(self, vector: tuple[float, ...], feature_hash: str) -> tuple[float, ...]:
        if (feature_hash != self.feature_hash or type(vector) is not tuple or
                len(vector) != len(self.mean)):
            raise DataError("scaler feature hash/shape mismatch")
        if any(type(v) not in (int, float) or not math.isfinite(v) for v in vector):
            raise DataError("non-finite scaler input")
        result = tuple((v - m) / s for v, m, s in zip(vector, self.mean, self.std))
        if not all(math.isfinite(x) for x in result):
            raise DataError("non-finite scaled observation")
        return result


def fit_train(spec: FeatureSpec, window: Window,
              samples: Iterable[tuple[int, tuple[float, ...]]],
              *, expected_train_window_hash: str | None = None) -> Scaler:
    """No fit/update API for validation and test; train scores only (never warmup)."""
    dataset = list(samples)
    if not dataset:
        raise DataError("cannot fit scaler without train score samples")
    if expected_train_window_hash is not None and expected_train_window_hash != digest(window):
        raise DataError("train fold hash mismatch")
    timestamps: set[int] = set()
    indexed_rows: list[tuple[int, tuple[float, ...]]] = []
    for ts, row in dataset:
        if not window.scoring(ts) or ts in timestamps:
            raise DataError("scaler fit outside train score window or duplicate time")
        timestamps.add(ts)
        if len(row) != len(spec.fields) or any(not isinstance(x, (float, int)) or
                                              isinstance(x, bool) or not math.isfinite(x) for x in row):
            raise DataError("non-finite or schema-incompatible scaler row")
        indexed_rows.append((ts, tuple(float(x) for x in row)))
    rows = [v for _, v in sorted(indexed_rows, key=lambda pair: pair[0])]
    cols = tuple(zip(*rows))
    means = tuple(math.fsum(col) / len(col) for col in cols)
    stds = tuple(max(math.sqrt(math.fsum((x - m)**2 for x in col) / len(col)), 1e-12)
                 for col, m in zip(cols, means))
    return Scaler(spec.hash, means, stds, digest(window), len(rows))
