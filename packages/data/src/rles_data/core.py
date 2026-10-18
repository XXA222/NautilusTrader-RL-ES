"""L2 Data Truth: immutable market facts, availability and quality gates.

This package deliberately contains no NautilusTrader imports and no L1 DTO clone.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from functools import cached_property
from datetime import datetime, timezone
from enum import StrEnum
from bisect import bisect_left, bisect_right
from hashlib import sha256
import json
import math
import re
from typing import Any, Iterable, Mapping, TYPE_CHECKING, cast

if TYPE_CHECKING:
    from .coverage import CoveragePolicy

MINUTE_NS = 60_000_000_000
DAY_NS = 1_440 * MINUTE_NS
MAX_NS = 2**63 - 1
CORE_KINDS = frozenset({"ohlcv_1m", "mark", "index", "funding_update", "funding_settlement", "instrument_rule"})
KINDS = CORE_KINDS | {"best_quote", "trade_flow_1s"}
HASH_RE = re.compile(r"[0-9a-f]{64}\Z")


class DataError(ValueError):
    """Fail-closed data, timeline or protocol violation."""


class FrozenDict(dict):
    """Deeply frozen JSON object; retains json/asdict interoperability."""

    def __init__(self, values: Mapping[str, Any] | Iterable[tuple[str, Any]] = ()):
        pairs = dict(values)
        if not all(isinstance(k, str) for k in pairs):
            raise DataError("JSON object keys must be strings")
        dict.__init__(self, ((k, freeze(v)) for k, v in pairs.items()))

    def _blocked(self, *args: Any, **kwargs: Any) -> None:
        raise TypeError("immutable market payload")

    # These immutable overrides intentionally replace several differently
    # typed built-in signatures; every operation fails before touching state.
    __setitem__ = __delitem__ = clear = pop = popitem = setdefault = update = _blocked  # type: ignore[assignment]
    __ior__ = _blocked  # type: ignore[assignment]

    def __copy__(self) -> FrozenDict:
        return self

    def __deepcopy__(self, memo: dict[int, Any]) -> FrozenDict:
        return self


def freeze(value: Any) -> Any:
    if isinstance(value, FrozenDict):
        return value
    if isinstance(value, dict):
        return FrozenDict(value)
    if isinstance(value, (tuple, list)):
        return tuple(freeze(v) for v in value)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise DataError(f"non-canonical JSON value: {type(value).__name__}")


def _plain(value: Any) -> Any:
    if isinstance(value, StrEnum):
        return value.value
    if is_dataclass(value) and not isinstance(value, type):
        return _plain(asdict(value))
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


def canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(_plain(value), sort_keys=True, separators=(",", ":"),
                          allow_nan=False, ensure_ascii=False).encode("utf-8")
    except (TypeError, ValueError, OverflowError) as exc:
        raise DataError("non-canonical manifest JSON") from exc


def digest(value: Any) -> str:
    return sha256(canonical_bytes(value)).hexdigest()


def require_hash(value: str) -> str:
    if not isinstance(value, str) or HASH_RE.fullmatch(value) is None:
        raise DataError("expected SHA-256 lowercase hexadecimal digest")
    return value


def require_ns(value: int) -> int:
    if type(value) is not int or not 0 <= value <= MAX_NS:
        raise DataError("timestamp must be nonnegative UTC int64 nanoseconds")
    return value


def parse_ns(value: str) -> int:
    """Explicit UTC ISO-8601 only, preserving up to nine fractional digits."""
    if not isinstance(value, str) or re.fullmatch(
        r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,9})?Z", value
    ) is None:
        raise DataError("expected explicit UTC ISO-8601 timestamp ending in Z")
    main, _, tail = value[:-1].partition(".")
    try:
        dt = datetime.fromisoformat(main).replace(tzinfo=timezone.utc)
        seconds = int((dt - datetime(1970, 1, 1, tzinfo=timezone.utc)).total_seconds())
        return require_ns(seconds * 1_000_000_000 + int(tail.ljust(9, "0") or "0"))
    except (ValueError, OverflowError) as exc:
        raise DataError(f"invalid UTC time {value}") from exc


class Availability(StrEnum):
    NATIVE = "native"
    DERIVED = "derived"
    MISSING = "missing"


class Grade(StrEnum):
    A = "DQ-A"
    B = "DQ-B"
    C = "DQ-C"
    D = "DQ-D"


@dataclass(frozen=True)
class AvailabilityPolicy:
    """Availability is never silently the exchange event time.

    Derived availability requires a strictly positive, source-approved buffer.
    ts_init is also observed as a causal lower bound.
    """
    conservative_delay_ns: int | None = None

    def __post_init__(self) -> None:
        if self.conservative_delay_ns is not None:
            if type(self.conservative_delay_ns) is not int or self.conservative_delay_ns <= 0:
                raise DataError("derivation delay must be strictly positive ns")

    def resolve(self, event_ns: int, init_ns: int, native_ns: int | None) -> tuple[int | None, Availability]:
        require_ns(event_ns)
        require_ns(init_ns)
        if native_ns is not None:
            available = require_ns(native_ns)
            if available < max(event_ns, init_ns):
                raise DataError("native availability precedes event/init timestamp")
            return available, Availability.NATIVE
        if self.conservative_delay_ns is None:
            return None, Availability.MISSING
        available = require_ns(max(event_ns, init_ns) + self.conservative_delay_ns)
        return available, Availability.DERIVED


@dataclass(frozen=True)
class Event:
    event_id: str
    kind: str
    ts_event_ns: int
    ts_init_ns: int
    available_at_ns: int | None
    availability_source: Availability
    source_hash: str
    payload: Mapping[str, Any]
    provenance: str

    def __post_init__(self) -> None:
        require_ns(self.ts_event_ns)
        require_ns(self.ts_init_ns)
        require_hash(self.source_hash)
        if not isinstance(self.availability_source, Availability):
            raise DataError("invalid availability source")
        if self.available_at_ns is not None:
            require_ns(self.available_at_ns)
        if (self.available_at_ns is None) != (self.availability_source == Availability.MISSING):
            raise DataError("missing availability marker inconsistent")
        if self.available_at_ns is not None and self.available_at_ns < max(self.ts_event_ns, self.ts_init_ns):
            raise DataError("availability precedes event/init time")
        if not self.event_id or self.kind not in KINDS or not self.provenance:
            raise DataError("event_id, kind, provenance required")
        if not isinstance(self.payload, Mapping):
            raise DataError("event payload must be a JSON object")
        object.__setattr__(self, "payload", FrozenDict(self.payload))
        from .schemas import validate_payload
        validate_payload(self.kind, dict(self.payload), self.ts_event_ns)
        canonical_bytes(self.payload)

    def visible(self, decision_ns: int) -> bool:
        return self.available_at_ns is not None and self.available_at_ns <= require_ns(decision_ns)

    @cached_property
    def hash(self) -> str:
        return digest(self)


@dataclass(frozen=True)
class DataQualityReport:
    grade: Grade
    findings: tuple[str, ...]
    missing_minutes: int
    counts: Mapping[str, int]
    snapshot_hash: str
    start_ns: int | None = None
    end_ns: int | None = None
    duplicate_count: int = 0
    conflict_count: int = 0
    verified_coverage: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.grade, Grade):
            raise DataError("invalid data quality grade")
        if (type(self.verified_coverage) is not tuple
                or self.verified_coverage != tuple(sorted(set(self.verified_coverage)))
                or set(self.verified_coverage) - KINDS):
            raise DataError("invalid verified economic coverage")
        if (type(self.findings) is not tuple or
                any(type(item) is not str or not item for item in self.findings) or
                self.findings != tuple(sorted(set(self.findings)))):
            raise DataError("quality findings must be sorted, unique nonempty strings")
        if not isinstance(self.counts, Mapping):
            raise DataError("quality counts must be a mapping")
        counts = dict(self.counts)
        if any(type(kind) is not str or kind not in KINDS or type(count) is not int or count < 0
               for kind, count in counts.items()):
            raise DataError("invalid event counts")
        object.__setattr__(self, "counts", FrozenDict(counts))
        require_hash(self.snapshot_hash)
        for name in ("missing_minutes", "duplicate_count", "conflict_count"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise DataError(f"{name} must be a nonnegative integer")
        if self.start_ns is not None:
            require_ns(self.start_ns)
        if self.end_ns is not None:
            require_ns(self.end_ns)
        if self.start_ns is not None and self.end_ns is not None and self.start_ns >= self.end_ns:
            raise DataError("invalid quality report interval")

    @property
    def hash(self) -> str:
        return digest(self)


@dataclass(frozen=True)
class DatasetCard:
    schema_version: str
    snapshot_hash: str
    sources: tuple[str, ...]
    start_ns: int
    end_ns: int
    event_count: int
    quality_hash: str
    feature_hash: str | None = None
    raw_hashes: tuple[str, ...] = ()
    missing_minutes: int = 0
    grade: Grade | None = None

    def __post_init__(self) -> None:
        if self.schema_version != "0.4.0":
            raise DataError("unsupported dataset card schema")
        require_hash(self.snapshot_hash)
        require_hash(self.quality_hash)
        if type(self.sources) is not tuple or not self.sources or any(type(x) is not str or not x for x in self.sources):
            raise DataError("dataset card sources must be nonempty strings")
        if self.sources != tuple(sorted(set(self.sources))):
            raise DataError("dataset card sources must be sorted and unique")
        for hash_value in self.raw_hashes:
            require_hash(hash_value)
        if type(self.raw_hashes) is not tuple or self.raw_hashes != tuple(sorted(set(self.raw_hashes))):
            raise DataError("dataset card raw hashes must be sorted and unique")
        if self.feature_hash is not None:
            require_hash(self.feature_hash)
        require_ns(self.start_ns)
        require_ns(self.end_ns)
        if (self.start_ns >= self.end_ns or type(self.event_count) is not int or self.event_count < 1 or
                type(self.missing_minutes) is not int or self.missing_minutes < 0 or
                (self.grade is not None and not isinstance(self.grade, Grade))):
            raise DataError("invalid dataset card")

    @property
    def hash(self) -> str:
        return digest(self)


@dataclass(frozen=True)
class FeatureField:
    name: str
    dtype: str
    unit: str
    source_kind: str
    lookback_bars: int
    timeframe_minutes: int
    recipe: str | None = None
    max_age_ns: int | None = None

    def __post_init__(self) -> None:
        if (type(self.name) is not str or not self.name or type(self.unit) is not str or not self.unit or
                self.dtype not in ("float64", "float32") or type(self.source_kind) is not str or
                type(self.timeframe_minutes) is not int):
            raise DataError("invalid feature schema")
        if self.source_kind not in KINDS or self.timeframe_minutes not in (1, 5, 15, 60, 240, 480, 1440):
            raise DataError("invalid source/timeframe")
        if type(self.lookback_bars) is not int or self.lookback_bars < 1:
            raise DataError("lookback must be a positive integer")
        if self.max_age_ns is not None and (type(self.max_age_ns) is not int or self.max_age_ns <= 0):
            raise DataError("max_age_ns must be positive")
        if self.recipe is not None and self.recipe not in (
            "return", "close", "volume_mean", "volatility", "trend_efficiency", "latest", "basis",
        ):
            raise DataError("unsupported feature recipe")
        if self.source_kind == "instrument_rule":
            raise DataError("instrument rules are accessed via RuleTimeline, not scalar features")
        if self.source_kind != "ohlcv_1m":
            if self.timeframe_minutes != 1 or self.lookback_bars != 1 or self.max_age_ns is None:
                raise DataError("event feature must use 1m, lookback=1 and explicit TTL")
            if self.recipe not in ("latest", "basis"):
                raise DataError("non-OHLCV source requires explicit latest or basis recipe")
            if self.recipe == "basis" and self.source_kind != "mark":
                raise DataError("basis must use mark/index as-of prices")
        elif self.recipe is not None and self.recipe not in (
            "return", "close", "volume_mean", "volatility", "trend_efficiency",
        ):
            raise DataError("unsupported OHLCV feature recipe")


@dataclass(frozen=True)
class FeatureSpec:
    schema_version: str
    fields: tuple[FeatureField, ...]

    def __post_init__(self) -> None:
        if (self.schema_version != "0.4.0" or type(self.fields) is not tuple or not self.fields or
                any(not isinstance(field, FeatureField) for field in self.fields) or
                len({f.name for f in self.fields}) != len(self.fields)):
            raise DataError("duplicate/empty feature schema")

    @cached_property
    def hash(self) -> str:
        return digest(self)

    @property
    def max_lookback_ns(self) -> int:
        return max(max((field.lookback_bars + int(
                           field.recipe in {"return", "trend_efficiency"} or field.name.startswith("return_")))
                       * field.timeframe_minutes * MINUTE_NS, field.max_age_ns or 0)
                   for field in self.fields)


@dataclass(frozen=True)
class Window:
    warmup_start_ns: int
    score_start_ns: int
    score_end_ns: int

    def __post_init__(self) -> None:
        if not 0 <= require_ns(self.warmup_start_ns) <= require_ns(self.score_start_ns) < require_ns(self.score_end_ns):
            raise DataError("invalid score/context window")

    def scoring(self, ts: int) -> bool:
        return self.score_start_ns <= require_ns(ts) < self.score_end_ns

    def readable(self, ts: int) -> bool:
        return self.warmup_start_ns <= require_ns(ts) < self.score_end_ns


@dataclass(frozen=True)
class FoldManifest:
    schema_version: str
    dataset_hash: str
    feature_hash: str
    train: Window
    validation: Window
    test: Window
    purge_ns: int
    embargo_ns: int
    scaler_hash: str | None
    terminal_policy: str = "force_flatten_with_cost"
    reset_positions: bool = True
    reset_orders: bool = True
    reset_replay: bool = True
    freeze_validation_scaler: bool = True

    def __post_init__(self) -> None:
        if self.schema_version != "0.4.0":
            raise DataError("unsupported fold schema")
        require_hash(self.dataset_hash)
        require_hash(self.feature_hash)
        if self.scaler_hash is not None:
            require_hash(self.scaler_hash)
        if self.terminal_policy != "force_flatten_with_cost":
            raise DataError("unsupported fold/terminal policy")
        if (not all(isinstance(window, Window) for window in (self.train, self.validation, self.test)) or
                type(self.purge_ns) is not int or type(self.embargo_ns) is not int or
                min(self.purge_ns, self.embargo_ns) < 0):
            raise DataError("invalid purge/embargo")
        flags = (self.reset_positions, self.reset_orders, self.reset_replay, self.freeze_validation_scaler)
        if any(type(flag) is not bool for flag in flags) or not all(flags):
            raise DataError("cross-split state carry is prohibited")
        gap = self.purge_ns + self.embargo_ns
        if (self.train.score_end_ns + gap > self.validation.score_start_ns or
                self.validation.score_end_ns + gap > self.test.score_start_ns):
            raise DataError("fold score windows violate purge/embargo")

    @property
    def hash(self) -> str:
        return digest(self)


def _snapshot_digest(events: Iterable[Event]) -> str:
    """Hash the same canonical JSON array without materializing the full array.

    The per-event canonical representation is unchanged; brackets and comma
    separators preserve existing snapshot IDs and commit manifests exactly.
    Temporary encoding memory is bounded by the largest individual event.
    """
    value = sha256(b"[")
    first = True
    for event in events:
        if not first:
            value.update(b",")
        # Event is a validated frozen DTO. Encode its known fields directly so
        # hashing millions of rows does not recursively dataclasses.asdict/
        # deepcopy every immutable payload before JSON encoding.
        encoded = json.dumps({
            "event_id": event.event_id,
            "kind": event.kind,
            "ts_event_ns": event.ts_event_ns,
            "ts_init_ns": event.ts_init_ns,
            "available_at_ns": event.available_at_ns,
            "availability_source": event.availability_source.value,
            "source_hash": event.source_hash,
            "payload": event.payload,
            "provenance": event.provenance,
        }, sort_keys=True, separators=(",", ":"), allow_nan=False,
            ensure_ascii=False).encode("utf-8")
        value.update(encoded)
        first = False
    value.update(b"]")
    return value.hexdigest()


class CuratedStore:
    def __init__(self, events: Iterable[Event]):
        seen: dict[str, Event] = {}
        duplicates = 0
        for ev in events:
            if not isinstance(ev, Event):
                raise DataError("curated input requires Event")
            previous = seen.get(ev.event_id)
            if previous is not None:
                if previous.hash != ev.hash:
                    raise DataError(f"conflicting event ID: {ev.event_id}")
                duplicates += 1
            else:
                seen[ev.event_id] = ev
        self.events = tuple(sorted(seen.values(), key=lambda e: (
            e.available_at_ns if e.available_at_ns is not None else MAX_NS + 1,
            e.ts_event_ns, e.ts_init_ns, e.kind, e.event_id)))
        self.instruments = frozenset(cast(str, event.payload["instrument"])
                                     for event in self.events
                                     if isinstance(event.payload.get("instrument"), str))
        visible_events = tuple(event for event in self.events if event.available_at_ns is not None)
        self._visible_events = visible_events
        self._visible_times = tuple(cast(int, event.available_at_ns) for event in visible_events)
        visible_by_kind: dict[str, list[Event]] = {}
        for event in visible_events:
            visible_by_kind.setdefault(event.kind, []).append(event)
        self._visible_by_kind = {kind: tuple(events) for kind, events in visible_by_kind.items()}
        self._visible_times_by_kind = {
            kind: tuple(cast(int, event.available_at_ns) for event in events)
            for kind, events in self._visible_by_kind.items()
        }
        visible_by_event_time: dict[str, list[Event]] = {}
        for event in visible_events:
            visible_by_event_time.setdefault(event.kind, []).append(event)
        self._visible_by_event_time = {
            kind: tuple(sorted(events, key=lambda event: (event.ts_event_ns,
                                                           cast(int, event.available_at_ns),
                                                           event.event_id)))
            for kind, events in visible_by_event_time.items()
        }
        self._visible_event_times_by_kind = {
            kind: tuple(event.ts_event_ns for event in events)
            for kind, events in self._visible_by_event_time.items()
        }
        # Derived feature indexes are lazy and excluded from snapshot identity.
        self._feature_cache: dict[tuple[Any, ...], Any] = {}
        rules_by_instrument: dict[str, list[Event]] = {}
        for event in self._visible_by_kind.get("instrument_rule", ()):
            rules_by_instrument.setdefault(str(event.payload["instrument"]), []).append(event)
        self._rule_indexes: dict[str, tuple[tuple[int, ...], tuple[Event, ...], tuple[bool, ...]]] = {}
        for instrument, rules in rules_by_instrument.items():
            # A rule becomes queryable only when both source availability and
            # exchange-effective time have arrived. Prefix maxima preserve the
            # old visible/filter/sort selection for as-of queries.
            ordered = sorted(rules, key=lambda event: (
                max(cast(int, event.available_at_ns), int(event.payload.get("effective_ns", event.ts_event_ns))),
                int(event.payload.get("effective_ns", event.ts_event_ns)),
                cast(int, event.available_at_ns), event.event_id,
            ))
            eligible_times: list[int] = []
            latest_events: list[Event] = []
            latest_conflicts: list[bool] = []
            payload_digests_by_effective: dict[int, set[str]] = {}
            latest: Event | None = None
            for event in ordered:
                effective = int(event.payload.get("effective_ns", event.ts_event_ns))
                available = cast(int, event.available_at_ns)
                payload_digests_by_effective.setdefault(effective, set()).add(digest(event.payload))
                if latest is None or (
                    effective, available, event.event_id
                ) > (
                    int(latest.payload.get("effective_ns", latest.ts_event_ns)),
                    cast(int, latest.available_at_ns), latest.event_id,
                ):
                    latest = event
                latest_effective = int(latest.payload.get("effective_ns", latest.ts_event_ns))
                eligible_times.append(max(effective, available))
                latest_events.append(latest)
                latest_conflicts.append(len(payload_digests_by_effective[latest_effective]) > 1)
            self._rule_indexes[instrument] = (
                tuple(eligible_times), tuple(latest_events), tuple(latest_conflicts),
            )
        self.duplicate_count = duplicates
        self.snapshot_hash = _snapshot_digest(self.events)
        self._conflicts = self._detect_conflicts()

    def _detect_conflicts(self) -> tuple[str, ...]:
        economic_keys: dict[tuple[Any, ...], str] = {}
        conflicting: set[str] = set()
        for event in self.events:
            p = event.payload
            key: tuple[Any, ...]
            if event.kind == "ohlcv_1m":
                key = (event.kind, p.get("instrument"), event.ts_event_ns)
            elif event.kind == "instrument_rule":
                key = (event.kind, p.get("instrument"), p.get("effective_ns", event.ts_event_ns))
            elif event.kind == "funding_settlement":
                key = (event.kind, p.get("instrument"), p.get("settlement_ns"))
            else:
                continue  # mark/index/funding updates may legitimately share event timestamps
            fingerprint = digest(p)
            old = economic_keys.setdefault(key, fingerprint)
            if old != fingerprint:
                conflicting.add(str(key))
        rules = sorted((e for e in self.events if e.kind == "instrument_rule"),
                       key=lambda e: (str(e.payload["instrument"]), e.payload["effective_ns"], e.event_id))
        for i, earlier in enumerate(rules):
            bound = earlier.payload.get("valid_until_ns")
            if bound is None:
                continue
            for later in rules[i + 1:]:
                if earlier.payload["instrument"] != later.payload["instrument"]:
                    break
                if later.payload["effective_ns"] >= bound:
                    break
                if digest(earlier.payload) != digest(later.payload):
                    conflicting.add(f"rule_overlap:{earlier.event_id}:{later.event_id}")
        return tuple(sorted(conflicting))

    @property
    def conflicts(self) -> tuple[str, ...]:
        return self._conflicts

    def visible(self, decision_ns: int, kind: str | None = None) -> tuple[Event, ...]:
        require_ns(decision_ns)
        if kind is None:
            events, times = self._visible_events, self._visible_times
        else:
            events = self._visible_by_kind.get(kind, ())
            times = self._visible_times_by_kind.get(kind, ())
        return events[:bisect_right(times, decision_ns)]

    def visible_event_window(self, decision_ns: int, start_event_ns: int,
                             kind: str) -> tuple[Event, ...]:
        """Return as-of facts in an event-time window without copying older history."""
        require_ns(decision_ns)
        require_ns(start_event_ns)
        if type(kind) is not str or not kind or start_event_ns > decision_ns:
            raise DataError("invalid visible event-time window")
        events = self._visible_by_event_time.get(kind, ())
        times = self._visible_event_times_by_kind.get(kind, ())
        first = bisect_left(times, start_event_ns)
        last = bisect_right(times, decision_ns)
        return tuple(event for event in events[first:last]
                     if event.available_at_ns is not None
                     and event.available_at_ns <= decision_ns)

    def available_event_window(self, start_available_ns: int, end_available_ns: int,
                               *, kind: str | None = None) -> tuple[Event, ...]:
        """Return source-visible facts in an inclusive availability-time window."""
        require_ns(start_available_ns)
        require_ns(end_available_ns)
        if start_available_ns > end_available_ns:
            raise DataError("invalid availability-time window")
        if kind is None:
            events, times = self._visible_events, self._visible_times
        else:
            if type(kind) is not str or kind not in KINDS:
                raise DataError("unknown availability-time event kind")
            events = self._visible_by_kind.get(kind, ())
            times = self._visible_times_by_kind.get(kind, ())
        first = bisect_left(times, start_available_ns)
        last = bisect_right(times, end_available_ns)
        return events[first:last]

    def latest_rule(self, decision_ns: int, instrument: str) -> Event:
        require_ns(decision_ns)
        index = self._rule_indexes.get(instrument)
        if index is None:
            raise DataError("instrument rule unavailable at this decision time")
        eligible_times, latest_events, latest_conflicts = index
        position = bisect_right(eligible_times, decision_ns) - 1
        if position < 0:
            raise DataError("instrument rule unavailable at this decision time")
        latest = latest_events[position]
        if latest.payload.get("valid_until_ns") is not None and decision_ns >= latest.payload["valid_until_ns"]:
            raise DataError("latest known rule has expired; fallback to older version forbidden")
        if latest_conflicts[position]:
            raise DataError("conflicting instrument rules at decision time")
        return latest

    def quality(self, required: tuple[str, ...] = (
        "ohlcv_1m", "mark", "index", "funding_update", "funding_settlement", "instrument_rule"
    ), start_ns: int | None = None, end_ns: int | None = None,
                expected_instruments: tuple[str, ...] | None = None,
                coverage_policy: CoveragePolicy | None = None) -> DataQualityReport:
        if coverage_policy is not None:
            from .coverage import CoveragePolicy
            if not isinstance(coverage_policy, CoveragePolicy):
                raise DataError("coverage policy must be a CoveragePolicy")
        if start_ns is not None:
            require_ns(start_ns)
        if end_ns is not None:
            require_ns(end_ns)
        if start_ns is not None and end_ns is not None and start_ns >= end_ns:
            raise DataError("invalid quality interval")
        if expected_instruments is not None:
            if (type(expected_instruments) is not tuple or not expected_instruments or
                    any(type(symbol) is not str or not symbol for symbol in expected_instruments) or
                    len(set(expected_instruments)) != len(expected_instruments)):
                raise DataError("expected_instruments must be a nonempty tuple of unique symbols")
        if len(set(required)) != len(required) or set(required) - KINDS:
            raise DataError("invalid required market event types")
        selected = [e for e in self.events if (start_ns is None or e.ts_event_ns >= start_ns)
                    and (end_ns is None or e.ts_event_ns < end_ns)]
        bar_instruments = {e.payload["instrument"] for e in selected if e.kind == "ohlcv_1m"}
        if expected_instruments is not None:
            unexpected = bar_instruments - set(expected_instruments)
            if unexpected:
                raise DataError(f"unexpected OHLCV instruments: {sorted(unexpected)}")
            instrument_scope = expected_instruments
        else:
            instrument_scope = tuple(sorted(bar_instruments))
        allowed_instruments = set(instrument_scope)
        observed_instruments = {e.payload["instrument"] for e in selected}
        unexpected_facts = observed_instruments - allowed_instruments
        if unexpected_facts:
            raise DataError(f"unexpected market-fact instruments: {sorted(unexpected_facts)}")
        counts = {kind: sum(e.kind == kind for e in selected) for kind in required}
        counts_by_instrument = {
            symbol: {kind: sum(e.kind == kind and e.payload.get("instrument") == symbol
                                for e in selected) for kind in required}
            for symbol in instrument_scope
        }
        findings: set[str] = set()
        grade = Grade.A
        bars_by_instrument = {
            symbol: sorted({e.ts_event_ns for e in selected
                            if e.kind == "ohlcv_1m" and e.payload["instrument"] == symbol})
            for symbol in instrument_scope
        }
        missing_instrument_bars = any(not minutes for minutes in bars_by_instrument.values())
        if not selected or not any(bars_by_instrument.values()):
            grade = Grade.D
            findings.add("empty_or_missing_ohlcv")
        elif missing_instrument_bars:
            grade = Grade.D
            findings.add("expected_instrument_ohlcv_missing")
        missing = 0
        if start_ns is not None and end_ns is not None:
            first = ((start_ns + MINUTE_NS - 1) // MINUTE_NS) * MINUTE_NS
            expected = max(0, (end_ns - 1 - first) // MINUTE_NS + 1)
            missing = sum(max(0, expected - len(minutes)) for minutes in bars_by_instrument.values())
        else:
            missing = sum(max(0, (b - a) // MINUTE_NS - 1)
                          for minutes in bars_by_instrument.values()
                          for a, b in zip(minutes, minutes[1:]))
        if missing:
            grade = Grade.D
            findings.add("critical_ohlcv_gap")
        funding_kinds = tuple(k for k in ("funding_update", "funding_settlement") if k in required)
        if instrument_scope:
            funding_missing = any(counts_by_instrument[symbol][kind] == 0
                                  and not (kind == "funding_settlement" and coverage_policy is not None
                                           and not coverage_policy.settlement_times_ns.get(symbol, (1,)))
                                  for symbol in instrument_scope for kind in funding_kinds)
        else:
            funding_missing = any(counts.get(kind, 0) == 0 for kind in funding_kinds)
        if funding_missing and coverage_policy is None:
            grade = Grade.D
            findings.add("funding_untrusted")
        if self._conflicts:
            grade = Grade.D
            findings.add("conflicting_economic_facts")
        if any(e.availability_source == Availability.MISSING for e in selected):
            grade = Grade.D
            findings.add("availability_missing")
        if instrument_scope:
            optional_missing = any(
                counts_by_instrument[symbol][kind] == 0
                for symbol in instrument_scope for kind in ("mark", "index") if kind in required
            )
        else:
            optional_missing = any(counts.get(kind, 0) == 0
                                   for kind in ("mark", "index") if kind in required)
        rule_missing = False
        if "instrument_rule" in required:
            if instrument_scope:
                for symbol in instrument_scope:
                    selected_rule = counts_by_instrument[symbol]["instrument_rule"] > 0
                    prior_rule = start_ns is not None and any(
                        e.kind == "instrument_rule" and e.payload.get("instrument") == symbol
                        and e.available_at_ns is not None and e.available_at_ns <= start_ns
                        and e.payload["effective_ns"] <= start_ns
                        and (e.payload.get("valid_until_ns") is None or e.payload["valid_until_ns"] > start_ns)
                        for e in self.events
                    )
                    rule_missing = rule_missing or (not selected_rule and not prior_rule)
            else:
                selected_rule = counts.get("instrument_rule", 0) > 0
                prior_rule = start_ns is not None and any(
                    e.kind == "instrument_rule" and e.available_at_ns is not None
                    and e.available_at_ns <= start_ns and e.payload["effective_ns"] <= start_ns
                    and (e.payload.get("valid_until_ns") is None or e.payload["valid_until_ns"] > start_ns)
                    for e in self.events
                )
                rule_missing = not selected_rule and not prior_rule
        quote_missing = (
            "best_quote" in required and any(counts_by_instrument[symbol]["best_quote"] == 0
                                               for symbol in instrument_scope)
        )
        if grade != Grade.D and (optional_missing or rule_missing) and coverage_policy is None:
            grade = Grade.C
            findings.add("mark_index_or_rule_missing")
        if grade != Grade.D and quote_missing:
            grade = Grade.C
            findings.add("best_quote_missing")
        if any(e.availability_source == Availability.DERIVED for e in selected):
            findings.add("conservative_availability")
            if grade == Grade.A:
                grade = Grade.B
        if not required or set(required) not in (CORE_KINDS, KINDS):
            findings.add("partial_capability_assessment")
        verified_coverage: tuple[str, ...] = ()
        if coverage_policy is None:
            findings.add("economic_coverage_unverified")
            if grade in (Grade.A, Grade.B):
                grade = Grade.C
        else:
            from .coverage import CoveragePolicy, assess_coverage
            if not isinstance(coverage_policy, CoveragePolicy) or start_ns is None or end_ns is None:
                raise DataError("coverage audit requires policy and explicit bounded interval")
            coverage_findings, verified_coverage = assess_coverage(
                self, coverage_policy, start_ns=start_ns, end_ns=end_ns, instruments=instrument_scope,
            )
            findings.update(coverage_findings)
            if {"funding_update_coverage_gap", "funding_settlement_coverage_gap"} & findings:
                grade = Grade.D
            elif coverage_findings and grade != Grade.D:
                grade = Grade.C
        return DataQualityReport(grade, tuple(sorted(findings)), missing, counts,
                                 self.snapshot_hash, start_ns, end_ns,
                                 self.duplicate_count, len(self._conflicts), verified_coverage)

    def require_usable(self, report: DataQualityReport, purpose: str,
                       *, time_sensitivity_approved: bool = False) -> None:
        if report.snapshot_hash != self.snapshot_hash:
            raise DataError("quality report snapshot mismatch")
        if purpose not in ("directional", "rl_es", "leveraged", "liquidation", "settlement"):
            raise DataError("unknown data use case")
        if report.grade == Grade.D:
            raise DataError("DQ-D blocks all training and evaluation")
        if report.grade == Grade.C and purpose != "directional":
            raise DataError("DQ-C permits directional-only, low-leverage research")
        if report.grade == Grade.B and purpose in ("leveraged", "liquidation", "settlement") and not time_sensitivity_approved:
            raise DataError("DQ-B requires explicit timing sensitivity approval")
        if "partial_capability_assessment" in report.findings and purpose != "directional":
            raise DataError("partial DQ audit does not authorize full capability")
        if purpose != "directional" and not {
            "mark", "index", "funding_update", "funding_settlement", "instrument_rule"
        }.issubset(report.verified_coverage):
            raise DataError("full economic interval coverage required for this capability")

    def card(self, report: DataQualityReport, *, feature_hash: str | None = None,
             start_ns: int | None = None, end_ns: int | None = None) -> DatasetCard:
        if report.snapshot_hash != self.snapshot_hash or not self.events:
            raise DataError("cannot make card from empty/mismatched snapshot")
        start = start_ns if start_ns is not None else min(e.ts_event_ns for e in self.events)
        end = end_ns if end_ns is not None else max(e.ts_event_ns for e in self.events) + 1
        return DatasetCard("0.4.0", self.snapshot_hash,
                           tuple(sorted({e.provenance for e in self.events})),
                           start, end, len(self.events), report.hash, feature_hash,
                           tuple(sorted({e.source_hash for e in self.events})),
                           report.missing_minutes, report.grade)
