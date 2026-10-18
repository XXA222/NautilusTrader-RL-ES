"""Causal projection from the canonical L2 store into Route B market rows."""
from __future__ import annotations

from rles_data.core import CuratedStore, DataError, MINUTE_NS, Window, require_ns

from .contracts import MarketReplay


def causal_frames_as_of(frames, *, decision_ts_ns: int):
    """Return only closed bars that were available at a declared decision barrier."""
    require_ns(decision_ts_ns)
    visible = tuple(frame for frame in frames if frame.available_at_ns <= decision_ts_ns)
    if any(frame.ts_close_ns > decision_ts_ns for frame in visible):
        raise DataError("future candle is visible before its close")
    if any(right.ts_close_ns - left.ts_close_ns != MINUTE_NS
           for left, right in zip(visible, visible[1:])):
        raise DataError("causal market history contains a gap or duplicate")
    return visible


def project_market_replay(store: CuratedStore, *, instrument: str, window: Window) -> MarketReplay:
    """Project only available 1m OHLCV whose close lies in a declared window."""
    if not isinstance(store, CuratedStore) or not isinstance(window, Window):
        raise DataError("canonical CuratedStore and explicit Window are required")
    events = tuple(event for event in store.events
                   if event.kind == "ohlcv_1m"
                   and event.payload.get("instrument") == instrument
                   and window.warmup_start_ns <= event.ts_event_ns < window.score_end_ns
                   and event.available_at_ns is not None
                   and event.available_at_ns <= window.score_end_ns)
    if any(event.available_at_ns is None for event in events):
        raise DataError("market replay contains bars with missing availability")
    return MarketReplay.from_events(events, dataset_hash=store.snapshot_hash,
                                    instrument=instrument)


def project_registered_train_replay(config: dict, window: dict, *, marker_sha256: str,
                                    label: str, instrument: str,
                                    allow_legacy_isolated_snapshot: bool = False) -> MarketReplay:
    """Reserve the development Train interval before decoding a curated payload."""
    from rles_orchestrator.research_access import read_registered_train_window

    store = read_registered_train_window(
        config, window, marker_sha256=marker_sha256, label=label,
        allow_legacy_isolated_snapshot=allow_legacy_isolated_snapshot,
    )
    replay_window = Window(window["start_ns"], window["start_ns"], window["end_ns"])
    return project_market_replay(store, instrument=instrument, window=replay_window)
