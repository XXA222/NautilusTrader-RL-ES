"""Deterministic 60/15/15 and 360/15/15 causal WFO fold manifests."""
from __future__ import annotations

from dataclasses import replace
from .core import DAY_NS, DataError, FeatureSpec, FoldManifest, Window, digest, require_hash, require_ns
from .features import Scaler


def build_folds(*, start_ns: int, end_ns: int, train_days: int,
                validation_days: int = 15, test_days: int = 15, step_days: int = 15,
                warmup_ns: int, feature_lookback_ns: int,
                execution_lag_guard_ns: int, embargo_ns: int,
                dataset_hash: str, feature_hash: str,
                scaler_hash: str | None = None,
                data_start_ns: int | None = None) -> tuple[FoldManifest, ...]:
    """start_ns is first train *score* timestamp, not file beginning.

    data_start_ns should equal the earliest verified source time.  Warmup is read-only;
    purge and embargo gaps never earn rewards or fit statistics. Each split resets
    positions/orders/replay and freezes fitted statistics (consumer must enforce).
    """
    for name, value in (("start_ns", start_ns), ("end_ns", end_ns),
                        ("warmup_ns", warmup_ns), ("feature_lookback_ns", feature_lookback_ns),
                        ("execution_lag_guard_ns", execution_lag_guard_ns), ("embargo_ns", embargo_ns)):
        require_ns(value)
    if data_start_ns is not None:
        require_ns(data_start_ns)
    require_hash(dataset_hash)
    require_hash(feature_hash)
    if scaler_hash is not None:
        raise DataError("one scaler hash cannot be assigned to every WFO fold; bind after each train fit")
    if start_ns >= end_ns or (train_days, validation_days, test_days, step_days) not in (
        (60, 15, 15, 15), (360, 15, 15, 15)):
        raise DataError("only approved rolling 60/15/15 or 360/15/15 protocol")
    if warmup_ns < feature_lookback_ns:
        raise DataError("warmup must cover max feature lookback")
    if start_ns < warmup_ns or (data_start_ns is not None and start_ns - warmup_ns < data_start_ns):
        raise DataError("missing dataset coverage for initial warmup")
    purge = max(feature_lookback_ns, execution_lag_guard_ns)
    gap = purge + embargo_ns
    results: list[FoldManifest] = []
    anchor = start_ns
    while True:
        train_end = anchor + train_days * DAY_NS
        val_start = train_end + gap
        val_end = val_start + validation_days * DAY_NS
        test_start = val_end + gap
        test_end = test_start + test_days * DAY_NS
        if test_end > end_ns:
            break
        result = FoldManifest("0.4.0", dataset_hash, feature_hash,
                              Window(anchor - warmup_ns, anchor, train_end),
                              Window(val_start - warmup_ns, val_start, val_end),
                              Window(test_start - warmup_ns, test_start, test_end),
                              purge, embargo_ns, None)
        results.append(result)
        anchor += step_days * DAY_NS
    return tuple(results)


def build_folds_from_spec(*, spec: FeatureSpec, **kwargs) -> tuple[FoldManifest, ...]:
    """Avoid hard-coded feature lookback in fold builders."""
    if "feature_lookback_ns" in kwargs or "feature_hash" in kwargs:
        raise DataError("derived spec parameters may not be overridden")
    return build_folds(feature_lookback_ns=spec.max_lookback_ns,
                       feature_hash=spec.hash, **kwargs)


def bind_fold_scaler(fold: FoldManifest, scaler: Scaler) -> FoldManifest:
    """Commit a scaler only if it was fitted on this *exact* fold train window."""
    # Warmup is read-only input needed to construct the first score row.  It is
    # deliberately excluded from fitted statistics, so the scaler identity is
    # bound to the Train score window rather than the wider read window.
    score_window = Window(fold.train.score_start_ns, fold.train.score_start_ns,
                          fold.train.score_end_ns)
    if scaler.feature_hash != fold.feature_hash or scaler.train_window_hash != digest(score_window):
        raise DataError("scaler was not fitted on this fold's train score window")
    if fold.scaler_hash is not None and fold.scaler_hash != scaler.hash:
        raise DataError("fold scaler already bound to a different artifact")
    return replace(fold, scaler_hash=scaler.hash)
