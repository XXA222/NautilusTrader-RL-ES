"""Nautilus-RLES AI-02 L2 Data Truth production-source package."""

from .core import (
    Availability, AvailabilityPolicy, CuratedStore, DataError, DataQualityReport,
    DatasetCard, Event, FeatureField, FeatureSpec, FoldManifest, Grade, Window,
    digest, parse_ns, require_ns,
)
from .features import (
    FeatureRow, Scaler, aggregate_closed, fit_train, make_feature_row, make_features,
)
from .folds import bind_fold_scaler, build_folds, build_folds_from_spec
from .io import (
    ingest_arrow, ingest_parquet, read_curated, store_raw, store_raw_file, validate_payload,
    write_curated, write_fact_json,
)
from .quality import GapAction, GapDecision, RuleTimeline, decide_gap
from .contracts import to_market_event
from .binance_archive import import_day, import_range

__all__ = [
    "Availability", "AvailabilityPolicy", "CuratedStore", "DataError", "DataQualityReport",
    "DatasetCard", "Event", "FeatureField", "FeatureRow", "FeatureSpec", "FoldManifest",
    "Grade", "Window", "Scaler", "GapAction", "GapDecision", "RuleTimeline", "digest",
    "parse_ns", "require_ns", "aggregate_closed", "fit_train", "make_feature_row",
    "make_features", "bind_fold_scaler", "build_folds", "build_folds_from_spec", "ingest_parquet",
    "ingest_arrow", "read_curated", "store_raw", "store_raw_file", "validate_payload", "write_curated",
    "write_fact_json", "decide_gap",
    "to_market_event",
    "import_day", "import_range",
]
