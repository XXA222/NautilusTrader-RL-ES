"""Explicit structural boundary from L2 features into a frozen L3 observation.

The bridge consumes L2 objects without importing that package, so the engine,
Gym, RL and ES distributions keep their declared dependency boundaries.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from .observation import ObservationSchema, VisibleFeature


def _field_columns(schema: ObservationSchema, feature_spec: Any) -> tuple[int, ...]:
    fields = getattr(feature_spec, "fields", None)
    if not isinstance(fields, tuple) or not fields:
        raise ValueError("L2 feature spec must contain immutable fields")
    expected = {"market." + field.name: field.unit for field in fields}
    actual = {name: unit for name, unit in zip(schema.names, schema.units)
              if name.startswith("market.")}
    if expected != actual or len(expected) != len(fields):
        raise ValueError("L2 feature fields do not match market observation schema")
    return tuple(schema.names.index("market." + field.name) for field in fields)


def visible_market_features(row: Any, feature_spec: Any, schema: ObservationSchema,
                            *, instrument: str, decision_ts_ns: int) -> dict[str, VisibleFeature]:
    """Map one point-in-time L2 FeatureRow to the Gym market observation fields."""
    _field_columns(schema, feature_spec)
    if (type(decision_ts_ns) is not int or decision_ts_ns <= 0 or
            type(instrument) is not str or not instrument or
            row.schema_hash != feature_spec.hash or row.instrument != instrument or
            row.decision_ts_ns != decision_ts_ns or
            type(row.available_at_ns) is not int or
            not 0 <= row.available_at_ns <= decision_ts_ns or
            not isinstance(row.values, tuple) or len(row.values) != len(feature_spec.fields)):
        raise ValueError("L2 feature row identity, clock or shape mismatch")
    result: dict[str, VisibleFeature] = {}
    for field, value in zip(feature_spec.fields, row.values):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
            raise ValueError("L2 feature row contains a non-finite value")
        result[field.name] = VisibleFeature(float(value), row.available_at_ns, field.unit)
    return result


class L2FrozenScalerAdapter:
    """Apply an L2 train-only Scaler to market columns, leaving account facts raw."""

    def __init__(self, schema: ObservationSchema, feature_spec: Any, scaler: Any,
                 *, expected_scaler_hash: str) -> None:
        self._columns = _field_columns(schema, feature_spec)
        if (not isinstance(expected_scaler_hash, str) or
                scaler.hash != expected_scaler_hash or scaler.training_only is not True or
                scaler.feature_hash != feature_spec.hash or
                len(scaler.mean) != len(self._columns) or
                len(scaler.std) != len(self._columns) or
                not callable(getattr(scaler, "transform", None))):
            raise ValueError("L2 train-fitted scaler hash, feature spec or shape mismatch")
        self._scaler = scaler
        self._expected_hash = expected_scaler_hash
        self._feature_hash = feature_spec.hash
        self._width = len(schema.names)

    @property
    def fingerprint(self) -> str:
        return self._scaler.hash

    @property
    def frozen(self) -> bool:
        return self._scaler.training_only is True

    def transform(self, matrix: np.ndarray) -> np.ndarray:
        if self.fingerprint != self._expected_hash or not self.frozen:
            raise ValueError("L2 scaler mutated after observation binding")
        values = np.asarray(matrix, dtype=np.float64)
        if values.shape != (1, self._width) or not np.isfinite(values).all():
            raise ValueError("invalid observation shape or values for L2 scaler")
        market = tuple(float(values[0, column]) for column in self._columns)
        scaled = self._scaler.transform(market, self._feature_hash)
        if len(scaled) != len(self._columns):
            raise ValueError("L2 scaler changed market feature count")
        result = values.copy()
        for column, value in zip(self._columns, scaled):
            result[0, column] = value
        if self.fingerprint != self._expected_hash or not self.frozen:
            raise ValueError("L2 scaler mutated during observation transform")
        return result
