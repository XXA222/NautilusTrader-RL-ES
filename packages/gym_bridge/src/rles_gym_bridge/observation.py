"""Fixed-order observations from L2-visible features and native account facts."""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from typing import Mapping, Any

import numpy as np


@dataclass(frozen=True, slots=True)
class ObservationSchema:
    schema_version: str
    names: tuple[str, ...]
    units: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.names, (tuple, list)) or not isinstance(self.units, (tuple, list)):
            raise ValueError("observation names and units must be sequences of strings")
        object.__setattr__(self, "names", tuple(self.names))
        object.__setattr__(self, "units", tuple(self.units))
        if self.schema_version != "0.4.0" or not self.names or len(self.names) != len(self.units):
            raise ValueError("invalid observation schema")
        if (any(type(n) is not str or not n or type(u) is not str or not u
                for n, u in zip(self.names, self.units))
                or len(set(self.names)) != len(self.names)):
            raise ValueError("duplicate/missing observation name or unit")
        mandatory = {"account.drawdown", "account.peak_equity", "account.equity", "risk.guard_intervention", "execution.prev_executed_target"}
        if not mandatory.issubset(self.names):
            raise ValueError("Markov reward/execution state absent from observation schema")

    @property
    def hash(self) -> str:
        body = json.dumps({"v":self.schema_version, "names":self.names, "units":self.units}, separators=(",", ":"))
        return sha256(body.encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class VisibleFeature:
    value: float
    available_at_ns: int
    unit: str


class ObservationBuilder:
    def __init__(self, schema: ObservationSchema, *, scaler: Any | None = None,
                 expected_scaler_hash: str | None = None, split: str = "train") -> None:
        if split not in {"train", "validation", "test", "holdout"}:
            raise ValueError("unknown split")
        if not isinstance(schema, ObservationSchema):
            raise TypeError("frozen observation schema required")
        self.schema, self.scaler, self.split = schema, scaler, split
        self._scaler_hash: str | None
        if scaler is not None:
            if not expected_scaler_hash or scaler.fingerprint != expected_scaler_hash:
                raise ValueError("train-fitted L2 scaler hash mismatch")
            if scaler.frozen is not True:
                raise ValueError("scaler must be frozen before any episode")
            if not callable(getattr(scaler, "transform", None)):
                raise TypeError("L2 frozen scaler.transform callable required")
            self._scaler_hash = expected_scaler_hash
        else:
            self._scaler_hash = None

    def build(self, *, decision_ts_ns: int,
              market: Mapping[str, VisibleFeature],
              account: Mapping[str, float], risk: Mapping[str, float],
              execution: Mapping[str, float], time: Mapping[str, float]) -> np.ndarray:
        if type(decision_ts_ns) is not int or decision_ts_ns <= 0:
            raise ValueError("positive UTC ns decision clock required")
        combined: dict[str, float] = {}
        for name, feat in market.items():
            if name in combined or not isinstance(feat, VisibleFeature):
                raise TypeError("L2 market values require VisibleFeature metadata")
            if type(feat.available_at_ns) is not int or feat.available_at_ns < 0 or feat.available_at_ns > decision_ts_ns:
                raise ValueError("future market feature detected")
            full_name = "market." + name
            if full_name not in self.schema.names:
                raise ValueError("unknown L2 market feature")
            unit = self.schema.units[self.schema.names.index(full_name)]
            if feat.unit != unit:
                raise ValueError("market feature unit/schema mismatch")
            combined[full_name] = feat.value
        for prefix, mapping in (("account", account), ("risk", risk), ("execution", execution), ("time", time)):
            for name, value in mapping.items():
                key = prefix + "." + name
                if key in combined:
                    raise ValueError("duplicate observation feature")
                combined[key] = value
        if set(combined) != set(self.schema.names):
            raise ValueError(f"observation feature mismatch missing={sorted(set(self.schema.names)-set(combined))} unknown={sorted(set(combined)-set(self.schema.names))}")
        values = np.asarray([combined[k] for k in self.schema.names], dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError("non-finite observation")
        if self.scaler is not None:
            # Detect accidental fit/mutation even in a nominally frozen transformer.
            if self.scaler.frozen is not True or self.scaler.fingerprint != self._scaler_hash:
                raise ValueError("L2 frozen scaler mutated before transform")
            transformed = np.asarray(self.scaler.transform(values[None, :]), dtype=np.float64)
            if transformed.shape != (1, len(self.schema.names)):
                raise ValueError("frozen scaler shape mismatch")
            if self.scaler.frozen is not True or self.scaler.fingerprint != self._scaler_hash:
                raise ValueError("L2 frozen scaler mutated during transform")
            values = transformed[0]
        if values.shape != (len(self.schema.names),) or not np.isfinite(values).all():
            raise ValueError("normalized observation invalid")
        # No silent infinity from float64 -> float32 conversion.
        output = values.astype(np.float32)
        if not np.isfinite(output).all():
            raise ValueError("float32 observation overflow")
        output.setflags(write=False)
        return output
