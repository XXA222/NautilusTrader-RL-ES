"""Strict resolved execution settings. L1 owns the canonical ExecutionSpec DTO."""
from __future__ import annotations

from dataclasses import dataclass, fields
from decimal import Decimal
from hashlib import sha256
import json


def dec(value: object, *, positive: bool = False) -> Decimal:
    if isinstance(value, bool) or isinstance(value, float):
        raise TypeError("Decimal input must be a decimal string, integer, or Decimal")
    result = Decimal(str(value))
    if not result.is_finite() or (positive and result <= 0):
        raise ValueError("non-finite or non-positive decimal")
    return result


@dataclass(frozen=True, slots=True)
class LatencyProfile:
    model_id: str
    decision_ns: int
    submit_ns: int
    venue_ns: int

    def __post_init__(self) -> None:
        if not self.model_id or any(type(x) is not int or x < 0 for x in (self.decision_ns, self.submit_ns, self.venue_ns)):
            raise ValueError("invalid latency model")

    @property
    def total_ns(self) -> int:
        return self.decision_ns + self.submit_ns + self.venue_ns


@dataclass(frozen=True, slots=True)
class FillProfile:
    model_id: str
    slippage_bps: Decimal
    data_granularity: str = "bar"
    fee_multiplier: Decimal = Decimal("1")

    def __post_init__(self) -> None:
        if not self.model_id or self.data_granularity not in {"bar", "trade", "quote"}:
            raise ValueError("unknown fill model or data granularity")
        slip = dec(self.slippage_bps)
        fee_multiplier = dec(self.fee_multiplier, positive=True)
        if slip < 0 or slip > Decimal("10000"):
            raise ValueError("slippage must be in [0, 10000] bps")
        if fee_multiplier < 1:
            raise ValueError("fee multiplier cannot be more optimistic than the historical fee")
        if self.data_granularity == "quote" and slip != 0:
            raise ValueError("quote execution uses native displayed depth and requires zero synthetic slippage")
        object.__setattr__(self, "slippage_bps", slip)
        object.__setattr__(self, "fee_multiplier", fee_multiplier)


@dataclass(frozen=True, slots=True)
class ResolvedExecutionConfig:
    schema_version: str
    deadband: Decimal
    pending_order_policy: str
    order_type: str
    tif: str
    terminal_policy: str
    base_latency: LatencyProfile
    stress_latency: LatencyProfile
    base_fill: FillProfile
    stress_fill: FillProfile

    def __post_init__(self) -> None:
        object.__setattr__(self, "deadband", dec(self.deadband))
        if not all(isinstance(getattr(self, key), typ) for key, typ in (("base_latency", LatencyProfile), ("stress_latency", LatencyProfile), ("base_fill", FillProfile), ("stress_fill", FillProfile))):
            raise TypeError("native latency/fill model profiles must be resolved")
        if self.schema_version != "0.4.0":
            raise ValueError("unsupported execution schema")
        if not (Decimal(0) <= dec(self.deadband) <= Decimal(1)):
            raise ValueError("deadband outside [0,1]")
        if self.pending_order_policy not in {"cancel_replace", "wait", "fail"}:
            raise ValueError("pending order policy must be explicit")
        if self.order_type != "MARKET" or self.tif != "GTC":
            raise ValueError("L3 v0.4.0 supports market-like GTC execution with explicit bar or quote data")
        if self.terminal_policy != "force_flatten_with_cost":
            raise ValueError("only cost-bearing terminal flatten supported")
        if self.base_latency.model_id == self.stress_latency.model_id:
            raise ValueError("stress latency needs distinct model ID")
        if self.base_fill.model_id == self.stress_fill.model_id:
            raise ValueError("stress fill needs distinct model ID")
        if self.base_fill.data_granularity != self.stress_fill.data_granularity:
            raise ValueError("fill modes cannot change data granularity")
        if self.stress_latency.total_ns < self.base_latency.total_ns:
            raise ValueError("stress latency may not be more optimistic than base")
        if self.stress_fill.slippage_bps < self.base_fill.slippage_bps:
            raise ValueError("stress slippage may not be more optimistic than base")
        if self.base_fill.fee_multiplier != 1:
            raise ValueError("base execution must use the historical instrument fee")
        if self.stress_fill.fee_multiplier < self.base_fill.fee_multiplier:
            raise ValueError("stress fees may not be more optimistic than base")
        if self.base_fill.data_granularity == "bar" and self.base_fill.slippage_bps == 0:
            raise ValueError("bar-only research cannot assume zero baseline slippage")

    @classmethod
    def from_dict(cls, data: dict) -> ResolvedExecutionConfig:
        names = set(cls.__dataclass_fields__)
        if set(data) != names:
            raise ValueError(f"execution config fields mismatch: missing={sorted(names-set(data))}, unknown={sorted(set(data)-names)}")
        def profile(t: type[LatencyProfile] | type[FillProfile], value: object) -> LatencyProfile | FillProfile:
            if not isinstance(value, dict):
                raise ValueError(f"invalid {t.__name__} keys")
            if t is FillProfile:
                allowed = {field.name for field in fields(t)}
                required = allowed - {"fee_multiplier"}
                if not required <= set(value) or set(value) - allowed:
                    raise ValueError("invalid FillProfile keys")
                value = {**value, "slippage_bps": dec(value["slippage_bps"]),
                         "fee_multiplier": dec(value.get("fee_multiplier", "1"))}
            elif set(value) != {field.name for field in fields(t)}:
                raise ValueError(f"invalid {t.__name__} keys")
            return t(**value)  # type: ignore[call-arg]
        values = dict(data)
        values["deadband"] = dec(values["deadband"])
        for n in ("base_latency", "stress_latency"):
            values[n] = profile(LatencyProfile, values[n])
        for n in ("base_fill", "stress_fill"):
            values[n] = profile(FillProfile, values[n])
        return cls(**values)

    def fingerprint(self) -> str:
        from dataclasses import asdict
        body = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"), default=str)
        return sha256(body.encode()).hexdigest()

    def latency(self, mode: str) -> LatencyProfile:
        if mode not in {"base", "stress"}:
            raise ValueError("unknown latency mode")
        return self.base_latency if mode == "base" else self.stress_latency

    def fill(self, mode: str) -> FillProfile:
        if mode not in {"base", "stress"}:
            raise ValueError("unknown fill mode")
        return self.base_fill if mode == "base" else self.stress_fill
