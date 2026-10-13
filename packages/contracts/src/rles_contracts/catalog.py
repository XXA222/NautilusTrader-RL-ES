"""Public unit/type catalogue for schema compatibility review (no inferred units)."""

from __future__ import annotations

from dataclasses import fields
from typing import get_args, get_origin, get_type_hints
from types import UnionType

from . import dto
from .schema import CONTRACT_VERSION, ContractSchema, FieldSpec, Version
from .serialization import canonical_hash

# Explicit units for financially meaningful or time-bearing scalars.
# Generic symbolic fields are identifiers, units are NOT inferred from their names.
FIELD_UNITS = {
    "MarketEvent": {"ts_event_ns": "utc_ns", "ts_init_ns": "utc_ns", "available_at_ns": "utc_ns"},
    "Observation": {"decision_ts_ns": "utc_ns", "available_through_ns": "utc_ns",
                    "prev_action": "signed_target_fraction", "vector": "per_field_units"},
    "PolicyAction": {"raw_action": "signed_target_fraction"},
    "TargetExposure": {"requested": "exposure_unit", "soft_risk_adjusted": "exposure_unit",
                       "hard_risk_capped": "exposure_unit"},
    "ExecutionPlan": {"delta_qty": "quantity_unit", "quantity_increment": "quantity_unit"},
    "ExecutionSnapshot": {"decision_ts_ns": "utc_ns", "fees": "currency", "funding": "currency",
                          "margin": "currency", "equity": "currency", "mark": "currency_per_asset"},
    "StepOutcome": {"reward": "reward_unit", "barrier_ts_ns": "utc_ns"},
    "ComputeRequest": {"step_id": "step_index", "shape": "element_counts"},
    "EpisodeSpec": {"score_start_ns": "utc_ns", "score_end_ns": "utc_ns"},
    "EngineAdvanceResult": {"barrier_ts_ns": "utc_ns"},
    "InstrumentSpec": {"quantity_increment": "quantity_unit", "price_increment": "price_unit"},
}


def _wire_type_name(annotation: object) -> str:
    """Stable across CPython type-repr rendering changes."""
    origin = get_origin(annotation)
    args = get_args(annotation)
    if origin is UnionType:
        return "|".join(sorted(_wire_type_name(item) for item in args))
    if origin is tuple:
        if len(args) == 2 and args[1] is Ellipsis:
            return f"tuple[{_wire_type_name(args[0])},...]"
        return "tuple[" + ",".join(_wire_type_name(item) for item in args) + "]"
    if annotation is type(None):
        return "null"
    if isinstance(annotation, type):
        return annotation.__name__
    raise TypeError("unsupported wire type annotation; schema requires explicit review")


def dto_schema(name: str) -> ContractSchema:
    cls = getattr(dto, name, None)
    if not isinstance(cls, type) or not hasattr(cls, "__dataclass_fields__") or cls.__module__ != dto.__name__:
        raise ValueError("not a registered public DTO type")
    annotations = get_type_hints(cls)
    units = FIELD_UNITS.get(name, {})
    specs = tuple(FieldSpec(field.name, _wire_type_name(annotations[field.name]),
                            units.get(field.name, "not_applicable")) for field in fields(cls))
    return ContractSchema(name, Version.parse(CONTRACT_VERSION), specs)


def dto_schema_hash(name: str) -> str:
    return canonical_hash(dto_schema(name))
