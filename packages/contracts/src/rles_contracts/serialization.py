"""Canonical JSON and optional Arrow IPC codec with strictly registered DTO types.

No pickle, magic import path or arbitrary object constructors.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import fields, is_dataclass
from functools import lru_cache
from types import MappingProxyType
from collections.abc import Mapping
from decimal import Decimal
from enum import StrEnum
from typing import Any, get_args, get_origin, get_type_hints

from . import dto
from .errors import ErrorCategory
from .frozen import FrozenFields
from .schema import CONTRACT_VERSION, require_current_version

MAX_METADATA_BYTES = 8 * 1024 * 1024

# Do not expose a mutable registration hook: incoming wire tags must never be
# allowed to instantiate dataclasses from plug-ins or user-imported modules.
_LOCAL_DTO_TYPES: Mapping[str, type] = MappingProxyType({
    **{name: obj for name, obj in vars(dto).items()
       if isinstance(obj, type) and is_dataclass(obj) and obj.__module__ == dto.__name__},
    "FrozenFields": FrozenFields,
})


@lru_cache(maxsize=1)
def _registered_types() -> Mapping[str, type]:
    """Closed-world registry built only from L1's known source modules."""
    from . import config, errors, manifest, risk_contracts, seed, schema
    registry = dict(_LOCAL_DTO_TYPES)
    for module in (config, errors, manifest, risk_contracts, seed, schema):
        for name, klass in vars(module).items():
            if isinstance(klass, type) and is_dataclass(klass) and klass.__module__ == module.__name__:
                if name in registry and registry[name] is not klass:
                    raise RuntimeError(f"ambiguous built-in DTO name {name}")
                registry[name] = klass
    return MappingProxyType(registry)


_ENUMS: Mapping[str, type[StrEnum]] = MappingProxyType({
    **{name: obj for name, obj in vars(dto).items()
       if isinstance(obj, type) and issubclass(obj, StrEnum) and obj.__module__ == dto.__name__},
    "ErrorCategory": ErrorCategory,
})


def _encode(value: Any) -> Any:
    if value is None or type(value) in (bool, int, str):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("nonfinite float forbidden")
        return {"$float": value.hex()}
    if type(value) is Decimal:
        if not value.is_finite():
            raise ValueError("nonfinite Decimal forbidden")
        return {"$decimal": str(value)}
    if type(value) in _ENUMS.values():
        return {"$enum": type(value).__name__, "value": value.value}
    if type(value) is tuple:
        return {"$tuple": [_encode(item) for item in value]}
    if type(value) in _registered_types().values():
        return {
            "$dto": type(value).__name__,
            "fields": {field.name: _encode(getattr(value, field.name)) for field in fields(value)},
        }
    if type(value) is dict:
        if any(type(k) is not str for k in value):
            raise TypeError("canonical JSON requires string keys")
        if any(k.startswith("$") for k in value):
            raise ValueError("plain canonical JSON objects cannot use reserved $ keys")
        return {k: _encode(v) for k, v in value.items()}
    if type(value) is list:
        return [_encode(v) for v in value]
    raise TypeError(f"cannot serialize unsupported type {type(value).__name__}")


def canonical_bytes(value: Any) -> bytes:
    encoded = json.dumps(_encode(value), sort_keys=True, ensure_ascii=True,
                         separators=(",", ":"), allow_nan=False).encode("ascii")
    if len(encoded) > MAX_METADATA_BYTES:
        raise ValueError("canonical metadata exceeds 8 MiB limit; use L2 storage refs")
    return encoded


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _no_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for k, v in pairs:
        if k in result:
            raise ValueError(f"duplicate JSON key {k!r}")
        result[k] = v
    return result


def _decode(value: Any) -> Any:
    if type(value) is list:
        return [_decode(v) for v in value]
    if type(value) is not dict:
        return value
    if "$decimal" in value:
        if set(value) != {"$decimal"} or type(value["$decimal"]) is not str:
            raise ValueError("malformed Decimal tag")
        dec = Decimal(value["$decimal"])
        if not dec.is_finite():
            raise ValueError("nonfinite Decimal forbidden")
        return dec
    if "$float" in value:
        if set(value) != {"$float"} or type(value["$float"]) is not str:
            raise ValueError("malformed float tag")
        result = float.fromhex(value["$float"])
        if not math.isfinite(result):
            raise ValueError("nonfinite float forbidden")
        return result
    if "$tuple" in value:
        if set(value) != {"$tuple"} or type(value["$tuple"]) is not list:
            raise ValueError("malformed tuple tag")
        return tuple(_decode(v) for v in value["$tuple"])
    if "$enum" in value:
        if set(value) != {"$enum", "value"} or type(value["$enum"]) is not str:
            raise ValueError("malformed enum tag")
        enum_cls = _ENUMS.get(value["$enum"])
        if enum_cls is None or type(value["value"]) is not str:
            raise ValueError("unregistered/invalid enum")
        return enum_cls(value["value"])
    if "$dto" in value:
        if set(value) != {"$dto", "fields"}:
            raise ValueError("malformed DTO envelope")
        if type(value["$dto"]) is not str:
            raise ValueError("invalid DTO type tag")
        cls = _registered_types().get(value["$dto"])
        if cls is None or type(value["fields"]) is not dict:
            raise ValueError("unregistered DTO name or malformed fields")
        required = {f.name for f in fields(cls)}
        supplied = set(value["fields"])
        if supplied != required:
            raise ValueError(f"{cls.__name__}: unknown/missing fields: {sorted(supplied ^ required)}")
        decoded = {k: _decode(v) for k, v in value["fields"].items()}
        for fld in fields(cls):
            _validate_field_type(decoded[fld.name], get_type_hints(cls)[fld.name], fld.name)
        return cls(**decoded)
    if any(k.startswith("$") for k in value):
        raise ValueError("unknown reserved JSON tag")
    return {k: _decode(v) for k, v in value.items()}


def _validate_field_type(value: Any, annotation: Any, field_path: str) -> None:
    """Reject implicit int->float, bool->int, list->tuple, unknown nested DTOs."""
    import types
    import typing

    origin = get_origin(annotation)
    if origin in (types.UnionType, typing.Union):
        for arm in get_args(annotation):
            try:
                _validate_field_type(value, arm, field_path)
                return
            except (TypeError, ValueError):
                continue
        raise TypeError(f"{field_path}: union mismatch")
    if origin is tuple:
        if type(value) is not tuple:
            raise TypeError(f"{field_path}: tuple required")
        args = get_args(annotation)
        if len(args) == 2 and args[1] is Ellipsis:
            for item in value:
                _validate_field_type(item, args[0], field_path)
        else:
            if len(args) != len(value):
                raise TypeError(f"{field_path}: fixed-size tuple mismatch")
            for item, ty in zip(value, args, strict=True):
                _validate_field_type(item, ty, field_path)
        return
    if annotation is Any:
        raise TypeError("unbounded Any fields forbidden in wire DTO")
    if annotation is type(None):
        if value is not None:
            raise TypeError(f"{field_path}: expected null")
        return
    if type(value) is not annotation:
        raise TypeError(f"{field_path}: expected {annotation}, got {type(value).__name__}")


def from_canonical_bytes(blob: bytes) -> Any:
    if type(blob) is not bytes:
        raise TypeError("canonical input must be bytes")
    if len(blob) > MAX_METADATA_BYTES:
        raise ValueError("canonical metadata exceeds 8 MiB limit")
    payload = json.loads(blob.decode("ascii"), parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)),
                         object_pairs_hook=_no_duplicate_pairs)
    obj = _decode(payload)
    if canonical_bytes(obj) != blob:
        raise ValueError("not canonical JSON (ordering, normalization, or numeric representation)")
    if type(obj) in _registered_types().values() and hasattr(obj, "schema_version"):
        require_current_version(obj.schema_version)
    return obj


def to_arrow_ipc(objects: tuple[object, ...]) -> bytes:
    """Small metadata DTOs only; large Arrow/Parquet market data belongs to rles-data."""
    try:
        import pyarrow as pa
        import pyarrow.ipc as ipc
    except ImportError as exc:
        from .errors import ErrorCategory, RLESError
        raise RLESError(ErrorCategory.DEPENDENCY, "ARROW_UNAVAILABLE", "install rles-contracts[arrow]") from exc
    if type(objects) is not tuple or not objects:
        raise ValueError("Arrow records must be a nonempty tuple")
    blob = [canonical_bytes(obj) for obj in objects]
    schema = pa.schema([("canonical_json", pa.binary()), ("sha256", pa.string())],
                       metadata={b"rles.contracts.schema": CONTRACT_VERSION.encode("ascii")})
    batch = pa.record_batch([pa.array(blob, type=pa.binary()),
                             pa.array([hashlib.sha256(item).hexdigest() for item in blob])], schema=schema)
    sink = pa.BufferOutputStream()
    with ipc.new_stream(sink, schema) as stream:
        stream.write_batch(batch)
    return sink.getvalue().to_pybytes()


def from_arrow_ipc(blob: bytes) -> tuple[object, ...]:
    try:
        import pyarrow as pa
        import pyarrow.ipc as ipc
    except ImportError as exc:
        from .errors import ErrorCategory, RLESError
        raise RLESError(ErrorCategory.DEPENDENCY, "ARROW_UNAVAILABLE", "install rles-contracts[arrow]") from exc
    if type(blob) is not bytes:
        raise TypeError("Arrow input must be bytes")
    if len(blob) > MAX_METADATA_BYTES * 2:
        raise ValueError("Arrow IPC metadata transport exceeds 16 MiB limit")
    with ipc.open_stream(pa.BufferReader(blob)) as reader:
        if reader.schema.names != ["canonical_json", "sha256"]:
            raise ValueError("Arrow IPC schema mismatch")
        expected = pa.schema([("canonical_json", pa.binary()), ("sha256", pa.string())],
                             metadata={b"rles.contracts.schema": CONTRACT_VERSION.encode("ascii")})
        if not reader.schema.equals(expected, check_metadata=True):
            raise ValueError("Arrow IPC schema, field types or version mismatch")
        table = reader.read_all()
    if table.num_rows == 0:
        raise ValueError("Arrow IPC empty batch not allowed")
    result = []
    for row in table.to_pylist():
        data = row["canonical_json"]
        if type(data) is not bytes or type(row["sha256"]) is not str:
            raise ValueError("Arrow IPC null/nonbinary payload")
        if hashlib.sha256(data).hexdigest() != row["sha256"]:
            raise ValueError("Arrow IPC payload hash mismatch")
        result.append(from_canonical_bytes(data))
    return tuple(result)
