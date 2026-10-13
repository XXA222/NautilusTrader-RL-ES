"""Explicit schema metadata and append-only, no-unit-change minor policy."""

from __future__ import annotations

from dataclasses import dataclass

from .errors import ErrorCategory, RLESError
from .validation import require_name

CONTRACT_VERSION = "0.4.0"
PROTOCOL_VERSION = "0.4.0"


@dataclass(frozen=True, order=True, slots=True)
class Version:
    major: int
    minor: int
    patch: int

    def __post_init__(self) -> None:
        if any(type(v) is not int or v < 0 for v in (self.major, self.minor, self.patch)):
            raise ValueError("version components must be nonnegative integers")

    @classmethod
    def parse(cls, value: str) -> Version:
        if type(value) is not str:
            raise ValueError("schema_version requires a string")
        items = value.split(".")
        if len(items) != 3 or any(not s.isascii() or not s.isdigit() or (len(s) > 1 and s[0] == "0") for s in items):
            raise ValueError("schema_version must be MAJOR.MINOR.PATCH")
        return cls(*(int(x) for x in items))

    def __str__(self) -> str:
        return f"{self.major}.{self.minor}.{self.patch}"


@dataclass(frozen=True, slots=True)
class FieldSpec:
    name: str
    wire_type: str
    unit: str
    required: bool = True
    default_json: str | None = None

    def __post_init__(self) -> None:
        require_name(self.name, "field name")
        if type(self.wire_type) is not str or not self.wire_type or len(self.wire_type) > 256:
            raise ValueError("wire_type must be a nonempty type descriptor")
        require_name(self.unit, "field unit")
        if not self.required and self.default_json is None:
            raise ValueError("new optional field must specify a canonical JSON default")
        if self.required and self.default_json is not None:
            raise ValueError("required field must not have a default")
        if not self.required:
            import json
            try:
                default_json = self.default_json
                if not isinstance(default_json, str):
                    raise ValueError("optional field default must be a string")
                def _check_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
                    value: dict[str, object] = {}
                    for key, item in pairs:
                        if key in value:
                            raise ValueError("duplicate default JSON field")
                        value[key] = item
                    return value
                parsed = json.loads(default_json, object_pairs_hook=_check_pairs,
                                    parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
            except (ValueError, TypeError) as exc:
                raise ValueError("invalid canonical JSON default") from exc
            if json.dumps(parsed, ensure_ascii=True, sort_keys=True,
                          separators=(",", ":"), allow_nan=False) != default_json:
                raise ValueError("default_json must use canonical JSON encoding")


@dataclass(frozen=True, slots=True)
class ContractSchema:
    type_name: str
    version: Version
    fields: tuple[FieldSpec, ...]

    def __post_init__(self) -> None:
        require_name(self.type_name, "type name")
        if type(self.fields) is not tuple or any(type(x) is not FieldSpec for x in self.fields):
            raise TypeError("fields must be tuple[FieldSpec,...]")
        if len(set(x.name for x in self.fields)) != len(self.fields):
            raise ValueError("duplicate schema field")


def assert_compatible(old: ContractSchema, new: ContractSchema) -> None:
    """Minor: append-only optional+default fields; patch: identical; major requires ICR."""
    if old.type_name != new.type_name or old.version.major != new.version.major:
        raise RLESError(ErrorCategory.CONTRACT, "SCHEMA_MAJOR_BREAK", "major contract change requires ICR")
    if new.version < old.version:
        raise RLESError(ErrorCategory.CONTRACT, "SCHEMA_DOWNGRADE", "schema downgrade forbidden")
    prefix = new.fields[: len(old.fields)]
    if old.fields != prefix:
        raise RLESError(ErrorCategory.CONTRACT, "SCHEMA_FIELD_BREAK", "field/type/unit/order changed")
    tail = new.fields[len(old.fields) :]
    if new.version.minor == old.version.minor and tail:
        raise RLESError(ErrorCategory.CONTRACT, "SCHEMA_PATCH_BREAK", "patch cannot add fields")
    if any(item.required or item.default_json is None for item in tail):
        raise RLESError(ErrorCategory.CONTRACT, "SCHEMA_REQUIRED_ADDITION", "minor additions must default")


def require_current_version(value: str) -> None:
    if value != CONTRACT_VERSION:
        raise RLESError(
            ErrorCategory.CONTRACT, "SCHEMA_VERSION_MISMATCH",
            f"expected contracts schema {CONTRACT_VERSION}; got {value!r}",
        )
