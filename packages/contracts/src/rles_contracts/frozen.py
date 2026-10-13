"""Immutable scalar map for small, cross-process DTO metadata (never large market data)."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import TypeAlias

from .validation import require_name

Scalar: TypeAlias = str | int | float | bool | Decimal | None


@dataclass(frozen=True, slots=True)
class FrozenFields:
    """Sorted unique (key, scalar) tuples, no shared mutable mappings."""

    entries: tuple[tuple[str, Scalar], ...] = ()

    def __post_init__(self) -> None:
        if type(self.entries) is not tuple:
            raise TypeError("FrozenFields.entries must be a tuple")
        prior = ""
        from math import isfinite

        for entry in self.entries:
            if type(entry) is not tuple or len(entry) != 2:
                raise TypeError("FrozenFields requires (key, scalar) tuples")
            key, value = entry
            require_name(key, "metadata key")
            if key <= prior:
                raise ValueError("FrozenFields keys must be sorted and unique")
            prior = key
            if type(value) not in (str, int, float, bool, Decimal, type(None)):
                raise TypeError(f"FrozenFields: unsupported value for {key}")
            if type(value) is float and not isfinite(value):
                raise ValueError(f"FrozenFields: nonfinite float at {key}")
            if type(value) is Decimal and not value.is_finite():
                raise ValueError(f"FrozenFields: nonfinite Decimal at {key}")

    @classmethod
    def from_mapping(cls, mapping: dict[str, Scalar]) -> FrozenFields:
        if type(mapping) is not dict:
            raise TypeError("metadata input must be dict")
        return cls(tuple(sorted(mapping.items())))

    def get(self, key: str) -> Scalar:
        for k, v in self.entries:
            if k == key:
                return v
        raise KeyError(key)
