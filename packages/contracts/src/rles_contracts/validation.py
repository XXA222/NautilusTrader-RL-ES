"""Shared value guards. No implicit numeric coercions or permissive fallbacks."""

from __future__ import annotations

import math
import re
from decimal import Decimal
from typing import Any

HASH_RE = re.compile(r"^[a-f0-9]{64}$")
NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_.:/-]{0,127}$")
INT64_MAX = (1 << 63) - 1


def require_name(value: str, label: str) -> None:
    if type(value) is not str or NAME_RE.fullmatch(value) is None:
        raise ValueError(f"{label}: invalid name")


def require_hash(value: str, label: str) -> None:
    if type(value) is not str or HASH_RE.fullmatch(value) is None:
        raise ValueError(f"{label}: expected lowercase sha256 hex")


def require_ns(value: int, label: str) -> None:
    if type(value) is not int or not 0 <= value <= INT64_MAX:
        raise ValueError(f"{label}: UTC nanosecond timestamp must fit signed int64")


def require_decimal(value: Decimal, label: str) -> None:
    if type(value) is not Decimal or not value.is_finite():
        raise ValueError(f"{label}: must be finite Decimal (not float)")


def require_float(value: float, label: str) -> None:
    if type(value) is not float or not math.isfinite(value):
        raise ValueError(f"{label}: expected finite float")


def require_exact_type(value: Any, target: type, label: str) -> None:
    if type(value) is not target:
        raise TypeError(f"{label}: expected {target.__name__}, got {type(value).__name__}")


def require_keys(value: dict[str, Any], required: frozenset[str], *, path: str) -> None:
    if type(value) is not dict or any(type(k) is not str for k in value):
        raise TypeError(f"{path}: expected object with string keys")
    unknown = set(value) - required
    missing = required - set(value)
    if unknown or missing:
        raise ValueError(f"{path}: unknown={sorted(unknown)} missing={sorted(missing)}")


def exact_decimal_multiple(value: Decimal, increment: Decimal) -> bool:
    """Exact divisibility without Decimal context rounding or float conversion."""
    require_decimal(value, "quantity")
    require_decimal(increment, "increment")
    if increment <= 0:
        raise ValueError("increment must be positive")
    if value.is_zero():
        return True

    def parts(number: Decimal) -> tuple[int, int]:
        raw = number.as_tuple()
        if len(raw.digits) > 4096:
            raise ValueError("Decimal precision exceeds 4096-digit DTO limit")
        # DecimalTuple types the exponent as int | str for NaN/Infinity cases;
        # require_decimal above excludes those cases at runtime.
        if not isinstance(raw.exponent, int):
            raise ValueError("quantity exponent must be an integer")
        coeff = 0
        for digit in raw.digits:
            coeff = coeff * 10 + digit
        return coeff, raw.exponent

    numerator, exp_n = parts(value)
    denominator, exp_d = parts(increment)
    if exp_n >= exp_d:
        return (numerator * pow(10, exp_n - exp_d, denominator)) % denominator == 0
    if numerator % denominator:
        return False
    quotient = numerator // denominator
    gap = exp_d - exp_n
    # A positive integer has at most len(str(n)) - 1 trailing zeros.
    if gap >= len(str(quotient)):
        return False
    return str(quotient).endswith("0" * gap)
