"""Finite numeric guards for compute evidence and process deadlines."""
from __future__ import annotations

import math
import threading
from typing import TypeGuard


def finite_real(value: object) -> TypeGuard[int | float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(value)
    except (OverflowError, TypeError, ValueError):
        return False


def valid_timeout(value: object) -> TypeGuard[int | float]:
    """Only deadlines supported by Python's platform lock implementation."""
    return finite_real(value) and 0 < value <= threading.TIMEOUT_MAX
