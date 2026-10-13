"""Shared finite real-number guard for untrusted research inputs."""
from __future__ import annotations

import math


def finite_real(value: object) -> bool:
    """Return false, rather than leaking OverflowError, for oversized integers."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(value)
    except (OverflowError, TypeError, ValueError):
        return False
