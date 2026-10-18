"""Minimal adapter boundary for the pinned upstream DreamerV3 RSSM."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


class WorldModelAdapterError(ValueError):
    """Invalid Route B worker or RSSM adapter input."""


UPSTREAM_REPOSITORY = "https://github.com/danijar/dreamerv3.git"
UPSTREAM_COMMIT = "e01491fad6434b2245a3b8ca201dd7faedcc458c"
ADAPTER_VERSION = "rles-market-rssm-v0"


@dataclass(frozen=True)
class UpstreamStatus:
    repository: str
    required_commit: str
    checkout: str | None
    observed_commit: str | None
    jax_version: str | None
    devices: tuple[str, ...]
    status: str
    detail: str | None = None


def constant_market_action(trading_action: Any, *, shape: tuple[int, ...] | None = None) -> dict[str, Any]:
    """Return an all-zero dummy action; actor actions never enter RSSM dynamics."""
    try:
        import numpy as np
    except ImportError as exc:
        raise WorldModelAdapterError("numpy is required for the Dreamer adapter") from exc
    if shape is None:
        source = next(iter(trading_action.values())) if isinstance(trading_action, dict) else trading_action
        inferred = getattr(source, "shape", None)
        shape = tuple(inferred) if inferred is not None else (1,)
    if not shape or any(type(size) is not int or size <= 0 for size in shape):
        raise WorldModelAdapterError("market dummy-action shape must be positive")
    # Validate that a caller did provide a policy action, but never inspect its value.
    if trading_action is None:
        raise WorldModelAdapterError("trading action argument is required")
    try:
        import jax.numpy as jnp
    except ImportError:
        return {"market_dummy": np.zeros(shape, dtype=np.float32)}
    return {"market_dummy": jnp.zeros(shape, dtype=jnp.float32)}


def inspect_upstream(checkout: str | Path) -> UpstreamStatus:
    """Report installed JAX devices and enforce the pinned checkout identity."""
    path = Path(checkout)
    if not path.is_dir():
        return UpstreamStatus(UPSTREAM_REPOSITORY, UPSTREAM_COMMIT, str(path), None,
                              None, (), "BLOCKED", "DreamerV3 checkout is missing")
    import subprocess

    result = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"],
                            capture_output=True, check=False, text=True)
    observed = result.stdout.strip() if result.returncode == 0 else None
    if observed != UPSTREAM_COMMIT:
        return UpstreamStatus(UPSTREAM_REPOSITORY, UPSTREAM_COMMIT, str(path), observed,
                              None, (), "BLOCKED", "checkout commit differs from pinned upstream")
    try:
        import jax
    except ImportError:
        return UpstreamStatus(UPSTREAM_REPOSITORY, UPSTREAM_COMMIT, str(path), observed,
                              None, (), "BLOCKED", "JAX is not installed in this interpreter")
    return UpstreamStatus(UPSTREAM_REPOSITORY, UPSTREAM_COMMIT, str(path), observed,
                          jax.__version__, tuple(str(device) for device in jax.devices()), "READY")


def rssm_market_observe(rssm: Any, carry: Any, tokens: Any, reset: Any, training: bool,
                        *, trading_action: Any, single: bool = False) -> Any:
    """Call upstream RSSM observation update with only a constant zero action."""
    action = constant_market_action(trading_action)
    return rssm.observe(carry, tokens, action, reset, training, single=single)


def rssm_market_imagine(rssm: Any, carry: Any, *, trading_action: Any,
                        training: bool = False) -> Any:
    """Advance the upstream RSSM prior with the constant market dummy action."""
    action = constant_market_action(trading_action)
    return rssm.imagine(carry, action, length=1, training=training, single=True)
