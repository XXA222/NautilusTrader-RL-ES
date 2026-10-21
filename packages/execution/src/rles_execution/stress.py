"""Deterministic scenario identities; never adjust native fill/account state here."""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json

from .spec import ResolvedExecutionConfig


@dataclass(frozen=True, slots=True)
class ExecutionScenario:
    """Resolved model selectors to bind into the official native venue."""
    config_hash: str
    latency_model_id: str
    fill_model_id: str
    total_latency_ns: int
    slippage_bps: str
    fee_multiplier: str
    simulated_depth_fraction: str
    granularity: str
    scenario_hash: str


def resolve_scenario(config: ResolvedExecutionConfig, mode: str) -> ExecutionScenario:
    if not isinstance(config, ResolvedExecutionConfig):
        raise TypeError("resolved execution configuration required")
    if mode not in {"base", "stress"}:
        raise ValueError("scenario must be base or stress")
    latency, fill = config.latency(mode), config.fill(mode)
    config_hash = config.fingerprint()
    latency_model_id = latency.model_id
    fill_model_id = fill.model_id
    total_latency_ns = latency.total_ns
    slippage_bps = str(fill.slippage_bps)
    fee_multiplier = str(fill.fee_multiplier)
    simulated_depth_fraction = str(fill.simulated_depth_fraction)
    granularity = fill.data_granularity
    payload = {
        "config_hash": config_hash, "latency_model_id": latency_model_id,
        "fill_model_id": fill_model_id, "total_latency_ns": total_latency_ns,
        "slippage_bps": slippage_bps, "fee_multiplier": fee_multiplier,
        "granularity": granularity,
    }
    if fill.simulated_depth_fraction != 1:
        payload["simulated_depth_fraction"] = simulated_depth_fraction
    digest = sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return ExecutionScenario(config_hash, latency_model_id, fill_model_id, total_latency_ns,
                             slippage_bps, fee_multiplier, simulated_depth_fraction, granularity, digest)
