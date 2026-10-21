"""Application-bound Native Gym candidate evaluation for L6 spawn workers.

The installed application supplies a reviewed L3 environment factory and its
frozen data/engine configuration. This adapter owns no NautilusTrader source,
RL algorithm, ES searcher, or risk policy.
"""
from __future__ import annotations

from collections.abc import Mapping
from hashlib import sha256
import importlib
import json
import math
from pathlib import Path
import re
from typing import Any, cast

import numpy as np

from rles_es_evotorch.actor import ActorGenomeSpec, act, decode as decode_actor, genome_hash
from rles_es_evotorch.evaluation import CandidateRequest
from rles_es_evotorch.soft_risk import SoftRiskDimension, SoftRiskSpace
from rles_gym_bridge.env import NautilusBarrierEnv


_HASH = re.compile(r"[0-9a-f]{64}\Z")
_CONTEXT_FIELDS = frozenset({
    "version", "request_hash", "run_id", "attempt_id", "generation",
    "candidate_id", "episode_id", "seed", "genome", "actor_signature",
    "candidate_kind", "candidate_schema_hash", "account_isolation_id", "split",
})


def _digest(value: object) -> str:
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                             allow_nan=False).encode()).hexdigest()


def _factory(ref: str):
    if not isinstance(ref, str) or ref.count(":") != 1:
        raise ValueError("installed L3 environment factory reference required")
    module_name, attribute = ref.split(":")
    if not module_name or not attribute or any(part.startswith("_") or not part
                                                for part in attribute.split(".")):
        raise ValueError("public installed L3 environment factory required")
    target: Any = importlib.import_module(module_name)
    for part in attribute.split("."):
        target = getattr(target, part)
    if not callable(target):
        raise TypeError("L3 environment factory is not callable")
    return target


def _actor_spec(raw: object) -> ActorGenomeSpec:
    if not isinstance(raw, Mapping) or not isinstance(raw.get("observation_names"), list) or not isinstance(raw.get("hidden_sizes"), list):
        raise ValueError("actor_spec requires ordered observation_names and hidden_sizes")
    names = raw["observation_names"]
    hidden = raw["hidden_sizes"]
    allowed = {"observation_names", "hidden_sizes", "activation", "output_size",
               "action_scale", "dtype", "version", "input_scale"}
    if set(raw) - allowed:
        raise ValueError("unknown actor_spec field")
    return ActorGenomeSpec(
        tuple(names), tuple(hidden), activation=raw.get("activation", "tanh"),
        output_size=raw.get("output_size", 1), action_scale=raw.get("action_scale", 1.0),
        dtype=raw.get("dtype", "float32"), version=raw.get("version", 1),
        input_scale=None if raw.get("input_scale") is None else tuple(raw["input_scale"]),
    )


def _risk_space(raw: object) -> SoftRiskSpace:
    if not isinstance(raw, list) or not raw:
        raise ValueError("E2 risk_dimensions must be a nonempty JSON array")
    dimensions = []
    for item in raw:
        if not isinstance(item, Mapping) or set(item) != {"name", "lower", "upper"}:
            raise ValueError("invalid E2 risk dimension")
        dimensions.append(SoftRiskDimension(item["name"], item["lower"], item["upper"]))
    return SoftRiskSpace(tuple(dimensions))


class NativeESCandidateSession:
    """One candidate, one fresh L3 Gym episode and native account."""

    def __init__(self, config: Mapping[str, object]):
        if not isinstance(config, Mapping):
            raise TypeError("frozen native ES worker config required")
        kind = config.get("candidate_kind")
        inference_mode = config.get("actor_inference", "canonical")
        allowed = {"candidate_kind", "env_factory_ref", "env_config", "actor_spec",
                   "actor_signature", "transition_hash", "episode_steps", "worker_index"}
        if "actor_inference" in config:
            allowed.add("actor_inference")
        if kind == "soft_risk":
            allowed |= {"risk_dimensions", "frozen_actor_hash"}
            allowed.add("frozen_actor_genome" if inference_mode == "canonical"
                        else "frozen_actor_artifact")
        if (kind not in {"actor", "soft_risk"}
                or inference_mode not in {"canonical", "l4_inference"}
                or (kind == "actor" and inference_mode != "canonical")
                or set(config) != allowed):
            raise ValueError("native ES worker recipe does not match E1/E2 schema")
        if (not isinstance(config["env_config"], Mapping)
                or not isinstance(config["actor_signature"], str)
                or _HASH.fullmatch(config["actor_signature"]) is None
                or not isinstance(config["transition_hash"], str)
                or _HASH.fullmatch(config["transition_hash"]) is None
                or type(config["episode_steps"]) is not int or config["episode_steps"] < 1
                or type(config["worker_index"]) is not int or config["worker_index"] < 0):
            raise ValueError("native ES worker requires frozen actor, transition and episode bounds")
        self.kind = kind
        self.inference_mode = inference_mode
        self.actor_spec = _actor_spec(config["actor_spec"])
        self.actor_signature = config["actor_signature"]
        self.transition_hash = config["transition_hash"]
        self.episode_steps = config["episode_steps"]
        self.env_config = dict(config["env_config"])
        self.env_factory = _factory(cast(str, config["env_factory_ref"]))
        self.risk_space = _risk_space(config["risk_dimensions"]) if kind == "soft_risk" else None
        self.frozen_actor: np.ndarray | None = None
        self.l4_actor: Any | None = None
        if kind == "soft_risk" and inference_mode == "canonical":
            raw = config["frozen_actor_genome"]
            if (not isinstance(raw, list) or any(type(value) not in {int, float}
                                                or not math.isfinite(value) for value in raw)):
                raise ValueError("frozen R1 actor genome must contain finite JSON numbers")
            frozen = np.asarray(raw, dtype=np.float32)
            decode_actor(self.actor_spec, frozen)
            if (not isinstance(config["frozen_actor_hash"], str)
                    or config["frozen_actor_hash"] != genome_hash(self.actor_spec, frozen)):
                raise ValueError("E2 frozen R1 actor weights differ from the declared actor hash")
            self.frozen_actor = frozen
        elif kind == "soft_risk":
            artifact = config["frozen_actor_artifact"]
            if not isinstance(artifact, str) or not Path(artifact).is_absolute():
                raise ValueError("E2 L4 inference artifact must have an absolute path")
            from rles_rl_sb3.actor import InferenceActor
            self.l4_actor = InferenceActor(Path(artifact), verify=True)
            marker = (Path(artifact) / "COMPLETED").read_text(encoding="ascii")
            if config["frozen_actor_hash"] != marker:
                raise ValueError("E2 L4 inference artifact differs from the attested R1 actor")

    def evaluate(self, *, rles_context: Mapping[str, object]) -> dict[str, object]:
        if (not isinstance(rles_context, Mapping) or set(rles_context) != _CONTEXT_FIELDS
                or rles_context["version"] != 1 or not isinstance(rles_context["genome"], list)):
            raise ValueError("invalid frozen L5 candidate context")
        request = CandidateRequest(
            run_id=cast(str, rles_context["run_id"]),
            attempt_id=cast(str, rles_context["attempt_id"]),
            generation=cast(int, rles_context["generation"]),
            candidate_id=cast(str, rles_context["candidate_id"]),
            episode_id=cast(str, rles_context["episode_id"]),
            seed=cast(int, rles_context["seed"]),
            genome=cast(tuple[float, ...], tuple(rles_context["genome"])),
            actor_signature=cast(str, rles_context["actor_signature"]),
            candidate_kind=cast(str, rles_context["candidate_kind"]),
            candidate_schema_hash=cast(str, rles_context["candidate_schema_hash"]),
            account_isolation_id=cast(str, rles_context["account_isolation_id"]),
            split=cast(str, rles_context["split"]),
        )
        if (rles_context["request_hash"] != request.request_hash
                or request.actor_signature != self.actor_signature
                or request.candidate_kind != self.kind):
            raise ValueError("candidate request identity differs from frozen worker recipe")
        candidate = np.asarray(request.genome, dtype=np.float32)
        if self.kind == "actor":
            if request.candidate_schema_hash != self.actor_spec.signature:
                raise ValueError("E1 actor candidate schema changed")
            decode_actor(self.actor_spec, candidate)
            actor: np.ndarray | None = candidate
            multipliers: dict[str, float] = {}
        else:
            if self.risk_space is None:
                raise RuntimeError("E2 risk space missing")
            if request.candidate_schema_hash != self.risk_space.signature:
                raise ValueError("E2 SoftRisk candidate schema changed")
            multipliers = self.risk_space.decode(candidate)
            actor = self.frozen_actor
        env = self.env_factory(
            env_config=dict(self.env_config), candidate_context=dict(rles_context),
            soft_risk_multipliers=multipliers,
        )
        if not isinstance(env, NautilusBarrierEnv):
            raise TypeError("reviewed L3 factory must return a NautilusBarrierEnv")
        try:
            if (env.rles_transition_hash != self.transition_hash
                    or tuple(env.observation_schema.names) != self.actor_spec.observation_names):
                raise ValueError("native environment transition or observation schema drift")
            if (self.l4_actor is not None
                    and self.l4_actor.mapping["observation_schema"] != env.observation_schema.hash):
                raise ValueError("R1 inference artifact observation schema differs from E2 episode")
            observation, reset_info = env.reset(seed=request.seed)
            if env.rles_transition_hash != self.transition_hash:
                raise ValueError("native episode differs from frozen transition recipe")
            initial_equity = float(reset_info["initial_equity"])
            if not math.isfinite(initial_equity) or initial_equity <= 0:
                raise ValueError("invalid initial native equity")
            returns: list[float] = []
            drawdowns: list[float] = []
            native_step_evidence: list[str] = []
            risk_interventions = 0
            insolvency_seen = False
            margin_facts_known = True
            maintenance_breach_seen = False
            liquidation_detected = False
            terminal_info: Mapping[str, object] | None = None
            for _ in range(self.episode_steps):
                if self.l4_actor is not None:
                    action = self.l4_actor.predict(observation[None, :])[0]
                else:
                    if actor is None:
                        raise RuntimeError("frozen canonical actor is missing")
                    action = act(self.actor_spec, actor, observation[None, :])[0]
                observation, _shaped_reward, terminated, truncated, info = env.step(action)
                if not isinstance(info, Mapping) or not isinstance(info.get("evidence_hash"), str):
                    raise ValueError("native episode step evidence is missing")
                evidence = info["evidence_hash"]
                if _HASH.fullmatch(evidence) is None:
                    raise ValueError("invalid native episode step evidence")
                if info.get("risk_intervention_components_known") is not True:
                    raise ValueError("native ES fitness requires classified soft and hard risk adjustments")
                if type(info.get("insolvency")) is not bool:
                    raise ValueError("native ES fitness requires explicit solvency facts")
                insolvency_seen = insolvency_seen or bool(info["insolvency"])
                if any(type(info.get(name)) is not bool for name in (
                        "margin_facts_known", "maintenance_margin_breach",
                        "native_liquidation_detected")):
                    raise ValueError("native ES fitness requires explicit margin/liquidation facts")
                margin_facts_known = margin_facts_known and bool(info["margin_facts_known"])
                maintenance_breach_seen = maintenance_breach_seen or bool(info["maintenance_margin_breach"])
                liquidation_detected = liquidation_detected or bool(info["native_liquidation_detected"])
                native_step_evidence.append(evidence)
                returns.append(float(info["r_econ"]))
                drawdowns.append(float(info["drawdown"]))
                hard_adjustment = float(info["risk_hard_adjustment"])
                if not math.isfinite(hard_adjustment) or hard_adjustment < 0:
                    raise ValueError("native hard-risk adjustment is invalid")
                risk_interventions += int(hard_adjustment > 0)
                if terminated or truncated:
                    if info.get("terminal_settled") is not True:
                        raise ValueError("native candidate terminal settlement is unproven")
                    terminal_info = info
                    break
            if terminal_info is None:
                raise ValueError("native candidate did not reach a settled terminal boundary")
            final_equity = float(cast(Any, terminal_info["native_equity"]))
            if (not math.isfinite(final_equity)
                    or any(not math.isfinite(value) for value in (*returns, *drawdowns))):
                raise ValueError("nonfinite native episode fitness facts")
            if not returns:
                raise ValueError("native candidate episode contains no scored decisions")
            if final_equity > 0 and not insolvency_seen:
                try:
                    realized_log_return = math.fsum(returns)
                except OverflowError as exc:
                    raise ValueError("native episode log returns do not reconcile to endpoint equity") from exc
                endpoint_log_return = math.log(final_equity) - math.log(initial_equity)
                if not math.isclose(realized_log_return, endpoint_log_return,
                                    rel_tol=0.0, abs_tol=1e-8):
                    raise ValueError("native episode log returns do not reconcile to endpoint equity")
            risk_intervention_rate = risk_interventions / len(returns)
            evidence_hash = _digest((
                request.request_hash, request.account_isolation_id, initial_equity,
                final_equity, returns, drawdowns, risk_interventions, len(returns),
                risk_intervention_rate, insolvency_seen, margin_facts_known,
                maintenance_breach_seen, liquidation_detected, native_step_evidence,
            ))
            return {
                "initial_equity": initial_equity, "final_equity": final_equity,
                "step_log_returns": returns, "drawdowns": drawdowns,
                "risk_intervention_count": risk_interventions,
                "risk_intervention_steps": len(returns),
                "risk_intervention_rate": risk_intervention_rate,
                "insolvency": insolvency_seen,
                "margin_facts_known": margin_facts_known,
                "maintenance_margin_breach": maintenance_breach_seen,
                "native_liquidation_detected": liquidation_detected,
                "evidence_hash": evidence_hash,
                "request_hash": request.request_hash,
                "account_isolation_id": request.account_isolation_id,
            }
        finally:
            env.close()

    def close(self) -> None:
        """Each evaluation closes its own episode before L6 releases the session."""


def make_native_es_session(config: Mapping[str, object]) -> NativeESCandidateSession:
    """Public import target for ``CPUWorkerPool(factory_ref=...)``."""
    return NativeESCandidateSession(config)


def make_registered_historical_es_env(*, env_config: dict, candidate_context: dict,
                                      soft_risk_multipliers: dict) -> NautilusBarrierEnv:
    """Build an ES Train environment from the parent-registered scoped market window."""
    from rles_gym_bridge.historical import make_historical_es_env

    registered_access = env_config.get("registered_train_access")
    if registered_access is not None:
        if not isinstance(registered_access, dict):
            raise ValueError("ES worker Train access must be an object")
        # Import lazily so the worker adapter remains independent at module import
        # time; this path repeats the immutable registration idempotently in each
        # process before decoding its scoped market rows.
        from .application import _read_registered_train_store

        prepared_store = _read_registered_train_store(registered_access)
        env_config = {**env_config, "prepared_store": prepared_store}
    return make_historical_es_env(
        env_config=env_config, candidate_context=candidate_context,
        soft_risk_multipliers=soft_risk_multipliers,
    )
