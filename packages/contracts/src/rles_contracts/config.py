"""Strict v0.4.0 config contract; no YAML dependency or hidden auto resolution."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from .errors import ErrorCategory, RLESError
from .serialization import canonical_bytes, canonical_hash
from .validation import require_hash, require_keys

_SCHEME_RE = re.compile(r"^[1-9][0-9]*/[1-9][0-9]*/[1-9][0-9]*$")
_DURATION_RE = re.compile(r"^(0|[1-9][0-9]*)d$")


def _obj(value: object, names: tuple[str, ...], path: str) -> dict[str, Any]:
    if type(value) is not dict:
        raise RLESError(ErrorCategory.CONFIG, "CONFIG_TYPE", f"{path}: expected mapping")
    try:
        require_keys(value, frozenset(names), path=path)
    except (TypeError, ValueError) as exc:
        raise RLESError(ErrorCategory.CONFIG, "CONFIG_FIELDS", str(exc)) from exc
    return value


def _str(value: object, *, label: str, options: tuple[str, ...] | None = None) -> str:
    if type(value) is not str or not value or (options is not None and value not in options):
        raise RLESError(ErrorCategory.CONFIG, "CONFIG_VALUE", f"invalid {label}")
    return value


def _integer(value: object, label: str, *, minimum: int = 1) -> int:
    if type(value) is not int or value < minimum:
        raise RLESError(ErrorCategory.CONFIG, "CONFIG_VALUE", f"{label} must be integer >= {minimum}")
    return value


def _bool(value: object, label: str) -> bool:
    if type(value) is not bool:
        raise RLESError(ErrorCategory.CONFIG, "CONFIG_VALUE", f"{label} must be boolean")
    return value


def _duration(value: object, label: str) -> str:
    if type(value) is not str or not _DURATION_RE.fullmatch(value):
        raise RLESError(ErrorCategory.CONFIG, "CONFIG_VALUE", f"{label} must be duration Nd")
    return value


def _nonempty_list(value: object, label: str, allowed: frozenset[str]) -> tuple[str, ...]:
    if type(value) is not list or not value or any(type(x) is not str or x not in allowed for x in value):
        raise RLESError(ErrorCategory.CONFIG, "CONFIG_VALUE", f"{label}: unknown entry or not list")
    if len(value) != len(set(value)):
        raise RLESError(ErrorCategory.CONFIG, "CONFIG_VALUE", f"{label}: duplicate entries")
    return tuple(value)


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    python: str
    topology: str
    research_sandbox: bool

    def __post_init__(self) -> None:
        _str(self.python, label="runtime.python", options=("3.14.7",))
        _str(self.topology, label="runtime.topology", options=("audit", "cpu", "gpu", "hybrid"))
        if not _bool(self.research_sandbox, "runtime.research_sandbox"):
            raise RLESError(ErrorCategory.SECURITY, "RESEARCH_SANDBOX_REQUIRED", "3.14.7 is research-only")


@dataclass(frozen=True, slots=True)
class FoldConfig:
    scheme: str
    step_days: int
    warmup: str | int
    purge: str | int
    embargo: str
    terminal_policy: str

    def __post_init__(self) -> None:
        scheme = _str(self.scheme, label="fold.scheme")
        if not _SCHEME_RE.fullmatch(scheme):
            raise RLESError(ErrorCategory.CONFIG, "CONFIG_VALUE", "fold.scheme must be N/N/N")
        _integer(self.step_days, "fold.step_days")
        if scheme not in {"60/15/15", "360/15/15"} or self.step_days != 15:
            raise RLESError(
                ErrorCategory.CONFIG,
                "CONFIG_VALUE",
                "only 60/15/15 or 360/15/15 with a 15-day step is supported",
            )
        for field in ("warmup", "purge"):
            value = getattr(self, field)
            if type(value) is str and value == "auto":
                continue
            _integer(value, f"fold.{field}", minimum=1 if field == "warmup" else 0)
        _duration(self.embargo, "fold.embargo")
        _str(self.terminal_policy, label="fold.terminal_policy",
             options=("force_flatten_with_cost", "mark_to_market"))


@dataclass(frozen=True, slots=True)
class ActionConfig:
    space: str
    bounds: tuple[float, float]
    deadband: float

    def __post_init__(self) -> None:
        _str(self.space, label="action.space", options=("target_net_exposure",))
        if (type(self.bounds) is not tuple or len(self.bounds) != 2 or
                any(type(x) is not float for x in self.bounds) or self.bounds != (-1.0, 1.0)):
            raise RLESError(ErrorCategory.CONFIG, "CONFIG_VALUE", "action.bounds must be (-1.0,1.0)")
        if type(self.deadband) is not float or not 0.0 <= self.deadband < 1.0:
            raise RLESError(ErrorCategory.CONFIG, "CONFIG_VALUE", "action.deadband invalid")


@dataclass(frozen=True, slots=True)
class ExecutionConfig:
    policy: str
    pending_order_policy: str
    latency_model: str
    fill_model: str

    def __post_init__(self) -> None:
        _str(self.policy, label="execution.policy", options=("rebalance_market_like",))
        _str(self.pending_order_policy, label="execution.pending_order_policy",
             options=("cancel_replace", "wait", "fail"))
        _str(self.latency_model, label="execution.latency_model")
        _str(self.fill_model, label="execution.fill_model")


@dataclass(frozen=True, slots=True)
class RiskPaths:
    hard_config: str
    soft_params: str

    def __post_init__(self) -> None:
        _str(self.hard_config, label="risk.hard_config")
        _str(self.soft_params, label="risk.soft_params")
        if self.hard_config == self.soft_params:
            raise RLESError(ErrorCategory.CONFIG, "CONFIG_VALUE", "hard and soft files must differ")


@dataclass(frozen=True, slots=True)
class ComputeConfig:
    rl_lane: str
    es_lane: str
    max_cpu_workers: int
    gpu_owner: int
    fallback_policy: str
    auto_select_by: str

    def __post_init__(self) -> None:
        _str(self.rl_lane, label="compute.rl_lane", options=("sync_vec",))
        _str(self.es_lane, label="compute.es_lane", options=("async_candidates",))
        _integer(self.max_cpu_workers, "compute.max_cpu_workers")
        _integer(self.gpu_owner, "compute.gpu_owner", minimum=0)
        _str(self.fallback_policy, label="compute.fallback_policy", options=("fail_fast",))
        _str(self.auto_select_by, label="compute.auto_select_by", options=("end_to_end_wallclock",))


@dataclass(frozen=True, slots=True)
class ResearchConfig:
    baselines: tuple[str, ...]
    statistics: tuple[str, ...]
    holdout_policy: str

    def __post_init__(self) -> None:
        for key, choices in (
            ("baselines", frozenset(("no_trade", "always_long", "simple_momentum", "random_action"))),
            ("statistics", frozenset(("block_bootstrap", "deflated_sharpe", "pbo"))),
        ):
            value = getattr(self, key)
            if (type(value) is not tuple or not value or
                    any(type(x) is not str or x not in choices for x in value) or
                    len(set(value)) != len(value)):
                raise RLESError(ErrorCategory.CONFIG, "CONFIG_VALUE", f"research.{key}: invalid selections")
        _str(self.holdout_policy, label="research.holdout_policy", options=("one_shot",))


@dataclass(frozen=True, slots=True)
class ProjectConfig:
    runtime: RuntimeConfig
    fold: FoldConfig
    action: ActionConfig
    execution: ExecutionConfig
    risk: RiskPaths
    compute: ComputeConfig
    research: ResearchConfig

    def __post_init__(self) -> None:
        for key, klass in (("runtime", RuntimeConfig), ("fold", FoldConfig),
                           ("action", ActionConfig), ("execution", ExecutionConfig),
                           ("risk", RiskPaths), ("compute", ComputeConfig),
                           ("research", ResearchConfig)):
            if type(getattr(self, key)) is not klass:
                raise TypeError(f"config.{key} must be {klass.__name__}")


@dataclass(frozen=True, slots=True)
class ResolvedConfig:
    """Same input+explicit auto resolution yields stable hashes, never overwrites source."""

    config: ProjectConfig
    source_hash: str
    config_hash: str
    contract_version: str = "0.4.0"

    def __post_init__(self) -> None:
        from .schema import require_current_version
        require_current_version(self.contract_version)
        if type(self.config) is not ProjectConfig:
            raise TypeError("resolved.config must be ProjectConfig")
        require_hash(self.source_hash, "source_hash")
        require_hash(self.config_hash, "config_hash")
        if any(getattr(self.config.fold, name) == "auto" for name in ("warmup", "purge")):
            raise ValueError("resolved config may not contain auto fields")
        if self.config_hash != canonical_hash(self.config):
            raise ValueError("resolved config hash mismatch")

    def json_bytes(self) -> bytes:
        return canonical_bytes(self)


def parse_config(document: dict[str, Any]) -> ProjectConfig:
    """Accept already-loaded YAML/JSON mapping, reject every unknown field."""
    root = _obj(document, ("runtime", "fold", "action", "execution", "risk", "compute", "research"), "root")
    r = _obj(root["runtime"], ("python", "topology", "research_sandbox"), "runtime")
    runtime = RuntimeConfig(
        _str(r["python"], label="runtime.python", options=("3.14.7",)),
        _str(r["topology"], label="runtime.topology", options=("audit", "cpu", "gpu", "hybrid")),
        _bool(r["research_sandbox"], "runtime.research_sandbox"),
    )
    if not runtime.research_sandbox:
        raise RLESError(ErrorCategory.SECURITY, "RESEARCH_SANDBOX_REQUIRED", "3.14.7 is research-only")
    f = _obj(root["fold"], ("scheme", "step_days", "warmup", "purge", "embargo", "terminal_policy"), "fold")
    scheme = _str(f["scheme"], label="fold.scheme")
    if not _SCHEME_RE.fullmatch(scheme):
        raise RLESError(ErrorCategory.CONFIG, "CONFIG_VALUE", "fold.scheme must be N/N/N")
    warmup: str | int = f["warmup"]
    purge: str | int = f["purge"]
    for label, value in (("warmup", warmup), ("purge", purge)):
        if (value != "auto" if type(value) is str else (type(value) is not int or value < 0)):
            raise RLESError(ErrorCategory.CONFIG, "CONFIG_VALUE", f"fold.{label} requires auto or nonnegative days")
    fold = FoldConfig(scheme, _integer(f["step_days"], "fold.step_days"), warmup, purge,
                      _duration(f["embargo"], "fold.embargo"),
                      _str(f["terminal_policy"], label="fold.terminal_policy",
                           options=("force_flatten_with_cost", "mark_to_market")))
    a = _obj(root["action"], ("space", "bounds", "deadband"), "action")
    bounds = a["bounds"]
    if type(bounds) is not list or len(bounds) != 2 or any(type(x) not in (int, float) for x in bounds):
        raise RLESError(ErrorCategory.CONFIG, "CONFIG_VALUE", "action.bounds must be numeric pair")
    if bounds[0] != -1 or bounds[1] != 1:
        raise RLESError(ErrorCategory.CONFIG, "CONFIG_VALUE", "action.bounds must equal [-1,1]")
    band = a["deadband"]
    if type(band) not in (int, float) or not 0 <= band < 1:
        raise RLESError(ErrorCategory.CONFIG, "CONFIG_VALUE", "action.deadband invalid")
    action = ActionConfig(_str(a["space"], label="action.space", options=("target_net_exposure",)),
                          (-1.0, 1.0), float(band))
    e = _obj(root["execution"], ("policy", "pending_order_policy", "latency_model", "fill_model"), "execution")
    execution = ExecutionConfig(
        _str(e["policy"], label="execution.policy", options=("rebalance_market_like",)),
        _str(e["pending_order_policy"], label="execution.pending_order_policy",
             options=("cancel_replace", "wait", "fail")),
        _str(e["latency_model"], label="execution.latency_model"),
        _str(e["fill_model"], label="execution.fill_model"),
    )
    p = _obj(root["risk"], ("hard_config", "soft_params"), "risk")
    hard_path = _str(p["hard_config"], label="risk.hard_config")
    soft_path = _str(p["soft_params"], label="risk.soft_params")
    if hard_path == soft_path:
        raise RLESError(ErrorCategory.CONFIG, "CONFIG_VALUE", "hard and soft risk files must be separate")
    risk = RiskPaths(hard_path, soft_path)
    c = _obj(root["compute"], ("rl_lane", "es_lane", "max_cpu_workers", "gpu_owner",
                               "fallback_policy", "auto_select_by"), "compute")
    compute = ComputeConfig(
        _str(c["rl_lane"], label="compute.rl_lane", options=("sync_vec",)),
        _str(c["es_lane"], label="compute.es_lane", options=("async_candidates",)),
        _integer(c["max_cpu_workers"], "compute.max_cpu_workers"),
        _integer(c["gpu_owner"], "compute.gpu_owner", minimum=0),
        _str(c["fallback_policy"], label="compute.fallback_policy", options=("fail_fast",)),
        _str(c["auto_select_by"], label="compute.auto_select_by", options=("end_to_end_wallclock",)),
    )
    t = _obj(root["research"], ("baselines", "statistics", "holdout_policy"), "research")
    research = ResearchConfig(
        _nonempty_list(t["baselines"], "research.baselines",
                       frozenset(("no_trade", "always_long", "simple_momentum", "random_action"))),
        _nonempty_list(t["statistics"], "research.statistics",
                       frozenset(("block_bootstrap", "deflated_sharpe", "pbo"))),
        _str(t["holdout_policy"], label="research.holdout_policy", options=("one_shot",)),
    )
    return ProjectConfig(runtime, fold, action, execution, risk, compute, research)


def resolve_config(document: dict[str, Any], *, auto_days: dict[str, int]) -> ResolvedConfig:
    """Explicit auto_days keys 'fold.warmup' / 'fold.purge'; fail if unresolved or unused."""
    from dataclasses import replace
    if type(auto_days) is not dict:
        raise TypeError("auto_days must be a dict")
    config = parse_config(document)
    used: set[str] = set()
    changes: dict[str, Any] = {}
    for field in ("warmup", "purge"):
        current = getattr(config.fold, field)
        key = f"fold.{field}"
        if current == "auto":
            if key not in auto_days:
                raise RLESError(ErrorCategory.CONFIG, "AUTO_UNRESOLVED", f"must resolve {key}")
            days = _integer(auto_days[key], key, minimum=0)
            changes[field] = days
            used.add(key)
    if set(auto_days) != used:
        raise RLESError(ErrorCategory.CONFIG, "AUTO_EXTRA", f"unexpected auto overrides: {sorted(set(auto_days) - used)}")
    resolved = replace(config, fold=replace(config.fold, **changes))
    # A plain typed DTO uses generic canonical serialization without auto importing YAML.
    return ResolvedConfig(resolved, canonical_hash(document), canonical_hash(resolved))
