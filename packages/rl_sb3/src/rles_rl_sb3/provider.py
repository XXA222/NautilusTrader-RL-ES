"""The standalone L4 RLProvider, with native SB3 algorithms and Q-gate injection."""
from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from array import array
from dataclasses import dataclass
import heapq
from pathlib import Path
import time
import sys
from typing import Any, Literal

import numpy as np

from rles_contracts.registry import ProviderDescriptor

from .actor import (ParityReport, action_parity, actor_mapping,
                    behavioral_clone, export_l0, transfer_actor)
from .checkpoint import Boundary, checkpoint_load, checkpoint_save
from .config import RLConfig
from .envs import (EnvFactory, assert_resume_proof, assert_transition_proof,
                   check_budget, make_sync_env)
from .errors import ContractError, GateBlocked
from .noise import make_noise
from .normalize import (assert_frozen, attach_train, export_stats,
                        freeze_for_eval)

GateAuthorizer = Callable[[tuple[str, ...], RLConfig], bool]

# Fixed, raw-ratio boundaries keep volatility summaries comparable across
# learn calls. They are descriptive bins, not strategy thresholds or fitted
# estimates; equality belongs to the lower (upper-closed) bucket.
_VOLATILITY_REGIME_CUTS = (0.00025, 0.0005, 0.001)


@dataclass(frozen=True)
class Evaluation:
    """Gym returns only: L6 computes financial metrics from L3 account facts."""
    algorithm: str
    episode_returns: tuple[float, ...]
    episode_lengths: tuple[int, ...]
    deterministic: bool
    normalization_frozen: bool


@dataclass(frozen=True)
class TrainingDiagnostics:
    """Deterministic learner and native-transition facts for one learn call."""
    samples: int
    episodes_completed: int
    decision_start_ns: int
    decision_end_ns: int
    unique_decision_timestamps: int
    decision_timestamp_revisit_fraction: float
    r_econ_sum: float
    r_econ_mean: float
    r_econ_std: float
    r_shape_sum: float
    r_train_sum: float
    r_train_mean: float
    r_train_std: float
    requested_action_mean: float
    requested_action_abs_mean: float
    requested_action_std: float
    executed_action_mean: float
    executed_action_abs_mean: float
    requested_target_change_abs_mean: float
    executed_exposure_change_abs_mean: float
    guard_intervention_mean: float
    guard_intervention_fraction: float
    soft_risk_adjustment_mean: float
    soft_risk_intervention_fraction: float
    hard_risk_adjustment_mean: float
    hard_risk_intervention_fraction: float
    requested_to_executed_abs_mean: float
    execution_plan_reasons: tuple[tuple[str, int], ...]
    submit_fraction: float
    noop_fraction: float
    optimizer_metrics: tuple[tuple[str, float], ...]
    optimizer_undefined_metrics: tuple[str, ...]
    # (requested-|action| bucket, count, r_econ sum/mean, r_shape sum,
    # hard-risk intervention fraction, submit fraction, mean executed |exposure|).
    # This is descriptive on-policy attribution, not a causal action effect.
    requested_action_economic_bins: tuple[
        tuple[str, int, float, float, float, float, float, float], ...
    ]
    # Exact per-learn-call facts. Market summaries are over observed L2 features;
    # they do not claim calibrated regime boundaries.
    position_state_counts: tuple[tuple[str, int], ...]
    market_feature_summaries: tuple[tuple[str, int, float, float, float, float], ...]
    # (family, feature, scheme, bucket, lower, upper, count, r_econ sum, mean).
    # Volatility cut points are empirical quartiles from this Train-only learn call.
    market_regime_economic_bins: tuple[
        tuple[str, str, str, str, float | None, float | None, int, float, float], ...
    ]
    episode_window_overlap_pairs: int


@dataclass(frozen=True)
class Performance:
    samples: int
    steps_per_second: float
    updates: int | None
    updates_per_second: float | None
    elapsed_seconds: float
    cpu_rss_bytes: int | None
    gpu_max_allocated_bytes: int | None
    diagnostics: TrainingDiagnostics | None = None
    gpu_max_reserved_bytes: int | None = None


class _Moments:
    """Stable-enough streaming moments for bounded diagnostic values."""

    def __init__(self) -> None:
        self.count = 0
        self.total = 0.0
        self.mean = 0.0
        self.m2 = 0.0

    def add(self, value: Any, name: str) -> None:
        if type(value) not in (int, float, np.float32, np.float64):
            raise ContractError(f"native training diagnostic {name} is not numeric")
        observed = float(value)
        if not np.isfinite(observed):
            raise ContractError(f"native training diagnostic {name} is not finite")
        self.count += 1
        self.total += observed
        delta = observed - self.mean
        self.mean += delta / self.count
        self.m2 += delta * (observed - self.mean)

    @property
    def std(self) -> float:
        return float(np.sqrt(max(self.m2 / self.count, 0.0))) if self.count else 0.0


def _perf_rss() -> int | None:
    try:
        import psutil  # type: ignore[import-untyped]
    except ImportError:
        return None
    return int(psutil.Process().memory_info().rss)


def _count_episode_window_overlap_pairs(windows: Iterable[tuple[int, int]]) -> int:
    """Count inclusive timestamp-window intersections in O(n log n) time."""
    active_ends: list[int] = []
    overlap_pairs = 0
    for start_ns, end_ns in sorted(windows):
        while active_ends and active_ends[0] < start_ns:
            heapq.heappop(active_ends)
        overlap_pairs += len(active_ends)
        heapq.heappush(active_ends, end_ns)
    return overlap_pairs


def _market_regime_bins(
    samples: Mapping[str, tuple[str, array, array]],
) -> tuple[tuple[str, str, str, str, float | None, float | None, int, float, float], ...]:
    """Describe Train-call rewards across fixed, predeclared feature regimes."""
    output = []
    for family in ("volatility", "trend", "funding"):
        item = samples.get(family)
        if item is None:
            continue
        feature_name, raw_values, raw_rewards = item
        values = np.frombuffer(raw_values, dtype=np.float64)
        rewards = np.frombuffer(raw_rewards, dtype=np.float64)
        if len(values) != len(rewards) or not len(values):
            raise ContractError("market regime diagnostic sample alignment failed")
        labels: tuple[str, ...]
        if family == "volatility":
            cuts = np.asarray(_VOLATILITY_REGIME_CUTS, dtype=np.float64)
            bucket_indices = np.searchsorted(cuts, values, side="left")
            scheme = "fixed_absolute_volatility_upper_closed_v1"
            labels = ("le_0.00025", "0.00025_to_0.0005", "0.0005_to_0.001", "gt_0.001")
            bounds: tuple[tuple[float | None, float | None], ...] = (
                (None, float(cuts[0])), (float(cuts[0]), float(cuts[1])),
                (float(cuts[1]), float(cuts[2])), (float(cuts[2]), None),
            )
        else:
            bucket_indices = np.where(values < 0, 0, np.where(values > 0, 2, 1))
            scheme = "signed_feature_v1"
            labels = ("negative", "zero", "positive")
            bounds = ((None, 0.0), (0.0, 0.0), (0.0, None))
        for index, label in enumerate(labels):
            selected = rewards[bucket_indices == index]
            total = float(np.sum(selected, dtype=np.float64))
            count = int(selected.size)
            lower, upper = bounds[index]
            output.append((family, feature_name, scheme, label, lower, upper,
                           count, total, total / count if count else 0.0))
    return tuple(output)


class RLProvider:
    """Train/save/load/evaluate actor on an externally owned frozen Gym L3 Env."""

    def __init__(self, config: RLConfig, env_factory: EnvFactory,
                 gate_authorizer: GateAuthorizer):
        if not isinstance(config, RLConfig):
            raise ContractError("Config must be strictly parsed RLConfig")
        if not callable(env_factory) or not callable(gate_authorizer):
            raise ContractError("L3 environment factory and external gate authorizer are mandatory")
        self.config = config
        self.env_factory = env_factory
        self.gate_authorizer = gate_authorizer
        self._normalization_calibrated = False
        self.model: Any | None = None
        self.vec: Any | None = None
        self._perf: Performance | None = None
        self._tainted = False
        self._restored = False
        self._recipe_hash = config.hash

    def _safe(self) -> None:
        if self._tainted:
            raise GateBlocked("L4 attempt invalidated by a previous training/restore failure")
        if self.config.hash != self._recipe_hash:
            self._tainted = True
            raise ContractError("Live algorithm recipe has changed after provider construction")

    def _gates(self, *keys: str) -> None:
        self._safe()
        # The owner/orchestrator must implement certificate verification. A
        # provider never self-signs Q0/Q3 nor silently substitutes sample Env.
        required = tuple(dict.fromkeys(("Q0", "Q3", *keys)))
        if self.gate_authorizer(required, self.config) is not True:
            raise GateBlocked(f"External sign-off required: {required}")

    def _runtime(self) -> None:
        """G0 dependency pin: no fallback across Python/SB3/Torch ABI."""
        if sys.implementation.name != "cpython" or sys.version_info[:2] != (3, 14):
            raise GateBlocked("L4 production runtime requires CPython 3.14.x G0 approval")
        import stable_baselines3 as sb3
        import gymnasium
        import torch
        if sb3.__version__ != "2.9.0":
            raise GateBlocked("L4 contract requires exact Stable-Baselines3 2.9.0")
        if tuple(int(part) for part in torch.__version__.split("+")[0].split(".")[:2]) != (2, 14):
            raise GateBlocked("L4 contract requires PyTorch 2.14.x")
        if tuple(int(part) for part in gymnasium.__version__.split(".")[:2]) not in ((1, 2), (1, 3)):
            raise GateBlocked("L4 contract requires Gymnasium 1.2.x/1.3.x")

    def _new_vec(self, phase: Literal["train", "eval"]):
        seeds = self.config.seeds.environments
        if phase == "eval":
            # Independent deterministic eval stream, never a training episode.
            seeds = tuple((s + 0x9e3779b9) % 2**32 for s in seeds)
        return make_sync_env(self.env_factory, seeds, phase, self.config.policy)

    def _configure_cuda_allocator(self, torch: Any) -> None:
        fraction = self.config.cuda_allocator_fraction
        if fraction is None:
            return
        if self.config.device != "cuda":
            raise ContractError("CUDA allocator budget requires CUDA device")
        if not torch.cuda.is_available():
            raise GateBlocked("CUDA requested but unavailable; never silently fall back to CPU")
        # PyTorch applies this to its caching allocator only, so set it before
        # constructing vectors or loading a model that may allocate CUDA tensors.
        torch.cuda.set_per_process_memory_fraction(fraction, torch.cuda.current_device())

    def create(self) -> RLProvider:
        self._gates("L3-020", "L3-021", "L1-013")
        if self.model is not None:
            raise ContractError("Learner already constructed")
        self._runtime()
        if self.config.device == "cuda":
            import torch
            if not torch.cuda.is_available():
                raise GateBlocked("CUDA requested but unavailable; never silently fall back to CPU")
            self._configure_cuda_allocator(torch)
        vec = self._new_vec("train")
        try:
            assert_transition_proof(vec, self.config.fingerprint.hash)
            self.vec = attach_train(vec, self.config.normalize_observation)
        except BaseException:
            vec.close()
            raise
        import torch
        from stable_baselines3 import DDPG, PPO, SAC, TD3
        activation = torch.nn.Tanh if self.config.policy.activation == "tanh" else torch.nn.ReLU
        policy_kwargs: dict[str, Any]
        algo: Any
        kwargs = dict(self.config.hyperparameters)
        if self.config.algorithm == "PPO":
            log_std_init = kwargs.pop("log_std_init", 0.0)
            zero_init_action_head = kwargs.pop("zero_init_action_head", False)
            policy_kwargs = {
                "net_arch": {"pi": list(self.config.policy.hidden_layers),
                             "vf": list(self.config.policy.value_layers)},
                # PPO receives `use_sde` at algorithm level; duplicating it in
                # policy_kwargs raises TypeError in SB3 2.9.0.
                "activation_fn": activation, "ortho_init": False,
                "log_std_init": log_std_init,
            }
            algo = PPO
        else:
            zero_init_action_head = False
            policy_kwargs = {
                "net_arch": list(self.config.policy.hidden_layers),
                "activation_fn": activation,
            }
            algo = {"SAC": SAC, "TD3": TD3, "DDPG": DDPG}[self.config.algorithm]
            if self.config.algorithm == "DDPG":
                # SB3 DDPG is implemented as a TD3 subclass with a single Q critic.
                # Only expose DDPG's native API; do not pass TD3 smoothing/delay keys.
                policy_kwargs["n_critics"] = 1
        if self.config.algorithm in ("TD3", "DDPG"):
            kwargs["action_noise"] = make_noise(
                self.config.noise_sigma, self.config.seeds.noise, self.config.num_envs)
        # noise_sigma for SAC is not consumed; entropy is native SAC.
        try:
            self.model = algo("MlpPolicy", self.vec, policy_kwargs=policy_kwargs,
                              seed=self.config.seeds.algorithm, device=self.config.device,
                              verbose=0, **kwargs)
            if zero_init_action_head:
                import torch
                with torch.no_grad():
                    self.model.policy.action_net.weight.zero_()
                    self.model.policy.action_net.bias.zero_()
        except BaseException:
            self.close()
            raise
        # Native SB3 replay sampling uses the NumPy process-global generator;
        # L4 records a dedicated L1 replay seed and snapshots that RNG at L1.
        np.random.seed(self.config.seeds.replay)
        return self

    def train(self, total_timesteps: int, *, resumed: bool = False,
              callback: Any | None = None) -> Performance:
        self._gates("L3-021")
        if self.model is None:
            raise ContractError("Call create() or resume() first")
        if self._restored and not resumed:
            raise ContractError("Loaded L1 stage must be resumed, not restarted")
        if resumed and self.model.num_timesteps == 0:
            raise ContractError("Cannot continue an untrained learner with resumed=True")
        if not resumed and self.model.num_timesteps != 0:
            raise ContractError("Existing learner must be continued with resumed=True")
        check_budget(self.config.algorithm, dict(self.config.hyperparameters),
                     self.config.num_envs, total_timesteps)
        previous = getattr(self.model, "_n_updates", None)
        t0 = time.perf_counter()
        import torch
        if self.config.device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        start_steps = int(self.model.num_timesteps)
        from stable_baselines3.common.callbacks import BaseCallback, CallbackList, ConvertCallback

        expected = self.config.fingerprint.hash
        environment_count = self.config.num_envs
        econ = _Moments()
        shape = _Moments()
        train_reward = _Moments()
        requested = _Moments()
        requested_abs = _Moments()
        executed = _Moments()
        executed_abs = _Moments()
        requested_change = _Moments()
        executed_change = _Moments()
        guard_values = _Moments()
        soft_risk_values = _Moments()
        hard_risk_values = _Moments()
        requested_executed_delta = _Moments()
        execution_reason_counts: dict[str, int] = {}
        action_bin_edges = (0.05, 0.15, 0.30, 0.60)
        action_bin_names = ("abs_lt_0.05", "abs_0.05_0.15", "abs_0.15_0.30",
                            "abs_0.30_0.60", "abs_0.60_1.00")
        action_bin_values = {name: [0, 0.0, 0.0, 0, 0, 0.0] for name in action_bin_names}
        decision_timestamps: set[int] = set()
        decision_start_ns: int | None = None
        decision_end_ns: int | None = None
        market_feature_values: dict[str, _Moments] = {}
        market_feature_minmax: dict[str, tuple[float, float]] = {}
        market_regime_samples: dict[str, tuple[str, array, array]] = {}
        position_state_counts: dict[str, int] = {"flat": 0, "long": 0, "short": 0}
        episode_windows: dict[tuple[int, int, int], tuple[int, int]] = {}
        previous_requested: list[float | None] = [None] * self.config.num_envs
        previous_executed: list[float | None] = [None] * self.config.num_envs
        diagnostic_counts = {
            "episodes": 0, "guarded": 0, "soft_guarded": 0,
            "hard_guarded": 0, "submit": 0, "noop": 0,
        }

        class TransitionGuard(BaseCallback):
            def _on_step(self) -> bool:
                assert_transition_proof(self.training_env, expected)
                return True

        class DiagnosticsCallback(BaseCallback):
            def _on_step(self) -> bool:
                nonlocal decision_start_ns, decision_end_ns
                infos = self.locals.get("infos")
                rewards = np.asarray(self.locals.get("rewards"), dtype=np.float64).reshape(-1)
                dones = np.asarray(self.locals.get("dones"), dtype=bool).reshape(-1)
                if (not isinstance(infos, list) or len(infos) != environment_count
                        or rewards.shape != (environment_count,)
                        or dones.shape != (environment_count,)):
                    raise ContractError("SB3 training callback lacks per-environment native facts")
                required = {
                    "r_econ", "r_shape", "r_train", "requested_action",
                    "policy_executed_action_pre_terminal", "risk_guard_intervention",
                    "risk_soft_adjustment", "risk_hard_adjustment",
                    "execution_plan_action", "execution_plan_reason", "decision_ts_ns",
                    "episode_window", "market_features", "net_exposure",
                }
                for rank, info in enumerate(infos):
                    if not isinstance(info, dict) or not required <= set(info):
                        raise ContractError("native training info lacks learning diagnostics")
                    decision_ts_ns = info["decision_ts_ns"]
                    if type(decision_ts_ns) is not int or decision_ts_ns <= 0:
                        raise ContractError("native training decision timestamp is invalid")
                    decision_timestamps.add(decision_ts_ns)
                    decision_start_ns = (decision_ts_ns if decision_start_ns is None
                                         else min(decision_start_ns, decision_ts_ns))
                    decision_end_ns = (decision_ts_ns if decision_end_ns is None
                                       else max(decision_end_ns, decision_ts_ns))
                    window = info["episode_window"]
                    if not isinstance(window, dict) or not {
                        "start_ns", "end_ns", "episode_index", "start_offset", "decision_steps",
                    } <= set(window):
                        raise ContractError("native training info lacks the exact episode window")
                    window_values = tuple(window[key] for key in (
                        "start_ns", "end_ns", "episode_index", "start_offset", "decision_steps",
                    ))
                    if any(type(value) is not int or value < 0 for value in window_values):
                        raise ContractError("native training episode window is invalid")
                    window_key = (rank, window["episode_index"], window["start_offset"])
                    episode_windows[window_key] = (window["start_ns"], window["end_ns"])
                    features = info["market_features"]
                    if not isinstance(features, dict):
                        raise ContractError("native training market features are invalid")
                    for feature_name, feature_value in features.items():
                        if type(feature_name) is not str or not feature_name:
                            raise ContractError("native training market feature name is invalid")
                        moments = market_feature_values.setdefault(feature_name, _Moments())
                        moments.add(feature_value, f"market.{feature_name}")
                        number = float(feature_value)
                        low, high = market_feature_minmax.get(feature_name, (number, number))
                        market_feature_minmax[feature_name] = (min(low, number), max(high, number))
                    regime_features: tuple[tuple[str, tuple[str, ...]], ...] = (
                        ("volatility", ("volatility_5m", "volatility_15m", "volatility_60m")),
                        ("trend", ("return_15m", "return_5m", "return_1m")),
                        ("funding", ("last_settled_funding",)),
                    )
                    for family, candidates in regime_features:
                        feature_name = next((name for name in candidates if name in features), None)
                        if feature_name is None:
                            continue
                        current = market_regime_samples.get(family)
                        if current is None:
                            current = (feature_name, array("d"), array("d"))
                            market_regime_samples[family] = current
                        elif current[0] != feature_name:
                            raise ContractError("market regime feature changed during one learn call")
                        current[1].append(float(features[feature_name]))
                        current[2].append(float(info["r_econ"]))
                    net_exposure = info["net_exposure"]
                    if type(net_exposure) not in (int, float) or not np.isfinite(float(net_exposure)):
                        raise ContractError("native training net exposure is invalid")
                    state = "flat" if abs(float(net_exposure)) <= 1e-12 else (
                        "long" if float(net_exposure) > 0 else "short"
                    )
                    position_state_counts[state] += 1
                    econ.add(info["r_econ"], "r_econ")
                    shape.add(info["r_shape"], "r_shape")
                    train_reward.add(info["r_train"], "r_train")
                    if not np.isclose(float(info["r_train"]), rewards[rank], atol=1e-7, rtol=1e-6):
                        raise ContractError("SB3 reward differs from native r_train")
                    requested_value = float(info["requested_action"])
                    executed_value = float(info["policy_executed_action_pre_terminal"])
                    requested.add(requested_value, "requested_action")
                    requested_abs.add(abs(requested_value), "requested_action_abs")
                    executed.add(executed_value, "executed_action")
                    executed_abs.add(abs(executed_value), "executed_action_abs")
                    guard = float(info["risk_guard_intervention"])
                    guard_values.add(guard, "risk_guard_intervention")
                    diagnostic_counts["guarded"] += int(guard > 0)
                    soft_adjustment = float(info["risk_soft_adjustment"])
                    hard_adjustment = float(info["risk_hard_adjustment"])
                    soft_risk_values.add(soft_adjustment, "risk_soft_adjustment")
                    hard_risk_values.add(hard_adjustment, "risk_hard_adjustment")
                    diagnostic_counts["soft_guarded"] += int(soft_adjustment > 0)
                    diagnostic_counts["hard_guarded"] += int(hard_adjustment > 0)
                    requested_executed_delta.add(
                        abs(requested_value - executed_value), "requested_to_executed_action_delta",
                    )
                    action = info["execution_plan_action"]
                    if action not in {"SUBMIT", "NOOP", "WAIT", "CANCEL"}:
                        raise ContractError("native execution-plan action is invalid")
                    reason = info["execution_plan_reason"]
                    if not isinstance(reason, str) or not reason:
                        raise ContractError("native execution-plan reason is invalid")
                    execution_reason_counts[reason] = execution_reason_counts.get(reason, 0) + 1
                    diagnostic_counts["submit"] += int(action == "SUBMIT")
                    diagnostic_counts["noop"] += int(action == "NOOP")
                    magnitude = abs(requested_value)
                    bin_index = next((i for i, edge in enumerate(action_bin_edges) if magnitude < edge),
                                     len(action_bin_names) - 1)
                    bin_values = action_bin_values[action_bin_names[bin_index]]
                    bin_values[0] += 1
                    bin_values[1] += float(info["r_econ"])
                    bin_values[2] += float(info["r_shape"])
                    bin_values[3] += int(float(info["risk_hard_adjustment"]) > 0)
                    bin_values[4] += int(action == "SUBMIT")
                    bin_values[5] += abs(executed_value)
                    previous_requested_value = previous_requested[rank]
                    previous_executed_value = previous_executed[rank]
                    if (previous_requested_value is None) != (previous_executed_value is None):
                        raise ContractError("training action diagnostic state is inconsistent")
                    if previous_requested_value is not None and previous_executed_value is not None:
                        requested_change.add(
                            abs(requested_value - previous_requested_value),
                            "requested_target_change",
                        )
                        executed_change.add(
                            abs(executed_value - previous_executed_value),
                            "executed_exposure_change",
                        )
                    if dones[rank]:
                        diagnostic_counts["episodes"] += 1
                        previous_requested[rank] = None
                        previous_executed[rank] = None
                    else:
                        previous_requested[rank] = requested_value
                        previous_executed[rank] = executed_value
                return True

        guard = TransitionGuard()
        diagnostics_callback = DiagnosticsCallback()
        callbacks: list[BaseCallback]
        if callback is None:
            callbacks = [guard, diagnostics_callback]
        elif isinstance(callback, BaseCallback):
            callbacks = [guard, diagnostics_callback, callback]
        elif callable(callback):
            callbacks = [guard, diagnostics_callback, ConvertCallback(callback)]
        else:
            raise ContractError("SB3 callback must be BaseCallback or callable")
        assert_transition_proof(self.vec, expected)
        try:
            self.model.learn(total_timesteps=total_timesteps,
                             reset_num_timesteps=not resumed,
                             callback=CallbackList(callbacks), progress_bar=False)
            assert_transition_proof(self.vec, expected)
        except BaseException:
            self._tainted = True
            raise
        elapsed = time.perf_counter() - t0
        n = int(self.model.num_timesteps) - start_steps
        if n != total_timesteps:
            self._tainted = True
            raise ContractError(f"SB3 sampled {n}, expected exactly {total_timesteps}")
        after = getattr(self.model, "_n_updates", None)
        delta = (after - previous) if isinstance(after, int) and isinstance(previous, int) else None
        if econ.count != n or any(values.count != n for values in (
            shape, train_reward, requested, requested_abs, executed, executed_abs,
            guard_values, soft_risk_values, hard_risk_values, requested_executed_delta,
        )):
            self._tainted = True
            raise ContractError("native learning diagnostic count differs from sampled transitions")
        optimizer_metrics: list[tuple[str, float]] = []
        optimizer_undefined_metrics: list[str] = []
        for name, value in sorted(self.model.logger.name_to_value.items()):
            if not name.startswith("train/"):
                continue
            scalar = np.asarray(value)
            if scalar.size != 1 or not np.issubdtype(scalar.dtype, np.number):
                continue
            parsed = float(scalar.reshape(()))
            if not np.isfinite(parsed):
                if name == "train/explained_variance":
                    optimizer_undefined_metrics.append(name)
                    continue
                self._tainted = True
                raise ContractError(f"SB3 optimizer diagnostic {name} is not finite")
            optimizer_metrics.append((name, parsed))
        diagnostics = TrainingDiagnostics(
            samples=n,
            episodes_completed=diagnostic_counts["episodes"],
            decision_start_ns=decision_start_ns or 0,
            decision_end_ns=decision_end_ns or 0,
            unique_decision_timestamps=len(decision_timestamps),
            decision_timestamp_revisit_fraction=1.0 - len(decision_timestamps) / n,
            r_econ_sum=econ.total, r_econ_mean=econ.mean, r_econ_std=econ.std,
            r_shape_sum=shape.total,
            r_train_sum=train_reward.total, r_train_mean=train_reward.mean,
            r_train_std=train_reward.std,
            requested_action_mean=requested.mean,
            requested_action_abs_mean=requested_abs.mean,
            requested_action_std=requested.std,
            executed_action_mean=executed.mean,
            executed_action_abs_mean=executed_abs.mean,
            requested_target_change_abs_mean=requested_change.mean,
            executed_exposure_change_abs_mean=executed_change.mean,
            guard_intervention_mean=guard_values.mean,
            guard_intervention_fraction=diagnostic_counts["guarded"] / n,
            soft_risk_adjustment_mean=soft_risk_values.mean,
            soft_risk_intervention_fraction=diagnostic_counts["soft_guarded"] / n,
            hard_risk_adjustment_mean=hard_risk_values.mean,
            hard_risk_intervention_fraction=diagnostic_counts["hard_guarded"] / n,
            requested_to_executed_abs_mean=requested_executed_delta.mean,
            execution_plan_reasons=tuple(sorted(execution_reason_counts.items())),
            submit_fraction=diagnostic_counts["submit"] / n,
            noop_fraction=diagnostic_counts["noop"] / n,
            optimizer_metrics=tuple(optimizer_metrics),
            optimizer_undefined_metrics=tuple(optimizer_undefined_metrics),
            requested_action_economic_bins=tuple(
                (name, int(values[0]), values[1], values[1] / values[0] if values[0] else 0.0,
                 values[2], values[3] / values[0] if values[0] else 0.0,
                 values[4] / values[0] if values[0] else 0.0,
                 values[5] / values[0] if values[0] else 0.0)
                for name, values in action_bin_values.items()
            ),
            position_state_counts=tuple(sorted(position_state_counts.items())),
            market_feature_summaries=tuple(
                (name, values.count, values.mean, values.std,
                 market_feature_minmax[name][0], market_feature_minmax[name][1])
                for name, values in sorted(market_feature_values.items())
            ),
            market_regime_economic_bins=_market_regime_bins(market_regime_samples),
            episode_window_overlap_pairs=_count_episode_window_overlap_pairs(
                episode_windows.values(),
            ),
        )
        self._perf = Performance(samples=n, steps_per_second=n / max(elapsed, 1e-9),
                                 updates=delta, updates_per_second=None if delta is None else delta / max(elapsed, 1e-9),
                                 elapsed_seconds=elapsed, cpu_rss_bytes=_perf_rss(),
                                 gpu_max_allocated_bytes=torch.cuda.max_memory_allocated() if self.config.device == "cuda" else None,
                                 gpu_max_reserved_bytes=torch.cuda.max_memory_reserved() if self.config.device == "cuda" else None,
                                 diagnostics=diagnostics)
        return self._perf

    def calibrate_observation_normalization(self, raw_observations: np.ndarray, *,
                                            training_split: str) -> dict[str, Any]:
        """Fit once on the E1 Train corpus before R1 updates or BC.

        Raw account amounts otherwise collapse to the same VecNormalize clip
        value before any samples have fitted its statistics.
        """
        self._gates("L5-003")
        if (training_split != "train_inner" or self.model is None or self.model.num_timesteps != 0
                or self._restored or self._normalization_calibrated):
            raise ContractError("normalization calibration requires fresh pre-training Train-inner input")
        self.policy_observations(raw_observations)  # Validate raw shape/dtype/finiteness.
        from stable_baselines3.common.vec_env import VecNormalize

        if not isinstance(self.vec, VecNormalize) or not self.vec.training or self.vec.norm_reward:
            raise ContractError("calibration requires training observation-only VecNormalize")
        rms = self.vec.obs_rms
        if isinstance(rms, dict):
            raise ContractError("Dict observation normalization is not supported")
        rms.update(np.asarray(raw_observations, dtype=np.float64))
        standardized = (np.asarray(raw_observations, dtype=np.float64) - rms.mean) / np.sqrt(rms.var + self.vec.epsilon)
        # Preserve information in the complete Train transfer corpus. The
        # fitted bound is frozen into normalizer/checkpoint/inference metadata.
        self.vec.clip_obs = max(float(self.vec.clip_obs), float(np.max(np.abs(standardized))) * 1.001)
        self._normalization_calibrated = True
        stats = export_stats(self.vec)
        if stats is None:
            raise ContractError("calibrated observation statistics are unavailable")
        return stats

    def policy_observations(self, raw_observation: np.ndarray) -> np.ndarray:
        """Apply the current inference transform without fitting running statistics.

        Transfer/BC accept policy-space inputs; teachers and public inference
        consume raw inputs. Keeping this conversion explicit prevents mixing
        the two domains when VecNormalize is enabled.
        """
        self._safe()
        if self.model is None or self.vec is None:
            raise ContractError("No trained/loaded model")
        x = np.asarray(raw_observation)
        if (x.dtype != np.float32 or x.ndim != 2 or x.shape[0] == 0
                or x.shape[1] != self.config.policy.observation_dim or not np.isfinite(x).all()):
            raise ContractError("Expected finite float32 batch raw observations")
        if self.config.normalize_observation:
            x = self.vec.normalize_obs(x)
        return np.asarray(x, dtype=np.float32).copy()

    def predict(self, raw_observation: np.ndarray, *, deterministic: bool = True) -> np.ndarray:
        x = self.policy_observations(raw_observation)
        if self.model is None:
            raise ContractError("No trained/loaded model")
        result, _ = self.model.predict(x, deterministic=deterministic)
        result = np.asarray(result, dtype=np.float32).reshape((-1, 1))
        if (result.shape != (len(x), 1) or not np.isfinite(result).all()
                or np.any(result < -1.000001) or np.any(result > 1.000001)):
            raise ContractError("Model returned invalid action")
        return result

    def evaluate(self, *, episodes: int, max_steps: int = 1_000_000) -> Evaluation:
        self._gates("Q3", "L3-020", "L3-021")
        if self.model is None or self.vec is None:
            raise ContractError("Model missing")
        if type(episodes) is not int or episodes <= 0 or type(max_steps) is not int or max_steps <= 0:
            raise ContractError("Invalid evaluation budget")
        raw = self._new_vec("eval")
        assert_transition_proof(raw, self.config.fingerprint.hash)
        ev = None
        try:
            ev = freeze_for_eval(self.vec, raw)
            assert_frozen(ev)
            before = export_stats(ev)
            obs = ev.reset()
            totals = np.zeros(ev.num_envs, dtype=np.float64)
            lengths = np.zeros(ev.num_envs, dtype=np.int64)
            returns: list[float] = []
            durations: list[int] = []
            step_count = 0
            while len(returns) < episodes:
                if step_count >= max_steps:
                    raise GateBlocked("Evaluation step budget exceeded")
                actions, _ = self.model.predict(obs, deterministic=True)
                obs, rewards, done, info = ev.step(actions)
                assert_transition_proof(ev, self.config.fingerprint.hash)
                if not np.isfinite(obs).all() or not np.isfinite(rewards).all():
                    raise ContractError("Evaluation produced NaN/Inf")
                totals += rewards
                lengths += 1
                step_count += 1
                for i in range(ev.num_envs):
                    if done[i]:
                        if len(returns) < episodes:
                            returns.append(float(totals[i]))
                            durations.append(int(lengths[i]))
                        totals[i] = 0
                        lengths[i] = 0
            if before != export_stats(ev):
                raise ContractError("Evaluation updated frozen normalization statistics")
            return Evaluation(self.config.algorithm, tuple(returns), tuple(durations), True, True)
        finally:
            (ev if ev is not None else raw).close()

    def save(self, destination: Path, boundary: Boundary, *, attempt_id: str) -> Path:
        self._gates("L3-020", "L3-021", "L1-013")
        if self.model is None or self.vec is None:
            raise ContractError("No learner to save")
        assert_transition_proof(self.vec, self.config.fingerprint.hash)
        assert_resume_proof(self.vec, boundary.env_tokens)
        return checkpoint_save(self.model, self.vec, self.config, destination, boundary,
                               attempt_id=attempt_id)

    def load(self, source: Path, *, trusted_source: bool, restored_boundary: Boundary) -> RLProvider:
        self._gates("L3-020", "L3-021", "L1-013")
        if self.model is not None:
            raise ContractError("Must load into a fresh provider")
        self._runtime()
        if self.config.cuda_allocator_fraction is not None:
            import torch
            self._configure_cuda_allocator(torch)
        base = self._new_vec("train")
        try:
            # Rebuild L3's native episode boundary before reading its token or
            # observation. Spawn VecEnvs are lazy and do not reset at creation;
            # L3 uses the restored replay recipe supplied by env_factory here.
            base.reset()
            assert_transition_proof(base, self.config.fingerprint.hash)
            assert_resume_proof(base, restored_boundary.env_tokens)
            self.model, self.vec = checkpoint_load(source, self.config, base,
                                                   trusted_source=trusted_source,
                                                   expected_boundary=restored_boundary)
            self._restored = True
        except BaseException:
            self._tainted = True
            base.close()
            raise
        return self

    def mapping(self) -> dict:
        self._safe()
        if self.model is None:
            raise ContractError("No policy initialized")
        return actor_mapping(self.model, self.config)

    def transfer(self, weights: Mapping[str, np.ndarray], actor_signature: str,
                 policy_observations: np.ndarray, reference_actions: np.ndarray,
                 *, tolerance: float = 1e-5,
                 source_actor_signature: str | None = None) -> ParityReport:
        self._gates("L5-003")
        if self.model is None:
            raise ContractError("No actor destination initialized")
        return transfer_actor(self.model, self.config, weights, actor_signature,
                              policy_observations, reference_actions, tolerance=tolerance,
                              source_actor_signature=source_actor_signature)

    def distill(self, policy_observations: np.ndarray, reference_actions: np.ndarray,
                *, epochs: int, batch_size: int, learning_rate: float,
                tolerance: float = 1e-3, early_stop: bool = False) -> ParityReport:
        self._gates("L5-003")
        if self.model is None:
            raise ContractError("No actor destination initialized")
        return behavioral_clone(self.model, self.config, policy_observations,
                                reference_actions, epochs=epochs, batch_size=batch_size,
                                learning_rate=learning_rate, tolerance=tolerance, early_stop=early_stop)

    def parity(self, policy_observations: np.ndarray, reference_actions: np.ndarray,
               *, tolerance: float = 1e-5) -> ParityReport:
        self._safe()
        if self.model is None:
            raise ContractError("No actor initialized")
        return action_parity(self.model, self.config, policy_observations,
                             reference_actions, tolerance=tolerance)

    def export_inference(self, path: Path, fixed_raw_observations: np.ndarray) -> Path:
        self._safe()
        if self.model is None or self.vec is None:
            raise ContractError("No learner initialized")
        assert_transition_proof(self.vec, self.config.fingerprint.hash)
        return export_l0(self.model, self.config, self.vec, path, fixed_raw_observations)

    def close(self):
        if self.vec is not None:
            self.vec.close()
        self.vec = None
        self.model = None

    @property
    def performance(self) -> Performance | None:
        return self._perf


def _create_rl(settings: Mapping[str, object]) -> RLProvider:
    required = {"config", "env_factory", "gate_authorizer"}
    if set(settings) != required:
        raise ValueError("RL provider settings must contain config, env_factory, gate_authorizer")
    config = settings["config"]
    env_factory = settings["env_factory"]
    gate_authorizer = settings["gate_authorizer"]
    if not isinstance(config, RLConfig) or not callable(env_factory) or not callable(gate_authorizer):
        raise TypeError("invalid typed dependencies for the SB3 provider")
    return RLProvider(config, env_factory, gate_authorizer)


provider = ProviderDescriptor(kind="rl", contract_version="0.4.0", factory=_create_rl)
