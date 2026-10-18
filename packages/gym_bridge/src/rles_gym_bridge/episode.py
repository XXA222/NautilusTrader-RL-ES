"""L3 episode coordination shared by independent RL and ES callers.

Native execution, L2 features, and risk/planning remain injected. This module
owns clocks, action ordering, terminal drain and deterministic journal replay;
it neither matches orders nor computes an alternative account balance.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from decimal import Decimal
import re
from types import MappingProxyType, SimpleNamespace
from typing import Any

from rles_contracts.dto import ExecutionPlan
from rles_contracts.serialization import canonical_hash

from .observation import VisibleFeature
from .reward import RewardCalculator, RewardConfig, RewardState


@dataclass(frozen=True, slots=True)
class BarrierFrame:
    timestamp_ns: int
    batches: tuple[Any, ...]
    market: Mapping[str, VisibleFeature]
    time_features: Mapping[str, float]

    def __post_init__(self) -> None:
        if (type(self.timestamp_ns) is not int or self.timestamp_ns <= 0 or
                type(self.batches) is not tuple or not self.batches or
                self.batches[-1].timestamp_ns != self.timestamp_ns):
            raise ValueError("complete decision frame required")
        for feature in self.market.values():
            if not isinstance(feature, VisibleFeature) or feature.available_at_ns > self.timestamp_ns:
                raise ValueError("frame contains unavailable market features")
        object.__setattr__(self, "market", MappingProxyType(dict(self.market)))
        object.__setattr__(self, "time_features", MappingProxyType(dict(self.time_features)))


@dataclass(frozen=True, slots=True)
class EpisodeReplay:
    """Persist alongside a learner checkpoint; contains no native objects."""
    transition_hash: str
    actions: tuple[Decimal, ...]
    boundary_token: str
    episode_index: int | None = None
    episode_start_offset: int | None = None

    def __post_init__(self) -> None:
        if (re.fullmatch(r"[0-9a-f]{64}", self.transition_hash) is None
                or re.fullmatch(r"[0-9a-f]{64}", self.boundary_token) is None
                or type(self.actions) is not tuple
                or any(type(action) is not Decimal or not action.is_finite()
                       or not -1 <= action <= 1 for action in self.actions)
                or (self.episode_index is None) != (self.episode_start_offset is None)
                or (self.episode_index is not None and (
                    type(self.episode_index) is not int or self.episode_index < 0
                    or type(self.episode_start_offset) is not int
                    or self.episode_start_offset < 0))):
            raise ValueError("invalid immutable episode replay")

    def to_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "schema": "rles.episode-replay.v1", "transition_hash": self.transition_hash,
            "actions": [str(action) for action in self.actions],
            "boundary_token": self.boundary_token,
        }
        if self.episode_index is not None:
            value.update({"schema": "rles.episode-replay.v2",
                          "episode_index": self.episode_index,
                          "episode_start_offset": self.episode_start_offset})
        return value

    @classmethod
    def from_dict(cls, value: Any) -> EpisodeReplay:
        if not isinstance(value, dict) or not isinstance(value.get("actions"), list):
            raise ValueError("episode replay snapshot schema mismatch")
        schema = value.get("schema")
        v1 = {"schema", "transition_hash", "actions", "boundary_token"}
        v2 = v1 | {"episode_index", "episode_start_offset"}
        if (schema == "rles.episode-replay.v1" and set(value) != v1) or (
            schema == "rles.episode-replay.v2" and set(value) != v2
        ) or schema not in {"rles.episode-replay.v1", "rles.episode-replay.v2"}:
            raise ValueError("episode replay snapshot schema mismatch")
        try:
            actions = tuple(Decimal(item) for item in value["actions"]
                            if isinstance(item, str))
        except Exception as exc:
            raise ValueError("invalid serialized replay action") from exc
        if len(actions) != len(value["actions"]):
            raise ValueError("serialized replay actions must be decimal strings")
        return cls(value["transition_hash"], actions, value["boundary_token"],
                   value.get("episode_index"), value.get("episode_start_offset"))


@dataclass(frozen=True, slots=True)
class EpisodeResumeState:
    reward: RewardState
    step_count: int
    guard_intervention: float


@dataclass(frozen=True, slots=True)
class RiskActionReview:
    """Risk adjustments for one target, retaining soft and hard contributions."""
    plan: ExecutionPlan
    total_intervention: Decimal
    soft_adjustment: Decimal
    hard_adjustment: Decimal
    components_known: bool
    soft_target: Decimal | None = None
    risk_target: Decimal | None = None
    rounded_target_exposure: Decimal | None = None

    def __post_init__(self) -> None:
        if type(self.plan) is not ExecutionPlan or type(self.components_known) is not bool:
            raise TypeError("risk action review requires a native execution plan and known-component flag")
        values = (self.total_intervention, self.soft_adjustment, self.hard_adjustment)
        if any(type(value) is not Decimal or not value.is_finite() or not 0 <= value <= 2 for value in values):
            raise ValueError("risk action adjustment fractions must be finite and within [0,2]")
        if self.components_known and self.total_intervention + Decimal("1e-24") < max(
            self.soft_adjustment, self.hard_adjustment,
        ):
            raise ValueError("total action adjustment cannot be smaller than a component")
        if (self.soft_target is None) != (self.risk_target is None):
            raise ValueError("soft and final risk targets must be recorded together")
        for target in (self.soft_target, self.risk_target):
            if target is not None and (type(target) is not Decimal or not target.is_finite() or not -1 <= target <= 1):
                raise ValueError("normalized risk targets must be finite Decimals in [-1,1]")
        if (self.rounded_target_exposure is not None
                and (type(self.rounded_target_exposure) is not Decimal
                     or not self.rounded_target_exposure.is_finite())):
            raise ValueError("quantity-rounded target exposure must be a finite Decimal")


class BarrierEpisode:
    def __init__(self, *, native_session: Any, native_identity: Mapping[str, str],
                 transition_hash: str, frames: tuple[BarrierFrame, ...], scoring_frames: int,
                 plan_action: Callable[[Decimal, Any, Decimal], tuple[ExecutionPlan, Decimal] | RiskActionReview],
                 plan_terminal: Callable[[Any], ExecutionPlan],
                 executed_exposure: Callable[[Any], Decimal], reward_config: RewardConfig,
                 replay: EpisodeReplay | None = None,
                 episode_index: int = 0, episode_start_offset: int = 0,
                 observation_state: Callable[[Any], Mapping[str, Mapping[str, float]]] | None = None) -> None:
        if re.fullmatch(r"[0-9a-f]{64}", transition_hash) is None:
            raise ValueError("complete transition fingerprint required")
        if set(native_identity) != {"dataset_hash", "fold_hash", "config_hash"}:
            raise ValueError("frozen native episode identity required")
        if (type(frames) is not tuple or type(scoring_frames) is not int or
                scoring_frames < 2 or len(frames) < scoring_frames + 2 or
                any(type(f) is not BarrierFrame for f in frames) or
                any(b.timestamp_ns <= a.timestamp_ns for a, b in zip(frames, frames[1:]))):
            raise ValueError("ordered scoring frames plus two terminal drain frames required")
        if not all(callable(f) for f in (plan_action, plan_terminal, executed_exposure)):
            raise TypeError("risk planner, terminal planner and native exposure reader required")
        if replay is not None and (type(replay) is not EpisodeReplay or
                replay.transition_hash != transition_hash or type(replay.actions) is not tuple or
                len(replay.actions) >= scoring_frames - 1 or
                any(type(a) is not Decimal or not a.is_finite() or not -1 <= a <= 1 for a in replay.actions)):
            raise ValueError("replay must describe a live boundary of the same transition recipe")
        if (type(episode_index) is not int or episode_index < 0
                or type(episode_start_offset) is not int or episode_start_offset < 0
                or replay is not None and replay.episode_index is not None
                and (replay.episode_index != episode_index
                     or replay.episode_start_offset != episode_start_offset)):
            raise ValueError("episode scheduling identity does not match replay")
        self._session = native_session
        self._identity = dict(native_identity)
        self._transition = transition_hash
        self._frames, self._scoring_frames = frames, scoring_frames
        self._plan, self._terminal_plan, self._exposure = plan_action, plan_terminal, executed_exposure
        self._reward = RewardCalculator(reward_config)
        self._replay = replay
        self._episode_index = episode_index
        self._episode_start_offset = episode_start_offset
        self._observation_state = observation_state
        self._snapshot: Any = None
        self._index = 0
        self._actions: list[Decimal] = []
        self._pending: tuple[ExecutionPlan, Decimal, RiskActionReview] | None = None
        self._executed = Decimal(0)
        self._last_guard = Decimal(0)
        self._state = "CREATED"

    @property
    def reward_config(self) -> RewardConfig:
        return self._reward.config

    @property
    def previous_decision_ts_ns(self) -> int:
        if self._snapshot is None:
            raise RuntimeError("episode not bootstrapped")
        return self._snapshot.decision_ts_ns

    @property
    def episode_window(self) -> dict[str, int]:
        """Return the exact causal decision segment selected for this episode."""
        return {
            "start_ns": self._frames[0].timestamp_ns,
            "end_ns": self._frames[self._scoring_frames - 1].timestamp_ns,
            "decision_steps": self._scoring_frames - 1,
            "episode_index": self._episode_index,
            "start_offset": self._episode_start_offset,
        }

    def transition_fingerprint_hash(self) -> str:
        return self._transition

    def _financial_state(self) -> dict:
        snapshot = self._snapshot
        # Read native ledger facts; omit random event UUIDs and nested event
        # histories. Replaying from the frozen recipe restores native state.
        return {
            "clock": snapshot.decision_ts_ns, "equity": snapshot.equity,
            "currency": snapshot.currency,
            "positions": sorted((
                {k: row.get(k) for k in ("instrument_id", "account_id", "side", "quantity", "realized_pnl", "avg_px_open")}
                for row in snapshot.positions), key=canonical_hash),
            "orders": sorted((
                {k: row.get(k) for k in ("client_order_id", "instrument_id", "side", "quantity", "filled_qty", "status", "price")}
                for row in snapshot.orders), key=canonical_hash),
        }

    def checkpoint_boundary_token(self) -> str:
        if (self._state != "RUNNING" or self._pending is not None or
                self._index >= self._scoring_frames - 1 or self._snapshot.equity <= 0):
            raise RuntimeError("checkpoint requires a completed live decision barrier")
        reward = self._reward.export_state()
        return canonical_hash({
            "transition": self._transition, "identity": self._identity,
            "actions": tuple(self._actions), "financial_state": self._financial_state(),
            "reward": (reward.previous_equity, reward.peak_equity, reward.drawdown),
            "executed": self._executed, "guard": self._last_guard,
        })

    def export_replay(self) -> EpisodeReplay:
        return EpisodeReplay(self._transition, tuple(self._actions), self.checkpoint_boundary_token(),
                             self._episode_index, self._episode_start_offset)

    def restored_env_state(self) -> EpisodeResumeState | None:
        if self._replay is None:
            return None
        return EpisodeResumeState(self._reward.export_state(), len(self._actions), float(self._last_guard))

    def bootstrap(self) -> Any:
        if self._state != "CREATED":
            raise RuntimeError("single bootstrap per episode")
        try:
            self._session.open(SimpleNamespace(**self._identity))
            first = self._frames[0]
            self._snapshot = self._session.bootstrap(first.batches, decision_ts_ns=first.timestamp_ns)
            self._reward.reset(float(self._snapshot.equity))
            self._executed = self._read_exposure()
            self._state = "RUNNING"
            if self._replay is not None:
                for action in self._replay.actions:
                    plan, _guard = self.action_to_plan(action)
                    self.advance(plan)
                if self.checkpoint_boundary_token() != self._replay.boundary_token:
                    raise RuntimeError("native journal replay did not restore checkpoint boundary")
            return self._snapshot
        except BaseException:
            self.close()
            raise

    def _read_exposure(self) -> Decimal:
        value = self._exposure(self._snapshot)
        if type(value) is not Decimal or not value.is_finite():
            raise ValueError("executed exposure must come from finite native facts")
        return value

    def action_to_plan(self, raw: float | Decimal) -> tuple[ExecutionPlan, Decimal]:
        if self._state != "RUNNING" or self._pending is not None or self._index >= self._scoring_frames - 1:
            raise RuntimeError("action requires a live unused scoring barrier")
        action = Decimal(str(raw))
        if not action.is_finite() or not -1 <= action <= 1:
            raise ValueError("normalized finite policy action required")
        result = self._plan(action, self._snapshot, Decimal(str(self._reward.drawdown)))
        if type(result) is RiskActionReview:
            review = result
        elif (type(result) is tuple and len(result) == 2 and type(result[0]) is ExecutionPlan
              and type(result[1]) is Decimal and result[1].is_finite() and 0 <= result[1] <= 2):
            # Legacy injected planners retain reward compatibility, but cannot
            # claim hard-risk attribution for candidate selection.
            review = RiskActionReview(result[0], result[1], result[1], Decimal(0), False)
        else:
            raise ValueError("risk pipeline must return an L1 plan and intervention fraction")
        plan, guard = review.plan, review.total_intervention
        self._pending = plan, action, review
        return plan, guard

    @property
    def pending_risk_review(self) -> RiskActionReview:
        if self._pending is None:
            raise RuntimeError("risk review is available only after planning and before advance")
        return self._pending[2]

    def advance(self, plan: ExecutionPlan) -> Any:
        if self._state != "RUNNING" or self._pending is None or plan is not self._pending[0]:
            raise RuntimeError("advance must consume the exact risk-reviewed action plan")
        try:
            _plan, action, review = self._pending
            guard = review.total_intervention
            next_frame = self._frames[self._index + 1]
            self._snapshot = self._session.advance(plan, batches=next_frame.batches,
                                                   next_decision_ts_ns=next_frame.timestamp_ns)
            self._index += 1
            self._executed = self._read_exposure()
            self._actions.append(action)
            self._last_guard = guard
            self._reward.calculate(native_equity=float(self._snapshot.equity), guard_intervention=float(guard))
            self._pending = None
            return SimpleNamespace(
                snapshot=self._snapshot, decision_ts_ns=next_frame.timestamp_ns,
                executed_action=float(self._executed),
                evidence_hash=canonical_hash({"action": plan, "facts": self._financial_state()}),
                terminated=self._snapshot.equity <= 0,
                truncated=self._index == self._scoring_frames - 1,
            )
        except BaseException:
            self.close()
            raise

    def observation_inputs(self, *, snapshot: Any, reward_state: RewardCalculator,
                           guard_intervention: float) -> dict:
        if snapshot is not self._snapshot:
            raise ValueError("observation must use the completed native snapshot")
        frame = self._frames[self._index]
        inputs = dict(
            decision_ts_ns=snapshot.decision_ts_ns, market=frame.market,
            account={"equity": float(snapshot.equity), "peak_equity": reward_state.peak_equity,
                     "drawdown": reward_state.drawdown},
            risk={"guard_intervention": guard_intervention},
            execution={"prev_executed_target": float(self._executed)}, time=frame.time_features,
        )
        if self._observation_state is not None:
            for group, values in self._observation_state(snapshot).items():
                if group not in {"account", "risk", "execution"}:
                    raise ValueError("native observation state has unknown group")
                if set(values) & set(inputs[group]):
                    raise ValueError("native observation state cannot replace reward/barrier memory")
                inputs[group].update(values)
            inputs["execution"]["prev_raw_target"] = float(self._actions[-1]) if self._actions else 0.0
        return inputs

    def drain_and_close(self, policy: str) -> Any:
        if self._state != "RUNNING" or self._pending is not None:
            raise RuntimeError("terminal drain requires a settled decision barrier")
        try:
            kwargs: dict[str, Any] = {}
            drain_index = self._index + 1
            if self._snapshot.orders:
                cancel_frame = self._frames[drain_index]
                kwargs.update(cancel_plan=self._terminal_plan(self._snapshot),
                              cancel_batches=cancel_frame.batches, cancel_ts_ns=cancel_frame.timestamp_ns)
                drain_index += 1
            frame = self._frames[drain_index]
            self._snapshot = self._session.drain_and_close(
                policy, flatten_plan=self._terminal_plan, drain_batches=frame.batches,
                terminal_ts_ns=frame.timestamp_ns, **kwargs,
            )
            self._index = drain_index
            self._executed = self._read_exposure()
            self._state = "SETTLED"
            return SimpleNamespace(
                snapshot=self._snapshot, unsettled_orders=bool(self._snapshot.orders),
                open_positions=bool(self._snapshot.positions),
                evidence_hash=canonical_hash(self._financial_state()),
            )
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        try:
            self._session.close()
        finally:
            self._state = "CLOSED"
