"""Real historical perpetual environment assembly, with no test imports."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict
from decimal import Decimal
from hashlib import sha256
from bisect import bisect_left, bisect_right
import json
from math import gcd
from pathlib import Path
import re
from types import SimpleNamespace
from typing import Any

from rles_contracts.dto import DataQualityGrade, ExecutionPlan
from rles_data.contracts import InstrumentContractSpec, to_market_event
from rles_data.core import Availability, CuratedStore, DataError, Event, FeatureField, FeatureSpec, KINDS, MINUTE_NS, Window, digest
from rles_data.coverage import CoveragePolicy
from rles_data.features import Scaler, fit_train, make_feature_row
from rles_data.io import read_curated
from rles_execution import InstrumentRules, PendingOrder, ResolvedExecutionConfig, TargetQuantityPlanner, to_contract_plan
from rles_nautilus_adapter.perpetual import QUOTE_MAX_AGE_NS, make_perpetual_session
from rles_nautilus_adapter.session import VisibleBatch
from rles_risk import HardRiskConfig, HardRiskGuard, SoftRiskParams, SoftRiskTransform

from .episode import BarrierEpisode, BarrierFrame, EpisodeReplay, RiskActionReview, SessionRebuildInterrupted
from .env import NautilusBarrierEnv
from .l2_bridge import L2FrozenScalerAdapter, visible_market_features
from .observation import ObservationBuilder, ObservationSchema
from .reward import RewardConfig


_FUNDING_TIME_HORIZON_NS = 24 * 60 * 60 * 1_000_000_000


class _NativePositionClock:
    """Track open-exposure and latest-fill clocks from native fill events only."""

    def __init__(self) -> None:
        self._quantity = Decimal(0)
        self._opened_at_ns: int | None = None
        self._last_fill_at_ns: int | None = None
        self._fills: dict[str, dict] = {}

    @staticmethod
    def _timestamp_ns(value: Any, label: str) -> int:
        if type(value) is not int or value <= 0:
            raise DataError(f"native {label} timestamp must be a positive integer")
        return value

    def observe(self, snapshot: Any) -> dict[str, float]:
        fresh = []
        for fill in snapshot.fills:
            event_id = fill.get("event_id")
            if not isinstance(event_id, str) or not event_id:
                raise DataError("native fill lacks a stable identity for causal clocks")
            previous = self._fills.get(event_id)
            if previous is not None:
                if previous != fill:
                    raise DataError("native fill identity has conflicting facts")
                continue
            fresh.append(fill)
        fresh.sort(key=lambda row: (self._timestamp_ns(row.get("ts_event"), "fill"),
                                    str(row["event_id"])))
        for fill in fresh:
            side = fill.get("order_side")
            try:
                quantity = Decimal(str(fill["last_qty"]))
                timestamp_ns = self._timestamp_ns(fill.get("ts_event"), "fill")
            except (KeyError, ValueError, TypeError, ArithmeticError) as exc:
                raise DataError("native fill lacks valid quantity or event time") from exc
            if (side not in {"BUY", "SELL"} or not quantity.is_finite() or quantity <= 0
                    or timestamp_ns <= 0 or timestamp_ns > snapshot.decision_ts_ns):
                raise DataError("native fill facts are invalid for causal clocks")
            signed_fill = quantity if side == "BUY" else -quantity
            updated = self._quantity + signed_fill
            if self._quantity == 0 and updated != 0:
                self._opened_at_ns = timestamp_ns
            elif updated == 0:
                self._opened_at_ns = None
            elif (self._quantity > 0 > updated) or (self._quantity < 0 < updated):
                self._opened_at_ns = timestamp_ns
            self._quantity = updated
            self._last_fill_at_ns = timestamp_ns
            self._fills[str(fill["event_id"])] = dict(fill)

        pending_ages = []
        pending_ids: set[str] = set()
        for order in snapshot.orders:
            client_id = order.get("client_order_id")
            try:
                submitted_at_ns = self._timestamp_ns(order.get("ts_init"), "pending order")
            except (KeyError, ValueError, TypeError, ArithmeticError) as exc:
                raise DataError("native pending order lacks a valid creation timestamp") from exc
            if (not isinstance(client_id, str) or not client_id or client_id in pending_ids
                    or submitted_at_ns <= 0 or submitted_at_ns > snapshot.decision_ts_ns):
                raise DataError("native pending order identity/time is invalid")
            pending_ids.add(client_id)
            pending_ages.append((snapshot.decision_ts_ns - submitted_at_ns) / MINUTE_NS)

        native_quantity = signed_quantity(snapshot)
        if native_quantity != self._quantity:
            raise DataError("native fill clock quantity does not reconcile to the position ledger")
        now = snapshot.decision_ts_ns
        has_position_age = self._opened_at_ns is not None
        has_fill_age = self._last_fill_at_ns is not None
        return {
            "position_age_minutes": ((now - self._opened_at_ns) / MINUTE_NS
                                     if has_position_age else 0.0),
            "has_position_age": float(has_position_age),
            "minutes_since_fill": ((now - self._last_fill_at_ns) / MINUTE_NS
                                   if has_fill_age else 0.0),
            "has_seen_fill": float(has_fill_age),
            "pending_order_age_minutes": max(pending_ages, default=0.0),
        }


def scheduled_funding_time_features(decision_ts_ns: int, settlement_times_ns: tuple[int, ...]) -> dict[str, float]:
    """Encode only the source-declared next funding time, never realized events."""
    if type(decision_ts_ns) is not int or decision_ts_ns <= 0:
        raise DataError("funding time feature requires a positive decision timestamp")
    if (not isinstance(settlement_times_ns, tuple)
            or any(type(ts) is not int or ts <= 0 for ts in settlement_times_ns)
            or settlement_times_ns != tuple(sorted(set(settlement_times_ns)))):
        raise DataError("funding time feature requires an ordered immutable schedule")
    future = next((ts for ts in settlement_times_ns if ts > decision_ts_ns), None)
    if future is None:
        return {"funding_settlement_within_24h": 0.0,
                "funding_settlement_time_to_event_24h": 1.0}
    fraction = min((future - decision_ts_ns) / _FUNDING_TIME_HORIZON_NS, 1.0)
    return {
        "funding_settlement_within_24h": float(future - decision_ts_ns <= _FUNDING_TIME_HORIZON_NS),
        "funding_settlement_time_to_event_24h": fraction,
    }


def _money(value: Any) -> Decimal:
    # Native Money strings carry a currency suffix; account balance dicts may
    # instead carry bare decimal strings. No market-derived balance estimates.
    result = Decimal(str(value).replace(",", "").split()[0])
    if not result.is_finite():
        raise DataError("nonfinite native account amount")
    return result


def _native_unrealized_pnl_pct(snapshot: Any, *, settlement_currency: str) -> float:
    """Compute aggregate native position PnL over native position notional."""
    if not snapshot.positions:
        return 0.0
    unrealized = Decimal(0)
    notional = Decimal(0)
    for position in snapshot.positions:
        if (position.get("unrealized_pnl_currency") != settlement_currency
                or position.get("notional_currency") != settlement_currency):
            raise DataError("native position PnL and notional currencies do not match settlement")
        unrealized += _money(position.get("unrealized_pnl"))
        position_notional = _money(position.get("notional_value"))
        if position_notional <= 0:
            raise DataError("native position notional must be positive")
        notional += position_notional
    if not notional.is_finite() or notional <= 0 or not unrealized.is_finite():
        raise DataError("native position PnL percentage denominator is invalid")
    return float(unrealized / notional)


def signed_quantity(snapshot) -> Decimal:
    return sum((Decimal(str(row["quantity"])) * (-1 if row["side"] == "SHORT" else 1)
                for row in snapshot.positions), Decimal(0))


def native_balances(snapshot) -> tuple[Decimal, Decimal]:
    balances = snapshot.account_events[-1]["balances"]
    matches = [b for b in balances if b["currency"] == snapshot.currency]
    if len(matches) != 1:
        raise DataError("exactly one settlement-currency native balance required")
    return _money(matches[0]["total"]), _money(matches[0]["free"])


def load_rules(path: str | Path) -> tuple[Event, ...]:
    """Require source bytes and explicit historical availability/effective dates."""
    source = Path(path).resolve()
    if not source.is_file():
        raise DataError(f"historical instrument rules unavailable: {source}")
    document = json.loads(source.read_text(encoding="utf-8"))
    if set(document) != {"sources", "events"} or not document["sources"] or not document["events"]:
        raise DataError("rules manifest requires source documents and normalized events")
    verified = set()
    for evidence in document["sources"]:
        p = Path(evidence["path"])
        p = p if p.is_absolute() else source.parent / p
        actual = sha256(p.read_bytes()).hexdigest()
        if actual != evidence["sha256"] or not evidence.get("provenance"):
            raise DataError("historical rule source integrity/provenance missing")
        verified.add(actual)
    events = []
    for row in document["events"]:
        event = Event(**{**row, "availability_source": Availability(row["availability_source"])})
        if event.kind != "instrument_rule" or event.source_hash not in verified:
            raise DataError("rule event is not bound to a verified historical source")
        events.append(event)
    return tuple(events)


def historical_feature_spec(maximum_age_ns: dict[str, int], *, profile: str = "legacy",
                            base_currency: str) -> FeatureSpec:
    """Return a frozen causal feature recipe selected by the run configuration.

    ``legacy`` preserves the original observation identity for existing bundles.
    ``causal_mtf_v1`` adds closed-bar return and volatility context at longer
    horizons. ``causal_mtf_v2`` also adds bounded trend-efficiency features that
    distinguish persistent directional movement from choppy price paths. The
    ``causal_mtf_v3`` adds a source-scheduled next-settlement countdown.
    ``causal_mtf_v4`` adds native-fill-derived position and fill clocks while
    retaining the scheduled funding context. ``causal_mtf_v5`` adds a complete
    UTC-closed daily return and a five-minute realized-volatility input. Each
    recipe binds its own identity.
    """
    if set(maximum_age_ns) != {"mark", "index", "funding_update"}:
        raise DataError("historical feature recipe requires complete freshness limits")
    if not isinstance(profile, str) or profile not in {
        "legacy", "causal_mtf_v1", "causal_mtf_v2", "causal_mtf_v3", "causal_mtf_v4",
        "causal_mtf_v5",
    }:
        raise DataError("unsupported historical feature profile")
    if not isinstance(base_currency, str) or re.fullmatch(r"[A-Z][A-Z0-9]{0,11}", base_currency) is None:
        raise DataError("feature base currency must be explicit")
    fields: tuple[FeatureField, ...] = (
        FeatureField("return_1m", "float64", "ratio", "ohlcv_1m", 1, 1, "return"),
        FeatureField("volatility_5m", "float64", "ratio", "ohlcv_1m", 5, 1, "volatility"),
        FeatureField("volume_5m", "float64", base_currency, "ohlcv_1m", 5, 1, "volume_mean"),
        FeatureField("basis", "float64", "ratio", "mark", 1, 1, "basis", maximum_age_ns["mark"]),
        FeatureField("last_settled_funding", "float64", "ratio", "funding_update", 1, 1,
                     "latest", maximum_age_ns["funding_update"]),
    )
    if profile == "causal_mtf_v5":
        fields += (
            FeatureField("volatility_1m", "float64", "ratio", "ohlcv_1m", 60, 1,
                         "volatility"),
            FeatureField("closed_return_1h", "float64", "ratio", "ohlcv_1m", 1, 60, "return"),
            FeatureField("closed_return_1d", "float64", "ratio", "ohlcv_1m", 1, 1440, "return"),
            FeatureField("closed_trend_efficiency_1h", "float64", "ratio", "ohlcv_1m", 1, 60,
                         "trend_efficiency"),
            FeatureField("closed_trend_efficiency_1d", "float64", "ratio", "ohlcv_1m", 1, 1440,
                         "trend_efficiency"),
        )
    if profile in {"causal_mtf_v1", "causal_mtf_v2", "causal_mtf_v3", "causal_mtf_v4",
                   "causal_mtf_v5"}:
        fields += (
            FeatureField("return_5m", "float64", "ratio", "ohlcv_1m", 5, 1, "return"),
            FeatureField("return_15m", "float64", "ratio", "ohlcv_1m", 15, 1, "return"),
            FeatureField("return_60m", "float64", "ratio", "ohlcv_1m", 60, 1, "return"),
            FeatureField("volatility_15m", "float64", "ratio", "ohlcv_1m", 15, 1, "volatility"),
            FeatureField("volatility_60m", "float64", "ratio", "ohlcv_1m", 60, 1, "volatility"),
        )
    if profile in {"causal_mtf_v2", "causal_mtf_v3", "causal_mtf_v4", "causal_mtf_v5"}:
        fields += (
            FeatureField("trend_efficiency_15m", "float64", "ratio", "ohlcv_1m", 15, 1, "trend_efficiency"),
            FeatureField("trend_efficiency_60m", "float64", "ratio", "ohlcv_1m", 60, 1, "trend_efficiency"),
        )
    return FeatureSpec("0.4.0", fields)


def _visible_batches_by_decision(batches: tuple[tuple[int, Any], ...],
                                 decision_times: tuple[int, ...]) -> tuple[tuple[Any, ...], ...]:
    """Partition ordered market batches once instead of rescanning history per row."""
    result = []
    offset = 0
    for decision_ns in decision_times:
        first = offset
        while offset < len(batches) and batches[offset][0] <= decision_ns:
            offset += 1
        result.append(tuple(batch for _, batch in batches[first:offset]))
    return tuple(result)


def partition_decision_lanes(full_steps: int, lane_count: int) -> tuple[tuple[int, int], ...]:
    """Partition decision offsets into contiguous, rank-exclusive half-open lanes."""
    if (type(full_steps) is not int or type(lane_count) is not int
            or full_steps < 1 or not 1 <= lane_count <= full_steps):
        raise DataError("invalid historical decision lane partition")
    return tuple(
        (rank * full_steps // lane_count, (rank + 1) * full_steps // lane_count)
        for rank in range(lane_count)
    )


def _episode_lane(*, full_steps: int, episode_rank: int,
                  episode_count: int, episode_steps: int) -> tuple[int, int]:
    """Return a rank-exclusive decision lane for deterministic episode rotation."""
    if (type(full_steps) is not int or type(episode_rank) is not int
            or type(episode_count) is not int or type(episode_steps) is not int
            or full_steps < 1 or not 0 <= episode_rank < episode_count
            or episode_steps < 1):
        raise DataError("invalid rotating historical episode partition")
    start, end = partition_decision_lanes(full_steps, episode_count)[episode_rank]
    if end - start < episode_steps:
        raise DataError("historical episode must fit inside every rank-exclusive time lane")
    return start, end


def _episode_start(*, lane_start: int, lane_end: int,
                   episode_steps: int, episode_index: int) -> int:
    """Walk every legal lane start exactly once before the sequence repeats."""
    if type(episode_index) is not int or episode_index < 0:
        raise DataError("historical episode index must be a non-negative integer")
    positions = lane_end - lane_start - episode_steps + 1
    if positions <= 0:
        raise DataError("historical episode has no legal start position in its lane")
    if positions == 1:
        return lane_start
    stride = max(1, int(positions * 0.6180339887498948))
    while gcd(stride, positions) != 1:
        stride += 1
        if stride >= positions:
            stride = 1
    return lane_start + episode_index * stride % positions


class HistoricalEnvironmentFactory:
    """Freeze data, rules and training-only scaler; create a fresh Engine per Env."""
    def __init__(self, *, market_path: str | Path, rules_path: str | Path,
                 instrument: str, start_ns: int, end_ns: int, coverage_policy: dict,
                 execution: dict, hard_risk: dict, soft_risk: dict,
                 starting_balance: str, risk_budget_notional: str,
                 mode: str = "base", scaler: Scaler | None = None,
                 fit_scaler: bool = False, split: str = "train_inner",
                 compact_native_history: bool = True,
                 compact_report_snapshots: bool = False,
                 feature_profile: str = "legacy", reward_scale: float = 1.0,
                 reward_lambda_dd: float = 0.0, reward_lambda_guard: float = 0.0,
                 base_currency: str | None = None,
                 prepared_store: CuratedStore | None = None) -> None:
        # Set only by the application layer after prepare() computes its
        # frozen root-config recipe identity.
        self.application_training_recipe_hash: str | None = None
        if split not in {"train_inner", "validation", "test", "holdout"}:
            raise DataError("unknown historical research split")
        if fit_scaler and split != "train_inner":
            raise DataError("only Train-inner may fit feature scaling")
        if not fit_scaler and scaler is None:
            raise DataError("evaluation must reuse the frozen training feature scaler")
        if type(compact_native_history) is not bool:
            raise DataError("native history compaction flag must be Boolean")
        if type(compact_report_snapshots) is not bool:
            raise DataError("report snapshot compaction flag must be Boolean")
        if not isinstance(feature_profile, str) or feature_profile not in {
            "legacy", "causal_mtf_v1", "causal_mtf_v2", "causal_mtf_v3", "causal_mtf_v4",
            "causal_mtf_v5",
        }:
            raise DataError("unsupported historical feature profile")
        self.compact_native_history = compact_native_history
        self.compact_report_snapshots = compact_report_snapshots
        self.feature_profile = feature_profile
        self.reward_config = RewardConfig(
            lambda_dd=reward_lambda_dd, lambda_guard=reward_lambda_guard,
            reward_scale=reward_scale,
        )
        if prepared_store is not None and not isinstance(prepared_store, CuratedStore):
            raise DataError("prepared history must be a validated CuratedStore")
        self.store = (prepared_store if prepared_store is not None else
                      CuratedStore((*read_curated(market_path).events, *load_rules(rules_path))))
        self.instrument, self.split, self.mode = instrument, split, mode
        self.start_ns, self.end_ns = start_ns, end_ns
        self.coverage = CoveragePolicy(**coverage_policy)
        self.execution = ResolvedExecutionConfig.from_dict(execution)
        required_kinds = (tuple(sorted(KINDS)) if self.execution.base_fill.data_granularity == "quote" else None)
        self.quality = self.store.quality(start_ns=start_ns, end_ns=end_ns,
                                         expected_instruments=(instrument,), coverage_policy=self.coverage,
                                         **({"required": required_kinds} if required_kinds is not None else {}))
        self.store.require_usable(self.quality, "rl_es")
        hard_values: dict[str, Any] = {k: v if type(v) is bool else Decimal(v) for k, v in hard_risk.items()}
        self.hard = HardRiskConfig(**hard_values)
        self.soft = SoftRiskParams.from_search_dict(soft_risk)
        self.capital, self.budget = Decimal(starting_balance), Decimal(risk_budget_notional)
        if not self.capital.is_finite() or self.capital <= 0 or not self.budget.is_finite() or self.budget <= 0:
            raise DataError("positive finite capital and risk budget required")
        if self.hard.max_leverage > 1 and self.quality.grade.value not in {"DQ-A", "DQ-B"}:
            raise DataError("leverage above 1x requires DQ-A/B market and liquidation evidence")
        self.rule = self.store.latest_rule(start_ns, instrument)
        self.contract_spec = InstrumentContractSpec.from_rule_payload(self.rule.payload, instrument_id=instrument)
        fill_cost = self.execution.fill(self.mode)
        taker_fee = Decimal(self.rule.payload["taker_fee"])
        if self.mode == "stress" and taker_fee > 0:
            taker_fee *= fill_cost.fee_multiplier
        self.hard_execution_cost_rate = (
            max(Decimal(0), taker_fee) + max(Decimal(0), fill_cost.slippage_bps) / Decimal(10000)
        )
        initial_margin_rate = Decimal(self.rule.payload["initial_margin_rate"])
        maintenance_rate = Decimal(self.rule.payload["maintenance_margin_rate"])
        if (initial_margin_rate <= 0
                or initial_margin_rate / self.hard.max_leverage < maintenance_rate):
            raise DataError("configured leverage would reduce initial margin below dated maintenance margin")
        if (self.feature_profile in {"causal_mtf_v4", "causal_mtf_v5"}
                and (self.contract_spec.is_inverse
                     or self.contract_spec.settlement_currency != self.contract_spec.quote_currency)):
            raise DataError("causal_mtf_v4 PnL percentage requires a linear quote-settled contract")
        if base_currency is not None and base_currency != self.contract_spec.base_currency:
            raise DataError("configured base currency differs from dated contract metadata")
        if any(dict(self.store.latest_rule(t, instrument).payload) != dict(self.rule.payload)
               for t in (end_ns - 1,)):
            raise DataError("split episode at historical instrument-rule changes")
        self.features = historical_feature_spec(dict(self.coverage.maximum_age_ns), profile=self.feature_profile,
                                                base_currency=self.contract_spec.base_currency)
        account: tuple[str, ...] = (
            "equity", "peak_equity", "drawdown", "cash", "free_margin", "initial_margin",
            "maintenance_margin", "unrealized_pnl", "net_exposure", "position_quantity",
        )
        if self.feature_profile in {"causal_mtf_v4", "causal_mtf_v5"}:
            account += ("unrealized_pnl_pct",)
        execution_names = (
            "prev_executed_target", "prev_raw_target", "pending_order_count",
            "pending_signed_exposure", "pending_gross_exposure", "fill_rate",
        )
        names = tuple("market." + f.name for f in self.features.fields) + tuple("account." + n for n in account)
        quote, base = self.contract_spec.quote_currency, self.contract_spec.base_currency
        units = tuple(f.unit for f in self.features.fields) + (
            quote, quote, "ratio", quote, quote, quote, quote, quote, "ratio", base,
        )
        if self.feature_profile in {"causal_mtf_v4", "causal_mtf_v5"}:
            units += ("ratio",)
        time_names = (("time.funding_settlement_within_24h",
                       "time.funding_settlement_time_to_event_24h")
                      if self.feature_profile in {"causal_mtf_v3", "causal_mtf_v4", "causal_mtf_v5"} else ())
        if (self.feature_profile in {"causal_mtf_v3", "causal_mtf_v4", "causal_mtf_v5"}
                and self.coverage.known_schedule_times_ns is None):
            raise DataError(f"{self.feature_profile} requires a source-declared future funding calendar")
        execution_features = (("position_age_minutes", "has_position_age", "minutes_since_fill", "has_seen_fill",
                               "pending_order_age_minutes")
                              if self.feature_profile in {"causal_mtf_v4", "causal_mtf_v5"} else ())
        observation_version = ("0.6.0" if self.feature_profile == "causal_mtf_v5" else
                               "0.5.0" if self.feature_profile == "causal_mtf_v4" else "0.4.0")
        self.schema = ObservationSchema(observation_version, names + time_names + ("risk.guard_intervention", "risk.headroom")
                                        + tuple("execution." + n for n in execution_names + execution_features),
                                        units + ("ratio",) * len(time_names)
                                        + ("ratio", "ratio", "ratio", "ratio", "count", "ratio", "ratio", "ratio")
                                        + (("minutes", "boolean", "minutes", "boolean", "minutes")
                                           if execution_features else ()))
        # Fold membership uses economic event time. Availability still becomes
        # each row's decision clock, so delayed facts never become observable
        # early and cannot drift into a different score split merely due to
        # ingestion latency.
        bars = [e for e in self.store.events if e.kind == "ohlcv_1m" and e.payload["instrument"] == instrument
                and e.available_at_ns is not None and start_ns <= e.ts_event_ns < end_ns]
        if len(bars) < 5:
            raise DataError("at least five scoring/drain bars plus feature warmup required")
        self.rows = tuple(make_feature_row(self.store, self.features, int(e.available_at_ns or 0), instrument=instrument) for e in bars)
        if self.execution.base_fill.data_granularity == "quote":
            quotes = tuple(e for e in self.store.events if e.kind == "best_quote"
                           and e.payload["instrument"] == instrument and e.available_at_ns is not None)
            quote_cursor = 0
            latest_quote: Event | None = None
            for row in self.rows:
                while quote_cursor < len(quotes):
                    candidate = quotes[quote_cursor]
                    available_at_ns = candidate.available_at_ns
                    if available_at_ns is None or available_at_ns > row.decision_ts_ns:
                        break
                    latest_quote = candidate
                    quote_cursor += 1
                if (latest_quote is None or latest_quote.available_at_ns is None
                        or row.decision_ts_ns - latest_quote.available_at_ns > QUOTE_MAX_AGE_NS):
                    raise DataError("quote execution requires fresh best-quote coverage at every decision")
        selected_scaler = (fit_train(self.features, Window(start_ns, start_ns, end_ns),
                                     ((r.decision_ts_ns, r.values) for r in self.rows[:-2]))
                           if fit_scaler else scaler)
        assert selected_scaler is not None
        self.scaler: Scaler = selected_scaler
        if any(dict(self.store.latest_rule(row.decision_ts_ns, instrument).payload) != dict(self.rule.payload)
               for row in self.rows):
            raise DataError("split episode at historical instrument-rule changes")
        self._bars = tuple(bars)
        config_identity: dict[str, Any] = {
            "execution": execution,
            "quote_max_age_ns": (QUOTE_MAX_AGE_NS if self.execution.base_fill.data_granularity == "quote" else None),
            "hard": hard_risk, "soft": soft_risk,
            "capital": starting_balance, "budget": risk_budget_notional,
        }
        # Preserve historical default run identities; bind every experimental
        # non-default optimizer reward scale to its new artifacts.
        if self.reward_config.reward_scale != 1.0:
            config_identity["reward_scale"] = self.reward_config.reward_scale
        if self.reward_config.lambda_dd != 0.0:
            config_identity["reward_lambda_dd"] = self.reward_config.lambda_dd
        if self.reward_config.lambda_guard != 0.0:
            config_identity["reward_lambda_guard"] = self.reward_config.lambda_guard
        self.base_config_hash = digest({**config_identity, "mode": "base"})
        self.config_hash = digest({**config_identity, "mode": mode})
        transition_identity: dict[str, Any] = {
            "terminal_value_semantics": "finite_settlement_no_bootstrap_v1",
            "data": self.store.snapshot_hash, "config": self.config_hash,
            "schema": self.schema.hash, "features": self.features.hash,
            "scaler": self.scaler.hash, "window": [start_ns, end_ns],
        }
        if self.reward_config.reward_scale != 1.0:
            transition_identity["reward_scale"] = self.reward_config.reward_scale
        if self.reward_config.lambda_dd != 0.0:
            transition_identity["reward_lambda_dd"] = self.reward_config.lambda_dd
        if self.reward_config.lambda_guard != 0.0:
            transition_identity["reward_lambda_guard"] = self.reward_config.lambda_guard
        if self.feature_profile in {"causal_mtf_v3", "causal_mtf_v4", "causal_mtf_v5"}:
            known_schedule = self.coverage.known_schedule_times_ns
            if known_schedule is None:
                raise DataError(f"{self.feature_profile} requires a source-declared future funding calendar")
            transition_identity["feature_profile"] = self.feature_profile
            transition_identity["known_schedule"] = digest({
                symbol: tuple(times)
                for symbol, times in known_schedule.items()
            })
        self.transition_hash = digest(transition_identity)
        self.snapshots: list[Any] = []
        self.fill_references: dict[str, Decimal] = {}
        self._market_batch_index: tuple[tuple[int, VisibleBatch], ...] | None = None
        self._market_batch_times: tuple[int, ...] = ()
        self._market_batch_groups_by_row: tuple[tuple[VisibleBatch, ...], ...] = ()
        self._barrier_frames_by_row: tuple[BarrierFrame, ...] | None = None

    def _visible_market_batch_index(self) -> tuple[tuple[int, VisibleBatch], ...]:
        """Build one immutable availability-time index for all Episodes in this factory."""
        if self._market_batch_index is not None:
            return self._market_batch_index
        warmup_minutes = max(
            (field.lookback_bars * field.timeframe_minutes
             for field in self.features.fields if field.source_kind == "ohlcv_1m"),
            default=1,
        ) + 2
        first_decision_ns = self.rows[0].decision_ts_ns
        last_decision_ns = self.rows[-1].decision_ts_ns
        feed_start_ns = max(0, first_decision_ns - warmup_minutes * MINUTE_NS)
        groups: dict[int, list] = defaultdict(list)
        for event in self.store.available_event_window(feed_start_ns, last_decision_ns):
            if event.kind in {"instrument_rule", "funding_update"}:
                continue
            if event.payload.get("instrument") != self.instrument:
                continue
            available_at_ns = event.available_at_ns
            if available_at_ns is None:
                continue
            groups[available_at_ns].append(to_market_event(
                event, quality_grade=DataQualityGrade(self.quality.grade.value)))
        batches = tuple(
            (timestamp, VisibleBatch(timestamp, tuple(events), True))
            for timestamp, events in sorted(groups.items())
        )
        self._market_batch_times = tuple(timestamp for timestamp, _batch in batches)
        self._market_batch_groups_by_row = _visible_batches_by_decision(
            batches, tuple(row.decision_ts_ns for row in self.rows),
        )
        self._market_batch_index = batches
        return batches

    def _episode_market_batch_groups(self, *, start_offset: int,
                                     episode_rows: tuple[Any, ...],
                                     feed_start_ns: int) -> tuple[tuple[VisibleBatch, ...], ...]:
        """Reuse row deltas while restoring the episode's required warmup prefix."""
        batches = self._visible_market_batch_index()
        first_batch = bisect_left(self._market_batch_times, feed_start_ns)
        first_decision_end = bisect_right(
            self._market_batch_times, episode_rows[0].decision_ts_ns,
        )
        initial = tuple(batch for _timestamp, batch in batches[first_batch:first_decision_end])
        following = self._market_batch_groups_by_row[
            start_offset + 1:start_offset + len(episode_rows)
        ]
        if len(following) != len(episode_rows) - 1:
            raise DataError("cached visible-batch rows do not cover the Episode window")
        return (initial, *following)

    def _cached_barrier_frames(self) -> tuple[BarrierFrame, ...]:
        """Build immutable observation/event frames once for this frozen factory."""
        if self._barrier_frames_by_row is not None:
            return self._barrier_frames_by_row
        self._visible_market_batch_index()
        frames = []
        for row, batches in zip(self.rows, self._market_batch_groups_by_row, strict=True):
            market_features = visible_market_features(
                row, self.features, self.schema, instrument=self.instrument,
                decision_ts_ns=row.decision_ts_ns,
            )
            if self.feature_profile in {"causal_mtf_v3", "causal_mtf_v4", "causal_mtf_v5"}:
                known_schedule = self.coverage.known_schedule_times_ns
                if known_schedule is None:
                    raise DataError(f"{self.feature_profile} requires a source-declared future funding calendar")
                time_features = scheduled_funding_time_features(
                    row.decision_ts_ns, known_schedule[self.instrument],
                )
            else:
                time_features = {}
            frames.append(BarrierFrame(row.decision_ts_ns, batches,
                                       market_features, time_features))
        self._barrier_frames_by_row = tuple(frames)
        return self._barrier_frames_by_row

    def _episode_barrier_frames(self, *, start_offset: int,
                                episode_rows: tuple[Any, ...],
                                feed_start_ns: int) -> tuple[BarrierFrame, ...]:
        """Reuse cached frames, rebuilding only a shifted Episode's warmup frame."""
        cached_frames = self._cached_barrier_frames()
        stop = start_offset + len(episode_rows)
        selected = cached_frames[start_offset:stop]
        if len(selected) != len(episode_rows):
            raise DataError("cached barrier frames do not cover the Episode window")
        if start_offset == 0:
            return selected
        first_batches = self._episode_market_batch_groups(
            start_offset=start_offset, episode_rows=episode_rows[:1],
            feed_start_ns=feed_start_ns,
        )[0]
        first = BarrierFrame(selected[0].timestamp_ns, first_batches,
                             selected[0].market, selected[0].time_features)
        return (first, *selected[1:])

    def observation_state(self, snapshot) -> dict:
        cash, free = native_balances(snapshot)
        if snapshot.mark is None:
            raise DataError("native mark required for exposure observations")
        notional = signed_quantity(snapshot) * snapshot.mark * self.contract_spec.multiplier
        if snapshot.position_initial_margin is None:
            raise DataError("native mark required to calculate position initial margin")
        position_initial = Decimal(str(snapshot.position_initial_margin))
        if not position_initial.is_finite() or position_initial < 0:
            raise DataError("native position initial margin is invalid")
        initial = (sum((_money(m["initial"]) for m in snapshot.margins), Decimal(0))
                   + position_initial)
        maintenance = sum((_money(m["maintenance"]) for m in snapshot.margins), Decimal(0))
        submitted = Decimal(str(getattr(snapshot, "cumulative_submitted_qty", 0)))
        filled = Decimal(str(getattr(snapshot, "cumulative_filled_qty", 0)))
        if not hasattr(snapshot, "cumulative_submitted_qty"):
            submitted = sum((Decimal(o["quantity"]) for o in snapshot.all_orders), Decimal(0))
            filled = sum((Decimal(o["filled_qty"]) for o in snapshot.all_orders), Decimal(0))
        pending_signed = Decimal(0)
        pending_gross = Decimal(0)
        for order in snapshot.orders:
            side = order.get("side")
            leaves = Decimal(str(order.get("leaves_qty", "0")))
            if side not in {"BUY", "SELL"} or not leaves.is_finite() or leaves < 0:
                raise DataError("native pending order side/quantity invalid for observation")
            signed = leaves if side == "BUY" else -leaves
            pending_signed += signed * snapshot.mark * self.contract_spec.multiplier / self.budget
            pending_gross += leaves * snapshot.mark * self.contract_spec.multiplier / self.budget
        funding_available = bool(self.store.visible(snapshot.decision_ts_ns, "funding_update"))
        direction = Decimal(-1) if notional < 0 else Decimal(1)
        decision = HardRiskGuard(self.hard).apply(
            direction, budget_notional=self.budget, equity=snapshot.equity,
            free_margin=free, current_notional=notional, mark_available=True,
            funding_available=funding_available, liquidation_capability=True,
            data_quality_grade=self.quality.grade.value,
            estimated_cost_rate=self.hard_execution_cost_rate,
        )
        allowed_notional = abs(decision.adjusted * self.budget)
        headroom = max(Decimal(0), allowed_notional - abs(notional)) / self.budget
        return {
            "account": {"cash": float(cash), "free_margin": float(free), "initial_margin": float(initial),
                        "maintenance_margin": float(maintenance), "unrealized_pnl": float(snapshot.equity - cash),
                        "net_exposure": float(notional / self.budget), "position_quantity": float(signed_quantity(snapshot))},
            "risk": {"headroom": float(headroom)},
            "execution": {"pending_order_count": float(len(snapshot.orders)),
                          "pending_signed_exposure": float(pending_signed),
                          "pending_gross_exposure": float(pending_gross),
                          "fill_rate": float(filled / submitted) if submitted else 0.0},
        }

    def make_env(self, transition_hash: str | None = None, *, replay: EpisodeReplay | None = None,
                 soft_multipliers: dict | None = None, episode_rank: int = 0,
                 episode_count: int = 1, episode_steps: int | None = None) -> NautilusBarrierEnv:
        transition = transition_hash or self.transition_hash
        pending_replay = replay
        full_steps = len(self.rows) - 3
        selected_steps = full_steps if episode_steps is None else episode_steps
        if (type(episode_rank) is not int or type(episode_count) is not int
                or type(selected_steps) is not int or not 0 <= episode_rank < episode_count
                or selected_steps < 1 or selected_steps > full_steps):
            raise DataError("invalid rotating historical episode partition")
        lane_start, lane_end = _episode_lane(
            full_steps=full_steps, episode_rank=episode_rank,
            episode_count=episode_count, episode_steps=selected_steps,
        )

        def episode_factory(*, seed, options):
            nonlocal pending_replay
            if not isinstance(options, dict):
                raise DataError("historical episode options must be a dictionary")
            requested_index = options.get("rles_episode_index", 0)
            current_replay = pending_replay
            episode_index = (current_replay.episode_index
                             if current_replay is not None
                             and current_replay.episode_index is not None
                             else requested_index)
            if type(episode_index) is not int or episode_index < 0:
                raise DataError("historical episode index must be a non-negative integer")
            start_offset = _episode_start(
                lane_start=lane_start, lane_end=lane_end,
                episode_steps=selected_steps, episode_index=episode_index,
            )
            if (current_replay is not None
                    and current_replay.episode_start_offset is not None
                    and current_replay.episode_start_offset != start_offset):
                raise DataError("checkpoint episode window differs from deterministic schedule")
            episode_rows = self.rows[start_offset:start_offset + selected_steps + 3]
            episode_fold_hash = (digest([self.start_ns, self.end_ns])
                                 if episode_steps is None and episode_count == 1 else digest({
                                     "window": [episode_rows[0].decision_ts_ns,
                                                episode_rows[-1].decision_ts_ns],
                                     "partition": [episode_rank, episode_count, selected_steps],
                                     "episode_index": episode_index,
                                     "start_offset": start_offset,
                                 }))
            self.snapshots = []
            self.fill_references = {}
            seen_fills: dict[str, dict] = {}
            seen_funding: dict[str, dict] = {}
            seen_liquidations: dict[str, dict] = {}
            seen_orders: dict[str, dict] = {}
            order_filled: dict[str, Decimal] = {}
            cumulative_submitted = Decimal(0)
            cumulative_filled = Decimal(0)
            suppress_rebuild_bootstrap = [False]

            def unique_delta(records, seen, identity_field="event_id"):
                delta = []
                for record in records:
                    identity = str(record.get(identity_field, ""))
                    if not identity:
                        raise DataError("native history fact has no stable identity")
                    previous = seen.get(identity)
                    if previous is not None:
                        if previous != record:
                            raise DataError("native history identity has conflicting facts")
                        continue
                    copied = dict(record)
                    seen[identity] = copied
                    delta.append(copied)
                return tuple(delta)

            def record_snapshot(snapshot):
                nonlocal cumulative_submitted, cumulative_filled
                if suppress_rebuild_bootstrap[0]:
                    suppress_rebuild_bootstrap[0] = False
                    return
                for order in snapshot.all_orders:
                    identity = str(order.get("client_order_id", ""))
                    if not identity:
                        raise DataError("native order history has no client identity")
                    quantity = Decimal(str(order["quantity"]))
                    filled = Decimal(str(order["filled_qty"]))
                    previous = order_filled.get(identity)
                    if previous is None:
                        cumulative_submitted += quantity
                        previous = Decimal(0)
                    if filled < previous or filled > quantity:
                        raise DataError("native order fill total regressed or exceeded order size")
                    cumulative_filled += filled - previous
                    order_filled[identity] = filled
                    prior_order = seen_orders.get(identity)
                    if prior_order is not None and any(
                        prior_order.get(field) != order.get(field)
                        for field in ("instrument_id", "side", "quantity")
                    ):
                        raise DataError("native order identity has conflicting immutable facts")
                    seen_orders[identity] = dict(order)
                fills = unique_delta(snapshot.fills, seen_fills)
                funding = unique_delta(snapshot.funding, seen_funding)
                liquidations = (None if snapshot.liquidation_orders is None else
                                unique_delta(snapshot.liquidation_orders, seen_liquidations,
                                            "client_order_id"))
                final_boundary = snapshot.decision_ts_ns >= episode_rows[-2].decision_ts_ns
                if self.compact_report_snapshots:
                    # Retain only the native contract scalars needed to verify
                    # open linear equity. Full Position.to_dict histories can
                    # grow with every fill and are not required by the report.
                    position_fields = ("position_id", "side", "quantity", "multiplier", "is_inverse",
                                       "settlement_currency")
                    if self.feature_profile in {"causal_mtf_v4", "causal_mtf_v5"}:
                        position_fields += ("unrealized_pnl", "unrealized_pnl_currency",
                                             "notional_value", "notional_currency")
                    report_positions = tuple({key: row.get(key) for key in position_fields}
                                              for row in snapshot.positions)
                else:
                    report_positions = snapshot.positions
                compact = SimpleNamespace(
                    equity=snapshot.equity, currency=snapshot.currency,
                    positions=report_positions,
                    orders=(snapshot.orders if not self.compact_report_snapshots or final_boundary else ()),
                    account=(snapshot.account if not self.compact_report_snapshots else {}),
                    account_events=(snapshot.account_events[-1:] if not self.compact_report_snapshots else ()),
                    fills=tuple(seen_fills.values()) if final_boundary else fills,
                    funding=tuple(seen_funding.values()) if final_boundary else funding,
                    all_orders=(tuple(seen_orders.values()) if final_boundary else
                                snapshot.all_orders if not self.compact_report_snapshots else ()),
                    liquidation_orders=(None if snapshot.liquidation_orders is None else
                                        tuple(seen_liquidations.values()) if final_boundary else liquidations),
                    margins=(snapshot.margins if not self.compact_report_snapshots else ()),
                    mark=snapshot.mark,
                    liquidation_capability=snapshot.liquidation_capability,
                    decision_ts_ns=snapshot.decision_ts_ns,
                    cumulative_submitted_qty=cumulative_submitted,
                    cumulative_filled_qty=cumulative_filled,
                    fill_references=dict(self.fill_references),
                    contract_multiplier=self.contract_spec.multiplier,
                )
                self.snapshots.append(compact)

            def record_reference(client_id, reference):
                previous = self.fill_references.get(client_id)
                if previous is not None and previous != reference:
                    raise DataError("multiple native fill references per order require event-level allocation")
                self.fill_references[client_id] = reference

            def make_session(starting_balance, leverage):
                return make_perpetual_session(
                    rule=self.rule.payload, contract_spec=self.contract_spec,
                    execution=self.execution, mode=self.mode,
                    starting_balance=starting_balance, leverage=leverage,
                    record_snapshot=record_snapshot,
                    record_fill_reference=record_reference,
                    compact_native_history=self.compact_native_history,
                    include_position_pnl=(self.feature_profile in {
                        "causal_mtf_v4", "causal_mtf_v5"}),
                )

            session = make_session(self.capital, self.hard.max_leverage)
            hard_ref = [HardRiskGuard(self.hard)]
            planner = TargetQuantityPlanner(self.execution)
            rules = InstrumentRules(self.instrument, Decimal(self.rule.payload["step_size"]),
                                    Decimal(self.rule.payload["min_qty"]), Decimal(self.rule.payload["min_notional"]),
                                    self.contract_spec.multiplier,
                                    self.rule.payload.get("reduce_only_min_notional_exempt", False))
            soft = SoftRiskTransform(self.soft if not soft_multipliers else self.soft.with_multipliers(soft_multipliers))
            def plan_with_rounding(target, snapshot, *, terminal=False, force_reduce=False):
                decision = planner.plan(
                    decision_id=f"{'terminal' if terminal else 'policy'}-{snapshot.decision_ts_ns}",
                    instrument=rules, target_exposure=target, risk_budget_notional=self.budget,
                    equity=snapshot.equity, mark_price=snapshot.mark, current_qty=signed_quantity(snapshot),
                    pending=tuple(PendingOrder(o["client_order_id"], Decimal(o["leaves_qty"]) * (-1 if o["side"] == "SELL" else 1), o["status"])
                                  for o in snapshot.orders), terminal=terminal, force_reduce=force_reduce,
                )
                contract_plan = to_contract_plan(
                    decision, ExecutionPlan, latency_model_id=self.execution.latency(self.mode).model_id,
                    quantity_unit=self.contract_spec.base_currency,
                )
                rounded_exposure = (decision.target_qty * snapshot.mark * rules.contract_multiplier
                                    / self.budget)
                return contract_plan, rounded_exposure

            def plan(target, snapshot, *, terminal=False, force_reduce=False):
                return plan_with_rounding(target, snapshot, terminal=terminal,
                                          force_reduce=force_reduce)[0]

            def plan_action(action, snapshot, drawdown):
                current = (signed_quantity(snapshot) * snapshot.mark * self.contract_spec.multiplier
                           / self.budget)
                known = self.store.visible(snapshot.decision_ts_ns, "funding_update")
                rate = Decimal(str(known[-1].payload["rate"])) if known else None
                softened = soft.apply(action, current_exposure=current, drawdown=drawdown, funding_abs=rate)
                adjusted = hard_ref[0].apply(softened.adjusted, budget_notional=self.budget, equity=snapshot.equity,
                                      free_margin=native_balances(snapshot)[1], current_notional=current * self.budget,
                                      mark_available=True, funding_available=rate is not None,
                                      liquidation_capability=True, data_quality_grade=self.quality.grade.value,
                                      estimated_cost_rate=self.hard_execution_cost_rate)
                contract_plan, rounded_exposure = plan_with_rounding(
                    adjusted.adjusted, snapshot, force_reduce=adjusted.force_reduce,
                )
                return RiskActionReview(
                    contract_plan, abs(action - adjusted.adjusted),
                    softened.intervened_fraction, adjusted.intervened_fraction, True,
                    softened.adjusted, adjusted.adjusted, rounded_exposure,
                )

            warmup_minutes = max(
                (field.lookback_bars * field.timeframe_minutes
                 for field in self.features.fields if field.source_kind == "ohlcv_1m"),
                default=1,
            ) + 2
            feed_start_ns = max(0, episode_rows[0].decision_ts_ns - warmup_minutes * MINUTE_NS)
            frames = self._episode_barrier_frames(
                start_offset=start_offset, episode_rows=episode_rows,
                feed_start_ns=feed_start_ns,
            )

            position_clock = (_NativePositionClock()
                              if self.feature_profile in {"causal_mtf_v4", "causal_mtf_v5"}
                              else None)

            def native_observation_state(snapshot):
                values = self.observation_state(snapshot)
                if position_clock is not None:
                    values["account"]["unrealized_pnl_pct"] = _native_unrealized_pnl_pct(
                        snapshot, settlement_currency=self.contract_spec.settlement_currency,
                    )
                    values["execution"].update(position_clock.observe(snapshot))
                return values

            def rebuild_session(old_session, leverage, snapshot, frame, action_step, identity):
                if self.quality.grade.value not in {"DQ-A", "DQ-B"} and leverage > 1:
                    raise DataError("high leverage requires verified DQ-A or DQ-B market data")
                if (snapshot.orders or snapshot.positions or snapshot.equity <= 0
                        or snapshot.currency != self.contract_spec.quote_currency):
                    raise DataError("leverage change requires a positive-equity flat account with no orders")
                next_hard = HardRiskConfig(**{
                    **asdict(self.hard), "max_leverage": leverage,
                })
                next_guard = HardRiskGuard(next_hard)
                next_identity = dict(identity)
                next_identity["config_hash"] = digest({
                    "parent_config_hash": identity["config_hash"],
                    "leverage": str(leverage),
                    "action_step": action_step,
                })
                next_session = make_session(Decimal(str(snapshot.equity)), leverage)
                try:
                    # Nautilus rc6 permits one live native engine per process.
                    # The episode has already proven this barrier flat and order-free.
                    old_session.close()
                    next_session.open(SimpleNamespace(**next_identity))
                    suppress_rebuild_bootstrap[0] = True
                    next_snapshot = next_session.bootstrap(
                        frame.batches, decision_ts_ns=frame.timestamp_ns,
                    )
                    if suppress_rebuild_bootstrap[0]:
                        raise DataError("rebuilt native session did not report its bootstrap snapshot")
                    if (Decimal(str(next_snapshot.equity)) != Decimal(str(snapshot.equity))
                            or next_snapshot.currency != snapshot.currency
                            or next_snapshot.orders or next_snapshot.positions):
                        raise DataError("rebuilt native session changed account facts at the leverage barrier")
                except BaseException as exc:
                    suppress_rebuild_bootstrap[0] = False
                    next_session.close()
                    raise SessionRebuildInterrupted(
                        "new native session failed after the old engine was closed; reset the episode",
                    ) from exc
                hard_ref[0] = next_guard
                return next_session, next_identity, next_snapshot

            episode = BarrierEpisode(
                native_session=session,
                native_identity={"dataset_hash": self.store.snapshot_hash,
                                 "fold_hash": episode_fold_hash,
                                 "config_hash": self.config_hash},
                transition_hash=transition, frames=frames, scoring_frames=len(frames) - 2,
                plan_action=plan_action, plan_terminal=lambda s: plan(Decimal(0), s, terminal=True),
                executed_exposure=lambda s: (signed_quantity(s) * s.mark * self.contract_spec.multiplier
                                             / self.budget),
                reward_config=self.reward_config,
                replay=current_replay,
                episode_index=episode_index, episode_start_offset=start_offset,
                observation_state=native_observation_state,
                rebuild_session=rebuild_session,
            )
            pending_replay = None
            return episode

        bridge = L2FrozenScalerAdapter(self.schema, self.features, self.scaler, expected_scaler_hash=self.scaler.hash)
        return NautilusBarrierEnv(episode_factory=episode_factory,
                                  observation_builder=ObservationBuilder(self.schema, scaler=bridge, expected_scaler_hash=bridge.fingerprint,
                                                                        split="train" if self.split == "train_inner" else self.split),
                                  reward_config=self.reward_config, max_steps=selected_steps,
                                  transition_hash=transition,
                                  fill_reference_reader=lambda: dict(self.fill_references))

    def metadata(self) -> dict:
        return {
            "quality": asdict(self.quality), "features": asdict(self.features),
            "contract_spec": {
                "instrument_id": self.contract_spec.instrument_id,
                "raw_symbol": self.contract_spec.raw_symbol,
                "base_currency": self.contract_spec.base_currency,
                "quote_currency": self.contract_spec.quote_currency,
                "settlement_currency": self.contract_spec.settlement_currency,
                "is_inverse": self.contract_spec.is_inverse,
                "multiplier": str(self.contract_spec.multiplier),
            },
            "feature_profile": self.feature_profile,
            "reward_config": asdict(self.reward_config),
            "known_schedule_times_ns": (None if self.coverage.known_schedule_times_ns is None else
                                         {symbol: tuple(times) for symbol, times in
                                          self.coverage.known_schedule_times_ns.items()}),
            "schema": asdict(self.schema), "scaler": asdict(self.scaler),
            "transition_hash": self.transition_hash,
            "identities": {
                "dataset": self.store.snapshot_hash,
                "features": self.features.hash,
                "schema": self.schema.hash,
                "scaler": self.scaler.hash,
                "execution": self.execution.fingerprint(),
                "hard_risk": self.hard.signature(),
                "config": self.config_hash,
            },
        }


def make_historical_es_env(*, env_config: dict, candidate_context: dict,
                           soft_risk_multipliers: dict) -> NautilusBarrierEnv:
    """Installed production factory for candidate-isolated L6 spawn workers."""
    if candidate_context.get("split") != "train_inner":
        raise DataError("ES may evaluate only the frozen Train-inner window")
    raw = dict(env_config["scaler"])
    raw["mean"], raw["std"] = tuple(raw["mean"]), tuple(raw["std"])
    prepared_store = env_config.get("prepared_store")
    if prepared_store is not None and not isinstance(prepared_store, CuratedStore):
        raise DataError("ES worker prepared history must be a validated CuratedStore")
    factory = HistoricalEnvironmentFactory(**{
        **env_config["recipe"], "fit_scaler": False, "scaler": Scaler(**raw),
        "prepared_store": prepared_store,
    })
    return factory.make_env(env_config["transition_hash"], soft_multipliers=soft_risk_multipliers)
