"""Real historical perpetual environment assembly, with no test imports."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict
from decimal import Decimal
from hashlib import sha256
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from rles_contracts.dto import DataQualityGrade, ExecutionPlan
from rles_data.contracts import to_market_event
from rles_data.core import Availability, CuratedStore, DataError, Event, FeatureField, FeatureSpec, KINDS, Window, digest
from rles_data.coverage import CoveragePolicy
from rles_data.features import Scaler, fit_train, make_feature_row
from rles_data.io import read_curated
from rles_execution import InstrumentRules, PendingOrder, ResolvedExecutionConfig, TargetQuantityPlanner, to_contract_plan
from rles_nautilus_adapter.perpetual import QUOTE_MAX_AGE_NS, make_perpetual_session
from rles_nautilus_adapter.session import VisibleBatch
from rles_risk import HardRiskConfig, HardRiskGuard, SoftRiskParams, SoftRiskTransform

from .episode import BarrierEpisode, BarrierFrame, EpisodeReplay, RiskActionReview
from .env import NautilusBarrierEnv
from .l2_bridge import L2FrozenScalerAdapter, visible_market_features
from .observation import ObservationBuilder, ObservationSchema
from .reward import RewardConfig


_FUNDING_TIME_HORIZON_NS = 24 * 60 * 60 * 1_000_000_000


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


def historical_feature_spec(maximum_age_ns: dict[str, int], *, profile: str = "legacy") -> FeatureSpec:
    """Return a frozen causal feature recipe selected by the run configuration.

    ``legacy`` preserves the original observation identity for existing bundles.
    ``causal_mtf_v1`` adds closed-bar return and volatility context at longer
    horizons. ``causal_mtf_v2`` also adds bounded trend-efficiency features that
    distinguish persistent directional movement from choppy price paths. The
    ``causal_mtf_v3`` adds a source-scheduled next-settlement countdown to the
    observation schema. Its market features, observation schema and transition
    identity remain bound to the selected recipe.
    """
    if set(maximum_age_ns) != {"mark", "index", "funding_update"}:
        raise DataError("historical feature recipe requires complete freshness limits")
    if not isinstance(profile, str) or profile not in {"legacy", "causal_mtf_v1", "causal_mtf_v2", "causal_mtf_v3"}:
        raise DataError("unsupported historical feature profile")
    fields: tuple[FeatureField, ...] = (
        FeatureField("return_1m", "float64", "ratio", "ohlcv_1m", 1, 1, "return"),
        FeatureField("volatility_5m", "float64", "ratio", "ohlcv_1m", 5, 1, "volatility"),
        FeatureField("volume_5m", "float64", "ETH", "ohlcv_1m", 5, 1, "volume_mean"),
        FeatureField("basis", "float64", "ratio", "mark", 1, 1, "basis", maximum_age_ns["mark"]),
        FeatureField("last_settled_funding", "float64", "ratio", "funding_update", 1, 1,
                     "latest", maximum_age_ns["funding_update"]),
    )
    if profile in {"causal_mtf_v1", "causal_mtf_v2", "causal_mtf_v3"}:
        fields += (
            FeatureField("return_5m", "float64", "ratio", "ohlcv_1m", 5, 1, "return"),
            FeatureField("return_15m", "float64", "ratio", "ohlcv_1m", 15, 1, "return"),
            FeatureField("return_60m", "float64", "ratio", "ohlcv_1m", 60, 1, "return"),
            FeatureField("volatility_15m", "float64", "ratio", "ohlcv_1m", 15, 1, "volatility"),
            FeatureField("volatility_60m", "float64", "ratio", "ohlcv_1m", 60, 1, "volatility"),
        )
    if profile in {"causal_mtf_v2", "causal_mtf_v3"}:
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


class HistoricalEnvironmentFactory:
    """Freeze data, rules and training-only scaler; create a fresh Engine per Env."""
    def __init__(self, *, market_path: str | Path, rules_path: str | Path,
                 instrument: str, start_ns: int, end_ns: int, coverage_policy: dict,
                 execution: dict, hard_risk: dict, soft_risk: dict,
                 starting_balance: str, risk_budget_notional: str,
                 mode: str = "base", scaler: Scaler | None = None,
                 fit_scaler: bool = False, split: str = "train_inner",
                 compact_native_history: bool = True,
                 feature_profile: str = "legacy") -> None:
        if split not in {"train_inner", "validation", "test", "holdout"}:
            raise DataError("unknown historical research split")
        if fit_scaler and split != "train_inner":
            raise DataError("only Train-inner may fit feature scaling")
        if not fit_scaler and scaler is None:
            raise DataError("evaluation must reuse the frozen training feature scaler")
        if type(compact_native_history) is not bool:
            raise DataError("native history compaction flag must be Boolean")
        if not isinstance(feature_profile, str) or feature_profile not in {
            "legacy", "causal_mtf_v1", "causal_mtf_v2", "causal_mtf_v3",
        }:
            raise DataError("unsupported historical feature profile")
        self.compact_native_history = compact_native_history
        self.feature_profile = feature_profile
        self.store = CuratedStore((*read_curated(market_path).events, *load_rules(rules_path)))
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
        if self.hard.max_leverage > 1 or self.hard.incomplete_data_max_leverage > 1:
            raise DataError("leveraged liquidation replay is not yet certified by this factory")
        self.rule = self.store.latest_rule(start_ns, instrument)
        if any(dict(self.store.latest_rule(t, instrument).payload) != dict(self.rule.payload)
               for t in (end_ns - 1,)):
            raise DataError("split episode at historical instrument-rule changes")
        self.features = historical_feature_spec(dict(self.coverage.maximum_age_ns), profile=self.feature_profile)
        account = ("equity", "peak_equity", "drawdown", "cash", "free_margin", "initial_margin",
                   "maintenance_margin", "unrealized_pnl", "net_exposure", "position_quantity")
        execution_names = (
            "prev_executed_target", "prev_raw_target", "pending_order_count",
            "pending_signed_exposure", "pending_gross_exposure", "fill_rate",
        )
        names = tuple("market." + f.name for f in self.features.fields) + tuple("account." + n for n in account)
        units = tuple(f.unit for f in self.features.fields) + ("USDT", "USDT", "ratio", "USDT", "USDT", "USDT", "USDT", "USDT", "ratio", "ETH")
        time_names = (("time.funding_settlement_within_24h",
                       "time.funding_settlement_time_to_event_24h")
                      if self.feature_profile == "causal_mtf_v3" else ())
        self.schema = ObservationSchema("0.4.0", names + time_names + ("risk.guard_intervention", "risk.headroom")
                                        + tuple("execution." + n for n in execution_names),
                                        units + ("ratio",) * len(time_names)
                                        + ("ratio", "ratio", "ratio", "ratio", "count", "ratio", "ratio", "ratio"))
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
        config_identity = {
            "execution": execution,
            "quote_max_age_ns": (QUOTE_MAX_AGE_NS if self.execution.base_fill.data_granularity == "quote" else None),
            "hard": hard_risk, "soft": soft_risk,
            "capital": starting_balance, "budget": risk_budget_notional,
        }
        self.base_config_hash = digest({**config_identity, "mode": "base"})
        self.config_hash = digest({**config_identity, "mode": mode})
        transition_identity: dict[str, Any] = {
            "data": self.store.snapshot_hash, "config": self.config_hash,
            "schema": self.schema.hash, "features": self.features.hash,
            "scaler": self.scaler.hash, "window": [start_ns, end_ns],
        }
        if self.feature_profile == "causal_mtf_v3":
            transition_identity["feature_profile"] = self.feature_profile
        self.transition_hash = digest(transition_identity)
        self.snapshots: list[Any] = []
        self.fill_references: dict[str, Decimal] = {}

    def observation_state(self, snapshot) -> dict:
        cash, free = native_balances(snapshot)
        if snapshot.mark is None:
            raise DataError("native mark required for exposure observations")
        notional = signed_quantity(snapshot) * snapshot.mark
        initial = sum((_money(m["initial"]) for m in snapshot.margins), Decimal(0))
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
            pending_signed += signed * snapshot.mark / self.budget
            pending_gross += leaves * snapshot.mark / self.budget
        cap = min(self.hard.max_notional, snapshot.equity * self.hard.max_leverage)
        return {
            "account": {"cash": float(cash), "free_margin": float(free), "initial_margin": float(initial),
                        "maintenance_margin": float(maintenance), "unrealized_pnl": float(snapshot.equity - cash),
                        "net_exposure": float(notional / self.budget), "position_quantity": float(signed_quantity(snapshot))},
            "risk": {"headroom": float(max(Decimal(0), cap - abs(notional)) / self.budget)},
            "execution": {"pending_order_count": float(len(snapshot.orders)),
                          "pending_signed_exposure": float(pending_signed),
                          "pending_gross_exposure": float(pending_gross),
                          "fill_rate": float(filled / submitted) if submitted else 0.0},
        }

    def make_env(self, transition_hash: str | None = None, *, replay: EpisodeReplay | None = None,
                 soft_multipliers: dict | None = None) -> NautilusBarrierEnv:
        transition = transition_hash or self.transition_hash
        pending_replay = replay

        def episode_factory(*, seed, options):
            nonlocal pending_replay
            self.snapshots = []
            self.fill_references = {}
            seen_fills: dict[str, dict] = {}
            seen_funding: dict[str, dict] = {}
            seen_liquidations: dict[str, dict] = {}
            order_filled: dict[str, Decimal] = {}
            cumulative_submitted = Decimal(0)
            cumulative_filled = Decimal(0)

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
                fills = unique_delta(snapshot.fills, seen_fills)
                funding = unique_delta(snapshot.funding, seen_funding)
                liquidations = (None if snapshot.liquidation_orders is None else
                                unique_delta(snapshot.liquidation_orders, seen_liquidations,
                                            "client_order_id"))
                final_boundary = snapshot.decision_ts_ns >= self.rows[-2].decision_ts_ns
                compact = SimpleNamespace(
                    equity=snapshot.equity, currency=snapshot.currency,
                    positions=snapshot.positions, orders=snapshot.orders,
                    account=snapshot.account,
                    account_events=snapshot.account_events[-1:],
                    fills=tuple(seen_fills.values()) if final_boundary else fills,
                    funding=tuple(seen_funding.values()) if final_boundary else funding,
                    all_orders=snapshot.all_orders if final_boundary else (),
                    liquidation_orders=(None if snapshot.liquidation_orders is None else
                                        tuple(seen_liquidations.values()) if final_boundary else liquidations),
                    margins=snapshot.margins,
                    mark=snapshot.mark, liquidation_capability=snapshot.liquidation_capability,
                    decision_ts_ns=snapshot.decision_ts_ns,
                    cumulative_submitted_qty=cumulative_submitted,
                    cumulative_filled_qty=cumulative_filled,
                )
                self.snapshots.append(compact)

            def record_reference(client_id, reference):
                previous = self.fill_references.get(client_id)
                if previous is not None and previous != reference:
                    raise DataError("multiple native fill references per order require event-level allocation")
                self.fill_references[client_id] = reference

            session = make_perpetual_session(rule=self.rule.payload, execution=self.execution, mode=self.mode,
                                             starting_balance=self.capital, record_snapshot=record_snapshot,
                                             record_fill_reference=record_reference,
                                             compact_native_history=self.compact_native_history)
            planner = TargetQuantityPlanner(self.execution)
            rules = InstrumentRules(self.instrument, Decimal(self.rule.payload["step_size"]),
                                    Decimal(self.rule.payload["min_qty"]), Decimal(self.rule.payload["min_notional"]),
                                    Decimal(1), self.rule.payload.get("reduce_only_min_notional_exempt", False))
            soft = SoftRiskTransform(self.soft if not soft_multipliers else self.soft.with_multipliers(soft_multipliers))
            hard = HardRiskGuard(self.hard)

            def plan(target, snapshot, *, terminal=False, force_reduce=False):
                decision = planner.plan(
                    decision_id=f"{'terminal' if terminal else 'policy'}-{snapshot.decision_ts_ns}",
                    instrument=rules, target_exposure=target, risk_budget_notional=self.budget,
                    equity=snapshot.equity, mark_price=snapshot.mark, current_qty=signed_quantity(snapshot),
                    pending=tuple(PendingOrder(o["client_order_id"], Decimal(o["leaves_qty"]) * (-1 if o["side"] == "SELL" else 1), o["status"])
                                  for o in snapshot.orders), terminal=terminal, force_reduce=force_reduce,
                )
                return to_contract_plan(decision, ExecutionPlan, latency_model_id=self.execution.latency(self.mode).model_id, quantity_unit="ETH")

            def plan_action(action, snapshot, drawdown):
                current = signed_quantity(snapshot) * snapshot.mark / self.budget
                known = self.store.visible(snapshot.decision_ts_ns, "funding_update")
                rate = Decimal(str(known[-1].payload["rate"])) if known else None
                softened = soft.apply(action, current_exposure=current, drawdown=drawdown, funding_abs=rate)
                adjusted = hard.apply(softened.adjusted, budget_notional=self.budget, equity=snapshot.equity,
                                      free_margin=native_balances(snapshot)[1], current_notional=current * self.budget,
                                      mark_available=True, funding_available=rate is not None,
                                      liquidation_capability=True, data_quality_grade=self.quality.grade.value)
                return RiskActionReview(
                    plan(adjusted.adjusted, snapshot, force_reduce=adjusted.force_reduce), abs(action - adjusted.adjusted),
                    softened.intervened_fraction, adjusted.intervened_fraction, True,
                    softened.adjusted, adjusted.adjusted,
                )

            groups: dict[int, list] = defaultdict(list)
            for e in self.store.events:
                if e.kind in {"instrument_rule", "funding_update"} or e.available_at_ns is None:
                    continue
                groups[e.available_at_ns].append(to_market_event(e, quality_grade=DataQualityGrade(self.quality.grade.value)))
            batches = [(ts, VisibleBatch(ts, tuple(items), True)) for ts, items in sorted(groups.items())]
            frames = []
            frame_batch_groups = _visible_batches_by_decision(
                tuple(batches), tuple(row.decision_ts_ns for row in self.rows),
            )
            for row, frame_batches in zip(self.rows, frame_batch_groups, strict=True):
                frames.append(BarrierFrame(row.decision_ts_ns, frame_batches,
                                          visible_market_features(row, self.features, self.schema, instrument=self.instrument,
                                                                  decision_ts_ns=row.decision_ts_ns),
                                          scheduled_funding_time_features(
                                              row.decision_ts_ns,
                                              self.coverage.settlement_times_ns[self.instrument],
                                          ) if self.feature_profile == "causal_mtf_v3" else {}))
            episode = BarrierEpisode(
                native_session=session,
                native_identity={"dataset_hash": self.store.snapshot_hash, "fold_hash": digest([self.start_ns, self.end_ns]), "config_hash": self.config_hash},
                transition_hash=transition, frames=tuple(frames), scoring_frames=len(frames) - 2,
                plan_action=plan_action, plan_terminal=lambda s: plan(Decimal(0), s, terminal=True),
                executed_exposure=lambda s: signed_quantity(s) * s.mark / self.budget,
                reward_config=RewardConfig(), replay=pending_replay, observation_state=self.observation_state,
            )
            pending_replay = None
            return episode

        bridge = L2FrozenScalerAdapter(self.schema, self.features, self.scaler, expected_scaler_hash=self.scaler.hash)
        return NautilusBarrierEnv(episode_factory=episode_factory,
                                  observation_builder=ObservationBuilder(self.schema, scaler=bridge, expected_scaler_hash=bridge.fingerprint,
                                                                        split="train" if self.split == "train_inner" else self.split),
                                  reward_config=RewardConfig(), max_steps=len(self.rows) - 3, transition_hash=transition)

    def metadata(self) -> dict:
        return {
            "quality": asdict(self.quality), "features": asdict(self.features),
            "feature_profile": self.feature_profile,
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
    factory = HistoricalEnvironmentFactory(**{**env_config["recipe"], "fit_scaler": False, "scaler": Scaler(**raw)})
    return factory.make_env(env_config["transition_hash"], soft_multipliers=soft_risk_multipliers)
