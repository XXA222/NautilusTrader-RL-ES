"""Synthetic contract tests for production assembly; never real-data claims."""
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pytest

from rles_data.core import Availability, CuratedStore, DataError, Event, MINUTE_NS
from rles_data.features import make_feature_row
from rles_data.io import write_curated
from rles_gym_bridge.historical import (HistoricalEnvironmentFactory, _visible_batches_by_decision,
                                        historical_feature_spec, scheduled_funding_time_features)
from rles_risk import SoftRiskParams

pytestmark = pytest.mark.micro_golden


def test_checked_in_ethusdt_real_config_has_valid_soft_risk_domain():
    project = Path(__file__).resolve().parents[1]
    config = json.loads((project / "configs/ethusdt_20240101_real.json").read_text(encoding="utf-8"))
    SoftRiskParams.from_search_dict(config["environment"]["soft_risk"])


def test_actor_account_scale_is_causal_and_legacy_checkpoint_defaults_stay_raw(tmp_path):
    from rles_orchestrator.application import _training_actor_spec
    from rles_orchestrator.workflow import Route

    factory = HistoricalEnvironmentFactory(**recipe(tmp_path))
    config = {"output": str(tmp_path / "run")}
    before = _training_actor_spec(factory, (2,), config, Route.A1)
    future = Event("future-mark", "mark", 25 * MINUTE_NS, 25 * MINUTE_NS, 25 * MINUTE_NS,
                   Availability.NATIVE, "b" * 64, {"instrument": factory.instrument, "price": "1000000"},
                   "synthetic:future-perturbation")
    factory.store = CuratedStore((*factory.store.events, future))
    after = _training_actor_spec(factory, (2,), config, Route.A1)
    assert after.signature == before.signature
    manifest = Path(config["output"]) / "run.json"
    manifest.parent.mkdir()
    manifest.write_text(json.dumps({"schema": "rles.historical-run.v1", "run_id": "legacy", "attempt_id": "legacy",
                                    "recipe_hash": "a" * 64, "run_fingerprint": "b" * 64}))
    legacy = _training_actor_spec(factory, (2,), {**config, "resume_checkpoint": "legacy-boundary"}, Route.A1)
    assert legacy.version == 1 and legacy.input_scale is None


def test_historical_frame_batch_partition_is_ordered_complete_and_linear_windowed():
    batches = ((10, "pre-window"), (20, "first"), (30, "boundary"), (40, "later"))
    frames = _visible_batches_by_decision(batches, (25, 30, 40))

    assert frames == (("pre-window", "first"), ("boundary",), ("later",))
    assert tuple(batch for frame in frames for batch in frame) == tuple(batch for _, batch in batches)


def test_stress_execution_identity_changes_only_the_mode_from_frozen_base(tmp_path):
    environment = recipe(tmp_path)
    base = HistoricalEnvironmentFactory(**environment)
    stress = HistoricalEnvironmentFactory(**{
        **environment, "fit_scaler": False, "scaler": base.scaler, "mode": "stress",
    })

    assert stress.mode == "stress"
    assert stress.config_hash != base.config_hash
    assert stress.base_config_hash == base.config_hash
    assert stress.execution.fingerprint() == base.execution.fingerprint()
    assert stress.scaler.hash == base.scaler.hash


def test_nondefault_reward_scale_is_bound_to_historical_training_identity(tmp_path):
    environment = recipe(tmp_path)
    raw = HistoricalEnvironmentFactory(**environment)
    legacy_default = HistoricalEnvironmentFactory(**environment, reward_scale=1.0)
    scaled = HistoricalEnvironmentFactory(**environment, reward_scale=10_000.0)

    assert raw.config_hash == legacy_default.config_hash
    assert raw.transition_hash == legacy_default.transition_hash
    assert scaled.config_hash != raw.config_hash
    assert scaled.transition_hash != raw.transition_hash
    assert scaled.metadata()["reward_config"]["reward_scale"] == 10_000.0
    assert scaled.make_env()._reward.config.reward_scale == 10_000.0


def test_historical_factory_accepts_v2_profile_and_binds_its_schema(tmp_path):
    factory = HistoricalEnvironmentFactory(**recipe(tmp_path, bar_count=80, start_minute=62),
                                           feature_profile="causal_mtf_v2")

    assert factory.feature_profile == "causal_mtf_v2"
    assert "market.trend_efficiency_15m" in factory.schema.names
    assert "market.trend_efficiency_60m" in factory.schema.names


def test_historical_factory_v3_binds_schedule_time_inputs_separately(tmp_path):
    environment = recipe(tmp_path, bar_count=80, start_minute=62)
    symbol = environment["instrument"]
    environment["coverage_policy"]["known_schedule_times_ns"] = {symbol: (90 * MINUTE_NS,)}
    factory = HistoricalEnvironmentFactory(**environment, feature_profile="causal_mtf_v3")
    v2 = HistoricalEnvironmentFactory(**environment, feature_profile="causal_mtf_v2")

    assert "market.trend_efficiency_60m" in factory.schema.names
    assert "time.funding_settlement_within_24h" in factory.schema.names
    assert "time.funding_settlement_time_to_event_24h" in factory.schema.names
    assert factory.schema.hash != v2.schema.hash
    assert "time.funding_settlement_within_24h" not in v2.schema.names


def test_funding_schedule_countdown_is_bounded_and_uses_strict_future_schedule():
    schedule = (10 * MINUTE_NS, 40 * MINUTE_NS)

    assert scheduled_funding_time_features(9 * MINUTE_NS, schedule) == {
        "funding_settlement_within_24h": 1.0,
        "funding_settlement_time_to_event_24h": pytest.approx(1 / 1440),
    }
    assert scheduled_funding_time_features(10 * MINUTE_NS, schedule)[
        "funding_settlement_time_to_event_24h"
    ] == pytest.approx(30 / 1440)
    assert scheduled_funding_time_features(20 * MINUTE_NS, schedule)[
        "funding_settlement_time_to_event_24h"
    ] == pytest.approx(20 / 1440)
    assert scheduled_funding_time_features(40 * MINUTE_NS, schedule) == {
        "funding_settlement_within_24h": 0.0,
        "funding_settlement_time_to_event_24h": 1.0,
    }


def recipe(tmp_path, *, price_jump=False, bar_count=20, start_minute=6):
    symbol = "ETHUSDT-PERP.SIM"
    rule = {"instrument": symbol, "tick_size": "0.01", "step_size": "0.001", "min_qty": "0.001",
            "min_notional": "1", "maker_fee": "0.001", "taker_fee": "0.001", "initial_margin_rate": "1",
            "maintenance_margin_rate": "0.35", "effective_ns": MINUTE_NS,
            "valid_until_ns": (bar_count + 1) * MINUTE_NS,
            "rule_version": "SYNTHETIC-TEST-ONLY"}
    source = tmp_path / "synthetic-rule.json"
    source.write_text(json.dumps(rule), encoding="utf-8")
    fingerprint = sha256(source.read_bytes()).hexdigest()
    rule_event = Event("rule", "instrument_rule", MINUTE_NS, MINUTE_NS, MINUTE_NS, Availability.NATIVE,
                       fingerprint, rule, "synthetic:factory-test-rule")
    rules = tmp_path / "rules.json"
    rules.write_text(json.dumps({"sources": [{"path": str(source), "sha256": fingerprint, "provenance": "synthetic:test"}],
                                  "events": [asdict(rule_event)]}), encoding="utf-8")
    events = []
    for i in range(1, bar_count + 1):
        timestamp = i * MINUTE_NS
        price = "1040" if price_jump and i >= 8 else "1000"
        for kind, payload in {
            "ohlcv_1m": {"open": price, "high": price, "low": price, "close": price, "volume": "1000"},
            "mark": {"price": price}, "index": {"price": price},
        }.items():
            events.append(Event(f"{kind}-{i}", kind, timestamp, timestamp, timestamp, Availability.NATIVE,
                                "a" * 64, {"instrument": symbol, **payload}, f"synthetic:{kind}"))
    for minute in range(1, bar_count + 1, 20):
        timestamp = minute * MINUTE_NS
        events.append(Event(f"rate-{minute}", "funding_update", timestamp, timestamp, timestamp,
                            Availability.NATIVE, "a" * 64,
                            {"instrument": symbol, "rate": "0.001"}, "synthetic:funding_update"))
    market = tmp_path / "market.parquet"
    write_curated(market, CuratedStore(events))
    latency = {"model_id": "base", "decision_ns": 1, "submit_ns": 1, "venue_ns": 1}
    execution = {"schema_version": "0.4.0", "deadband": "0.01", "pending_order_policy": "cancel_replace",
                 "order_type": "MARKET", "tif": "GTC", "terminal_policy": "force_flatten_with_cost",
                 "base_latency": latency, "stress_latency": {**latency, "model_id": "stress", "venue_ns": 2},
                 "base_fill": {"model_id": "base", "slippage_bps": "1", "data_granularity": "bar"},
                 "stress_fill": {"model_id": "stress", "slippage_bps": "2", "data_granularity": "bar"}}
    return dict(market_path=market, rules_path=rules, instrument=symbol,
                start_ns=start_minute * MINUTE_NS, end_ns=(bar_count - 1) * MINUTE_NS,
                coverage_policy={"maximum_age_ns": {"mark": MINUTE_NS, "index": MINUTE_NS, "funding_update": 30 * MINUTE_NS},
                                 "settlement_times_ns": {symbol: ()}, "provenance": "synthetic:factory-test"},
                execution=execution, starting_balance="10000", risk_budget_notional="1000",
                hard_risk={"max_notional": "1000", "max_leverage": "1", "margin_buffer_fraction": "0.1",
                           "min_equity": "1", "incomplete_data_max_leverage": "1", "require_mark": True, "require_funding": True},
                soft_risk={"target_scale": "1", "max_turnover_fraction": "1", "drawdown_throttle": "0", "funding_sensitivity": "0"},
                fit_scaler=True)


def test_causal_mtf_profile_adds_regime_features_without_future_leakage():
    maximum_age = {"mark": 60 * MINUTE_NS, "index": 60 * MINUTE_NS,
                   "funding_update": 60 * MINUTE_NS}
    legacy = historical_feature_spec(maximum_age)
    profile_v1 = historical_feature_spec(maximum_age, profile="causal_mtf_v1")
    profile = historical_feature_spec(maximum_age, profile="causal_mtf_v2")
    assert profile.fields[:len(legacy.fields)] == legacy.fields
    assert profile.fields[:len(profile_v1.fields)] == profile_v1.fields
    assert tuple(field.name for field in profile.fields[len(legacy.fields):len(profile_v1.fields)]) == (
        "return_5m", "return_15m", "return_60m", "volatility_15m", "volatility_60m",
    )
    assert tuple(field.name for field in profile.fields[len(profile_v1.fields):]) == (
        "trend_efficiency_15m", "trend_efficiency_60m",
    )
    assert profile.max_lookback_ns == 61 * MINUTE_NS

    symbol = "ETHUSDT-PERP.SIM"
    events = []
    closes = []
    for minute in range(1, 62):
        timestamp = minute * MINUTE_NS
        close = float(100 + minute)
        closes.append(close)
        for kind, payload in {
            "ohlcv_1m": {"open": str(close), "high": str(close), "low": str(close),
                         "close": str(close), "volume": "10"},
            "mark": {"price": str(close)},
            "index": {"price": str(close)},
        }.items():
            events.append(Event(f"{kind}-{minute}", kind, timestamp, timestamp, timestamp,
                                Availability.NATIVE, "a" * 64, {"instrument": symbol, **payload},
                                f"synthetic:causal-mtf:{kind}"))
    events.append(Event("funding", "funding_update", MINUTE_NS, MINUTE_NS, MINUTE_NS,
                        Availability.NATIVE, "a" * 64,
                        {"instrument": symbol, "rate": "0.001"}, "synthetic:causal-mtf:funding"))

    decision = 61 * MINUTE_NS
    store = CuratedStore(events)
    row = make_feature_row(store, profile, decision, instrument=symbol)
    values = dict(zip((field.name for field in profile.fields), row.values))
    assert values["return_60m"] == pytest.approx(closes[-1] / closes[0] - 1)
    assert values["trend_efficiency_15m"] == pytest.approx(1.0)
    assert values["trend_efficiency_60m"] == pytest.approx(1.0)
    assert row.available_at_ns <= decision

    future = Event("future-bar", "ohlcv_1m", 62 * MINUTE_NS, 62 * MINUTE_NS, 62 * MINUTE_NS,
                   Availability.NATIVE, "b" * 64,
                   {"instrument": symbol, "open": "1000000", "high": "1000000", "low": "1000000",
                    "close": "1000000", "volume": "1000000"}, "synthetic:future-perturbation")
    extended_store = CuratedStore((*events, future))
    extended_row = make_feature_row(extended_store, profile, decision, instrument=symbol)
    assert extended_row.values == row.values


def test_trend_efficiency_encodes_chop_is_bounded_and_rejects_gaps():
    from math import isfinite
    from rles_data.core import FeatureField, FeatureSpec

    symbol = "ETHUSDT-PERP.SIM"
    events = []
    for minute in range(1, 62):
        close = 100.0 if minute < 47 or minute % 2 == 0 else 101.0
        timestamp = minute * MINUTE_NS
        for kind, payload in {
            "ohlcv_1m": {"open": str(close), "high": str(close), "low": str(close),
                         "close": str(close), "volume": "1"},
            "mark": {"price": str(close)}, "index": {"price": str(close)},
        }.items():
            events.append(Event(f"{kind}-{minute}", kind, timestamp, timestamp, timestamp,
                                Availability.NATIVE, "c" * 64, {"instrument": symbol, **payload},
                                f"synthetic:trend-efficiency:{kind}"))
    spec = FeatureSpec("0.4.0", (
        FeatureField("trend_efficiency_15m", "float64", "ratio", "ohlcv_1m", 15, 1,
                     "trend_efficiency"),
    ))
    decision = 61 * MINUTE_NS
    row = make_feature_row(CuratedStore(events), spec, decision, instrument=symbol)
    efficiency = row.values[0]
    assert isfinite(efficiency) and -1 <= efficiency <= 1
    assert efficiency == pytest.approx(1 / 15, rel=1e-4)

    gap_events = [event for event in events if event.event_id != "ohlcv_1m-55"]
    with pytest.raises(DataError, match="gap|not enough"):
        make_feature_row(CuratedStore(gap_events), spec, decision, instrument=symbol)


def test_production_factory_uses_native_models_full_observations_and_terminal_costs(tmp_path):
    factory = HistoricalEnvironmentFactory(**recipe(tmp_path))
    env = factory.make_env()
    try:
        obs, info = env.reset(seed=31)
        assert obs.shape == (23,) and np.isfinite(obs).all()
        assert float(info["initial_equity"]) == 10000
        while True:
            obs, _, terminated, truncated, info = env.step(np.array([0.5], dtype=np.float32))
            assert info["risk_intervention_components_known"] is True
            assert info["risk_soft_adjustment"] == pytest.approx(0.0)
            assert info["risk_hard_adjustment"] == pytest.approx(0.0)
            if terminated or truncated:
                break
        final = factory.snapshots[-1]
        assert info["terminal_settled"] and final.positions == final.orders == ()
        assert final.equity < 10000
        observed_positions = [row for snapshot in factory.snapshots for row in snapshot.positions]
        assert observed_positions
        assert all(not ({"events", "adjustments", "commissions", "venue_order_ids", "trade_ids"}
                        & set(row)) for row in observed_positions)
        assert {f["order_side"] for f in final.fills} == {"BUY", "SELL"}
        assert all(float(f["last_px"]) > 1000 for f in final.fills if f["order_side"] == "BUY")
        assert all(float(f["last_px"]) < 1000 for f in final.fills if f["order_side"] == "SELL")
        assert obs[factory.schema.names.index("execution.prev_raw_target")] == np.float32(0.5)
        from rles_research.native_metrics import native_report

        report = native_report(factory.snapshots, fill_references=factory.fill_references, sample_interval_ns=MINUTE_NS)
        metrics = report["metrics"]
        assert metrics["total_fee"] > 0 and metrics["total_slippage_cost"] > 0
        assert metrics["total_funding"] == 0
        assert metrics["net_return"] == pytest.approx(float(final.equity / 10000 - 1))
        assert report["terminal_flat"]
        assert report["liquidations"] == 0
        assert report["liquidations_observed"] is True
    finally:
        env.close()


def test_soft_funding_adjustment_is_not_reported_as_hard_risk_violation(tmp_path):
    config = recipe(tmp_path)
    config["soft_risk"] = {**config["soft_risk"], "funding_sensitivity": "1"}
    factory = HistoricalEnvironmentFactory(**config)
    env = factory.make_env()
    try:
        env.reset(seed=31)
        _observation, _reward, terminated, truncated, info = env.step(np.array([0.5], dtype=np.float32))
        assert not terminated and not truncated
        assert info["risk_intervention_components_known"] is True
        assert info["risk_soft_adjustment"] == pytest.approx(0.0005)
        assert info["risk_hard_adjustment"] == pytest.approx(0.0)
        assert info["risk_guard_intervention"] == pytest.approx(info["risk_soft_adjustment"])
    finally:
        env.close()


@pytest.mark.parametrize("requested, expected_plan", ((0.1, "NOOP"), (0.4, "SUBMIT")))
def test_native_execution_distinguishes_scaled_deadband_from_risk_target(tmp_path, requested, expected_plan):
    config = recipe(tmp_path)
    config["execution"]["deadband"] = "0.05"
    factory = HistoricalEnvironmentFactory(**config)
    env = factory.make_env(soft_multipliers={
        "exposure_multiplier": 0.5, "volatility_throttle": 0.5, "drawdown_throttle": 1.0,
    })
    try:
        env.reset(seed=31)
        _observation, _reward, _terminated, _truncated, info = env.step(np.array([requested], dtype=np.float32))
        assert info["risk_soft_target"] == pytest.approx(requested * 0.25)
        assert info["risk_adjusted_target"] == pytest.approx(requested * 0.25)
        assert info["risk_hard_adjustment"] == 0.0
        assert info["execution_plan_action"] == expected_plan
        assert info["execution_plan_reason"] == ("deadband" if expected_plan == "NOOP" else "rebalance")
        if expected_plan == "NOOP":
            assert info["policy_executed_action_pre_terminal"] == 0.0
            assert factory.snapshots[-1].fills == ()
        else:
            assert info["policy_executed_action_pre_terminal"] == pytest.approx(0.1)
            assert factory.snapshots[-1].fills
    finally:
        env.close()


def test_native_hard_cap_price_drift_reduces_inside_deadband(tmp_path):
    from decimal import Decimal

    config = recipe(tmp_path, price_jump=True)
    config["execution"]["deadband"] = "0.05"
    config["hard_risk"]["max_notional"] = "500"
    factory = HistoricalEnvironmentFactory(**config, compact_native_history=False)
    env = factory.make_env()
    try:
        env.reset(seed=31)
        observed_reduction = False
        while True:
            _, _, terminated, truncated, info = env.step(np.array([1.0], dtype=np.float32))
            if info["execution_plan_reason"] == "hard_risk_reduce":
                observed_reduction = True
                assert info["execution_plan_action"] == "SUBMIT"
                assert info["risk_adjusted_target"] == 0.5
                snapshot = factory.snapshots[-1]
                quantity = Decimal(snapshot.positions[0]["signed_qty"])
                assert 0 < quantity * snapshot.mark <= 500
                assert snapshot.fills[-1]["order_side"] == "SELL"
            if terminated or truncated:
                break
        assert observed_reduction
        terminal = factory.snapshots[-1]
        assert terminal.positions == terminal.orders == ()
        assert sum(order["is_reduce_only"] for order in terminal.all_orders) == 2
    finally:
        env.close()


def test_native_cache_compaction_preserves_historical_accounting(tmp_path):
    from rles_research.native_metrics import native_report

    config = recipe(tmp_path)

    def run(compact: bool):
        factory = HistoricalEnvironmentFactory(**config, compact_native_history=compact)
        env = factory.make_env()
        try:
            observation, _ = env.reset(seed=91)
            while True:
                observation, _, terminated, truncated, _ = env.step(
                    np.asarray([0.5], dtype=np.float32),
                )
                if terminated or truncated:
                    break
            report = native_report(
                factory.snapshots, fill_references=factory.fill_references,
                sample_interval_ns=MINUTE_NS,
            )
            fill_economics = sorted(
                (row["order_side"], row["last_qty"], row["last_px"], row["commission"])
                for row in report["fills"]
            )
            position_economics = [
                sorted((row["side"], row["signed_qty"], row["quantity"],
                        row["avg_px_open"], row["avg_px_close"], row["realized_pnl"])
                       for row in snapshot.positions)
                for snapshot in factory.snapshots
            ]
            funding_economics = sorted(
                (row["pnl_change"], row["currency"])
                for row in report["funding"]
            )
            return (
                [snapshot.equity for snapshot in factory.snapshots],
                position_economics,
                [snapshot.orders for snapshot in factory.snapshots],
                report["metrics"], fill_economics, funding_economics,
                report["liquidations"], report["terminal_flat"],
            )
        finally:
            env.close()

    assert run(compact=True) == run(compact=False)


def test_native_observation_exposes_pending_order_direction_and_size(tmp_path):
    from decimal import Decimal
    from types import SimpleNamespace

    factory = HistoricalEnvironmentFactory(**recipe(tmp_path))
    orders = (
        {"side": "BUY", "leaves_qty": "0.5"},
        {"side": "SELL", "leaves_qty": "0.25"},
    )
    all_orders = (
        {"quantity": "1", "filled_qty": "0.5"},
        {"quantity": "1", "filled_qty": "0.5"},
    )
    snapshot = SimpleNamespace(
        equity=Decimal("10000"), currency="USDT", mark=Decimal("1000"), positions=(),
        orders=orders, all_orders=all_orders, margins=(),
        account_events=({"balances": ({"currency": "USDT", "total": "10000 USDT", "free": "9000 USDT"},)},),
    )
    state = factory.observation_state(snapshot)
    assert state["execution"]["pending_order_count"] == 2
    assert state["execution"]["pending_signed_exposure"] == pytest.approx(0.25)
    assert state["execution"]["pending_gross_exposure"] == pytest.approx(0.75)
    assert state["execution"]["fill_rate"] == pytest.approx(0.5)

    invalid = SimpleNamespace(**{**vars(snapshot), "orders": ({"side": "UNKNOWN", "leaves_qty": "0.1"},)})
    with pytest.raises(DataError, match="pending order side/quantity"):
        factory.observation_state(invalid)


def test_production_factory_rejects_missing_rules_and_evaluation_fit(tmp_path):
    config = recipe(tmp_path)
    with pytest.raises(DataError, match="only Train-inner"):
        HistoricalEnvironmentFactory(**{**config, "split": "test"})
    with pytest.raises(DataError, match="rules unavailable"):
        HistoricalEnvironmentFactory(**{**config, "rules_path": tmp_path / "missing.json"})


def test_market_audit_reports_economic_coverage_without_granting_missing_rules(tmp_path):
    from rles_orchestrator.application import audit_market

    config = recipe(tmp_path)
    result = audit_market({"environment": config, "evaluation": {
        "start_ns": config["start_ns"], "end_ns": config["end_ns"],
        "coverage_policy": config["coverage_policy"],
    }})
    report = result["splits"]["train"]
    assert report["grade"] == "DQ-C"
    assert "instrument_rule_coverage_gap" in report["findings"]
    assert set(report["verified_coverage"]) >= {"mark", "index", "funding_update"}


@pytest.mark.parametrize("algorithm", ("PPO", "SAC", "TD3", "DDPG"))
def test_application_rl_train_bundle_load_and_disjoint_native_backtest(tmp_path, algorithm):
    from rles_orchestrator.application import backtest, load_config, train

    config = recipe(tmp_path)
    config["end_ns"] = 12 * MINUTE_NS
    config["market_path"], config["rules_path"] = str(config["market_path"]), str(config["rules_path"])
    run = {"environment": config, "evaluation": {"start_ns": 12 * MINUTE_NS, "end_ns": 20 * MINUTE_NS,
                                                   "coverage_policy": config["coverage_policy"]},
           "output": str(tmp_path / "run"), "algorithm": algorithm, "steps": 4, "hidden_layers": [2],
           "hyperparameters": {"n_steps": 2, "batch_size": 2, "n_epochs": 1} if algorithm == "PPO" else
                                {"learning_starts": 2, "buffer_size": 32, "batch_size": 2, "train_freq": 1, "gradient_steps": 1},
           "normalize_observation": True}
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(run), encoding="utf-8")
    resolved = load_config(config_path)
    result = train(resolved)
    restored = backtest(resolved, result["output"])
    assert result["terminal_flat"] and restored["terminal_flat"]
    assert result["metrics"] == restored["metrics"]
    assert restored["market_data_provenance"] == [
        "synthetic:funding_update", "synthetic:index", "synthetic:mark", "synthetic:ohlcv_1m",
    ]
    assert restored["rule_provenance"] == ["synthetic:factory-test-rule"]
    if algorithm == "PPO":
        bundle = tmp_path / "run" / "bundle.json"
        original = bundle.read_bytes()
        bundle.write_bytes(original + b"\n")
        with pytest.raises(DataError, match="integrity mismatch"):
            backtest(resolved, result["output"])
        bundle.write_bytes(original)


@pytest.mark.parametrize("route", ("A1", "A4"))
def test_application_es_and_hybrid_use_installed_historical_spawn_factory(tmp_path, route, monkeypatch):
    from rles_orchestrator.application import backtest, diagnose_train, train

    config = recipe(tmp_path)
    config["end_ns"] = 12 * MINUTE_NS
    config["market_path"], config["rules_path"] = str(config["market_path"]), str(config["rules_path"])
    run = {"environment": config, "evaluation": {"start_ns": 12 * MINUTE_NS, "end_ns": 20 * MINUTE_NS,
                                                   "coverage_policy": config["coverage_policy"]},
           "output": str(tmp_path / "run"), "algorithm": "PPO", "route": route,
           "steps": 4, "hidden_layers": [2], "hyperparameters": {"n_steps": 2, "batch_size": 2, "n_epochs": 1},
           "distillation": {"epochs": 120, "tolerance": 0.12},
           "checkpoint_interval_steps": 2}
    if route == "A4":
        from rles_rl_sb3.provider import RLProvider

        original_train = RLProvider.train
        calls = 0

        def interrupt_after_first_durable_chunk(provider, total_timesteps, *, resumed=False, callback=None):
            nonlocal calls
            if calls:
                raise RuntimeError("injected A4 interruption after R1 checkpoint")
            calls += 1
            return original_train(provider, total_timesteps, resumed=resumed, callback=callback)

        monkeypatch.setattr(RLProvider, "train", interrupt_after_first_durable_chunk)
        with pytest.raises(RuntimeError, match="injected A4 interruption"):
            train(run)
        checkpoint = Path(run["output"]) / "R1-checkpoints" / "step-000000000002"
        assert (checkpoint / "COMPLETED").is_file()
        monkeypatch.setattr(RLProvider, "train", original_train)
        run["resume_checkpoint"] = str(checkpoint)
    result = train(run)
    restored = backtest(run, result["output"])
    assert result["metrics"] == restored["metrics"]
    assert restored["terminal_flat"]
    bundle_root = Path(result["output"])
    bundle_files_before = {str(path.relative_to(bundle_root)): sha256(path.read_bytes()).hexdigest()
                           for path in bundle_root.rglob("*") if path.is_file()}
    with monkeypatch.context() as diagnostic_patch:
        def forbidden(*_args, **_kwargs):
            pytest.fail("Train diagnostics must not refit scaling or construct an evaluation")
        diagnostic_patch.setattr("rles_gym_bridge.historical.fit_train", forbidden)
        diagnostic_patch.setattr("rles_orchestrator.application._evaluation_factory", forbidden)
        diagnostic_patch.setattr("rles_orchestrator.application.ResearchLedger", forbidden)
        diagnostic = diagnose_train({**run, "evaluation": None}, bundle_root)
        assert diagnostic["diagnostic_scope"]["split"] == "train_inner"
        assert diagnostic["diagnostic_scope"]["evaluation_queries"] == 0
        assert diagnostic["diagnostic_scope"]["end_ns"] == 12 * MINUTE_NS
        assert diagnostic["action_diagnostics"]["count"] > 0
        assert diagnostic["execution_attribution"]["components_known"] is True
        assert diagnostic["terminal_flat"] is True
        with pytest.raises(DataError, match="may not replace"):
            diagnose_train({**run, "environment": {**config, "end_ns": 20 * MINUTE_NS}}, bundle_root)
    assert bundle_files_before == {
        str(path.relative_to(bundle_root)): sha256(path.read_bytes()).hexdigest()
        for path in bundle_root.rglob("*") if path.is_file()
    }
    from os import name as os_name

    resource_paths = list(Path(result["output"]).glob("E1-resources-*.json"))
    assert resource_paths
    resource_report = json.loads(resource_paths[0].read_text())
    assert resource_report["configured_policy"]["enforcement"] == (
        "windows_job_commit_memory" if os_name == "nt" else "quota_only")
    if os_name == "nt":
        assert resource_report["workers"][0]["active_processes"] == 0
        assert resource_report["workers"][0]["peak_process_commit_bytes"] > 0
    manifest = json.loads((Path(result["output"]) / "run.json").read_text())
    assert manifest["actor_spec"]["version"] == 2
    scales = manifest["actor_spec"]["input_scale"]
    names = manifest["actor_spec"]["observation_names"]
    assert scales[names.index("account.cash")] == pytest.approx(0.0001)
    assert scales[names.index("account.position_quantity")] == pytest.approx(1.0)
    if route == "A4":
        assert manifest["canonical_ppo_output"] is True
        transfer = json.loads((Path(result["output"]) / "R1-transfer.json").read_text())["result"]
        assert transfer["method"] == "DIRECT"
        assert transfer["parity"]["max_abs_error"] < 1e-5
    if route == "A4":
        from rles_research.ledger import ResearchLedger

        root = Path(result["output"])
        receipts = ResearchLedger(root / "research.sqlite").evidence_summary(
            json.loads((root / "run.json").read_text(encoding="utf-8"))["run_id"],
        )["stage_receipts"]
        assert {item["stage"] for item in receipts} == {"E1", "R1", "E2"}
        assert sum(json.loads(item["usage_json"]).get("lane_requests", 0) for item in receipts) == 8
        assert len(list(root.glob("FAILED-*.json"))) == 1
        assert (root / "R1-transfer.json").is_file()
        checkpoint_dirs = sorted((Path(result["output"]) / "R1-checkpoints").glob("step-*"))
        assert [path.name for path in checkpoint_dirs] == ["step-000000000002", "step-000000000004"]
        for checkpoint in checkpoint_dirs:
            manifest = json.loads((checkpoint / "manifest.json").read_text(encoding="utf-8"))
            assert manifest["schema"] == "rles.l4.stage.v4"
            assert "episode_replays.json" in manifest["files"]
            assert (checkpoint / "COMPLETED").is_file()
            replays = json.loads((checkpoint / "episode_replays.json").read_text(encoding="utf-8"))
            assert len(replays) == 1
            assert replays[0]["actions"]
            assert len(replays[0]["actions"]) <= int(checkpoint.name.rsplit("-", 1)[1])
            assert replays[0]["boundary_token"] == manifest["boundary"]["env_tokens"][0]


@pytest.mark.parametrize("route", ("A0", "A2", "A3"))
def test_application_routes_resume_partial_r1_from_replay_checkpoint(tmp_path, monkeypatch, route):
    from rles_orchestrator.application import train
    from rles_rl_sb3.provider import RLProvider

    environment = recipe(tmp_path)
    environment["end_ns"] = 12 * MINUTE_NS
    environment["market_path"], environment["rules_path"] = (
        str(environment["market_path"]), str(environment["rules_path"]),
    )
    run = {"environment": environment,
           "evaluation": {"start_ns": 12 * MINUTE_NS, "end_ns": 20 * MINUTE_NS,
                          "coverage_policy": environment["coverage_policy"]},
           "output": str(tmp_path / f"resume-{route.lower()}-run"), "algorithm": "PPO", "route": route,
           "steps": 4, "hidden_layers": [2],
           "hyperparameters": {"n_steps": 2, "batch_size": 2, "n_epochs": 1},
           "checkpoint_interval_steps": 2}
    if route in {"A2", "A3"}:
        run.update(population=4, generations=1,
                   distillation={"epochs": 120, "tolerance": 0.12})
    original_train = RLProvider.train
    calls = 0

    def stop_after_first_boundary(provider, total_timesteps, *, resumed=False, callback=None):
        nonlocal calls
        if calls:
            raise RuntimeError("injected interruption after durable R1 boundary")
        calls += 1
        return original_train(provider, total_timesteps, resumed=resumed, callback=callback)

    monkeypatch.setattr(RLProvider, "train", stop_after_first_boundary)
    with pytest.raises(RuntimeError, match="injected interruption"):
        train(run)
    checkpoint = Path(run["output"]) / "R1-checkpoints" / "step-000000000002"
    assert (checkpoint / "COMPLETED").is_file()
    assert (Path(run["output"]) / "FAILED.json").is_file()

    monkeypatch.setattr(RLProvider, "train", original_train)
    changed_recipe = dict(run, seed=run.get("seed", 31) + 1,
                          resume_checkpoint=str(checkpoint))
    with pytest.raises(DataError, match="configuration differs"):
        train(changed_recipe)
    assert (Path(run["output"]) / "FAILED.json").is_file()
    run["resume_checkpoint"] = str(checkpoint)
    result = train(run)
    root = Path(result["output"])
    assert (root / "COMPLETED.json").is_file()
    assert (root / "FAILED.json").is_file() is False
    assert list(root.glob("FAILED-*.json"))
    bundle = json.loads((root / "bundle.json").read_text(encoding="utf-8"))
    assert [item["stage"] for item in bundle["lineage"]] == {
        "A0": ["R1"], "A2": ["R1", "E2"], "A3": ["E1", "R1"],
    }[route]
    rl_entry = next(item for item in bundle["lineage"] if item["stage"] == "R1")
    assert rl_entry["usage"]["rl_steps"] == 4
    assert (root / "R1-checkpoints" / "step-000000000004" / "COMPLETED").is_file()


@pytest.mark.parametrize("route", ("A1", "A3", "A4"))
def test_application_routes_resume_e1_from_completed_generation(tmp_path, monkeypatch, route):
    from rles_es_evotorch.provider import EvoTorchProvider
    from rles_orchestrator.application import train

    environment = recipe(tmp_path)
    environment["end_ns"] = 12 * MINUTE_NS
    environment["market_path"], environment["rules_path"] = (
        str(environment["market_path"]), str(environment["rules_path"]),
    )
    run = {"environment": environment,
           "evaluation": {"start_ns": 12 * MINUTE_NS, "end_ns": 20 * MINUTE_NS,
                          "coverage_policy": environment["coverage_policy"]},
           "output": str(tmp_path / f"resume-{route.lower()}-e1-run"), "algorithm": "PPO", "route": route,
           "steps": 4, "population": 4, "generations": 2, "hidden_layers": [2],
           "hyperparameters": {"n_steps": 2, "batch_size": 2, "n_epochs": 1},
           "checkpoint_interval_steps": 2}
    if route in {"A3", "A4"}:
        run["distillation"] = {"epochs": 120, "tolerance": 0.12}

    original_continue = EvoTorchProvider.continue_evolution
    calls = 0

    def stop_before_second_generation(provider, *args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("injected E1 interruption after generation one")
        return original_continue(provider, *args, **kwargs)

    monkeypatch.setattr(EvoTorchProvider, "continue_evolution", stop_before_second_generation)
    with pytest.raises(RuntimeError, match="injected E1 interruption"):
        train(run)

    root = Path(run["output"])
    checkpoint = root / "E1" / "generation-00000001" / json.loads(
        (root / "run.json").read_text(encoding="utf-8"),
    )["run_id"]
    manifest = json.loads((checkpoint / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["generation"] == 1
    assert (root / "FAILED.json").is_file()

    monkeypatch.setattr(EvoTorchProvider, "continue_evolution", original_continue)
    run["resume_checkpoint"] = str(checkpoint)
    result = train(run)
    assert (root / "COMPLETED.json").is_file()
    assert not (root / "FAILED.json").exists()
    assert list(root.glob("FAILED-*.json"))
    assert (root / "E1" / "generation-00000002" / manifest["attempt_id"] / "COMPLETED").is_file()
    assert result["route"] == route


@pytest.mark.parametrize("route", ("A2", "A4"))
def test_application_routes_resume_e2_and_revalidate_existing_r1_actor(tmp_path, monkeypatch, route):
    from rles_es_evotorch.provider import EvoTorchProvider
    from rles_orchestrator.application import backtest, train

    environment = recipe(tmp_path)
    environment["end_ns"] = 12 * MINUTE_NS
    environment["market_path"], environment["rules_path"] = (
        str(environment["market_path"]), str(environment["rules_path"]),
    )
    run = {"environment": environment,
           "evaluation": {"start_ns": 12 * MINUTE_NS, "end_ns": 20 * MINUTE_NS,
                          "coverage_policy": environment["coverage_policy"]},
           "output": str(tmp_path / f"resume-{route.lower()}-e2-run"), "algorithm": "PPO", "route": route,
           "steps": 4, "population": 4, "generations": 2, "hidden_layers": [2],
           "hyperparameters": {"n_steps": 2, "batch_size": 2, "n_epochs": 1},
           "checkpoint_interval_steps": 2}
    if route == "A4":
        run["distillation"] = {"epochs": 120, "tolerance": 0.12}

    original_continue = EvoTorchProvider.continue_evolution
    e2_calls = 0

    def stop_during_second_e2_generation(provider, *args, **kwargs):
        nonlocal e2_calls
        if provider.config.algorithm == "cmaes":
            e2_calls += 1
            if e2_calls == 2:
                raise RuntimeError("injected E2 interruption after generation one")
        return original_continue(provider, *args, **kwargs)

    monkeypatch.setattr(EvoTorchProvider, "continue_evolution", stop_during_second_e2_generation)
    with pytest.raises(RuntimeError, match="injected E2 interruption"):
        train(run)

    root = Path(run["output"])
    run_id = json.loads((root / "run.json").read_text(encoding="utf-8"))["run_id"]
    checkpoint = root / "E2" / "generation-00000001" / run_id
    manifest = json.loads((checkpoint / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["generation"] == 1
    actor_hash = (root / "r1" / "COMPLETED").read_text(encoding="ascii")
    assert (root / "FAILED.json").is_file()

    monkeypatch.setattr(EvoTorchProvider, "continue_evolution", original_continue)
    run["resume_checkpoint"] = str(checkpoint)
    result = train(run)
    report = backtest(run, result["output"])
    assert report["terminal_flat"]
    assert (root / "r1" / "COMPLETED").read_text(encoding="ascii") == actor_hash
    assert (root / "E2" / "generation-00000002" / run_id / "COMPLETED").is_file()
    assert (root / "COMPLETED.json").is_file()


def test_application_finalizes_full_budget_r1_checkpoint_without_retraining(tmp_path, monkeypatch):
    from rles_orchestrator.application import train
    from rles_rl_sb3.provider import RLProvider

    environment = recipe(tmp_path)
    environment["end_ns"] = 12 * MINUTE_NS
    environment["market_path"], environment["rules_path"] = (
        str(environment["market_path"]), str(environment["rules_path"]),
    )
    run = {"environment": environment,
           "evaluation": {"start_ns": 12 * MINUTE_NS, "end_ns": 20 * MINUTE_NS,
                          "coverage_policy": environment["coverage_policy"]},
           "output": str(tmp_path / "finalize-run"), "algorithm": "PPO", "route": "A0",
           "steps": 4, "hidden_layers": [2],
           "hyperparameters": {"n_steps": 2, "batch_size": 2, "n_epochs": 1},
           "checkpoint_interval_steps": 2}
    original_export = RLProvider.export_inference

    def interrupt_before_inference_export(_provider, _directory, _fixed_observations):
        raise RuntimeError("injected interruption after full-budget checkpoint")

    monkeypatch.setattr(RLProvider, "export_inference", interrupt_before_inference_export)
    with pytest.raises(RuntimeError, match="full-budget checkpoint"):
        train(run)
    checkpoint = Path(run["output"]) / "R1-checkpoints" / "step-000000000004"
    assert (checkpoint / "COMPLETED").is_file()

    monkeypatch.setattr(RLProvider, "export_inference", original_export)

    def reject_retraining(*_args, **_kwargs):
        raise AssertionError("finalize-only recovery must not train or charge new steps")

    monkeypatch.setattr(RLProvider, "train", reject_retraining)
    run["resume_checkpoint"] = str(checkpoint)
    result = train(run)
    root = Path(result["output"])
    bundle = json.loads((root / "bundle.json").read_text(encoding="utf-8"))
    assert bundle["lineage"][0]["usage"]["rl_steps"] == 4
    assert (root / "COMPLETED.json").is_file()
    assert list(root.glob("FAILED-*.json"))
