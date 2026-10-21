"""Synthetic contract tests for production assembly; never real-data claims."""
from bisect import bisect_left, bisect_right
from dataclasses import asdict
from decimal import Decimal
from hashlib import sha256
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from rles_data.core import Availability, CuratedStore, DataError, Event, MINUTE_NS
from rles_data.features import make_feature_row
from rles_data.io import write_curated
from rles_data.contracts import to_market_event
from rles_gym_bridge.historical import (HistoricalEnvironmentFactory, _NativePositionClock,
                                        _visible_batches_by_decision,
                                        historical_feature_spec, partition_decision_lanes,
                                        scheduled_funding_time_features)
from rles_gym_bridge.episode import BarrierFrame, EpisodeReplay
from rles_nautilus_adapter.session import VisibleBatch
from rles_contracts.dto import DataQualityGrade
from rles_risk import SoftRiskParams

pytestmark = pytest.mark.micro_golden


def test_checked_in_ethusdt_real_config_has_valid_soft_risk_domain():
    project = Path(__file__).resolve().parents[1]
    config = json.loads((project / "configs/ethusdt_20240101_real.json").read_text(encoding="utf-8"))
    SoftRiskParams.from_search_dict(config["environment"]["soft_risk"])


def test_factory_rejects_instrument_rule_change_inside_native_session(tmp_path):
    config = recipe(tmp_path, bar_count=20, start_minute=6)
    rules_path = Path(config["rules_path"])
    document = json.loads(rules_path.read_text(encoding="utf-8"))
    original_source = Path(document["sources"][0]["path"])
    first = json.loads(original_source.read_text(encoding="utf-8"))
    first["valid_until_ns"] = 10 * MINUTE_NS
    original_source.write_text(json.dumps(first), encoding="utf-8")
    first_hash = sha256(original_source.read_bytes()).hexdigest()
    document["sources"][0]["sha256"] = first_hash
    first_event = document["events"][0]
    first_event["source_hash"] = first_hash
    first_event["payload"] = first

    second = {
        **first,
        "initial_margin_rate": "0.9",
        "maintenance_margin_rate": "0.3",
        "effective_ns": 10 * MINUTE_NS,
        "valid_until_ns": 21 * MINUTE_NS,
        "rule_version": "SYNTHETIC-TEST-ONLY-v2",
    }
    second_source = tmp_path / "synthetic-rule-v2.json"
    second_source.write_text(json.dumps(second), encoding="utf-8")
    second_hash = sha256(second_source.read_bytes()).hexdigest()
    document["sources"].append({
        "path": str(second_source), "sha256": second_hash,
        "provenance": "synthetic:test-rule-update",
    })
    document["events"].append(asdict(Event(
        "rule-v2", "instrument_rule", 10 * MINUTE_NS, 10 * MINUTE_NS, 10 * MINUTE_NS,
        Availability.NATIVE, second_hash, second, "synthetic:test-rule-update",
    )))
    rules_path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(DataError, match="split episode at historical instrument-rule changes"):
        HistoricalEnvironmentFactory(**config)


def test_factory_applies_configured_high_leverage_to_native_account(tmp_path):
    from nautilus_trader.model import AccountId, InstrumentId

    config = recipe(tmp_path, initial_margin_rate="0.8")
    one_x = HistoricalEnvironmentFactory(**config)
    config["hard_risk"] = {
        **config["hard_risk"], "max_notional": "30000", "max_leverage": "2",
        "margin_buffer_fraction": "0",
    }
    config["risk_budget_notional"] = "30000"
    factory = HistoricalEnvironmentFactory(**config)
    assert factory.quality.grade.value in {"DQ-A", "DQ-B"}
    assert factory.config_hash != one_x.config_hash
    assert factory.transition_hash != one_x.transition_hash
    env = factory.make_env()
    try:
        env.reset(seed=13)
        account = env._episode._session._engine.portfolio.account(
            account_id=AccountId.from_str("SIM-001"),
        )
        instrument = env._episode._session._engine.cache.instrument(
            InstrumentId.from_str(factory.instrument),
        )
        assert env._episode._snapshot.orders == ()
        assert env._episode._snapshot.positions == ()
        assert account.default_leverage == Decimal("2")
        assert instrument.margin_init == Decimal("0.4")
        assert instrument.margin_maint == Decimal("0.35")
        env.step(np.asarray([1.0], dtype=np.float32))
        snapshot = env._episode._snapshot
        position = snapshot.positions[0]
        notional = (
            Decimal(position["quantity"]) * snapshot.mark * factory.contract_spec.multiplier
        )
        maintenance = Decimal(snapshot.margins[0]["maintenance"].split()[0])
        assert notional > Decimal("10000")
        assert notional / snapshot.equity >= Decimal("1.99")
        assert notional <= Decimal("20000")
        assert abs(maintenance / notional - Decimal("0.35")) < Decimal("0.0001")
    finally:
        env.close()

    excessive = {
        **config,
        "hard_risk": {**config["hard_risk"], "max_leverage": "3"},
    }
    with pytest.raises(DataError, match="initial margin below dated maintenance margin"):
        HistoricalEnvironmentFactory(**excessive)

    # Leverage above 2x is accepted when the dated initial/maintenance rates
    # support it, while maintenance remains tied to the frozen rule.
    three_x_dir = tmp_path / "three-x"
    three_x_dir.mkdir()
    three_x_config = recipe(three_x_dir, initial_margin_rate="0.9",
                            maintenance_margin_rate="0.2")
    three_x_config["hard_risk"] = {
        **three_x_config["hard_risk"], "max_notional": "30000", "max_leverage": "3",
        "margin_buffer_fraction": "0",
    }
    three_x_config["risk_budget_notional"] = "30000"
    three_x_factory = HistoricalEnvironmentFactory(**three_x_config)
    three_x_env = three_x_factory.make_env()
    try:
        three_x_env.reset(seed=17)
        account = three_x_env._episode._session._engine.portfolio.account(
            account_id=AccountId.from_str("SIM-001"),
        )
        instrument = three_x_env._episode._session._engine.cache.instrument(
            InstrumentId.from_str(three_x_factory.instrument),
        )
        assert account.default_leverage == Decimal("3")
        assert instrument.margin_init == Decimal("0.3")
        assert instrument.margin_maint == Decimal("0.2")
    finally:
        three_x_env.close()

    ten_x_dir = tmp_path / "ten-x"
    ten_x_dir.mkdir()
    ten_x_config = recipe(ten_x_dir, initial_margin_rate="1",
                          maintenance_margin_rate="0.004")
    ten_x_config["hard_risk"] = {
        **ten_x_config["hard_risk"], "max_notional": "100000", "max_leverage": "10",
        "margin_buffer_fraction": "0",
    }
    ten_x_config["risk_budget_notional"] = "100000"
    ten_x_factory = HistoricalEnvironmentFactory(**ten_x_config)
    ten_x_env = ten_x_factory.make_env()
    try:
        ten_x_env.reset(seed=19)
        account = ten_x_env._episode._session._engine.portfolio.account(
            account_id=AccountId.from_str("SIM-001"),
        )
        instrument = ten_x_env._episode._session._engine.cache.instrument(
            InstrumentId.from_str(ten_x_factory.instrument),
        )
        assert account.default_leverage == Decimal("10")
        assert instrument.margin_init == Decimal("0.1")
        assert instrument.margin_maint == Decimal("0.004")
        ten_x_env.step(np.asarray([1.0], dtype=np.float32))
        snapshot = ten_x_env._episode._snapshot
        position = snapshot.positions[0]
        notional = (Decimal(position["quantity"]) * snapshot.mark
                    * ten_x_factory.contract_spec.multiplier)
        maintenance = Decimal(snapshot.margins[0]["maintenance"].split()[0])
        assert notional / snapshot.equity > Decimal("2")
        assert notional / snapshot.equity <= Decimal("10")
        assert abs(maintenance / notional - Decimal("0.004")) < Decimal("0.00001")
    finally:
        ten_x_env.close()


def test_flat_barrier_leverage_change_rebuilds_session_and_replays(tmp_path):
    from nautilus_trader.model import AccountId

    config = recipe(tmp_path, maintenance_margin_rate="0.2")
    config["hard_risk"] = {
        **config["hard_risk"], "max_notional": "30000",
        "incomplete_data_max_leverage": "1",
    }
    config["risk_budget_notional"] = "30000"
    factory = HistoricalEnvironmentFactory(**config)
    original = factory.make_env()
    resumed = None
    try:
        _observation, _info = original.reset(seed=101)
        before = original._episode._snapshot
        old_engine = original._episode._session
        observation = original.set_leverage("3")
        changed = original._episode._snapshot
        assert original._episode._session is not old_engine
        assert changed.decision_ts_ns == before.decision_ts_ns
        assert changed.equity == before.equity
        assert changed.orders == changed.positions == ()
        assert original._episode._session._engine.portfolio.account(
            account_id=AccountId.from_str("SIM-001"),
        ).default_leverage == Decimal("3")
        assert original._last_observation is not None
        assert np.array_equal(observation, original._last_observation)

        original.step(np.asarray([0.9], dtype=np.float32))
        position = original._episode._snapshot.positions[0]
        position_notional = (Decimal(position["quantity"])
                             * original._episode._snapshot.mark
                             * factory.contract_spec.multiplier)
        assert position_notional / original._episode._snapshot.equity > 2.5
        native_margin = original._episode._snapshot.margins[0]
        maintenance_margin = Decimal(native_margin["maintenance"].split()[0])
        assert abs(maintenance_margin / position_notional - Decimal("0.2")) < Decimal("0.001")
        with pytest.raises(RuntimeError, match="no open orders and no positions"):
            original.set_leverage("2")
        replay = EpisodeReplay.from_dict(original.rles_export_replay())
        assert replay.leverage_changes == ((0, Decimal("3")),)
        assert replay.to_dict()["schema"] == "rles.episode-replay.v3"
        expected_token = original.rles_boundary_token
        original.close()

        resumed = factory.make_env(replay=replay)
        resumed.reset(seed=101)
        assert resumed.rles_boundary_token == expected_token
        assert resumed._episode._session._engine.portfolio.account(
            account_id=AccountId.from_str("SIM-001"),
        ).default_leverage == Decimal("3")
    finally:
        original.close()
        if resumed is not None:
            resumed.close()


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


def test_factory_market_batch_index_matches_uncached_visible_event_grouping(tmp_path):
    factory = HistoricalEnvironmentFactory(**recipe(tmp_path, bar_count=30))
    warmup_minutes = max(
        field.lookback_bars * field.timeframe_minutes
        for field in factory.features.fields if field.source_kind == "ohlcv_1m"
    ) + 2
    feed_start_ns = max(0, factory.rows[0].decision_ts_ns - warmup_minutes * MINUTE_NS)
    feed_end_ns = factory.rows[-1].decision_ts_ns
    expected = {}
    for event in factory.store.available_event_window(feed_start_ns, feed_end_ns):
        if event.kind in {"instrument_rule", "funding_update"}:
            continue
        if event.payload.get("instrument") != factory.instrument:
            continue
        expected.setdefault(event.available_at_ns, []).append(event.event_id)

    first = factory._visible_market_batch_index()
    second = factory._visible_market_batch_index()

    assert first is second
    assert [(timestamp, [event.event_id for event in batch.events])
            for timestamp, batch in first] == list(expected.items())
    batch_times = tuple(timestamp for timestamp, _batch in first)
    for start_offset in (0, 7):
        episode_rows = factory.rows[start_offset:start_offset + 8]
        episode_feed_start = max(
            0, episode_rows[0].decision_ts_ns - warmup_minutes * MINUTE_NS,
        )
        left = bisect_left(batch_times, episode_feed_start)
        right = bisect_right(batch_times, episode_rows[-1].decision_ts_ns)
        uncached = _visible_batches_by_decision(
            first[left:right], tuple(row.decision_ts_ns for row in episode_rows),
        )
        cached = factory._episode_market_batch_groups(
            start_offset=start_offset, episode_rows=episode_rows,
            feed_start_ns=episode_feed_start,
        )
        assert cached == uncached
        frames = factory._episode_barrier_frames(
            start_offset=start_offset, episode_rows=episode_rows,
            feed_start_ns=episode_feed_start,
        )
        cached_frames = factory._barrier_frames_by_row
        assert cached_frames is not None
        assert frames[0].batches == uncached[0]
        assert frames[0].market == cached_frames[start_offset].market
        assert frames[0].time_features == cached_frames[start_offset].time_features
        assert frames[1:] == cached_frames[start_offset + 1:start_offset + len(episode_rows)]


def test_cached_episode_frames_preserve_native_replay_boundary_and_ledger(tmp_path, monkeypatch):
    from collections import defaultdict

    from rles_research.native_metrics import native_report
    from rles_gym_bridge.l2_bridge import visible_market_features

    config = recipe(tmp_path, bar_count=24)
    cached_factory = HistoricalEnvironmentFactory(**config)
    reference_factory = HistoricalEnvironmentFactory(**config)

    def uncached_frames(*, start_offset, episode_rows, feed_start_ns):
        groups = defaultdict(list)
        for event in reference_factory.store.available_event_window(
                feed_start_ns, episode_rows[-1].decision_ts_ns):
            if event.kind in {"instrument_rule", "funding_update"}:
                continue
            if event.payload.get("instrument") != reference_factory.instrument:
                continue
            if event.available_at_ns is None:
                continue
            groups[event.available_at_ns].append(to_market_event(
                event, quality_grade=DataQualityGrade(reference_factory.quality.grade.value),
            ))
        batches = tuple((timestamp, VisibleBatch(timestamp, tuple(events), True))
                        for timestamp, events in sorted(groups.items()))
        batch_groups = _visible_batches_by_decision(
            batches, tuple(row.decision_ts_ns for row in episode_rows),
        )
        frames = []
        for row, visible_batches in zip(episode_rows, batch_groups, strict=True):
            frames.append(BarrierFrame(
                row.decision_ts_ns, visible_batches,
                visible_market_features(row, reference_factory.features,
                                        reference_factory.schema,
                                        instrument=reference_factory.instrument,
                                        decision_ts_ns=row.decision_ts_ns),
                {},
            ))
        return tuple(frames)

    monkeypatch.setattr(reference_factory, "_episode_barrier_frames", uncached_frames)
    reference_env = reference_factory.make_env()
    cached_resume_env = None
    try:
        reference_env.reset(seed=317)
        prefix = []
        for action in (0.5, -0.5):
            prefix.append(reference_env.step(np.asarray([action], dtype=np.float32)))
        replay = EpisodeReplay.from_dict(reference_env.rles_export_replay())
        continuation = [reference_env.step(np.asarray([action], dtype=np.float32))
                        for action in (0.25, 0.0, 0.5)]
        reference_report = native_report(
            reference_factory.snapshots,
            fill_references=reference_factory.fill_references,
            sample_interval_ns=MINUTE_NS,
        )
        reference_env.close()

        cached_resume_env = cached_factory.make_env(replay=replay)
        resumed_observation, _info = cached_resume_env.reset(seed=317)
        assert np.array_equal(resumed_observation, prefix[-1][0])
        assert cached_resume_env.rles_boundary_token == replay.boundary_token
        for action, expected in zip((0.25, 0.0, 0.5), continuation, strict=True):
            actual = cached_resume_env.step(np.asarray([action], dtype=np.float32))
            assert np.array_equal(actual[0], expected[0])
            assert actual[1:3] == expected[1:3]
            def without_random_event_ids(value):
                if isinstance(value, dict):
                    return {key: without_random_event_ids(item) for key, item in value.items()
                            if key != "event_id"}
                if isinstance(value, tuple):
                    return tuple(without_random_event_ids(item) for item in value)
                if isinstance(value, list):
                    return [without_random_event_ids(item) for item in value]
                return value

            assert without_random_event_ids(actual[3]) == without_random_event_ids(expected[3])
        cached_report = native_report(
            cached_factory.snapshots,
            fill_references=cached_factory.fill_references,
            sample_interval_ns=MINUTE_NS,
        )
        assert cached_report["metrics"] == reference_report["metrics"]
        for field in ("total_fee", "total_slippage_cost", "total_funding",
                      "net_return", "max_drawdown", "turnover_ratio"):
            assert cached_report["metrics"][field] == reference_report["metrics"][field]
        economic_fill_fields = (
            "instrument_id", "order_side", "last_qty", "last_px", "commission", "ts_event",
        )
        assert [tuple(fill.get(key) for key in economic_fill_fields)
                for fill in cached_report["fills"]] == [
                    tuple(fill.get(key) for key in economic_fill_fields)
                    for fill in reference_report["fills"]
                ]
        assert cached_factory.snapshots[-1].equity == reference_factory.snapshots[-1].equity
        assert cached_factory.snapshots[-1].positions == reference_factory.snapshots[-1].positions
    finally:
        reference_env.close()
        if cached_resume_env is not None:
            cached_resume_env.close()


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


def test_nondefault_drawdown_reward_shaping_is_bound_to_historical_identity(tmp_path):
    environment = recipe(tmp_path)
    raw = HistoricalEnvironmentFactory(**environment)
    default_explicit = HistoricalEnvironmentFactory(**environment, reward_lambda_dd=0.0)
    shaped = HistoricalEnvironmentFactory(**environment, reward_lambda_dd=0.1)

    assert raw.config_hash == default_explicit.config_hash
    assert raw.transition_hash == default_explicit.transition_hash
    assert shaped.config_hash != raw.config_hash
    assert shaped.transition_hash != raw.transition_hash
    assert shaped.metadata()["reward_config"]["lambda_dd"] == 0.1
    assert shaped.metadata()["reward_config"]["lambda_guard"] == 0.0
    env = shaped.make_env()
    try:
        assert env._reward.config.lambda_dd == 0.1
        assert env._reward.config.lambda_guard == 0.0
    finally:
        env.close()


def test_rank_partitioned_training_episodes_start_in_distinct_time_lanes(tmp_path):
    factory = HistoricalEnvironmentFactory(**recipe(tmp_path, bar_count=30))
    full_steps = len(factory.rows) - 3
    episode_steps = 4
    first = factory.make_env(episode_rank=0, episode_count=2, episode_steps=episode_steps)
    try:
        _first_obs, first_info = first.reset(seed=11)
    finally:
        first.close()
    last = factory.make_env(episode_rank=1, episode_count=2, episode_steps=episode_steps)
    try:
        _last_obs, last_info = last.reset(seed=22)
        assert first_info["episode_window"]["decision_steps"] == episode_steps
        assert last_info["episode_window"]["decision_steps"] == episode_steps
        assert first_info["episode_window"]["start_ns"] < last_info["episode_window"]["start_ns"]
        assert last_info["episode_window"]["start_ns"] == factory.rows[
            full_steps // 2
        ].decision_ts_ns
        assert first_info["episode_window"]["episode_index"] == 0
        assert last_info["episode_window"]["episode_index"] == 0
    finally:
        last.close()


def test_rank_decision_lanes_partition_every_step_with_balanced_remainders():
    lanes = partition_decision_lanes(129_597, 90)
    sizes = [end - start for start, end in lanes]

    assert lanes[0][0] == 0
    assert lanes[-1][1] == 129_597
    assert all(previous[1] == following[0]
               for previous, following in zip(lanes, lanes[1:]))
    assert sum(sizes) == 129_597
    assert sizes.count(1_440) == 87
    assert sizes.count(1_439) == 3


def test_rank_partition_rejects_duplicate_or_oversized_segments(tmp_path):
    factory = HistoricalEnvironmentFactory(**recipe(tmp_path, bar_count=12))
    full_steps = len(factory.rows) - 3
    with pytest.raises(DataError, match="rank-exclusive time lane"):
        factory.make_env(episode_rank=0, episode_count=2, episode_steps=full_steps)
    with pytest.raises(DataError, match="invalid rotating historical episode partition"):
        factory.make_env(episode_steps=full_steps + 1)


def test_historical_episode_windows_rotate_reproducibly_within_rank_lane(tmp_path):
    factory = HistoricalEnvironmentFactory(**recipe(tmp_path, bar_count=30))
    episode_steps = 4
    env = factory.make_env(episode_rank=0, episode_count=2, episode_steps=episode_steps)
    try:
        _, first = env.reset(seed=11, options={"rles_episode_index": 0})
        _, second = env.reset(seed=12, options={"rles_episode_index": 1})
    finally:
        env.close()
    duplicate = factory.make_env(episode_rank=0, episode_count=2, episode_steps=episode_steps)
    try:
        _, repeated = duplicate.reset(seed=99, options={"rles_episode_index": 1})
    finally:
        duplicate.close()
    assert first["episode_window"]["start_offset"] != second["episode_window"]["start_offset"]
    assert second["episode_window"] == repeated["episode_window"]
    assert 0 <= second["episode_window"]["start_offset"] < (len(factory.rows) - 3) // 2


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


def test_historical_factory_v4_adds_native_position_clocks_without_changing_v3(tmp_path):
    environment = recipe(tmp_path, bar_count=80, start_minute=62)
    symbol = environment["instrument"]
    environment["coverage_policy"]["known_schedule_times_ns"] = {symbol: (90 * MINUTE_NS,)}
    v3 = HistoricalEnvironmentFactory(**environment, feature_profile="causal_mtf_v3")
    v4 = HistoricalEnvironmentFactory(**environment, feature_profile="causal_mtf_v4")

    assert v3.schema.schema_version == "0.4.0"
    assert v4.schema.schema_version == "0.5.0"
    assert v3.schema.names == tuple(name for name in v4.schema.names
                                    if name not in {"execution.position_age_minutes",
                                                    "execution.has_position_age",
                                                    "execution.minutes_since_fill",
                                                    "execution.has_seen_fill",
                                                    "execution.pending_order_age_minutes",
                                                    "account.unrealized_pnl_pct"})
    assert v4.schema.hash != v3.schema.hash
    assert v4.transition_hash != v3.transition_hash


def test_native_position_clock_resets_on_flat_and_reversal_only():
    clock = _NativePositionClock()
    fills = (
        ("open", "BUY", "1", 1 * MINUTE_NS),
        ("add", "BUY", "1", 2 * MINUTE_NS),
        ("reduce", "SELL", "0.5", 3 * MINUTE_NS),
        ("close", "SELL", "1.5", 4 * MINUTE_NS),
        ("short", "SELL", "2", 5 * MINUTE_NS),
        ("reverse", "BUY", "2.5", 6 * MINUTE_NS),
    )
    quantities = ("1", "2", "1.5", None, "-2", "0.5")
    opened_at = (1, 1, 1, None, 5, 6)
    for (event_id, side, quantity, timestamp), current, opening in zip(
        fills, quantities, opened_at, strict=True,
    ):
        positions = (() if current is None else ({
            "side": "LONG" if Decimal(current) > 0 else "SHORT",
            "quantity": str(abs(Decimal(current))),
        },))
        snapshot = SimpleNamespace(
            decision_ts_ns=timestamp + MINUTE_NS,
            fills=({"event_id": event_id, "order_side": side,
                    "last_qty": quantity, "ts_event": timestamp},),
            orders=(),
            positions=positions,
        )
        values = clock.observe(snapshot)
        assert values["has_position_age"] == float(opening is not None)
        assert values["position_age_minutes"] == (0.0 if opening is None else timestamp / MINUTE_NS + 1 - opening)
        assert values["minutes_since_fill"] == 1.0
        assert values["has_seen_fill"] == 1.0
        assert values["pending_order_age_minutes"] == 0.0

    pending = SimpleNamespace(
        decision_ts_ns=10 * MINUTE_NS,
        fills=(), orders=({"client_order_id": "pending-1", "ts_init": 7 * MINUTE_NS},
                          {"client_order_id": "pending-2", "ts_init": 8 * MINUTE_NS}),
        positions=({"side": "LONG", "quantity": "0.5"},),
    )
    # Preserve the open position clock while one pending order ages from native creation time.
    values = clock.observe(pending)
    assert values["position_age_minutes"] == 4.0
    assert values["minutes_since_fill"] == 4.0
    assert values["pending_order_age_minutes"] == 3.0

    for minute in range(11, 127):
        values = clock.observe(SimpleNamespace(
            decision_ts_ns=minute * MINUTE_NS, fills=(), orders=(),
            positions=({"side": "LONG", "quantity": "0.5"},),
        ))
    assert values["position_age_minutes"] == 120.0
    assert values["minutes_since_fill"] == 120.0


def test_v4_position_clocks_replay_identically_from_native_barriers(tmp_path):
    config = recipe(tmp_path, bar_count=80, start_minute=62)
    config["coverage_policy"]["known_schedule_times_ns"] = {
        config["instrument"]: (70 * MINUTE_NS,),
    }
    factory = HistoricalEnvironmentFactory(**config, feature_profile="causal_mtf_v4")
    original = factory.make_env()
    resumed = None
    try:
        original.reset(seed=41)
        prefix = []
        for action in (0.5, -0.5):
            prefix.append(original.step(np.asarray([action], dtype=np.float32)))
        replay = EpisodeReplay.from_dict(original.rles_export_replay())
        expected = original.step(np.asarray([0.25], dtype=np.float32))
        pnl_pct_index = factory.schema.names.index("account.unrealized_pnl_pct")
        quantity_index = factory.schema.names.index("account.position_quantity")
        live_observations = [step[0] for step in (*prefix, expected)]
        assert any(observation[quantity_index] != 0
                   and observation[pnl_pct_index] != 0 for observation in live_observations)
        assert any("unrealized_pnl_currency" in position
                   for snapshot in factory.snapshots for position in snapshot.positions)
        original.close()

        resumed = factory.make_env(replay=replay)
        observation, _info = resumed.reset(seed=41)
        assert np.array_equal(observation, prefix[-1][0])
        actual = resumed.step(np.asarray([0.25], dtype=np.float32))
        assert np.array_equal(actual[0], expected[0])
        assert actual[1:3] == expected[1:3]
    finally:
        original.close()
        if resumed is not None:
            resumed.close()


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


def recipe(tmp_path, *, price_jump=False, bar_count=20, start_minute=6,
           base_currency="ETH", contract_multiplier="1", quantity_step="0.001",
           min_quantity="0.001", min_notional="1", initial_margin_rate="1",
           maintenance_margin_rate="0.35"):
    symbol = f"{base_currency}USDT-PERP.SIM"
    rule = {"instrument": symbol, "tick_size": "0.01", "step_size": quantity_step,
            "min_qty": min_quantity,
            "min_notional": min_notional, "maker_fee": "0.001", "taker_fee": "0.001",
            "initial_margin_rate": initial_margin_rate,
            "maintenance_margin_rate": maintenance_margin_rate, "effective_ns": MINUTE_NS,
            "valid_until_ns": (bar_count + 1) * MINUTE_NS,
            "rule_version": "SYNTHETIC-TEST-ONLY",
            "contract_spec": {"raw_symbol": f"{base_currency}USDT", "base_currency": base_currency,
                              "quote_currency": "USDT", "settlement_currency": "USDT",
                              "is_inverse": False, "multiplier": contract_multiplier}}
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


@pytest.mark.parametrize(("base_currency", "multiplier", "quantity_step", "min_quantity", "min_notional"), (
    ("BTC", "0.001", "0.001", "0.001", "5"),
    ("ETH", "1", "0.001", "0.001", "1"),
    ("SOL", "1", "0.1", "0.1", "10"),
))
def test_explicit_contract_metadata_drives_units_and_native_order_planning(
    tmp_path, base_currency, multiplier, quantity_step, min_quantity, min_notional,
):
    factory = HistoricalEnvironmentFactory(**recipe(
        tmp_path, base_currency=base_currency, contract_multiplier=multiplier,
        quantity_step=quantity_step, min_quantity=min_quantity, min_notional=min_notional,
    ))
    assert factory.contract_spec.base_currency == base_currency
    assert factory.contract_spec.multiplier == Decimal(multiplier)
    assert next(field for field in factory.features.fields if field.name == "volume_5m").unit == base_currency
    assert factory.schema.units[factory.schema.names.index("account.position_quantity")] == base_currency
    assert factory.schema.units[factory.schema.names.index("account.equity")] == "USDT"
    assert factory.metadata()["contract_spec"]["multiplier"] == multiplier
    from rles_orchestrator.application import _training_actor_spec
    from rles_orchestrator.workflow import Route

    actor_spec = _training_actor_spec(factory, (2,), {"output": str(tmp_path / "actor")}, Route.A1)
    quantity_index = factory.schema.names.index("account.position_quantity")
    assert actor_spec.input_scale is not None
    assert actor_spec.input_scale[quantity_index] == pytest.approx(float(Decimal("1000") * Decimal(multiplier)
                                                                          / factory.budget))

    env = factory.make_env()
    try:
        env.reset(seed=17)
        _obs, _reward, terminated, truncated, info = env.step(np.array([0.5], dtype=np.float32))
        assert not terminated and not truncated
        assert info["execution_plan_action"] == "SUBMIT"
        quantity = Decimal(info["execution_delta_qty"])
        assert quantity != 0 and quantity % Decimal(quantity_step) == 0
        assert info["net_exposure"] == pytest.approx(info["quantity_rounded_target_exposure"], abs=0.002)
        assert info["policy_equity_currency"] == "USDT"
    finally:
        env.close()


@pytest.mark.parametrize(("minimum_kind", "minimum_value"), (
    ("quantity", "1000"),
    ("notional", "1001"),
))
@pytest.mark.parametrize(("base_currency", "multiplier", "quantity_step"), (
    ("BTC", "0.001", "0.001"),
    ("ETH", "1", "0.001"),
    ("SOL", "1", "0.1"),
))
def test_contract_metadata_minimums_block_non_executable_orders(
    tmp_path, minimum_kind, minimum_value, base_currency, multiplier, quantity_step,
):
    """Dated quantity and quote-notional limits must govern each contract's plan."""
    minimum_quantity = minimum_value if minimum_kind == "quantity" else quantity_step
    minimum_notional = minimum_value if minimum_kind == "notional" else "1"
    factory = HistoricalEnvironmentFactory(**recipe(
        tmp_path, base_currency=base_currency, contract_multiplier=multiplier,
        quantity_step=quantity_step, min_quantity=minimum_quantity,
        min_notional=minimum_notional,
    ))
    env = factory.make_env()
    try:
        env.reset(seed=23)
        _obs, _reward, terminated, truncated, info = env.step(np.array([0.5], dtype=np.float32))
        assert not terminated and not truncated
        assert info["execution_plan_action"] == "NOOP"
        assert info["execution_plan_reason"] == "below minimum executable quantity/notional"
        assert Decimal(info["execution_delta_qty"]) == 0
        assert info["native_fills"] == ()
        assert info["fees"] == "0" and info["funding_signed"] == "0"
        assert info["slippage_cost"] == "0"
    finally:
        env.close()


def test_nonlegacy_instrument_rule_requires_explicit_contract_spec():
    from rles_data.contracts import InstrumentContractSpec

    with pytest.raises(DataError, match="missing explicit contract_spec"):
        InstrumentContractSpec.from_rule_payload(
            {"instrument": "BTCUSDT-PERP.SIM"}, instrument_id="BTCUSDT-PERP.SIM",
        )


def test_configured_contract_currency_must_match_frozen_rule(tmp_path):
    config = recipe(tmp_path, base_currency="BTC", contract_multiplier="0.001")
    with pytest.raises(DataError, match="differs from dated contract metadata"):
        HistoricalEnvironmentFactory(**config, base_currency="ETH")


def test_causal_mtf_profile_adds_regime_features_without_future_leakage():
    maximum_age = {"mark": 60 * MINUTE_NS, "index": 60 * MINUTE_NS,
                   "funding_update": 60 * MINUTE_NS}
    legacy = historical_feature_spec(maximum_age, base_currency="ETH")
    profile_v1 = historical_feature_spec(maximum_age, profile="causal_mtf_v1", base_currency="ETH")
    profile = historical_feature_spec(maximum_age, profile="causal_mtf_v2", base_currency="ETH")
    profile_v5 = historical_feature_spec(maximum_age, profile="causal_mtf_v5", base_currency="ETH")
    btc = historical_feature_spec(maximum_age, base_currency="BTC")
    assert profile.fields[:len(legacy.fields)] == legacy.fields
    assert profile.fields[:len(profile_v1.fields)] == profile_v1.fields
    assert tuple(field.name for field in profile.fields[len(legacy.fields):len(profile_v1.fields)]) == (
        "return_5m", "return_15m", "return_60m", "volatility_15m", "volatility_60m",
    )
    assert tuple(field.name for field in profile.fields[len(profile_v1.fields):]) == (
        "trend_efficiency_15m", "trend_efficiency_60m",
    )
    assert profile.max_lookback_ns == 61 * MINUTE_NS
    profile_v5_names = {field.name for field in profile_v5.fields}
    assert {"closed_return_1h", "closed_return_1d", "closed_trend_efficiency_1d",
            "volatility_1m"}.issubset(profile_v5_names)
    daily = next(field for field in profile_v5.fields if field.name == "closed_return_1d")
    assert (daily.lookback_bars, daily.timeframe_minutes, daily.recipe) == (1, 1440, "return")
    assert profile_v5.max_lookback_ns >= 2 * 24 * 60 * MINUTE_NS
    assert next(field for field in btc.fields if field.name == "volume_5m").unit == "BTC"

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

def test_causal_mtf_v5_daily_feature_uses_only_published_closed_bars():
    from rles_data.core import FeatureField, FeatureSpec

    daily_spec = FeatureSpec("0.4.0", (
        FeatureField("closed_return_1d", "float64", "ratio", "ohlcv_1m", 1, 1440, "return"),
    ))
    events = []
    for minute in range(1, 2 * 1440 + 1):
        timestamp = minute * MINUTE_NS
        close = 100.0 + minute / 10
        events.append(Event(
            f"daily-minute-{minute}", "ohlcv_1m", timestamp, timestamp, timestamp,
            Availability.NATIVE, "a" * 64,
            {"instrument": "ETHUSDT-PERP.SIM", "open": str(close), "high": str(close),
             "low": str(close), "close": str(close), "volume": "1"},
            "synthetic:closed-daily-feature",
        ))
    decision = 2 * 1440 * MINUTE_NS
    store = CuratedStore(events)
    row = make_feature_row(store, daily_spec, decision, instrument="ETHUSDT-PERP.SIM")
    assert row.available_at_ns == decision
    assert row.values == pytest.approx((float(events[-1].payload["close"])
                                       / float(events[1439].payload["close"]) - 1,))

    delayed_last = Event(
        events[-1].event_id, events[-1].kind, events[-1].ts_event_ns,
        events[-1].ts_init_ns, decision + MINUTE_NS, events[-1].availability_source,
        events[-1].source_hash, dict(events[-1].payload), events[-1].provenance,
    )
    delayed_store = CuratedStore((*events[:-1], delayed_last))
    with pytest.raises(DataError, match="not enough closed bars"):
        make_feature_row(delayed_store, daily_spec, decision, instrument="ETHUSDT-PERP.SIM")

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


def test_short_episode_boundary_has_extra_native_flatten_cost_vs_continuous_account(tmp_path):
    from rles_research.native_metrics import native_report

    config = recipe(tmp_path)
    continuous = HistoricalEnvironmentFactory(**config)
    short = HistoricalEnvironmentFactory(**config)
    full_env = continuous.make_env()
    try:
        full_env.reset(seed=31)
        for index in range(5):
            _full_obs, _, full_done, full_truncated, full_info = full_env.step(
                np.asarray([0.5], dtype=np.float32))
            if index < 4:
                assert not (full_done or full_truncated)
        assert not (full_done or full_truncated)
        assert full_info["terminal_settled"] is False
        full_report = native_report(continuous.snapshots,
                                    fill_references=continuous.fill_references,
                                    sample_interval_ns=MINUTE_NS)
    finally:
        full_env.close()
    short_env = short.make_env(episode_steps=5)
    try:
        short_env.reset(seed=31)
        for index in range(5):
            _short_obs, _, short_done, short_truncated, short_info = short_env.step(
                np.asarray([0.5], dtype=np.float32))
            if index < 4:
                assert not (short_done or short_truncated)
        assert short_done or short_truncated
        assert short_info["terminal_settled"] is True
        short_report = native_report(short.snapshots,
                                     fill_references=short.fill_references,
                                     sample_interval_ns=MINUTE_NS)
        assert not full_report["terminal_flat"] and short_report["terminal_flat"]
        assert short_report["metrics"]["total_fee"] > full_report["metrics"]["total_fee"]
        assert len(short_report["fills"]) > len(full_report["fills"])
    finally:
        short_env.close()


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


@pytest.mark.parametrize(("base_currency", "contract_multiplier", "quantity_step", "min_quantity", "min_notional"), (
    ("BTC", "0.001", "0.001", "0.001", "5"),
    ("ETH", "1", "0.001", "0.001", "1"),
    ("SOL", "1", "0.001", "0.001", "1"),
))
def test_exact_neutral_multiplier_replays_identical_perpetual_funding_ledger(
    tmp_path, base_currency, contract_multiplier, quantity_step, min_quantity, min_notional,
):
    from rles_rl_sb3.actor import InferenceActor
    from rles_rl_sb3.config import PolicySpec, RLConfig, SeedSpec, TransitionFingerprint, digest
    from rles_rl_sb3.provider import RLProvider
    from rles_research.native_metrics import native_report
    from rles_data.io import read_curated

    source_recipe = recipe(
        tmp_path, bar_count=50, start_minute=6, base_currency=base_currency,
        contract_multiplier=contract_multiplier, quantity_step=quantity_step,
        min_quantity=min_quantity, min_notional=min_notional,
    )
    source_path = source_recipe["market_path"]
    events = list(read_curated(source_path).events)
    settlements = (20 * MINUTE_NS, 40 * MINUTE_NS)
    source_recipe["coverage_policy"]["settlement_times_ns"] = {
        source_recipe["instrument"]: settlements,
    }
    for timestamp in settlements:
        events.append(Event(
            f"funding-settlement-{timestamp}", "funding_settlement", timestamp,
            timestamp, timestamp, Availability.NATIVE, "a" * 64,
            {"instrument": source_recipe["instrument"], "rate": "0.001",
             "settlement_ns": timestamp}, "synthetic:neutral-parity-settlement",
        ))
    market_path = tmp_path / "neutral-parity-market.parquet"
    write_curated(market_path, CuratedStore(events))
    source_recipe["market_path"] = market_path

    actor_factory = HistoricalEnvironmentFactory(**source_recipe)
    fingerprint = TransitionFingerprint(
        env_contract=digest("NautilusBarrierEnv-v0.4.0"),
        dataset=actor_factory.store.snapshot_hash,
        feature=actor_factory.features.hash, feature_scaler=actor_factory.scaler.hash,
        observation=actor_factory.schema.hash,
        action=digest({"target_net_exposure": [-1, 1], "shape": [1]}),
        action_transform=digest("soft-hard-native-quantity"),
        execution=actor_factory.execution.fingerprint(), hard_risk=actor_factory.hard.signature(),
        soft_risk=digest({name: str(value) for name, value in asdict(actor_factory.soft).items()}),
        reward=digest({"lambda_dd": 0, "lambda_guard": 0}),
        normalization=digest({"VecNormalize": False}),
        engine=digest("nautilus_trader-2.0.0rc6"), adapter=digest("rles.perpetual-native-v2"),
        fold=digest([actor_factory.start_ns, actor_factory.end_ns]),
        terminal=digest("force_flatten_with_cost:finite_settlement_no_bootstrap_v1"),
    )
    rl_config = RLConfig(
        algorithm="PPO",
        policy=PolicySpec(
            observation_dim=len(actor_factory.schema.names), hidden_layers=(4,), value_layers=(4,),
            observation_schema_hash=actor_factory.schema.hash, action_schema_hash="c" * 64,
        ),
        fingerprint=fingerprint, seeds=SeedSpec(812, (813,), 814, 815), device="cpu",
        hyperparameters={"n_steps": 2, "batch_size": 2, "n_epochs": 1},
    )
    provider = RLProvider(
        rl_config, lambda _rank, _seed, _phase: actor_factory.make_env(fingerprint.hash),
        lambda _gates, _config: True,
    )
    try:
        provider.create()
        provider.train(4)
        artifact = provider.export_inference(
            tmp_path / "neutral-parity-l4-actor",
            np.zeros((4, len(actor_factory.schema.names)), dtype=np.float32),
        )
    finally:
        provider.close()
    actor = InferenceActor(artifact, verify=True)

    def replay(multipliers):
        factory = HistoricalEnvironmentFactory(**source_recipe)
        env = factory.make_env(soft_multipliers=multipliers)
        try:
            observation, _ = env.reset(seed=812)
            trace = []
            while True:
                action = actor.predict(observation[None, :])[0]
                observation, _reward, terminated, truncated, info = env.step(action)
                trace.append(tuple(info[key] for key in (
                    "requested_action", "risk_soft_target", "risk_adjusted_target",
                    "execution_plan_action", "execution_plan_reason", "executed_action",
                    "native_equity", "r_econ", "drawdown",
                )))
                if terminated or truncated:
                    break
            report = native_report(
                factory.snapshots, fill_references=factory.fill_references,
                sample_interval_ns=MINUTE_NS,
                linear_equity_rounding=Decimal("0.00000001"),
            )
            fills = tuple(tuple(fill[key] for key in (
                "order_side", "last_qty", "last_px", "commission", "ts_event",
            )) for fill in report["fills"])
            funding = tuple(tuple(row[key] for key in (
                "pnl_change", "ts_event", "ts_init",
            )) for row in report["funding"])
            currencies = tuple(sorted({snapshot.currency for snapshot in factory.snapshots}))
            return (tuple(trace), fills, funding, tuple(report["account_rows"]), report["metrics"],
                    currencies, report["equity_reconciliation"])
        finally:
            env.close()

    baseline = replay({})
    neutral = replay({"exposure_multiplier": 1.0, "drawdown_throttle": 1.0})
    assert neutral == baseline
    assert baseline[1]  # Native fills include both opening and terminal-close costs.
    assert len(baseline[2]) == 2 and sum(float(row[0].split()[0]) for row in baseline[2]) < 0
    assert all(fill[3].split()[1] == "USDT" for fill in baseline[1])
    assert all(row[0].split()[1] == "USDT" for row in baseline[2])
    assert baseline[5] == ("USDT",)
    assert Decimal(baseline[6]["max_abs_residual"]) <= Decimal("0.00000001")
    assert sum(row["turnover_notional"] for row in baseline[3]) > 0


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
        assert info["execution_decision_id"] == (
            f"policy-{factory.snapshots[-2].decision_ts_ns}")
        assert info["execution_plan_id"] and info["execution_idempotency_key"]
        if expected_plan == "NOOP":
            assert float(info["execution_delta_qty"]) == 0
            assert info["execution_client_order_id"] is None
            assert info["execution_cancel_order_ids"] == ()
            assert info["policy_executed_action_pre_terminal"] == 0.0
            assert factory.snapshots[-1].fills == ()
        else:
            assert float(info["execution_delta_qty"]) != 0
            assert info["execution_client_order_id"].startswith("RLES-")
            assert info["policy_executed_action_pre_terminal"] == pytest.approx(0.1)
            assert factory.snapshots[-1].fills
            assert {fill["client_order_id"] for fill in factory.snapshots[-1].fills} == {
                info["execution_client_order_id"]
            }
    finally:
        env.close()


def test_rollout_execution_trace_joins_decisions_to_native_orders_and_fills(tmp_path):
    from decimal import Decimal
    from rles_orchestrator.application import rollout

    factory = HistoricalEnvironmentFactory(**recipe(tmp_path, bar_count=14))
    report, _ = rollout(
        factory, lambda _batch: np.asarray([[0.5]], dtype=np.float32), seed=31,
        collect_observations=False, include_account_rows=False, include_execution_trace=True,
    )
    summary = report["execution_decision_diagnostics"]
    trace = report["execution_decision_trace"]
    terminal_trace = report["terminal_settlement_trace"]
    assert summary["components_known"] is True
    assert summary["trace_included"] is True
    assert summary["decision_count"] == report["action_diagnostics"]["count"]
    assert len(trace) == summary["decision_count"]
    assert all(type(row["net_exposure"]) is float
               and isinstance(row["market_features"], dict)
               and "volatility_5m" in row["market_features"]
               for row in trace)
    assert all(row["actor_raw_action_unit"] == "normalized_box[-1,1]"
               and row["risk_target_unit"] == "fraction_of_risk_budget"
               and row["actor_raw_action"] == row["actor_requested_target"] == 0.5
               for row in trace)
    assert all(row["equity_currency"] == "USDT"
               and Decimal(row["policy_equity_after"]) - Decimal(row["policy_equity_before"])
               == Decimal(row["policy_equity_change"])
               for row in trace)
    assert all(row["soft_target"] is None or -1 <= row["soft_target"] <= 1
               for row in trace)
    submitted = [row for row in trace if row["action"] == "SUBMIT"]
    noops = [row for row in trace if row["action"] == "NOOP"]
    assert submitted and noops
    assert summary["native_order_count"] == len(submitted)
    assert summary["submitted_decisions_with_fills"] == len(submitted)
    all_report_fills = {
        fill["event_id"]: fill for fill in report["fills"]
    }
    linked_fill_ids = set()
    for row in submitted:
        assert row["client_order_id"].startswith("RLES-")
        assert row["submission_status"] == "native_order_observed"
        assert row["native_order_status"] == "FILLED"
        assert row["unfilled_reason"] is None
        assert float(row["filled_qty"]) > 0
        assert row["fee_currency"] == "USDT"
        assert Decimal(row["fee"]) == Decimal(row["fees_by_currency"]["USDT"])
        for fill in row["fills"]:
            linked_fill_ids.add(fill["event_id"])
            assert fill["fee"].endswith(" USDT")
    assert linked_fill_ids <= set(all_report_fills)
    assert any(row["fills"] for row in submitted)
    assert all(row["client_order_id"] is None and row["fills"] == []
               and row["related_order_ids"] == []
               for row in noops)
    assert len(terminal_trace) == 1
    settlement = terminal_trace[0]
    assert settlement["decision_id"] == trace[-1]["decision_id"]
    assert settlement["settlement_barrier_ts_ns"] >= settlement["policy_barrier_ts_ns"]
    assert settlement["pre_terminal_net_exposure"] > 0
    assert Decimal(settlement["equity_before"]) == Decimal(trace[-1]["policy_equity_after"])
    assert settlement["post_terminal_open_position_count"] == 0
    assert Decimal(settlement["equity_after"]) - Decimal(settlement["equity_before"]) == Decimal(
        settlement["equity_delta"]
    )
    terminal_order_ids = {row["client_order_id"] for row in settlement["orders"]}
    assert terminal_order_ids - {row["client_order_id"] for row in submitted}
    terminal_fill_ids = {row["event_id"] for row in settlement["fills"]}
    assert terminal_fill_ids
    assert terminal_fill_ids <= set(all_report_fills)
    json.dumps(terminal_trace, allow_nan=False)


def test_partial_native_fill_exposes_remaining_quantity_for_cancel_replace(tmp_path):
    config = recipe(tmp_path, bar_count=14)
    config["execution"]["stress_fill"]["simulated_depth_fraction"] = "0.5"
    factory = HistoricalEnvironmentFactory(**config, mode="stress")
    env = factory.make_env()
    try:
        env.reset(seed=31)
        action = np.asarray([0.8], dtype=np.float32)
        _observation, _reward, terminated, truncated, submitted = env.step(action)
        assert not terminated and not truncated
        pending = factory.snapshots[-1].orders
        assert len(pending) == 1
        order = pending[0]
        assert order["status"] == "PARTIALLY_FILLED"
        assert order["quantity"] == "0.800"
        assert order["filled_qty"] == "0.400"
        assert order["leaves_qty"] == "0.400"
        assert submitted["execution_client_order_id"] == order["client_order_id"]
        assert submitted["native_policy_fills"]
        assert submitted["native_policy_fills"][0]["client_order_id"] == order["client_order_id"]
        assert submitted["native_order_statuses_after"][0]["status"] == "PARTIALLY_FILLED"
        _observation, _reward, terminated, truncated, canceled = env.step(action)
        assert not terminated and not truncated
        assert canceled["execution_plan_action"] == "CANCEL"
        assert canceled["execution_plan_reason"] == "cancel then resnapshot and replan after ACK"
        assert canceled["execution_related_order_ids"] == (order["client_order_id"],)
        assert factory.snapshots[-1].orders == ()
        final_order = next(row for row in factory.snapshots[-1].all_orders
                           if row["client_order_id"] == order["client_order_id"])
        assert final_order["status"] == "CANCELED"
        assert final_order["filled_qty"] == "0.400"
        assert final_order["quantity"] == "0.800"
        assert any(fill["client_order_id"] == order["client_order_id"]
                   for snapshot in factory.snapshots for fill in snapshot.fills)
    finally:
        env.close()


@pytest.mark.parametrize(
    "order, filled_qty, expected",
    ((None, "0", "unknown"),
     ({"quantity": "1", "filled_qty": "0", "status": "REJECTED",
       "reject_reason": "insufficient margin"},
      "0", "insufficient margin"),
     ({"quantity": "1", "filled_qty": "0", "status": "REJECTED",
       "reject_reason": {"code": "MARGIN"}},
      "0", "REJECTED"),
     ({"quantity": "1", "filled_qty": "0", "status": "DENIED",
       "reject_reason": [], "reason": "risk limit"},
      "0", "risk limit"),
     ({"quantity": "1", "filled_qty": "0", "status": "CANCELED"},
      "0", "canceled_unfilled"),
     ({"quantity": "1", "filled_qty": "0.4", "status": "CANCELED"},
      "0.4", "canceled_after_partial_fill"),
     ({"quantity": "1", "filled_qty": "0.4", "status": "PARTIALLY_FILLED"},
      "0.4", "partial_fill_open"),
     ({"quantity": "1", "filled_qty": "0.4", "status": "PENDING_CANCEL"},
      "0.4", "cancel_pending"),
     ({"quantity": "1", "filled_qty": "0", "status": "PENDING_UPDATE"},
      "0", "update_pending"),
     ({"quantity": "1", "filled_qty": "0", "status": "ACCEPTED"},
      "0", "awaiting_fill"),
     ({"quantity": "1", "filled_qty": "0", "status": "INITIALIZED"},
      "0", "initialized_unfilled"),
     ({"quantity": "1", "filled_qty": "0", "status": "EMULATED"},
      "0", "emulated_unfilled"),
     ({"quantity": "1", "filled_qty": "0", "status": "RELEASED"},
      "0", "released_unfilled"),
     ({"quantity": "1", "filled_qty": "0", "status": "TRIGGERED"},
      "0", "triggered_unfilled"),
     ({"quantity": "1", "filled_qty": "0", "status": "VOIDED"},
      "0", "voided_unfilled"),
     ({"quantity": "1", "filled_qty": "1", "status": "FILLED"}, "1", None)),
)
def test_execution_trace_unfilled_reason_uses_final_native_order_state(
        order, filled_qty, expected) -> None:
    from decimal import Decimal
    from rles_orchestrator.application import _unfilled_order_reason

    assert _unfilled_order_reason(order, Decimal(filled_qty)) == expected


def test_execution_trace_classifies_every_locked_nautilus_order_status() -> None:
    from decimal import Decimal
    from nautilus_trader.model import OrderStatus
    from rles_orchestrator.application import _unfilled_order_reason

    for native_status in OrderStatus.variants():
        status = str(native_status)
        filled = Decimal("1") if status == "FILLED" else (
            Decimal("0.5") if status == "PARTIALLY_FILLED" else Decimal(0)
        )
        result = _unfilled_order_reason(
            {"quantity": "1", "filled_qty": str(filled), "status": status},
            filled,
        )
        assert result != "unknown", f"native order status has no trace classification: {status}"


def test_execution_trace_rejects_mismatch_between_native_order_and_fill_events() -> None:
    from decimal import Decimal
    from rles_data.core import DataError
    from rles_orchestrator.application import _unfilled_order_reason

    order = {"quantity": "1", "filled_qty": "0.5", "status": "PARTIALLY_FILLED"}
    with pytest.raises(DataError, match="disagree with cumulative order"):
        _unfilled_order_reason(order, Decimal("0.4"))
    with pytest.raises(DataError, match="no matching order"):
        _unfilled_order_reason(None, Decimal("0.1"))


@pytest.mark.parametrize(
    "order, filled_qty, message",
    (({"quantity": "1", "filled_qty": "0.4", "status": "FILLED"},
      "0.4", "FILLED status conflicts"),
     ({"quantity": "1", "filled_qty": "0", "status": "PARTIALLY_FILLED"},
      "0", "PARTIALLY_FILLED status conflicts"),
     ({"quantity": "1", "filled_qty": "1", "status": "PARTIALLY_FILLED"},
      "1", "PARTIALLY_FILLED status conflicts")),
)
def test_execution_trace_rejects_order_status_fill_quantity_conflicts(
        order, filled_qty, message) -> None:
    from decimal import Decimal
    from rles_data.core import DataError
    from rles_orchestrator.application import _unfilled_order_reason

    with pytest.raises(DataError, match=message):
        _unfilled_order_reason(order, Decimal(filled_qty))


def test_non_submit_trace_preserves_related_partial_fill_and_fee_facts() -> None:
    from rles_orchestrator.application import _related_native_order_trace

    order = {"client_order_id": "RLES-partial", "quantity": "1.000",
             "filled_qty": "0.400", "status": "CANCELED"}
    fills = ({"event_id": "fill-1", "last_qty": "0.400", "commission": "0.1 USDT"},)

    related = _related_native_order_trace("RLES-partial", order, fills)

    assert related == {
        "client_order_id": "RLES-partial",
        "status": "CANCELED",
        "order_quantity": "1.000",
        "cumulative_filled_qty": "0.400",
        "fill_event_ids": ["fill-1"],
        "fees": {"fee": "0.1", "fee_currency": "USDT", "fees_by_currency": {"USDT": "0.1"}},
        "unfilled_reason": "canceled_after_partial_fill",
    }


@pytest.mark.parametrize("identity", [None, 7, [], {}, "", " order-1 "])
def test_execution_trace_rejects_invalid_native_order_identity(identity) -> None:
    from rles_data.core import DataError
    from rles_orchestrator.application import _index_native_orders

    with pytest.raises(DataError, match="native order history lacks valid client identity"):
        _index_native_orders(({"client_order_id": identity, "quantity": "1"},))


def test_execution_trace_rejects_conflicting_duplicate_native_order_identity() -> None:
    from rles_data.core import DataError
    from rles_orchestrator.application import _index_native_orders

    orders = (
        {"client_order_id": "order-1", "quantity": "1", "status": "CANCELED"},
        {"client_order_id": "order-1", "quantity": "2", "status": "FILLED"},
    )
    with pytest.raises(DataError, match="native order identity has conflicting final facts"):
        _index_native_orders(orders)


def test_execution_trace_deduplicates_identical_final_native_order_facts() -> None:
    from rles_orchestrator.application import _index_native_orders

    order = {"client_order_id": "order-1", "quantity": "1", "status": "CANCELED"}
    assert _index_native_orders((order, dict(order))) == {"order-1": order}


def test_execution_trace_fee_summary_preserves_currency_units() -> None:
    from rles_data.core import DataError
    from rles_orchestrator.application import _fee_summary

    summary = _fee_summary([
        {"commission": "0.10 USDT"},
        {"commission": "0.002 BNB"},
        {"commission": "0.20 USDT"},
    ])
    assert summary == {
        "fee": None,
        "fee_currency": None,
        "fees_by_currency": {"BNB": "0.002", "USDT": "0.30"},
    }
    assert _fee_summary([{"commission": "0.30 USDT"}]) == {
        "fee": "0.30",
        "fee_currency": "USDT",
        "fees_by_currency": {"USDT": "0.30"},
    }
    with pytest.raises(DataError, match="amount and currency"):
        _fee_summary([{"commission": "0.30"}])
    with pytest.raises(DataError, match="finite"):
        _fee_summary([{"commission": "NaN USDT"}])


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


def test_prepared_history_reuses_verified_store_and_preserves_native_trajectory(tmp_path, monkeypatch):
    import rles_gym_bridge.historical as historical

    config = recipe(tmp_path)
    original = HistoricalEnvironmentFactory(**config)
    monkeypatch.setattr(historical, "read_curated", lambda *_: pytest.fail("verified history must be shared"))
    shared = HistoricalEnvironmentFactory(**config, prepared_store=original.store)
    assert shared.store is original.store
    assert shared.metadata() == original.metadata()
    assert shared.transition_hash == original.transition_hash
    trajectories = []
    for factory in (original, shared):
        env = factory.make_env()
        try:
            obs, _ = env.reset(seed=31)
            rows = [(obs.tolist(),)]
            while True:
                obs, reward, terminated, truncated, info = env.step(np.array([0.5], dtype=np.float32))
                rows.append((obs.tolist(), reward, terminated, truncated, info["native_equity"],
                             info["execution_plan_action"], info["execution_plan_reason"]))
                if terminated or truncated:
                    break
            trajectories.append(rows)
        finally:
            env.close()
    assert trajectories[0] == trajectories[1]


def test_market_audit_reports_only_registered_train_coverage_without_granting_missing_rules(
    monkeypatch, tmp_path,
):
    from rles_orchestrator.application import audit_market
    from rles_data.io import read_curated

    config = recipe(tmp_path)
    calls = []

    def read_train_window(run, window, *, marker_sha256, label,
                          allow_legacy_isolated_snapshot=False):
        calls.append((run, window, marker_sha256, label, allow_legacy_isolated_snapshot))
        return read_curated(run["environment"]["market_path"])

    monkeypatch.setattr("rles_orchestrator.application.read_registered_train_window",
                        read_train_window)
    access = {"start_ns": config["start_ns"], "end_ns": config["end_ns"],
              "marker_sha256": "a" * 64, "label": "train-market-audit-test"}
    run = {"output": str(tmp_path / "audit"), "environment": config,
           "train_data_access": access}
    result = audit_market(run)
    report = result["splits"]["train"]
    assert result["schema"] == "rles.market-audit.v2"
    assert result["access_scope"] == {"split": "train_inner", "start_ns": config["start_ns"],
                                      "end_ns": config["end_ns"]}
    assert set(result["splits"]) == {"train"}
    assert calls == [(run, {"start_ns": access["start_ns"], "end_ns": access["end_ns"]},
                      access["marker_sha256"], access["label"], False)]
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
           "normalize_observation": True, "rl_corpus_episode_steps": 2}
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(run), encoding="utf-8")
    resolved = load_config(config_path)
    result = train(resolved)
    run_manifest = json.loads((tmp_path / "run" / "run.json").read_text(encoding="utf-8"))
    assert run_manifest["gpu_lease"] is None
    exploratory = {**resolved, "evaluation": {**resolved["evaluation"], "split": "validation"}}
    restored = backtest(exploratory, result["output"])
    assert restored["evaluation_scope"]["formal"] is False
    with pytest.raises(DataError, match="requires WFO research access"):
        backtest(resolved, result["output"])
    assert result["terminal_flat"] and restored["terminal_flat"]
    assert result["metrics"] == restored["metrics"]
    assert restored["market_data_provenance"] == [
        "synthetic:funding_update", "synthetic:index", "synthetic:mark", "synthetic:ohlcv_1m",
    ]
    assert restored["rule_provenance"] == ["synthetic:factory-test-rule"]
    if algorithm == "PPO":
        learning = tmp_path / "run" / "R1-learning.json"
        learning_record = json.loads(learning.read_text(encoding="utf-8"))
        assert learning_record["schema"] == "rles.r1-learning.v5"
        assert learning_record["final_steps"] == 4
        assert sum(block["samples"] for block in learning_record["blocks"]) == 4
        resources = learning_record["blocks"][0]["resource_measurements"]
        assert resources["scope"] == (
            "cpu_rss_at_learn_call_end; torch_cuda_allocator_peaks_since_learn_call_start"
        )
        assert resources["cpu_rss_bytes"] is None or resources["cpu_rss_bytes"] > 0
        assert resources["gpu_max_allocated_bytes"] is None
        assert resources["gpu_max_reserved_bytes"] is None
        bundle_record = json.loads((tmp_path / "run" / "bundle.json").read_text(encoding="utf-8"))
        assert bundle_record["runtime_resources"]["gpu_lease"] is None
        assert bundle_record["runtime_resources"]["cuda_allocator_fraction"] is None
        assert bundle_record["actor"]["training_diagnostics_hash"] == sha256(
            learning.read_bytes()
        ).hexdigest()
        original_learning = learning.read_bytes()
        learning.write_bytes(original_learning + b"\n")
        with pytest.raises(DataError, match="integrity mismatch"):
            backtest(exploratory, result["output"])
        learning.write_bytes(original_learning)
        bundle = tmp_path / "run" / "bundle.json"
        original = bundle.read_bytes()
        bundle.write_bytes(original + b"\n")
        with pytest.raises(DataError, match="integrity mismatch"):
            backtest(exploratory, result["output"])
        bundle.write_bytes(original)


def test_application_parallel_rl_uses_rotating_rank_exclusive_train_lanes(tmp_path):
    from rles_orchestrator.application import train

    config = recipe(tmp_path)
    config["end_ns"] = 13 * MINUTE_NS
    config["market_path"], config["rules_path"] = str(config["market_path"]), str(config["rules_path"])
    run = {
        "environment": config,
        "evaluation": {"start_ns": 13 * MINUTE_NS, "end_ns": 20 * MINUTE_NS,
                       "coverage_policy": config["coverage_policy"]},
        "output": str(tmp_path / "parallel-run"), "algorithm": "PPO", "steps": 4,
        "hidden_layers": [2], "rl_num_envs": 2, "rl_episode_steps": 2,
        "rl_corpus_episode_steps": 2,
        "hyperparameters": {"n_steps": 2, "batch_size": 4, "n_epochs": 1},
    }
    result = train(run)
    coverage = result["training_coverage"]
    assert coverage["environment_count"] == 2
    assert coverage["nominal_episode_decision_steps"] == 2
    assert coverage["sampling_policy"] == "rank_exclusive_coprime_rotating_windows_v1"
    assert coverage["initial_episode_start_offsets"][0] == 0
    assert coverage["initial_episode_start_offsets"][1] > 0
    assert coverage["upper_bound_unique_train_window_steps"] == coverage[
        "train_window_decision_steps"
    ]
    assert coverage["upper_bound_train_window_coverage_fraction"] == 1.0


def test_parallel_rl_checkpoint_restores_each_partition_mid_episode(tmp_path, monkeypatch):
    from rles_orchestrator.application import train
    from rles_rl_sb3.provider import RLProvider

    environment = recipe(tmp_path)
    environment["end_ns"] = 13 * MINUTE_NS
    environment["market_path"], environment["rules_path"] = (
        str(environment["market_path"]), str(environment["rules_path"]),
    )
    run = {
        "environment": environment,
        "evaluation": {"start_ns": 13 * MINUTE_NS, "end_ns": 20 * MINUTE_NS,
                       "coverage_policy": environment["coverage_policy"]},
        "output": str(tmp_path / "parallel-resume"), "algorithm": "PPO", "steps": 4,
        "hidden_layers": [2], "rl_num_envs": 2, "rl_episode_steps": 2,
        "rl_corpus_episode_steps": 2,
        "hyperparameters": {"n_steps": 1, "batch_size": 2, "n_epochs": 1},
        "checkpoint_interval_steps": 2,
    }
    original_train = RLProvider.train
    calls = 0

    def stop_after_first_boundary(provider, total_timesteps, *, resumed=False, callback=None):
        nonlocal calls
        if calls:
            raise RuntimeError("interrupt parallel learner after durable boundary")
        calls += 1
        return original_train(provider, total_timesteps, resumed=resumed, callback=callback)

    monkeypatch.setattr(RLProvider, "train", stop_after_first_boundary)
    with pytest.raises(RuntimeError, match="interrupt parallel learner"):
        train(run)
    checkpoint = Path(run["output"]) / "R1-checkpoints" / "step-000000000002"
    replays = json.loads((checkpoint / "episode_replays.json").read_text(encoding="utf-8"))
    assert len(replays) == 2
    assert all(len(replay["actions"]) == 1 for replay in replays)
    assert all(replay["schema"] == "rles.episode-replay.v2" for replay in replays)
    assert [replay["episode_index"] for replay in replays] == [0, 0]

    monkeypatch.setattr(RLProvider, "train", original_train)
    run["resume_checkpoint"] = str(checkpoint)
    result = train(run)
    assert result["training_coverage"]["environment_count"] == 2
    assert (Path(run["output"]) / "COMPLETED.json").is_file()


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
           "checkpoint_interval_steps": 2, "rl_corpus_episode_steps": 2}
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
    restored = backtest({**run, "evaluation": {**run["evaluation"], "split": "validation"}}, result["output"])
    assert result["metrics"] == restored["metrics"]
    assert restored["terminal_flat"]
    bundle_root = Path(result["output"])
    bundle_files_before = {str(path.relative_to(bundle_root)): sha256(path.read_bytes()).hexdigest()
                           for path in bundle_root.rglob("*") if path.is_file()}
    with monkeypatch.context() as diagnostic_patch:
        from rles_data.io import curated_source_identity
        from rles_orchestrator import research_access

        def forbidden(*_args, **_kwargs):
            pytest.fail("Train diagnostics must not refit scaling or construct an evaluation")
        diagnostic_patch.setattr("rles_gym_bridge.historical.fit_train", forbidden)
        diagnostic_patch.setattr("rles_orchestrator.application._evaluation_factory", forbidden)
        diagnostic_patch.setattr("rles_orchestrator.application.ResearchLedger", forbidden)
        # Keep the real registered reader, marker verification, and row-group
        # scoping in this integration path. Stub only the project registry write
        # so the test remains isolated from the workspace's persistent ledger.
        diagnostic_patch.setattr(
            research_access, "record_development",
            lambda *_args, **_kwargs: "synthetic-train-reservation",
        )
        scoped_run = {
            **run, "evaluation": None,
            "train_data_access": {
                "start_ns": MINUTE_NS, "end_ns": 31 * MINUTE_NS,
                "marker_sha256": curated_source_identity(config["market_path"])[
                    "marker_sha256"],
                "label": "train-diagnostic-test",
            },
        }
        diagnostic = diagnose_train(scoped_run, bundle_root)
        assert diagnostic["diagnostic_scope"]["split"] == "train_inner"
        assert diagnostic["diagnostic_scope"]["evaluation_queries"] == 0
        assert diagnostic["diagnostic_scope"]["end_ns"] == 12 * MINUTE_NS
        assert diagnostic["action_diagnostics"]["count"] > 0
        assert diagnostic["execution_attribution"]["components_known"] is True
        assert diagnostic["execution_attribution"]["quantity_rounding_components_known"] is True
        assert diagnostic["quantity_rounding_diagnostics"]["components_known"] is True
        assert diagnostic["quantity_rounding_diagnostics"]["count"] == diagnostic["action_diagnostics"]["count"]
        assert diagnostic["terminal_flat"] is True
        assert "account_rows" not in diagnostic
        with pytest.raises(DataError, match="may not replace"):
            diagnose_train({**scoped_run,
                            "environment": {**config, "end_ns": 20 * MINUTE_NS}}, bundle_root)
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
        # E2 now includes one paired exact-neutral candidate in each generation.
        assert sum(json.loads(item["usage_json"]).get("lane_requests", 0) for item in receipts) == 9
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
           "checkpoint_interval_steps": 2, "rl_corpus_episode_steps": 2}
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
           "checkpoint_interval_steps": 2, "rl_corpus_episode_steps": 2}
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
           "checkpoint_interval_steps": 2, "rl_corpus_episode_steps": 2}
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
    if route == "A4":
        # Simulate interruption after the payload is durable but before its
        # digest sidecar is restored by the E1 recovery pass.
        e1_digest_path = root / "E1-selected-actor.json.sha256"
        assert e1_digest_path.is_file()
        e1_digest_path.unlink()

    monkeypatch.setattr(EvoTorchProvider, "continue_evolution", original_continue)
    run["resume_checkpoint"] = str(checkpoint)
    result = train(run)
    report = backtest({**run, "evaluation": {**run["evaluation"], "split": "validation"}}, result["output"])
    assert report["terminal_flat"]
    assert (root / "r1" / "COMPLETED").read_text(encoding="ascii") == actor_hash
    assert (root / "E2" / "generation-00000002" / run_id / "COMPLETED").is_file()
    assert (root / "COMPLETED.json").is_file()
    bundle = json.loads((root / "bundle.json").read_text(encoding="utf-8"))
    if route == "A4":
        e1_artifact = bundle["stage_actor_artifacts"]["E1"]
        e1_path = root / e1_artifact["path"]
        assert sha256(e1_path.read_bytes()).hexdigest() == e1_artifact["sha256"]
        assert e1_path.with_suffix(e1_path.suffix + ".sha256").read_text(encoding="ascii") == e1_artifact["sha256"]
        e1_payload = json.loads(e1_path.read_text(encoding="utf-8"))
        e1_lineage = next(row for row in bundle["lineage"] if row["stage"] == "E1")
        assert e1_payload["actor_hash"] == e1_lineage["actor_hash"] == e1_artifact["actor_hash"]
        assert e1_payload["lane_evidence_hash"] == e1_artifact["lane_evidence_hash"] == (
            e1_lineage["metadata"]["lane_evidence_hash"]
        )
        from rles_es_evotorch.actor import ActorGenomeSpec, genome_hash

        e1_spec_values = dict(e1_payload["actor_spec"])
        e1_spec_values["observation_names"] = tuple(e1_spec_values["observation_names"])
        e1_spec_values["hidden_sizes"] = tuple(e1_spec_values["hidden_sizes"])
        e1_spec_values["input_scale"] = (
            None if e1_spec_values.get("input_scale") is None else tuple(e1_spec_values["input_scale"])
        )
        assert genome_hash(ActorGenomeSpec(**e1_spec_values),
                           np.asarray(e1_payload["genome"], dtype=np.float32)) == e1_payload["actor_hash"]
    else:
        assert bundle["stage_actor_artifacts"] == {}


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
           "checkpoint_interval_steps": 2, "rl_corpus_episode_steps": 2}
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
