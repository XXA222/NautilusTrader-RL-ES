import math
import random

import numpy as np
import pytest

from rles_data.core import DataError, FoldManifest, MINUTE_NS, Window
from rles_world_model.baselines import crps_ensemble, score_baselines, stationary_block_bootstrap
from rles_world_model.contracts import MarketFrame, MarketReplay
from rles_world_model.data_audit import audit_replay
from rles_world_model.train_dataset import causal_frames_as_of
from rles_world_model.upstream import (constant_market_action, rssm_market_imagine,
                                       rssm_market_observe)


def _frame(index, *, available_delay=0, close=None):
    value = float(close if close is not None else 100 + index * 0.1 + math.sin(index / 4))
    ts = index * MINUTE_NS
    return MarketFrame(ts, ts + available_delay, "ETHUSDT-PERP.SIM",
                       value, value * 1.001, value * 0.999, value, 1.0)


def _fold():
    return FoldManifest(
        "0.4.0", "a" * 64, "b" * 64,
        Window(MINUTE_NS, MINUTE_NS, 70 * MINUTE_NS),
        Window(70 * MINUTE_NS, 130 * MINUTE_NS, 140 * MINUTE_NS),
        Window(140 * MINUTE_NS, 140 * MINUTE_NS, 150 * MINUTE_NS),
        0, 0, None,
    )


def test_market_frame_rejects_future_availability_and_invalid_ohlc():
    with pytest.raises(DataError, match="before its close"):
        _frame(1, available_delay=-1)
    with pytest.raises(DataError, match="OHLC bounds"):
        MarketFrame(MINUTE_NS, MINUTE_NS, "ETH", 100, 100, 101, 102, 1)


def test_causal_history_hides_unavailable_and_future_bars():
    frames = (_frame(1), _frame(2, available_delay=MINUTE_NS), _frame(3))
    history = causal_frames_as_of(frames, decision_ts_ns=2 * MINUTE_NS)
    assert tuple(frame.ts_close_ns for frame in history) == (MINUTE_NS,)


def test_audit_reports_gap_and_baselines_fail_closed():
    replay = MarketReplay((_frame(1), _frame(3)), "a" * 64)
    report = audit_replay(replay)
    assert report.missing_minutes == 1
    assert not report.market_model_eligible
    with pytest.raises(DataError, match="gap-free"):
        score_baselines(replay, _fold())


def test_crps_ensemble_exact_for_point_mass():
    assert crps_ensemble([0.0, 0.0], 2.0) == 2.0
    assert crps_ensemble([0.0, 2.0], 1.0) == 0.5


def test_stationary_bootstrap_is_seeded_and_shapes_paths():
    returns = tuple(float(index) for index in range(16))
    left = stationary_block_bootstrap(returns, paths=8, horizon=5,
                                      mean_block_length=4, rng=random.Random(7))
    right = stationary_block_bootstrap(returns, paths=8, horizon=5,
                                       mean_block_length=4, rng=random.Random(7))
    assert left == right
    assert len(left) == 8 and all(len(path) == 5 for path in left)
    assert all(value in returns for path in left for value in path)


def test_b0_baselines_score_validation_using_train_and_causal_warmup():
    frames = tuple(_frame(index) for index in range(1, 150))
    replay = MarketReplay(frames, "a" * 64)
    scores = score_baselines(replay, _fold(), seed=19, ensemble_size=64,
                             volatility_lookback=10, bootstrap_block=8)
    assert [score.name for score in scores] == [
        "S0_ZERO_DRIFT", "S1_CONDITIONAL_VOL", "S2_BLOCK_BOOTSTRAP",
    ]
    assert all(score.observations == 10 and score.mean_crps >= 0 for score in scores)


def test_market_rssm_adapter_zeroes_policy_action_before_upstream_call():
    class FakeRSSM:
        def observe(self, carry, tokens, action, reset, training, *, single):
            return action

    adapter_outputs = [rssm_market_observe(
        FakeRSSM(), None, None, None, False,
        trading_action=np.full((2, 1), policy_action, dtype=np.float32), single=True,
    )["market_dummy"] for policy_action in (-1.0, 0.0, 1.0)]
    assert all(np.array_equal(adapter_outputs[0], action) for action in adapter_outputs[1:])
    assert not np.any(adapter_outputs[0])
    assert np.array_equal(constant_market_action(np.ones((3, 1)))['market_dummy'],
                          np.zeros((3, 1), dtype=np.float32))


def test_market_imagination_adapter_zeroes_policy_action_before_upstream_call():
    class FakeRSSM:
        def imagine(self, carry, policy, *, length, training, single):
            return carry, policy

    outputs = [rssm_market_imagine(
        FakeRSSM(), {"state": 1},
        trading_action=np.full((2, 1), action, dtype=np.float32),
    )[1]["market_dummy"] for action in (-1.0, 0.0, 1.0)]
    assert all(np.array_equal(outputs[0], output) for output in outputs[1:])
    assert not np.any(outputs[0])
