import pytest

from rles_research.baselines import BaselinePolicy, BaselineSpec


@pytest.mark.parametrize("closes,trend", [([100.0] * 21, 0.0),
                                         ([100.0 + i for i in range(21)], 0.4),
                                         ([100.0 - i for i in range(21)], -0.4)])
def test_causal_reversal_is_opposite_trend_with_same_bounded_risk_target(closes, trend):
    momentum = BaselinePolicy(BaselineSpec("B2_simple_momentum", 31, target_long=0.4))
    reversal = BaselinePolicy(BaselineSpec("B5_simple_reversal", 31, target_long=0.4))
    assert momentum.action(closes) == trend
    assert reversal.action(closes) == -trend
    assert momentum.action(closes[:2]) == reversal.action(closes[:2]) == 0


def test_fixed_short_is_a_direction_baseline_and_validates_market_inputs():
    policy = BaselinePolicy(BaselineSpec("B4_always_short", 31, target_long=0.4))
    assert policy.action([100.0]) == -0.4
    with pytest.raises(ValueError, match="historic close"):
        policy.action([float("nan")])
