from __future__ import annotations

import math

import pytest

from rles_gym_bridge.reward import RewardCalculator, RewardConfig


def test_fixed_reward_scale_preserves_native_return_components() -> None:
    initial, middle, final = 10_000.0, 10_001.0, 9_998.0
    raw = RewardCalculator(RewardConfig())
    scaled = RewardCalculator(RewardConfig(reward_scale=10_000.0))
    raw.reset(initial)
    scaled.reset(initial)

    raw_first = raw.calculate(native_equity=middle, guard_intervention=0.0)
    scaled_first = scaled.calculate(native_equity=middle, guard_intervention=0.0)
    raw_second = raw.calculate(native_equity=final, guard_intervention=0.0)
    scaled_second = scaled.calculate(native_equity=final, guard_intervention=0.0)

    assert scaled_first.r_econ == raw_first.r_econ
    assert scaled_first.r_shape == raw_first.r_shape
    assert scaled_first.r_total == pytest.approx(raw_first.r_total * 10_000.0)
    assert scaled_second.r_total == pytest.approx(raw_second.r_total * 10_000.0)
    assert sum((raw_first.r_econ, raw_second.r_econ)) == pytest.approx(math.log(final / initial))
    assert scaled.export_state() == raw.export_state()


def test_reward_scale_applies_to_shaping_and_insolvency_penalty() -> None:
    config = RewardConfig(lambda_dd=0.5, lambda_guard=0.25, reward_scale=100.0)
    calculator = RewardCalculator(config)
    calculator.reset(100.0)
    losing = calculator.calculate(native_equity=80.0, guard_intervention=1.0)
    assert losing.r_shape == pytest.approx(-0.5 * 0.2 - 0.25)
    assert losing.r_total == pytest.approx((losing.r_econ + losing.r_shape) * 100.0)

    insolvent = calculator.calculate(native_equity=0.0, guard_intervention=1.0)
    assert insolvent.insolvency
    assert insolvent.r_total == pytest.approx((-10.0 + insolvent.r_shape) * 100.0)


def test_drawdown_shaping_changes_learner_reward_but_preserves_economic_return() -> None:
    raw = RewardCalculator(RewardConfig(reward_scale=1_000.0))
    shaped = RewardCalculator(RewardConfig(lambda_dd=0.1, reward_scale=1_000.0))
    raw.reset(100.0)
    shaped.reset(100.0)

    raw_step = raw.calculate(native_equity=95.0, guard_intervention=0.0)
    shaped_step = shaped.calculate(native_equity=95.0, guard_intervention=0.0)

    assert shaped_step.r_econ == raw_step.r_econ
    assert shaped_step.r_shape == pytest.approx(-0.1 * 0.05)
    assert shaped_step.r_total == pytest.approx((shaped_step.r_econ - 0.005) * 1_000.0)
    assert raw_step.r_total == pytest.approx(raw_step.r_econ * 1_000.0)


@pytest.mark.parametrize("value", [0, -1, 1_000_001, True, float("inf"), float("nan")])
def test_reward_scale_rejects_invalid_values(value: object) -> None:
    with pytest.raises(ValueError):
        RewardConfig(reward_scale=value)  # type: ignore[arg-type]


@pytest.mark.parametrize("field,value", (
    ("lambda_dd", True), ("lambda_dd", -0.1), ("lambda_guard", float("inf")),
    ("insolvent_penalty", 0.0),
))
def test_invalid_reward_shaping_parameters_are_rejected(field: str, value: object) -> None:
    with pytest.raises(ValueError):
        RewardConfig(**{field: value})  # type: ignore[arg-type]
