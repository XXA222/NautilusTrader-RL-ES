from __future__ import annotations

import pytest

from rles_es_evotorch.fitness import FitnessSpec


def test_episode_fitness_uses_native_facts_and_worst_tail_losses() -> None:
    score = FitnessSpec().score_episode(
        split="train_inner", initial_equity=100.0, final_equity=90.0,
        step_log_returns=(0.1, -0.2, 0.05), drawdowns=(0.0, 0.2, 0.15),
        risk_intervention_rate=0.5, tail_fraction=0.34,
    )
    # net return -.1, mean of worst ceil(3*.34)=2 losses (.2, 0),
    # max drawdown .2, risk intervention .5 at a 0.05 default weight.
    assert score == pytest.approx(-0.425)


def test_single_hard_risk_intervention_does_not_overwhelm_one_percent_return() -> None:
    score = FitnessSpec().score_episode(
        split="train_inner", initial_equity=100.0, final_equity=101.0,
        step_log_returns=(0.01,) * 64, drawdowns=(0.0,) * 64,
        risk_intervention_rate=1 / 64, tail_fraction=0.25,
    )
    legacy_score = 0.01 - 1.0
    inactive_score = 0.0
    assert legacy_score < inactive_score < score
    assert score == pytest.approx(0.01 - 0.05 / 64)


@pytest.mark.parametrize("facts", [
    {"drawdowns": (0.0, 0.1)},
    {"step_log_returns": ()},
    {"step_log_returns": (float("nan"),), "drawdowns": (0.0,)},
    {"step_log_returns": (0.0,), "drawdowns": (1.1,)},
])
def test_episode_fitness_rejects_incomplete_or_invalid_facts(facts: dict) -> None:
    arguments = {
        "split": "train_inner", "initial_equity": 100.0, "final_equity": 100.0,
        "step_log_returns": (0.0,), "drawdowns": (0.0,),
        "risk_intervention_rate": 0.0, "tail_fraction": 0.1,
    }
    arguments.update(facts)
    with pytest.raises(ValueError):
        FitnessSpec().score_episode(**arguments)


def test_episode_fitness_cannot_consume_non_training_split() -> None:
    with pytest.raises(PermissionError):
        FitnessSpec().score_episode(
            split="validation", initial_equity=100.0, final_equity=101.0,
            step_log_returns=(0.01,), drawdowns=(0.0,),
            risk_intervention_rate=0.0, tail_fraction=0.1,
        )
