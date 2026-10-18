from __future__ import annotations

import pytest

from rles_es_evotorch.fitness import (
    ConstraintPriorityFitnessSpec, FeasibilityFirstFitnessSpec, FitnessSpec,
)


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


def test_explicit_economic_fitness_ablation_can_rank_profitable_candidate_above_flat() -> None:
    legacy = FitnessSpec()
    economic = FitnessSpec(tail_weight=0.0, drawdown_weight=0.0,
                           risk_intervention_weight=0.0)
    facts = dict(split="train_inner", net_return=0.01, tail_loss=0.002,
                 max_drawdown=0.004, risk_intervention_rate=0.2)
    flat = dict(split="train_inner", net_return=0.0, tail_loss=0.0,
                max_drawdown=0.0, risk_intervention_rate=0.0)
    assert legacy.score(**facts) < legacy.score(**flat)
    assert economic.score(**facts) > economic.score(**flat)


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


def test_constraint_priority_ranks_profitable_flat_and_losing_episodes_after_feasibility():
    spec = ConstraintPriorityFitnessSpec()

    def decide(final_equity: float, step_return: float, *, intervention_rate: float = 0.0):
        return spec.evaluate_episode(
            split="train_inner", initial_equity=100.0, final_equity=final_equity,
            step_log_returns=(step_return,), drawdowns=(max(0.0, -step_return),),
            risk_intervention_rate=intervention_rate, tail_fraction=1.0,
        )

    profitable = decide(110.0, 0.1)
    flat = decide(100.0, 0.0)
    losing = decide(90.0, -0.1)
    insolvent = decide(0.0, -1.0)
    assert profitable.feasible and flat.feasible and losing.feasible
    assert profitable.score > flat.score > losing.score
    assert insolvent.feasible is False
    assert insolvent.score == -2.0 < losing.score
    assert insolvent.rejection_reason == "episode_insolvency_detected"
    transiently_insolvent = spec.evaluate_episode(
        split="train_inner", initial_equity=100.0, final_equity=105.0,
        step_log_returns=(0.05,), drawdowns=(0.0,), risk_intervention_rate=0.0,
        tail_fraction=1.0, insolvency=True,
    )
    assert transiently_insolvent.feasible is False
    assert transiently_insolvent.score == -2.0

    margin_breach = spec.evaluate_episode(
        split="train_inner", initial_equity=100.0, final_equity=100.0,
        step_log_returns=(0.0,), drawdowns=(0.0,), risk_intervention_rate=0.0,
        tail_fraction=1.0, maintenance_margin_breach=True,
    )
    liquidation = spec.evaluate_episode(
        split="train_inner", initial_equity=100.0, final_equity=100.0,
        step_log_returns=(0.0,), drawdowns=(0.0,), risk_intervention_rate=0.0,
        tail_fraction=1.0, native_liquidation_detected=True,
    )
    assert margin_breach.score == liquidation.score == -2.0
    assert margin_breach.rejection_reason == "maintenance_margin_breach_detected"
    assert liquidation.rejection_reason == "native_liquidation_detected"


def test_constraint_priority_is_bounded_and_does_not_penalize_hard_risk_interventions():
    spec = ConstraintPriorityFitnessSpec()
    facts = dict(split="train_inner", initial_equity=100.0, final_equity=1e100,
                 step_log_returns=(1e100,), drawdowns=(0.0,), tail_fraction=1.0)
    no_intervention = spec.evaluate_episode(**facts, risk_intervention_rate=0.0)
    many_interventions = spec.evaluate_episode(**facts, risk_intervention_rate=1.0)
    assert no_intervention.score == many_interventions.score == pytest.approx(1.0)
    assert no_intervention.score == 1.0  # v1 retains its prior floating-point endpoint.
    assert no_intervention.rejection_reason is None


@pytest.mark.parametrize("spec", [ConstraintPriorityFitnessSpec(), FeasibilityFirstFitnessSpec()])
def test_constraint_priority_classifies_negative_terminal_equity_as_insolvency(spec):
    decision = spec.evaluate_episode(
        split="train_inner", initial_equity=100.0, final_equity=-5.0,
        step_log_returns=(-1.0,), drawdowns=(1.0,), risk_intervention_rate=0.0,
        tail_fraction=1.0,
    )

    assert decision.feasible is False
    assert decision.score == -2.0
    assert decision.rejection_reason == "episode_insolvency_detected"

    # The legacy protocol remains strict; only the versioned constraint-first
    # protocols reinterpret a finite negative terminal balance as insolvency.
    with pytest.raises(ValueError, match="episode equity facts"):
        FitnessSpec().score_episode(
            split="train_inner", initial_equity=100.0, final_equity=-5.0,
            step_log_returns=(-1.0,), drawdowns=(1.0,), risk_intervention_rate=0.0,
            tail_fraction=1.0,
        )


def test_feasibility_first_scores_stay_strictly_inside_selection_bounds() -> None:
    spec = FeasibilityFirstFitnessSpec()

    profitable = spec.evaluate_episode(
        split="train_inner", initial_equity=100.0, final_equity=1e100,
        step_log_returns=(1e100,), drawdowns=(0.0,), risk_intervention_rate=0.0,
        tail_fraction=1.0,
    )
    losing = spec.evaluate_episode(
        split="train_inner", initial_equity=100.0, final_equity=1.0,
        step_log_returns=(-1e100,), drawdowns=(1.0,), risk_intervention_rate=0.0,
        tail_fraction=1.0,
    )

    assert profitable.feasible and losing.feasible
    assert -1.0 < losing.score < 0.0 < profitable.score < 1.0


def test_three_fitness_protocols_compare_the_same_deterministic_episode_facts():
    candidates = {
        "profitable": dict(final_equity=101.5, step_log_returns=(0.01, 0.005),
                            drawdowns=(0.0, 0.0), risk_intervention_rate=0.75),
        "flat": dict(final_equity=100.0, step_log_returns=(0.0, 0.0),
                     drawdowns=(0.0, 0.0), risk_intervention_rate=0.0),
        "losing": dict(final_equity=98.0, step_log_returns=(-0.01, -0.01),
                       drawdowns=(0.01, 0.02), risk_intervention_rate=0.25),
        "insolvent": dict(final_equity=0.0, step_log_returns=(-1.0,),
                          drawdowns=(1.0,), risk_intervention_rate=0.0),
    }
    legacy, no_intervention_penalty = FitnessSpec(), FitnessSpec(risk_intervention_weight=0.0)
    constraint_priority = ConstraintPriorityFitnessSpec()

    def legacy_scores(spec: FitnessSpec) -> dict[str, float]:
        return {name: spec.score_episode(
            split="train_inner", initial_equity=100.0, tail_fraction=0.5, **facts,
        ) for name, facts in candidates.items()}

    legacy_scores_by_candidate = legacy_scores(legacy)
    no_intervention_scores = legacy_scores(no_intervention_penalty)
    constraint_scores = {
        name: constraint_priority.evaluate_episode(
            split="train_inner", initial_equity=100.0, tail_fraction=0.5, **facts,
        ).score for name, facts in candidates.items()
    }
    assert legacy_scores_by_candidate["flat"] > legacy_scores_by_candidate["profitable"]
    assert no_intervention_scores["profitable"] > no_intervention_scores["flat"]
    assert constraint_scores["profitable"] > constraint_scores["flat"]
    assert constraint_scores["flat"] > constraint_scores["losing"]
    assert constraint_scores["losing"] > constraint_scores["insolvent"] == -2.0
