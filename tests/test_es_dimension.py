from __future__ import annotations

import numpy as np
import pytest
from decimal import Decimal

from rles_es_evotorch.provider import EvolutionConfig, EvoTorchProvider, EvolutionResult
from rles_es_evotorch.soft_risk import SoftRiskDimension, SoftRiskSpace
from rles_risk.guards import SoftRiskParams
from rles_orchestrator.es_stage import _select_e2_candidate

pytestmark = pytest.mark.algorithm_smoke


def test_two_independent_e2_multipliers_have_exact_neutral_physical_candidate() -> None:
    baseline = SoftRiskParams(Decimal("0.8"), Decimal("0.4"), Decimal("0.2"), Decimal("0"))
    space = SoftRiskSpace((SoftRiskDimension("exposure_multiplier", 0.1, 1.0),
                           SoftRiskDimension("drawdown_throttle", 0.1, 1.0)))
    assert baseline.with_multipliers(space.decode(np.asarray([1.0, 1.0]))) == baseline
    exposure = baseline.with_multipliers(space.decode(np.asarray([0.5, 1.0])))
    drawdown = baseline.with_multipliers(space.decode(np.asarray([1.0, 0.5])))
    assert exposure.target_scale == Decimal("0.4")
    assert exposure.drawdown_throttle == baseline.drawdown_throttle
    assert drawdown.target_scale == baseline.target_scale
    assert drawdown.drawdown_throttle == Decimal("0.6")


def test_e2_selection_prefers_exact_neutral_on_equal_or_better_fitness() -> None:
    space = SoftRiskSpace((SoftRiskDimension("exposure_multiplier", 0.1, 1.0),
                           SoftRiskDimension("drawdown_throttle", 0.1, 1.0)))
    optimizer = EvolutionResult("cmaes", 2, 8, 0.5, (0.2, -0.1),
                                (("drawdown_throttle", 0.8), ("exposure_multiplier", 0.7)))

    neutral_win = _select_e2_candidate(optimizer, space, (0.5, 1, "a" * 64))
    assert neutral_win.evaluations == 10
    assert neutral_win.best_fitness == 0.5
    assert neutral_win.selected_candidate_kind == "neutral_r1"
    assert neutral_win.risk_parameters == (("drawdown_throttle", 1.0),
                                           ("exposure_multiplier", 1.0))

    optimizer_win = _select_e2_candidate(optimizer, space, (0.4, 1, "b" * 64))
    assert optimizer_win.evaluations == 10
    assert optimizer_win.selected_candidate_kind == "optimizer"
    assert optimizer_win.risk_parameters == optimizer.risk_parameters


@pytest.mark.parametrize("center", (None, np.asarray([0.0], dtype=np.float32)))
def test_cmaes_supports_one_dimensional_soft_risk_space(center: np.ndarray | None) -> None:
    risk_space = SoftRiskSpace((SoftRiskDimension("exposure_multiplier", 0.1, 1.0),))
    config = EvolutionConfig(
        algorithm="cmaes",
        population_size=4,
        generations=1,
        seed=17,
        max_evaluations=4,
    )

    def fitness(candidate: np.ndarray | dict[str, float], *, split: str) -> float:
        assert split == "train_inner"
        values = np.asarray(list(candidate.values()) if isinstance(candidate, dict) else candidate)
        assert values.shape == (1,)
        return -float(np.square(values).sum())

    result = EvoTorchProvider(config, fitness, risk_space=risk_space).evolve(
        dimension=1, center=center
    )
    assert result.evaluations == 4
    assert len(result.best_search_vector) == 1
    assert len(result.risk_parameters or ()) == 1
