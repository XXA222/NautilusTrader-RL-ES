from __future__ import annotations

import numpy as np
import pytest

from rles_es_evotorch.provider import EvolutionConfig, EvoTorchProvider
from rles_es_evotorch.soft_risk import SoftRiskDimension, SoftRiskSpace

pytestmark = pytest.mark.algorithm_smoke


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
