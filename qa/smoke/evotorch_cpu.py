"""One-generation CPU smoke for PGPE and CMA-ES provider wiring."""

from __future__ import annotations

import numpy as np

from rles_es_evotorch.provider import EvolutionConfig, EvoTorchProvider
from rles_es_evotorch.soft_risk import SoftRiskDimension, SoftRiskSpace


def fitness(candidate: np.ndarray | dict[str, float], *, split: str) -> float:
    assert split == "train_inner"
    vector = np.asarray(
        list(candidate.values()) if isinstance(candidate, dict) else candidate,
        dtype=np.float64,
    )
    return -float(np.square(vector).sum())


def main() -> None:
    risk_space = SoftRiskSpace(
        (
            SoftRiskDimension("exposure_multiplier", 0.1, 1.0),
            SoftRiskDimension("drawdown_throttle", 0.1, 1.0),
            SoftRiskDimension("volatility_throttle", 0.1, 1.0),
        )
    )
    for algorithm in ("pgpe", "cmaes"):
        risk = None if algorithm == "pgpe" else risk_space
        config = EvolutionConfig(
            algorithm=algorithm,
            population_size=4,
            generations=1,
            seed=731,
            max_evaluations=4,
        )
        result = EvoTorchProvider(config, fitness, risk_space=risk).evolve(dimension=3)
        assert result.generations == 1 and result.evaluations == 4
        if result.risk_parameters is not None:
            assert all(0.1 <= value <= 1.0 for _, value in result.risk_parameters)
        print(f"PASS {algorithm}: generations=1 evaluations=4")


if __name__ == "__main__":
    main()
