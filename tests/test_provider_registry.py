from __future__ import annotations

from hashlib import sha256

import numpy as np
import pytest

from rles_contracts.registry import ProviderDescriptor, ProviderRegistry
from rles_es_evotorch.provider import EvolutionConfig

pytestmark = pytest.mark.contracts


@pytest.mark.parametrize(
    ("kind", "name"),
    (("engine", "nt_rc6"), ("rl", "sb3"), ("es", "evotorch"), ("compute", "local")),
)
def test_installed_provider_entrypoints_satisfy_l1_contract(kind: str, name: str) -> None:
    info, descriptor = ProviderRegistry().load(kind, name)
    assert info.kind == kind
    assert isinstance(descriptor, ProviderDescriptor)
    assert descriptor.kind == kind
    assert descriptor.contract_version == "0.4.0"
    assert callable(descriptor.create)


def test_compute_provider_descriptor_creates_with_explicit_empty_settings() -> None:
    _, descriptor = ProviderRegistry().load("compute", "local")
    implementation = descriptor.create({})
    assert implementation.__class__.__name__ == "LocalComputeProvider"


def test_es_entrypoint_factory_supports_batch_only_evolution() -> None:
    _, descriptor = ProviderRegistry().load("es", "evotorch")
    calls: list[tuple[int, tuple[int, ...]]] = []

    def batch_fitness(candidates: np.ndarray, *, generation: int) -> np.ndarray:
        calls.append((generation, candidates.shape))
        return -np.square(candidates).sum(axis=1)

    provider = descriptor.create({
        "config": EvolutionConfig("pgpe", 4, 1, 812, max_evaluations=4),
        "batch_fitness": batch_fitness,
        "batch_fitness_id": sha256(b"registry-batch-fitness-v1").hexdigest(),
    })
    result = provider.evolve(dimension=2)

    assert result.generations == 1
    assert result.evaluations == 4
    assert np.isfinite(result.best_fitness)
    assert calls == [(0, (4, 2))]


def test_provider_descriptor_rejects_non_mapping_settings() -> None:
    descriptor = ProviderDescriptor("compute", "0.4.0", lambda settings: settings)
    with pytest.raises(TypeError, match="mapping"):
        descriptor.create([])  # type: ignore[arg-type]
