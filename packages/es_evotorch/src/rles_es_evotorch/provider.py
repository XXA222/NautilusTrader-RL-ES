"""EvoTorch-only PGPE(E1) and CMA-ES(E2) search adapter.

The optimiser update/sampling equations belong exclusively to EvoTorch.
For E2, CMA-ES searches unbounded latent values; SoftRiskSpace maps every
candidate into risk-reducing multipliers. Searcher native resume remains
supported on the pinned EvoTorch 0.6.1 CPU path by an explicit native-state
codec; CUDA and external data-cursor restoration remain blocked.
"""
from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping
from typing import Any, Protocol, cast
from io import BytesIO
import importlib.metadata
import json
import math
import random
import re
from pathlib import Path

import numpy as np

from rles_contracts.registry import ProviderDescriptor

from .soft_risk import SoftRiskSpace
from .checkpoint import ESCheckpointEnvelope, ESCheckpointStore


_EVOTORCH_CODEC = "evotorch-0.6.1-native-v2"
_CHECKPOINT_COMPONENTS = frozenset({"distribution", "optimizer", "population",
                                    "rng_torch", "rng_numpy", "rng_python", "rng_evotorch",
                                    "batch_evaluator"})


def _encode_state(value: object) -> bytes:
    """Encode primitive trees and numeric arrays without pickle/object arrays."""
    arrays: dict[str, np.ndarray] = {}

    def visit(item: object) -> object:
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - guarded by provider construction
            raise RuntimeError("PyTorch required for native ES state codec") from exc
        if isinstance(item, torch.Tensor):
            item = item.detach().cpu().numpy()
            tag = "tensor"
        elif isinstance(item, np.ndarray):
            tag = "array"
        elif isinstance(item, np.generic):
            item = item.item()
            return visit(item)
        else:
            tag = ""
        if tag:
            array = np.ascontiguousarray(item)
            if array.dtype.hasobject or array.dtype.kind not in "biuf":
                raise ValueError("native ES state accepts numeric arrays only")
            key = f"a{len(arrays):06d}"
            arrays[key] = array
            return {"$array": key, "$kind": tag}
        if item is None or type(item) in {bool, int, str}:
            return item
        if type(item) is float:
            if not math.isfinite(item):
                raise ValueError("native ES state contains nonfinite scalar")
            return item
        if isinstance(item, tuple):
            return {"$tuple": [visit(part) for part in item]}
        if isinstance(item, list):
            return [visit(part) for part in item]
        if isinstance(item, dict):
            if any(not isinstance(key, str) for key in item):
                raise ValueError("native ES state mappings require string keys")
            return {key: visit(part) for key, part in sorted(item.items())}
        raise TypeError(f"unsupported native ES state type: {type(item).__name__}")

    metadata = json.dumps(visit(value), sort_keys=True, separators=(",", ":"), allow_nan=False)
    stream = BytesIO()
    np.savez_compressed(stream, __metadata__=np.asarray(metadata), **arrays)  # type: ignore[arg-type]
    return stream.getvalue()


def _decode_state(blob: bytes) -> object:
    try:
        with np.load(BytesIO(blob), allow_pickle=False) as stored:
            if "__metadata__" not in stored.files:
                raise ValueError("missing native state metadata")
            metadata = json.loads(str(stored["__metadata__"].item()))
            arrays = {key: stored[key].copy() for key in stored.files if key != "__metadata__"}
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError("invalid native ES state component") from exc

    def visit(item: object) -> object:
        if isinstance(item, list):
            return [visit(part) for part in item]
        if isinstance(item, dict):
            if set(item) == {"$array", "$kind"}:
                key, kind = item["$array"], item["$kind"]
                if not isinstance(key, str) or key not in arrays or kind not in {"array", "tensor"}:
                    raise ValueError("invalid native state array reference")
                value = arrays.pop(key)
                if value.dtype.hasobject or value.dtype.kind not in "biuf":
                    raise ValueError("object or nonnumeric state array rejected")
                return value
            if set(item) == {"$tuple"}:
                return tuple(visit(part) for part in item["$tuple"])
            if any(key.startswith("$") for key in item):
                raise ValueError("unknown native state metadata tag")
            return {key: visit(part) for key, part in item.items()}
        return item

    result = visit(metadata)
    if arrays:
        raise ValueError("unreferenced native state arrays")
    return result


class TrainInnerEvaluator(Protocol):
    def __call__(self, candidate: np.ndarray | dict[str, float], *, split: str) -> float: ...


class BatchTrainInnerEvaluator(Protocol):
    def __call__(self, candidates: np.ndarray, *, generation: int) -> np.ndarray: ...


@dataclass(frozen=True)
class EvolutionConfig:
    algorithm: str
    population_size: int
    generations: int
    seed: int
    stdev_init: float = 0.1
    center_learning_rate: float = 0.05
    stdev_learning_rate: float = 0.1
    max_evaluations: int = 10_000
    device: str = "cpu"
    gpu_lease_id: str | None = None

    def __post_init__(self) -> None:
        if self.algorithm not in {"pgpe", "cmaes"}:
            raise ValueError("algorithm must be pgpe or cmaes")
        if (type(self.population_size) is not int or self.population_size < 4
                or (self.algorithm == "pgpe" and self.population_size % 2 != 0)
                or type(self.generations) is not int or self.generations < 1
                or type(self.seed) is not int or self.seed < 0):
            raise ValueError("invalid ES generation, seed, or population")
        if type(self.max_evaluations) is not int or self.max_evaluations < 1:
            raise ValueError("invalid ES evaluation budget")
        if self.population_size * self.generations > self.max_evaluations:
            raise ValueError("population * generations exceeds evaluation cap")
        if self.device != "cpu" and not (isinstance(self.device, str) and
                re.fullmatch(r"cuda:[0-9]+", self.device)):
            raise ValueError("ES device must be explicit cpu or cuda:<index>")
        if self.device != "cpu" and not self.gpu_lease_id:
            raise ValueError("CUDA ES requires explicit L6 GPU lease identity")
        if self.device == "cpu" and self.gpu_lease_id is not None:
            raise ValueError("CPU ES must not retain a GPU lease")
        for name in ("stdev_init", "center_learning_rate", "stdev_learning_rate"):
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, (float, int)) or not math.isfinite(v) or not (0 < v <= 1):
                raise ValueError(f"invalid EvoTorch {name}")


@dataclass(frozen=True)
class EvolutionResult:
    algorithm: str
    generations: int
    evaluations: int
    best_fitness: float
    best_search_vector: tuple[float, ...]
    risk_parameters: tuple[tuple[str, float], ...] | None
    selected_candidate_kind: str = "optimizer"


class EvoTorchProvider:
    def __init__(self, config: EvolutionConfig, fitness: TrainInnerEvaluator | None = None,
                 *, risk_space: SoftRiskSpace | None = None,
                 batch_fitness: BatchTrainInnerEvaluator | None = None,
                 batch_fitness_id: str | None = None):
        if (not isinstance(config, EvolutionConfig)
                or (fitness is not None and not callable(fitness))
                or (fitness is None and batch_fitness is None)):
            raise TypeError("invalid EvoTorch provider dependencies")
        if (config.algorithm == "cmaes") != (risk_space is not None):
            raise ValueError("CMA-ES requires a bounded SoftRiskSpace; PGPE must not receive it")
        if (batch_fitness is None) != (batch_fitness_id is None):
            raise ValueError("batch fitness and its frozen identity must be supplied together")
        if batch_fitness is not None and not callable(batch_fitness):
            raise TypeError("batch fitness must be callable")
        if batch_fitness_id is not None and (not isinstance(batch_fitness_id, str)
                or not re.fullmatch(r"[0-9a-f]{64}", batch_fitness_id)):
            raise ValueError("batch fitness identity must be lowercase SHA256")
        self.config = config
        self.fitness = fitness
        self.batch_fitness = batch_fitness
        self.batch_fitness_id = batch_fitness_id
        self.risk_space = risk_space
        self._searcher: Any | None = None
        self._generation = 0
        self._evaluations = 0
        self._best_fitness = -math.inf
        self._best_vector: np.ndarray | None = None
        self._dimension: int | None = None
        self._search_dimension: int | None = None
        self._initial_bounds: tuple[float, float] | None = None
        self._center: np.ndarray | None = None
        self._failed = False

    def _config_identity(self) -> dict[str, object]:
        return {
            "algorithm": self.config.algorithm,
            "population_size": self.config.population_size,
            "generations": self.config.generations,
            "seed": self.config.seed,
            "stdev_init": float(self.config.stdev_init),
            "center_learning_rate": float(self.config.center_learning_rate),
            "stdev_learning_rate": float(self.config.stdev_learning_rate),
            "max_evaluations": self.config.max_evaluations,
            "device": self.config.device,
            "risk_space_signature": self.risk_space.signature if self.risk_space else None,
            "batch_fitness_id": self.batch_fitness_id,
        }

    def evolve(self, *, dimension: int, initial_bounds: tuple[float, float] = (-0.1, 0.1),
               center: np.ndarray | None = None,
               generations: int | None = None) -> EvolutionResult:
        if self._searcher is not None or type(dimension) is not int or dimension < 1:
            raise RuntimeError("provider is single-use and requires positive dimensionality")
        if self.risk_space is not None and dimension != len(self.risk_space.dimensions):
            raise ValueError("CMA-ES dimension does not match SoftRiskSpace")
        if (len(initial_bounds) != 2 or not np.isfinite(initial_bounds).all()
                or initial_bounds[0] >= initial_bounds[1]):
            raise ValueError("invalid finite search initialization interval")
        if center is not None:
            center = np.asarray(center)
            if center.shape != (dimension,) or center.dtype != np.float32 or not np.isfinite(center).all():
                raise ValueError("initial center must be finite float32 with exact genome shape")
        if generations is not None and (type(generations) is not int or not 1 <= generations <= self.config.generations):
            raise ValueError("initial generation count must fit the frozen generation budget")
        self._dimension = dimension
        self._search_dimension = 2 if self.config.algorithm == "cmaes" and dimension == 1 else dimension
        self._initial_bounds = (float(initial_bounds[0]), float(initial_bounds[1]))
        self._center = None if center is None else center.copy()
        self._searcher = self._create_searcher(dimension, self._search_dimension,
                                               self._initial_bounds, self._center)
        return self.continue_evolution(generations=generations)

    def _create_searcher(self, dimension: int, search_dimension: int,
                         initial_bounds: tuple[float, float], center: np.ndarray | None):
        try:
            from evotorch import Problem
            from evotorch.algorithms import PGPE, CMAES
            import torch
        except ImportError as exc:
            raise RuntimeError("G0 BLOCKED: EvoTorch dependency is not installed") from exc

        if self.config.device != "cpu" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable; no automatic CPU fallback")

        def objective(x: object) -> object:
            # Scalar mode receives one candidate; batch mode receives a population matrix.
            output_device = x.device if isinstance(x, torch.Tensor) else torch.device("cpu")
            if isinstance(x, torch.Tensor):
                vector = x.detach().to(device="cpu", dtype=torch.float32).numpy()
            else:
                vector = np.asarray(x, dtype=np.float32)
            if self.batch_fitness is not None:
                if (vector.ndim != 2 or vector.shape[0] < 1
                        or vector.shape[1] != search_dimension or not np.isfinite(vector).all()):
                    raise ValueError("EvoTorch population dimension or finiteness mismatch")
                if self._evaluations + vector.shape[0] > self.config.max_evaluations:
                    raise RuntimeError("ES evaluation cap exceeded")
                genomes = vector[:, :dimension].copy()
                if self.risk_space is None:
                    physical_batch = genomes
                else:
                    names = tuple(item.name for item in self.risk_space.dimensions)
                    physical_batch = np.asarray([
                        [self.risk_space.from_latent(row)[name] for name in names]
                        for row in genomes
                    ], dtype=np.float32)
                values = np.asarray(self.batch_fitness(physical_batch, generation=self._generation))
                if values.dtype.kind not in "fiu" or values.shape != (vector.shape[0],) or not np.isfinite(values).all():
                    raise ValueError("batch Train-inner fitness must return one finite score per candidate")
                values = values.astype(np.float32, copy=False)
                self._evaluations += vector.shape[0]
                for genome, value in zip(genomes, values, strict=True):
                    if float(value) > self._best_fitness:
                        self._best_fitness = float(value)
                        self._best_vector = genome.copy()
                return torch.as_tensor(values, dtype=torch.float32, device=output_device)
            if vector.shape != (search_dimension,) or not np.isfinite(vector).all():
                raise ValueError("EvoTorch candidate dimension or finiteness mismatch")
            if self._evaluations >= self.config.max_evaluations:
                raise RuntimeError("ES evaluation cap exceeded")
            genome = vector[:dimension].copy()
            if self.fitness is None:
                raise RuntimeError("scalar fitness is unavailable for this batch-only provider")
            physical_candidate = self.risk_space.from_latent(genome) if self.risk_space else genome
            value = float(self.fitness(physical_candidate, split="train_inner"))
            if not math.isfinite(value):
                raise ValueError("nonfinite Train-inner fitness")
            self._evaluations += 1
            if value > self._best_fitness:
                self._best_fitness = value
                self._best_vector = genome
            return value

        # PGPE needs distribution initialisation, NOT a bounded problem.
        # CMAES may reject bounded problems in EvoTorch: never set `bounds`.
        problem = Problem("max", objective, solution_length=search_dimension,
                          initial_bounds=initial_bounds, dtype="float32", seed=self.config.seed,
                          vectorized=self.batch_fitness is not None, device=self.config.device)
        if self.config.algorithm == "pgpe":
            pgpe_args: dict[str, Any] = dict(popsize=self.config.population_size,
                        center_learning_rate=self.config.center_learning_rate,
                        stdev_learning_rate=self.config.stdev_learning_rate,
                        stdev_init=self.config.stdev_init, symmetric=True)
            if center is not None:
                pgpe_args["center_init"] = torch.as_tensor(center, dtype=torch.float32, device=self.config.device)
            searcher = PGPE(problem, **pgpe_args)
        else:
            cmaes_args: dict[str, Any] = dict(popsize=self.config.population_size, stdev_init=self.config.stdev_init)
            if center is not None:
                center_init = center
                if search_dimension != dimension:
                    center_init = np.pad(center, (0, search_dimension - dimension))
                cmaes_args["center_init"] = torch.as_tensor(
                    center_init, dtype=torch.float32, device=self.config.device
                )
            searcher = CMAES(problem, **cmaes_args)
        return searcher

    def continue_evolution(self, *, generations: int | None = None) -> EvolutionResult:
        if self._searcher is None or self._failed or self._generation >= self.config.generations:
            raise RuntimeError("provider is uninitialized or its frozen generation budget is exhausted")
        remaining = self.config.generations - self._generation
        count = remaining if generations is None else generations
        if type(count) is not int or not 1 <= count <= remaining:
            raise ValueError("continuation generation count exceeds frozen budget")
        for _ in range(count):
            try:
                self._searcher.step()
            except BaseException:
                self._failed = True
                raise
            self._generation += 1
        return self.result()

    def result(self) -> EvolutionResult:
        if self._searcher is None or self._best_vector is None:
            raise RuntimeError("provider has no completed generation")
        return EvolutionResult(
            self.config.algorithm, self._generation, self._evaluations, self._best_fitness,
            tuple(float(x) for x in self._best_vector),
            tuple(sorted(self.risk_space.from_latent(self._best_vector).items())) if self.risk_space else None,
        )

    def evaluate(self, genome: np.ndarray, *, split: str) -> float:
        if split != "train_inner":
            raise PermissionError("ES evaluation cannot consume Validation/Test")
        if self._searcher is None:
            raise RuntimeError("provider has not been initialised")
        value = np.asarray(genome)
        if value.shape != (self._dimension,) or value.dtype != np.float32 or not np.isfinite(value).all():
            raise ValueError("invalid genome")
        if self.risk_space is not None and len(value) != len(self.risk_space.dimensions):
            raise ValueError("invalid E2 risk latent dimension")
        if self._evaluations >= self.config.max_evaluations:
            raise RuntimeError("ES evaluation budget exhausted")
        if self.fitness is None:
            raise RuntimeError("scalar evaluate is unavailable for this batch-only provider")
        result = float(self.fitness(self.risk_space.from_latent(value) if self.risk_space else value.copy(),
                                    split="train_inner"))
        if not math.isfinite(result):
            raise ValueError("nonfinite evaluator result")
        self._evaluations += 1
        return result

    @staticmethod
    def _tensor_like(value: object, reference: object):
        import torch
        if not isinstance(value, np.ndarray):
            raise ValueError("native state tensor shape/type mismatch")
        if isinstance(reference, torch.Tensor):
            dtype, device, expected_shape = reference.dtype, reference.device, reference.shape
        elif type(reference) in {float, int}:
            dtype, device, expected_shape = torch.float32, torch.device("cpu"), value.shape
        else:
            raise ValueError("native state template type mismatch")
        tensor = torch.as_tensor(value, dtype=dtype, device=device)
        if tensor.shape != expected_shape or not torch.isfinite(tensor).all():
            raise ValueError("native state tensor shape/finiteness mismatch")
        return tensor.clone()

    def _state_components(self) -> dict[str, bytes]:
        if (self._searcher is None or self._failed or self._generation < 1
                or self._dimension is None or self._search_dimension is None):
            raise RuntimeError("checkpoint requires a successful generation boundary")
        if self.config.device != "cpu":
            raise RuntimeError("native EvoTorch checkpoint codec currently certifies CPU only")
        import torch

        searcher = self._searcher
        if self.config.algorithm == "pgpe":
            distribution = {"kind": "pgpe", "steps_count": searcher._steps_count,
                            "first_iter": searcher._first_iter,
                            "parameters": searcher._distribution._Distribution__parameters}
            optimizer = searcher._optimizer
            if optimizer is None or type(optimizer).__name__ != "ClipUp":
                raise RuntimeError("unsupported PGPE optimizer state codec")
            optimizer_state = {"class": "ClipUp", "velocity": optimizer._velocity}
        else:
            fields = ("m", "sigma", "C", "A", "p_sigma", "p_c")
            distribution = {"kind": "cmaes", "steps_count": searcher._steps_count,
                            "fields": {name: getattr(searcher, name) for name in fields}}
            optimizer_state = {"class": "none"}

        population = searcher._population
        if population is None:
            raise RuntimeError("searcher has no completed population to checkpoint")
        population_state = {
            "values": population.access_values(keep_evals=True),
            "evaluations": population.access_evals(),
        }
        provider_state = {
            "config": self._config_identity(),
            "dimension": self._dimension,
            "search_dimension": self._search_dimension,
            "initial_bounds": self._initial_bounds,
            "center": self._center,
            "generation": self._generation,
            "evaluations": self._evaluations,
            "best_fitness": self._best_fitness,
            "best_vector": self._best_vector,
        }
        evaluator_state = None
        if self.batch_fitness is not None:
            snapshot = getattr(self.batch_fitness, "checkpoint_state", None)
            if not callable(snapshot):
                raise RuntimeError("batch evaluator must expose checkpoint_state for safe resume")
            evaluator_state = snapshot()
        components = {
            "distribution": _encode_state(distribution),
            "optimizer": _encode_state(optimizer_state),
            "population": _encode_state(population_state),
            "rng_torch": _encode_state(torch.get_rng_state()),
            "rng_numpy": _encode_state(np.random.get_state()),
            "rng_python": _encode_state(random.getstate()),
            "rng_evotorch": _encode_state({
                "version": importlib.metadata.version("evotorch"),
                "problem_generator": searcher.problem._generator.get_state(),
            }),
            "provider": _encode_state(provider_state),
            "batch_evaluator": _encode_state(evaluator_state),
        }
        return components

    def save(self, path: str | Path, *, attempt_id: str, actor_signature: str,
             dataset_hash: str, config_hash: str, data_cursor: int = 0) -> str:
        """Save a CPU generation-boundary checkpoint using a versioned, pickle-free codec."""
        if type(data_cursor) is not int or data_cursor != 0:
            raise ValueError("provider can checkpoint only at its generation boundary with cursor 0")
        components = self._state_components()
        codec = _EVOTORCH_CODEC
        if importlib.metadata.version("evotorch") != "0.6.1":
            raise RuntimeError("native checkpoint codec is pinned to EvoTorch 0.6.1")
        envelope = ESCheckpointEnvelope(
            algorithm=self.config.algorithm, attempt_id=attempt_id,
            actor_signature=actor_signature, dataset_hash=dataset_hash,
            config_hash=config_hash, generation=self._generation,
            evaluations=self._evaluations, data_cursor=data_cursor,
            native_codec_id=codec, state_components=components,
        )
        return str(ESCheckpointStore(path).save(envelope))

    def resume(self, path: str | Path, *, attempt_id: str, actor_signature: str,
               dataset_hash: str, config_hash: str) -> int:
        """Atomically restore complete optimizer, population, provider and RNG state."""
        if self._searcher is not None or self._generation != 0 or self._evaluations != 0:
            raise RuntimeError("resume requires a fresh provider")
        import torch
        torch_rng = torch.get_rng_state().clone()
        numpy_rng = np.random.get_state()
        python_rng = random.getstate()
        try:
            return self._resume_impl(path, attempt_id=attempt_id,
                                     actor_signature=actor_signature,
                                     dataset_hash=dataset_hash, config_hash=config_hash)
        except BaseException:
            torch.set_rng_state(torch_rng)
            np.random.set_state(numpy_rng)
            random.setstate(python_rng)
            self._searcher = None
            self._dimension = None
            self._search_dimension = None
            self._initial_bounds = None
            self._center = None
            self._generation = 0
            self._evaluations = 0
            self._best_fitness = -math.inf
            self._best_vector = None
            self._failed = True
            raise

    def _resume_impl(self, path: str | Path, *, attempt_id: str, actor_signature: str,
                     dataset_hash: str, config_hash: str) -> int:
        """Version-pinned native restore implementation; invoked transactionally by resume."""
        if self._searcher is not None or self._generation != 0 or self._evaluations != 0:
            raise RuntimeError("resume requires a fresh provider")
        if self.config.device != "cpu":
            raise RuntimeError("native EvoTorch checkpoint codec currently certifies CPU only")
        if importlib.metadata.version("evotorch") != "0.6.1":
            raise RuntimeError("native checkpoint codec is pinned to EvoTorch 0.6.1")
        envelope = ESCheckpointStore(path).load(attempt_id)
        if (envelope.native_codec_id != _EVOTORCH_CODEC
                or envelope.algorithm != self.config.algorithm
                or envelope.actor_signature != actor_signature
                or envelope.dataset_hash != dataset_hash
                or envelope.config_hash != config_hash
                or envelope.generation < 1 or envelope.generation > self.config.generations
                or envelope.evaluations > self.config.max_evaluations
                or envelope.data_cursor != 0):
            raise ValueError("checkpoint identity, codec, budget or cursor mismatch")
        if set(envelope.state_components) != _CHECKPOINT_COMPONENTS | {"provider"}:
            raise ValueError("native checkpoint component set mismatch")
        decoded = {key: _decode_state(value) for key, value in envelope.state_components.items()}
        provider_state = decoded["provider"]
        evaluator_state = decoded["batch_evaluator"]
        if ((self.batch_fitness is None and evaluator_state is not None)
                or (self.batch_fitness is not None and evaluator_state is None)):
            raise ValueError("checkpoint batch evaluator mode differs from current provider")
        if not isinstance(provider_state, dict):
            raise ValueError("invalid provider continuation state")
        if provider_state.get("config") != self._config_identity():
            raise ValueError("checkpoint optimizer configuration mismatch")
        if (provider_state.get("generation") != envelope.generation
                or provider_state.get("evaluations") != envelope.evaluations):
            raise ValueError("provider continuation counters mismatch checkpoint envelope")
        dimension = provider_state.get("dimension")
        search_dimension = provider_state.get("search_dimension")
        bounds = provider_state.get("initial_bounds")
        center = provider_state.get("center")
        if (type(dimension) is not int or dimension < 1 or type(search_dimension) is not int
                or search_dimension != (2 if self.config.algorithm == "cmaes" and dimension == 1 else dimension)
                or not isinstance(bounds, tuple) or len(bounds) != 2
                or any(type(value) not in {float, int} or not math.isfinite(value) for value in bounds)
                or bounds[0] >= bounds[1]):
            raise ValueError("invalid provider dimensions/bounds in checkpoint")
        if self.risk_space is not None and dimension != len(self.risk_space.dimensions):
            raise ValueError("checkpoint latent dimension differs from SoftRiskSpace")
        if center is not None and (not isinstance(center, np.ndarray) or center.dtype != np.float32
                                   or center.shape != (dimension,)):
            raise ValueError("invalid initial center in checkpoint")
        self._dimension, self._search_dimension = dimension, search_dimension
        self._initial_bounds = (float(bounds[0]), float(bounds[1]))
        self._center = center
        self._searcher = self._create_searcher(dimension, search_dimension,
                                               self._initial_bounds, center)
        searcher = self._searcher
        if searcher is None:  # defensive narrowing for static checkers
            raise RuntimeError("searcher construction failed")
        distribution = decoded["distribution"]
        optimizer = decoded["optimizer"]
        population_state = decoded["population"]
        if not isinstance(distribution, dict) or not isinstance(optimizer, dict) or not isinstance(population_state, dict):
            raise ValueError("malformed optimizer checkpoint components")
        if type(distribution.get("steps_count")) is not int or distribution["steps_count"] != envelope.generation:
            raise ValueError("optimizer step counter differs from checkpoint generation")
        if self.config.algorithm == "pgpe":
            if distribution.get("kind") != "pgpe" or distribution.get("first_iter") is not False:
                raise ValueError("PGPE checkpoint is not at a complete generation boundary")
            parameters = distribution.get("parameters")
            if not isinstance(parameters, dict) or set(parameters) != {"mu", "sigma", "divide_mu_grad_by", "divide_sigma_grad_by"}:
                raise ValueError("PGPE distribution parameter schema drift")
            searcher._distribution = searcher._distribution.modified_copy(
                **{key: self._tensor_like(parameters[key], getattr(searcher._distribution, key))
                   if key in {"mu", "sigma"} else parameters[key] for key in parameters}
            )
            if optimizer.get("class") != "ClipUp":
                raise ValueError("PGPE optimizer codec mismatch")
            searcher._optimizer._velocity = self._tensor_like(optimizer.get("velocity"),
                                                               searcher._optimizer._velocity)
            searcher._first_iter = False
        else:
            fields = distribution.get("fields")
            expected = {"m", "sigma", "C", "A", "p_sigma", "p_c"}
            if distribution.get("kind") != "cmaes" or not isinstance(fields, dict) or set(fields) != expected:
                raise ValueError("CMA-ES distribution state schema drift")
            for name, value in fields.items():
                reference = getattr(searcher, name)
                if name == "sigma" and hasattr(reference, "numel") and reference.numel() == 1:
                    reference = reference.reshape(1)
                restored_value = self._tensor_like(value, reference)
                expected_shape = {
                    "m": (search_dimension,), "sigma": (1,),
                    "C": (search_dimension, search_dimension),
                    "A": (search_dimension, search_dimension),
                    "p_sigma": (search_dimension,), "p_c": (search_dimension,),
                }[name]
                if tuple(restored_value.shape) != expected_shape:
                    raise ValueError(f"CMA-ES {name} shape mismatch")
                setattr(searcher, name, restored_value)
            if optimizer.get("class") != "none":
                raise ValueError("unexpected CMA-ES optimizer state")
        searcher._steps_count = distribution["steps_count"]

        values, evaluations = population_state.get("values"), population_state.get("evaluations")
        if not isinstance(values, np.ndarray) or not isinstance(evaluations, np.ndarray):
            raise ValueError("missing checkpoint population/evaluations")
        if searcher._population is None:
            from evotorch import SolutionBatch
            searcher._population = SolutionBatch(searcher.problem, popsize=len(values), empty=True)
        if tuple(values.shape) != tuple(searcher._population.access_values().shape):
            raise ValueError("checkpoint population shape mismatch")
        searcher._population.access_values()[:] = self._tensor_like(
            values, searcher._population.access_values())
        searcher._population.access_evals()[:] = self._tensor_like(
            evaluations, searcher._population.access_evals())

        rng_evo = decoded["rng_evotorch"]
        if (not isinstance(rng_evo, dict) or rng_evo.get("version") != "0.6.1"):
            raise ValueError("EvoTorch RNG codec version mismatch")
        import torch
        searcher.problem._generator.set_state(torch.as_tensor(
            rng_evo["problem_generator"], dtype=torch.uint8, device="cpu"))
        torch.set_rng_state(torch.as_tensor(decoded["rng_torch"], dtype=torch.uint8, device="cpu"))
        np.random.set_state(cast(Any, decoded["rng_numpy"]))
        random.setstate(cast(Any, decoded["rng_python"]))
        self._generation = envelope.generation
        self._evaluations = envelope.evaluations
        self._best_fitness = float(provider_state["best_fitness"])
        best_vector = provider_state["best_vector"]
        if not isinstance(best_vector, np.ndarray) or best_vector.shape != (dimension,) or not np.isfinite(best_vector).all():
            raise ValueError("invalid best-so-far candidate")
        self._best_vector = best_vector.astype(np.float32, copy=True)
        if self.batch_fitness is not None:
            restore = getattr(self.batch_fitness, "restore_checkpoint_state", None)
            if not callable(restore):
                raise RuntimeError("batch evaluator must expose restore_checkpoint_state")
            restore(evaluator_state)
        return envelope.data_cursor


def _create_es(settings: Mapping[str, object]) -> EvoTorchProvider:
    allowed = {"config", "fitness", "risk_space", "batch_fitness", "batch_fitness_id"}
    if "config" not in settings or set(settings) - allowed:
        raise ValueError("ES provider settings must contain config and a fitness evaluator")
    config = settings["config"]
    fitness = settings.get("fitness")
    batch_fitness = settings.get("batch_fitness")
    batch_fitness_id = settings.get("batch_fitness_id")
    risk_space = settings.get("risk_space")
    if (not isinstance(config, EvolutionConfig)
            or (fitness is not None and not callable(fitness))
            or (batch_fitness is not None and not callable(batch_fitness))
            or (fitness is None and batch_fitness is None)
            or (batch_fitness_id is not None and not isinstance(batch_fitness_id, str))
            or (risk_space is not None and not isinstance(risk_space, SoftRiskSpace))):
        raise TypeError("invalid typed dependencies for the EvoTorch provider")
    return EvoTorchProvider(config, fitness, risk_space=risk_space,
                            batch_fitness=batch_fitness, batch_fitness_id=batch_fitness_id)


provider = ProviderDescriptor(kind="es", contract_version="0.4.0", factory=_create_es)
