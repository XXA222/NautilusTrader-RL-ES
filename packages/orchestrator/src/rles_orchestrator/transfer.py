"""Four-algorithm E1 -> R1 transfer with explicit L4 evidence and failure gates.

L4 provides native SB3 actor load/reset/BC operations through callables.
No SB3 internals or files are accessed here. BC is *never* weight inheritance.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import math
from typing import Callable

import numpy as np

_ALGOS = frozenset({"PPO", "SAC", "TD3", "DDPG"})
_HEX = frozenset("0123456789abcdef")


def _hash(value: str) -> None:
    if not isinstance(value, str) or len(value) != 64 or any(x not in _HEX for x in value):
        raise ValueError("required lowercase SHA256 hash")


def _obs(value: np.ndarray, name: str) -> np.ndarray:
    x = np.asarray(value)
    if x.dtype != np.float32 or x.ndim != 2 or x.shape[0] < 1 or x.shape[1] < 1 or not np.isfinite(x).all():
        raise ValueError(f"{name}: nonempty finite float32 [batch, features] required")
    return x


def _actions(f: Callable[[np.ndarray], np.ndarray], obs: np.ndarray) -> np.ndarray:
    y = np.asarray(f(obs))
    if y.dtype != np.float32 or y.shape != (obs.shape[0], 1) or not np.isfinite(y).all():
        raise ValueError("deterministic normalized actor action must be finite float32 [batch, 1]")
    if (y < -1.0).any() or (y > 1.0).any():
        raise ValueError("SB3/Canonical action is outside Box[-1,1]")
    return y


@dataclass(frozen=True)
class ParityReport:
    algorithm: str
    actor_signature: str
    observations_hash: str
    count: int
    max_abs_error: float
    mean_abs_error: float
    p95_abs_error: float
    root_mean_squared_error: float
    tolerance: float
    passed: bool

    def __post_init__(self) -> None:
        if self.algorithm not in _ALGOS:
            raise ValueError("unsupported parity algorithm")
        _hash(self.actor_signature)
        _hash(self.observations_hash)
        if type(self.count) is not int or self.count < 1 or type(self.passed) is not bool:
            raise ValueError("invalid parity sample count / pass flag")
        values = (self.max_abs_error, self.mean_abs_error, self.p95_abs_error,
                  self.root_mean_squared_error, self.tolerance)
        if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) or x < 0 for x in values):
            raise ValueError("invalid parity error distribution")
        if any(x > self.max_abs_error + 1e-12 for x in values[1:4]):
            raise ValueError("parity distribution exceeds stated maximum")
        if self.passed != (self.max_abs_error <= self.tolerance):
            raise ValueError("parity pass flag conflicts with actual threshold")


def check_parity(algorithm: str, teacher: Callable[[np.ndarray], np.ndarray],
                 student: Callable[[np.ndarray], np.ndarray], observations: np.ndarray,
                 *, actor_signature: str, absolute_tolerance: float) -> ParityReport:
    if algorithm not in _ALGOS:
        raise ValueError("unsupported SB3 algorithm")
    _hash(actor_signature)
    if isinstance(absolute_tolerance, bool) or not isinstance(absolute_tolerance, (float, int)) or not math.isfinite(absolute_tolerance) or absolute_tolerance < 0:
        raise ValueError("invalid action parity tolerance")
    x = _obs(observations, "fixed parity observations")
    t1, t2 = _actions(teacher, x), _actions(teacher, x)
    s1, s2 = _actions(student, x), _actions(student, x)
    if not np.array_equal(t1, t2) or not np.array_equal(s1, s2):
        raise RuntimeError("stochastic actor used in deterministic transfer parity")
    errors = np.abs(t1.astype(np.float64) - s1.astype(np.float64)).ravel()
    max_error = float(errors.max())
    return ParityReport(algorithm, actor_signature, sha256(x.tobytes(order="C")).hexdigest(),
                        int(len(x)), max_error, float(errors.mean()),
                        float(np.percentile(errors, 95)), float(np.sqrt(np.mean(errors ** 2))),
                        float(absolute_tolerance), max_error <= absolute_tolerance)


@dataclass(frozen=True)
class ParameterLoadEvidence:
    actor_signature: str
    expected_parameters: int
    loaded_parameters: int
    missing_keys: tuple[str, ...]
    unexpected_keys: tuple[str, ...]
    shape_dtype_verified: bool

    def __post_init__(self) -> None:
        _hash(self.actor_signature)
        if (type(self.expected_parameters) is not int or self.expected_parameters <= 0
                or type(self.loaded_parameters) is not int or self.loaded_parameters < 0
                or self.loaded_parameters > self.expected_parameters):
            raise ValueError("invalid parameter coverage")
        if (type(self.shape_dtype_verified) is not bool
                or not isinstance(self.missing_keys, tuple)
                or not isinstance(self.unexpected_keys, tuple)
                or any(not isinstance(x, str) or not x for x in self.missing_keys + self.unexpected_keys)):
            raise ValueError("invalid per-layer key/shape/dtype evidence")

    @property
    def exact(self) -> bool:
        return (self.expected_parameters == self.loaded_parameters and not self.missing_keys
                and not self.unexpected_keys and self.shape_dtype_verified)


@dataclass(frozen=True)
class DistillationEvidence:
    training_split: str
    training_samples: int
    teacher_dataset_hash: str
    imitation_mse: float
    epochs: int

    def __post_init__(self) -> None:
        if (self.training_split != "train_inner" or type(self.training_samples) is not int
                or self.training_samples < 1 or type(self.epochs) is not int or self.epochs < 1
                or not math.isfinite(self.imitation_mse) or self.imitation_mse < 0):
            raise ValueError("invalid Train-inner behavior-cloning evidence")
        _hash(self.teacher_dataset_hash)


@dataclass(frozen=True)
class TransferResult:
    algorithm: str
    method: str
    parity: ParityReport
    load_evidence: ParameterLoadEvidence | None
    distillation: DistillationEvidence | None
    training_split: str

    def __post_init__(self) -> None:
        if self.algorithm not in _ALGOS or self.algorithm != self.parity.algorithm:
            raise ValueError("transfer/parity algorithm mismatch")
        if self.training_split != "train_inner" or not self.parity.passed:
            raise ValueError("unverified training split or action parity")
        if self.method == "DIRECT":
            if self.load_evidence is None or not self.load_evidence.exact or self.distillation is not None:
                raise ValueError("direct transfer requires full native parameter coverage")
            if self.load_evidence.actor_signature != self.parity.actor_signature:
                raise ValueError("native transfer signature mismatch")
        elif self.method == "DISTILLATION_NOT_WEIGHT_INHERITANCE":
            if self.distillation is None or self.distillation.training_split != "train_inner":
                raise ValueError("distillation requires train-inner imitation evidence")
        else:
            raise ValueError("unknown transfer method")


class StructuralTransferMismatch(RuntimeError):
    """Only L4 can establish a genuine native-actor structural mismatch."""


def adapt_l4_load(loader: Callable[[], object], *, source_actor_signature: str,
                  action_parity_mismatch: type[Exception]) -> Callable[[], ParameterLoadEvidence]:
    """Adapt an L4 ``RLProvider.transfer`` callback to the L5 evidence contract.

    L4 remains independent of L5. The caller injects L4's typed action-mismatch
    exception; only that condition becomes an eligible L5 distillation fallback.
    Mapping, signature, coverage, and dtype errors propagate unchanged.
    """
    if not callable(loader):
        raise TypeError("L4 transfer loader must be callable")
    _hash(source_actor_signature)
    if (not isinstance(action_parity_mismatch, type)
            or not issubclass(action_parity_mismatch, Exception)):
        raise TypeError("typed L4 action-parity exception class required")

    def load() -> ParameterLoadEvidence:
        try:
            report = loader()
        except action_parity_mismatch as exc:
            # L4 transfer_actor rolls back the exact load before raising this
            # specific type, so the generic L5 transfer can start BC cleanly.
            raise StructuralTransferMismatch(str(exc)) from exc
        method = getattr(report, "method", None)
        signature = getattr(report, "actor_signature", None)
        passed = getattr(report, "passed", None)
        expected = getattr(report, "expected_tensors", None)
        loaded = getattr(report, "loaded_tensors", None)
        if (method != "exact_weight_transfer" or signature != source_actor_signature
                or passed is not True or type(expected) is not int or expected < 1
                or type(loaded) is not int or loaded != expected):
            raise RuntimeError("L4 did not attest a complete, parity-passing exact actor load")
        # L4 reports these counts only after its mapping/dtype validators and
        # exact copy succeed; this conversion does not infer coverage from keys.
        return ParameterLoadEvidence(source_actor_signature, expected, loaded, (), (), True)

    return load


def adapt_l4_distillation(distill: Callable[[np.ndarray, np.ndarray], object], *,
                          student: Callable[[np.ndarray], np.ndarray],
                          epochs: int) -> Callable[[np.ndarray, np.ndarray], DistillationEvidence]:
    """Adapt L4's checked BC result into L5's dataset/MSE evidence contract."""
    if not callable(distill) or not callable(student) or type(epochs) is not int or epochs < 1:
        raise ValueError("L4 distiller, student actor, and positive epoch count required")

    def run(observations: np.ndarray, labels: np.ndarray) -> DistillationEvidence:
        train = _obs(observations, "Train-inner distillation observations")
        target = _actions(lambda _batch: labels, train)
        report = distill(train.copy(), target.copy())
        if getattr(report, "method", None) != "behavioral_distillation":
            raise RuntimeError("L4 did not attest successful Train-inner behavioral distillation")
        if getattr(report, "samples", None) != len(train):
            raise RuntimeError("L4 behavioral distillation sample count differs from Train-inner input")
        if getattr(report, "passed", None) is not True:
            raise RuntimeError(
                "L4 Train-inner behavioral distillation parity failed: "
                f"max_abs_error={getattr(report, 'max_abs_error', 'unknown')}, "
                f"tolerance={getattr(report, 'tolerance', 'unknown')}"
            )
        completed_epochs = getattr(report, "epochs_completed", epochs)
        if type(completed_epochs) is not int:
            raise RuntimeError("L4 behavioral distillation epoch count must be an integer")
        if completed_epochs == 0:
            completed_epochs = epochs  # Older L4 reports omitted actual epoch accounting.
        if not 1 <= completed_epochs <= epochs:
            raise RuntimeError("L4 behavioral distillation epoch count exceeds its frozen budget")
        actual = _actions(student, train)
        mse = float(np.mean((target.astype(np.float64) - actual.astype(np.float64)) ** 2))
        dataset_hash = sha256(train.tobytes(order="C") + target.tobytes(order="C")).hexdigest()
        return DistillationEvidence("train_inner", len(train), dataset_hash, mse, completed_epochs)

    return run


def transfer(*, algorithm: str, actor_signature: str,
             teacher: Callable[[np.ndarray], np.ndarray],
             student: Callable[[np.ndarray], np.ndarray],
             fixed_observations: np.ndarray,
             training_observations: np.ndarray,
             load_exact: Callable[[], ParameterLoadEvidence],
             reset_student: Callable[[], None],
             distill: Callable[[np.ndarray, np.ndarray], DistillationEvidence],
             tolerance: float, training_split: str = "train_inner") -> TransferResult:
    if training_split != "train_inner":
        raise PermissionError("transfer adaptation may only train on Train-inner")
    _hash(actor_signature)
    train = _obs(training_observations, "behavior cloning train observations")
    fixed = _obs(fixed_observations, "fixed parity observations")
    if fixed.shape[1] != train.shape[1]:
        raise ValueError("transfer observation dimension mismatch")
    evidence: ParameterLoadEvidence | None = None
    structurally_incompatible = False
    try:
        evidence = load_exact()
    except StructuralTransferMismatch:
        structurally_incompatible = True
    if not structurally_incompatible:
        if not isinstance(evidence, ParameterLoadEvidence):
            raise TypeError("L4 loader must return ParameterLoadEvidence")
        if evidence.actor_signature != actor_signature:
            raise RuntimeError("native actor signature differs from canonical actor")
        if not evidence.exact:
            raise RuntimeError("partial strict weight load: missing / extra / dtype mismatch")
        before = check_parity(algorithm, teacher, student, fixed,
                              actor_signature=actor_signature, absolute_tolerance=tolerance)
        if before.passed:
            return TransferResult(algorithm, "DIRECT", before, evidence, None, "train_inner")
    # Direct parity failed or math non-isomorphism established by L4.
    # Reset is mandatory so a mixed partially loaded student never masquerades as BC.
    reset_student()
    labels = _actions(teacher, train).copy()
    distillation = distill(train.copy(), labels)
    if not isinstance(distillation, DistillationEvidence):
        raise TypeError("L4 distiller must return train-inner imitation evidence")
    if distillation.training_samples != len(train):
        raise RuntimeError("distillation sample count does not match provided teacher corpus")
    observed_mse = float(np.mean((labels.astype(np.float64) -
                                   _actions(student, train).astype(np.float64)) ** 2))
    if not math.isclose(distillation.imitation_mse, observed_mse, abs_tol=1e-8, rel_tol=1e-4):
        raise RuntimeError("distillation reported imitation error not supported by student actions")
    if distillation.teacher_dataset_hash != sha256(train.tobytes(order="C") + labels.tobytes(order="C")).hexdigest():
        raise RuntimeError("distillation teacher dataset lineage mismatch")
    parity = check_parity(algorithm, teacher, student, fixed,
                          actor_signature=actor_signature, absolute_tolerance=tolerance)
    if not parity.passed:
        raise RuntimeError("distillation action parity failed: transfer must be blocked")
    return TransferResult(algorithm, "DISTILLATION_NOT_WEIGHT_INHERITANCE", parity,
                          evidence, distillation, "train_inner")


@dataclass(frozen=True)
class TransferJob:
    """A typed L4 direct-load/BC capability for exactly one SB3 algorithm."""
    algorithm: str
    student_id: str
    student: Callable[[np.ndarray], np.ndarray]
    load_exact: Callable[[], ParameterLoadEvidence]
    reset_student: Callable[[], None]
    distill: Callable[[np.ndarray, np.ndarray], DistillationEvidence]

    def __post_init__(self) -> None:
        if self.algorithm not in _ALGOS or not isinstance(self.student_id, str) or not self.student_id:
            raise ValueError("invalid native SB3 transfer job identity")
        if not all(callable(x) for x in (self.student, self.load_exact, self.reset_student, self.distill)):
            raise ValueError("L4 native transfer callbacks must be explicit")


def transfer_all(*, jobs: tuple[TransferJob, ...], actor_signature: str,
                 teacher: Callable[[np.ndarray], np.ndarray],
                 fixed_observations: np.ndarray, training_observations: np.ndarray,
                 tolerance: float) -> tuple[TransferResult, ...]:
    """Full PPO/SAC/TD3/DDPG matrix or exception. Never report partial as PASS."""
    if not isinstance(jobs, tuple) or {j.algorithm for j in jobs} != _ALGOS or len(jobs) != 4:
        raise ValueError("all four SB3 transfer algorithms required")
    if len({j.student_id for j in jobs}) != 4:
        raise ValueError("one independent destination model is required per algorithm")
    results = []
    for job in sorted(jobs, key=lambda x: x.algorithm):
        result = transfer(
            algorithm=job.algorithm, actor_signature=actor_signature,
            teacher=teacher, student=job.student,
            fixed_observations=fixed_observations,
            training_observations=training_observations,
            load_exact=job.load_exact, reset_student=job.reset_student,
            distill=job.distill, tolerance=tolerance,
            training_split="train_inner",
        )
        results.append(result)
    return tuple(results)
