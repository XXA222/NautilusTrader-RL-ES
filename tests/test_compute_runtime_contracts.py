from __future__ import annotations

import pytest

from rles_compute_runtime.benchmark import (
    benchmark, device_decision, validate_benchmark_report, worker_scaling_benchmark,
)
from rles_compute_runtime.errors import ComputeError, ComputeProtocolError
from rles_compute_runtime.workers import _json_object

pytestmark = pytest.mark.unit


_FINGERPRINTS = {
    name: "a" * 64
    for name in (
        "dataset_sha256",
        "seed_tree_sha256",
        "actor_signature_sha256",
        "execution_sha256",
        "risk_sha256",
        "algorithm_config_sha256",
    )
}


def _sample(_mode: str, _trial: int) -> dict[str, int | float]:
    return {
        "wall_sec": 2.0,
        "p95_action_latency_ms": 3.0,
        "peak_rss_bytes": 1024,
        "peak_vram_bytes": 0,
        "pcie_txrx_bytes": 0,
        "numerical_max_abs_error": 0.0,
        "env_steps": 100,
        "es_evaluations": 0,
        "model_params": 500,
    }


def test_benchmark_report_is_recomputed_and_cpu_decision_is_stable() -> None:
    report = benchmark(["cpu"], _sample, frozen_fingerprints=_FINGERPRINTS)

    assert validate_benchmark_report(report)
    decision = device_decision(
        report,
        max_p95_latency_ms=10,
        max_numeric_error=1e-5,
        reviewer_approved=True,
    )
    assert decision["selected_topology"] == "cpu"


def test_benchmark_rejects_nonfinite_probe_values() -> None:
    def invalid(_mode: str, _trial: int) -> dict[str, int | float]:
        sample = _sample(_mode, _trial)
        sample["wall_sec"] = float("nan")
        return sample

    with pytest.raises(ComputeError, match="invalid benchmark reading"):
        benchmark(["cpu"], invalid, frozen_fingerprints=_FINGERPRINTS)


def test_benchmark_verifier_rejects_tampered_samples() -> None:
    report = benchmark(["cpu"], _sample, frozen_fingerprints=_FINGERPRINTS)
    report["samples"]["cpu"][0]["wall_sec"] = float("inf")

    with pytest.raises(ComputeError, match="invalid benchmark reading"):
        validate_benchmark_report(report)


def test_worker_payload_must_be_a_json_object() -> None:
    assert _json_object({"step": 1}, "payload") == {"step": 1}
    with pytest.raises(ComputeProtocolError, match="string-keyed JSON object"):
        _json_object([1, 2], "payload")


def test_benchmark_maps_oversized_integer_to_validation_error() -> None:
    def invalid(mode: str, trial: int) -> dict[str, int | float]:
        sample = _sample(mode, trial)
        sample["wall_sec"] = 10 ** 1000
        return sample

    with pytest.raises(ComputeError, match="invalid benchmark reading"):
        benchmark(["cpu"], invalid, frozen_fingerprints=_FINGERPRINTS)


@pytest.mark.parametrize("field", ["env_steps", "es_evaluations", "model_params"])
def test_benchmark_requires_integer_workload_counters(field: str) -> None:
    def invalid(mode: str, trial: int) -> dict[str, int | float]:
        sample = _sample(mode, trial)
        sample[field] = 1.5
        return sample

    with pytest.raises(ComputeError, match="integer counter"):
        benchmark(["cpu"], invalid, frozen_fingerprints=_FINGERPRINTS)


@pytest.mark.parametrize("repetitions", [3.5, "3", None])
def test_benchmark_validates_repeat_budget_before_running_probe(repetitions: object) -> None:
    with pytest.raises(ValueError, match="independent repeats"):
        benchmark(
            ["cpu"], lambda *_: pytest.fail("invalid budget must not run a probe"),
            frozen_fingerprints=_FINGERPRINTS, repetitions=repetitions,
        )


def test_benchmark_approval_requires_explicit_boolean_true() -> None:
    report = benchmark(["cpu"], _sample, frozen_fingerprints=_FINGERPRINTS)
    with pytest.raises(ComputeError, match="explicit human approval"):
        device_decision(
            report, max_p95_latency_ms=10, max_numeric_error=1e-5,
            reviewer_approved="yes",
        )


def test_benchmark_report_rejects_boolean_schema_version() -> None:
    report = benchmark(["cpu"], _sample, frozen_fingerprints=_FINGERPRINTS)
    report["schema_version"] = True
    with pytest.raises(ComputeError, match="schema drift"):
        validate_benchmark_report(report)


def test_benchmark_summary_cannot_hide_boolean_values_behind_numeric_equality() -> None:
    report = benchmark(["cpu"], _sample, frozen_fingerprints=_FINGERPRINTS)
    report["summary"]["cpu"]["peak_vram_bytes"] = False
    with pytest.raises(ComputeError, match="summary does not agree"):
        validate_benchmark_report(report)


def test_benchmark_workload_budget_cannot_use_boolean_zero() -> None:
    report = benchmark(["cpu"], _sample, frozen_fingerprints=_FINGERPRINTS)
    report["sample_budget"]["es_evaluations"] = False
    with pytest.raises(ComputeError, match="invalid benchmark sample budget"):
        validate_benchmark_report(report)


def test_device_decision_rejects_oversized_numeric_threshold() -> None:
    report = benchmark(["cpu"], _sample, frozen_fingerprints=_FINGERPRINTS)
    with pytest.raises(ValueError, match="acceptance threshold"):
        device_decision(report, max_p95_latency_ms=10, max_numeric_error=10 ** 1000,
                        reviewer_approved=True)


def test_worker_scaling_validates_repeat_budget_before_running_probe() -> None:
    with pytest.raises(ComputeError, match="3 repeats"):
        worker_scaling_benchmark(
            lambda *_: pytest.fail("invalid budget must not run a probe"),
            frozen_fingerprints=_FINGERPRINTS, repetitions=3.5, max_peak_rss_bytes=4096,
        )
