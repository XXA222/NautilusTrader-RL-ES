from __future__ import annotations

import pytest

from rles_research.baselines import BaselinePolicy, BaselineSpec
from rles_research.metrics import MetricsError, series_metrics
from rles_research.selection import SelectionSpec, choose
from rles_research.statistics import StatisticsError, block_bootstrap

pytestmark = pytest.mark.unit

_OVERSIZED_INTEGER = 10**10_000


def test_economic_metrics_reject_integers_outside_float_domain_as_metrics_errors() -> None:
    with pytest.raises(MetricsError, match="finite numeric"):
        series_metrics([_OVERSIZED_INTEGER, 0.0], periods_per_year=1)
    with pytest.raises(MetricsError, match="valid frequency/quantile"):
        series_metrics([0.01, -0.01], periods_per_year=_OVERSIZED_INTEGER)


def test_bootstrap_rejects_oversized_integer_as_statistics_error() -> None:
    with pytest.raises(StatisticsError, match="nonfinite or non-real returns"):
        block_bootstrap(
            [0.01, -0.02, 0.015, _OVERSIZED_INTEGER],
            block_length=2,
            draws=100,
            seed=7,
            confidence=0.95,
            statistic=lambda sample: sum(sample) / len(sample),
        )


def test_candidate_selection_rejects_oversized_integer_metric() -> None:
    spec = SelectionSpec(0.5, 2.0, 1, "median_fold_sharpe")
    candidate = {
        "candidate_id": "oversized",
        "liquidations": 0,
        "max_drawdown": 0.2,
        "turnover": 0.5,
        "risk_violations": 0,
        "median_fold_net_return": 0.01,
        "median_fold_sharpe": _OVERSIZED_INTEGER,
    }
    with pytest.raises(ValueError, match="nonfinite candidate metric"):
        choose(spec, (candidate,))


def test_candidate_selection_rejects_negative_validation_net_return() -> None:
    spec = SelectionSpec(0.5, 2.0, 1, "median_fold_sharpe")
    candidate = {
        "candidate_id": "negative",
        "liquidations": 0,
        "max_drawdown": 0.2,
        "turnover": 0.5,
        "risk_violations": 0,
        "median_fold_net_return": -0.001,
        "median_fold_sharpe": 1.0,
    }
    assert choose(spec, (candidate,)) == {
        "status": "NO_FEASIBLE_MODEL",
        "rejected": {"negative": "NONPOSITIVE_NET_RETURN"},
        "selection_spec_sha256": spec.sha256(),
    }


def test_candidate_selection_rejects_zero_net_return_as_nonprofitable() -> None:
    spec = SelectionSpec(0.5, 2.0, 1, "median_fold_sharpe")
    candidate = {
        "candidate_id": "flat",
        "liquidations": 0,
        "max_drawdown": 0.0,
        "turnover": 0.0,
        "risk_violations": 0,
        "median_fold_net_return": 0.0,
        "median_fold_sharpe": 0.0,
    }
    result = choose(spec, (candidate,))
    assert result["status"] == "NO_FEASIBLE_MODEL"
    assert result["rejected"] == {"flat": "NONPOSITIVE_NET_RETURN"}


def test_baseline_rejects_oversized_market_price() -> None:
    policy = BaselinePolicy(BaselineSpec("B2_simple_momentum", seed=1))
    with pytest.raises(ValueError, match="invalid historic close"):
        policy.action([1.0, _OVERSIZED_INTEGER])


def test_baseline_spec_rejects_oversized_target() -> None:
    with pytest.raises(ValueError, match="invalid preregistered baseline parameters"):
        BaselineSpec("B1_always_long", seed=1, target_long=_OVERSIZED_INTEGER)
