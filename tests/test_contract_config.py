from __future__ import annotations

import pytest

from rles_contracts.config import FoldConfig
from rles_contracts.errors import RLESError

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("scheme", ("60/15/15", "360/15/15"))
def test_l1_accepts_only_frozen_wfo_schemes_with_fifteen_day_step(scheme: str) -> None:
    config = FoldConfig(
        scheme=scheme,
        step_days=15,
        warmup=5,
        purge="auto",
        embargo="1d",
        terminal_policy="force_flatten_with_cost",
    )
    assert config.scheme == scheme


@pytest.mark.parametrize(
    ("scheme", "step_days"),
    (("30/15/15", 15), ("60/15/15", 1), ("360/15/15", 30)),
)
def test_l1_rejects_wfo_config_that_downstream_cannot_execute(
    scheme: str, step_days: int
) -> None:
    with pytest.raises(RLESError, match="15-day step is supported"):
        FoldConfig(
            scheme=scheme,
            step_days=step_days,
            warmup=5,
            purge="auto",
            embargo="1d",
            terminal_policy="force_flatten_with_cost",
        )


def test_l1_rejects_zero_warmup_days() -> None:
    with pytest.raises(RLESError, match="fold.warmup must be integer >= 1"):
        FoldConfig(
            scheme="60/15/15",
            step_days=15,
            warmup=0,
            purge=0,
            embargo="1d",
            terminal_policy="force_flatten_with_cost",
        )
