"""Small native-engine golden driven by an authentic Binance market archive.

Economics are explicitly scenario-only because dated historical account fees
and all instrument filters are not established by the public archive fixture.
"""
from __future__ import annotations

from bisect import bisect_right
from decimal import Decimal
from hashlib import sha256
import json
from math import log
from pathlib import Path

import numpy as np
import pytest

from rles_data.io import read_curated
from rles_gym_bridge.historical import HistoricalEnvironmentFactory
from rles_research.native_metrics import native_report


pytestmark = pytest.mark.real_golden

FIXTURE = Path(__file__).parent / "fixtures" / "real_ethusdt_20240101"
MARKET_SHA256 = "6957e1237d69cbe2fd2ee37a0a92486bf8f0114229c0a649c0ecee7a53c97c8f"


def test_real_ethusdt_native_account_trace_matches_scenario_golden():
    market_path = FIXTURE / "market.parquet"
    assert sha256(market_path.read_bytes()).hexdigest() == MARKET_SHA256

    archive = json.loads((FIXTURE / "import.json").read_text(encoding="utf-8"))
    assert archive["schema"] == "rles.binance-archive.v1"
    assert archive["event_count"] == 4328
    assert len(archive["sources"]) == 4
    for source in archive["sources"]:
        raw_path = FIXTURE / source["path"]
        assert sha256(raw_path.read_bytes()).hexdigest() == source["sha256"]

    events = read_curated(market_path).events
    assert len(events) == archive["event_count"]
    assert {event.kind for event in events} >= {
        "ohlcv_1m", "mark", "index", "funding_update", "funding_settlement"
    }

    project = Path(__file__).resolve().parents[1]
    config = json.loads((project / "configs" / "ethusdt_20240101_real.json").read_text(encoding="utf-8"))
    environment = config["environment"]
    policy = dict(environment["coverage_policy"])
    policy["settlement_times_ns"] = {"ETHUSDT-PERP.SIM": [1704096000000000000]}
    factory = HistoricalEnvironmentFactory(
        market_path=market_path,
        rules_path=FIXTURE / "scenario_rules.json",
        instrument="ETHUSDT-PERP.SIM",
        start_ns=1704067500000000000,  # 00:05 UTC, with feature warmup available.
        end_ns=1704096300000000000,  # 08:05 UTC; includes the real 08:00 settlement.
        coverage_policy=policy,
        execution=environment["execution"],
        hard_risk=environment["hard_risk"],
        soft_risk=environment["soft_risk"],
        starting_balance=environment["starting_balance"],
        risk_budget_notional=environment["risk_budget_notional"],
        fit_scaler=True,
    )

    env = factory.make_env()
    try:
        observation, info = env.reset(seed=31)
        assert observation.shape == (23,)
        assert float(info["initial_equity"]) == 10000.0
        step = 0
        economic_reward_sum = 0.0
        while True:
            target = 0.5 if step < 475 else 0.0
            observation, reward, terminated, truncated, info = env.step(
                np.array([target], dtype=np.float32)
            )
            economic_reward_sum += float(info["r_econ"])
            assert reward == pytest.approx(float(info["r_train"]))
            step += 1
            if terminated or truncated:
                break

        final = factory.snapshots[-1]
        report = native_report(
            factory.snapshots,
            fill_references=factory.fill_references,
            sample_interval_ns=60_000_000_000,
            linear_equity_rounding=Decimal("0.00000001"),
        )
        metrics = report["metrics"]
        assert step == 477
        assert factory.quality.grade.value == "DQ-B"
        assert factory.rule.provenance == "scenario-only:fixture-assumptions-not-historical-evidence"
        assert final.decision_ts_ns == 1704096180001000000
        assert str(final.equity) == "9998.21338803"
        assert len(report["fills"]) == 3
        assert metrics["observations"] == 478
        assert metrics["gross_return_before_reported_costs"] == pytest.approx(-0.00010322299999992852)
        assert metrics["total_fee"] == pytest.approx(0.49911263)
        assert metrics["total_slippage_cost"] == pytest.approx(0.10028)
        assert metrics["total_funding"] == pytest.approx(-0.15498934)
        assert metrics["net_return"] == pytest.approx(-0.000178661197001162)
        # The per-step economic reward must reconcile to the final native
        # account equity after actual fee, slippage and funding adjustments.
        assert economic_reward_sum == pytest.approx(log(float(final.equity) / 10000.0), abs=1e-12)
        assert info["terminal_settled"]
        assert report["terminal_flat"]
        assert report["liquidations_observed"] is True
        assert report["liquidations"] == 0
        assert report["equity_reconciliation"]["terminal_checked"] is True
        assert report["equity_reconciliation"]["unique_funding"] > 0
        assert report["equity_reconciliation"]["unique_fills"] == 3
        check = report["equity_reconciliation"]
        assert check["open_barriers"] == 475
        assert check["checked_open_barriers"] == 475
        assert check["unchecked_open_barriers"] == 0
        tolerance = (Decimal(check["rounding_per_event"])
                     * (check["unique_fills"] + check["unique_funding"] + 1))
        assert Decimal(check["max_abs_residual"]) <= tolerance
    finally:
        env.close()


@pytest.mark.micro_golden
def test_real_ethusdt_bookticker_drives_native_quote_fills():
    """Replay the real sampled Binance top of book through native matching."""
    project = Path(__file__).resolve().parents[1]
    market_path = project / "tests" / "fixtures" / "binance_ethusdt_20240101_quotes_1s" / "market.parquet"
    assert sha256(market_path.read_bytes()).hexdigest() == (
        "d4bfb69209f3ca86f6f9a473aaf20d0db2c0225a6fd72a6e2eafd92ca466b68b"
    )
    archive = json.loads((market_path.parent / "import.json").read_text(encoding="utf-8"))
    quote_source = next(source for source in archive["sources"] if "bookTicker" in source["url"])
    assert quote_source["sha256"] == (
        "8b0ba77526b089682e541fd4d80dcf7d228f50227ec8e5c81ca7431e0e98eb2f"
    )

    config = json.loads((project / "configs" / "ethusdt_20240101_real.json").read_text(encoding="utf-8"))
    environment = config["environment"]
    execution = dict(environment["execution"])
    execution["base_fill"] = {
        "model_id": "binance-bookticker-top1",
        "slippage_bps": "0",
        "data_granularity": "quote",
    }
    execution["stress_fill"] = {
        "model_id": "binance-bookticker-top1-stress",
        "slippage_bps": "0",
        "data_granularity": "quote",
    }
    coverage = dict(environment["coverage_policy"])
    coverage["settlement_times_ns"] = {"ETHUSDT-PERP.SIM": [1704096000000000000]}
    factory = HistoricalEnvironmentFactory(
        market_path=market_path,
        rules_path=project / "tests" / "fixtures" / "real_ethusdt_20240101" / "scenario_rules.json",
        instrument="ETHUSDT-PERP.SIM",
        start_ns=1704067500000000000,
        end_ns=1704096300000000000,
        coverage_policy=coverage,
        execution=execution,
        hard_risk=environment["hard_risk"],
        soft_risk=environment["soft_risk"],
        starting_balance=environment["starting_balance"],
        risk_budget_notional=environment["risk_budget_notional"],
        fit_scaler=True,
    )

    env = factory.make_env()
    try:
        _observation, _info = env.reset(seed=31)
        step = 0
        while True:
            target = 0.25 if step < 60 else 0.0
            _observation, _reward, terminated, truncated, info = env.step(
                np.array([target], dtype=np.float32)
            )
            step += 1
            if terminated or truncated:
                break

        report = native_report(
            factory.snapshots,
            fill_references=factory.fill_references,
            sample_interval_ns=60_000_000_000,
        )
        quote_events = [
            event for event in read_curated(market_path).events
            if event.kind == "best_quote" and event.payload["instrument"] == "ETHUSDT-PERP.SIM"
            and event.available_at_ns is not None
        ]
        quote_available_ns = [int(event.available_at_ns) for event in quote_events]
        assert factory.quality.grade.value == "DQ-B"
        assert step == 477
        assert len(report["fills"]) == len(factory.fill_references) == 2
        for fill in report["fills"]:
            quote_index = bisect_right(quote_available_ns, fill["ts_event"]) - 1
            assert quote_index >= 0
            quote = quote_events[quote_index]
            side = "ask" if fill["order_side"] == "BUY" else "bid"
            assert Decimal(fill["last_qty"]) <= Decimal(str(quote.payload[f"{side}_qty"]))
            assert Decimal(fill["last_px"]) == Decimal(str(quote.payload[f"{side}_price"]))
        assert report["metrics"]["observations"] == 478
        assert info["terminal_settled"] is True
        assert report["terminal_flat"] is True
        assert report["liquidations"] == 0
        assert np.isfinite(report["metrics"]["net_return"])
    finally:
        env.close()
