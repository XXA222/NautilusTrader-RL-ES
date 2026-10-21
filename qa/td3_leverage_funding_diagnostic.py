"""Replay one frozen TD3 actor under paired 1x and 10x account-risk caps.

This is a development diagnostic only. It records the full legacy snapshot
scope before decoding and never queries Test or Holdout data.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from decimal import Decimal
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

from rles_data.core import CuratedStore, DataError
from rles_data.features import Scaler
from rles_data.io import curated_source_identity
from rles_gym_bridge.historical import HistoricalEnvironmentFactory, load_rules
from rles_orchestrator.application import InferenceActor, rollout
from rles_orchestrator.research_access import read_registered_train_window_with_receipt


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _account_exposure(factory: HistoricalEnvironmentFactory) -> dict[str, float]:
    maximum = 0.0
    observed = 0
    multiplier = factory.contract_spec.multiplier
    for snapshot in factory.snapshots:
        equity = Decimal(str(snapshot.equity))
        mark = Decimal(str(snapshot.mark))
        if equity <= 0 or mark <= 0:
            continue
        gross = Decimal(0)
        for position in snapshot.positions:
            quantity = Decimal(str(position["quantity"]))
            gross += abs(quantity) * mark * multiplier
        maximum = max(maximum, float(gross / equity))
        observed += 1
    return {"maximum_gross_leverage": maximum, "native_snapshot_count": observed}


def _funding_attribution(factory: HistoricalEnvironmentFactory, report: dict) -> list[dict]:
    events = [event for event in factory.store.events
              if event.kind == "funding_settlement"
              and factory.start_ns <= event.ts_event_ns < factory.end_ns
              and event.payload.get("instrument") == factory.instrument]
    payments = list(report.get("funding", ()))
    fills = list(report.get("fills", ()))
    rows = []
    for event in events:
        payload = dict(event.payload)
        event_timestamp = event.ts_event_ns
        timestamp = int(payload.get("settlement_ns", event_timestamp))
        payment = next((item for item in payments
                        if isinstance(item.get("ts_event"), int)
                        and abs(item["ts_event"] - event_timestamp) <= 2_000_000), None)
        rate_value = payload.get("rate")
        settlement_price = payload.get("mark_price", payload.get("markPrice"))
        fills_before = [item for item in fills
                        if item.get("instrument_id") == factory.instrument
                        and isinstance(item.get("ts_event"), int)
                        and item["ts_event"] <= event_timestamp]
        quantity = sum((Decimal(str(item["last_qty"]))
                        * (1 if item.get("order_side") == "BUY" else -1)
                        for item in fills_before), Decimal(0))
        marks = [item for item in factory.store.events
                 if item.kind == "mark"
                 and item.payload.get("instrument") == factory.instrument
                 and item.ts_event_ns <= timestamp]
        mark_event = max(marks, key=lambda item: item.ts_event_ns) if marks else None
        mark = (Decimal(str(settlement_price)) if settlement_price is not None else
                Decimal(str(mark_event.payload["price"])) if mark_event is not None else None)
        expected = (None if rate_value is None or mark is None else
                    -quantity * mark * factory.contract_spec.multiplier
                    * Decimal(str(rate_value)))
        actual = (Decimal(str(payment["pnl_change"]).split()[0])
                  if payment is not None else Decimal(0))
        rows.append({
            "event_id": event.event_id,
            "settlement_ns": timestamp,
            "rate": None if rate_value is None else str(rate_value),
            "settlement_mark": None if mark is None else str(mark),
            "settlement_mark_event_id": None if mark_event is None else mark_event.event_id,
            "settlement_mark_event_ns": None if mark_event is None else mark_event.ts_event_ns,
            "fills_before_settlement": len(fills_before),
            "signed_position_quantity_before_settlement": str(quantity),
            "native_payment_usdt": str(actual),
            "linear_formula_payment_usdt": None if expected is None else str(expected),
            "formula_residual_usdt": None if expected is None else str(actual - expected),
            "native_payment_event_id": None if payment is None else payment.get("event_id"),
        })
    return rows


def run_diagnostic(*, config_path: Path, bundle_path: Path,
                   registration_path: Path, output_dir: Path) -> dict:
    registration = json.loads(registration_path.read_text(encoding="utf-8"))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    bundle = json.loads((bundle_path / "bundle.json").read_text(encoding="utf-8"))
    if registration.get("schema") != "rles.td3-leverage-funding-diagnostic.v1":
        raise DataError("unsupported TD3 leverage diagnostic registration")
    if (config.get("algorithm") != "TD3"
            or registration.get("saved_split") != "validation"):
        raise DataError("frozen source config must be registered as the saved TD3 Validation recipe")
    if bundle.get("schema") != "rles.historical-bundle.v1" or bundle.get("actor", {}).get("kind") != "rl":
        raise DataError("frozen actor bundle is not a historical RL bundle")
    source_bundle_sha = _file_sha256(bundle_path / "bundle.json")
    source_config_sha = _file_sha256(config_path)
    if (source_bundle_sha != registration["bundle_sha256"]
            or source_config_sha != registration["config_sha256"]):
        raise DataError("frozen TD3 bundle/config hash differs from registration")
    baseline_report = bundle_path / "report.json"
    if _file_sha256(baseline_report) != registration["baseline_report_sha256"]:
        raise DataError("registered prior TD3 report hash differs")
    for relative, expected in registration["source_code_sha256"].items():
        if _file_sha256(Path(relative)) != expected:
            raise DataError(f"source code changed after registration: {relative}")
    if _canonical_json(config["environment"]) != _canonical_json(bundle["environment"]):
        raise DataError("saved TD3 config and bundle environment differ")
    evaluation = config["evaluation"]
    if evaluation["start_ns"] < bundle["training_end_ns"]:
        raise DataError("diagnostic window overlaps the frozen TD3 training boundary")
    output_dir.mkdir(parents=True, exist_ok=False)

    source_environment = dict(bundle["environment"])
    access_window = registration["registered_source_window"]
    source = curated_source_identity(source_environment["market_path"])
    if source != registration["market_source_identity"]:
        raise DataError("curated source identity differs from registration")
    frozen_rules_sha256 = registration.get(
        "frozen_training_rules_sha256", registration["scenario_rules_sha256"]
    )
    if _file_sha256(Path(source_environment["rules_path"])) != frozen_rules_sha256:
        raise DataError("frozen TD3 source rule file changed after registration")
    evaluation_rules_path = Path(registration.get(
        "evaluation_rules_path", source_environment["rules_path"]
    ))
    if _file_sha256(evaluation_rules_path) != registration["scenario_rules_sha256"]:
        raise DataError("registered evaluation scenario rule file changed")
    if not (access_window["start_ns"] <= source_environment["start_ns"]
            and access_window["end_ns"] >= evaluation["end_ns"]):
        raise DataError("registered source window does not cover training and evaluation")

    evaluation_environment = {
        **source_environment,
        "rules_path": str(evaluation_rules_path),
        **evaluation,
        "risk_budget_notional": "100000",
        "hard_risk": {
            **source_environment["hard_risk"],
            "max_notional": "100000",
            "max_leverage": "10",
        },
    }
    access_config = {
        "output": str(output_dir / "scope-owner"),
        "environment": evaluation_environment,
    }
    receipt = read_registered_train_window_with_receipt(
        access_config,
        access_window,
        marker_sha256=source["marker_sha256"],
        label=registration["access_label"],
        allow_legacy_isolated_snapshot=True,
    )
    rules = load_rules(evaluation_environment["rules_path"])
    prepared_store = CuratedStore((*receipt.store.events, *rules))
    raw_scaler = dict(bundle["preprocessing"]["scaler"])
    raw_scaler["mean"] = tuple(raw_scaler["mean"])
    raw_scaler["std"] = tuple(raw_scaler["std"])
    scaler = Scaler(**raw_scaler)
    inference = InferenceActor(bundle_path / bundle["actor"]["path"], verify=True)
    actor_marker = (bundle_path / bundle["actor"]["path"] / "COMPLETED").read_text(encoding="ascii")
    if actor_marker != bundle["actor"]["actor_hash"]:
        raise DataError("frozen TD3 actor hash differs from bundle")
    original_hard = dict(source_environment["hard_risk"])
    scenarios = []
    for leverage in (Decimal("1"), Decimal("10")):
        environment = {
            **evaluation_environment,
            "risk_budget_notional": "100000",
            "hard_risk": {
                **original_hard,
                "max_notional": "100000",
                "max_leverage": str(leverage),
            },
        }
        factory = HistoricalEnvironmentFactory(**{
            **environment,
            "split": "train_inner",
            "fit_scaler": False,
            "scaler": scaler,
            "prepared_store": prepared_store,
            "reward_scale": bundle.get("reward_scale", 1.0),
            "compact_report_snapshots": True,
        })
        if factory.schema.hash != bundle["preprocessing"]["identities"]["schema"]:
            raise DataError("evaluation observation schema differs from frozen TD3 actor")
        if factory.features.hash != bundle["preprocessing"]["identities"]["features"]:
            raise DataError("evaluation features differ from frozen TD3 actor")
        if factory.scaler.hash != bundle["preprocessing"]["identities"]["scaler"]:
            raise DataError("evaluation scaler differs from frozen TD3 actor")
        if factory.execution.fingerprint() != bundle["preprocessing"]["identities"]["execution"]:
            raise DataError("evaluation execution recipe differs from frozen TD3 actor")
        report, _ = rollout(
            factory,
            inference.predict,
            seed=config.get("seed", 31),
            soft_parameters=bundle["soft_multipliers"],
            collect_observations=False,
            include_account_rows=True,
            include_execution_trace=True,
        )
        report["dataset_quality"] = asdict(factory.quality)
        report["evaluation_scope"] = {
            "split": "train_inner",
            "formal": False,
            "start_ns": factory.start_ns,
            "end_ns": factory.end_ns,
            "project_access_key": receipt.reservation_key,
        }
        scenarios.append({
            "leverage_cap": str(leverage),
            "risk_budget_notional": "100000",
            "max_notional": "100000",
            "metrics": report["metrics"],
            "fee_total_usdt": report["metrics"]["total_fee"],
            "funding_total_usdt": report["metrics"]["total_funding"],
            "slippage_total_usdt": report["metrics"]["total_slippage_cost"],
            "fills": len(report["fills"]),
            "funding_payments": report["funding"],
            "funding_settlement_facts": _funding_attribution(factory, report),
            "liquidations": report["liquidations"],
            "liquidations_observed": report["liquidations_observed"],
            "liquidation_orders": report["liquidation_orders"],
            "risk_guard_interventions": report["risk_guard_interventions"],
            "exposure": _account_exposure(factory),
            "dataset_quality": report["dataset_quality"],
            "evaluation_scope": report["evaluation_scope"],
        })
    result = {
        "schema": "rles.td3-leverage-funding-diagnostic-result.v1",
        "registration_sha256": _file_sha256(registration_path),
        "source_bundle_sha256": source_bundle_sha,
        "source_actor_hash": bundle["actor"]["actor_hash"],
        "source_config_sha256": source_config_sha,
        "market_source_identity": source,
        "scenario_rule_classification": "NON_HISTORICAL_SCENARIO_ONLY",
        "training_window": {
            "start_ns": source_environment["start_ns"],
            "end_ns": source_environment["end_ns"],
            "training_end_ns": bundle["training_end_ns"],
        },
        "evaluation_window": {
            "start_ns": evaluation["start_ns"],
            "end_ns": evaluation["end_ns"],
            "previously_used_validation": True,
            "independent_evaluation": False,
        },
        "access_receipt": {
            "reservation_key": receipt.reservation_key,
            "registered_before_read": receipt.registered_before_read,
            "source_mode": "whole_snapshot_isolated_v1",
            "registered_source_window": access_window,
        },
        "paired_scenarios": scenarios,
        "limitations": [
            "The frozen TD3 actor was trained at 1x with a 1000 USDT notional budget; 10x is an out-of-distribution fixed-actor exposure-cap scenario.",
            "The evaluation interval was previously used as Validation; this replay is not independent evidence and is for development diagnosis only.",
            "Initial/maintenance margin, fees and liquidation settings are scenario assumptions, not authenticated historical Binance tiers.",
            "The paired 1x control shares the 100000 USDT risk budget and maximum notional; only the account leverage cap changes between paired scenarios.",
        ],
    }
    result_path = output_dir / "summary.json"
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                           encoding="utf-8")
    result_hash = _file_sha256(result_path)
    result_path.with_suffix(".json.sha256").write_text(result_hash, encoding="ascii")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = run_diagnostic(config_path=args.config, bundle_path=args.bundle,
                            registration_path=args.registration, output_dir=args.output_dir)
    print(json.dumps({"output": str(args.output_dir / "summary.json"),
                      "paired_scenarios": [
                          {"leverage_cap": item["leverage_cap"],
                           "net_return": item["metrics"]["net_return"],
                           "total_funding": item["funding_total_usdt"]}
                          for item in result["paired_scenarios"]],
                      "limitation": "development-only, previously used Validation"},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
