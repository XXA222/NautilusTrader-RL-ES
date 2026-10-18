"""Read-only production-code audit probes; use synthetic fixtures only.

Run with .venv/Scripts/python.exe qa/audit_20261016_probe.py.
Creates a fresh audit evidence directory under .tmp and prints its location.
These probes record current behavior; they are not acceptance tests.
"""
from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path
import runpy
import sqlite3
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for source in (ROOT / "packages").glob("*/src"):
    sys.path.insert(0, str(source))


def main() -> None:
    from rles_data.core import MINUTE_NS, digest
    from rles_data.features import Scaler
    from rles_gym_bridge import L2FrozenScalerAdapter
    from rles_execution.planner import InstrumentRules, TargetQuantityPlanner
    from rles_execution.spec import ResolvedExecutionConfig
    from rles_orchestrator.application import backtest, train
    from rles_research.ledger import ResearchLedger, ResearchLockError

    output = Path(tempfile.mkdtemp(prefix="audit-20261016-probe-", dir=ROOT / ".tmp"))
    fixtures = runpy.run_path(str(ROOT / "tests/test_l2_gym_bridge.py"))
    _, spec, schema = fixtures["_feature_inputs"]()
    scaler = Scaler(spec.hash, (0.02, 90.0), (0.01, 5.0), "c" * 64, 10)
    expected = scaler.hash
    adapter = L2FrozenScalerAdapter(schema, spec, scaler, expected_scaler_hash=expected)
    matrix = np.zeros((1, len(schema.names)))
    before = adapter.transform(matrix).tolist()
    # Simulate accidental low-level mutation/restore code bypassing frozen=True.
    # Ordinary attribute assignment is already rejected by the dataclass.
    object.__setattr__(scaler, "mean", (1.02, 90.0))
    after = adapter.transform(matrix).tolist()
    result = {"scaler": {
        "cached_hash_unchanged": scaler.hash == expected,
        "fresh_digest_changed": digest(scaler) != expected,
        "transform_changed_without_guard_error": before != after,
        "before": before, "after": after,
    }}

    opened = []
    for index in range(2):
        ledger = ResearchLedger(output / f"ledger-{index}.sqlite")
        protocol = f"audit-{index}"
        ledger.create(protocol, protocol, validation_budget=1, hpo_budget=0, es_evaluation_budget=0)
        ledger.freeze_recipe(protocol, recipe_sha="a" * 64, code_sha="b" * 64,
                             selection_spec_sha="c" * 64)
        ledger.register_retrain(protocol, retrain_sha="d" * 64, pre_holdout_end_ns=100)
        opened.append(ledger.open_holdout(protocol, holdout_dataset_sha="e" * 64,
                                         reviewer_approval_sha="f" * 64, holdout_start_ns=101))
        try:
            ledger.open_holdout(protocol, holdout_dataset_sha="e" * 64,
                                reviewer_approval_sha="f" * 64, holdout_start_ns=101)
        except ResearchLockError:
            same_ledger_blocked = True
        else:
            same_ledger_blocked = False
    result["ledger"] = {"same_ledger_repeat_blocked": same_ledger_blocked,
                        "same_holdout_opened_in_two_databases": opened}

    native_fixture = runpy.run_path(str(ROOT / "tests/test_historical_factory.py"))
    environment = native_fixture["recipe"](output)
    environment["end_ns"] = 12 * MINUTE_NS
    for key in ("market_path", "rules_path"):
        environment[key] = str(environment[key])
    execution = ResolvedExecutionConfig.from_dict({**environment["execution"], "deadband": "0.10"})
    plan = TargetQuantityPlanner(execution).plan(
        decision_id="audit-flat", instrument=InstrumentRules(
            "ETHUSDT-PERP.SIM", Decimal("0.001"), Decimal("0.001"),
            Decimal("20"), Decimal("1")),
        target_exposure=Decimal("0"), risk_budget_notional=Decimal("1000"),
        equity=Decimal("10000"), mark_price=Decimal("1000"),
        current_qty=Decimal("0.05"), pending=(),
    )
    result["voluntary_flat"] = {
        "current_exposure_risk_budget_units": 0.05, "requested_target": 0,
        "deadband": 0.1, "action": plan.action, "reason": plan.reason,
        "interpretation": "ordinary zero target can retain a valid nonzero position inside deadband",
    }
    config = {"environment": environment, "evaluation": None,
              "output": str(output / "train"), "algorithm": "PPO", "steps": 4,
              "hidden_layers": [2], "normalize_observation": True,
              "rl_corpus_episode_steps": 2,
              "hyperparameters": {"n_steps": 2, "batch_size": 2, "n_epochs": 1}}
    trained = train(config, evaluate=False)
    database = output / "train/research.sqlite"

    def ledger_counts():
        with sqlite3.connect(database) as connection:
            return {table: connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                    for table in ("query", "audit")}

    counts_before = ledger_counts()
    config["evaluation"] = {"start_ns": 12 * MINUTE_NS, "end_ns": 20 * MINUTE_NS,
                            "split": "holdout", "coverage_policy": environment["coverage_policy"]}
    first = backtest(config, trained["output"])
    second = backtest(config, trained["output"])
    result["standalone_holdout"] = {
        "two_calls_succeeded": True, "same_metrics": first["metrics"] == second["metrics"],
        "ledger_before": counts_before, "ledger_after": ledger_counts(),
        "scope": "synthetic fixture only; no real Holdout was accessed",
    }
    (output / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(output / "result.json")


if __name__ == "__main__":
    main()
