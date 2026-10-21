"""Matched-seed A2 Train-inner comparison of three ES fitness protocols.

Uses the checksum-verified ETHUSDT archive fixture and scenario-only execution
rules. Evaluation is disabled; the script does not query Validation/Test.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from rles_es_evotorch.evaluation import scenario_seed
from rles_orchestrator.application import backtest, prepare, train


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"
BASE_NS = 1_704_067_200_000_000_000
DAY_NS = 86_400_000_000_000
HOUR_NS = 3_600_000_000_000
MINUTE_NS = 60_000_000_000
FOLD_ID = "a02-three-fitness-crn-20261017"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _base_config(output: Path, fitness: dict[str, object]) -> tuple[dict[str, Any], dict[str, object]]:
    fixture = FIXTURES / "real_ethusdt_20240104"
    manifest = json.loads((fixture / "import.json").read_text(encoding="utf-8"))
    if manifest.get("schema") not in {"rles.binance-archive.v1", "rles.binance-archive-range.v1"}:
        raise ValueError("archive fixture manifest schema is unsupported")
    for source in manifest["sources"]:
        if _sha(fixture / source["path"]) != source["sha256"]:
            raise ValueError(f"archive fixture checksum mismatch: {source['path']}")

    base = json.loads((ROOT / "configs" / "ethusdt_20240101_real.json").read_text(encoding="utf-8"))
    day_start = BASE_NS + 3 * DAY_NS
    environment = base["environment"]
    environment["market_path"] = str(fixture / "market.parquet")
    environment["rules_path"] = str(FIXTURES / "real_ethusdt_20240101" / "scenario_rules.json")
    environment["start_ns"] = day_start + 5 * MINUTE_NS
    environment["end_ns"] = day_start + 8 * HOUR_NS + 5 * MINUTE_NS
    environment["coverage_policy"]["settlement_times_ns"] = {
        "ETHUSDT-PERP.SIM": [day_start + 8 * HOUR_NS],
    }
    environment["coverage_policy"]["settlement_timestamp_tolerance_ns"] = 4_000_000
    validation_coverage = json.loads(json.dumps(environment["coverage_policy"]))
    validation_coverage["settlement_times_ns"] = {"ETHUSDT-PERP.SIM": []}
    base.update({
        "route": "A2", "algorithm": "PPO", "steps": 64, "seed": 31,
        "hidden_layers": [16, 16], "normalize_observation": True,
        "hyperparameters": {"n_steps": 32, "batch_size": 32, "n_epochs": 2},
        "population": 4, "generations": 2,
        "checkpoint_interval_steps": 32,
        "evaluation": {"start_ns": environment["end_ns"],
                        "end_ns": day_start + 12 * HOUR_NS + 5 * MINUTE_NS,
                        "coverage_policy": validation_coverage},
        "output": str(output),
        "es_fitness": fitness,
        "es_scenario_fold_id": FOLD_ID,
        "es_candidate_evidence": True,
    })
    return base, manifest


def run(output_root: Path, *, replay_only: bool = False) -> dict[str, object]:
    arms: dict[str, dict[str, object]] = {
        "legacy_weighted": {"protocol": "legacy_weighted_v1"},
        "zero_intervention_penalty": {
            "protocol": "legacy_weighted_v1", "risk_intervention_weight": 0.0,
        },
        "constraint_priority": {"protocol": "constraint_priority_train_inner_v1"},
    }
    records = []
    expected_actor_hash: str | None = None
    for arm_name, fitness in arms.items():
        config, manifest = _base_config(output_root / arm_name, fitness)
        run_root = Path(config["output"])
        if replay_only:
            if not (run_root / "COMPLETED.json").is_file():
                raise FileNotFoundError(f"completed arm missing for Train-inner replay: {arm_name}")
        else:
            result = train(config, evaluate=False)
            run_root = Path(result["output"])
        train_factory = prepare(config)
        train_report = backtest(config, run_root, factory=train_factory, summary_only=True)
        if (train_report["evaluation_scope"]["split"] != "train_inner"
                or train_report["evaluation_scope"]["formal"] is not False):
            raise RuntimeError("economic replay escaped the Train-inner development scope")
        bundle = json.loads((run_root / "bundle.json").read_text(encoding="utf-8"))
        run_manifest = json.loads((run_root / "run.json").read_text(encoding="utf-8"))
        if bundle["es_scenario_fold_id"] != FOLD_ID or run_manifest["es_scenario_fold_id"] != FOLD_ID:
            raise RuntimeError("application did not preserve the common ES scenario fold")
        if bundle["es_candidate_evidence"] is not True:
            raise RuntimeError("candidate evidence was not included in the training bundle")
        actor_hash = bundle["actor"]["actor_hash"]
        if expected_actor_hash is None:
            expected_actor_hash = actor_hash
        elif actor_hash != expected_actor_hash:
            raise RuntimeError("A2 arms did not reuse an identical R1 actor")
        e2 = next(row for row in bundle["lineage"] if row["stage"] == "E2")
        metadata = e2["metadata"]
        if metadata["scenario_fold_id"] != FOLD_ID:
            raise RuntimeError("E2 stage metadata lost the CRN fold id")
        candidate_rows = json.loads(metadata["constraint_evidence_json"])
        if len(candidate_rows) != 2 or any(len(generation["candidates"]) != 5
                                           for generation in candidate_rows):
            raise RuntimeError("E2 candidate evidence differs from the frozen population/neutral budget")
        records.append({
            "arm": arm_name,
            "fitness_config": fitness,
            "fitness_protocol": metadata["fitness_protocol"],
            "fitness_spec": json.loads(metadata["fitness_spec_json"]),
            "run_id": run_manifest["run_id"],
            "run_recipe_hash": run_manifest["recipe_hash"],
            "actor_hash": actor_hash,
            "scenario_fold_id": FOLD_ID,
            "scenario_seeds": [scenario_seed(31, FOLD_ID, generation, "train-inner")
                               for generation in range(2)],
            "optimizer": {"algorithm": "cmaes", "population": 4,
                          "generations": 2, "candidate_evaluations": 8,
                          "e2_lane_requests": e2["usage"]["lane_requests"]},
            "selected_candidate_kind": metadata["selected_candidate_kind"],
            "selected_fitness": float(metadata["selected_fitness"]),
            "selected_soft_multipliers": bundle["soft_multipliers"],
            "selected_candidate_train_inner_native_metrics": train_report["metrics"],
            "generation_candidate_scores": [
                {candidate["candidate_id"]: {
                    "mean_fitness": candidate["mean_fitness"],
                    "infeasible_episodes": candidate["infeasible_episodes"],
                    "rejection_reasons": candidate["rejection_reasons"],
                    "evidence_hash": candidate["evidence_hash"],
                } for candidate in generation["candidates"]}
                for generation in candidate_rows
            ],
            "bundle_sha256": _sha(run_root / "bundle.json"),
            "source_archive": {
                "schema": manifest["schema"], "event_count": manifest["event_count"],
                "sources": manifest["sources"],
                "rule_provenance": "scenario-only:fixture-assumptions-not-historical-evidence",
            },
            "validation_evaluated": False,
        })
    if len({row["actor_hash"] for row in records}) != 1:
        raise RuntimeError("R1 actor hash differs across matched A2 objective arms")
    summary: dict[str, object] = {
        "schema": "rles.es-fitness-comparison.v1",
        "experiment": "A02 matched-seed Train-inner A2 objective comparison",
        "date": "2026-10-17",
        "data_boundary": "checksum-verified ETHUSDT Train-inner archive fixture; no Validation/Test access",
        "shared": {"seed": 31, "scenario_fold_id": FOLD_ID,
                   "route": "A2", "same_frozen_r1_actor": True,
                   "population": 4, "generations": 2,
                   "e2_candidate_evaluations_per_arm": 8,
                   "e2_lane_requests_per_arm": 10,
                   "neutral_r1_included_each_generation": True},
        "arms": records,
        "interpretation_limit": "fitness values use protocol-specific scales; compare within-arm candidate ranks and selected economic parameters, not raw scores across protocols. This is Train-inner objective evidence, not Validation performance or profitability certification.",
    }
    output_root.mkdir(parents=True, exist_ok=True)
    summary_path = output_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    summary["summary_path"] = str(summary_path.resolve())
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True,
                        help="fresh output directory for the three isolated A2 runs")
    parser.add_argument("--replay-only", action="store_true",
                        help="reuse completed arms and run the Train-inner native economic replay")
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    if output.exists() and any(output.iterdir()) and not args.replay_only:
        raise SystemExit("output directory must be new or empty")
    summary = run(output, replay_only=args.replay_only)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
