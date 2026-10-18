from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pytest

from rles_es_evotorch.actor import ActorGenomeSpec, genome_hash
from qa.analyze_m13_stage_contribution_v2 import (
    analyze,
    summarize_stage_chain,
    verify_e1_actor,
)


def _digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    digest = _digest(path)
    path.with_suffix(path.suffix + ".sha256").write_text(digest, encoding="ascii")
    return digest


def _bundle(root: Path) -> tuple[dict, str]:
    spec = ActorGenomeSpec(observation_names=("feature",), hidden_sizes=(1,))
    genome = np.zeros(spec.size, dtype=np.float32)
    actor_hash = genome_hash(spec, genome)
    lane_hash = "b" * 64
    artifact_path = root / "E1-selected-actor.json"
    artifact = {
        "schema": "rles.selected-stage-actor.v1",
        "stage": "E1",
        "actor_spec": {
            "observation_names": list(spec.observation_names),
            "hidden_sizes": list(spec.hidden_sizes),
            "activation": spec.activation,
            "output_size": spec.output_size,
            "action_scale": spec.action_scale,
            "dtype": spec.dtype,
            "version": spec.version,
            "input_scale": spec.input_scale,
        },
        "actor_hash": actor_hash,
        "lane_evidence_hash": lane_hash,
        "genome": genome.tolist(),
    }
    artifact_sha = _write_json(artifact_path, artifact)
    bundle = {
        "schema": "rles.historical-bundle.v1",
        "actor": {"kind": "rl", "actor_hash": "c" * 64},
        "soft_multipliers": {"exposure_multiplier": 0.75, "drawdown_throttle": 0.8},
        "stage_actor_artifacts": {
            "E1": {
                "path": artifact_path.name,
                "sha256": artifact_sha,
                "actor_hash": actor_hash,
                "lane_evidence_hash": lane_hash,
            },
        },
        "lineage": [{
            "stage": "E1",
            "actor_hash": actor_hash,
            "metadata": {"lane_evidence_hash": lane_hash},
        }],
    }
    bundle_sha = _write_json(root / "bundle.json", bundle)
    return bundle, bundle_sha


def _native_report(final_equity: float, fee: float, decision_count: int) -> dict:
    return {
        "costs_already_in_native_equity": True,
        "terminal_flat": True,
        "equity_reconciliation": {
            "rows": [{"ts_ns": 2, "residual": "0", "tolerance": "0.000001"}],
            "max_abs_residual": "0",
        },
        "account_rows": [
            {"ts_ns": 1, "equity": 10_000.0},
            {"ts_ns": 2, "equity": final_equity},
        ],
        "metrics": {
            "net_return": (final_equity / 10_000.0) - 1.0,
            "max_drawdown": 0.01,
            "sharpe": 0.2,
            "sortino": 0.3,
            "turnover_ratio": 0.1,
            "total_fee": fee,
            "total_funding": -0.25,
            "total_slippage_cost": 0.15,
            "gross_return_before_reported_costs": 0.003,
        },
        "fills": [{}] * decision_count,
        "execution_decision_diagnostics": {
            "decision_count": 2,
            "submit_decision_count": decision_count,
            "native_order_count": decision_count,
        },
        "risk_hard_interventions": 1,
    }


def _envelopes(e1_actor_hash: str, prereg_hash: str = "d" * 64) -> dict[str, dict]:
    pairing = {
        "market_sha256": "1" * 64,
        "rules_sha256": "2" * 64,
        "window": {"start_ns": 1, "end_ns": 2},
        "seed": 31,
        "starting_balance": "10000",
        "environment_sha256": "3" * 64,
        "execution_sha256": "4" * 64,
        "hard_risk_sha256": "5" * 64,
        "scaler_sha256": "6" * 64,
    }
    actor_hashes = {"E1": e1_actor_hash, "R1": "c" * 64, "E2": "c" * 64}
    payloads = {
        "E1": _native_report(10_002.0, 1.0, 1),
        "R1": _native_report(10_001.0, 2.0, 2),
        "E2": _native_report(9_997.0, 3.0, 3),
    }
    return {
        stage: {
            "schema": "rles.m13-stage-native-report.v1",
            "stage": stage,
            "mode": "base",
            "actor_sha256": actor_hashes[stage],
            "preregistration_sha256": prereg_hash,
            "pairing_identity": json.loads(json.dumps(pairing)),
            "soft_multipliers": (
                {"exposure_multiplier": 1.0, "drawdown_throttle": 1.0}
                if stage != "E2" else
                {"exposure_multiplier": 0.75, "drawdown_throttle": 0.8}
            ),
            "native_report": payloads[stage],
        }
        for stage in ("E1", "R1", "E2")
    }


def test_v2_analyzer_verifies_e1_artifact_and_recomputes_all_stage_deltas(tmp_path: Path) -> None:
    _, _ = _bundle(tmp_path)
    actor = verify_e1_actor(tmp_path)
    result = summarize_stage_chain("base", _envelopes(actor["actor_sha256"]),
                                   e1_actor_sha256=actor["actor_sha256"],
                                   preregistration_sha256="d" * 64)

    assert result["paired_account_rows"] == 2
    assert result["same_native_timestamps"] is True
    assert result["stage_deltas"]["E1_to_R1"]["native_final_equity_delta"] == -1.0
    assert result["stage_deltas"]["R1_to_E2"]["native_final_equity_delta"] == -4.0
    assert result["stage_deltas"]["E1_to_E2"]["metric_deltas"]["total_fee"] == 2.0
    assert result["stage_deltas"]["R1_to_E2"]["execution_deltas"]["fills"] == 1


def test_v2_analyzer_rejects_e1_genome_that_does_not_match_artifact_hash(tmp_path: Path) -> None:
    bundle, _ = _bundle(tmp_path)
    artifact_path = tmp_path / bundle["stage_actor_artifacts"]["E1"]["path"]
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    artifact["genome"][0] = 1.0
    artifact_sha = _write_json(artifact_path, artifact)
    bundle["stage_actor_artifacts"]["E1"]["sha256"] = artifact_sha
    _write_json(tmp_path / "bundle.json", bundle)

    with pytest.raises(ValueError, match="genome hash does not match"):
        verify_e1_actor(tmp_path)


@pytest.mark.parametrize("mutation, message", [
    ("timestamp", "timestamps are not strictly paired"),
    ("pairing", "same frozen pairing identity"),
    ("actor", "same frozen Actor"),
    ("costs", "native equity cost inclusion is not confirmed"),
    ("decisions", "same number of policy decisions"),
])
def test_v2_analyzer_rejects_unpaired_or_unreconciled_stage_report(
    mutation: str, message: str, tmp_path: Path,
) -> None:
    _, _ = _bundle(tmp_path)
    actor = verify_e1_actor(tmp_path)
    envelopes = _envelopes(actor["actor_sha256"])
    if mutation == "timestamp":
        envelopes["E2"]["native_report"]["account_rows"][1]["ts_ns"] = 3
    elif mutation == "pairing":
        envelopes["R1"]["pairing_identity"]["seed"] = 32
    elif mutation == "actor":
        envelopes["E2"]["actor_sha256"] = "e" * 64
    elif mutation == "decisions":
        envelopes["E2"]["native_report"]["execution_decision_diagnostics"]["decision_count"] += 1
    else:
        envelopes["R1"]["native_report"]["costs_already_in_native_equity"] = False

    with pytest.raises(ValueError, match=message):
        summarize_stage_chain("base", envelopes, e1_actor_sha256=actor["actor_sha256"],
                              preregistration_sha256="d" * 64)


def test_v2_analyze_accepts_only_registered_train_inner_reports(tmp_path: Path) -> None:
    _, bundle_sha = _bundle(tmp_path / "bundle-root")
    report_root = tmp_path / "reports"
    registration_path = tmp_path / "preregistration.json"
    actor = verify_e1_actor(tmp_path / "bundle-root")
    envelopes = _envelopes(actor["actor_sha256"])
    registration = {
        "schema": "rles.m13-stage-contribution-preregistration.v2",
        "status": "frozen_before_native_replay",
        "split": "train_inner",
        "stage_order": ["E1", "R1", "E2"],
        "no_retraining_or_selection": True,
        "formal_evaluation_queries": {"validation": 0, "test": 0, "holdout": 0},
        "bundle_root": str((tmp_path / "bundle-root").resolve()),
        "bundle_sha256": bundle_sha,
        "output_root": str(report_root.resolve()),
        "modes": ["base"],
        "actor_sha256_by_stage": {"E1": actor["actor_sha256"], "R1": "c" * 64, "E2": "c" * 64},
        "pairing_identity_by_mode": {"base": envelopes["E1"]["pairing_identity"]},
        "soft_multipliers_by_mode": {"base": {
            "E1": {"exposure_multiplier": 1.0, "drawdown_throttle": 1.0},
            "R1": {"exposure_multiplier": 1.0, "drawdown_throttle": 1.0},
            "E2": {"exposure_multiplier": 0.75, "drawdown_throttle": 0.8},
        }},
        "interpretation_limit": "synthetic unit-test contract only",
    }
    prereg_sha = _write_json(registration_path, registration)
    envelopes = _envelopes(actor["actor_sha256"], prereg_sha)
    for stage, envelope in envelopes.items():
        _write_json(report_root / f"base_{stage}.json", envelope)

    result = analyze(tmp_path / "bundle-root", registration_path, report_root)

    assert result["schema"] == "rles.m13-stage-contribution-analysis.v2"
    assert result["market_data_opened"] is False
    assert result["formal_evaluation_queries"] == {"validation": 0, "test": 0, "holdout": 0}
    assert result["modes"]["base"]["stage_deltas"]["R1_to_E2"]["native_final_equity_delta"] == -4.0


def test_v2_analyze_rejects_boolean_formal_query_counts(tmp_path: Path) -> None:
    bundle_root = tmp_path / "bundle-root"
    _, bundle_sha = _bundle(bundle_root)
    report_root = tmp_path / "reports"
    registration_path = tmp_path / "preregistration.json"
    actor = verify_e1_actor(bundle_root)
    envelopes = _envelopes(actor["actor_sha256"])
    registration = {
        "schema": "rles.m13-stage-contribution-preregistration.v2",
        "status": "frozen_before_native_replay",
        "split": "train_inner",
        "stage_order": ["E1", "R1", "E2"],
        "no_retraining_or_selection": True,
        "formal_evaluation_queries": {"validation": False, "test": 0, "holdout": 0},
        "bundle_root": str(bundle_root.resolve()),
        "bundle_sha256": bundle_sha,
        "output_root": str(report_root.resolve()),
        "modes": ["base"],
        "actor_sha256_by_stage": {"E1": actor["actor_sha256"], "R1": "c" * 64, "E2": "c" * 64},
        "pairing_identity_by_mode": {"base": envelopes["E1"]["pairing_identity"]},
        "soft_multipliers_by_mode": {"base": {
            "E1": {"exposure_multiplier": 1.0, "drawdown_throttle": 1.0},
            "R1": {"exposure_multiplier": 1.0, "drawdown_throttle": 1.0},
            "E2": {"exposure_multiplier": 0.75, "drawdown_throttle": 0.8},
        }},
    }
    _write_json(registration_path, registration)

    with pytest.raises(ValueError, match="incomplete or outside"):
        analyze(bundle_root, registration_path, report_root)
