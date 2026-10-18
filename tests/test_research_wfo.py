from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from rles_data.core import DAY_NS, MINUTE_NS, DataError, Window, digest
from rles_data.features import Scaler
from rles_data.folds import bind_fold_scaler, build_folds
from rles_orchestrator.wfo_application import (
    _candidate, _coverage, _dsr_trial_count_evidence, _with_window,
)
from rles_research.wfo import FoldBoundaryError, FoldRequest, plan_folds, validate_l2_manifest


def _market_source_identity() -> dict:
    return {"schema_version": "0.4.1", "snapshot_hash": "a" * 64,
            "file_sha256": "b" * 64, "marker_sha256": "c" * 64, "events": 1}

pytestmark = pytest.mark.unit


def _execution_config() -> dict:
    return {
        "schema_version": "0.4.0", "deadband": "0.01", "pending_order_policy": "cancel_replace",
        "order_type": "MARKET", "tif": "GTC", "terminal_policy": "force_flatten_with_cost",
        "base_latency": {"model_id": "base", "decision_ns": 1, "submit_ns": 1, "venue_ns": 5},
        "stress_latency": {"model_id": "stress", "decision_ns": 5, "submit_ns": 5, "venue_ns": 20},
        "base_fill": {"model_id": "base-fill", "slippage_bps": "1", "data_granularity": "bar"},
        "stress_fill": {"model_id": "stress-fill", "slippage_bps": "5", "data_granularity": "bar"},
    }


def test_wfo_candidate_deadband_is_applied_and_bound_to_its_environment(tmp_path) -> None:
    execution = _execution_config()
    environment = {"instrument": "ETHUSDT-PERP.SIM", "execution": execution}
    base = {"environment": environment, "algorithm": "PPO", "output": "base", "wfo": {}}

    candidate = _candidate(
        base, {"candidate_id": "ppo-deadband-5pct", "algorithm": "PPO", "deadband": "0.05",
               "feature_profile": "causal_mtf_v2"},
        tmp_path / "candidate",
    )

    assert candidate["environment"]["execution"]["deadband"] == "0.05"
    assert candidate["environment"]["feature_profile"] == "causal_mtf_v2"
    assert "feature_profile" not in candidate
    assert candidate["environment"]["execution"]["base_fill"] == execution["base_fill"]
    assert base["environment"] is environment
    assert base["environment"]["execution"]["deadband"] == "0.01"
    assert "deadband" not in candidate
    explicit_flat = _candidate(base, {"candidate_id": "explicit-flat", "zero_target_policy": "close"},
                               tmp_path / "flat")
    assert explicit_flat["environment"]["execution"]["zero_target_policy"] == "close"
    assert "zero_target_policy" not in explicit_flat
    train_environment = _with_window(
        candidate["environment"], start_ns=10, end_ns=20, coverage_policy={"marker": "train"},
    )
    assert train_environment["execution"]["deadband"] == "0.05"
    assert train_environment["coverage_policy"] == {"marker": "train"}

    scheduled = _candidate(
        base, {"candidate_id": "ppo-funding-calendar", "feature_profile": "causal_mtf_v3"},
        tmp_path / "scheduled",
    )
    assert scheduled["environment"]["feature_profile"] == "causal_mtf_v3"

    with pytest.raises(DataError, match="deadband is invalid"):
        _candidate(base, {"candidate_id": "bad", "deadband": "1.01"}, tmp_path / "bad")
    with pytest.raises(DataError, match="decimal string or integer"):
        _candidate(base, {"candidate_id": "float", "deadband": 0.05}, tmp_path / "float")


@pytest.mark.artifact_restore
def test_final_retrain_resume_checkpoint_uses_latest_completed_supported_stage(tmp_path) -> None:
    from rles_orchestrator.wfo_application import _latest_final_retrain_checkpoint

    run_root = tmp_path / "final-retrain"
    run_root.mkdir()
    (run_root / "run.json").write_text(json.dumps({
        "schema": "rles.historical-run.v1", "run_id": "attempt-1", "attempt_id": "attempt-1",
    }), encoding="utf-8")

    def es_checkpoint(stage: str, generation: int, *, completed: bool = True) -> Path:
        checkpoint = run_root / stage / f"generation-{generation:08d}" / "attempt-1"
        checkpoint.mkdir(parents=True)
        (checkpoint / "manifest.json").write_text("{}", encoding="utf-8")
        if completed:
            (checkpoint / "COMPLETED").write_text("receipt", encoding="ascii")
        return checkpoint

    es_checkpoint("E1", 1)
    es_checkpoint("E1", 2, completed=False)
    es_checkpoint("E2", 1)
    (run_root / "R1-checkpoints" / "step-000000000032").mkdir(parents=True)
    (run_root / "R1-checkpoints" / "step-000000000032" / "COMPLETED").write_text(
        "receipt", encoding="ascii",
    )
    (run_root / "R1-checkpoints" / "step-000000000064").mkdir()

    assert _latest_final_retrain_checkpoint(run_root, "A0").name == "step-000000000032"
    assert _latest_final_retrain_checkpoint(run_root, "A1").parent.parent.name == "E1"
    assert _latest_final_retrain_checkpoint(run_root, "A2").parent.parent.name == "E2"
    assert _latest_final_retrain_checkpoint(run_root, "A3").name == "step-000000000032"
    assert _latest_final_retrain_checkpoint(run_root, "A4").parent.parent.name == "E2"

    (run_root / "E2" / "generation-00000001" / "attempt-1" / "COMPLETED").unlink()
    assert _latest_final_retrain_checkpoint(run_root, "A4").name == "step-000000000032"


def test_dsr_trial_count_evidence_limits_claim_to_frozen_protocol_candidates() -> None:
    from rles_research.statistics import candidate_set_sha256

    evidence = _dsr_trial_count_evidence(["candidate-a", "candidate-b"], {
        "trial_count_inventory": {
            "raw_query_counts": {"validation": 4, "test": 0},
            "observed_candidate_ids": ["candidate-a", "candidate-b"],
            "observed_fold_ids": ["fold-0", "fold-1"],
            "es_lane_request_count": 12,
            "stage_receipt_count": 3,
        },
    }, {"protocol_registered_attempt_count": 2,
        "project_registered_attempt_count": 5, "attempts": []})
    assert evidence["status"] == "partial"
    assert evidence["dsr_effective_trials"] == 2
    assert evidence["candidate_set_sha256"] == candidate_set_sha256(["candidate-a", "candidate-b"])
    assert "not canonicalized" in evidence["deduplication"]
    assert "not modeled" in evidence["dependence_assumption"]
    assert evidence["protocol_observed_usage"]["raw_query_counts"]["validation"] == 4
    assert evidence["project_historical_effective_trials"]["status"] == "not_estimated"
    assert "standalone application training attempts" in evidence[
        "project_historical_effective_trials"
    ]["reason"]
    assert evidence["project_trial_attempt_inventory"]["project_registered_attempt_count"] == 5


def _ns(value: datetime) -> int:
    return int(value.timestamp() * 1_000_000_000)


def _paired_folds():
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    end = datetime(2024, 5, 1, tzinfo=timezone.utc)
    warmup = timedelta(days=2)
    purge = timedelta(days=1)
    embargo = timedelta(days=1)
    dataset_hash, feature_hash = "a" * 64, "b" * 64
    research = plan_folds(
        start=start, end=end, scheme="60/15/15", warmup=warmup,
        purge=purge, embargo=embargo, data_available_from=start - warmup,
    )
    l2 = build_folds(
        start_ns=_ns(start), end_ns=_ns(end), train_days=60,
        warmup_ns=warmup.days * DAY_NS, feature_lookback_ns=purge.days * DAY_NS,
        execution_lag_guard_ns=0, embargo_ns=embargo.days * DAY_NS,
        dataset_hash=dataset_hash, feature_hash=feature_hash,
        data_start_ns=_ns(start - warmup),
    )
    return research, l2, dataset_hash, feature_hash


def test_wfo_request_matches_actual_nested_l2_fold_manifest_and_mapping() -> None:
    research, l2, dataset_hash, feature_hash = _paired_folds()
    assert len(research) == len(l2) == 2
    for request, manifest in zip(research, l2, strict=True):
        validate_l2_manifest(
            request, manifest, expected_dataset_hash=dataset_hash,
            expected_feature_hash=feature_hash,
        )
        validate_l2_manifest(
            request, asdict(manifest), expected_dataset_hash=dataset_hash,
            expected_feature_hash=feature_hash,
        )


def test_fold_scaler_binds_score_window_and_excludes_read_only_warmup() -> None:
    _research, folds, _dataset_hash, feature_hash = _paired_folds()
    fold = folds[0]
    score_window = Window(fold.train.score_start_ns, fold.train.score_start_ns,
                          fold.train.score_end_ns)
    scaler = Scaler(feature_hash, (1.0,), (2.0,), digest(score_window), 10)
    bound = bind_fold_scaler(fold, scaler)
    assert bound.scaler_hash == scaler.hash

    warmup_bound = Scaler(feature_hash, (1.0,), (2.0,), digest(fold.train), 10)
    with pytest.raises(DataError, match="train score window"):
        bind_fold_scaler(fold, warmup_bound)


def test_application_wfo_expands_declared_funding_schedule_without_guessing() -> None:
    template = {
        "maximum_age_ns": {"mark": MINUTE_NS, "index": MINUTE_NS,
                           "funding_update": 8 * 60 * MINUTE_NS},
        "settlement_interval_minutes": 480,
        "settlement_anchor_ns": 0,
        "settlement_max_delay_ns": MINUTE_NS,
        "provenance": "synthetic:wfo-contract-test",
    }
    instrument = "ETHUSDT-PERP.SIM"
    start, end = 7 * 60 * MINUTE_NS, 17 * 60 * MINUTE_NS
    policy = _coverage(template, instrument, start, end)
    assert policy["settlement_times_ns"][instrument] == (
        8 * 60 * MINUTE_NS,
        16 * 60 * MINUTE_NS,
    )
    assert policy["known_schedule_times_ns"][instrument] == (
        8 * 60 * MINUTE_NS,
        16 * 60 * MINUTE_NS,
        24 * 60 * MINUTE_NS,
        32 * 60 * MINUTE_NS,
        40 * 60 * MINUTE_NS,
    )
    assert policy["settlement_timestamp_tolerance_ns"] == 0
    overlapping = _coverage(template, instrument, 9 * 60 * MINUTE_NS, 19 * 60 * MINUTE_NS)
    point_in_time = 16 * 60 * MINUTE_NS + 30 * MINUTE_NS
    from rles_gym_bridge.historical import scheduled_funding_time_features

    assert scheduled_funding_time_features(
        point_in_time, policy["known_schedule_times_ns"][instrument],
    ) == scheduled_funding_time_features(
        point_in_time, overlapping["known_schedule_times_ns"][instrument],
    )
    tolerant = _coverage({**template, "settlement_timestamp_tolerance_ns": 1_000_000},
                         instrument, start, end)
    assert tolerant["settlement_timestamp_tolerance_ns"] == 1_000_000

    with pytest.raises(DataError, match="template fields"):
        _coverage({**template, "guessed_schedule": True}, instrument, start, end)


def test_wfo_refuses_selection_when_native_liquidation_facts_are_unavailable() -> None:
    from rles_orchestrator.wfo_application import _aggregate_candidate

    with pytest.raises(DataError, match="lack native liquidation facts"):
        _aggregate_candidate(
            "candidate-1",
            [{"metrics": {"sharpe": 1.0, "net_return": 0.0}, "liquidations": None,
              "liquidations_observed": False, "risk_guard_interventions": 0}],
            "median_fold_sharpe",
        )


def test_wfo_sums_only_explicit_native_liquidation_counts() -> None:
    from rles_orchestrator.wfo_application import _aggregate_candidate

    report = {"metrics": {"sharpe": 1.0, "net_return": 0.01, "max_drawdown": 0.1,
                          "turnover_ratio": 0.2},
              "liquidations": 2, "liquidations_observed": True,
              "risk_guard_interventions": 20, "risk_soft_adjustment_steps": 20,
              "risk_hard_interventions": 1,
              "risk_intervention_components_known": True}
    result = _aggregate_candidate("candidate-1", [report], "median_fold_sharpe")
    assert result["liquidations"] == 2
    assert result["risk_violations"] == 1


def test_wfo_selection_requires_classified_risk_and_counts_only_hard_guard_changes() -> None:
    from rles_data.core import DataError
    from rles_orchestrator.wfo_application import _aggregate_candidate

    base = {"metrics": {"sharpe": 1.0, "net_return": 0.01, "max_drawdown": 0.1,
                         "turnover_ratio": 0.2},
            "liquidations": 0, "liquidations_observed": True,
            "risk_guard_interventions": 25, "risk_soft_adjustment_steps": 25,
            "risk_hard_interventions": 0,
            "risk_intervention_components_known": True}
    assert _aggregate_candidate("soft-only", [base], "median_fold_sharpe")["risk_violations"] == 0
    with pytest.raises(DataError, match="hard-risk attribution"):
        _aggregate_candidate("unclassified", [{**base, "risk_intervention_components_known": False}],
                             "median_fold_sharpe")


def test_holdout_gate_fails_closed_without_classified_native_risk_facts() -> None:
    from rles_orchestrator.wfo_application import _holdout_constraints_pass, _holdout_return_gate
    from rles_research.selection import SelectionSpec

    spec = SelectionSpec(0.2, 2.0, 1, "median_fold_sharpe")
    report = {"terminal_flat": True, "liquidations_observed": True, "liquidations": 0,
              "metrics": {"max_drawdown": 0.1, "turnover_ratio": 0.5},
              "risk_intervention_components_known": True, "risk_hard_interventions": 1}
    assert _holdout_constraints_pass(report, spec)
    assert _holdout_return_gate({**report, "metrics": {**report["metrics"], "net_return": 0.01}}, spec) == (0.01, True)
    assert _holdout_return_gate({**report, "metrics": {**report["metrics"], "net_return": -0.01}}, spec) == (-0.01, False)
    assert _holdout_return_gate(report, spec) == (None, False)
    assert not _holdout_constraints_pass({**report, "risk_intervention_components_known": False}, spec)
    assert not _holdout_constraints_pass({**report, "risk_hard_interventions": 2}, spec)


def test_zero_return_fails_stress_and_holdout_profitability_gates() -> None:
    from rles_orchestrator.wfo_application import _holdout_return_gate, _stress_validation_gate
    from rles_research.selection import SelectionSpec

    spec = SelectionSpec(0.2, 2.0, 0, "median_fold_net_return", min_median_fold_net_return=0.0)
    stress_report = {
        "execution_mode": "stress", "terminal_flat": True,
        "liquidations_observed": True, "liquidations": 0,
        "risk_intervention_components_known": True, "risk_hard_interventions": 0,
        "metrics": {"net_return": 0.0, "max_drawdown": 0.0, "turnover_ratio": 0.0},
    }
    gate = _stress_validation_gate([stress_report], spec)
    assert gate["passed"] is False
    assert "fold_0:nonpositive_net_return" in gate["reasons"]
    holdout = {"metrics": {"net_return": 0.0}}
    assert _holdout_return_gate(holdout, spec) == (0.0, False)


def test_default_wfo_blocks_before_data_or_compute_when_adapter_lacks_liquidation_support(
    tmp_path, monkeypatch,
) -> None:
    from rles_orchestrator.wfo_application import run_wfo
    import rles_orchestrator.wfo_application as wfo_application

    settings = {
        "protocol_id": "wfo-no-liquidation-events", "scheme": "60/15/15",
        "start_ns": 1, "end_ns": 4, "data_start_ns": 0, "warmup_ns": 1,
        "execution_lag_guard_ns": 0, "embargo_ns": 0,
        "holdout_start_ns": 2, "holdout_end_ns": 3,
        "coverage_template": {}, "candidates": [{"candidate_id": "a"}, {"candidate_id": "b"}],
        "selection": {}, "bootstrap": {}, "pbo_max_partitions": 0,
        "reviewer_approval_sha": "", "cost_stress_validation": False,
    }
    monkeypatch.setattr(wfo_application, "NATIVE_LIQUIDATION_COUNTS_AVAILABLE", False)
    output = tmp_path / "must-not-start"
    with pytest.raises(DataError, match="cannot start.*liquidation event counts"):
        run_wfo({"output": str(output), "wfo": settings})
    assert not output.exists()


@pytest.mark.parametrize("policy", [
    {"enforce_worker_memory_limit": "true"},
    {"worker_ram_bytes": True},
    {"worker_ram_bytes": 0},
    {"worker_ram_bytes": 4 * 1024**3 + 1},
])
def test_wfo_invalid_worker_policy_cannot_consume_queries(tmp_path, monkeypatch, policy) -> None:
    import rles_orchestrator.wfo_application as wfo_application

    settings = {
        "protocol_id": "invalid-worker-policy", "scheme": "60/15/15",
        "start_ns": 1, "end_ns": 4, "data_start_ns": 0, "warmup_ns": 1,
        "execution_lag_guard_ns": 0, "embargo_ns": 0,
        "holdout_start_ns": 2, "holdout_end_ns": 3,
        "coverage_template": {}, "candidates": [{"candidate_id": "a"}, {"candidate_id": "b"}],
        "selection": {}, "bootstrap": {}, "pbo_max_partitions": 0,
        "reviewer_approval_sha": "", "cost_stress_validation": False,
    }
    monkeypatch.setattr(wfo_application, "read_curated_window",
                        lambda *_args, **_kwargs: pytest.fail("data read"))
    monkeypatch.setattr(wfo_application, "ResearchLedger", lambda _path: pytest.fail("query ledger opened"))
    output = tmp_path / "must-not-start"
    with pytest.raises(DataError, match="worker memory policy"):
        wfo_application.run_wfo({"output": str(output), "wfo": settings, **policy})
    assert not output.exists()


def test_wfo_invalid_rl_sampling_cannot_read_data_or_open_query_ledger(tmp_path, monkeypatch) -> None:
    import rles_orchestrator.wfo_application as wfo_application

    settings = {
        "protocol_id": "invalid-rl-sampling", "scheme": "60/15/15",
        "start_ns": 1, "end_ns": 4, "data_start_ns": 0, "warmup_ns": 1,
        "execution_lag_guard_ns": 0, "embargo_ns": 0,
        "holdout_start_ns": 2, "holdout_end_ns": 3,
        "coverage_template": {}, "candidates": [{"candidate_id": "a"}, {"candidate_id": "b"}],
        "selection": {}, "bootstrap": {}, "pbo_max_partitions": 0,
        "reviewer_approval_sha": "", "cost_stress_validation": False,
    }
    monkeypatch.setattr(wfo_application, "read_curated_window",
                        lambda *_args, **_kwargs: pytest.fail("data read"))
    monkeypatch.setattr(wfo_application, "ResearchLedger", lambda _path: pytest.fail("query ledger opened"))
    output = tmp_path / "must-not-start"
    with pytest.raises(DataError, match="explicit fixed episode steps"):
        wfo_application.run_wfo({
            "output": str(output), "wfo": settings, "rl_num_envs": 2,
        })
    assert not output.exists()


@pytest.mark.short_wfo
@pytest.mark.artifact_restore
def test_wfo_suspends_validation_protocol_on_keyboard_interrupt(tmp_path, monkeypatch) -> None:
    import rles_orchestrator.wfo_application as wfo_application
    from rles_orchestrator.wfo_application import run_wfo
    from rles_research.access import ProjectResearchRegistry
    from rles_research.ledger import ResearchLedger

    _research, manifests, _dataset_hash, _feature_hash = _paired_folds()
    manifest = manifests[0]
    access_order: list[str] = []
    source_identity = _market_source_identity()
    original_record_development = wfo_application.research_access.record_development
    registry = ProjectResearchRegistry(tmp_path / "project-research.sqlite")
    original_reserve_intent = registry.reserve_query_intent
    original_start_intents = registry.start_query_intents

    def record_development(*args, **kwargs):
        assert kwargs["source_identity"] == source_identity
        access_order.append("registered")
        return original_record_development(*args, **kwargs)

    def read_market(_path, start_ns, end_ns, **_kwargs):
        assert start_ns == settings["data_start_ns"]
        assert end_ns == settings["holdout_end_ns"]
        access_order.append("payload")
        return SimpleNamespace(events=(), snapshot_hash="a" * 64)

    def reserve_intent(**kwargs):
        assert kwargs["identity"]["source_sha256"] == source_identity["snapshot_hash"]
        access_order.append("intent-reserved")
        return original_reserve_intent(**kwargs)

    def start_intents(**kwargs):
        access_order.append("intent-started")
        return original_start_intents(**kwargs)

    monkeypatch.setattr(wfo_application, "CuratedStore",
                        lambda _events: SimpleNamespace(snapshot_hash="a" * 64))
    monkeypatch.setattr(wfo_application, "read_curated_window", read_market)
    monkeypatch.setattr(wfo_application, "curated_source_identity",
                        lambda _path: (access_order.append("metadata") or source_identity))
    monkeypatch.setattr(wfo_application.research_access, "record_development", record_development)
    monkeypatch.setattr(wfo_application.research_access, "project_registry", lambda: registry)
    monkeypatch.setattr(registry, "reserve_query_intent", reserve_intent)
    monkeypatch.setattr(registry, "start_query_intents", start_intents)
    monkeypatch.setattr(wfo_application, "load_rules", lambda _path: ())
    monkeypatch.setattr(wfo_application, "historical_feature_spec",
                        lambda _ages: SimpleNamespace(max_lookback_ns=1, hash="b" * 64))
    monkeypatch.setattr(wfo_application, "build_folds", lambda **_kwargs: [manifest])
    monkeypatch.setattr(wfo_application, "_require_window", lambda *_args: None)
    monkeypatch.setattr(wfo_application, "_source_tree_hash", lambda _root: "c" * 64)

    output = tmp_path / "interrupted-wfo"
    settings = {
        "protocol_id": "interrupted-wfo", "scheme": "60/15/15",
        "start_ns": manifest.train.score_start_ns,
        "end_ns": manifest.test.score_end_ns + 2,
        "data_start_ns": manifest.train.warmup_start_ns, "warmup_ns": 2 * DAY_NS,
        "execution_lag_guard_ns": 0, "embargo_ns": 0,
        "holdout_start_ns": manifest.test.score_end_ns + 1,
        "holdout_end_ns": manifest.test.score_end_ns + 2,
        "coverage_template": {
            "maximum_age_ns": {"mark": 1, "index": 1, "funding_update": 1},
            "settlement_interval_minutes": 480, "settlement_anchor_ns": 0,
            "settlement_max_delay_ns": 1, "provenance": "test",
        },
        "candidates": [{"candidate_id": "candidate-a"}, {"candidate_id": "candidate-b"}],
        "selection": {}, "bootstrap": {}, "pbo_max_partitions": 0,
        "reviewer_approval_sha": "", "cost_stress_validation": False,
    }
    config = {
        "output": str(output),
        "environment": {"market_path": "unused", "rules_path": str(Path(__file__).resolve()),
                        "instrument": "ETHUSDT-PERP.SIM"},
        "wfo": settings,
    }

    def interrupted(_run: dict) -> dict:
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        run_wfo(config, train_fn=interrupted)

    assert access_order == ["metadata", "registered", "intent-reserved",
                            "intent-reserved", "intent-started", "payload"]

    ledger = ResearchLedger(output / "research.sqlite")
    assert ledger.state("interrupted-wfo")["state"] == "SUSPENDED"
    assert ledger.query_counts("interrupted-wfo") == {"validation": 1}
    assert ledger.verify_audit_chain()
    attempts = wfo_application.research_access.project_registry().trial_attempt_summary(
        "interrupted-wfo",
    )
    assert attempts["protocol_registered_attempt_count"] == 2
    assert all(attempt["events"][-1]["status"] == "interrupted"
               for attempt in attempts["attempts"])
    failure = json.loads((output / "FAILED.json").read_text(encoding="utf-8"))
    assert failure == {"error": "KeyboardInterrupt", "detail": "interrupted"}


@pytest.mark.short_wfo
def test_wfo_persists_validation_rejection_report_without_opening_test_or_holdout(tmp_path, monkeypatch) -> None:
    import rles_orchestrator.wfo_application as wfo_application
    from rles_orchestrator.application import _verify_file, _write_verified
    from rles_orchestrator.wfo_application import run_wfo
    from rles_research.ledger import ResearchLedger

    _research, manifests, dataset_hash, feature_hash = _paired_folds()
    manifest = manifests[0]
    monkeypatch.setattr(wfo_application, "CuratedStore",
                        lambda _events: SimpleNamespace(snapshot_hash=dataset_hash))
    monkeypatch.setattr(wfo_application, "read_curated_window",
                        lambda *_args, **_kwargs: SimpleNamespace(events=(), snapshot_hash="a" * 64))
    monkeypatch.setattr(wfo_application, "curated_source_identity",
                        lambda _path: _market_source_identity())
    monkeypatch.setattr(wfo_application, "load_rules", lambda _path: ())
    def feature_spec(_ages, profile="legacy"):
        selected_hash = feature_hash if profile == "legacy" else "e" * 64
        return SimpleNamespace(max_lookback_ns=MINUTE_NS, hash=selected_hash)

    monkeypatch.setattr(wfo_application, "historical_feature_spec", feature_spec)
    monkeypatch.setattr(wfo_application, "build_folds", lambda **_kwargs: (manifest,))
    monkeypatch.setattr(wfo_application, "_require_window", lambda *_args: None)
    monkeypatch.setattr(wfo_application, "_source_tree_hash", lambda _root: "c" * 64)

    output = tmp_path / "rejected-wfo"
    candidates = [
        {"candidate_id": "candidate-a", "route": "A0", "algorithm": "PPO", "steps": 4,
         "seed": 1, "deadband": "0.01", "feature_profile": "legacy"},
        {"candidate_id": "candidate-b", "route": "A0", "algorithm": "PPO", "steps": 4,
         "seed": 2, "deadband": "0.05", "feature_profile": "causal_mtf_v2"},
    ]
    settings = {
        "protocol_id": "rejected-wfo", "scheme": "60/15/15",
        "start_ns": manifest.train.score_start_ns,
        "end_ns": manifest.test.score_end_ns + 2 * DAY_NS,
        "data_start_ns": manifest.train.warmup_start_ns, "warmup_ns": DAY_NS,
        "execution_lag_guard_ns": 0, "embargo_ns": 0,
        "holdout_start_ns": manifest.test.score_end_ns + DAY_NS,
        "holdout_end_ns": manifest.test.score_end_ns + 2 * DAY_NS,
        "coverage_template": {
            "maximum_age_ns": {"mark": MINUTE_NS, "index": MINUTE_NS, "funding_update": DAY_NS},
            "settlement_interval_minutes": 480, "settlement_anchor_ns": 0,
            "settlement_max_delay_ns": 1, "provenance": "test-only coverage",
        },
        "candidates": candidates,
        "selection": {
            "max_drawdown": 0.2, "max_turnover": 2.0, "max_risk_violations": 0,
            "primary_metric": "median_fold_net_return", "min_median_fold_net_return": 0.0,
        },
        "bootstrap": {}, "pbo_max_partitions": 1, "reviewer_approval_sha": "d" * 64,
        "cost_stress_validation": False,
    }
    config = {
        "output": str(output),
        "environment": {"market_path": "unused", "rules_path": str(Path(__file__).resolve()),
                        "instrument": "ETHUSDT-PERP.SIM", "execution": _execution_config()},
        "wfo": settings,
    }
    seen_candidates: list[tuple[str, str]] = []

    def train_negative_candidate(run: dict) -> dict:
        profile = run["environment"]["feature_profile"]
        seen_candidates.append((run["environment"]["execution"]["deadband"], profile))
        run_output = Path(run["output"])
        run_output.mkdir(parents=True)
        score_window = Window(manifest.train.score_start_ns, manifest.train.score_start_ns,
                              manifest.train.score_end_ns)
        _write_verified(run_output / "bundle.json", {
            "preprocessing": {"scaler": {
                "feature_hash": feature_hash if profile == "legacy" else "e" * 64,
                "mean": [0.0], "std": [1.0],
                "train_window_hash": digest(score_window), "sample_count": 10,
            }},
        })
        _write_verified(run_output / "report.json", {
            "metrics": {
                "net_return": -0.01, "max_drawdown": 0.01, "turnover_ratio": 0.1,
                "sharpe": -1.0, "sortino": -1.0,
            },
            "liquidations_observed": True, "liquidations": 0,
            "risk_intervention_components_known": True, "risk_hard_interventions": 0,
        })
        return {"output": str(run_output)}

    with pytest.raises(DataError, match="no feasible validation-selected candidate"):
        run_wfo(config, train_fn=train_negative_candidate)
    assert seen_candidates == [("0.01", "legacy"), ("0.05", "causal_mtf_v2")]

    report_path = output / "wfo-report.json"
    _verify_file(report_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["conclusion"] == "NO_VALIDATION_CANDIDATE"
    assert report["selection"]["status"] == "NO_FEASIBLE_MODEL"
    assert report["holdout_state"] == "LOCKED"
    assert report["fold_tests"] == []
    assert len(report["fold_validation_artifacts"]) == 2
    assert report["candidate_feature_hashes"] == {
        "candidate-a": feature_hash, "candidate-b": "e" * 64,
    }
    assert set(report["fold_manifests"]) == {"candidate-a", "candidate-b"}
    assert all(item["metrics"]["net_return"] == -0.01
               for item in report["fold_validation_artifacts"])
    ledger = ResearchLedger(output / "research.sqlite")
    assert ledger.state("rejected-wfo")["state"] == "ABORTED"
    assert ledger.query_counts("rejected-wfo") == {"validation": 2}
    assert ledger.verify_audit_chain()
    attempts = wfo_application.research_access.project_registry().trial_attempt_summary("rejected-wfo")
    assert attempts["protocol_registered_attempt_count"] == 2
    assert {attempt["seed"] for attempt in attempts["attempts"]} == {1, 2}
    assert all(attempt["feature_sha256"] for attempt in attempts["attempts"])
    assert all(attempt["events"][-1]["status"] == "not_selected"
               for attempt in attempts["attempts"])


@pytest.mark.short_wfo
def test_wfo_stress_validation_gate_blocks_candidates_before_test(tmp_path, monkeypatch) -> None:
    import rles_orchestrator.wfo_application as wfo_application
    from rles_orchestrator.application import _verify_file, _write_verified
    from rles_orchestrator.wfo_application import run_wfo
    from rles_research.ledger import ResearchLedger

    _research, manifests, dataset_hash, feature_hash = _paired_folds()
    manifest = manifests[0]
    monkeypatch.setattr(wfo_application, "CuratedStore",
                        lambda _events: SimpleNamespace(snapshot_hash=dataset_hash))
    monkeypatch.setattr(wfo_application, "read_curated_window",
                        lambda *_args, **_kwargs: SimpleNamespace(events=(), snapshot_hash="a" * 64))
    monkeypatch.setattr(wfo_application, "curated_source_identity",
                        lambda _path: _market_source_identity())
    monkeypatch.setattr(wfo_application, "load_rules", lambda _path: ())
    monkeypatch.setattr(wfo_application, "historical_feature_spec",
                        lambda _ages: SimpleNamespace(max_lookback_ns=MINUTE_NS, hash=feature_hash))
    monkeypatch.setattr(wfo_application, "build_folds", lambda **_kwargs: (manifest,))
    monkeypatch.setattr(wfo_application, "_require_window", lambda *_args: None)
    monkeypatch.setattr(wfo_application, "_source_tree_hash", lambda _root: "c" * 64)

    output = tmp_path / "stress-gated-wfo"
    candidates = [
        {"candidate_id": "candidate-a", "route": "A0", "algorithm": "PPO", "steps": 4,
         "seed": 1, "deadband": "0.01"},
        {"candidate_id": "candidate-b", "route": "A0", "algorithm": "PPO", "steps": 4,
         "seed": 2, "deadband": "0.05"},
    ]
    settings = {
        "protocol_id": "stress-gated-wfo", "scheme": "60/15/15",
        "start_ns": manifest.train.score_start_ns,
        "end_ns": manifest.test.score_end_ns + 2 * DAY_NS,
        "data_start_ns": manifest.train.warmup_start_ns, "warmup_ns": DAY_NS,
        "execution_lag_guard_ns": 0, "embargo_ns": 0,
        "holdout_start_ns": manifest.test.score_end_ns + DAY_NS,
        "holdout_end_ns": manifest.test.score_end_ns + 2 * DAY_NS,
        "coverage_template": {
            "maximum_age_ns": {"mark": MINUTE_NS, "index": MINUTE_NS, "funding_update": DAY_NS},
            "settlement_interval_minutes": 480, "settlement_anchor_ns": 0,
            "settlement_max_delay_ns": 1, "provenance": "test-only coverage",
        },
        "candidates": candidates,
        "selection": {
            "max_drawdown": 0.2, "max_turnover": 2.0, "max_risk_violations": 0,
            "primary_metric": "median_fold_net_return", "min_median_fold_net_return": 0.0,
        },
        "bootstrap": {}, "pbo_max_partitions": 1, "reviewer_approval_sha": "d" * 64,
        "cost_stress_validation": True,
    }
    config = {
        "output": str(output),
        "environment": {"market_path": "unused", "rules_path": str(Path(__file__).resolve()),
                        "instrument": "ETHUSDT-PERP.SIM", "execution": _execution_config()},
        "wfo": settings,
    }

    def train_positive_candidate(run: dict) -> dict:
        run_output = Path(run["output"])
        run_output.mkdir(parents=True)
        score_window = Window(manifest.train.score_start_ns, manifest.train.score_start_ns,
                              manifest.train.score_end_ns)
        _write_verified(run_output / "bundle.json", {
            "preprocessing": {"scaler": {
                "feature_hash": feature_hash, "mean": [0.0], "std": [1.0],
                "train_window_hash": digest(score_window), "sample_count": 10,
            }},
        })
        _write_verified(run_output / "report.json", {
            "execution_mode": "base", "terminal_flat": True,
            "metrics": {"net_return": 0.01, "max_drawdown": 0.01, "turnover_ratio": 0.1,
                        "sharpe": 1.0, "sortino": 1.0},
            "liquidations_observed": True, "liquidations": 0,
            "risk_intervention_components_known": True, "risk_hard_interventions": 0,
        })
        return {"output": str(run_output)}

    def stress_loss(_run: dict, _bundle: Path, **_window: dict) -> dict:
        return {
            "execution_mode": "stress", "terminal_flat": True,
            "metrics": {"net_return": -0.001, "max_drawdown": 0.02, "turnover_ratio": 0.1},
            "liquidations_observed": True, "liquidations": 0,
            "risk_intervention_components_known": True, "risk_hard_interventions": 0,
        }

    with pytest.raises(DataError, match="no feasible validation-selected candidate"):
        run_wfo(config, train_fn=train_positive_candidate, stress_validation_fn=stress_loss)

    report_path = output / "wfo-report.json"
    _verify_file(report_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["selection"]["status"] == "NO_FEASIBLE_MODEL"
    assert report["selection"]["rejected"] == {
        "candidate-a": "STRESS_VALIDATION_GATE", "candidate-b": "STRESS_VALIDATION_GATE",
    }
    assert report["fold_tests"] == []
    assert len(report["stress_validation_artifacts"]) == 2
    assert all(not row["gate"]["passed"] for row in report["stress_validation_artifacts"])
    ledger = ResearchLedger(output / "research.sqlite")
    assert ledger.query_counts("stress-gated-wfo") == {"validation": 4}
    assert ledger.verify_audit_chain()


@pytest.mark.parametrize("boundary", ["completed", "checkpoint", "no-checkpoint", "evaluation-started", "frozen-checkpoint"])
@pytest.mark.short_wfo
@pytest.mark.artifact_restore
def test_wfo_resumes_completed_validation_boundary_without_retraining_or_recharging(tmp_path, monkeypatch, boundary) -> None:
    import rles_orchestrator.wfo_application as wfo_application
    import rles_research.ledger as ledger_module
    from rles_orchestrator.application import _verify_file, _write_verified
    from rles_orchestrator.wfo_application import run_wfo
    from rles_research.ledger import ResearchLedger

    _research, manifests, dataset_hash, feature_hash = _paired_folds()
    manifest = manifests[0]
    monkeypatch.setattr(wfo_application, "CuratedStore",
                        lambda _events: SimpleNamespace(snapshot_hash=dataset_hash))
    monkeypatch.setattr(wfo_application, "read_curated_window",
                        lambda *_args, **_kwargs: SimpleNamespace(events=(), snapshot_hash="a" * 64))
    monkeypatch.setattr(wfo_application, "curated_source_identity",
                        lambda _path: _market_source_identity())
    monkeypatch.setattr(wfo_application, "load_rules", lambda _path: ())
    monkeypatch.setattr(wfo_application, "historical_feature_spec",
                        lambda _ages: SimpleNamespace(max_lookback_ns=MINUTE_NS, hash=feature_hash))
    monkeypatch.setattr(wfo_application, "build_folds", lambda **_kwargs: (manifest,))
    monkeypatch.setattr(wfo_application, "_require_window", lambda *_args: None)
    monkeypatch.setattr(wfo_application, "_source_tree_hash", lambda _root: "c" * 64)

    output = tmp_path / "resumable-wfo"
    candidates = [
        {"candidate_id": "candidate-a", "route": "A0", "algorithm": "PPO", "steps": 4, "seed": 1},
        {"candidate_id": "candidate-b", "route": "A0", "algorithm": "PPO", "steps": 4, "seed": 2},
    ]
    holdout_start = manifest.test.score_end_ns + DAY_NS
    settings = {
        "protocol_id": "resumable-wfo", "scheme": "60/15/15",
        "start_ns": manifest.train.score_start_ns,
        "end_ns": holdout_start + DAY_NS,
        "data_start_ns": manifest.train.warmup_start_ns, "warmup_ns": DAY_NS,
        "execution_lag_guard_ns": 0, "embargo_ns": 0,
        "holdout_start_ns": holdout_start, "holdout_end_ns": holdout_start + DAY_NS,
        "coverage_template": {
            "maximum_age_ns": {"mark": MINUTE_NS, "index": MINUTE_NS, "funding_update": DAY_NS},
            "settlement_interval_minutes": 480, "settlement_anchor_ns": 0,
            "settlement_max_delay_ns": 1, "provenance": "test-only coverage",
        },
        "candidates": candidates,
        "selection": {
            "max_drawdown": 0.2, "max_turnover": 2.0, "max_risk_violations": 0,
            "primary_metric": "median_fold_net_return", "min_median_fold_net_return": 0.0,
        },
        "bootstrap": {}, "pbo_max_partitions": 1, "reviewer_approval_sha": "d" * 64,
        "cost_stress_validation": False,
    }
    config = {
        "output": str(output),
        "environment": {"market_path": "unused", "rules_path": str(Path(__file__).resolve()),
                        "instrument": "ETHUSDT-PERP.SIM"},
        "wfo": settings,
    }
    train_calls: list[str] = []

    def train_candidate(run: dict) -> dict:
        candidate_id = Path(run["output"]).name
        train_calls.append(candidate_id)
        run_output = Path(run["output"])
        run_output.mkdir(parents=True, exist_ok=True)
        if boundary != "completed" and len(train_calls) == 1:
            (run_output / "run.json").write_text(json.dumps({
                "schema": "rles.historical-run.v1", "run_id": "attempt-1", "attempt_id": "attempt-1",
            }), encoding="utf-8")
            if boundary != "no-checkpoint":
                checkpoint = run_output / "R1-checkpoints" / "step-000000000002"
                checkpoint.mkdir(parents=True)
                (checkpoint / "COMPLETED").write_text("test-only-checkpoint", encoding="ascii")
            if boundary == "evaluation-started":
                _write_verified(run_output / "bundle.json", {"test_only": True})
            raise KeyboardInterrupt
        if boundary == "checkpoint" and candidate_id == "candidate-a":
            assert run.get("resume_checkpoint") == str(run_output / "R1-checkpoints" / "step-000000000002")
            assert run["environment"]["start_ns"] == manifest.train.score_start_ns
            assert run["evaluation"]["start_ns"] == manifest.validation.score_start_ns
        score_window = Window(manifest.train.score_start_ns, manifest.train.score_start_ns,
                              manifest.train.score_end_ns)
        bundle_hash = _write_verified(run_output / "bundle.json", {
            "preprocessing": {"scaler": {
                "feature_hash": feature_hash, "mean": [0.0], "std": [1.0],
                "train_window_hash": digest(score_window), "sample_count": 10,
            }},
        })
        report_hash = _write_verified(run_output / "report.json", {
            "metrics": {
                "net_return": -0.01, "max_drawdown": 0.01, "turnover_ratio": 0.1,
                "sharpe": -1.0, "sortino": -1.0,
            },
            "liquidations_observed": True, "liquidations": 0,
            "risk_intervention_components_known": True, "risk_hard_interventions": 0,
        })
        (run_output / "COMPLETED.json").write_text(json.dumps({
            "bundle_sha256": bundle_hash, "report_sha256": report_hash,
        }), encoding="utf-8")
        return {"output": str(run_output)}

    original_record_query = ResearchLedger.record_query
    interrupt_before_second_reservation = True

    def interrupt_at_boundary(self, protocol_id, query_id, kind, candidate, fold_id) -> None:
        nonlocal interrupt_before_second_reservation
        if boundary == "completed" and query_id == "validation-0-candidate-b" and interrupt_before_second_reservation:
            interrupt_before_second_reservation = False
            raise KeyboardInterrupt
        original_record_query(self, protocol_id, query_id, kind, candidate, fold_id)

    monkeypatch.setattr(ledger_module.ResearchLedger, "record_query", interrupt_at_boundary)
    with pytest.raises(KeyboardInterrupt):
        run_wfo(config, train_fn=train_candidate)
    assert train_calls == ["candidate-a"]
    assert ResearchLedger(output / "research.sqlite").state("resumable-wfo")["state"] == "SUSPENDED"

    if boundary in {"no-checkpoint", "evaluation-started", "frozen-checkpoint"}:
        message = {"no-checkpoint": "no completed stage checkpoint",
                   "evaluation-started": "evaluation may have started",
                   "frozen-checkpoint": "only resumable before recipe freeze"}[boundary]
        if boundary == "frozen-checkpoint":
            frozen_ledger = ResearchLedger(output / "research.sqlite")
            frozen_ledger.resume_suspended("resumable-wfo")
            frozen_ledger.freeze_recipe("resumable-wfo", recipe_sha="e" * 64,
                                        code_sha="c" * 64, selection_spec_sha="f" * 64)
        with pytest.raises(DataError, match=message):
            run_wfo(config, train_fn=train_candidate, resume=True)
        ledger = ResearchLedger(output / "research.sqlite")
        assert train_calls == ["candidate-a"]
        assert ledger.query_counts("resumable-wfo") == {"validation": 1}
        assert ledger.state("resumable-wfo")["state"] == "ABORTED"
        assert ledger.verify_audit_chain()
        return

    with pytest.raises(DataError, match="no feasible validation-selected candidate"):
        run_wfo(config, train_fn=train_candidate, resume=True)
    expected_calls = ["candidate-a", "candidate-b"] if boundary == "completed" else ["candidate-a", "candidate-a", "candidate-b"]
    assert train_calls == expected_calls
    report_path = output / "wfo-report.json"
    _verify_file(report_path)
    assert json.loads(report_path.read_text(encoding="utf-8"))["holdout_state"] == "LOCKED"
    ledger = ResearchLedger(output / "research.sqlite")
    assert ledger.state("resumable-wfo")["state"] == "ABORTED"
    assert ledger.query_counts("resumable-wfo") == {"validation": 2}
    assert ledger.verify_audit_chain()
    assert list(output.glob("FAILED-resumed-*.json"))


@pytest.mark.parametrize("interrupt_after_holdout_open", [False, True])
@pytest.mark.short_wfo
@pytest.mark.artifact_restore
def test_wfo_resumes_after_persisted_fold_test_without_replaying_one_shot_test(
        tmp_path, monkeypatch, interrupt_after_holdout_open: bool) -> None:
    import rles_orchestrator.wfo_application as wfo_application
    from rles_orchestrator.application import _verify_file, _write_verified
    from rles_orchestrator.wfo_application import run_wfo
    from rles_research.ledger import ResearchLedger

    _research, manifests, dataset_hash, feature_hash = _paired_folds()
    monkeypatch.setattr(wfo_application, "CuratedStore",
                        lambda _events: SimpleNamespace(snapshot_hash=dataset_hash))
    monkeypatch.setattr(wfo_application, "read_curated_window",
                        lambda *_args, **_kwargs: SimpleNamespace(events=(), snapshot_hash="a" * 64))
    monkeypatch.setattr(wfo_application, "curated_source_identity",
                        lambda _path: _market_source_identity())
    monkeypatch.setattr(wfo_application, "load_rules", lambda _path: ())
    monkeypatch.setattr(wfo_application, "historical_feature_spec",
                        lambda _ages: SimpleNamespace(max_lookback_ns=MINUTE_NS, hash=feature_hash))
    monkeypatch.setattr(wfo_application, "build_folds", lambda **_kwargs: manifests)
    monkeypatch.setattr(wfo_application, "_require_window", lambda *_args: None)
    monkeypatch.setattr(wfo_application, "_source_tree_hash", lambda _root: "c" * 64)

    output = tmp_path / "resumable-test-wfo"
    candidates = [
        {"candidate_id": "candidate-a", "route": "A0", "algorithm": "PPO", "steps": 4,
         "seed": 1, "deadband": "0.05"},
        {"candidate_id": "candidate-b", "route": "A0", "algorithm": "PPO", "steps": 4,
         "seed": 2, "deadband": "0.01"},
    ]
    holdout_start = manifests[-1].test.score_end_ns + DAY_NS
    settings = {
        "protocol_id": "resumable-test-wfo", "scheme": "60/15/15",
        "start_ns": manifests[0].train.score_start_ns,
        "end_ns": holdout_start + DAY_NS,
        "data_start_ns": manifests[0].train.warmup_start_ns, "warmup_ns": 2 * DAY_NS,
        "execution_lag_guard_ns": 0, "embargo_ns": 0,
        "holdout_start_ns": holdout_start, "holdout_end_ns": holdout_start + DAY_NS,
        "coverage_template": {
            "maximum_age_ns": {"mark": MINUTE_NS, "index": MINUTE_NS, "funding_update": DAY_NS},
            "settlement_interval_minutes": 480, "settlement_anchor_ns": 0,
            "settlement_max_delay_ns": 1, "provenance": "test-only coverage",
        },
        "candidates": candidates,
        "selection": {
            "max_drawdown": 0.2, "max_turnover": 2.0, "max_risk_violations": 0,
            "primary_metric": "median_fold_net_return", "min_median_fold_net_return": 0.0,
        },
        "bootstrap": {"block_length": 1, "draws": 100, "seed": 1, "confidence": 0.95},
        "pbo_max_partitions": 1, "reviewer_approval_sha": "d" * 64,
        "cost_stress_validation": False,
    }
    config = {
        "output": str(output),
        "environment": {"market_path": "unused", "rules_path": str(Path(__file__).resolve()),
                        "instrument": "ETHUSDT-PERP.SIM", "execution": _execution_config()},
        "wfo": settings,
    }
    train_calls: list[str] = []
    backtest_calls: list[str] = []
    deadband_runs: list[str] = []
    interrupt_final_retrain = True

    def account_report() -> dict:
        return {
            "account_rows": [
                {"equity": 10000.0, "external_cashflow": 0.0},
                {"equity": 10001.0, "external_cashflow": 0.0},
                {"equity": 10003.0, "external_cashflow": 0.0},
                {"equity": 10006.0, "external_cashflow": 0.0},
                {"equity": 10010.0, "external_cashflow": 0.0},
                {"equity": 10015.0, "external_cashflow": 0.0},
            ],
            "metrics": {
                "net_return": 0.01, "max_drawdown": 0.01, "turnover_ratio": 0.1,
                "sharpe": 1.0, "sortino": 1.0,
            },
            "liquidations_observed": True, "liquidations": 0,
            "risk_intervention_components_known": True, "risk_hard_interventions": 0,
            "terminal_flat": True,
        }

    def train_candidate(run: dict, *, evaluate: bool = True) -> dict:
        nonlocal interrupt_final_retrain
        run_output = Path(run["output"])
        run_output.mkdir(parents=True, exist_ok=True)
        deadband_runs.append(run["environment"]["execution"]["deadband"])
        if evaluate:
            candidate_id = run_output.name
            train_calls.append(candidate_id)
            score_window = Window(run["environment"]["start_ns"], run["environment"]["start_ns"],
                                  run["environment"]["end_ns"])
            bundle_hash = _write_verified(run_output / "bundle.json", {
                "environment": run["environment"],
                "preprocessing": {"scaler": {
                    "feature_hash": feature_hash, "mean": [0.0], "std": [1.0],
                    "train_window_hash": digest(score_window), "sample_count": 10,
                }},
            })
            report_hash = _write_verified(run_output / "report.json", account_report())
            (run_output / "COMPLETED.json").write_text(json.dumps({
                "bundle_sha256": bundle_hash, "report_sha256": report_hash,
            }), encoding="utf-8")
        else:
            recipe = {key: value for key, value in run.items()
                      if key not in {"output", "resume_checkpoint"}}
            if interrupt_final_retrain and "resume_checkpoint" not in run:
                train_calls.append("final-retrain-interrupted")
                run_id = "synthetic-final"
                (run_output / "run.json").write_text(json.dumps({
                    "schema": "rles.historical-run.v1", "run_id": run_id,
                    "attempt_id": run_id, "recipe_hash": digest(recipe),
                }), encoding="utf-8")
                checkpoint = run_output / "R1-checkpoints" / "step-000000000004"
                checkpoint.mkdir(parents=True)
                (checkpoint / "COMPLETED").write_text("synthetic-boundary", encoding="ascii")
                interrupt_final_retrain = False
                raise KeyboardInterrupt
            assert run.get("resume_checkpoint") == str(
                run_output / "R1-checkpoints" / "step-000000000004"
            )
            train_calls.append("final-retrain-resumed")
            bundle_hash = _write_verified(run_output / "bundle.json", {
                "environment": run["environment"],
                "training_end_ns": run["environment"]["end_ns"],
                "actor": {"kind": "rl", "path": "r1", "actor_hash": "e" * 64},
                "lineage": [],
            })
            (run_output / "COMPLETED.json").write_text(json.dumps({
                "bundle_sha256": bundle_hash, "route": run["route"],
                "algorithm": run["algorithm"],
            }), encoding="utf-8")
            (run_output / "run.json").write_text(json.dumps({
                "schema": "rles.historical-run.v1", "run_id": "synthetic-final",
                "attempt_id": "synthetic-final", "recipe_hash": digest(recipe),
            }), encoding="utf-8")
        return {"output": str(run_output)}

    def backtest_candidate(_run: dict, bundle_root: Path) -> dict:
        stage = "holdout" if Path(bundle_root) == output / "final-retrain" else "test"
        backtest_calls.append(stage)
        return account_report()

    baseline_calls = 0

    def interrupt_after_first_durable_test(*_args, **_kwargs) -> dict:
        nonlocal baseline_calls
        baseline_calls += 1
        if baseline_calls == 2:
            raise KeyboardInterrupt
        return {"B0": {"metrics": {"net_return": 0.0}}}

    # Interrupt before the second baseline access is reserved. The first fixed
    # baseline batch is complete, so resume must reuse both completion receipts.
    original_authorize = wfo_application.authorize_evaluation

    def interrupt_before_second_baseline(*args, **kwargs):
        if kwargs.get("component") == "baselines" and baseline_calls == 1:
            raise KeyboardInterrupt
        return original_authorize(*args, **kwargs)

    monkeypatch.setattr(wfo_application, "authorize_evaluation", interrupt_before_second_baseline)
    monkeypatch.setattr(wfo_application, "_baseline_reports", interrupt_after_first_durable_test)
    with pytest.raises(KeyboardInterrupt):
        run_wfo(config, train_fn=train_candidate, backtest_fn=backtest_candidate)
    assert backtest_calls == ["test", "test"]
    assert len(train_calls) == 4
    assert deadband_runs == ["0.05", "0.01", "0.05", "0.01"]
    first_test_report = output / "folds" / "60-15-15-0000" / "test" / "report.json"
    _verify_file(first_test_report)
    assert ResearchLedger(output / "research.sqlite").state("resumable-test-wfo")["state"] == "SUSPENDED"

    def baseline_ok(*_args, **_kwargs) -> dict:
        return {"B0": {"metrics": {"net_return": 0.0}}}

    monkeypatch.setattr(wfo_application, "_baseline_reports", baseline_ok)
    monkeypatch.setattr(wfo_application, "authorize_evaluation", original_authorize)
    with pytest.raises(KeyboardInterrupt):
        run_wfo(config, train_fn=train_candidate, backtest_fn=backtest_candidate, resume=True)
    ledger = ResearchLedger(output / "research.sqlite")
    assert ledger.state("resumable-test-wfo")["state"] == "SUSPENDED"
    assert ledger.state("resumable-test-wfo")["resume_state"] == "RECIPE_FROZEN"
    assert (output / "final-retrain" / "R1-checkpoints" / "step-000000000004"
            / "COMPLETED").is_file()
    assert train_calls[-1] == "final-retrain-interrupted"

    original_open_holdout = ResearchLedger.open_holdout

    def interrupt_before_holdout_open(self, *args, **kwargs):
        if interrupt_after_holdout_open:
            original_open_holdout(self, *args, **kwargs)
        raise KeyboardInterrupt

    monkeypatch.setattr(ResearchLedger, "open_holdout", interrupt_before_holdout_open)
    with pytest.raises(KeyboardInterrupt):
        run_wfo(config, train_fn=train_candidate, backtest_fn=backtest_candidate, resume=True)
    ledger = ResearchLedger(output / "research.sqlite")
    assert ledger.state("resumable-test-wfo")["state"] == "SUSPENDED"
    expected_resume_state = "HOLDOUT_OPENED" if interrupt_after_holdout_open else "RETRAINED"
    assert ledger.state("resumable-test-wfo")["resume_state"] == expected_resume_state
    assert backtest_calls == ["test", "test"]
    assert len(train_calls) == 6 and train_calls[-1] == "final-retrain-resumed"
    assert deadband_runs[-1] == "0.05"

    if interrupt_after_holdout_open:
        monkeypatch.setattr(ResearchLedger, "open_holdout", original_open_holdout)
        with pytest.raises(DataError, match="refusing to replay"):
            run_wfo(config, train_fn=train_candidate, backtest_fn=backtest_candidate, resume=True)
        ledger = ResearchLedger(output / "research.sqlite")
        assert ledger.state("resumable-test-wfo")["state"] == "ABORTED"
        assert backtest_calls == ["test", "test"]
        assert ledger.verify_audit_chain()
        return

    monkeypatch.setattr(ResearchLedger, "open_holdout", original_open_holdout)
    original_close_holdout = ResearchLedger.close_holdout

    def interrupt_before_holdout_close(self, *args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(ResearchLedger, "close_holdout", interrupt_before_holdout_close)
    with pytest.raises(KeyboardInterrupt):
        run_wfo(config, train_fn=train_candidate, backtest_fn=backtest_candidate, resume=True)
    _verify_file(output / "holdout-report.json")
    ledger = ResearchLedger(output / "research.sqlite")
    assert ledger.state("resumable-test-wfo")["state"] == "SUSPENDED"
    assert ledger.state("resumable-test-wfo")["resume_state"] == "HOLDOUT_OPENED"
    assert backtest_calls == ["test", "test", "holdout"]

    monkeypatch.setattr(ResearchLedger, "close_holdout", original_close_holdout)
    original_write_verified = wfo_application._write_verified

    def interrupt_before_final_report(path, payload):
        if Path(path) == output / "wfo-report.json":
            raise KeyboardInterrupt
        return original_write_verified(path, payload)

    monkeypatch.setattr(wfo_application, "_write_verified", interrupt_before_final_report)
    with pytest.raises(KeyboardInterrupt):
        run_wfo(config, train_fn=train_candidate, backtest_fn=backtest_candidate, resume=True)
    assert not (output / "wfo-report.json").exists()
    assert ResearchLedger(output / "research.sqlite").state("resumable-test-wfo")["state"] == "CLOSED"

    monkeypatch.setattr(wfo_application, "_write_verified", original_write_verified)
    registry = wfo_application.research_access.project_registry()
    registry.register_trial_attempt(
        attempt_key="standalone-before-wfo-report",
        protocol_id="standalone-application-training-v1", candidate_id="standalone-run",
        recipe={"steps": 64}, feature_sha="e" * 64, seed=13,
    )
    registry.record_trial_attempt_event(
        "standalone-before-wfo-report", event_key="standalone-started", status="started",
    )
    registry.record_trial_attempt_event(
        "standalone-before-wfo-report", event_key="standalone-failed", status="failed",
    )
    report = run_wfo(config, train_fn=train_candidate, backtest_fn=backtest_candidate, resume=True)
    _verify_file(output / "wfo-report.json")
    report = run_wfo(config, train_fn=train_candidate, backtest_fn=backtest_candidate, resume=True)
    assert backtest_calls == ["test", "test", "holdout"]
    assert len(train_calls) == 6 and train_calls[-1] == "final-retrain-resumed"
    assert len(report["fold_tests"]) == 2
    assert report["conclusion"] == "PASS"
    trial_evidence = report["deflated_sharpe"]["trial_count_evidence"]
    assert trial_evidence["dsr_effective_trials"] == 2
    assert trial_evidence["project_trial_attempt_inventory"][
        "protocol_registered_attempt_count"] == 2
    project_attempts = {
        row["protocol_id"]: row
        for row in trial_evidence["project_trial_attempt_inventory"]["project_protocol_inventory"]
    }
    assert project_attempts["standalone-application-training-v1"][
        "lifecycle_event_counts"]["failed"] == 1
    assert trial_evidence["project_historical_effective_trials"]["status"] == "not_estimated"
    sensitivity = trial_evidence["current_protocol_sensitivity"]
    assert sensitivity["effective_trial_count_range"] == [1, 2]
    assert sensitivity["correlation_estimated"] is False
    assert sensitivity["scenarios"][1]["dsr_probability"] == report["deflated_sharpe"]["dsr_probability"]
    attempt_statuses = {
        attempt["candidate_id"]: next(
            event["status"] for event in attempt["events"]
            if event["event_key"] == "validation-selection"
        )
        for attempt in trial_evidence["project_trial_attempt_inventory"]["attempts"]
    }
    assert attempt_statuses == {
        candidate_id: "selected" if candidate_id == report["selection"]["candidate_id"]
        else "not_selected"
        for candidate_id in ("candidate-a", "candidate-b")
    }
    ledger = ResearchLedger(output / "research.sqlite")
    assert ledger.state("resumable-test-wfo")["state"] == "CLOSED"
    assert ledger.query_counts("resumable-test-wfo") == {"validation": 4, "test": 2}
    assert ledger.verify_audit_chain()


def test_wfo_rejects_l2_boundary_hash_and_state_reset_drift() -> None:
    research, l2, dataset_hash, feature_hash = _paired_folds()
    manifest = asdict(l2[0])
    manifest["validation"]["score_start_ns"] += 1
    with pytest.raises(FoldBoundaryError, match="validation_start_ns"):
        validate_l2_manifest(research[0], manifest)

    manifest = asdict(l2[0])
    manifest["reset_orders"] = False
    with pytest.raises(FoldBoundaryError, match="isolation flags"):
        validate_l2_manifest(research[0], manifest)

    with pytest.raises(FoldBoundaryError, match="dataset_hash"):
        validate_l2_manifest(
            research[0], l2[0], expected_dataset_hash="c" * 64,
            expected_feature_hash=feature_hash,
        )


@pytest.mark.parametrize(("path", "value", "message"), [
    (("schema_version",), "0.4.1", "schema version"),
    (("train", "warmup_start_ns"), 1, "warmup_start_ns"),
    (("dataset_hash",), "not-a-sha256", "dataset_hash"),
    (("test", "score_end_ns"), True, "malformed L2 test window"),
])
def test_wfo_rejects_malformed_nested_l2_manifest(path, value, message: str) -> None:
    research, l2, _dataset_hash, _feature_hash = _paired_folds()
    manifest = asdict(l2[0])
    cursor = manifest
    for key in path[:-1]:
        cursor = cursor[key]
    cursor[path[-1]] = value
    with pytest.raises(FoldBoundaryError, match=message):
        validate_l2_manifest(research[0], manifest)


@pytest.mark.parametrize("field,value", [
    ("train_start_ns", True),
    ("purge_ns", -1),
    ("terminal_policy", "mark_to_market"),
])
def test_fold_request_rejects_invalid_boundary_contract(field: str, value: object) -> None:
    research, _l2, _dataset_hash, _feature_hash = _paired_folds()
    request = research[0]
    values = {name: getattr(request, name) for name in request.__dataclass_fields__}
    values[field] = value
    with pytest.raises(FoldBoundaryError):
        FoldRequest(**values)
