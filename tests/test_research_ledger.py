from __future__ import annotations

import sqlite3
from types import SimpleNamespace

import pytest

from rles_es_evotorch.evaluation import CandidateRequest, EvaluationBudget
from rles_research.ledger import ResearchLedger, ResearchLockError

pytestmark = pytest.mark.unit


def _sha(character: str) -> str:
    return character * 64


def _freeze_and_retrain(ledger: ResearchLedger, protocol_id: str, cutoff_ns: int) -> None:
    ledger.freeze_recipe(
        protocol_id,
        recipe_sha=_sha("1"),
        code_sha=_sha("2"),
        selection_spec_sha=_sha("3"),
    )
    ledger.register_retrain(protocol_id, _sha("4"), pre_holdout_end_ns=cutoff_ns)


def test_research_ledger_enforces_budgets_test_lock_and_one_shot_holdout(tmp_path) -> None:
    ledger = ResearchLedger(tmp_path / "research.sqlite")
    ledger.create("protocol-1", "run-1", validation_budget=1, hpo_budget=0, es_evaluation_budget=0)
    ledger.record_query("protocol-1", "query-1", "validation", "candidate-a", "fold-1")
    with pytest.raises(ResearchLockError, match="budget exhausted"):
        ledger.record_query("protocol-1", "query-2", "validation", "candidate-b", "fold-1")

    ledger.freeze_recipe(
        "protocol-1", recipe_sha=_sha("1"), code_sha=_sha("2"), selection_spec_sha=_sha("3")
    )
    with pytest.raises(ResearchLockError, match="frozen"):
        ledger.record_query("protocol-1", "query-3", "validation", "candidate-a", "fold-1")
    ledger.record_query("protocol-1", "test-1", "test", "candidate-a", "fold-1")
    with pytest.raises(ResearchLockError, match="one-shot fold Test"):
        ledger.record_query("protocol-1", "test-2", "test", "candidate-a", "fold-1")

    ledger.register_retrain("protocol-1", _sha("4"), pre_holdout_end_ns=900)
    ledger.open_holdout(
        "protocol-1", holdout_dataset_sha=_sha("a"),
        reviewer_approval_sha=_sha("b"), holdout_start_ns=1_000,
    )
    assert ledger.state("protocol-1")["state"] == "HOLDOUT_OPENED"
    ledger.record_holdout_report("protocol-1", _sha("c"))
    ledger.record_holdout_report("protocol-1", _sha("c"))
    with pytest.raises(ResearchLockError, match="differs from its durable"):
        ledger.record_holdout_report("protocol-1", _sha("d"))
    ledger.close_holdout("protocol-1", report_sha=_sha("c"), conclusion="PASS")
    assert ledger.state("protocol-1")["state"] == "CLOSED"

    ledger.create("protocol-2", "run-2", validation_budget=1, hpo_budget=0, es_evaluation_budget=0)
    _freeze_and_retrain(ledger, "protocol-2", cutoff_ns=900)
    with pytest.raises(ResearchLockError, match="already been opened"):
        ledger.open_holdout(
            "protocol-2", holdout_dataset_sha=_sha("a"),
            reviewer_approval_sha=_sha("c"), holdout_start_ns=1_000,
        )
    assert ledger.verify_audit_chain()


def test_research_ledger_duplicate_protocol_id_uses_domain_error(tmp_path) -> None:
    ledger = ResearchLedger(tmp_path / "research.sqlite")
    ledger.create("protocol-1", "run-1", validation_budget=1, hpo_budget=0, es_evaluation_budget=0)
    with pytest.raises(ResearchLockError, match="protocol ID already exists"):
        ledger.create("protocol-1", "run-2", validation_budget=1, hpo_budget=0, es_evaluation_budget=0)


def test_aborted_protocol_is_audited_and_cannot_continue(tmp_path) -> None:
    ledger = ResearchLedger(tmp_path / "research.sqlite")
    ledger.create("protocol-abort", "run-abort", validation_budget=2, hpo_budget=0,
                  es_evaluation_budget=0)
    ledger.record_query("protocol-abort", "validation-1", "validation", "ppo", "fold-0")
    ledger.abort("protocol-abort", error_type="DataError", detail="no feasible validation candidate")

    assert ledger.state("protocol-abort")["state"] == "ABORTED"
    assert ledger.query_counts("protocol-abort") == {"validation": 1}
    with pytest.raises(ResearchLockError, match="protocol frozen"):
        ledger.record_query("protocol-abort", "validation-2", "validation", "sac", "fold-0")
    with pytest.raises(ResearchLockError, match="terminal"):
        ledger.abort("protocol-abort", error_type="DataError", detail="duplicate abort")
    assert ledger.verify_audit_chain()


def test_evidence_summary_reports_observed_trial_inventory_without_estimating_history(tmp_path) -> None:
    ledger = ResearchLedger(tmp_path / "research.sqlite")
    ledger.create("protocol-trials", "run-trials", validation_budget=4,
                  hpo_budget=0, es_evaluation_budget=0)
    ledger.record_query("protocol-trials", "v-a-0", "validation", "candidate-a", "fold-0")
    ledger.record_query("protocol-trials", "v-b-0", "validation", "candidate-b", "fold-0")
    summary = ledger.evidence_summary("protocol-trials")
    inventory = summary["trial_count_inventory"]
    assert inventory["scope"] == "current_protocol_only"
    assert inventory["observed_candidate_ids"] == ["candidate-a", "candidate-b"]
    assert inventory["observed_fold_ids"] == ["fold-0"]
    assert inventory["raw_query_counts"]["validation"] == 2
    assert inventory["project_historical_effective_trials"] == "not_estimated"
    assert "manual selections" in " ".join(inventory["limitations"])


def test_research_ledger_audit_chain_detects_database_tampering(tmp_path) -> None:
    path = tmp_path / "research.sqlite"
    ledger = ResearchLedger(path)
    ledger.create("protocol-1", "run-1", validation_budget=1, hpo_budget=0, es_evaluation_budget=0)
    with sqlite3.connect(path) as connection:
        connection.execute("UPDATE audit SET payload_sha=? WHERE seq=1", (_sha("f"),))
    with pytest.raises(ResearchLockError, match="audit journal chain broken"):
        ledger.verify_audit_chain()


def test_es_lane_reservations_charge_failed_attempts_against_protocol_budget(tmp_path) -> None:
    ledger = ResearchLedger(tmp_path / "research.sqlite")
    ledger.create("protocol-es", "run-es", validation_budget=1,
                  hpo_budget=0, es_evaluation_budget=2)

    def spec(attempt: str):
        return SimpleNamespace(
            run_id="run-es", attempt_id=attempt,
            stage_budget={"E1": SimpleNamespace(max_lane_requests=2)},
        )

    def request(attempt: str, candidate: str):
        return CandidateRequest(
            run_id="run-es", attempt_id=attempt, generation=0,
            candidate_id=candidate, episode_id="episode-0", seed=17,
            genome=(0.1,), actor_signature=_sha("a"), candidate_kind="actor",
            candidate_schema_hash=_sha("b"),
            account_isolation_id=f"account-{attempt}-{candidate}",
        )

    # These reservations stand even though this attempt never reaches a stage receipt.
    failed_attempt=spec("attempt-failed")
    budget=EvaluationBudget(
        2, reserve_requests=lambda batch: ledger.reserve_es_lane_requests(
            "protocol-es", failed_attempt, stage="E1", requests=batch,
        ),
        verify_requests=lambda keys: ledger.verify_es_lane_reservations(
            "protocol-es","run-es","attempt-failed","E1",keys,
        ),
        reservation_id=_sha("c"),
    )
    assert budget.reserve((request("attempt-failed","candidate-0"),
                           request("attempt-failed","candidate-1"))) == 2
    assert budget.consumed == 2
    restored=EvaluationBudget(
        2, reserve_requests=lambda batch: ledger.reserve_es_lane_requests(
            "protocol-es", failed_attempt, stage="E1", requests=batch,
        ),
        verify_requests=lambda keys: ledger.verify_es_lane_reservations(
            "protocol-es","run-es","attempt-failed","E1",keys,
        ),
        reservation_id=_sha("c"),
    )
    restored.restore_checkpoint_state(budget.checkpoint_state())
    assert restored.consumed == 2
    empty_ledger=ResearchLedger(tmp_path / "empty.sqlite")
    empty_ledger.create("protocol-es", "run-es", validation_budget=1,
                        hpo_budget=0, es_evaluation_budget=2)
    missing_reservations=EvaluationBudget(
        2, reserve_requests=lambda batch: empty_ledger.reserve_es_lane_requests(
            "protocol-es", failed_attempt, stage="E1", requests=batch,
        ),
        verify_requests=lambda keys: empty_ledger.verify_es_lane_reservations(
            "protocol-es","run-es","attempt-failed","E1",keys,
        ),
        reservation_id=_sha("c"),
    )
    with pytest.raises(ResearchLockError, match="differs from durable"):
        missing_reservations.restore_checkpoint_state(budget.checkpoint_state())
    with pytest.raises(ValueError, match="schema mismatch"):
        EvaluationBudget(2).restore_checkpoint_state(budget.checkpoint_state())
    wrong_scope=EvaluationBudget(
        2, reserve_requests=lambda batch: ledger.reserve_es_lane_requests(
            "protocol-es", failed_attempt, stage="E1", requests=batch,
        ),
        verify_requests=lambda keys: ledger.verify_es_lane_reservations(
            "protocol-es","run-es","attempt-failed","E1",keys,
        ),
        reservation_id=_sha("d"),
    )
    with pytest.raises(ValueError, match="schema mismatch"):
        wrong_scope.restore_checkpoint_state(budget.checkpoint_state())
    assert ledger.evidence_summary("protocol-es")["es_lane_request_count"] == 2
    stored_keys=budget.checkpoint_state()["dispatched"]
    with pytest.raises(ResearchLockError, match="differs from durable"):
        ledger.verify_es_lane_reservations(
            "protocol-es","run-es","attempt-failed","E1",stored_keys[:-1],
        )
    with pytest.raises(ResearchLockError, match="evaluation budget exhausted"):
        retry_spec=spec("attempt-retry")
        retry_budget=EvaluationBudget(
            2, reserve_requests=lambda batch: ledger.reserve_es_lane_requests(
                "protocol-es", retry_spec, stage="E1", requests=batch,
            ),
            verify_requests=lambda keys: ledger.verify_es_lane_reservations(
                "protocol-es","run-es","attempt-retry","E1",keys,
            ),
            reservation_id=_sha("d"),
        )
        retry_budget.reserve((request("attempt-retry","candidate-0"),
                              request("attempt-retry","candidate-1")))
    with pytest.raises(ResearchLockError, match="budget exhausted"):
        ledger.record_query("protocol-es", "query-extra", "es_evaluation",
                            "manual-candidate", "fold-0")
    assert ledger.verify_audit_chain()
