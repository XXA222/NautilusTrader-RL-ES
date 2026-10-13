from __future__ import annotations

from hashlib import sha256

import pytest

from rles_orchestrator.workflow import (
    Orchestrator, Route, RunSpec, STAGES, StageArtifact, StageBudget,
    StageContext, StageUsage, lineage_hash, transfer_evidence_hash,
)
from rles_orchestrator.transfer import ParityReport, ParameterLoadEvidence, TransferResult
from rles_research.ledger import ResearchLedger, SelectionLedger

pytestmark = pytest.mark.unit

_HASHES = {name: sha256(name.encode()).hexdigest() for name in (
    "data", "feature", "config", "hard", "execution", "normalization",
    "soft-schema", "soft-initial", "actor", "actor-payload", "transfer-obs",
)}


def _transfer(algorithm: str) -> TransferResult:
    parity = ParityReport(
        algorithm, _HASHES["actor"], _HASHES["transfer-obs"], 4,
        0.0, 0.0, 0.0, 0.0, 0.0, True,
    )
    loaded = ParameterLoadEvidence(
        _HASHES["actor"], 4, 4, (), (), True,
    )
    return TransferResult(algorithm, "DIRECT", parity, loaded, None, "train_inner")


def _spec(route: Route) -> RunSpec:
    budgets = {}
    for stage in STAGES[route]:
        budgets[stage] = (StageBudget(4, 2, 8) if stage in {"E1", "E2"}
                          else StageBudget(max_rl_steps=12))
    return RunSpec(
        run_id=f"run-{route.value}", attempt_id=f"attempt-{route.value}", route=route,
        algorithm="PPO", seed=17,
        data_hash=_HASHES["data"], feature_hash=_HASHES["feature"],
        config_hash=_HASHES["config"], hard_risk_hash=_HASHES["hard"],
        execution_hash=_HASHES["execution"], normalization_hash=_HASHES["normalization"],
        soft_risk_schema_hash=_HASHES["soft-schema"],
        initial_soft_risk_hash=_HASHES["soft-initial"], actor_signature=_HASHES["actor"],
        stage_budget=budgets,
    )


@pytest.mark.parametrize("route", list(Route))
def test_a0_a4_routes_preserve_stage_lineage_budgets_and_freezes(route: Route) -> None:
    spec = _spec(route)
    stages_seen: list[str] = []
    receipts: list[str] = []

    def run_stage(context: StageContext) -> StageArtifact:
        stage = context.stage
        stages_seen.append(stage)
        parent = context.previous
        transfer = _transfer(spec.algorithm) if stage == "R1" and parent and parent.stage == "E1" else None
        actor_hash = parent.actor_hash if parent else _HASHES["actor-payload"]
        soft_hash = sha256(b"e2-soft-risk-update").hexdigest() if stage == "E2" else spec.initial_soft_risk_hash
        metadata: dict[str, str] = {}
        if stage == "E1":
            metadata["actor_signature"] = spec.actor_signature
        elif stage == "R1":
            metadata["rl_training_complete"] = "true"
            if transfer is not None:
                metadata["transfer_report_hash"] = transfer_evidence_hash(transfer)
        elif stage == "E2":
            metadata["soft_risk_bounds_gate"] = "PASS"
            metadata["soft_risk_schema_hash"] = spec.soft_risk_schema_hash
        budget = context.budget
        usage = (StageUsage(budget.max_population, budget.max_generations, budget.max_evaluations)
                 if stage in {"E1", "E2"}
                 else StageUsage(rl_steps=budget.max_rl_steps))
        receipt = sha256(f"receipt:{stage}:{route.value}".encode()).hexdigest()
        return StageArtifact(
            stage, f"artifact-{route.value}-{stage}",
            sha256(f"payload:{route.value}:{stage}".encode()).hexdigest(),
            spec.fingerprint, parent.artifact_id if parent else None,
            actor_hash, soft_hash, spec.hard_risk_hash, spec.data_hash,
            spec.execution_hash, spec.feature_hash, spec.normalization_hash,
            usage, receipt, metadata, transfer,
        )

    runners = {stage: run_stage for stage in {"E1", "R1", "E2"}}

    def attest(_spec: RunSpec, artifact: StageArtifact) -> str:
        receipts.append(artifact.ledger_receipt_hash)
        return artifact.ledger_receipt_hash

    orchestrator = Orchestrator(runners, attest_usage=attest)
    lineage = orchestrator.run(spec)
    assert stages_seen == list(STAGES[route])
    assert tuple(artifact.stage for artifact in lineage) == STAGES[route]
    assert orchestrator.state == "CLOSED"
    assert len(receipts) == len(STAGES[route])
    assert lineage_hash(lineage)


def test_orchestrator_fails_closed_when_e2_changes_frozen_actor() -> None:
    spec = _spec(Route.A2)

    def run_stage(context: StageContext) -> StageArtifact:
        parent = context.previous
        assert parent is not None or context.stage == "R1"
        actor_hash = sha256(b"mutated-actor").hexdigest() if context.stage == "E2" else _HASHES["actor-payload"]
        usage = (StageUsage(context.budget.max_population, context.budget.max_generations,
                            context.budget.max_evaluations) if context.stage == "E2"
                 else StageUsage(rl_steps=context.budget.max_rl_steps))
        metadata = ({"rl_training_complete": "true"} if context.stage == "R1" else {
            "soft_risk_bounds_gate": "PASS", "soft_risk_schema_hash": spec.soft_risk_schema_hash,
        })
        return StageArtifact(
            context.stage, f"bad-{context.stage}", sha256(context.stage.encode()).hexdigest(),
            spec.fingerprint, parent.artifact_id if parent else None, actor_hash,
            spec.initial_soft_risk_hash, spec.hard_risk_hash, spec.data_hash,
            spec.execution_hash, spec.feature_hash, spec.normalization_hash,
            usage, sha256(f"receipt:{context.stage}".encode()).hexdigest(), metadata,
        )

    orchestrator = Orchestrator(
        {stage: run_stage for stage in ("R1", "E2")},
        attest_usage=lambda _spec, artifact: artifact.ledger_receipt_hash,
    )
    with pytest.raises(RuntimeError, match="modified frozen R1 actor"):
        orchestrator.run(spec)
    assert orchestrator.state == "FAILED"


def test_a4_orchestrator_persists_stage_usage_through_l6_selection_ledger(tmp_path) -> None:
    from rles_es_evotorch.provider import EvolutionResult
    from rles_orchestrator.workflow import ProviderBackedStageRunner, ProviderStageOutput
    from rles_rl_sb3.provider import Performance

    spec = _spec(Route.A4)
    protocol_id = "research-a4"
    ledger = ResearchLedger(tmp_path / "research.sqlite")
    ledger.create(protocol_id, spec.run_id, validation_budget=1,
                  hpo_budget=8, es_evaluation_budget=16)
    selection = SelectionLedger(ledger, protocol_id)

    def execute_stage(context: StageContext) -> ProviderStageOutput:
        stage, parent = context.stage, context.previous
        transfer = _transfer(spec.algorithm) if stage == "R1" else None
        actor_hash = _HASHES["actor-payload"] if parent is None else parent.actor_hash
        soft_hash = sha256(b"a4-e2-soft-risk").hexdigest() if stage == "E2" else spec.initial_soft_risk_hash
        metadata: dict[str, str] = {}
        if stage == "E1":
            metadata["actor_signature"] = spec.actor_signature
        elif stage == "R1":
            metadata["rl_training_complete"] = "true"
            metadata["transfer_report_hash"] = transfer_evidence_hash(transfer)
        else:
            metadata.update({"soft_risk_bounds_gate": "PASS",
                             "soft_risk_schema_hash": spec.soft_risk_schema_hash})
        artifact_id = f"ledger-{stage}"
        payload_hash = sha256(f"payload-{stage}".encode()).hexdigest()
        if stage == "R1":
            return ProviderStageOutput(
                artifact_id, payload_hash, actor_hash, soft_hash, metadata,
                rl_performances=(Performance(12, 1.0, None, None, 12.0, None, None),),
                transfer_evidence=transfer,
            )
        evolution = EvolutionResult(
            "pgpe" if stage == "E1" else "cmaes", context.budget.max_generations,
            context.budget.max_evaluations, 0.25, (0.0, 0.1, 0.2), () if stage == "E2" else None,
        )
        return ProviderStageOutput(
            artifact_id, payload_hash, actor_hash, soft_hash, metadata,
            evolution_result=evolution, population_size=context.budget.max_population,
            transfer_evidence=transfer,
        )

    orchestrator = Orchestrator(
        {stage: ProviderBackedStageRunner(stage, execute_stage, selection)
         for stage in ("E1", "R1", "E2")},
        attest_usage=selection,
    )
    lineage = orchestrator.run(spec)

    summary = ledger.evidence_summary(protocol_id)
    assert [stage["stage"] for stage in summary["stage_receipts"]] == ["E1", "E2", "R1"]
    assert len({stage["receipt_hash"] for stage in summary["stage_receipts"]}) == 3
    assert ledger.verify_audit_chain()
    assert len(lineage) == 3 and orchestrator.state == "CLOSED"


def test_provider_stage_runner_rejects_wrong_es_searcher_for_stage() -> None:
    from types import SimpleNamespace
    from rles_es_evotorch.provider import EvolutionResult
    from rles_orchestrator.workflow import ProviderBackedStageRunner, ProviderStageOutput

    spec = _spec(Route.A1)
    output = ProviderStageOutput(
        "wrong-searcher", sha256(b"payload").hexdigest(), _HASHES["actor-payload"],
        spec.initial_soft_risk_hash, {},
        evolution_result=EvolutionResult("cmaes", 1, 4, 0.0, (0.0,), ()),
        population_size=4,
    )
    receipt_provider = SimpleNamespace(receipt_hash_for=lambda *_a, **_k: "0" * 64)
    runner = ProviderBackedStageRunner("E1", lambda _context: output, receipt_provider)
    context = StageContext(spec, "E1", spec.stage_budget["E1"], None)
    with pytest.raises(RuntimeError, match="E1 requires the frozen pgpe searcher"):
        runner(context)


def test_provider_stage_output_binds_lane_usage_to_evidence() -> None:
    from rles_es_evotorch.provider import EvolutionResult
    from rles_orchestrator.workflow import ProviderStageOutput

    evolution = EvolutionResult("pgpe", 1, 4, 0.0, (0.0,), ())
    base = ("lane-output", sha256(b"payload").hexdigest(), _HASHES["actor-payload"],
            _HASHES["soft-initial"])
    with pytest.raises(ValueError, match="provided together"):
        ProviderStageOutput(*base, {}, evolution_result=evolution,
                            population_size=4, lane_requests=8)
    with pytest.raises(ValueError, match="provided together"):
        ProviderStageOutput(*base, {"lane_evidence_hash": sha256(b"lane").hexdigest()},
                            evolution_result=evolution, population_size=4)
    with pytest.raises(ValueError, match="lowercase SHA256"):
        ProviderStageOutput(*base, {"lane_evidence_hash": "G" * 64},
                            evolution_result=evolution, population_size=4,
                            lane_requests=8)

    output = ProviderStageOutput(
        *base, {"lane_evidence_hash": sha256(b"lane-evidence").hexdigest()},
        evolution_result=evolution, population_size=4, lane_requests=8,
    )
    assert output.lane_requests == 8
