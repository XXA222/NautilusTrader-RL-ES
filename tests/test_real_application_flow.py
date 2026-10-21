"""End-to-end application routes on a checksum-verified Binance archive day."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest


pytestmark = pytest.mark.real_golden

FIXTURES = Path(__file__).parent / "fixtures"
_DAY_NS = 86_400_000_000_000
_BASE_DAY_NS = 1_704_067_200_000_000_000


def _write_route_config(tmp_path: Path, route: str, day_offset: int,
                        algorithm: str = "PPO") -> tuple[Path, Path, int]:
    project = Path(__file__).resolve().parents[1]
    fixture = FIXTURES / f"real_ethusdt_202401{day_offset + 1:02d}"
    day_start_ns = _BASE_DAY_NS + day_offset * _DAY_NS
    config = json.loads(
        (project / "configs" / "ethusdt_20240101_real.json").read_text(encoding="utf-8")
    )
    environment = config["environment"]
    environment["market_path"] = str(fixture / "market.parquet")
    # The normalized economic assumptions are intentionally shared and kept
    # scenario-only; the daily fixtures provide authentic market events.
    environment["rules_path"] = str(FIXTURES / "real_ethusdt_20240101" / "scenario_rules.json")
    environment["start_ns"] = day_start_ns + 5 * 60_000_000_000  # 00:05 UTC
    environment["end_ns"] = day_start_ns + 8 * 3_600_000_000_000 + 5 * 60_000_000_000  # 08:05 UTC
    environment["coverage_policy"]["settlement_times_ns"] = {
        "ETHUSDT-PERP.SIM": [day_start_ns + 8 * 3_600_000_000_000]
    }
    # Binance's archived settlement timestamp can differ from the nominal
    # 8-hour schedule by a few milliseconds; retain the event and audit it
    # under the same bounded tolerance already used by the range evidence.
    environment["coverage_policy"]["settlement_timestamp_tolerance_ns"] = 4_000_000

    validation = json.loads(json.dumps(environment["coverage_policy"]))
    validation["settlement_times_ns"] = {"ETHUSDT-PERP.SIM": []}
    config["evaluation"] = {
        "start_ns": environment["end_ns"],
        "end_ns": day_start_ns + 12 * 3_600_000_000_000 + 5 * 60_000_000_000,  # 12:05 UTC
        "coverage_policy": validation,
    }
    config.update(
        route=route,
        algorithm=algorithm,
        steps=64,
        seed=31,
        hidden_layers=[16, 16],
        normalize_observation=True,
        hyperparameters=(
            {"n_steps": 32, "batch_size": 32, "n_epochs": 2}
            if algorithm == "PPO" else
            {"learning_starts": 32, "buffer_size": 256, "batch_size": 32,
             "train_freq": 1, "gradient_steps": 1}
        ),
        population=4,
        generations=1,
        output=str(tmp_path / f"{route.lower()}-run"),
    )
    if route in {"A3", "A4"}:
        config["distillation"] = {
            "tolerance": 0.12,
            "epochs": 120,
            "batch_size": 16,
            "learning_rate": 0.01,
        }
    path = tmp_path / f"config-{route.lower()}.json"
    path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    return path, fixture, day_start_ns


def _run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    config_path = Path(args[args.index("--config") + 1])
    # Exercise the real CLI main in a new interpreter while isolating fixture
    # research accounting from the user's project-level history.
    bootstrap = (
        "import sys; from pathlib import Path; "
        "from rles_research.access import ProjectResearchRegistry; "
        "from rles_orchestrator import research_access; "
        "registry_path=Path(sys.argv.pop(1)); "
        "research_access.project_registry=lambda: ProjectResearchRegistry(registry_path); "
        "from rles_orchestrator.application import main; sys.exit(main())"
    )
    return subprocess.run(
        [sys.executable, "-c", bootstrap, str(config_path.parent / "project-research.sqlite"), *args],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )


@pytest.mark.parametrize(
    ("route", "day_offset", "expected_stages", "actor_kind", "algorithm"),
    (
        ("A0", 0, ["R1"], "rl", "PPO"),
        ("A1", 1, ["E1"], "es", "PPO"),
        ("A2", 3, ["R1", "E2"], "rl", "PPO"),
        ("A3", 4, ["E1", "R1"], "rl", "PPO"),
        ("A4", 2, ["E1", "R1", "E2"], "rl", "PPO"),
        ("A0", 5, ["R1"], "rl", "SAC"),
        ("A0", 6, ["R1"], "rl", "TD3"),
        ("A0", 7, ["R1"], "rl", "DDPG"),
    ),
)
def test_real_archive_train_bundle_and_governed_exploratory_backtest(
    tmp_path, route, day_offset, expected_stages, actor_kind, algorithm,
):
    config_path, fixture, day_start_ns = _write_route_config(tmp_path, route, day_offset, algorithm)
    import_manifest = json.loads((fixture / "import.json").read_text(encoding="utf-8"))
    assert import_manifest["schema"] in {
        "rles.binance-archive.v1",
        "rles.binance-archive-range.v1",
    }
    assert 4326 <= import_manifest["event_count"] <= 4328
    assert len(import_manifest["sources"]) == 4
    for source in import_manifest["sources"]:
        assert hashlib.sha256((fixture / source["path"]).read_bytes()).hexdigest() == source["sha256"]

    completed = _run_cli("train", "--config", str(config_path))
    assert completed.returncode == 0, completed.stderr[-4000:]

    output = tmp_path / f"{route.lower()}-run"
    run_manifest = json.loads((output / "run.json").read_text(encoding="utf-8"))
    from rles_research.access import ProjectResearchRegistry

    attempts = ProjectResearchRegistry(tmp_path / "project-research.sqlite").trial_attempt_summary(
        "standalone-application-training-v1",
    )
    assert attempts["protocol_registered_attempt_count"] == 1
    trial = attempts["attempts"][0]
    assert trial["candidate_id"] == run_manifest["run_id"]
    assert len(trial["feature_sha256"]) == 64
    assert [event["status"] for event in trial["events"]] == [
        "registered", "started", "completed",
    ]
    marker = json.loads((output / "COMPLETED.json").read_text(encoding="utf-8"))
    bundle_path = output / "bundle.json"
    report_path = output / "report.json"
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert trial["feature_sha256"] == bundle["preprocessing"]["identities"]["features"]

    assert hashlib.sha256(bundle_path.read_bytes()).hexdigest() == marker["bundle_sha256"]
    assert hashlib.sha256(report_path.read_bytes()).hexdigest() == marker["report_sha256"]
    assert bundle["actor"]["kind"] == actor_kind
    assert marker["algorithm"] == algorithm
    assert [item["stage"] for item in bundle["lineage"]] == expected_stages
    assert report["dataset_quality"]["grade"] == "DQ-B"
    assert report["market_data_provenance"]
    assert report["rule_provenance"] == [
        "scenario-only:fixture-assumptions-not-historical-evidence"
    ]
    assert report["terminal_flat"] is True
    assert report["liquidations_observed"] is True
    assert report["liquidations"] == 0
    assert report["metrics"]["observations"] > 0

    test_config = json.loads(config_path.read_text(encoding="utf-8"))
    test_coverage = json.loads(json.dumps(test_config["environment"]["coverage_policy"]))
    test_coverage["settlement_times_ns"] = {
        "ETHUSDT-PERP.SIM": [day_start_ns + 16 * 3_600_000_000_000]
    }
    test_config["evaluation"] = {
        "start_ns": day_start_ns + 12 * 3_600_000_000_000 + 5 * 60_000_000_000,
        "end_ns": day_start_ns + 20 * 3_600_000_000_000 + 5 * 60_000_000_000,
        "split": "test",
        "coverage_policy": test_coverage,
    }
    test_config["output"] = str(tmp_path / f"{route.lower()}-test-report")
    test_config_path = tmp_path / f"config-{route.lower()}-test.json"
    test_config_path.write_text(json.dumps(test_config, indent=2), encoding="utf-8")

    backtest = _run_cli("backtest", "--config", str(test_config_path), "--bundle", str(output))
    assert backtest.returncode == 2
    assert "requires WFO research access" in backtest.stderr
    assert not Path(test_config["output"]).exists()
    test_config["evaluation"]["split"] = "validation"
    test_config_path.write_text(json.dumps(test_config, indent=2), encoding="utf-8")
    backtest = _run_cli("backtest", "--config", str(test_config_path), "--bundle", str(output))
    assert backtest.returncode == 0, backtest.stderr[-4000:]
    test_output = tmp_path / f"{route.lower()}-test-report"
    test_marker = json.loads((test_output / "COMPLETED.json").read_text(encoding="utf-8"))
    test_report = json.loads((test_output / "report.json").read_text(encoding="utf-8"))
    assert test_marker["bundle_sha256"] == marker["bundle_sha256"]
    assert hashlib.sha256((test_output / "report.json").read_bytes()).hexdigest() == test_marker["report_sha256"]
    assert test_report["terminal_flat"] is True
    assert test_report["evaluation_scope"]["formal"] is False
    assert test_report["liquidations"] == 0
    assert test_report["dataset_quality"]["grade"] == "DQ-B"
    assert "funding_settlement" in test_report["dataset_quality"]["verified_coverage"]
    assert any("fundingRate" in source for source in test_report["market_data_provenance"])


@pytest.mark.parametrize("resume_stage", ("E1", "R1", "E2"))
@pytest.mark.artifact_restore
def test_real_archive_a4_recovers_from_durable_stage_checkpoint(tmp_path, monkeypatch, resume_stage):
    """Resume a real-data Hybrid run from durable E1, R1 and E2 boundaries."""
    from rles_es_evotorch.provider import EvoTorchProvider
    from rles_orchestrator.application import backtest, load_config, train
    from rles_rl_sb3.provider import RLProvider

    config_path, fixture, day_start_ns = _write_route_config(tmp_path, "A4", 2)
    config = load_config(config_path)
    config["generations"] = 2
    config["population"] = 4
    config["es_fitness"] = {"protocol": "constraint_priority_train_inner_v1"}
    config["es_scenario_fold_id"] = "constraint-recovery-test-fold"
    config["checkpoint_interval_steps"] = 32
    config["distillation"] = {"tolerance": 0.12, "epochs": 120, "batch_size": 16,
                              "learning_rate": 0.01}
    import_manifest = json.loads((fixture / "import.json").read_text(encoding="utf-8"))
    for source in import_manifest["sources"]:
        assert hashlib.sha256((fixture / source["path"]).read_bytes()).hexdigest() == source["sha256"]

    output = Path(config["output"])
    if resume_stage in {"E1", "E2"}:
        original_continue = EvoTorchProvider.continue_evolution
        continuation_calls = 0
        algorithm = "pgpe" if resume_stage == "E1" else "cmaes"

        def interrupt_es_after_checkpoint(provider, *args, **kwargs):
            nonlocal continuation_calls
            if provider.config.algorithm == algorithm:
                continuation_calls += 1
                # EvoTorch enters continue_evolution once inside evolve(); the
                # stage persists generation one before its next continuation.
                if continuation_calls == 2:
                    raise RuntimeError(f"injected real-data interruption after {resume_stage} generation one")
            return original_continue(provider, *args, **kwargs)

        monkeypatch.setattr(EvoTorchProvider, "continue_evolution", interrupt_es_after_checkpoint)
        failure = f"real-data interruption after {resume_stage}"
    else:
        original_train = RLProvider.train
        r1_train_calls = 0

        def interrupt_r1_after_checkpoint(provider, total_timesteps, *, resumed=False, callback=None):
            nonlocal r1_train_calls
            r1_train_calls += 1
            if r1_train_calls == 2:
                raise RuntimeError("injected real-data interruption after R1 step 32")
            return original_train(provider, total_timesteps, resumed=resumed, callback=callback)

        monkeypatch.setattr(RLProvider, "train", interrupt_r1_after_checkpoint)
        failure = "real-data interruption after R1 step 32"
    with pytest.raises(RuntimeError, match=failure):
        train(config)

    run_id = json.loads((output / "run.json").read_text(encoding="utf-8"))["run_id"]
    checkpoint = (output / "E1" / "generation-00000001" / run_id if resume_stage == "E1" else
                  output / "R1-checkpoints" / "step-000000000032" if resume_stage == "R1" else
                  output / "E2" / "generation-00000001" / run_id)
    if resume_stage == "R1":
        assert (checkpoint / "COMPLETED").is_file()
    else:
        checkpoint_manifest = json.loads((checkpoint / "manifest.json").read_text(encoding="utf-8"))
        assert checkpoint_manifest["generation"] == 1
        assert (checkpoint / "COMPLETED").is_file()
    assert (output / "FAILED.json").is_file()
    from rles_research.access import ProjectResearchRegistry

    attempts = ProjectResearchRegistry(tmp_path / "project-research.sqlite").trial_attempt_summary(
        "standalone-application-training-v1",
    )
    assert attempts["protocol_registered_attempt_count"] == 1
    assert [event["status"] for event in attempts["attempts"][0]["events"]] == [
        "registered", "started", "failed",
    ]

    if resume_stage in {"E1", "E2"}:
        monkeypatch.setattr(EvoTorchProvider, "continue_evolution", original_continue)
    else:
        monkeypatch.setattr(RLProvider, "train", original_train)
    config["resume_checkpoint"] = str(checkpoint)
    result = train(config)
    assert result["route"] == "A4"
    assert (output / "COMPLETED.json").is_file()
    assert not (output / "FAILED.json").exists()
    assert list(output.glob("FAILED-*.json"))
    from rles_orchestrator.research_access import project_registry

    attempts = project_registry().trial_attempt_summary("standalone-application-training-v1")
    assert attempts["protocol_registered_attempt_count"] == 1
    assert [event["status"] for event in attempts["attempts"][0]["events"]] == [
        "registered", "started", "failed", "resumed", "completed",
    ]
    bundle_path = output / "bundle.json"
    marker = json.loads((output / "COMPLETED.json").read_text(encoding="utf-8"))
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    assert hashlib.sha256(bundle_path.read_bytes()).hexdigest() == marker["bundle_sha256"]
    assert [item["stage"] for item in bundle["lineage"]] == ["E1", "R1", "E2"]
    e1_lineage = next(item for item in bundle["lineage"] if item["stage"] == "E1")
    e2_lineage = next(item for item in bundle["lineage"] if item["stage"] == "E2")
    for stage_lineage in (e1_lineage, e2_lineage):
        assert stage_lineage["metadata"]["fitness_protocol"] == "constraint_priority_train_inner_v1"
        assert stage_lineage["metadata"]["scenario_fold_id"] == "constraint-recovery-test-fold"
        constraint_rows = json.loads(stage_lineage["metadata"]["constraint_evidence_json"])
        assert len(constraint_rows) == 2
        assert all("candidates" in row for row in constraint_rows)
    assert (output / "E1" / "generation-00000002" / run_id / "COMPLETED").is_file()
    assert (output / "R1-checkpoints" / "step-000000000064" / "COMPLETED").is_file()
    assert (output / "E2" / "generation-00000002" / run_id / "COMPLETED").is_file()

    test_config = json.loads(config_path.read_text(encoding="utf-8"))
    test_config["generations"] = 2
    test_config["population"] = 4
    test_coverage = json.loads(json.dumps(test_config["environment"]["coverage_policy"]))
    test_coverage["settlement_times_ns"] = {
        "ETHUSDT-PERP.SIM": [day_start_ns + 16 * 3_600_000_000_000]
    }
    test_config["evaluation"] = {
        "start_ns": day_start_ns + 12 * 3_600_000_000_000 + 5 * 60_000_000_000,
        "end_ns": day_start_ns + 20 * 3_600_000_000_000 + 5 * 60_000_000_000,
        "split": "test",
        "coverage_policy": test_coverage,
    }
    test_config["output"] = str(tmp_path / "a4-real-resume-test-report")
    test_config["evaluation"]["split"] = "validation"
    test_report = backtest(load_config(_write_json_config(tmp_path, test_config)), output)
    assert test_report["dataset_quality"]["grade"] == "DQ-B"
    assert test_report["terminal_flat"] is True
    assert test_report["liquidations"] == 0
    assert "funding_settlement" in test_report["dataset_quality"]["verified_coverage"]


def _write_json_config(tmp_path: Path, config: dict) -> Path:
    path = tmp_path / "config-a4-resume-test.json"
    path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    return path
