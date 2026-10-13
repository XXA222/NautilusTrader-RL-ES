from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import rles_orchestrator.application as application
from rles_orchestrator.workflow import Route


def test_config_loader_accepts_es_scale_and_rejects_other_unknown_keys(tmp_path):
    from rles_data.core import DataError

    path = tmp_path / "config.json"
    config = {"environment": {"market_path": "market.parquet", "rules_path": "rules.json"},
              "output": "run", "es_input_scaling": "account_units",
              "enforce_worker_memory_limit": True, "worker_ram_bytes": 2147483648}
    path.write_text(json.dumps(config))
    assert application.load_config(path)["es_input_scaling"] == "account_units"
    assert application.load_config(path)["worker_ram_bytes"] == 2147483648
    path.write_text(json.dumps({**config, "unknown_policy_setting": True}))
    with pytest.raises(DataError, match="unknown application configuration"):
        application.load_config(path)


@pytest.mark.parametrize("settings", [
    {"enforce_worker_memory_limit": "true"}, {"worker_ram_bytes": True},
    {"worker_ram_bytes": 0}, {"worker_ram_bytes": 4 * 1024**3 + 1},
])
def test_invalid_worker_limits_fail_before_data_and_allocation(tmp_path, monkeypatch, settings):
    from rles_data.core import DataError

    monkeypatch.setattr(application, "prepare", lambda _config: pytest.fail("invalid limits reached data preparation"))
    output = tmp_path / "must-not-create"
    with pytest.raises(DataError, match="worker memory policy"):
        application.train({"route": "A1", "output": str(output), **settings})
    assert not output.exists()


pytestmark = pytest.mark.unit


@pytest.mark.parametrize("plan_metadata", (False, True))
@pytest.mark.parametrize("collect_observations", (False, True))
def test_rollout_reports_requested_and_post_guard_action_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
    plan_metadata: bool,
    collect_observations: bool,
) -> None:
    requested = iter((0.8, -0.2, 0.5))
    executed = (0.4, -0.2, 0.0)

    class Environment:
        index = 0

        def reset(self, *, seed: int) -> tuple[np.ndarray, dict]:
            assert seed == 7
            return np.zeros(2, dtype=np.float32), {}

        def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
            expected = float(action[0])
            assert float(action[0]) == pytest.approx(expected)
            post_guard = executed[self.index]
            self.index += 1
            info = {
                "requested_action": expected,
                "policy_executed_action_pre_terminal": post_guard,
                "risk_guard_intervention": abs(expected - post_guard),
                "risk_soft_adjustment": abs(expected - post_guard),
                "risk_hard_adjustment": 0.0,
                "risk_intervention_components_known": True,
                "risk_soft_target": expected * 0.5,
                "risk_adjusted_target": expected * 0.5,
            }
            if plan_metadata:
                info["execution_plan_action"] = ("SUBMIT", "NOOP", "NOOP")[self.index - 1]
                info["execution_plan_reason"] = ("rebalance", "deadband", "below minimum executable quantity/notional")[self.index - 1]
            return np.full(2, self.index, dtype=np.float32), 0.0, False, self.index == 3, info

        def close(self) -> None:
            return None

    class Factory:
        mode = "base"
        snapshots = ()
        fill_references = ()
        execution = SimpleNamespace(fill=lambda _mode: SimpleNamespace(data_granularity="bar"))

        @staticmethod
        def make_env(*, soft_multipliers: dict | None) -> Environment:
            assert soft_multipliers is None
            return Environment()

    monkeypatch.setattr(application, "native_report", lambda *_args, **_kwargs: {"metrics": {}})

    report, observations = application.rollout(
        Factory(), lambda _batch: np.asarray([[next(requested)]], dtype=np.float32), seed=7,
        collect_observations=collect_observations,
    )

    diagnostics = report["action_diagnostics"]
    assert observations.shape == ((4, 2) if collect_observations else (0, 2))
    assert diagnostics["count"] == 3
    assert diagnostics["requested_abs_mean"] == pytest.approx(0.5)
    assert diagnostics["requested_nonzero_fraction"] == 1.0
    assert diagnostics["executed_abs_mean"] == pytest.approx(0.2)
    assert diagnostics["executed_nonzero_fraction"] == pytest.approx(2 / 3)
    assert diagnostics["requested_executed_gap_abs_mean"] == pytest.approx(0.3)
    assert diagnostics["adjusted_fraction"] == pytest.approx(2 / 3)
    assert diagnostics["requested_short_fraction"] == pytest.approx(1 / 3)
    assert diagnostics["requested_long_fraction"] == pytest.approx(2 / 3)
    assert diagnostics["requested_target_change_abs_mean"] == pytest.approx(0.85)
    assert diagnostics["requested_direction_comparison_pair_count"] == 2
    assert diagnostics["requested_direction_flip_count"] == 2
    assert diagnostics["requested_direction_flip_fraction"] == pytest.approx(1.0)
    assert diagnostics["executed_exposure_change_abs_mean"] == pytest.approx(0.4)
    assert diagnostics["executed_direction_comparison_pair_count"] == 1
    assert diagnostics["executed_direction_flip_count"] == 1
    assert diagnostics["executed_direction_flip_fraction"] == pytest.approx(1.0)
    attribution = report["execution_attribution"]
    assert attribution["components_known"] is plan_metadata
    assert attribution["missing_plan_count"] == (0 if plan_metadata else 3)
    if plan_metadata:
        groups = {group["reason"]: group for group in attribution["groups"]}
        assert groups["deadband"]["requested_mean"] == pytest.approx(-0.2)
        assert groups["deadband"]["requested_short_fraction"] == 1
        assert groups["deadband"]["risk_targets"]["risk_target_abs_mean"] == pytest.approx(0.1)
        assert groups["deadband"]["executed_abs_mean"] == pytest.approx(0.2)
        minimum = groups["below minimum executable quantity/notional"]
        assert minimum["executed_nonzero_fraction"] == 0
        assert minimum["risk_targets"]["risk_to_executed_gap_abs_mean"] == pytest.approx(0.25)
        assert groups["rebalance"]["action"] == "SUBMIT"
    else:
        assert attribution["groups"] == []
    assert report["risk_target_diagnostics"] == {
        "components_known": True,
        "soft_target_abs_mean": pytest.approx(0.25),
        "risk_adjusted_target_abs_mean": pytest.approx(0.25),
        "requested_to_soft_gap_abs_mean": pytest.approx(0.25),
        "soft_to_risk_gap_abs_mean": pytest.approx(0.0),
        "risk_to_executed_gap_abs_mean": pytest.approx(0.1166666667),
    }
    assert report["execution_plan_diagnostics"] == {
        "components_known": plan_metadata,
        "count": 3,
        "missing_count": 0 if plan_metadata else 3,
        "action_counts": {"NOOP": 2, "SUBMIT": 1} if plan_metadata else {},
        "reason_counts": {
            "rebalance": 1, "deadband": 1, "below minimum executable quantity/notional": 1,
        } if plan_metadata else {},
    }


def test_application_resolves_explicit_cuda_for_routes_with_rl(monkeypatch: pytest.MonkeyPatch) -> None:
    requested: list[str] = []

    class Resolver:
        def resolve(self, value: str) -> SimpleNamespace:
            requested.append(value)
            return SimpleNamespace(learner_device="cuda")

    monkeypatch.setattr(application, "DeviceResolver", Resolver)

    assert application._resolve_rl_device("cuda", Route.A0) == "cuda"
    assert requested == ["gpu"]


def test_application_rejects_cuda_for_es_only_route_before_resolution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(application, "DeviceResolver", lambda: pytest.fail("must reject before probing"))

    with pytest.raises(application.DataError, match="has no R1 learner"):
        application._resolve_rl_device("cuda", Route.A1)


def test_application_rejects_unavailable_cuda_before_building_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Resolver:
        def resolve(self, _value: str) -> SimpleNamespace:
            raise application.DeviceUnavailable("CUDA not available: refusing CPU fallback")

    monkeypatch.setattr(application, "DeviceResolver", Resolver)
    monkeypatch.setattr(application, "prepare", lambda _config: pytest.fail("environment must not be built"))
    output = tmp_path / "must-not-be-created"

    with pytest.raises(application.DataError, match="requested R1 device is unavailable"):
        application.train({"route": "A0", "rl_device": "cuda", "output": str(output)})

    assert not output.exists()


def test_backtest_allows_only_explicit_cost_stress_mode_over_base_bundle(tmp_path, monkeypatch):
    from rles_data.core import DataQualityReport, Grade
    from rles_gym_bridge.observation import ObservationSchema
    from rles_orchestrator.application import _write_verified

    schema = ObservationSchema("0.4.0", (
        "account.equity", "account.peak_equity", "account.drawdown",
        "risk.guard_intervention", "execution.prev_executed_target",
    ), ("USDT", "USDT", "ratio", "ratio", "ratio"))
    quality = DataQualityReport(Grade.B, (), 0, {}, "d" * 64)
    factory = SimpleNamespace(
        mode="stress", base_config_hash="base-config", config_hash="stress-config",
        store=SimpleNamespace(snapshot_hash="d" * 64, events=()),
        features=SimpleNamespace(hash="f" * 64), schema=schema,
        scaler=SimpleNamespace(hash="s" * 64), hard=SimpleNamespace(signature=lambda: "h" * 64),
        execution=SimpleNamespace(
            fingerprint=lambda: "e" * 64,
            latency=lambda mode: SimpleNamespace(model_id=f"{mode}-latency"),
            fill=lambda mode: SimpleNamespace(model_id=f"{mode}-fill"),
        ), quality=quality,
    )
    bundle_root = tmp_path / "bundle"
    actor_root = bundle_root / "actor"
    actor_root.mkdir(parents=True)
    (actor_root / "COMPLETED").write_text("actor-hash", encoding="ascii")
    identities = {
        "dataset": "d" * 64, "features": "f" * 64, "schema": schema.hash,
        "scaler": "s" * 64, "execution": "e" * 64, "hard_risk": "h" * 64,
        "config": "base-config",
    }
    _write_verified(bundle_root / "bundle.json", {
        "preprocessing": {"identities": identities, "schema": {
            "schema_version": schema.schema_version, "names": schema.names, "units": schema.units,
        }, "scaler": {"mean": [], "std": []}},
        "actor": {"kind": "rl", "path": "actor", "actor_hash": "actor-hash"},
        "soft_multipliers": None,
    })
    monkeypatch.setattr(application, "InferenceActor",
                        lambda *_args, **_kwargs: SimpleNamespace(predict=lambda _batch: np.zeros((1, 1))))
    monkeypatch.setattr(application, "rollout", lambda _factory, *_args, **_kwargs: ({"metrics": {}}, np.empty((0, 5))))
    config = {"seed": 1, "environment": {}}

    report = application.backtest(config, bundle_root, factory=factory, allow_execution_stress=True)
    assert report["execution_mode"] == "stress"
    assert report["latency_model_id"] == "stress-latency"
    assert report["fill_model_id"] == "stress-fill"
    with pytest.raises(application.DataError, match="identity differs"):
        application.backtest(config, bundle_root, factory=factory)


@pytest.mark.parametrize("command", ["backtest", "diagnose-train"])
def test_public_backtest_cli_persists_checksummed_report_and_compact_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
    command: str,
) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    bundle_json = bundle / "bundle.json"
    bundle_json.write_text('{"schema":"test-bundle"}', encoding="utf-8")
    bundle_sha = hashlib.sha256(bundle_json.read_bytes()).hexdigest()
    bundle_json.with_suffix(".json.sha256").write_text(bundle_sha, encoding="ascii")
    output = tmp_path / "backtest-output"
    report = {
        "metrics": {"net_return": -0.01, "max_drawdown": 0.02},
        "terminal_flat": True,
        "liquidations": 0,
        "account_rows": [{"equity": "10000"}],
    }
    if command == "diagnose-train":
        report["diagnostic_scope"] = {"split": "train_inner", "evaluation_queries": 0}
    config = {"output": str(output), "evaluation": {"split": "test"}}
    monkeypatch.setattr(application, "load_config", lambda _path: config)
    monkeypatch.setattr(application, "backtest", lambda _config, _bundle: report)
    monkeypatch.setattr(application, "diagnose_train", lambda _config, _bundle: report)

    assert application.main([command, "--config", "unused.json", "--bundle", str(bundle)]) == 0

    saved = output / "report.json"
    expected_report_sha = hashlib.sha256(saved.read_bytes()).hexdigest()
    assert json.loads(saved.read_text(encoding="utf-8")) == report
    assert (output / "report.json.sha256").read_text(encoding="ascii") == expected_report_sha
    completed = json.loads((output / "COMPLETED.json").read_text(encoding="utf-8"))
    assert completed == {
        "schema": "rles.train-diagnostic.v1" if command == "diagnose-train" else "rles.backtest-cli.v1",
        "bundle_sha256": bundle_sha,
        "report_sha256": expected_report_sha,
    }
    summary = json.loads(capsys.readouterr().out)
    assert summary["schema"] == completed["schema"]
    if command == "diagnose-train":
        assert "evaluation" not in summary
        assert summary["diagnostic_scope"] == report["diagnostic_scope"]
    assert summary["output"] == str(output)
    assert summary["metrics"] == report["metrics"]
    assert summary["terminal_flat"] is True


@pytest.mark.parametrize("command", ["backtest", "diagnose-train"])
def test_public_backtest_cli_refuses_to_overwrite_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    command: str,
) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    bundle_json = bundle / "bundle.json"
    bundle_json.write_text("{}", encoding="utf-8")
    bundle_json.with_suffix(".json.sha256").write_text(
        hashlib.sha256(bundle_json.read_bytes()).hexdigest(), encoding="ascii",
    )
    output = tmp_path / "existing-output"
    output.mkdir()
    (output / "keep.txt").write_text("immutable", encoding="utf-8")
    monkeypatch.setattr(application, "load_config", lambda _path: {"output": str(output)})
    called = False

    def should_not_run(_config: dict, _bundle: str) -> dict:
        nonlocal called
        called = True
        raise AssertionError("existing outputs must be rejected before native replay")

    monkeypatch.setattr(application, "backtest", should_not_run)
    monkeypatch.setattr(application, "diagnose_train", should_not_run)

    assert application.main([command, "--config", "unused.json", "--bundle", str(bundle)]) == 2
    assert not called
    assert (output / "keep.txt").read_text(encoding="utf-8") == "immutable"


def test_execution_attribution_preserves_partial_risk_coverage():
    from rles_orchestrator.diagnostics import ExecutionAttribution

    attribution = ExecutionAttribution()
    for soft, risk in ((0.1, 0.05), (None, None)):
        attribution.observe(requested=-0.2, executed=0.0, soft=soft, risk=risk,
                            action="NOOP", reason="deadband")
    report = attribution.report()
    assert report["components_known"] is False
    group = report["groups"][0]
    assert group["count"] == 2 and group["fraction"] == 1
    assert group["requested_mean"] == -0.2
    assert group["risk_targets"]["count"] == 1
    assert group["risk_targets"]["components_known"] is False
    assert group["risk_targets"]["risk_target_abs_mean"] == 0.05
    unknown = ExecutionAttribution()
    unknown.observe(requested=0, executed=0, soft=None, risk=None, action="NOOP", reason="deadband")
    assert unknown.report()["groups"][0]["risk_targets"]["risk_target_abs_mean"] is None
