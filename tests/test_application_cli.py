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
              "es_scenario_fold_id": "shared-crn-fold",
              "es_candidate_evidence": True,
              "enforce_worker_memory_limit": True, "worker_ram_bytes": 2147483648,
              "rl_num_envs": 4, "rl_episode_steps": 1000,
              "rl_corpus_episodes": 8, "rl_corpus_episode_steps": 1024,
              "rl_gpu_memory_fraction": 0.75,
              "es_fitness": {"risk_intervention_weight": 0.0}}
    path.write_text(json.dumps(config))
    assert application.load_config(path)["es_input_scaling"] == "account_units"
    assert application.load_config(path)["worker_ram_bytes"] == 2147483648
    assert application.load_config(path)["rl_num_envs"] == 4
    assert application.load_config(path)["rl_corpus_episodes"] == 8
    assert application.load_config(path)["rl_gpu_memory_fraction"] == 0.75
    assert application.load_config(path)["es_scenario_fold_id"] == "shared-crn-fold"
    assert application.load_config(path)["es_candidate_evidence"] is True
    assert application._es_fitness_spec(application.load_config(path)).risk_intervention_weight == 0.0
    path.write_text(json.dumps({**config, "es_fitness": {
        "protocol": "constraint_priority_train_inner_v1", "return_weight": 1.0,
        "tail_weight": 0.0, "drawdown_weight": 0.5,
    }}))
    selected_fitness = application._es_fitness_spec(application.load_config(path))
    assert selected_fitness.protocol == "constraint_priority_train_inner_v1"
    assert selected_fitness.tail_weight == 0.0
    feasibility_first = application._es_fitness_spec({"es_fitness": {
        "protocol": "constraint_priority_train_inner_v2",
    }})
    assert feasibility_first.protocol == "constraint_priority_train_inner_v2"
    path.write_text(json.dumps({**config, "es_fitness": {"protocol": "unknown_v9"}}))
    with pytest.raises(DataError, match="es_fitness"):
        application.load_config(path)
    path.write_text(json.dumps({**config, "es_scenario_fold_id": "bad fold id"}))
    with pytest.raises(DataError, match="es_scenario_fold_id"):
        application.load_config(path)
    path.write_text(json.dumps({**config, "es_candidate_evidence": "yes"}))
    with pytest.raises(DataError, match="es_candidate_evidence"):
        application.load_config(path)
    path.write_text(json.dumps({**config, "unknown_policy_setting": True}))
    with pytest.raises(DataError, match="unknown application configuration"):
        application.load_config(path)
    path.write_text(json.dumps({**config, "es_fitness": {"unknown_weight": 1.0}}))
    with pytest.raises(DataError, match="es_fitness"):
        application.load_config(path)
    path.write_text(json.dumps({**config, "train_data_access": {
        "start_ns": 10, "end_ns": 20, "marker_sha256": "a" * 64,
        "label": "train-stage69", "source_mode": "whole_snapshot_isolated_v1",
    }}))
    assert application.load_config(path)["train_data_access"]["source_mode"] == "whole_snapshot_isolated_v1"
    path.write_text(json.dumps({**config, "train_data_access": {
        "start_ns": True, "end_ns": 20, "marker_sha256": "a" * 64,
        "label": "train-stage69",
    }}))
    with pytest.raises(DataError, match="scoped Train data access"):
        application.load_config(path)
    path.write_text(json.dumps({**config, "train_data_access": {
        "start_ns": 10, "end_ns": 20, "marker_sha256": "a" * 64,
        "label": "train-stage69", "source_mode": [],
    }}))
    with pytest.raises(DataError, match="scoped Train data access"):
        application.load_config(path)


def test_train_only_cli_skips_validation_and_is_limited_to_train_command(monkeypatch, capsys):
    config = {"output": "diagnostic-only-output"}
    calls = []
    monkeypatch.setattr(application, "load_config", lambda _path: config)
    monkeypatch.setattr(application, "train",
                        lambda supplied, *, evaluate: calls.append((supplied, evaluate)) or {
                            "training_coverage": {"rl_steps": 8},
                        })

    assert application.main(["train", "--config", "unused.json", "--train-only"]) == 0
    assert calls == [(config, False)]
    assert json.loads(capsys.readouterr().out) == {"training_coverage": {"rl_steps": 8}}
    assert application.main(["backtest", "--config", "unused.json", "--train-only"]) == 2
    assert calls == [(config, False)]
    assert "--train-only is supported only by the train command" in capsys.readouterr().err


def test_scoped_train_access_cli_fails_closed_for_full_snapshot_commands(monkeypatch, capsys):
    config = {"train_data_access": {"start_ns": 10, "end_ns": 20,
                                    "marker_sha256": "a" * 64,
                                    "label": "train-stage69"}}
    calls = []
    monkeypatch.setattr(application, "load_config", lambda _path: config)
    monkeypatch.setattr(application, "train", lambda supplied, *, evaluate:
                        calls.append((supplied, evaluate)) or {"status": "TRAIN_ONLY"})

    assert application.main(["train", "--config", "unused.json", "--train-only"]) == 0
    assert calls == [(config, False)]
    capsys.readouterr()
    assert application.main(["wfo", "--config", "unused.json"]) == 2
    assert "limited to Train-only commands" in capsys.readouterr().err
    assert application.main(["backtest", "--config", "unused.json", "--bundle", "unused"]) == 2
    assert "limited to Train-only commands" in capsys.readouterr().err


def test_training_coverage_report_separates_interaction_budget_from_full_episode_passes():
    short = application._training_coverage_report(
        rl_steps=64, num_envs=1, train_window_decision_steps=60 * 24 * 60,
        episode_decision_steps=60 * 24 * 60, episode_start_offsets=(0,),
    )
    assert short["upper_bound_unique_decision_steps_per_environment"] == [64]
    assert short["nominal_full_episode_passes_per_environment"] == [0]
    assert short["upper_bound_train_window_coverage_fraction"] == pytest.approx(
        64 / (60 * 24 * 60),
    )

    repeated = application._training_coverage_report(
        rl_steps=2_900, num_envs=2, train_window_decision_steps=3_000,
        episode_decision_steps=1_000, episode_start_offsets=(0, 1_500),
    )
    assert repeated["interaction_steps_per_environment"] == [1_450, 1_450]
    assert repeated["upper_bound_unique_decision_steps_per_environment"] == [1_450, 1_450]
    assert repeated["upper_bound_unique_train_window_steps"] == 2_900
    assert repeated["initial_episode_start_offsets"] == [0, 1_500]
    assert repeated["sampling_policy"] == "rank_exclusive_coprime_rotating_windows_v1"
    assert repeated["nominal_full_episode_passes_per_environment"] == [1, 1]
    assert repeated["nominal_partial_episode_steps_per_environment"] == [450, 450]


def test_rl_sampling_policy_requires_bounded_distinct_parallel_episodes():
    assert application._rl_sampling_policy({}) == (1, None)
    assert application._rl_sampling_policy({"rl_num_envs": 4, "rl_episode_steps": 10_000}) == (4, 10_000)
    with pytest.raises(Exception, match="explicit fixed episode steps"):
        application._rl_sampling_policy({"rl_num_envs": 2})
    with pytest.raises(Exception, match="1-4 environments"):
        application._rl_sampling_policy({"rl_num_envs": 5, "rl_episode_steps": 100})


def test_fixed_observation_corpus_uses_deterministic_stratified_train_windows(monkeypatch):
    factory = SimpleNamespace(rows=tuple(SimpleNamespace(decision_ts_ns=index)
                                         for index in range(8 * 100 + 3)))
    calls = []

    def fake_rollout(_factory, _predict, *, seed, episode_rank, episode_count, episode_steps,
                     reset_options):
        calls.append((seed, episode_rank, episode_count, episode_steps))
        start = episode_rank * 100
        values = np.arange(start, start + episode_steps + 1, dtype=np.float32)[:, None]
        return ({"initial_decision_ts_ns": start, "episode_window": {"test": True}},
                np.repeat(values, 2, axis=1))

    monkeypatch.setattr(application, "rollout", fake_rollout)
    corpus, metadata = application._fixed_observation_corpus(
        factory, lambda batch: batch, seed=31, episodes=8, episode_steps=20,
    )

    assert calls == [(31 + rank, rank, 8, 20) for rank in range(8)]
    assert corpus.shape == (8 * 21, 2)
    assert metadata["unique_decision_timestamps"] == 160
    assert metadata["initial_start_offsets"] == [rank * 100 for rank in range(8)]
    assert len(metadata["observation_sha256"]) == 64
    repeated, repeated_metadata = application._fixed_observation_corpus(
        factory, lambda batch: batch, seed=31, episodes=8, episode_steps=20,
    )
    np.testing.assert_array_equal(corpus, repeated)
    assert metadata == repeated_metadata


def test_fixed_corpus_policy_rejects_unbounded_values():
    assert application._fixed_corpus_policy({}) == (1, 256)
    with pytest.raises(Exception, match="one episode"):
        application._fixed_corpus_policy({"rl_corpus_episodes": 0})


def test_single_r1_environment_reuses_the_already_frozen_train_factory():
    calls = []

    class SharedFactory:
        def make_env(self, transition_hash, *, episode_rank, episode_count, episode_steps):
            calls.append((transition_hash, episode_rank, episode_count, episode_steps))
            return "shared-native-environment"

    factory = application._HistoricalRLEnvFactory(
        recipe_json="{}", scaler_json="{}", transition_hash="a" * 64,
        num_envs=1, episode_steps=8192, shared_factory=SharedFactory(),
    )
    assert factory(0, 123, "train") == "shared-native-environment"
    assert calls == [("a" * 64, 0, 1, 8192)]


def test_prepared_train_factory_recipe_identity_binds_window_and_reward_scale(monkeypatch):
    recipe = {"environment": {"instrument": "ETHUSDT-PERP.SIM", "start_ns": 10,
                               "end_ns": 20}, "reward_scale": 1000.0}
    assert application._training_factory_recipe_hash(recipe) == application._training_factory_recipe_hash(recipe)
    assert application._training_factory_recipe_hash({**recipe, "reward_scale": 10_000.0}) != \
        application._training_factory_recipe_hash(recipe)
    assert application._training_factory_recipe_hash({
        **recipe, "environment": {**recipe["environment"], "end_ns": 21},
    }) != application._training_factory_recipe_hash(recipe)

    created = SimpleNamespace()

    def fake_factory(**values):
        assert values["split"] == "train_inner" and values["fit_scaler"] is True
        return created

    monkeypatch.setattr(application, "HistoricalEnvironmentFactory", fake_factory)
    monkeypatch.setattr(application, "record_development", lambda *_args, **_kwargs: None)
    assert application.prepare(recipe) is created
    assert created.application_training_recipe_hash == application._training_factory_recipe_hash(recipe)


def test_prepare_uses_registered_scoped_market_before_factory(monkeypatch):
    recipe = {
        "environment": {"market_path": "train.parquet", "rules_path": "rules.json",
                        "instrument": "ETHUSDT-PERP.SIM", "start_ns": 10, "end_ns": 20},
        "train_data_access": {"start_ns": 10, "end_ns": 20,
                               "marker_sha256": "a" * 64, "label": "train-stage69"},
    }
    market = SimpleNamespace(events=("market-row",))
    calls = []
    monkeypatch.setattr(application, "read_registered_train_window",
                        lambda *args, **kwargs: calls.append((args, kwargs)) or market)
    monkeypatch.setattr(application, "load_rules", lambda _path: ("rule-row",))
    monkeypatch.setattr(application, "CuratedStore", lambda events: tuple(events))
    monkeypatch.setattr(application, "record_development",
                        lambda *_args, **_kwargs: pytest.fail("scoped access must reserve through its reader"))
    created = SimpleNamespace()

    def fake_factory(**values):
        assert values["prepared_store"] == ("market-row", "rule-row")
        assert values["split"] == "train_inner" and values["fit_scaler"] is True
        return created

    monkeypatch.setattr(application, "HistoricalEnvironmentFactory", fake_factory)
    assert application.prepare(recipe) is created
    assert calls == [((recipe, {"start_ns": 10, "end_ns": 20}), {
        "marker_sha256": "a" * 64, "label": "train-stage69",
        "allow_legacy_isolated_snapshot": False,
    })]
    recipe["train_data_access"]["source_mode"] = "whole_snapshot_isolated_v1"
    assert application.prepare(recipe) is created
    assert calls[1][1]["allow_legacy_isolated_snapshot"] is True


def test_prepare_rejects_unhashable_source_mode_before_registered_reader(monkeypatch):
    from rles_data.core import DataError

    config = {
        "output": "run",
        "environment": {"market_path": "train.parquet", "rules_path": "rules.json",
                        "instrument": "BTC/USD", "start_ns": 10, "end_ns": 20},
        "train_data_access": {"start_ns": 10, "end_ns": 20,
                               "marker_sha256": "a" * 64, "label": "train-scope",
                               "source_mode": []},
    }
    monkeypatch.setattr(application, "read_registered_train_window",
                        lambda *_args, **_kwargs: pytest.fail("invalid source mode reached reader"))
    with pytest.raises(DataError, match="invalid scoped Train data access"):
        application.prepare(config)


def test_spawned_r1_factory_reloads_registered_train_scope(monkeypatch):
    observed = {}
    prepared = object()

    def read_train(config):
        observed["access_config"] = config
        return prepared

    monkeypatch.setattr(application, "_read_registered_train_store", read_train)
    monkeypatch.setattr(application, "Scaler", lambda **_kwargs: "frozen-scaler")

    def make_factory(**kwargs):
        observed["factory_kwargs"] = kwargs
        return SimpleNamespace(make_env=lambda *args, **options: (args, options))

    monkeypatch.setattr(application, "HistoricalEnvironmentFactory", make_factory)
    access_config = {"output": "run", "environment": {"market_path": "train.parquet"},
                     "train_data_access": {"start_ns": 10, "end_ns": 20,
                                           "marker_sha256": "a" * 64,
                                           "label": "train-r1-worker"}}
    access_json = json.dumps(access_config)
    factory = application._HistoricalRLEnvFactory(
        json.dumps({"instrument": "BTC/USD"}), json.dumps({"mean": [0.0], "std": [1.0]}),
        "transition", 2, 8, train_access_json=access_json,
    )

    _args, options = factory(1, 7, "eval")

    assert observed["access_config"] == access_config
    assert observed["factory_kwargs"]["prepared_store"] is prepared
    assert options["episode_rank"] == 1 and options["episode_count"] == 2


def test_es_worker_factory_reloads_registered_train_scope(monkeypatch):
    from rles_gym_bridge import historical
    from rles_orchestrator.native_es_worker import make_registered_historical_es_env

    prepared = object()
    access = {"output": "run", "environment": {"market_path": "train.parquet"},
              "train_data_access": {"start_ns": 10, "end_ns": 20,
                                    "marker_sha256": "a" * 64, "label": "train-es-worker"}}
    captured = {}
    def read_train(supplied):
        captured["access"] = supplied
        return prepared

    def make_env(**kwargs):
        captured["env"] = kwargs
        return "candidate-env"

    monkeypatch.setattr(application, "_read_registered_train_store", read_train)
    monkeypatch.setattr(historical, "make_historical_es_env", make_env)

    result = make_registered_historical_es_env(
        env_config={"recipe": {}, "registered_train_access": access},
        candidate_context={"split": "train_inner"}, soft_risk_multipliers={},
    )

    assert result == "candidate-env"
    assert captured["access"] == access
    assert captured["env"]["env_config"]["prepared_store"] is prepared


def test_scoped_train_config_cannot_start_evaluation_or_full_market_audit():
    from rles_data.core import DataError

    config = {"train_data_access": {}}
    with pytest.raises(DataError, match="evaluate=False"):
        application.train(config, evaluate=True)
    with pytest.raises(DataError, match="registered Train data access"):
        application.audit_market(config)
    with pytest.raises(DataError, match="registered Train reader"):
        application.backtest(config, "unused-bundle")


@pytest.mark.parametrize("config", [
    {"environment": {"market_path": "market.parquet"}},
    {"environment": {"market_path": "market.parquet"},
     "train_data_access": {"start_ns": 10, "end_ns": 20,
                           "marker_sha256": "a" * 64, "label": "train-audit"},
     "evaluation": {"split": "test", "start_ns": 20, "end_ns": 30}},
    {"environment": {"market_path": "market.parquet", "split": "test"},
     "train_data_access": {"start_ns": 10, "end_ns": 20,
                           "marker_sha256": "a" * 64, "label": "train-audit"}},
    {"environment": {"market_path": "market.parquet", "instrument": "BTC/USD",
                      "start_ns": 10, "end_ns": 20, "coverage_policy": {}},
     "train_data_access": {"start_ns": 10, "end_ns": 20,
                           "marker_sha256": "a" * 64, "label": "train-audit",
                           "source_mode": {}}},
])
def test_market_audit_rejects_unscoped_or_formal_evaluation_config_before_reader(
    monkeypatch, config,
):
    from rles_data.core import DataError

    monkeypatch.setattr(
        application, "read_registered_train_window",
        lambda *_args, **_kwargs: pytest.fail("unauthorized market payload reader was called"),
    )
    with pytest.raises(DataError, match="Train-only|registered Train data access|complete Train-inner"):
        application.audit_market(config)


@pytest.mark.parametrize("config", [
    {},
    {"train_data_access": {"start_ns": 1, "end_ns": 2,
                           "marker_sha256": "a" * 64, "label": "train-diagnostic"},
     "evaluation": {"split": "test", "start_ns": 2, "end_ns": 3}},
    {"train_data_access": {"start_ns": 1, "end_ns": 2,
                           "marker_sha256": "a" * 64, "label": "train-diagnostic"},
     "evaluation": []},
    {"train_data_access": {"start_ns": 1, "end_ns": 2,
                           "marker_sha256": "a" * 64, "label": "train-diagnostic"},
     "evaluation": "test"},
])
def test_train_diagnostic_requires_scoped_access_and_rejects_evaluation_before_bundle_read(
    monkeypatch, config,
):
    from rles_data.core import DataError

    monkeypatch.setattr(application, "_verify_file",
                        lambda *_args, **_kwargs: pytest.fail("bundle read preceded scope validation"))
    monkeypatch.setattr(application, "read_registered_train_window",
                        lambda *_args, **_kwargs: pytest.fail("market payload reader was called"))
    with pytest.raises(DataError, match="registered scoped Train data access|reject Evaluation"):
        application.diagnose_train(config, "missing-bundle")


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
                info["quantity_rounded_target_exposure"] = (0.35, -0.15, 0.0)[self.index - 1]
                info["execution_plan_action"] = ("SUBMIT", "NOOP", "NOOP")[self.index - 1]
                info["execution_plan_reason"] = ("rebalance", "deadband", "below minimum executable quantity/notional")[self.index - 1]
            return np.full(2, self.index, dtype=np.float32), 0.0, False, self.index == 3, info

        def close(self) -> None:
            return None

    class Factory:
        mode = "base"
        snapshots = ()
        fill_references = ()
        schema = SimpleNamespace(names=("market.feature_1", "market.feature_2"))
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
    action_market_response = report["action_market_response"]
    expected_correlation = np.corrcoef([0.0, 1.0, 2.0], [0.8, -0.2, 0.5])[0, 1]
    assert action_market_response["count"] == 3
    assert action_market_response["pearson_correlation_with_requested_action"] == {
        "market.feature_1": pytest.approx(expected_correlation),
        "market.feature_2": pytest.approx(expected_correlation),
    }
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
        assert minimum["quantity_rounding"]["risk_to_rounded_gap_abs_mean"] == pytest.approx(0.25)
        assert minimum["quantity_rounding"]["rounded_to_executed_gap_abs_mean"] == 0
        assert groups["rebalance"]["action"] == "SUBMIT"
        assert attribution["quantity_rounding_components_known"] is True
        assert groups["rebalance"]["quantity_rounding"]["risk_to_rounded_gap_abs_mean"] == pytest.approx(0.05)
    else:
        assert attribution["groups"] == []
        assert attribution["quantity_rounding_components_known"] is False
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
    assert report["quantity_rounding_diagnostics"] == {
        "components_known": plan_metadata,
        "count": 3 if plan_metadata else 0,
        **({
            "risk_adjusted_target_abs_mean": pytest.approx(0.25),
            "quantity_rounded_target_abs_mean": pytest.approx(1 / 6),
            "risk_to_rounded_target_gap_abs_mean": pytest.approx(0.1166666667),
            "rounded_target_to_actual_exposure_gap_abs_mean": pytest.approx(1 / 30),
        } if plan_metadata else {}),
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
        split="validation", start_ns=1, end_ns=2, instrument="ETHUSDT-PERP.SIM",
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
    config = {"seed": 1, "output": str(tmp_path / "output"),
              "environment": {"instrument": "ETHUSDT-PERP.SIM"}}

    report = application.backtest(config, bundle_root, factory=factory, allow_execution_stress=True)
    assert report["execution_mode"] == "stress"
    assert report["latency_model_id"] == "stress-latency"
    assert report["fill_model_id"] == "stress-fill"
    assert report["simulated_depth_fraction"] == "1"
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
    trace_options = {}
    monkeypatch.setattr(application, "backtest",
                        lambda _config, _bundle, **kwargs: (trace_options.update(kwargs) or report))
    monkeypatch.setattr(application, "diagnose_train",
                        lambda _config, _bundle, **kwargs: (trace_options.update(kwargs) or report))

    assert application.main([command, "--config", "unused.json", "--bundle", str(bundle),
                             "--execution-trace"]) == 0
    assert trace_options == {"include_execution_trace": True}

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
