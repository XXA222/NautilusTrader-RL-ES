import json

from rles_orchestrator.application import main
import rles_world_model.cli as world_model_cli
from rles_world_model.contracts import MarketFrame, MarketReplay


def test_world_model_cli_dispatches_through_rles(tmp_path, capsys):
    minute = 60_000_000_000
    frames = []
    for index in range(1, 4):
        price = 100.0 + index
        frames.append({
            "ts_close_ns": minute * index,
            "available_at_ns": minute * index,
            "instrument": "ETHUSDT-PERP.SIM",
            "open": price,
            "high": price + 1,
            "low": price - 1,
            "close": price,
            "volume": 1.0,
        })
    replay = tmp_path / "replay.json"
    replay.write_text(json.dumps({
        "schema_version": "rles.market_replay.v0",
        "evidence_scope": "synthetic_test_only",
        "dataset_hash": "a" * 64,
        "instrument": "ETHUSDT-PERP.SIM",
        "frames": frames,
    }), encoding="utf-8")

    assert main(["world-model", "audit-data", "--synthetic-replay-json", str(replay)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["bars"] == 3
    assert report["market_model_eligible"] is True
    assert report["perpetual_economics_eligible"] is False
    assert report["evidence_scope"] == "synthetic_test_only"


def test_registered_train_cli_routes_through_scoped_reader_before_audit(tmp_path, monkeypatch, capsys):
    minute = 60_000_000_000
    replay = MarketReplay((
        MarketFrame(minute, minute, "ETHUSDT-PERP.SIM", 100, 101, 99, 100, 1),
        MarketFrame(2 * minute, 2 * minute, "ETHUSDT-PERP.SIM", 100, 101, 99, 100, 1),
    ), "b" * 64)
    captured = {}

    def fake_reader(config, window, **kwargs):
        captured["config"] = config
        captured["window"] = window
        captured.update(kwargs)
        return replay

    monkeypatch.setattr(world_model_cli, "project_registered_train_replay", fake_reader)
    config = {
        "schema_version": "rles.world_model.registered_train_audit.v0",
        "config": {"output": "routeb-b0", "environment": {
            "market_path": "market.parquet", "instrument": "ETHUSDT-PERP.SIM",
            "start_ns": minute, "end_ns": 3 * minute,
        }},
        "window": {"start_ns": minute, "end_ns": 3 * minute},
        "marker_sha256": "c" * 64,
        "label": "train-routeb-b0-smoke",
        "instrument": "ETHUSDT-PERP.SIM",
        "allow_legacy_isolated_snapshot": False,
    }
    path = tmp_path / "registered-train.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    assert main(["world-model", "audit-data", "--registered-train-config", str(path)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["evidence_scope"] == "registered_train"
    assert captured["label"] == "train-routeb-b0-smoke"
    assert captured["marker_sha256"] == "c" * 64


def test_registered_train_cli_rejects_evaluation_config_before_reader(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(world_model_cli, "project_registered_train_replay",
                        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("reader called")))
    config = {
        "schema_version": "rles.world_model.registered_train_audit.v0",
        "config": {"output": "routeb-b0", "environment": {"market_path": "market.parquet"},
                   "evaluation": {"start_ns": 100, "end_ns": 200}},
        "window": {"start_ns": 10, "end_ns": 20},
        "marker_sha256": "c" * 64,
        "label": "train-routeb-b0-smoke",
        "instrument": "ETHUSDT-PERP.SIM",
        "allow_legacy_isolated_snapshot": False,
    }
    path = tmp_path / "evaluation.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    assert main(["world-model", "audit-data", "--registered-train-config", str(path)]) == 2
    assert json.loads(capsys.readouterr().err)["status"] == "BLOCKED"
