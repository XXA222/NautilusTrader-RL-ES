"""JSON CLI for B0 audits and baseline scoring; never reads unregistered files."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

from rles_data.core import DataError, FoldManifest, Window

from .baselines import score_baselines
from .contracts import MarketFrame, MarketReplay
from .data_audit import audit_replay
from .train_dataset import project_registered_train_replay
from .upstream import inspect_upstream


def _read_json(path: str) -> dict:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise DataError(f"cannot read valid JSON input: {path}") from exc
    if not isinstance(value, dict):
        raise DataError("JSON input root must be an object")
    return value


def _replay(path: str) -> tuple[dict, MarketReplay]:
    payload = _read_json(path)
    if set(payload) != {"schema_version", "evidence_scope", "dataset_hash", "instrument", "frames"}:
        raise DataError("synthetic replay must declare schema, evidence scope, dataset hash, instrument, and frames")
    if (payload["schema_version"] != "rles.market_replay.v0"
            or payload["evidence_scope"] != "synthetic_test_only"
            or not isinstance(payload["frames"], list)):
        raise DataError("only synthetic_test_only replay JSON is accepted by this mechanism interface")
    frames = tuple(MarketFrame(**item) for item in payload["frames"])
    replay = MarketReplay(frames, payload["dataset_hash"])
    if any(frame.instrument != payload["instrument"] for frame in frames):
        raise DataError("market replay frames do not match declared instrument")
    return payload, replay


def _registered_train_replay(path: str) -> tuple[dict, MarketReplay]:
    payload = _read_json(path)
    expected = {"schema_version", "config", "window", "marker_sha256", "label", "instrument",
                "allow_legacy_isolated_snapshot"}
    if set(payload) != expected or payload["schema_version"] != "rles.world_model.registered_train_audit.v0":
        raise DataError("registered Train config does not match the supported audit schema")
    config = payload["config"]
    window = payload["window"]
    if (not isinstance(config, dict) or not isinstance(config.get("environment"), dict)
            or config.get("evaluation") is not None or config.get("test") is not None
            or config.get("holdout") is not None):
        raise DataError("B0 registered audit accepts a Train-only application config")
    if (not isinstance(window, dict) or set(window) != {"start_ns", "end_ns"}
            or not isinstance(payload["marker_sha256"], str)
            or not isinstance(payload["label"], str) or not payload["label"].startswith("train-")
            or not isinstance(payload["instrument"], str)
            or type(payload["allow_legacy_isolated_snapshot"]) is not bool):
        raise DataError("invalid registered Train audit scope")
    replay = project_registered_train_replay(
        config, window, marker_sha256=payload["marker_sha256"], label=payload["label"],
        instrument=payload["instrument"],
        allow_legacy_isolated_snapshot=payload["allow_legacy_isolated_snapshot"],
    )
    return payload, replay


def _fold(value: dict) -> FoldManifest:
    if set(value) != {"schema_version", "dataset_hash", "feature_hash", "train", "validation",
                      "test", "purge_ns", "embargo_ns", "scaler_hash"}:
        raise DataError("fold JSON does not match the supported FoldManifest schema")
    return FoldManifest(value["schema_version"], value["dataset_hash"], value["feature_hash"],
                        Window(**value["train"]), Window(**value["validation"]),
                        Window(**value["test"]), value["purge_ns"], value["embargo_ns"],
                        value["scaler_hash"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="rles-world-model")
    commands = parser.add_subparsers(dest="command", required=True)
    audit = commands.add_parser("audit-data")
    audit_source = audit.add_mutually_exclusive_group(required=True)
    audit_source.add_argument("--registered-train-config",
                              help="config that reserves an exact Train interval before reading curated rows")
    audit_source.add_argument("--synthetic-replay-json",
                              help="synthetic_test_only JSON for mechanism tests; never real research evidence")
    baseline = commands.add_parser("baselines")
    baseline.add_argument("--synthetic-replay-json", required=True,
                          help="synthetic_test_only JSON for mechanism tests; never real research evidence")
    baseline.add_argument("--fold-json", required=True)
    baseline.add_argument("--seed", type=int, default=0)
    baseline.add_argument("--ensemble-size", type=int, default=128)
    upstream = commands.add_parser("inspect-upstream")
    upstream.add_argument("--checkout", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "audit-data":
            if args.registered_train_config:
                _, replay = _registered_train_replay(args.registered_train_config)
                result = asdict(audit_replay(replay))
                result["evidence_scope"] = "registered_train"
            else:
                _, replay = _replay(args.synthetic_replay_json)
                result = asdict(audit_replay(replay))
                result["evidence_scope"] = "synthetic_test_only"
        elif args.command == "baselines":
            _, replay = _replay(args.synthetic_replay_json)
            fold = _fold(_read_json(args.fold_json))
            result = {"status": "COMPLETED", "scores": [asdict(score) for score in
                                                              score_baselines(replay, fold, seed=args.seed,
                                                                              ensemble_size=args.ensemble_size)]}
        else:
            result = asdict(inspect_upstream(args.checkout))
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        return 0 if result.get("status", "READY") not in {"BLOCKED", "FAILED"} else 2
    except Exception as exc:
        print(json.dumps({"status": "BLOCKED", "error": type(exc).__name__, "detail": str(exc)},
                         ensure_ascii=False), file=sys.stderr)
        return 2
