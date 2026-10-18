from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
STAGE58 = ROOT / ".tmp/rl-stage58-train-window-coverage-20261016"
OUTPUTS = ROOT / ".tmp/rl-stage59-train-diagnostics-v3-20261017"
EVIDENCE = ROOT / "docs/audit_evidence/20261017/stage59_train_diagnostics"
SUPERSEDED_EVIDENCE = ROOT / "docs/audit_evidence/20261017/stage59_train_diagnostics_revision2"
CASES = ((1, "2024-01-01"), (2, "2024-09-01"),
         (3, "2025-05-01"), (4, "2026-01-01"))
CODE_PATHS = (
    "packages/gym_bridge/src/rles_gym_bridge/env.py",
    "packages/gym_bridge/src/rles_gym_bridge/historical.py",
    "packages/rl_sb3/src/rles_rl_sb3/provider.py",
    "packages/orchestrator/src/rles_orchestrator/rl_stage.py",
    "packages/orchestrator/src/rles_orchestrator/application.py",
    "qa/preregister_stage59_train_diagnostics.py",
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def write_new(path: Path, payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(encoded)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=SUPERSEDED_EVIDENCE)
    parser.add_argument("--supersedes", type=Path, default=EVIDENCE / "preregistration.json")
    args = parser.parse_args()
    output: Path = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    summary = load(STAGE58 / "verification-summary.json")
    if (summary.get("status") != "passed"
            or summary.get("evaluation_queries") != {
                "validation": 0, "test": 0, "holdout": 0, "outer_2026": 0,
            }):
        raise ValueError("Stage58 Train-only source verification did not pass")
    config_hashes: dict[str, str] = {}
    windows = []
    for index, day in CASES:
        case_name = f"window-{index:02d}-{day}"
        prior = STAGE58 / case_name
        previous_record = next(row for row in summary["runs"] if row["window_id"] == index)
        if (sha(prior / "config.json") != previous_record["config_sha256"]
                or sha(prior / "preregistration.json") != previous_record["preregistration_sha256"]):
            raise ValueError(f"frozen Stage58 config/preregistration changed: {day}")
        previous_prereg = load(prior / "preregistration.json")
        artifact_entries = previous_prereg.get("sources", {})
        data_sources = {}
        for name in ("base_config", "market", "market_completion", "scenario_rules"):
            entry = artifact_entries.get(name)
            if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
                raise ValueError(f"Stage58 prerequisite source missing: {name}")
            source_path = Path(entry["path"])
            if not source_path.is_absolute():
                source_path = ROOT / source_path
            if sha(source_path) != entry.get("sha256"):
                raise ValueError(f"Stage58 prerequisite source hash changed: {name}")
            data_sources[name] = {"path": str(source_path), "sha256": entry["sha256"]}
        old = load(prior / "config.json")
        if old.get("evaluation") is not None or old.get("seed") != 102 or old.get("steps") != 8192:
            raise ValueError(f"Stage58 case is outside the fixed Train-only recipe: {day}")
        new_config = dict(old)
        new_config["output"] = str(OUTPUTS / case_name / "run")
        config_path = output / f"{case_name}.config.json"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_sha = write_new(config_path, new_config)
        config_hashes[case_name] = config_sha
        windows.append({
            "window_id": index,
            "start_date_utc": day,
            "config_path": str(config_path),
            "config_sha256": config_sha,
            "market_start_ns": new_config["environment"]["start_ns"],
            "market_end_ns": new_config["environment"]["end_ns"],
            "steps": new_config["steps"],
            "episodes": 1,
            "seed": new_config["seed"],
            "output": new_config["output"],
        })
    source_hashes = {relative: sha(ROOT / relative) for relative in CODE_PATHS}
    # Include only source/config evidence; no raw market file is opened here.
    prereg = {
        "schema": "rles.stage59-train-diagnostic-preregistration.v1",
        "status": "frozen_before_diagnostic_replay",
        "supersedes": {"path": str(args.supersedes), "sha256": sha(args.supersedes)},
        "prior_attempt": {
            "status": "blocked_before_learning",
            "detail": "CLI train command previously required a Validation window even though the Python training API supports evaluate=False; no R1 learner stage ran and no Validation query occurred.",
            "validation_queries": 0, "test_queries": 0, "holdout_queries": 0,
        },
        "objective": "Re-run the already-used Stage58 fixed PPO Train-only recipe solely to emit v3 episode, position, market-regime and r_econ diagnostics; no candidate selection or evaluation split.",
        "source_stage58_verification_sha256": sha(STAGE58 / "verification-summary.json"),
        "stage58_preregistrations_sha256": {
            f"window-{index:02d}-{day}": sha(
                STAGE58 / f"window-{index:02d}-{day}" / "preregistration.json"
            ) for index, day in CASES
        },
        "data_and_base_config_sha256": data_sources,
        "stage58_reused_train_dates": [day for _, day in CASES],
        "source_code_sha256": source_hashes,
        "windows": windows,
        "evaluation_queries": {"validation": 0, "test": 0, "holdout": 0, "outer_2026": 0},
        "fixed_policy": {
            "algorithm": "PPO", "seed": 102, "steps_per_window": 8192,
            "environment_count": 1, "episode_steps": 8192,
            "purpose": "diagnostic_replay_only_no_selection_no_validation",
            "expected_learning_schema": "rles.r1-learning.v3",
        },
        "market_regime_policy": {
            "volatility": "train-call empirical quartiles, linear quantile method, cutpoints persisted per block",
            "trend": "sign buckets from highest-available return_15m/return_5m/return_1m feature",
            "funding": "sign buckets from last_settled_funding feature",
            "economic_contribution": "sum and mean of native per-step r_econ within each bucket",
            "limitations": [
                "quartile thresholds are specific to each learn block and are descriptive, not cross-window fixed thresholds",
                "these previously reused Train dates do not constitute independent evidence",
                "all rewards retain Stage58 scenario-only rules and costs",
            ],
        },
    }
    prereg_path = output / "preregistration.json"
    prereg_sha = write_new(prereg_path, prereg)
    print(json.dumps({"preregistration": str(prereg_path),
                      "sha256": prereg_sha, "config_hashes": config_hashes,
                      "windows": len(windows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
