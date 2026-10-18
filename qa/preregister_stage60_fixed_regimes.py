from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
STAGE58 = ROOT / ".tmp/rl-stage58-train-window-coverage-20261016"
STAGE59 = ROOT / "docs/audit_evidence/20261017/stage59_train_diagnostics_revision2"
OUTPUTS = ROOT / ".tmp/rl-stage60-fixed-regimes-train-only-20261017"
EVIDENCE = ROOT / "docs/audit_evidence/20261017/stage60_fixed_regimes"
CASES = ((1, "2024-01-01"), (2, "2024-09-01"),
         (3, "2025-05-01"), (4, "2026-01-01"))
CODE_PATHS = (
    "packages/gym_bridge/src/rles_gym_bridge/env.py",
    "packages/gym_bridge/src/rles_gym_bridge/historical.py",
    "packages/rl_sb3/src/rles_rl_sb3/provider.py",
    "packages/orchestrator/src/rles_orchestrator/rl_stage.py",
    "packages/orchestrator/src/rles_orchestrator/application.py",
    "qa/preregister_stage60_fixed_regimes.py",
    "qa/analyze_stage60_fixed_regimes.py",
)
VOLATILITY_CUTS = [0.00025, 0.0005, 0.001]


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
    parser.add_argument("--output", type=Path, default=EVIDENCE)
    args = parser.parse_args()
    output: Path = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    stage58_summary = load(STAGE58 / "verification-summary.json")
    stage59_prereg = load(STAGE59 / "preregistration.json")
    stage59_analysis = ROOT / "docs/audit_evidence/20261017/stage59_train_diagnostics_analysis.json"
    stage59_analysis_sha = sha(stage59_analysis)
    if (stage58_summary.get("status") != "passed"
            or stage58_summary.get("evaluation_queries") != {
                "validation": 0, "test": 0, "holdout": 0, "outer_2026": 0,
            }
            or stage59_prereg.get("status") != "frozen_before_diagnostic_replay"
            or stage59_prereg.get("evaluation_queries") != {
                "validation": 0, "test": 0, "holdout": 0, "outer_2026": 0,
            }
            or sha(STAGE58 / "verification-summary.json")
            != stage59_prereg.get("source_stage58_verification_sha256")
            or stage59_prereg.get("market_regime_policy") is None
            or sha(stage59_analysis) != "395b2e2e314fda0a9cb219271c71704df9f2cfa68ad301561d1d967f95526acc"):
        raise ValueError("Stage58/59 frozen Train-only evidence is invalid")

    input_sources = stage59_prereg.get("data_and_base_config_sha256")
    if not isinstance(input_sources, dict):
        raise ValueError("Stage59 frozen input hashes are missing")
    for entry in input_sources.values():
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
            raise ValueError("Stage59 frozen source identity is invalid")
        if sha(Path(entry["path"])) != entry.get("sha256"):
            raise ValueError(f"Stage59 frozen source changed: {entry['path']}")

    configs: dict[str, dict[str, Any]] = {}
    windows = []
    for index, day in CASES:
        name = f"window-{index:02d}-{day}"
        prior_case = STAGE58 / name
        prior_config = load(prior_case / "config.json")
        prior_record = next(row for row in stage58_summary["runs"] if row["window_id"] == index)
        if (sha(prior_case / "config.json") != prior_record["config_sha256"]
                or prior_config.get("evaluation") is not None
                or prior_config.get("seed") != 102 or prior_config.get("steps") != 8192):
            raise ValueError(f"Stage58 frozen Train recipe changed: {day}")
        config = dict(prior_config)
        config["output"] = str(OUTPUTS / name / "run")
        config_path = output / f"{name}.config.json"
        config_hash = write_new(config_path, config)
        configs[name] = {"path": str(config_path), "sha256": config_hash}
        windows.append({
            "window_id": index,
            "start_date_utc": day,
            "market_start_ns": config["environment"]["start_ns"],
            "market_end_ns": config["environment"]["end_ns"],
            "steps": config["steps"],
            "episodes": 1,
            "seed": config["seed"],
            "config": configs[name],
            "output": config["output"],
        })

    payload = {
        "schema": "rles.stage60-fixed-regime-preregistration.v1",
        "status": "frozen_before_diagnostic_replay",
        "objective": (
            "Re-run the already-used Stage58 Train-only PPO recipe to emit v4 diagnostics "
            "using fixed cross-window volatility boundaries; no model selection or evaluation."
        ),
        "supersedes": {
            "stage59_preregistration": {
                "path": str(STAGE59 / "preregistration.json"),
                "sha256": sha(STAGE59 / "preregistration.json"),
            },
            "stage59_analysis": {"path": str(stage59_analysis), "sha256": stage59_analysis_sha},
        },
        "source_stage58_verification_sha256": sha(STAGE58 / "verification-summary.json"),
        "source_stage59_preregistration_sha256": sha(STAGE59 / "preregistration.json"),
        "frozen_inputs": input_sources,
        "source_code_sha256": {relative: sha(ROOT / relative) for relative in CODE_PATHS},
        "reused_train_dates": [day for _, day in CASES],
        "windows": windows,
        "evaluation_queries": {"validation": 0, "test": 0, "holdout": 0, "outer_2026": 0},
        "fixed_policy": {
            "algorithm": "PPO", "seed": 102, "steps_per_window": 8192,
            "environment_count": 1, "episode_steps": 8192,
            "purpose": "train_inner_diagnostic_only_no_selection_no_validation",
            "expected_learning_schema": "rles.r1-learning.v4",
        },
        "regime_policy": {
            "volatility_feature": "volatility_5m",
            "volatility_scheme": "fixed_absolute_volatility_upper_closed_v1",
            "volatility_ratio_upper_closed_cuts": VOLATILITY_CUTS,
            "trend_feature": "return_15m, fallback return_5m then return_1m",
            "trend_scheme": "signed_feature_v1",
            "funding_feature": "last_settled_funding",
            "funding_scheme": "signed_feature_v1",
            "economic_contribution": "per-step native r_econ sum and mean",
            "threshold_source": "fixed round raw-ratio engineering scales; not fitted, tuned, or used by policy",
            "limitations": [
                "all four windows were previously used Train data and are not independent evidence",
                "scenario-only costs and rules remain unchanged",
                "fixed bins improve descriptive comparability but do not validate regime predictive value",
                "no Validation, Test, Holdout, or outer-2026 query is allowed in this replay",
            ],
        },
    }
    prereg_path = output / "preregistration.json"
    prereg_sha = write_new(prereg_path, payload)
    print(json.dumps({
        "preregistration": str(prereg_path), "sha256": prereg_sha,
        "config_hashes": configs, "windows": len(windows),
        "volatility_ratio_upper_closed_cuts": VOLATILITY_CUTS,
        "evaluation_queries": payload["evaluation_queries"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
