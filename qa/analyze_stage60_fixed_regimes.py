from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
STAGE58 = ROOT / ".tmp/rl-stage58-train-window-coverage-20261016"
STAGE59 = ROOT / ".tmp/rl-stage59-train-diagnostics-v3-20261017"
PREREG_PATH = ROOT / "docs/audit_evidence/20261017/stage60_fixed_regimes/preregistration.json"
OUTPUT_PATH = ROOT / "docs/audit_evidence/20261017/stage60_fixed_regimes/analysis.json"
CASES = ((1, "2024-01-01"), (2, "2024-09-01"),
         (3, "2025-05-01"), (4, "2026-01-01"))
EXPECTED_CUTS = [0.00025, 0.0005, 0.001]
EXPECTED_SCHEME = "fixed_absolute_volatility_upper_closed_v1"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def analyze() -> dict[str, Any]:
    prereg = load(PREREG_PATH)
    stage58 = load(STAGE58 / "verification-summary.json")
    if (prereg.get("schema") != "rles.stage60-fixed-regime-preregistration.v1"
            or prereg.get("status") != "frozen_before_diagnostic_replay"
            or prereg.get("evaluation_queries") != {
                "validation": 0, "test": 0, "holdout": 0, "outer_2026": 0,
            }
            or stage58.get("status") != "passed"
            or sha(STAGE58 / "verification-summary.json")
            != prereg.get("source_stage58_verification_sha256")):
        raise ValueError("Stage60 preregistration or Stage58 evidence is invalid")
    if sha(Path(prereg["supersedes"]["stage59_preregistration"]["path"])) != prereg[
            "supersedes"]["stage59_preregistration"]["sha256"]:
        raise ValueError("Stage59 preregistration changed after Stage60 was frozen")
    if sha(Path(prereg["supersedes"]["stage59_analysis"]["path"])) != prereg[
            "supersedes"]["stage59_analysis"]["sha256"]:
        raise ValueError("Stage59 analysis changed after Stage60 was frozen")
    for relative, expected in prereg["source_code_sha256"].items():
        if sha(ROOT / relative) != expected:
            raise ValueError(f"Stage60 source changed after preregistration: {relative}")
    for entry in prereg["frozen_inputs"].values():
        if sha(Path(entry["path"])) != entry["sha256"]:
            raise ValueError(f"Stage60 frozen input changed: {entry['path']}")
    policy = prereg["regime_policy"]
    if (policy.get("volatility_scheme") != EXPECTED_SCHEME
            or policy.get("volatility_ratio_upper_closed_cuts") != EXPECTED_CUTS):
        raise ValueError("Stage60 fixed regime policy differs from expected thresholds")

    rows = []
    for (index, day), frozen in zip(CASES, prereg["windows"], strict=True):
        if (frozen["window_id"] != index or frozen["start_date_utc"] != day
                or sha(Path(frozen["config"]["path"])) != frozen["config"]["sha256"]):
            raise ValueError(f"Stage60 frozen window identity changed: {day}")
        config = load(Path(frozen["config"]["path"]))
        if config.get("evaluation") is not None:
            raise ValueError(f"Stage60 must remain Train-only: {day}")
        run = Path(config["output"])
        learning_path = run / "R1-learning.json"
        learning = load(learning_path)
        complete = load(run / "COMPLETED.json")
        bundle = load(run / "bundle.json")
        diagnostic = learning["blocks"][0]["diagnostics"]
        samples = int(diagnostic["samples"])
        if (learning.get("schema") != "rles.r1-learning.v4"
                or learning.get("final_steps") != frozen["steps"]
                or samples != frozen["steps"]
                or diagnostic["unique_decision_timestamps"] != samples
                or diagnostic["episodes_completed"] != frozen["episodes"]
                or diagnostic["episode_window_overlap_pairs"] != 0
                or complete.get("report_sha256") is not None):
            raise ValueError(f"Stage60 learning/evaluation invariant failed: {day}")
        learning_sha = sha(learning_path)
        marker = learning_path.with_suffix(learning_path.suffix + ".sha256")
        if marker.read_text(encoding="ascii") != learning_sha:
            raise ValueError(f"Stage60 learning checksum mismatch: {day}")
        if bundle["actor"]["training_diagnostics_hash"] != learning_sha:
            raise ValueError(f"Stage60 bundle does not bind diagnostics: {day}")
        if sum(item[1] for item in diagnostic["position_state_counts"]) != samples:
            raise ValueError(f"Stage60 position counts do not reconcile: {day}")
        if any(item[1] != samples for item in diagnostic["market_feature_summaries"]):
            raise ValueError(f"Stage60 feature counts do not reconcile: {day}")
        bins = diagnostic["market_regime_economic_bins"]
        volatility = [row for row in bins if row[0] == "volatility"]
        if (len(volatility) != 4
                or any(row[2] != EXPECTED_SCHEME for row in volatility)
                or [row[4:6] for row in volatility]
                != [[None, EXPECTED_CUTS[0]], [EXPECTED_CUTS[0], EXPECTED_CUTS[1]],
                    [EXPECTED_CUTS[1], EXPECTED_CUTS[2]], [EXPECTED_CUTS[2], None]]
                or sum(row[6] for row in volatility) != samples):
            raise ValueError(f"Stage60 volatility bins do not reconcile: {day}")
        for family in ("trend", "funding"):
            family_bins = [row for row in bins if row[0] == family]
            if family_bins and sum(row[6] for row in family_bins) != samples:
                raise ValueError(f"Stage60 {family} bins do not reconcile: {day}")

        previous_path = Path(load(STAGE58 / f"window-{index:02d}-{day}" / "config.json")["output"])
        previous_learning = load(previous_path / "R1-learning.json")
        previous_diag = previous_learning["blocks"][0]["diagnostics"]
        for key in ("samples", "episodes_completed", "r_econ_sum", "r_train_sum",
                    "r_shape_sum", "requested_action_economic_bins"):
            if diagnostic.get(key) != previous_diag.get(key):
                raise ValueError(f"Stage60 diagnostic-only replay changed {key}: {day}")
        stage59_learning_path = STAGE59 / f"window-{index:02d}-{day}" / "run" / "R1-learning.json"
        stage59_learning = load(stage59_learning_path)
        stage59_diag = stage59_learning["blocks"][0]["diagnostics"]
        if (diagnostic["position_state_counts"] != stage59_diag["position_state_counts"]
                or diagnostic["market_feature_summaries"] != stage59_diag["market_feature_summaries"]):
            raise ValueError(f"Stage60 changed non-regime training diagnostics: {day}")
        rows.append({
            "window_id": index, "start_date_utc": day,
            "learning_sha256": learning_sha,
            "samples": samples,
            "unique_decision_timestamps": diagnostic["unique_decision_timestamps"],
            "episode_window_overlap_pairs": diagnostic["episode_window_overlap_pairs"],
            "position_state_counts": diagnostic["position_state_counts"],
            "market_feature_summaries": diagnostic["market_feature_summaries"],
            "market_regime_economic_bins": bins,
            "fixed_volatility_bins_match_policy": True,
            "core_training_metrics_match_stage58": True,
            "non_regime_diagnostics_match_stage59": True,
        })

    total = sum(row["samples"] for row in rows)
    return {
        "schema": "rles.stage60-fixed-regime-analysis.v1",
        "status": "passed_train_only_fixed_regime_diagnostics",
        "preregistration_sha256": sha(PREREG_PATH),
        "evaluation_queries": prereg["evaluation_queries"],
        "fixed_volatility_policy": {
            "scheme": EXPECTED_SCHEME,
            "feature": "volatility_5m",
            "ratio_upper_closed_cuts": EXPECTED_CUTS,
            "fitted_from_evaluation_data": False,
            "used_by_policy": False,
        },
        "totals": {
            "environment_interaction_steps": total,
            "unique_decision_timestamps": sum(row["unique_decision_timestamps"] for row in rows),
            "episode_window_overlap_pairs": sum(row["episode_window_overlap_pairs"] for row in rows),
            "core_training_metrics_match_stage58": all(
                row["core_training_metrics_match_stage58"] for row in rows
            ),
            "fixed_boundaries_identical_across_windows": True,
        },
        "windows": rows,
        "limitations": [
            "all windows are previously reused Train data and are not independent evidence",
            "fixed round ratio boundaries are descriptive engineering scales, not calibrated or predictive regimes",
            "costs and trading rules remain scenario-only",
            "no Validation, Test, Holdout, or outer-2026 query was made",
        ],
    }


if __name__ == "__main__":
    payload = json.dumps(analyze(), ensure_ascii=False, indent=2) + "\n"
    with OUTPUT_PATH.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(payload)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    OUTPUT_PATH.with_suffix(OUTPUT_PATH.suffix + ".sha256").write_text(digest + "\n", encoding="ascii")
    print(json.dumps({"output": str(OUTPUT_PATH), "sha256": digest}, ensure_ascii=False))
