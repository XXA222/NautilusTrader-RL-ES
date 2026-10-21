from __future__ import annotations

from itertools import combinations
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
STAGE58 = ROOT / ".tmp/rl-stage58-train-window-coverage-20261016"
PREREG_PATH = ROOT / "docs/audit_evidence/20261017/stage59_train_diagnostics_revision2/preregistration.json"
CASES = ((1, "2024-01-01"), (2, "2024-09-01"),
         (3, "2025-05-01"), (4, "2026-01-01"))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _core_parity(new: dict[str, Any], old: dict[str, Any], window: str) -> None:
    for key in ("samples", "episodes_completed", "r_econ_sum", "r_train_sum",
                "r_shape_sum", "requested_action_economic_bins"):
        if new.get(key) != old.get(key):
            raise ValueError(f"instrumented Train replay changed {key}: {window}")


def analyze() -> dict[str, Any]:
    prereg = load(PREREG_PATH)
    summary_path = STAGE58 / "verification-summary.json"
    stage58 = load(summary_path)
    if (prereg.get("status") != "frozen_before_diagnostic_replay"
            or prereg.get("evaluation_queries") != {
                "validation": 0, "test": 0, "holdout": 0, "outer_2026": 0,
            }
            or stage58.get("status") != "passed"):
        raise ValueError("Stage59 preregistration or Stage58 source evidence is invalid")
    if sha(summary_path) != prereg["source_stage58_verification_sha256"]:
        raise ValueError("Stage58 verification summary changed after preregistration")
    if sha(PREREG_PATH.parent.parent / "stage59_train_diagnostics" / "preregistration.json") != prereg["supersedes"]["sha256"]:
        raise ValueError("Stage59 superseded registration hash mismatch")
    for relative, expected in prereg["source_code_sha256"].items():
        if sha(ROOT / relative) != expected:
            raise ValueError(f"Stage59 source code changed after preregistration: {relative}")
    for entry in prereg["data_and_base_config_sha256"].values():
        if sha(Path(entry["path"])) != entry["sha256"]:
            raise ValueError("Stage59 frozen data/base config hash changed")

    rows = []
    for (index, day), frozen in zip(CASES, prereg["windows"], strict=True):
        name = f"window-{index:02d}-{day}"
        if (frozen["window_id"] != index or frozen["start_date_utc"] != day
                or sha(Path(frozen["config_path"])) != frozen["config_sha256"]):
            raise ValueError(f"Stage59 window preregistration changed: {name}")
        config = load(Path(frozen["config_path"]))
        if config.get("evaluation") is not None:
            raise ValueError(f"Stage59 must not declare an evaluation split: {name}")
        run = Path(config["output"])
        learning_path = run / "R1-learning.json"
        learning = load(learning_path)
        complete = load(run / "COMPLETED.json")
        bundle = load(run / "bundle.json")
        diagnostic = learning["blocks"][0]["diagnostics"]
        samples = int(diagnostic["samples"])
        unique = int(diagnostic["unique_decision_timestamps"])
        if (learning.get("schema") != "rles.r1-learning.v3"
                or learning.get("final_steps") != samples or samples != frozen["steps"]
                or unique != samples or diagnostic["episodes_completed"] != frozen["episodes"]
                or diagnostic["episode_window_overlap_pairs"] != 0
                or complete.get("report_sha256") is not None):
            raise ValueError(f"Stage59 learning/evaluation invariant failed: {name}")
        learning_sha = sha(learning_path)
        marker = learning_path.with_suffix(learning_path.suffix + ".sha256")
        if marker.read_text(encoding="ascii") != learning_sha:
            raise ValueError(f"Stage59 learning diagnostic SHA marker mismatch: {name}")
        if bundle["actor"]["training_diagnostics_hash"] != learning_sha:
            raise ValueError(f"Stage59 bundle is not bound to learning diagnostics: {name}")
        if sum(item[1] for item in diagnostic["position_state_counts"]) != samples:
            raise ValueError(f"Stage59 position state counts do not reconcile: {name}")
        if any(item[1] != samples for item in diagnostic["market_feature_summaries"]):
            raise ValueError(f"Stage59 market feature counts do not reconcile: {name}")
        regime_bins = diagnostic["market_regime_economic_bins"]
        for family in {item[0] for item in regime_bins}:
            if sum(item[6] for item in regime_bins if item[0] == family) != samples:
                raise ValueError(f"Stage59 {family} regime coverage does not reconcile: {name}")

        prior_record = next(item for item in stage58["runs"] if item["window_id"] == index)
        prior_learning = load(Path(load(STAGE58 / name / "config.json")["output"]) / "R1-learning.json")
        prior_diag = prior_learning["blocks"][0]["diagnostics"]
        _core_parity(diagnostic, prior_diag, name)
        rows.append({
            "window_id": index, "start_date_utc": day,
            "config_sha256": frozen["config_sha256"],
            "learning_sha256": learning_sha,
            "samples": samples, "unique_decision_timestamps": unique,
            "decision_timestamp_revisit_fraction": diagnostic["decision_timestamp_revisit_fraction"],
            "decision_start_ns": diagnostic["decision_start_ns"],
            "decision_end_ns": diagnostic["decision_end_ns"],
            "episodes_completed": diagnostic["episodes_completed"],
            "episode_window_overlap_pairs": diagnostic["episode_window_overlap_pairs"],
            "position_state_counts": diagnostic["position_state_counts"],
            "market_feature_summaries": diagnostic["market_feature_summaries"],
            "market_regime_economic_bins": regime_bins,
            "r_econ_sum": diagnostic["r_econ_sum"],
            "stage58_r_econ_sum": prior_diag["r_econ_sum"],
            "core_train_economics_and_action_bins_match_stage58": True,
            "bundle_learning_diagnostics_sha256_match": True,
            "evaluation_report_sha256": complete.get("report_sha256"),
            "stage58_record_sha256": prior_record["learning_sha256"],
        })

    overlapping_pairs = [[left["window_id"], right["window_id"]]
                         for left, right in combinations(rows, 2)
                         if max(left["decision_start_ns"], right["decision_start_ns"])
                         <= min(left["decision_end_ns"], right["decision_end_ns"])]
    if overlapping_pairs:
        raise ValueError("Stage59 observed Train timestamp intervals overlap")
    total_samples = sum(row["samples"] for row in rows)
    total_unique = sum(row["unique_decision_timestamps"] for row in rows)
    return {
        "schema": "rles.stage59-train-diagnostics-analysis.v1",
        "status": "passed_train_only_diagnostics",
        "preregistration_sha256": sha(PREREG_PATH),
        "stage58_verification_sha256": sha(summary_path),
        "evaluation_queries": prereg["evaluation_queries"],
        "totals": {
            "environment_interaction_steps": total_samples,
            "unique_decision_timestamps": total_unique,
            "repeated_samples_within_or_between_windows": total_samples - total_unique,
            "unique_market_coverage_fraction": total_unique / total_samples,
            "episodes_completed": sum(row["episodes_completed"] for row in rows),
            "cross_window_overlap_pairs": len(overlapping_pairs),
            "core_train_reward_and_action_bins_match_stage58": all(
                row["core_train_economics_and_action_bins_match_stage58"] for row in rows
            ),
        },
        "windows": rows,
        "cross_window_overlap_pairs": overlapping_pairs,
        "limitations": [
            "All windows were previously reused Train dates; this diagnostic is not fresh independent evidence.",
            "Volatility cutpoints are computed from each Train learn block and are descriptive; they are not comparable fixed boundaries across windows or a Validation rule.",
            "Trend and funding regimes use signed feature values; zero is its own bucket. Empty funding coverage remains visible as an empty feature family.",
            "r_econ uses frozen Stage58 scenario costs/rules and does not establish account-level profitability.",
            "No Validation, Test, Holdout, or outer-2026 queries were made.",
        ],
    }


if __name__ == "__main__":
    output = ROOT / "docs/audit_evidence/20261017/stage59_train_diagnostics_analysis.json"
    payload = json.dumps(analyze(), ensure_ascii=False, indent=2) + "\n"
    with output.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(payload)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    output.with_suffix(output.suffix + ".sha256").write_text(digest + "\n", encoding="ascii")
    print(json.dumps({"output": str(output), "sha256": digest}, ensure_ascii=False))
