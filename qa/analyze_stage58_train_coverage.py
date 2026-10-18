from __future__ import annotations

import hashlib
import json
from itertools import combinations
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / ".tmp/rl-stage58-train-window-coverage-20261016"
SUMMARY = EVIDENCE / "verification-summary.json"
CASES = ((1, "2024-01-01"), (2, "2024-09-01"),
         (3, "2025-05-01"), (4, "2026-01-01"))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def analyze() -> dict[str, Any]:
    summary = _load(SUMMARY)
    if summary.get("status") != "passed" or len(summary.get("runs", [])) != len(CASES):
        raise ValueError("Stage58 source verification is incomplete")
    rows: list[dict[str, Any]] = []
    for (window_id, day), source in zip(CASES, summary["runs"], strict=True):
        case = EVIDENCE / f"window-{window_id:02d}-{day}"
        config_path = case / "config.json"
        prereg_path = case / "preregistration.json"
        config = _load(config_path)
        prereg = _load(prereg_path)
        if (_sha(config_path) != source["config_sha256"]
                or _sha(prereg_path) != source["preregistration_sha256"]):
            raise ValueError(f"frozen Stage58 input hash changed: {day}")
        frozen_config = prereg.get("sources", {}).get("frozen_config", {})
        if frozen_config.get("sha256") != source["config_sha256"]:
            raise ValueError(f"preregistration does not bind the frozen config: {day}")
        learning_path = Path(config["output"]) / "R1-learning.json"
        if _sha(learning_path) != source["learning_sha256"]:
            raise ValueError(f"frozen learning evidence hash changed: {day}")
        learning = _load(learning_path)
        if learning.get("schema") != "rles.r1-learning.v2":
            raise ValueError(f"unexpected legacy learning schema: {day}")
        diagnostics = learning["blocks"][0]["diagnostics"]
        samples = int(diagnostics["samples"])
        unique = int(diagnostics["unique_decision_timestamps"])
        start, end = int(diagnostics["decision_start_ns"]), int(diagnostics["decision_end_ns"])
        episodes = int(diagnostics["episodes_completed"])
        if samples != int(source["steps"]) or unique != samples or not start <= end:
            raise ValueError(f"Stage58 sample coverage invariant failed: {day}")
        if episodes != 1 or int(config["rl_episode_steps"]) != samples:
            raise ValueError(f"Stage58 episode boundary invariant failed: {day}")
        rows.append({
            "window_id": window_id,
            "start_date_utc": day,
            "samples": samples,
            "unique_decision_timestamps": unique,
            "within_window_repeated_samples": samples - unique,
            "decision_start_ns": start,
            "decision_end_ns": end,
            "episodes_completed": episodes,
            "episode_window_overlap_pairs": 0,
            "position_state_coverage": "not_recorded_in_legacy_schema",
            "market_regime_coverage": "not_recorded_in_legacy_schema",
            "regime_conditioned_economic_contribution": "not_recorded_in_legacy_schema",
        })

    overlap_pairs = []
    for left, right in combinations(rows, 2):
        if max(left["decision_start_ns"], right["decision_start_ns"]) <= min(
            left["decision_end_ns"], right["decision_end_ns"],
        ):
            overlap_pairs.append([left["window_id"], right["window_id"]])
    if overlap_pairs:
        raise ValueError("the frozen Stage58 decision timestamp ranges overlap")
    total_samples = sum(row["samples"] for row in rows)
    total_unique = sum(row["unique_decision_timestamps"] for row in rows)
    return {
        "schema": "rles.stage58-train-coverage-analysis.v1",
        "status": "partial",
        "source_verification_summary_sha256": _sha(SUMMARY),
        "evaluation_queries": summary["evaluation_queries"],
        "totals": {
            "environment_interaction_steps": total_samples,
            "unique_decision_timestamps": total_unique,
            "repeated_samples_within_or_between_windows": total_samples - total_unique,
            "unique_market_coverage_fraction": total_unique / total_samples,
            "completed_episodes": sum(row["episodes_completed"] for row in rows),
            "cross_window_episode_overlap_pairs": len(overlap_pairs),
        },
        "windows": rows,
        "cross_window_overlap_pairs": overlap_pairs,
        "limitations": [
            "Exact timestamp union follows from each verified window having unique_decision_timestamps == samples and pairwise-disjoint observed start/end intervals.",
            "All four windows have prior Train research reuse; unique timestamps describe this run's coverage, not fresh independent market evidence.",
            "The v2 learning files contain no per-step position state or market-feature/regime samples, so state coverage and conditioned reward contribution cannot be reconstructed without rerunning or reading extra market data.",
            "Rewards are Train diagnostics under scenario costs, not held-out/account-level profitability evidence.",
        ],
    }


if __name__ == "__main__":
    output = ROOT / "docs/audit_evidence/20261017/stage58_train_coverage_analysis.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(analyze(), ensure_ascii=False, indent=2) + "\n"
    with output.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(payload)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    output.with_suffix(output.suffix + ".sha256").write_text(digest + "\n", encoding="ascii")
    print(json.dumps({"output": str(output), "sha256": digest}, ensure_ascii=False))
