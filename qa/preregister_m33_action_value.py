from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "configs/train_signal_baselines_20261016.json"
BASELINE_SUMMARY = ROOT / "docs/audit_evidence/20261016/train_signal_baselines_summary.json"
BASELINE_RUNS = ROOT / ".artifacts/audit-development-20261016/signal-baselines-v3"
OUTPUT = ROOT / "docs/audit_evidence/20261017/m33_action_value"
RUN_OUTPUT = ROOT / ".artifacts/audit-development-20261017/m33-action-value-v1"
CODE_PATHS = (
    "packages/orchestrator/src/rles_orchestrator/application.py",
    "packages/orchestrator/src/rles_orchestrator/research_access.py",
    "packages/gym_bridge/src/rles_gym_bridge/env.py",
    "packages/gym_bridge/src/rles_gym_bridge/historical.py",
    "packages/rl_sb3/src/rles_rl_sb3/provider.py",
    "qa/train_signal_baselines.py",
    "qa/preregister_m33_action_value.py",
    "qa/analyze_m33_action_value.py",
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def main() -> None:
    plan = load(PLAN)
    previous = load(BASELINE_SUMMARY)
    if (plan.get("split") != "train_inner" or plan.get("policies") != [
            "B0_no_trade", "B1_always_long", "B4_always_short",
            "B2_simple_momentum", "B5_simple_reversal", "frozen_rl",
        ] or previous.get("formal_evaluation_queries") != 0
            or previous.get("candidate_gate") != "FAIL_NO_CROSS_WINDOW_BASELINE_ADVANTAGE"
            or previous.get("budget_expansion_allowed") is not False):
        raise ValueError("existing Train-only baseline study is not the frozen M3.3 source")
    for item in previous["runs"]:
        report_path = BASELINE_RUNS / item["report"]
        if sha(report_path) != item["report_sha256"]:
            raise ValueError(f"prior baseline report changed: {report_path}")
    bundle = Path(plan["bundle"])
    if sha(bundle / "bundle.json") != plan["bundle_sha256"]:
        raise ValueError("frozen M3.3 bundle changed")
    source_hashes = {}
    for name, path_key in (("market", "market_path"), ("rules", "rules_path")):
        path = Path(plan["source"][path_key])
        if sha(path) != plan["source"][name + "_sha256"]:
            raise ValueError(f"frozen M3.3 {name} changed")
        source_hashes[name] = {"path": str(path), "sha256": plan["source"][name + "_sha256"]}
    if RUN_OUTPUT.exists():
        raise ValueError(f"M3.3 run output must be new: {RUN_OUTPUT}")
    if OUTPUT.exists():
        raise ValueError(f"M3.3 evidence directory must be new: {OUTPUT}")
    OUTPUT.mkdir(parents=True)
    frozen_plan_path = OUTPUT / "plan.json"
    encoded_plan = json.dumps(plan, ensure_ascii=False, indent=2) + "\n"
    with frozen_plan_path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(encoded_plan)

    prereg = {
        "schema": "rles.m33-action-value-preregistration.v1",
        "status": "frozen_before_paired_replay",
        "objective": "Attribute the already-frozen actor and simple baselines by fixed market regimes, native exposure, execution and modeled costs; no retraining or selection.",
        "scope": "train_inner_development_only_scenario_costs",
        "frozen_plan": {"path": str(frozen_plan_path), "sha256": sha(frozen_plan_path)},
        "prior_baseline_plan": {"path": str(PLAN), "sha256": sha(PLAN)},
        "prior_baseline_summary": {"path": str(BASELINE_SUMMARY), "sha256": sha(BASELINE_SUMMARY)},
        "prior_baseline_reports": len(previous["runs"]),
        "bundle": {"path": str(bundle / "bundle.json"), "sha256": plan["bundle_sha256"]},
        "market_and_rules": source_hashes,
        "source_code_sha256": {relative: sha(ROOT / relative) for relative in CODE_PATHS},
        "replay_output": str(RUN_OUTPUT),
        "windows": plan["windows"],
        "modes": plan["modes"],
        "policies": plan["policies"],
        "seed": plan["seed"],
        "regime_policy": {
            "volatility": {
                "feature": "volatility_5m",
                "scheme": "fixed_absolute_volatility_upper_closed_v1",
                "ratio_upper_closed_cuts": [0.00025, 0.0005, 0.001],
            },
            "trend": {"feature": "return_15m", "scheme": "signed_feature_v1"},
            "funding": {"feature": "last_settled_funding", "scheme": "signed_feature_v1"},
            "thresholds_fit_from_evaluation_data": False,
        },
        "per_decision_metrics": [
            "native_equity_delta", "gross_marked_price_pnl", "fee", "funding",
            "slippage_attribution", "turnover", "plan_action_and_reason", "native_fill_count",
            "requested_and_executed_action", "net_exposure", "market_features",
        ],
        "holding_duration": "contiguous nonzero native net-exposure runs by direction; descriptive proxy, not exchange position identity",
        "paired_uncertainty": {
            "method": "paired nonoverlapping block bootstrap within each window and cost mode",
            "block_minutes": 60,
            "replicates": 5000,
            "seed": 61017,
            "interval": "percentile 95% interval for the full-window cumulative USDT PnL difference",
            "minimum_blocks": 6,
            "interpretation": "within-path dependence sensitivity only; windows are previously reused Train data and do not provide independent validation",
        },
        "evaluation_queries": {"validation": 0, "test": 0, "holdout": 0, "outer_2026": 0},
        "limitations": [
            "three one-day windows are previously reused Train dates and are not independent evidence",
            "Bar execution assumes sufficient simulated depth; historical rules and costs remain scenario-only",
            "zero fees or fees already embedded in native equity must not be subtracted twice",
            "descriptive regime attribution and bootstrap intervals do not establish predictive or causal value",
        ],
    }
    prereg_path = OUTPUT / "preregistration.json"
    encoded = json.dumps(prereg, ensure_ascii=False, indent=2) + "\n"
    with prereg_path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(encoded)
    digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    prereg_path.with_suffix(".json.sha256").write_text(digest + "\n", encoding="ascii")
    print(json.dumps({
        "preregistration": str(prereg_path), "sha256": digest,
        "frozen_plan_sha256": sha(frozen_plan_path),
        "prior_reports_verified": len(previous["runs"]),
        "replays": len(plan["windows"]) * len(plan["modes"]) * len(plan["policies"]),
        "evaluation_queries": prereg["evaluation_queries"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
