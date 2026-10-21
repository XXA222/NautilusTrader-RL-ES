"""Verify E1/R1/E2 native stage reports against a frozen E1 actor artifact.

This analyzer only reads a bundle, its E1 actor artifact, a preregistration,
and already-produced native reports. It never opens market data or runs a replay.
Each report file is an envelope with ``schema``, ``stage``, ``mode``,
``actor_sha256``, ``preregistration_sha256``, ``pairing_identity`` and
``native_report`` fields. The latter uses the native report shape emitted by
``rles_orchestrator.application.rollout``.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Any, TypeGuard, cast


ROOT = Path(__file__).resolve().parents[1]
STAGES = ("E1", "R1", "E2")
METRICS = (
    "net_return", "max_drawdown", "sharpe", "sortino", "turnover_ratio",
    "total_fee", "total_funding", "total_slippage_cost",
    "gross_return_before_reported_costs",
)
HEX = frozenset("0123456789abcdef")


def _sha(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read JSON object: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _is_sha(value: object) -> TypeGuard[str]:
    return (isinstance(value, str) and len(value) == 64
            and all(char in HEX for char in value))


def _number(value: object, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a finite number")
    try:
        result = float(cast(Any, value))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be a finite number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{label} must be a finite number")
    return result


def verify_e1_actor(bundle_root: Path) -> dict[str, Any]:
    """Verify bundle index, sidecar, E1 lineage, actor schema and genome hash."""
    from rles_es_evotorch.actor import ActorGenomeSpec, genome_hash
    import numpy as np

    root = bundle_root.resolve()
    bundle_path = root / "bundle.json"
    marker_path = bundle_path.with_suffix(bundle_path.suffix + ".sha256")
    if not marker_path.is_file() or marker_path.read_text(encoding="ascii").strip() != _sha(bundle_path):
        raise ValueError("bundle SHA-256 receipt mismatch")
    bundle = _json(bundle_path)
    actor_metadata = bundle.get("actor")
    r1_actor_hash = actor_metadata.get("actor_hash") if isinstance(actor_metadata, dict) else None
    if not _is_sha(r1_actor_hash):
        raise ValueError("bundle R1 actor identity is missing")
    index = bundle.get("stage_actor_artifacts")
    if not isinstance(index, dict) or not isinstance(index.get("E1"), dict):
        raise ValueError("bundle does not index a selected E1 actor")
    entry = index["E1"]
    relative = entry.get("path")
    if not isinstance(relative, str) or Path(relative).name != relative:
        raise ValueError("E1 actor path must be a bundle-local filename")
    artifact_path = root / relative
    if not artifact_path.is_file() or not _is_sha(entry.get("sha256")):
        raise ValueError("E1 actor artifact or indexed digest is missing")
    marker = artifact_path.with_suffix(artifact_path.suffix + ".sha256")
    actual_sha = _sha(artifact_path)
    if (actual_sha != entry["sha256"] or not marker.is_file()
            or marker.read_text(encoding="ascii").strip() != actual_sha):
        raise ValueError("E1 actor artifact SHA-256 receipt mismatch")
    artifact = _json(artifact_path)
    if artifact.get("schema") != "rles.selected-stage-actor.v1" or artifact.get("stage") != "E1":
        raise ValueError("unsupported selected-stage actor schema")
    actor_hash = artifact.get("actor_hash")
    lane_hash = artifact.get("lane_evidence_hash")
    if not _is_sha(actor_hash) or not _is_sha(lane_hash):
        raise ValueError("E1 actor or lane evidence identity is malformed")
    if entry.get("actor_hash") != actor_hash or entry.get("lane_evidence_hash") != lane_hash:
        raise ValueError("E1 artifact identity differs from the bundle index")
    spec_data = artifact.get("actor_spec")
    genome_data = artifact.get("genome")
    if not isinstance(spec_data, dict) or not isinstance(genome_data, list):
        raise ValueError("E1 actor schema or genome is missing")
    spec_values = dict(spec_data)
    for name in ("observation_names", "hidden_sizes"):
        if isinstance(spec_values.get(name), list):
            spec_values[name] = tuple(spec_values[name])
    if isinstance(spec_values.get("input_scale"), list):
        spec_values["input_scale"] = tuple(spec_values["input_scale"])
    try:
        spec = ActorGenomeSpec(**spec_values)
        genome = np.asarray(genome_data, dtype=np.float32)
        computed_hash = genome_hash(spec, genome)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("E1 actor spec or genome is invalid") from exc
    if computed_hash != actor_hash:
        raise ValueError("E1 genome hash does not match its actor identity")
    lineage = bundle.get("lineage")
    if not isinstance(lineage, list):
        raise ValueError("bundle stage lineage is missing")
    e1_lineage = [entry for entry in lineage if isinstance(entry, dict) and entry.get("stage") == "E1"]
    if len(e1_lineage) != 1 or e1_lineage[0].get("actor_hash") != actor_hash:
        raise ValueError("E1 actor identity does not match bundle lineage")
    metadata = e1_lineage[0].get("metadata")
    if not isinstance(metadata, dict) or metadata.get("lane_evidence_hash") != lane_hash:
        raise ValueError("E1 lane evidence identity does not match bundle lineage")
    return {
        "bundle_sha256": _sha(bundle_path),
        "artifact_path": relative,
        "artifact_sha256": actual_sha,
        "actor_sha256": actor_hash,
        "r1_actor_sha256": r1_actor_hash,
        "lane_evidence_sha256": lane_hash,
        "actor_spec": spec_data,
    }


def _verify_native_report(report: dict[str, Any], label: str) -> None:
    if report.get("costs_already_in_native_equity") is not True:
        raise ValueError(f"{label}: native equity cost inclusion is not confirmed")
    if report.get("terminal_flat") is not True:
        raise ValueError(f"{label}: terminal account is not confirmed flat")
    reconciliation = report.get("equity_reconciliation")
    if not isinstance(reconciliation, dict) or not isinstance(reconciliation.get("rows"), list):
        raise ValueError(f"{label}: event-level equity reconciliation is missing")
    if not reconciliation["rows"]:
        raise ValueError(f"{label}: event-level equity reconciliation is empty")
    for row in reconciliation["rows"]:
        if not isinstance(row, dict):
            raise ValueError(f"{label}: malformed equity reconciliation row")
        residual = abs(_number(row.get("residual"), f"{label} residual"))
        tolerance = _number(row.get("tolerance"), f"{label} tolerance")
        if tolerance < 0 or residual > tolerance:
            raise ValueError(f"{label}: equity residual exceeds tolerance at {row.get('ts_ns')}")
    metrics = report.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError(f"{label}: native economic metrics are missing")
    for metric in METRICS:
        _number(metrics.get(metric), f"{label} metric {metric}")
    rows = report.get("account_rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{label}: native account rows are missing")
    timestamps: list[int] = []
    for row in rows:
        if not isinstance(row, dict) or type(row.get("ts_ns")) is not int:
            raise ValueError(f"{label}: malformed native account row")
        timestamps.append(row["ts_ns"])
        _number(row.get("equity"), f"{label} equity")
    if any(left >= right for left, right in zip(timestamps, timestamps[1:])):
        raise ValueError(f"{label}: native account timestamps are not strictly increasing")
    diagnostics = report.get("execution_decision_diagnostics")
    if not isinstance(diagnostics, dict):
        raise ValueError(f"{label}: execution decision diagnostics are missing")
    for key in ("decision_count", "submit_decision_count", "native_order_count"):
        if type(diagnostics.get(key)) is not int or diagnostics[key] < 0:
            raise ValueError(f"{label}: invalid execution diagnostic {key}")
    if not isinstance(report.get("fills"), list):
        raise ValueError(f"{label}: native fill list is missing")


def summarize_stage_chain(mode: str, envelopes: dict[str, dict[str, Any]], *,
                          e1_actor_sha256: str,
                          preregistration_sha256: str,
                          expected_pairing_identity: dict[str, Any] | None = None,
                          expected_actor_sha256_by_stage: dict[str, str] | None = None,
                          expected_soft_multipliers_by_stage: dict[str, dict[str, Any]] | None = None
                          ) -> dict[str, Any]:
    """Calculate E1→R1, R1→E2 and E1→E2 paired deltas for one mode."""
    if tuple(envelopes.keys()) != STAGES and set(envelopes) != set(STAGES):
        raise ValueError(f"{mode}: reports must contain exactly E1, R1 and E2")
    normalized: dict[str, dict[str, Any]] = {}
    pairing: dict[str, Any] | None = None
    actor_hashes: dict[str, str] = {}
    timestamp_rows: dict[str, list[dict[str, Any]]] = {}
    for stage in STAGES:
        envelope = envelopes[stage]
        if envelope.get("schema") != "rles.m13-stage-native-report.v1":
            raise ValueError(f"{mode}/{stage}: unsupported stage report schema")
        if envelope.get("stage") != stage or envelope.get("mode") != mode:
            raise ValueError(f"{mode}/{stage}: report stage or mode identity mismatch")
        if envelope.get("preregistration_sha256") != preregistration_sha256:
            raise ValueError(f"{mode}/{stage}: report is not bound to the frozen preregistration")
        actor_hash = envelope.get("actor_sha256")
        if not _is_sha(actor_hash):
            raise ValueError(f"{mode}/{stage}: actor identity is malformed")
        actor_hashes[stage] = actor_hash
        if stage == "E1" and actor_hash != e1_actor_sha256:
            raise ValueError("E1 native report actor does not match the selected E1 artifact")
        identity = envelope.get("pairing_identity")
        required_identity_keys = {
            "market_sha256", "rules_sha256", "window", "seed", "starting_balance",
            "environment_sha256", "execution_sha256", "hard_risk_sha256", "scaler_sha256",
        }
        if not isinstance(identity, dict) or not required_identity_keys.issubset(identity):
            raise ValueError(f"{mode}/{stage}: frozen pairing identity is missing")
        for identity_key in ("market_sha256", "rules_sha256", "environment_sha256",
                             "execution_sha256", "hard_risk_sha256", "scaler_sha256"):
            if not _is_sha(identity[identity_key]):
                raise ValueError(f"{mode}/{stage}: invalid frozen identity {identity_key}")
        if (not isinstance(identity["window"], dict)
                or type(identity["window"].get("start_ns")) is not int
                or type(identity["window"].get("end_ns")) is not int
                or identity["window"]["start_ns"] >= identity["window"]["end_ns"]
                or type(identity["seed"]) is not int):
            raise ValueError(f"{mode}/{stage}: invalid frozen window or seed")
        if expected_pairing_identity is not None and identity != expected_pairing_identity:
            raise ValueError(f"{mode}/{stage}: pairing identity differs from preregistration")
        if pairing is None:
            pairing = identity
        elif identity != pairing:
            raise ValueError(f"{mode}: stage reports do not share the same frozen pairing identity")
        if expected_actor_sha256_by_stage is not None:
            if actor_hash != expected_actor_sha256_by_stage.get(stage):
                raise ValueError(f"{mode}/{stage}: actor identity differs from preregistration")
        soft_multipliers = envelope.get("soft_multipliers")
        if not isinstance(soft_multipliers, dict) or not soft_multipliers:
            raise ValueError(f"{mode}/{stage}: frozen SoftRisk parameters are missing")
        for key, value in soft_multipliers.items():
            _number(value, f"{mode}/{stage} SoftRisk parameter {key}")
        if (expected_soft_multipliers_by_stage is not None
                and soft_multipliers != expected_soft_multipliers_by_stage.get(stage)):
            raise ValueError(f"{mode}/{stage}: SoftRisk parameters differ from preregistration")
        native_report = envelope.get("native_report")
        if not isinstance(native_report, dict):
            raise ValueError(f"{mode}/{stage}: native report payload is missing")
        _verify_native_report(native_report, f"{mode}/{stage}")
        normalized[stage] = native_report
        timestamp_rows[stage] = native_report["account_rows"]
    if actor_hashes["R1"] != actor_hashes["E2"]:
        raise ValueError(f"{mode}: R1 and E2 must use the same frozen Actor")
    if actor_hashes["E1"] != e1_actor_sha256:
        raise ValueError("E1 native report actor does not match the selected E1 artifact")
    decision_counts = {
        stage: normalized[stage]["execution_decision_diagnostics"]["decision_count"]
        for stage in STAGES
    }
    if len(set(decision_counts.values())) != 1:
        raise ValueError(f"{mode}: stages do not cover the same number of policy decisions")
    timestamps = [row["ts_ns"] for row in timestamp_rows["E1"]]
    if not timestamps or any(timestamp_rows[stage] and
                             [row["ts_ns"] for row in timestamp_rows[stage]] != timestamps
                             for stage in STAGES[1:]):
        raise ValueError(f"{mode}: stage native account timestamps are not strictly paired")
    initials = {stage: _number(timestamp_rows[stage][0]["equity"], f"{mode}/{stage} initial equity")
                for stage in STAGES}
    if len(set(initials.values())) != 1:
        raise ValueError(f"{mode}: stage initial equities differ")

    pairs = (("E1", "R1"), ("R1", "E2"), ("E1", "E2"))
    stage_deltas: dict[str, Any] = {}
    for before, after in pairs:
        before_report, after_report = normalized[before], normalized[after]
        before_rows, after_rows = timestamp_rows[before], timestamp_rows[after]
        b_final = _number(before_rows[-1]["equity"], f"{mode}/{before} final equity")
        a_final = _number(after_rows[-1]["equity"], f"{mode}/{after} final equity")
        before_metrics, after_metrics = before_report["metrics"], after_report["metrics"]
        before_diag = before_report["execution_decision_diagnostics"]
        after_diag = after_report["execution_decision_diagnostics"]
        stage_deltas[f"{before}_to_{after}"] = {
            "native_final_equity_delta": a_final - b_final,
            "metric_deltas": {
                key: _number(after_metrics[key], f"{mode}/{after}/{key}")
                - _number(before_metrics[key], f"{mode}/{before}/{key}")
                for key in METRICS
            },
            "execution_deltas": {
                "fills": len(after_report["fills"]) - len(before_report["fills"]),
                "submit_decisions": after_diag["submit_decision_count"]
                - before_diag["submit_decision_count"],
                "native_orders": after_diag["native_order_count"]
                - before_diag["native_order_count"],
                "hard_risk_interventions": int(after_report.get("risk_hard_interventions", 0))
                - int(before_report.get("risk_hard_interventions", 0)),
            },
        }
    return {
        "mode": mode,
        "paired_account_rows": len(timestamps),
        "policy_decisions_per_stage": decision_counts,
        "same_native_timestamps": True,
        "initial_equity": initials["E1"],
        "actor_sha256": actor_hashes,
        "pairing_identity": pairing,
        "stage_deltas": stage_deltas,
        "native_metrics": {stage: normalized[stage]["metrics"] for stage in STAGES},
        "equity_reconciliation": {
            stage: {
                "rows": len(normalized[stage]["equity_reconciliation"]["rows"]),
                "max_abs_residual": normalized[stage]["equity_reconciliation"].get("max_abs_residual"),
                "terminal_flat": normalized[stage]["terminal_flat"],
            }
            for stage in STAGES
        },
    }


def analyze(bundle_root: Path, registration_path: Path, reports_root: Path) -> dict[str, Any]:
    """Verify a frozen Train-inner registration and summarize existing reports."""
    registration = registration_path.resolve()
    marker = registration.with_suffix(registration.suffix + ".sha256")
    if not marker.is_file() or marker.read_text(encoding="ascii").strip() != _sha(registration):
        raise ValueError("M1.3 v2 preregistration SHA receipt mismatch")
    frozen = _json(registration)
    formal_queries = frozen.get("formal_evaluation_queries")
    no_formal_queries = (
        isinstance(formal_queries, dict)
        and set(formal_queries) == {"validation", "test", "holdout"}
        and all(type(value) is int and value == 0 for value in formal_queries.values())
    )
    if (frozen.get("schema") != "rles.m13-stage-contribution-preregistration.v2"
            or frozen.get("status") != "frozen_before_native_replay"
            or frozen.get("split") != "train_inner"
            or frozen.get("stage_order") != list(STAGES)
            or frozen.get("no_retraining_or_selection") is not True
            or not no_formal_queries):
        raise ValueError("M1.3 v2 preregistration is incomplete or outside the Train-inner scope")
    root = bundle_root.resolve()
    output = reports_root.resolve()
    if Path(str(frozen.get("bundle_root"))).resolve() != root:
        raise ValueError("bundle path differs from M1.3 v2 preregistration")
    if Path(str(frozen.get("output_root"))).resolve() != output:
        raise ValueError("report path differs from M1.3 v2 preregistration")
    actor = verify_e1_actor(root)
    if actor["bundle_sha256"] != frozen.get("bundle_sha256"):
        raise ValueError("bundle digest differs from M1.3 v2 preregistration")
    modes = frozen.get("modes")
    if (not isinstance(modes, list) or not modes
            or any(not isinstance(mode, str) or not mode for mode in modes)
            or len(set(modes)) != len(modes)):
        raise ValueError("M1.3 v2 preregistration has no valid modes")
    pairings = frozen.get("pairing_identity_by_mode")
    soft_by_mode = frozen.get("soft_multipliers_by_mode")
    expected_actor_hashes = frozen.get("actor_sha256_by_stage")
    if (not isinstance(pairings, dict) or set(pairings) != set(modes)
            or not isinstance(soft_by_mode, dict) or set(soft_by_mode) != set(modes)
            or not isinstance(expected_actor_hashes, dict)):
        raise ValueError("M1.3 v2 preregistration is missing frozen pairing or stage identities")
    bundle_actor_hashes = {
        "E1": actor["actor_sha256"],
        "R1": actor["r1_actor_sha256"],
        "E2": actor["r1_actor_sha256"],
    }
    if expected_actor_hashes != bundle_actor_hashes:
        raise ValueError("preregistered stage actors differ from the verified bundle lineage")
    bundle = _json(root / "bundle.json")
    selected_soft = bundle.get("soft_multipliers")
    if not isinstance(selected_soft, dict):
        raise ValueError("bundle E2 SoftRisk parameters are missing")
    preregistration_sha = _sha(registration)
    mode_results: dict[str, Any] = {}
    report_hashes: dict[str, str] = {}
    for mode in modes:
        envelopes: dict[str, dict[str, Any]] = {}
        for stage in STAGES:
            path = output / f"{mode}_{stage}.json"
            envelopes[stage] = _json(path)
            report_hashes[path.name] = _sha(path)
        expected_soft = soft_by_mode[mode]
        if (not isinstance(expected_soft, dict) or set(expected_soft) != set(STAGES)
                or expected_soft["E2"] != selected_soft):
            raise ValueError(f"{mode}: preregistered stage SoftRisk parameters differ from bundle")
        if (any(not isinstance(expected_soft[stage], dict) or not expected_soft[stage]
                for stage in STAGES)
                or len({tuple(sorted(expected_soft[stage])) for stage in STAGES}) != 1):
            raise ValueError(f"{mode}: stage SoftRisk parameter schemas differ")
        mode_results[mode] = summarize_stage_chain(
            mode, envelopes, e1_actor_sha256=actor["actor_sha256"],
            preregistration_sha256=preregistration_sha,
            expected_pairing_identity=pairings[mode],
            expected_actor_sha256_by_stage=bundle_actor_hashes,
            expected_soft_multipliers_by_stage=expected_soft,
        )
    return {
        "schema": "rles.m13-stage-contribution-analysis.v2",
        "status": "verified_train_inner_stage_diagnostic",
        "analysis_only": True,
        "market_data_opened": False,
        "preregistration_sha256": preregistration_sha,
        "bundle": actor,
        "report_sha256": report_hashes,
        "stage_order": list(STAGES),
        "formal_evaluation_queries": frozen["formal_evaluation_queries"],
        "interpretation_limit": frozen.get("interpretation_limit"),
        "modes": mode_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--reports-root", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()
    result_path = args.result if args.result.is_absolute() else ROOT / args.result
    result = analyze(args.bundle, args.registration, args.reports_root)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
                           encoding="utf-8")
    result_path.with_suffix(result_path.suffix + ".sha256").write_text(_sha(result_path), encoding="ascii")
    print(json.dumps({"status": result["status"], "result": str(result_path),
                      "modes": list(result["modes"])}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
