"""Run the preregistered Train-only low-capacity forecast diagnostic."""
from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path
import time

import numpy as np

from rles_data.core import CuratedStore, DataError, MINUTE_NS
from rles_data.coverage import CoveragePolicy
from rles_data.features import make_feature_row
from rles_data.io import read_curated
from rles_gym_bridge.historical import historical_feature_spec, load_rules
from rles_orchestrator.application import _file_hash, _verify_file, _write_verified
from rles_orchestrator.research_access import record_development
from rles_orchestrator.wfo_application import _scaler

from profile_native_history_growth_20261016 import available_memory, coverage_for_window


def corr(left: np.ndarray, right: np.ndarray) -> float | None:
    if left.size < 2 or np.std(left) == 0 or np.std(right) == 0:
        return None
    value = float(np.corrcoef(left, right)[0, 1])
    return value if math.isfinite(value) else None


def rank(values: np.ndarray) -> np.ndarray:
    _unique, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    first = np.cumsum(counts) - counts
    mean_ranks = first + (counts - 1) / 2
    return mean_ranks[inverse]


def forecast_metrics(prediction: np.ndarray, target_bp: np.ndarray, cost_bp: float) -> dict:
    count = int(target_bp.size)
    order = np.argsort(prediction, kind="mergesort")
    tail = max(1, count // 10)
    lower, upper = target_bp[order[:tail]], target_bp[order[-tail:]]
    direction = np.sign(prediction)
    active = direction != 0
    selected_gross = float(np.mean(direction[active] * target_bp[active])) if np.any(active) else None
    return {
        "samples": count,
        "pearson_ic": corr(prediction, target_bp),
        "spearman_ic": corr(rank(prediction), rank(target_bp)),
        "direction_accuracy": float(np.mean((prediction > 0) == (target_bp > 0))),
        "positive_target_fraction": float(np.mean(target_bp > 0)),
        "prediction_active_fraction": float(np.mean(active)),
        "mae_bp": float(np.mean(np.abs(prediction - target_bp))),
        "rmse_bp": float(np.sqrt(np.mean(np.square(prediction - target_bp)))),
        "zero_score_rmse_bp": float(np.sqrt(np.mean(np.square(target_bp)))),
        "top_decile_long_gross_bp": float(np.mean(upper)),
        "top_decile_long_after_scenario_cost_bp": float(np.mean(upper) - cost_bp),
        "bottom_decile_short_gross_bp": float(-np.mean(lower)),
        "bottom_decile_short_after_scenario_cost_bp": float(-np.mean(lower) - cost_bp),
        "top_minus_bottom_gross_spread_bp": float(np.mean(upper) - np.mean(lower)),
        "all_signal_selected_gross_bp": selected_gross,
        "all_signal_selected_after_scenario_cost_bp": (
            None if selected_gross is None else selected_gross - cost_bp),
    }


def sample_indices(times: np.ndarray, start_ns: int, end_ns: int, cadence_ns: int) -> np.ndarray:
    grid = np.arange(start_ns, end_ns, cadence_ns, dtype=np.int64)
    indices = np.searchsorted(times, grid, side="left")
    indices = indices[indices < times.size]
    indices = np.unique(indices)
    return indices[(times[indices] >= start_ns) & (times[indices] < end_ns)]


def labels_for(indices: np.ndarray, times: np.ndarray, closes: np.ndarray,
               horizon_minutes: int, max_lag_minutes: int
               ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    horizon_ns = horizon_minutes * MINUTE_NS
    future_indices = np.searchsorted(times, times[indices] + horizon_ns, side="left")
    within = future_indices < times.size
    valid = np.zeros(indices.size, dtype=bool)
    valid[within] = (times[future_indices[within]] - (times[indices[within]] + horizon_ns)
                     <= max_lag_minutes * MINUTE_NS)
    selected, future = indices[valid], future_indices[valid]
    labels_bp = np.log(closes[future] / closes[selected]) * 10_000.0
    return (selected, labels_bp.astype(np.float64),
            (times[future] - times[selected]) / MINUTE_NS, times[future])


def fit_ridge(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray,
              alpha: float) -> tuple[np.ndarray, dict]:
    mean = np.mean(train_x, axis=0)
    std = np.std(train_x, axis=0)
    std = np.where(std > 1e-12, std, 1.0)
    x_train = (train_x - mean) / std
    x_test = (test_x - mean) / std
    x_train = np.column_stack((np.ones(x_train.shape[0]), x_train))
    x_test = np.column_stack((np.ones(x_test.shape[0]), x_test))
    gram = x_train.T @ x_train / x_train.shape[0]
    rhs = x_train.T @ train_y / x_train.shape[0]
    penalty = np.eye(gram.shape[0]) * alpha
    penalty[0, 0] = 0.0
    coefficients = np.linalg.solve(gram + penalty, rhs)
    return x_test @ coefficients, {"feature_count": int(train_x.shape[1]), "ridge_alpha": alpha}


def run(plan_path: Path, output: Path) -> dict:
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if plan.get("split") != "train_inner" or output.exists():
        raise DataError("Train-inner plan and a new output directory are required")
    marker = plan_path.with_name(plan_path.name + ".sha256")
    if not marker.is_file() or marker.read_text(encoding="ascii").strip() != _file_hash(plan_path):
        raise DataError("preregistered forecast plan checksum mismatch")
    bundle_root = Path(plan["bundle"])
    if _verify_file(bundle_root / "bundle.json") != plan["bundle_sha256"]:
        raise DataError("frozen source bundle checksum changed")
    bundle = json.loads((bundle_root / "bundle.json").read_text(encoding="utf-8"))
    if (bundle["preprocessing"]["identities"]["dataset"] != plan["dataset_sha256"]
            or available_memory() < 10 * 1024**3):
        raise DataError("dataset identity mismatch or insufficient start memory")
    if any(fold["source_start_ns"] < bundle["environment"]["start_ns"]
           or fold["source_end_ns"] > bundle["training_end_ns"]
           or fold["train_start_ns"] + 180 * 24 * 60 * MINUTE_NS != fold["test_start_ns"]
           or fold["test_end_ns"] - fold["test_start_ns"] != 30 * 24 * 60 * MINUTE_NS
           for fold in plan["folds"]):
        raise DataError("forecast folds differ from the frozen Train-only protocol")
    source = plan["source"]
    for name in ("market", "rules"):
        if _file_hash(Path(source[name + "_path"])) != source[name + "_sha256"]:
            raise DataError(f"{name} source checksum mismatch")

    output.mkdir(parents=True)
    _write_verified(output / "preregistration.json", {
        "plan_path": str(plan_path.resolve()), "plan_sha256": _file_hash(plan_path), "plan": plan})
    try:
        rules = load_rules(source["rules_path"])
        store = CuratedStore((*read_curated(source["market_path"]).events, *rules))
        if store.snapshot_hash != plan["dataset_sha256"]:
            raise DataError("source market/rules do not match the frozen data identity")
        environment = bundle["environment"]
        instrument = environment["instrument"]
        spec = historical_feature_spec(environment["coverage_policy"]["maximum_age_ns"],
                                       profile=plan["feature_profile"])
        scaler = _scaler(bundle_root)
        if scaler.feature_hash != spec.hash:
            raise DataError("frozen feature/scaler identity mismatch")
        bars = sorted((event for event in store.events if event.kind == "ohlcv_1m"
                       and event.payload["instrument"] == instrument
                       and event.available_at_ns is not None),
                      key=lambda event: int(event.available_at_ns or 0))
        times = np.asarray([int(event.available_at_ns or 0) for event in bars], dtype=np.int64)
        closes = np.asarray([float(event.payload["close"]) for event in bars], dtype=np.float64)
        if times.size < 10 or np.any(np.diff(times) <= 0):
            raise DataError("one-minute bar availability times must be unique and increasing")
        del bars
        gc.collect()

        registry_config = {"environment": environment, "output": str(output), "seed": 0}
        fold_results = []
        for fold in plan["folds"]:
            if available_memory() < 2 * 1024**3:
                raise MemoryError("resource stop floor reached before next predictor fold")
            record_development(registry_config,
                               {"start_ns": fold["source_start_ns"], "end_ns": fold["source_end_ns"]},
                               label=f"low-capacity-forecast-fold-{fold['fold']}")
            quality_window_start = fold["source_start_ns"]
            quality_window_end = fold["source_end_ns"]
            quality = store.quality(
                start_ns=quality_window_start, end_ns=quality_window_end,
                expected_instruments=(instrument,),
                coverage_policy=CoveragePolicy(**coverage_for_window(
                    environment["coverage_policy"],
                    {"start_ns": quality_window_start, "end_ns": quality_window_end})))
            store.require_usable(quality, "rl_es")
            cadence_ns = plan["sample_interval_minutes"] * MINUTE_NS
            train_indices = sample_indices(times, fold["train_start_ns"],
                                           fold["test_start_ns"], cadence_ns)
            test_indices = sample_indices(times, fold["test_start_ns"],
                                          fold["test_end_ns"], cadence_ns)
            selected_indices = np.unique(np.concatenate((train_indices, test_indices)))
            started = time.perf_counter()
            feature_rows = [make_feature_row(store, spec, int(times[index]), instrument=instrument)
                            for index in selected_indices]
            features = np.asarray([row.values for row in feature_rows], dtype=np.float64)
            feature_times = np.asarray([row.decision_ts_ns for row in feature_rows], dtype=np.int64)
            if (features.ndim != 2 or not np.isfinite(features).all()
                    or any(row.available_at_ns > row.decision_ts_ns for row in feature_rows)):
                raise DataError("non-finite or future-visible predictor feature row")
            del feature_rows
            lookup = {int(index): offset for offset, index in enumerate(selected_indices)}
            rows = []
            for horizon in plan["horizons_minutes"]:
                train_idx, train_y, train_actual, train_future_ns = labels_for(
                    train_indices, times, closes, horizon, plan["label_max_lag_minutes"])
                train_keep = train_future_ns <= fold["test_start_ns"]
                train_idx, train_y, train_actual = (train_idx[train_keep], train_y[train_keep],
                                                     train_actual[train_keep])
                test_idx, test_y, test_actual, test_future_ns = labels_for(
                    test_indices, times, closes, horizon, plan["label_max_lag_minutes"])
                test_keep = ((times[test_idx] >= fold["test_start_ns"])
                             & (times[test_idx] < fold["test_end_ns"])
                             & (test_future_ns <= fold["source_end_ns"]))
                test_idx, test_y, test_actual = (test_idx[test_keep], test_y[test_keep],
                                                 test_actual[test_keep])
                if train_idx.size < 10_000 or test_idx.size < 2_000:
                    raise DataError("fold has too few samples for the preregistered forecast diagnostic")
                train_x = features[[lookup[int(index)] for index in train_idx]]
                test_x = features[[lookup[int(index)] for index in test_idx]]
                alpha = float(plan["model"]["alpha"])
                ridge_prediction, fit_meta = fit_ridge(train_x, train_y, test_x, alpha)
                past_indices = np.searchsorted(times, times[test_idx] - horizon * MINUTE_NS, side="right") - 1
                past_valid = past_indices >= 0
                past_delay = times[test_idx[past_valid]] - horizon * MINUTE_NS - times[past_indices[past_valid]]
                past_valid_positions = np.flatnonzero(past_valid)
                past_valid[past_valid_positions[past_delay > plan["label_max_lag_minutes"] * MINUTE_NS]] = False
                current_close = closes[test_idx]
                momentum = np.zeros(test_idx.size, dtype=np.float64)
                momentum[past_valid] = np.log(current_close[past_valid] / closes[past_indices[past_valid]]) * 10_000.0
                if np.count_nonzero(past_valid) != test_idx.size:
                    raise DataError("test window lacks causal same-horizon momentum observations")
                cost_bp = float(plan["round_trip_cost_bps"])
                row = {
                    "fold": fold["fold"], "horizon_minutes": horizon,
                    "train_samples": int(train_idx.size), "test_samples": int(test_idx.size),
                    "train_actual_horizon_minutes_mean": float(np.mean(train_actual)),
                    "test_actual_horizon_minutes_mean": float(np.mean(test_actual)),
                    "feature_build_seconds": time.perf_counter() - started,
                    "data_quality": {"grade": quality.grade.value,
                                     "verified_coverage": quality.verified_coverage},
                    "ridge": {**fit_meta, **forecast_metrics(ridge_prediction, test_y, cost_bp)},
                    "zero_score": forecast_metrics(np.zeros_like(test_y), test_y, cost_bp),
                    "same_horizon_momentum": forecast_metrics(momentum, test_y, cost_bp),
                    "same_horizon_reversal": forecast_metrics(-momentum, test_y, cost_bp),
                    "round_trip_cost_bps_scenario_only": cost_bp,
                    "interpretation": "forecast ranking diagnostics; no native execution replay",
                }
                rows.append(row)
                print(json.dumps({"fold": fold["fold"], "horizon": horizon,
                                  "train_samples": train_idx.size, "test_samples": test_idx.size,
                                  "ridge_spearman_ic": row["ridge"]["spearman_ic"],
                                  "ridge_top_decile_net_hurdle_bp": row["ridge"]["top_decile_long_after_scenario_cost_bp"]}),
                      flush=True)
                del train_x, test_x, ridge_prediction, momentum
            fold_result = {"fold": fold["fold"], "rows": rows,
                           "feature_count": int(features.shape[1]),
                           "feature_names": [field.name for field in spec.fields],
                           "feature_samples": int(feature_times.size),
                           "source_interval_ns": [fold["source_start_ns"], fold["source_end_ns"]]}
            _write_verified(output / f"fold-{fold['fold']}.json", fold_result)
            fold_results.append(fold_result)
            del features, feature_times, lookup, selected_indices
            gc.collect()

        summary = {
            "schema": "rles.low-capacity-forecast-results.v1", "split": "train_inner",
            "scope": "development-only internal diagnostics; not independent OOS",
            "source_event_count": len(store.events), "dataset_snapshot_hash": store.snapshot_hash,
            "bundle_sha256": plan["bundle_sha256"], "plan_sha256": _file_hash(plan_path),
            "folds": fold_results,
            "conclusion": "diagnostic only; no model selection, training budget expansion, or strategy promotion",
            "limitations": [
                "All selected periods are previously used Train data and now registered as development.",
                "Fee/slippage hurdle is an explicit scenario assumption, not verified historical economics.",
                "Forecast labels and tail hurdle summaries do not model position sizing, turnover, funding, or execution.",
                "Forecast rows are sampled every five minutes; overlapping 15m/60m targets are dependent.",
                "No block bootstrap or statistical significance claim is made.",
            ],
        }
        _write_verified(output / "summary.json", summary)
        return summary
    except BaseException as exc:
        _write_verified(output / "FAILED.json", {"error": type(exc).__name__, "detail": str(exc)})
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.plan, args.output)
    print(json.dumps({"folds": len(result["folds"]), "summary": str(args.output / "summary.json")}))


if __name__ == "__main__":
    main()
