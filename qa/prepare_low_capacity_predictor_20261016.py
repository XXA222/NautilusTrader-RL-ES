"""Write the fixed, Train-only low-capacity forecast diagnostic protocol."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path

from rles_orchestrator.application import _file_hash, _verify_file, _write_verified


def ns(value: datetime) -> int:
    return int(value.timestamp()) * 1_000_000_000


def prepare(root: Path) -> Path:
    source_plan_path = root / "configs/train_signal_baselines_20261016.json"
    source_plan = json.loads(source_plan_path.read_text(encoding="utf-8"))
    bundle_root = Path(source_plan["bundle"])
    if _verify_file(bundle_root / "bundle.json") != source_plan["bundle_sha256"]:
        raise ValueError("frozen candidate bundle changed")
    bundle = json.loads((bundle_root / "bundle.json").read_text(encoding="utf-8"))
    test_starts = ("2025-02-01", "2025-09-01", "2026-04-01")
    folds = []
    for index, date in enumerate(test_starts, start=1):
        test_start = datetime.fromisoformat(date).replace(tzinfo=timezone.utc)
        test_end = test_start + timedelta(days=30)
        train_start = test_start - timedelta(days=180)
        source_start = train_start - timedelta(days=1)  # causal feature warmup
        source_end = test_end + timedelta(hours=1)  # maximum 60m forward label
        folds.append({
            "fold": index, "source_start_ns": ns(source_start),
            "train_start_ns": ns(train_start), "test_start_ns": ns(test_start),
            "test_end_ns": ns(test_end), "source_end_ns": ns(source_end),
        })
    plan = {
        "schema": "rles.low-capacity-forecast-plan.v1", "date": "2026-10-16",
        "split": "train_inner", "scope": "development-only; no independent OOS claim",
        "bundle": str(bundle_root), "bundle_sha256": source_plan["bundle_sha256"],
        "dataset_sha256": bundle["preprocessing"]["identities"]["dataset"],
        "source": source_plan["source"], "source_plan_sha256": _file_hash(source_plan_path),
        "feature_profile": bundle["environment"].get("feature_profile", "legacy"),
        "folds": folds, "sample_interval_minutes": 5,
        "horizons_minutes": [5, 15, 60], "training_days": 180, "test_days": 30,
        "purge_minutes": 60, "label_max_lag_minutes": 2,
        "model": {"kind": "multi-output ridge, fit separately by horizon",
                  "alpha": 0.1, "features": "all market fields from frozen feature profile",
                  "feature_scaling": "fold-local train-only mean and population standard deviation",
                  "intercept": True, "random_seeds": []},
        "baselines": ["zero-score", "same-horizon causal momentum", "same-horizon causal reversal"],
        "round_trip_cost_bps": 12.0,
        "cost_assumption": "2x 5bp taker fee + 2x 1bp bar slippage; scenario-only, not historical fact",
        "selection": "none; report all folds/horizons without model tuning or promotion gate",
        "label": "log(future_close/current_visible_close)*10_000, where future_close is first available at or after decision+H",
        "guardrails": ["future prices are labels only", "test labels end within frozen Train boundary",
                       "test windows avoid all formally reserved Test/Holdout intervals",
                       "no training/feature fitting uses the test windows",
                       "forecast net-cost summaries are hurdle diagnostics, not execution replay"],
    }
    path = root / "configs/low_capacity_predictor_20261016.json"
    if path.exists():
        raise FileExistsError(path)
    _write_verified(path, plan)
    return path


if __name__ == "__main__":
    print(prepare(Path(__file__).resolve().parents[1]))
