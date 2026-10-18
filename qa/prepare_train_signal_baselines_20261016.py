"""Archive existing inference artifacts and preregister bounded Train diagnostics."""
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

from rles_orchestrator.application import _file_hash, _verify_file, _write_verified


def prepare(root: Path) -> Path:
    prior = root / ".tmp/rl-stage66-memory-bounded-full-train-replays-20261016/preregistration.json"
    run = json.loads(prior.read_text(encoding="utf-8"))["runs"][0]
    source_bundle = Path(run["bundle"])
    if _verify_file(source_bundle / "bundle.json") != run["bundle_sha256"]:
        raise ValueError("existing candidate differs from Stage66 frozen artifact")
    bundle = json.loads((source_bundle / "bundle.json").read_text(encoding="utf-8"))
    archive = root / ".artifacts/audit-development-20261016"
    archive.mkdir(parents=True, exist_ok=False)
    model = archive / "models/candidate-seed-102"
    model.mkdir(parents=True)
    for name in ("bundle.json", "bundle.json.sha256", "COMPLETED.json", "run.json",
                 "R1-learning.json", "R1-learning.json.sha256"):
        shutil.copy2(source_bundle / name, model / name)
    shutil.copytree(source_bundle / bundle["actor"]["path"], model / bundle["actor"]["path"])
    data = archive / "data"
    data.mkdir()
    market = Path(bundle["environment"]["market_path"])
    rules = Path(bundle["environment"]["rules_path"])
    shutil.copy2(market, data / "market.parquet")
    shutil.copy2(market.with_name(market.name + ".COMPLETED.json"), data / "market.parquet.COMPLETED.json")
    shutil.copy2(rules, data / "scenario-rules.json")
    for evidence in json.loads(rules.read_text(encoding="utf-8"))["sources"]:
        relative = Path(evidence["path"])
        target = (data / relative).resolve()
        if relative.is_absolute() or not target.is_relative_to(data.resolve()):
            raise ValueError("rule evidence must have an archive-contained relative path")
        source = rules.parent / relative
        if _file_hash(source) != evidence["sha256"]:
            raise ValueError("rule evidence checksum changed")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    windows = []
    for date in ("2024-01-15", "2024-02-15", "2024-03-15"):
        start = int(datetime.fromisoformat(date).replace(tzinfo=timezone.utc).timestamp()) * 10**9
        windows.append({"start_ns": start, "end_ns": start + 86_400_000_000_000})
    plan = {
        "schema": "rles.train-signal-plan.v1", "date": "2026-10-16", "split": "train_inner",
        "bundle": str(model), "bundle_sha256": _file_hash(model / "bundle.json"),
        "source": {"market_path": str(data / "market.parquet"),
                   "market_sha256": _file_hash(data / "market.parquet"),
                   "rules_path": str(data / "scenario-rules.json"),
                   "rules_sha256": _file_hash(data / "scenario-rules.json")},
        "windows": windows, "modes": ["base", "stress"], "seed": 102,
        "policies": ["B0_no_trade", "B1_always_long", "B4_always_short",
                     "B2_simple_momentum", "B5_simple_reversal", "frozen_rl"],
        "target": 1.0, "momentum_lookback": 20,
        "minimum_start_free_bytes": 10 * 1024**3, "stop_free_bytes": 2 * 1024**3,
        "gate": "RL must strictly exceed every fixed baseline in every window and cost mode; screening only",
        "on_failure": "retain all reports; do not expand current candidate to 100k/300k or formal OOS",
        "scope": "frozen old RL inference only; no retraining, selection or independent OOS claim",
    }
    plan_path = root / "configs/train_signal_baselines_20261016.json"
    _write_verified(plan_path, plan)
    manifest = []
    for path in sorted(archive.rglob("*")):
        if path.is_file():
            manifest.append({"path": str(path.relative_to(archive)).replace("\\", "/"),
                             "bytes": path.stat().st_size, "sha256": _file_hash(path)})
    _write_verified(root / "docs/audit_evidence/20261016/persistent_artifacts.json", {
        "schema": "rles.persistent-artifacts.v1", "archive": str(archive),
        "files": manifest, "model_source": str(source_bundle), "market_source": str(market),
        "rules_source": str(rules), "model_bundle_sha256": run["bundle_sha256"],
        "availability": "local archive; not a public release or clean-checkout reproduction",
    })
    return plan_path


if __name__ == "__main__":
    print(prepare(Path(__file__).resolve().parents[1]))
