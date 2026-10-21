"""Read legacy WFO ledgers and configurations without loading market data.

The inventory is conservative: an old query reservation counts even when its
report is absent. Unknown Holdout end boundaries quarantine the remaining future
interval until the evidence is repaired. This command never edits legacy DBs.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sqlite3
import subprocess

from rles_data.core import MAX_NS, MINUTE_NS, digest


def inventory(root: Path) -> dict:
    scan_roots = [root / ".tmp" / "real-ethusdt-scenario",
                  root / ".tmp" / "wfo-validation-recovery-20261013"]
    configs = []
    for folder in scan_roots:
        for path in sorted(folder.glob("config*.json")):
            value = json.loads(path.read_text(encoding="utf-8"))
            configs.append((path, value))
    discovery = subprocess.run([
        "rg", "--files", "--hidden", "--no-ignore", "--no-messages",
        *[str(path) for path in scan_roots], "-g", "wfo-run.json",
    ], capture_output=True, text=True, check=False)
    entries: list[dict] = []
    databases: list[dict] = []
    unresolved: list[dict] = []
    for name in sorted(discovery.stdout.splitlines()):
        manifest_path = Path(name)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        db = manifest_path.parent / "research.sqlite"
        if not db.is_file():
            unresolved.append({"manifest": str(manifest_path.relative_to(root)), "reason": "missing ledger"})
            continue
        with sqlite3.connect(db.resolve().as_uri() + "?mode=ro", uri=True) as connection:
            connection.row_factory = sqlite3.Row
            states = [dict(row) for row in connection.execute("SELECT * FROM protocol")]
            queries = [dict(row) for row in connection.execute("SELECT * FROM query")]
        db_rel = str(db.relative_to(root)).replace("\\", "/")
        databases.append({"path": db_rel, "sha256": sha256(db.read_bytes()).hexdigest(),
                          "protocols": states, "queries": queries})
        for state in states:
            pid = state["id"]
            matching = [value for _, value in configs
                        if value.get("wfo", {}).get("protocol_id") == pid]
            instrument = matching[0]["environment"]["instrument"] if matching else "ETHUSDT-PERP.SIM"
            market = instrument.split(".")[0].removesuffix("-PERP").upper()
            folds = manifest["fold_manifests"]

            def add(purpose, start, end, evidence):
                identity = {"database": db_rel, "protocol_id": pid,
                            "source_sha": manifest["dataset_hash"], "evidence": evidence}
                entries.append({"key": digest([db_rel, purpose, start, end, evidence]),
                                "market": market, "family": "perpetual-market-v1",
                                "start_ns": start, "end_ns": end, "purpose": purpose,
                                "identity": identity})

            for query in queries:
                if query["protocol_id"] != pid:
                    continue
                if query["kind"] not in {"validation", "test"}:
                    unresolved.append({"database": db_rel, "query": query, "reason": "unknown query scope"})
                    continue
                index = int(query["fold_id"].rsplit("-", 1)[-1])
                fold = folds[index]
                scope = fold[query["kind"]]
                add("test" if query["kind"] == "test" else "development",
                    scope["score_start_ns"], scope["score_end_ns"], query)
                train = fold["train"]
                add("development", train["score_start_ns"], train["score_end_ns"],
                    {"training_for": query["query_id"]})
            if state.get("holdout_sha"):
                start = state.get("holdout_start_ns")
                if start is None and matching:
                    start = matching[0]["wfo"]["holdout_start_ns"]
                end = matching[0]["wfo"]["holdout_end_ns"] if matching else None
                if end is None:
                    report_path = db.parent / "holdout-report.json"
                    if report_path.is_file():
                        rows = json.loads(report_path.read_text(encoding="utf-8")).get("account_rows", [])
                        if rows:
                            end = max(row["ts_ns"] for row in rows) + MINUTE_NS
                if start is None or end is None:
                    unresolved.append({"database": db_rel, "reason": "unknown Holdout boundary; quarantined"})
                    start, end = start or 1, end or MAX_NS
                add("holdout", start, end, {"holdout_sha": state["holdout_sha"],
                                             "state": state["state"], "conservative_end": not bool(matching)})
    standalone_reports = []
    for config_path, config in configs:
        evaluation = config.get("evaluation")
        if config.get("wfo") or not isinstance(evaluation, dict):
            continue
        output = Path(config["output"])
        output = output if output.is_absolute() else config_path.parent / output
        report_path = output / "report.json"
        if not report_path.is_file():
            continue
        split = evaluation.get("split", "validation")
        purpose = split if split in {"test", "holdout"} else "development"
        identity = {"config": str(config_path.relative_to(root)).replace("\\", "/"),
                    "config_sha": sha256(config_path.read_bytes()).hexdigest(),
                    "report_sha": sha256(report_path.read_bytes()).hexdigest(),
                    "split": split, "scope": "completed standalone report; conservatively used"}
        standalone_reports.append(identity)
        entries.append({"key": digest(identity),
                        "market": config["environment"]["instrument"].split(".")[0].removesuffix("-PERP").upper(),
                        "family": "perpetual-market-v1", "start_ns": evaluation["start_ns"],
                        "end_ns": evaluation["end_ns"], "purpose": purpose, "identity": identity})
    return {"schema": "rles.legacy-research-access.v1", "date": "2026-10-16",
            "scope": [str(path.relative_to(root)).replace("\\", "/") for path in scan_roots],
            "market_data_loaded": False, "legacy_databases_modified": False,
            "discovery_exit_code": discovery.returncode,
            "limitations": "Known WFO roots only; unledgered historical accesses cannot be inferred.",
            "databases": databases, "standalone_reports": standalone_reports,
            "entries": entries, "unresolved": unresolved}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = inventory(args.root.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as target:
        json.dump(result, target, indent=2, ensure_ascii=False, allow_nan=False)
    args.output.with_suffix(args.output.suffix + ".sha256").write_text(
        sha256(args.output.read_bytes()).hexdigest(), encoding="ascii")
    print(json.dumps({"databases": len(result["databases"]), "entries": len(result["entries"]),
                      "unresolved": len(result["unresolved"]), "output": str(args.output)}))


if __name__ == "__main__":
    main()
