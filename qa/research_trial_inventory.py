"""Inventory legacy WFO candidate metadata without loading market data.

The scan is deliberately limited to the two roots named in the frozen legacy
research-access inventory. Query rows are reservations, not proof that training
or evaluation completed. Candidate config fingerprints are descriptive and are
not promoted to project-wide effective-trial estimates.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sqlite3


def _file_sha(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _canonical_sha(value: object) -> str:
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                             ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()


def inventory(root: Path) -> dict:
    manifest_path = root / "docs/audit_evidence/20261016/legacy_research_access_v2.json"
    marker_path = manifest_path.with_suffix(manifest_path.suffix + ".sha256")
    if not manifest_path.is_file() or not marker_path.is_file():
        raise ValueError("frozen legacy research-access inventory or SHA marker is missing")
    manifest_sha = _file_sha(manifest_path)
    if marker_path.read_text(encoding="ascii") != manifest_sha:
        raise ValueError("frozen legacy research-access inventory SHA mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "rles.legacy-research-access.v1":
        raise ValueError("unexpected legacy research-access inventory schema")

    config_rows: list[dict] = []
    for relative_root in manifest["scope"]:
        scan_root = root / relative_root
        if not scan_root.is_dir():
            raise ValueError(f"legacy research root missing: {relative_root}")
        for path in sorted(scan_root.rglob("config*.json")):
            config = json.loads(path.read_text(encoding="utf-8"))
            wfo = config.get("wfo")
            if not isinstance(wfo, dict) or not isinstance(wfo.get("protocol_id"), str):
                continue
            candidates = []
            for value in wfo.get("candidates", []):
                if not isinstance(value, dict) or not isinstance(value.get("candidate_id"), str):
                    continue
                parameters = {
                    key: item for key, item in value.items()
                    if key not in {"candidate_id", "seed", "feature_profile"}
                }
                seed = value.get("seed", config.get("seed", 31))
                feature_profile = value.get(
                    "feature_profile", config.get("environment", {}).get("feature_profile", "legacy"),
                )
                candidates.append({
                    "candidate_id": value["candidate_id"],
                    "seed": seed,
                    "feature_profile": feature_profile,
                    "parameter_sha256": _canonical_sha(parameters),
                    "attempt_recipe_sha256": _canonical_sha({
                        "parameters": parameters, "seed": seed, "feature_profile": feature_profile,
                    }),
                })
            config_rows.append({
                "path": path.relative_to(root).as_posix(),
                "config_sha256": _file_sha(path),
                "protocol_id": wfo["protocol_id"],
                "candidates": candidates,
            })

    query_rows: list[dict] = []
    protocol_rows: list[dict] = []
    db_sources = []
    for database in manifest["databases"]:
        relative_db = database["path"]
        db_path = root / relative_db
        source_sha = database["sha256"]
        if _file_sha(db_path) != source_sha:
            raise ValueError(f"legacy research database SHA mismatch: {relative_db}")
        db_sources.append({"path": relative_db, "sha256": source_sha})
        uri = db_path.resolve().as_uri() + "?mode=ro"
        with sqlite3.connect(uri, uri=True) as connection:
            connection.row_factory = sqlite3.Row
            states = connection.execute(
                "SELECT id,state FROM protocol ORDER BY id",
            ).fetchall()
            queries = connection.execute(
                "SELECT query_id,protocol_id,kind,candidate,fold_id FROM query ORDER BY query_id",
            ).fetchall()
        for state in states:
            protocol_id = state["id"]
            candidates = [row for row in config_rows if row["protocol_id"] == protocol_id]
            protocol_queries = [row for row in queries if row["protocol_id"] == protocol_id]
            protocol_rows.append({
                "protocol_id": protocol_id,
                "state": state["state"],
                "config_files": [row["path"] for row in candidates],
                "config_candidates": [
                    {"config_path": config["path"], **candidate}
                    for config in candidates for candidate in config["candidates"]
                ],
                "query_reservations": [dict(row) for row in protocol_queries],
            })
            config_candidate_ids = {
                candidate["candidate_id"] for config in candidates
                for candidate in config["candidates"]
            }
            for query in protocol_queries:
                query_rows.append({
                    "database": relative_db,
                    **dict(query),
                    "matching_config_candidate": query["candidate"] in config_candidate_ids,
                    "execution_completion": "not_proven_by_query_reservation",
                })

    recipes = {
        candidate["attempt_recipe_sha256"]
        for protocol in protocol_rows for candidate in protocol["config_candidates"]
    }
    kinds: dict[str, int] = {}
    for query in query_rows:
        kinds[query["kind"]] = kinds.get(query["kind"], 0) + 1
    return {
        "schema": "rles.legacy-research-trial-inventory.v1",
        "source_inventory": {
            "path": manifest_path.relative_to(root).as_posix(),
            "sha256": manifest_sha,
            "database_sources": db_sources,
            "scan_roots": manifest["scope"],
            "market_data_loaded": False,
            "formal_test_or_holdout_data_loaded": False,
        },
        "coverage": {
            "protocol_count": len(protocol_rows),
            "config_file_count": len({path for row in protocol_rows for path in row["config_files"]}),
            "query_reservation_count": len(query_rows),
            "query_reservations_by_kind": kinds,
            "protocol_candidate_pairs": len({
                (query["protocol_id"], query["candidate"]) for query in query_rows
            }),
            "candidate_id_count": len({query["candidate"] for query in query_rows}),
            "distinct_config_recipe_fingerprints": len(recipes),
            "standalone_report_index_count": len(manifest["standalone_reports"]),
            "legacy_access_entry_count": len(manifest["entries"]),
            "unresolved_legacy_access_count": len(manifest["unresolved"]),
        },
        "protocols": protocol_rows,
        "query_reservations": query_rows,
        "effective_project_historical_trials": "not_estimated",
        "limitations": [
            "only the two roots in the frozen legacy access inventory are scanned",
            "query reservations do not prove training/evaluation completion or failed attempt coverage",
            "candidate recipe fingerprints omit code, data, window, and selection-history identity",
            "manual selections and unledgered research are not represented",
            "correlated candidate performance is unavailable for historical sensitivity analysis",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output if args.output.is_absolute() else root / args.output
    payload = inventory(root)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as target:
        json.dump(payload, target, indent=2, ensure_ascii=False, allow_nan=False)
        target.write("\n")
    output.with_suffix(output.suffix + ".sha256").write_text(
        _file_sha(output), encoding="ascii",
    )
    print(json.dumps({"coverage": payload["coverage"], "output": str(output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
