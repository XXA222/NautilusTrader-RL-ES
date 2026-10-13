"""L2-local CLI: python -m rles_data validate|build-fold. No L1 CLI shadow."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
from typing import NoReturn

from .core import CuratedStore, DataError
from .coverage import CoveragePolicy
from .folds import build_folds
from .io import ingest_parquet, write_fact_json


class JSONParser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:
        raise DataError(f"invalid CLI arguments: {message}")


def _parser() -> JSONParser:
    parser = JSONParser(prog="rles data", description="Nautilus-RLES independent L2 Data Truth CLI")
    sub = parser.add_subparsers(dest="command", required=True, parser_class=JSONParser)
    val = sub.add_parser("validate")
    inputs = val.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--manifest-json", help="JSON manifest with required inputs list")
    inputs.add_argument("--parquet", help="Single verified Parquet input (partial DQ only)")
    val.add_argument("--source-hash")
    val.add_argument("--kind")
    val.add_argument("--provenance")
    val.add_argument("--delay-ns", type=int)
    val.add_argument("--start-ns", type=int)
    val.add_argument("--end-ns", type=int)
    val.add_argument("--expected-instrument", action="append",
                     help="repeat for every required instrument; manifest mode uses expected_instruments")
    val.add_argument("--card-output")
    fold = sub.add_parser("build-fold")
    for field in ("start-ns", "end-ns", "train-days", "warmup-ns", "feature-lookback-ns",
                  "execution-lag-guard-ns", "embargo-ns"):
        fold.add_argument("--" + field, type=int, required=True)
    fold.add_argument("--data-start-ns", type=int, required=True)
    fold.add_argument("--dataset-hash", required=True)
    fold.add_argument("--feature-hash", required=True)
    fold.add_argument("--scaler-hash")
    fold.add_argument("--output")
    return parser


def _strict_input(value: dict) -> dict:
    allowed = {"path", "source_hash", "kind", "provenance", "conservative_delay_ns"}
    if not isinstance(value, dict) or set(value) - allowed or not {"path", "source_hash", "kind", "provenance"} <= set(value):
        raise DataError("manifest input has unknown or missing fields")
    return value


def _unique_json_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise DataError(f"duplicate validation manifest key: {key}")
        result[key] = value
    return result


def _validate(args: argparse.Namespace) -> tuple[dict, int]:
    if args.manifest_json:
        doc = json.loads(
            Path(args.manifest_json).read_text(encoding="utf-8"),
            object_pairs_hook=_unique_json_pairs,
            parse_constant=lambda value: (_ for _ in ()).throw(DataError(f"invalid JSON constant: {value}")),
        )
        if (not isinstance(doc, dict) or set(doc) - {"inputs", "start_ns", "end_ns", "expected_instruments", "coverage_policy"}
                or not isinstance(doc.get("inputs"), list) or not doc["inputs"]):
            raise DataError("invalid validation manifest")
        if any(getattr(args, x) is not None for x in ("source_hash", "kind", "provenance", "delay_ns")):
            raise DataError("single-file flags forbidden with manifest-json")
        if args.start_ns is not None or args.end_ns is not None or args.expected_instrument is not None:
            raise DataError("time range and expected instruments must be set by manifest-json")
        sources = [_strict_input(x) for x in doc["inputs"]]
        start_ns, end_ns = doc.get("start_ns"), doc.get("end_ns")
        expected_instruments = doc.get("expected_instruments")
        if expected_instruments is not None and (
            not isinstance(expected_instruments, list) or not expected_instruments or
            any(type(symbol) is not str or not symbol for symbol in expected_instruments) or
            len(set(expected_instruments)) != len(expected_instruments)
        ):
            raise DataError("expected_instruments must be a nonempty unique string list")
        expected_instruments = tuple(expected_instruments) if expected_instruments is not None else None
        coverage = doc.get("coverage_policy")
        coverage_policy = None if coverage is None else CoveragePolicy(**coverage)
    else:
        if not args.source_hash or not args.kind or not args.provenance:
            raise DataError("--parquet requires --source-hash, --kind, --provenance")
        sources = [{"path": args.parquet, "source_hash": args.source_hash,
                    "kind": args.kind, "provenance": args.provenance,
                    "conservative_delay_ns": args.delay_ns}]
        start_ns, end_ns = args.start_ns, args.end_ns
        expected_instruments = (tuple(args.expected_instrument)
                                if args.expected_instrument is not None else None)
        coverage_policy = None
    all_events: list = []
    for source in sources:
        all_events.extend(ingest_parquet(**source).events)
    store = CuratedStore(all_events)
    report = store.quality(start_ns=start_ns, end_ns=end_ns,
                            expected_instruments=expected_instruments, coverage_policy=coverage_policy)
    card = store.card(report)
    result = {"status": "BLOCKED" if report.grade.value == "DQ-D" else "VALIDATED",
              "snapshot_hash": store.snapshot_hash, "dataset_hash": card.hash,
              "dataset_card": asdict(card), "quality": asdict(report),
              "quality_hash": report.hash}
    if args.card_output:
        # Never hide missing dependencies and never overwrite another run's manifest.
        write_fact_json(args.card_output, {"card": card, "quality": report})
    return result, (2 if report.grade.value == "DQ-D" else 0)


def _fold(args: argparse.Namespace) -> tuple[dict, int]:
    folds = build_folds(start_ns=args.start_ns, end_ns=args.end_ns,
                        train_days=args.train_days, warmup_ns=args.warmup_ns,
                        feature_lookback_ns=args.feature_lookback_ns,
                        execution_lag_guard_ns=args.execution_lag_guard_ns,
                        embargo_ns=args.embargo_ns, dataset_hash=args.dataset_hash,
                        feature_hash=args.feature_hash, scaler_hash=args.scaler_hash,
                        data_start_ns=args.data_start_ns)
    if not folds:
        raise DataError("no complete scoring fold within verified dataset")
    result = {"folds": [{"manifest": asdict(f), "hash": f.hash} for f in folds],
              "count": len(folds)}
    if args.output:
        write_fact_json(args.output, result)
    return result, 0


def main(argv: list[str] | None = None) -> int:
    try:
        args = _parser().parse_args(argv)
        result, status = _validate(args) if args.command == "validate" else _fold(args)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True, allow_nan=False))
        return status
    except (DataError, ValueError, OSError, ImportError, TypeError, OverflowError) as exc:
        print(json.dumps({"error": type(exc).__name__, "detail": str(exc)},
                         ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
