"""Verified Binance USD-M archive ingestion; no simulated prices or rule defaults."""
from __future__ import annotations

import csv
from datetime import date, datetime, timezone
from decimal import Decimal, InvalidOperation
from hashlib import sha256
import io
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import urlopen
from uuid import uuid4
from zipfile import ZipFile

from .core import Availability, CuratedStore, DataError, Event, MINUTE_NS
from .io import write_curated, write_fact_json

BASE = "https://data.binance.vision/data/futures/um/"


def _read_with_curl(url: str, *, limit: int, temp_parent: Path | None = None) -> bytes:
    executable = shutil.which("curl")
    if executable is None:
        raise DataError("curl fallback unavailable")
    with tempfile.TemporaryDirectory(prefix="rles-binance-download-", dir=temp_parent) as temporary:
        destination = Path(temporary) / "response.bin"
        result = subprocess.run(
            [executable, "--fail", "--location", "--silent", "--show-error",
             "--retry", "2", "--retry-all-errors", "--max-time", "20",
             "--max-filesize", str(limit + 1),
             "--output", str(destination), url],
            capture_output=True, check=False, timeout=25,
        )
        if result.returncode != 0:
            detail = result.stderr.decode("utf-8", errors="replace")[:512]
            raise DataError(f"curl archive fallback failed: {detail}")
        content = destination.read_bytes()
        if len(content) > limit:
            raise DataError("archive exceeds bounded download size")
        return content


def _read_remote(url: str, *, limit: int, temp_parent: Path | None = None) -> bytes:
    """Retry short-lived archive transport failures without weakening checks."""
    last_error: Exception | None = None
    # Historical imports can require hundreds of sequential archive/checksum
    # requests. A brief EOF on one request must not abandon the whole window.
    for attempt in range(5):
        try:
            with urlopen(url, timeout=20) as response:
                return response.read(limit + 1)
        except HTTPError as exc:
            if exc.code not in {408, 429, 500, 502, 503, 504}:
                raise
            last_error = exc
        except (URLError, TimeoutError, ConnectionError) as exc:
            last_error = exc
            if attempt == 4:
                break
        if attempt < 4:
            time.sleep(0.25 * (2 ** attempt))
    if last_error is not None and shutil.which("curl") is not None:
        return _read_with_curl(url, limit=limit, temp_parent=temp_parent)
    if last_error is not None:
        raise last_error
    raise DataError("archive download retry loop ended unexpectedly")


def fetch_archive(relative_url: str, destination: Path) -> dict:
    """Cache source bytes only after validating the exchange-published checksum."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    url = BASE + relative_url
    source_receipt = destination.with_suffix(destination.suffix + ".source.json")
    if destination.is_symlink() or source_receipt.is_symlink():
        raise DataError("archive cache entries must not be symbolic links")
    if source_receipt.exists():
        if not destination.is_file() or not source_receipt.is_file():
            raise DataError("cached archive source receipt has no regular archive file")
        try:
            saved = json.loads(source_receipt.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise DataError("cached archive source receipt is unreadable") from exc
        if (not isinstance(saved, dict) or set(saved) != {"schema", "url", "sha256"}
                or saved["schema"] != "rles.binance-archive-source.v1"
                or saved["url"] != url
                or not isinstance(saved["sha256"], str)
                or re.fullmatch(r"[0-9a-f]{64}", saved["sha256"]) is None):
            raise DataError("cached archive source receipt identity is invalid")
        size_limit = (128 * 1024 * 1024 if any(stream in relative_url
                                               for stream in ("/bookTicker/", "/aggTrades/"))
                      else 64 * 1024 * 1024)
        if destination.stat().st_size > size_limit:
            raise DataError("cached archive exceeds bounded importer size")
        content_hash = sha256(destination.read_bytes()).hexdigest()
        if content_hash != saved["sha256"]:
            raise DataError("cached archive differs from its verified source receipt")
        return {"path": str(destination.resolve()), "url": url, "sha256": content_hash}

    checksum = _read_remote(url + ".CHECKSUM", limit=1024,
                            temp_parent=destination.parent).decode("ascii").split()
    if len(checksum) != 2 or re.fullmatch(r"[0-9a-f]{64}", checksum[0]) is None:
        raise DataError("invalid Binance archive checksum")
    if checksum[1].lstrip("*") != relative_url.rsplit("/", 1)[-1]:
        raise DataError("Binance checksum names a different archive")
    size_limit = (128 * 1024 * 1024 if any(stream in relative_url
                                           for stream in ("/bookTicker/", "/aggTrades/"))
                  else 64 * 1024 * 1024)
    if destination.exists():
        content = destination.read_bytes()
    else:
        content = _read_remote(url, limit=size_limit, temp_parent=destination.parent)
        if len(content) > size_limit:
            raise DataError("archive exceeds bounded importer size")
    if sha256(content).hexdigest() != checksum[0]:
        raise DataError("Binance archive checksum mismatch")
    if not destination.exists():
        with destination.open("xb") as target:
            target.write(content)
    receipt = {"schema": "rles.binance-archive-source.v1", "url": url, "sha256": checksum[0]}
    temporary = source_receipt.with_name(source_receipt.name + f".{uuid4().hex}.tmp")
    temporary.write_text(json.dumps(receipt, sort_keys=True, separators=(",", ":")),
                         encoding="utf-8")
    temporary.replace(source_receipt)
    return {"path": str(destination.resolve()), "url": url, "sha256": checksum[0]}


def _raw_cache_only(target: Path) -> bool:
    """Allow a staging directory containing only regular verified-cache entries."""
    raw = target / "raw"
    if (target.is_symlink() or not target.is_dir() or raw.is_symlink() or not raw.is_dir()
            or {path.name for path in target.iterdir()} != {"raw"}):
        return False
    paths = tuple(raw.iterdir())
    if not all(not path.is_symlink() and path.is_file() for path in paths):
        return False
    names = {path.name for path in paths}
    api_data = {name for name in names if name.endswith("-api.json")}
    api_receipts = {name for name in names if name.endswith("-api.json.source.json")}
    return (
        all(name.endswith((".zip", ".zip.source.json", "-api.json", "-api.json.source.json"))
            for name in names)
        and api_receipts == {name + ".source.json" for name in api_data}
    )


def archive_events(path: Path, *, stream: str, instrument: str, source_url: str,
                   expected_sha256: str, delay_ns: int) -> tuple[Event, ...]:
    """Use closed bars, and realized funding only after its source timestamp.

    Archives do not contain historical receipt/publication clocks. A positive
    declared delay is therefore marked DERIVED, never NATIVE. The funding
    feature is the last *settled* rate; it is not a future funding forecast.
    """
    if type(delay_ns) is not int or delay_ns <= 0:
        raise DataError("archive ingress requires an explicit positive availability delay")
    if stream not in {"klines", "markPriceKlines", "indexPriceKlines", "fundingRate",
                      "bookTicker", "aggTrades"}:
        raise DataError("unsupported Binance USD-M stream")
    content = path.read_bytes()
    if sha256(content).hexdigest() != expected_sha256:
        raise DataError("archive changed after verification")
    result = []
    with ZipFile(io.BytesIO(content)) as archive:
        members = archive.infolist()
        member_limit = 1024 * 1024 * 1024 if stream in {"bookTicker", "aggTrades"} else 128 * 1024 * 1024
        if len(members) != 1 or not members[0].filename.endswith(".csv") or members[0].file_size > member_limit:
            raise DataError("expected one bounded CSV archive member")
        with archive.open(members[0]) as source:
            if stream == "bookTicker":
                # Official daily bookTicker files are multi-GB when uncompressed.
                # Retain the last exchange quote in each 1-second event-time bucket,
                # bounding CURATED size while preserving a real quote at every
                # available sample point. The selected cadence is recorded in data provenance.
                last_bucket = None
                selected = None
                previous_event_ms = -1

                def emit_quote(row: list[str], row_number: int) -> Event:
                    if len(row) != 7:
                        raise DataError("unexpected bookTicker archive columns")
                    update_id = int(row[0])
                    transaction_ms, event_ms = int(row[5]), int(row[6])
                    if update_id < 0 or transaction_ms < 0 or event_ms < 0:
                        raise DataError("invalid bookTicker identity or timestamp")
                    ts = event_ms * 1_000_000
                    return Event(
                        f"binance:{expected_sha256[:16]}:best_quote:{update_id}", "best_quote",
                        ts, ts + delay_ns, ts + delay_ns, Availability.DERIVED, expected_sha256,
                        {"instrument": instrument, "bid_price": row[1], "bid_qty": row[2],
                         "ask_price": row[3], "ask_qty": row[4], "update_id": update_id,
                         "transaction_time_ns": transaction_ms * 1_000_000},
                        source_url + "#last-event-time-quote-per-second",
                    )

                for number, row in enumerate(csv.reader(io.TextIOWrapper(source, encoding="utf-8"))):
                    if number == 0 and row and row[0] == "update_id":
                        continue
                    if len(row) != 7:
                        raise DataError("unexpected bookTicker archive columns")
                    event_ms = int(row[6])
                    if event_ms < previous_event_ms:
                        raise DataError("bookTicker event timestamps are not ordered")
                    previous_event_ms = event_ms
                    bucket = event_ms // 1000
                    if last_bucket is not None and bucket != last_bucket:
                        if selected is not None:
                            result.append(emit_quote(selected, number))
                        selected = row
                    else:
                        selected = row
                    last_bucket = bucket
                if selected is not None:
                    result.append(emit_quote(selected, number))
                return tuple(result)
            if stream == "aggTrades":
                # Preserve exact one-second aggressor-flow aggregates instead of
                # retaining millions of individual Event objects in memory.
                bucket_ns = 1_000_000_000
                current_bucket: int | None = None
                aggregate: dict[str, Any] | None = None
                previous_event_ms = -1
                previous_trade_id = -1

                def emit_flow(bucket: int, values: dict[str, Any]) -> Event:
                    payload = {
                        "instrument": instrument,
                        "aggressor_buy_qty": str(values["buy_qty"]),
                        "aggressor_sell_qty": str(values["sell_qty"]),
                        "aggressor_buy_notional": str(values["buy_notional"]),
                        "aggressor_sell_notional": str(values["sell_notional"]),
                        "trade_count": values["count"],
                        "first_trade_id": values["first_trade_id"],
                        "last_trade_id": values["last_trade_id"],
                        "first_price": str(values["first_price"]),
                        "last_price": str(values["last_price"]),
                        "high_price": str(values["high_price"]),
                        "low_price": str(values["low_price"]),
                    }
                    ts = (bucket + 1) * bucket_ns
                    return Event(
                        f"binance:{expected_sha256[:16]}:trade_flow_1s:{bucket}",
                        "trade_flow_1s", ts, ts + delay_ns, ts + delay_ns,
                        Availability.DERIVED, expected_sha256, payload,
                        source_url + "#one-second-aggressor-flow",
                    )

                for number, row in enumerate(csv.reader(io.TextIOWrapper(source, encoding="utf-8"))):
                    if number == 0 and row and row[0] == "agg_trade_id":
                        continue
                    if len(row) != 7:
                        raise DataError("unexpected USD-M aggTrades archive columns")
                    try:
                        trade_id, price, quantity = int(row[0]), Decimal(row[1]), Decimal(row[2])
                        first_id, last_id, event_ms = int(row[3]), int(row[4]), int(row[5])
                    except (ValueError, InvalidOperation) as exc:
                        raise DataError("invalid USD-M aggTrades identity, price, or quantity") from exc
                    maker_flag = row[6].strip().lower()
                    if (maker_flag not in {"true", "false"} or trade_id < 0 or first_id < 0
                            or last_id < first_id or price <= 0 or quantity <= 0
                            or event_ms < previous_event_ms or trade_id <= previous_trade_id):
                        raise DataError("USD-M aggTrades identity, order, or value is invalid")
                    previous_event_ms, previous_trade_id = event_ms, trade_id
                    bucket = event_ms * 1_000_000 // bucket_ns
                    if current_bucket is not None and bucket != current_bucket:
                        if aggregate is None or bucket <= current_bucket:
                            raise DataError("USD-M aggTrades timestamps are not strictly bucket ordered")
                        result.append(emit_flow(current_bucket, aggregate))
                        aggregate = None
                    if aggregate is None:
                        aggregate = {
                            "buy_qty": Decimal(0), "sell_qty": Decimal(0),
                            "buy_notional": Decimal(0), "sell_notional": Decimal(0),
                            "count": 0, "first_trade_id": first_id, "last_trade_id": last_id,
                            "first_price": price, "last_price": price,
                            "high_price": price, "low_price": price,
                        }
                        current_bucket = bucket
                    side = "sell" if maker_flag == "true" else "buy"
                    aggregate[f"{side}_qty"] += quantity
                    aggregate[f"{side}_notional"] += price * quantity
                    aggregate["count"] += 1
                    aggregate["last_trade_id"] = last_id
                    aggregate["last_price"] = price
                    aggregate["high_price"] = max(aggregate["high_price"], price)
                    aggregate["low_price"] = min(aggregate["low_price"], price)
                if current_bucket is not None and aggregate is not None:
                    result.append(emit_flow(current_bucket, aggregate))
                return tuple(result)
            for number, row in enumerate(csv.reader(io.TextIOWrapper(source, encoding="utf-8"))):
                if number == 0 and row and row[0] in {"open_time", "calc_time"}:
                    continue
                if stream == "fundingRate":
                    if len(row) != 3:
                        raise DataError("unexpected funding archive columns")
                    ts = int(row[0]) * 1_000_000
                    interval = int(row[1]) * 60
                    if interval <= 0:
                        raise DataError("invalid funding interval")
                    payloads: dict[str, dict[str, Any]] = {
                        "funding_settlement": {"rate": row[2], "settlement_ns": ts, "interval_minutes": interval},
                        "funding_update": {"rate": row[2], "rate_semantics": "last_settled_rate"},
                    }
                else:
                    if len(row) != 12:
                        raise DataError("unexpected kline archive columns")
                    opened = int(row[0]) * 1_000_000
                    ts = opened + MINUTE_NS
                    if opened % MINUTE_NS or int(row[6]) * 1_000_000 != ts - 1_000_000:
                        raise DataError("archive is not an aligned one-minute closed candle")
                    if stream == "klines":
                        payloads = {"ohlcv_1m": dict(zip(("open", "high", "low", "close", "volume"), row[1:6]))}
                    else:
                        payloads = {"mark" if stream == "markPriceKlines" else "index": {"price": row[4]}}
                for kind, payload in payloads.items():
                    result.append(Event(
                        f"binance:{expected_sha256[:16]}:{kind}:{number}", kind, ts, ts + delay_ns,
                        ts + delay_ns, Availability.DERIVED, expected_sha256,
                        {"instrument": instrument, **payload}, source_url,
                    ))
    return tuple(result)


def funding_api_events(content: bytes, *, symbol: str, instrument: str, source_url: str,
                       expected_sha256: str, delay_ns: int) -> tuple[Event, ...]:
    """Normalize official REST funding history without claiming archive checksums.

    Funding interval is derived from adjacent observed settlement timestamps because
    the REST response does not publish a historical interval field. A single row
    therefore uses the next row's observed gap; settlement timestamps may carry
    millisecond jitter, so only gaps within one second of a whole minute pass.
    """
    if type(delay_ns) is not int or delay_ns <= 0:
        raise DataError("funding API ingress requires an explicit positive availability delay")
    if sha256(content).hexdigest() != expected_sha256:
        raise DataError("funding API response changed after capture")
    try:
        rows = json.loads(content)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DataError("funding API response is not valid JSON") from exc
    if not isinstance(rows, list) or not rows:
        raise DataError("funding API response must contain at least one settlement")
    normalized: list[tuple[int, str]] = []
    for row in rows:
        if not isinstance(row, dict) or row.get("symbol") != symbol:
            raise DataError("funding API response contains an unexpected symbol or row")
        ts_ms, rate = row.get("fundingTime"), row.get("fundingRate")
        if type(ts_ms) is not int or ts_ms < 0 or not isinstance(rate, str):
            raise DataError("funding API row has invalid timestamp or rate")
        if row.get("rateType") not in {"Regular", "Special"}:
            raise DataError("funding API row has an unknown rate type")
        normalized.append((ts_ms * 1_000_000, rate))
    if any(current[0] <= previous[0] for previous, current in zip(normalized, normalized[1:])):
        raise DataError("funding API settlement timestamps must be strictly ascending")

    result: list[Event] = []
    for index, (ts, rate) in enumerate(normalized):
        neighbor = normalized[index - 1][0] if index else normalized[index + 1][0]
        gap_ns = abs(ts - neighbor)
        minute_ns = 60 * 1_000_000_000
        interval_minutes = round(gap_ns / minute_ns)
        if interval_minutes <= 0 or abs(gap_ns - interval_minutes * minute_ns) > 1_000_000_000:
            raise DataError("funding API settlement gap must be within one second of whole minutes")
        for kind, payload in (
            ("funding_settlement", {"rate": rate, "settlement_ns": ts,
                                    "interval_minutes": interval_minutes}),
            ("funding_update", {"rate": rate, "rate_semantics": "last_settled_rate",
                                "interval_minutes": interval_minutes}),
        ):
            result.append(Event(
                f"binance-api:{expected_sha256[:16]}:{kind}:{ts}", kind, ts,
                ts + delay_ns, ts + delay_ns, Availability.DERIVED, expected_sha256,
                {"instrument": instrument, **payload}, source_url,
            ))
    return tuple(result)


def fetch_funding_api(symbol: str, month: date, end_day: date, raw: Path) -> tuple[dict, bytes]:
    """Fetch and retain a current-month REST supplement with an explicit receipt."""
    start_ms = int(datetime(month.year, month.month, 1, tzinfo=timezone.utc).timestamp() * 1000)
    end_ms = int(datetime(end_day.year, end_day.month, end_day.day, tzinfo=timezone.utc).timestamp() * 1000) + 86_400_000 - 1
    query = urlencode({"symbol": symbol, "startTime": start_ms, "endTime": end_ms, "limit": 1000})
    url = f"https://fapi.binance.com/fapi/v1/fundingRate?{query}"
    destination = raw / f"fundingRate-{month:%Y-%m}-api.json"
    receipt_path = destination.with_suffix(destination.suffix + ".source.json")
    if destination.is_symlink() or receipt_path.is_symlink():
        raise DataError("funding API cache entries must not be symbolic links")
    if destination.exists() or receipt_path.exists():
        if not destination.is_file() or not receipt_path.is_file():
            raise DataError("cached funding API source receipt has no response file")
        try:
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise DataError("cached funding API source receipt is unreadable") from exc
        content = destination.read_bytes()
        digest = sha256(content).hexdigest()
        if receipt != {"schema": "rles.binance-funding-api-source.v1", "url": url, "sha256": digest}:
            raise DataError("cached funding API response differs from its source receipt")
        return {"path": str(destination.resolve()), "url": url, "sha256": digest,
                "verification": "captured_response_sha256"}, content
    content = _read_remote(url, limit=8 * 1024 * 1024, temp_parent=raw)
    if len(content) > 8 * 1024 * 1024:
        raise DataError("funding API response exceeds bounded download size")
    digest = sha256(content).hexdigest()
    receipt = {"schema": "rles.binance-funding-api-source.v1", "url": url, "sha256": digest}
    temporary = destination.with_name(destination.name + f".{uuid4().hex}.tmp")
    temporary.write_bytes(content)
    temporary.replace(destination)
    receipt_tmp = receipt_path.with_name(receipt_path.name + f".{uuid4().hex}.tmp")
    receipt_tmp.write_text(json.dumps(receipt, sort_keys=True, separators=(",", ":")), encoding="utf-8")
    receipt_tmp.replace(receipt_path)
    return {"path": str(destination.resolve()), "url": url, "sha256": digest,
            "verification": "captured_response_sha256"}, content


def import_day(*, symbol: str, day: str, output: str | Path, delay_ns: int,
               instrument: str | None = None, include_bookticker: bool = False,
               include_agg_trades: bool = False) -> dict:
    if re.fullmatch(r"[A-Z0-9]{3,24}", symbol) is None:
        raise DataError("invalid Binance symbol")
    parsed_day = date.fromisoformat(day)
    if parsed_day.isoformat() != day:
        raise DataError("day must be YYYY-MM-DD")
    if type(include_bookticker) is not bool:
        raise DataError("include_bookticker must be an explicit boolean")
    if type(include_agg_trades) is not bool:
        raise DataError("include_agg_trades must be an explicit boolean")
    target = Path(output).resolve()
    if target.exists() and not (
        (target / "IMPORT_FAILED.json").is_file()
        and not (target / "market.parquet").exists()
        and not (target / "import.json").exists()
        or _raw_cache_only(target)
    ):
        raise DataError("output directory already exists; use a new immutable import")
    target.mkdir(parents=True, exist_ok=True)
    sources: list[dict] = []
    events: list[Event] = []
    try:
        streams = ["klines", "markPriceKlines", "indexPriceKlines", "fundingRate"]
        if include_bookticker:
            streams.append("bookTicker")
        if include_agg_trades:
            streams.append("aggTrades")
        for stream in streams:
            if stream == "fundingRate":
                relative = f"monthly/{stream}/{symbol}/{symbol}-fundingRate-{day[:7]}.zip"
            elif stream == "bookTicker":
                relative = f"daily/{stream}/{symbol}/{symbol}-{stream}-{day}.zip"
            elif stream == "aggTrades":
                relative = f"daily/{stream}/{symbol}/{symbol}-{stream}-{day}.zip"
            else:
                relative = f"daily/{stream}/{symbol}/1m/{symbol}-1m-{day}.zip"
            source = fetch_archive(relative, target / "raw" / (stream + ".zip"))
            sources.append(source)
            events.extend(archive_events(Path(source["path"]), stream=stream,
                                         instrument=instrument or f"{symbol}-PERP.SIM",
                                         source_url=source["url"], expected_sha256=source["sha256"],
                                         delay_ns=delay_ns))
        # Keep funding before the first bar for causal warmup, but never later
        # than the imported day's final close. No future monthly rates leak in.
        bars = [e.ts_event_ns for e in events if e.kind == "ohlcv_1m"]
        events = [e for e in events if e.ts_event_ns <= max(bars)]
        store = CuratedStore(events)
        write_curated(target / "market.parquet", store)
        metadata = {"schema": "rles.binance-archive.v1", "symbol": symbol, "day": day,
                    "instrument": instrument or f"{symbol}-PERP.SIM", "sources": sources,
                    "availability": "derived", "availability_delay_ns": delay_ns,
                    "funding_feature": "last_settled_rate", "event_count": len(store.events),
                    "start_ns": min(bars), "end_ns": max(bars) + MINUTE_NS,
                    "instrument_rules_supplied": False,
                    "bookticker_included": include_bookticker,
                    "bookticker_sampling": "last_event_time_quote_per_second" if include_bookticker else None,
                    "agg_trades_included": include_agg_trades,
                    "agg_trades_sampling": "one_second_aggressor_flow" if include_agg_trades else None}
        write_fact_json(target / "import.json", metadata)
        (target / "IMPORT_FAILED.json").unlink(missing_ok=True)
        return metadata
    except Exception as exc:
        # Preserve source bytes and a clear failure record for a future retry.
        (target / "IMPORT_FAILED.json").write_text(json.dumps({"error": type(exc).__name__, "detail": str(exc)}), encoding="utf-8")
        raise


def import_range(*, symbol: str, start_day: str, end_day: str,
                 output: str | Path, delay_ns: int,
                 instrument: str | None = None, include_bookticker: bool = False,
                 include_agg_trades: bool = False,
                 archive_interval: str = "daily") -> dict:
    """Import a closed inclusive date range from Binance's checksum archives.

    Daily OHLCV/mark/index files are retained individually; monthly funding
    files are fetched once per month and clipped to the imported bars. No
    instrument rules are inferred or inserted.
    """
    if re.fullmatch(r"[A-Z0-9]{3,24}", symbol) is None:
        raise DataError("invalid Binance symbol")
    first, last = date.fromisoformat(start_day), date.fromisoformat(end_day)
    if first.isoformat() != start_day or last.isoformat() != end_day or last < first:
        raise DataError("range must use ordered YYYY-MM-DD start/end dates")
    if type(include_bookticker) is not bool:
        raise DataError("include_bookticker must be an explicit boolean")
    if type(include_agg_trades) is not bool:
        raise DataError("include_agg_trades must be an explicit boolean")
    if type(archive_interval) is not str or archive_interval not in {"daily", "monthly"}:
        raise DataError("archive_interval must be daily or monthly")
    if (include_bookticker or include_agg_trades) and first != last:
        raise DataError("bookTicker/aggTrades import is limited to one UTC day to bound curated event memory")
    if include_agg_trades and archive_interval != "daily":
        raise DataError("aggTrades are available only from the daily archive importer")
    if type(delay_ns) is not int or delay_ns <= 0:
        raise DataError("archive ingress requires an explicit positive availability delay")
    target = Path(output).resolve()
    if target.exists() and not (
        (target / "IMPORT_FAILED.json").is_file()
        and not (target / "market.parquet").exists()
        and not (target / "import.json").exists()
        or _raw_cache_only(target)
    ):
        raise DataError("output directory already exists; use a new immutable import")
    target.mkdir(parents=True, exist_ok=True)
    raw = target / "raw"
    raw.mkdir(exist_ok=True)
    sources: list[dict] = []
    events: list[Event] = []
    try:
        if archive_interval == "daily":
            current = first
            while current <= last:
                day = current.isoformat()
                for stream in ("klines", "markPriceKlines", "indexPriceKlines"):
                    relative = f"daily/{stream}/{symbol}/1m/{symbol}-1m-{day}.zip"
                    source = fetch_archive(relative, raw / f"{stream}-{day}.zip")
                    sources.append(source)
                    events.extend(archive_events(
                        Path(source["path"]), stream=stream, instrument=instrument or f"{symbol}-PERP.SIM",
                        source_url=source["url"], expected_sha256=source["sha256"], delay_ns=delay_ns,
                    ))
                if include_bookticker:
                    stream = "bookTicker"
                    relative = f"daily/{stream}/{symbol}/{symbol}-{stream}-{day}.zip"
                    source = fetch_archive(relative, raw / f"{stream}-{day}.zip")
                    sources.append(source)
                    events.extend(archive_events(
                        Path(source["path"]), stream=stream, instrument=instrument or f"{symbol}-PERP.SIM",
                        source_url=source["url"], expected_sha256=source["sha256"], delay_ns=delay_ns,
                    ))
                if include_agg_trades:
                    stream = "aggTrades"
                    relative = f"daily/{stream}/{symbol}/{symbol}-{stream}-{day}.zip"
                    source = fetch_archive(relative, raw / f"{stream}-{day}.zip")
                    sources.append(source)
                    events.extend(archive_events(
                        Path(source["path"]), stream=stream, instrument=instrument or f"{symbol}-PERP.SIM",
                        source_url=source["url"], expected_sha256=source["sha256"], delay_ns=delay_ns,
                    ))
                current = date.fromordinal(current.toordinal() + 1)
        else:
            # Binance publishes monthly USD-M klines, mark and index archives.
            # Use these to reduce hundreds of daily checksum/download round trips;
            # exact requested dates are still clipped below and audited downstream.
            month = date(first.year, first.month, 1)
            final_month = date(last.year, last.month, 1)
            while month <= final_month:
                month_key = month.strftime("%Y-%m")
                for stream in ("klines", "markPriceKlines", "indexPriceKlines"):
                    relative = (f"monthly/{stream}/{symbol}/1m/"
                                f"{symbol}-1m-{month_key}.zip")
                    source = fetch_archive(relative, raw / f"{stream}-{month_key}.zip")
                    sources.append(source)
                    events.extend(archive_events(
                        Path(source["path"]), stream=stream, instrument=instrument or f"{symbol}-PERP.SIM",
                        source_url=source["url"], expected_sha256=source["sha256"], delay_ns=delay_ns,
                    ))
                month = date(month.year + (month.month == 12), month.month % 12 + 1, 1)

        month = date(first.year, first.month, 1)
        final_month = date(last.year, last.month, 1)
        current_month = datetime.now(timezone.utc).date().replace(day=1)
        while month <= final_month:
            month_key = month.strftime("%Y-%m")
            relative = f"monthly/fundingRate/{symbol}/{symbol}-fundingRate-{month_key}.zip"
            try:
                source = fetch_archive(relative, raw / f"fundingRate-{month_key}.zip")
            except HTTPError as exc:
                if exc.code != 404 or month != final_month or month != current_month:
                    raise
                source, content = fetch_funding_api(symbol, month, last, raw)
                sources.append(source)
                events.extend(funding_api_events(
                    content, symbol=symbol, instrument=instrument or f"{symbol}-PERP.SIM",
                    source_url=source["url"], expected_sha256=source["sha256"], delay_ns=delay_ns,
                ))
            else:
                sources.append(source)
                events.extend(archive_events(
                    Path(source["path"]), stream="fundingRate", instrument=instrument or f"{symbol}-PERP.SIM",
                    source_url=source["url"], expected_sha256=source["sha256"], delay_ns=delay_ns,
                ))
            month = date(month.year + (month.month == 12), month.month % 12 + 1, 1)

        bars = [event.ts_event_ns for event in events if event.kind == "ohlcv_1m"]
        if not bars:
            raise DataError("Binance archive range contains no one-minute bars")
        # Kline event timestamps are candle closes. Include the final close at
        # 00:00 UTC after the inclusive end date; funding stays inside dates.
        requested_start_ns = int(datetime.combine(first, datetime.min.time(), tzinfo=timezone.utc).timestamp() * 1e9)
        requested_end_ns = int(datetime.combine(
            date.fromordinal(last.toordinal() + 1), datetime.min.time(), tzinfo=timezone.utc,
        ).timestamp() * 1e9)
        # The prior settlement remains available from any adjacent warm-up archive.
        events = [
            event for event in events
            if (requested_start_ns <= event.ts_event_ns < requested_end_ns
                if event.kind in {"funding_update", "funding_settlement"}
                else requested_start_ns < event.ts_event_ns < requested_end_ns + MINUTE_NS)
        ]
        clipped_bars = [event.ts_event_ns for event in events if event.kind == "ohlcv_1m"]
        if not clipped_bars:
            raise DataError("Binance archives contain no bars inside the requested date range")
        first_bar, last_bar = min(clipped_bars), max(clipped_bars)
        store = CuratedStore(events)
        write_curated(target / "market.parquet", store)
        metadata = {
            "schema": "rles.binance-archive-range.v1", "symbol": symbol,
            "start_day": start_day, "end_day": end_day,
            "instrument": instrument or f"{symbol}-PERP.SIM", "sources": sources,
            "availability": "derived", "availability_delay_ns": delay_ns,
            "archive_interval": archive_interval,
            "agg_trades_included": include_agg_trades,
            "agg_trades_sampling": "one_second_aggressor_flow" if include_agg_trades else None,
            "funding_feature": "last_settled_rate", "event_count": len(store.events),
            "start_ns": first_bar, "end_ns": last_bar + MINUTE_NS,
            "instrument_rules_supplied": False,
            "bookticker_included": include_bookticker,
            "bookticker_sampling": "last_event_time_quote_per_second" if include_bookticker else None,
        }
        write_fact_json(target / "import.json", metadata)
        (target / "IMPORT_FAILED.json").unlink(missing_ok=True)
        return metadata
    except Exception as exc:
        (target / "IMPORT_FAILED.json").write_text(
            json.dumps({"error": type(exc).__name__, "detail": str(exc)}), encoding="utf-8",
        )
        raise


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", default="ETHUSDT")
    parser.add_argument("--day")
    parser.add_argument("--start-day")
    parser.add_argument("--end-day")
    parser.add_argument("--output", required=True)
    parser.add_argument("--delay-ns", type=int, required=True)
    parser.add_argument("--archive-interval", choices=("daily", "monthly"), default="daily",
                        help="use daily archives (default) or checksum-verified monthly kline archives")
    parser.add_argument("--include-bookticker", action="store_true",
                        help="import Binance best bid/ask sampled at one-second event-time cadence")
    parser.add_argument("--include-agg-trades", action="store_true",
                        help="import USD-M aggTrades as checksum-verified one-second aggressor-flow events")
    args = parser.parse_args()
    if args.day is not None and args.start_day is None and args.end_day is None:
        result = import_day(symbol=args.symbol, day=args.day, output=args.output, delay_ns=args.delay_ns,
                            include_bookticker=args.include_bookticker,
                            include_agg_trades=args.include_agg_trades)
    elif args.day is None and args.start_day is not None and args.end_day is not None:
        result = import_range(symbol=args.symbol, start_day=args.start_day, end_day=args.end_day,
                              output=args.output, delay_ns=args.delay_ns,
                              archive_interval=args.archive_interval,
                              include_bookticker=args.include_bookticker,
                              include_agg_trades=args.include_agg_trades)
    else:
        parser.error("provide either --day or both --start-day and --end-day")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
