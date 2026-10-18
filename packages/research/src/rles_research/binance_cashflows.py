"""Import explicitly typed Binance USDⓈ-M transfer income into native ledger facts."""
from __future__ import annotations

import hashlib
import json
from bisect import bisect_left
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any

from .metrics import MetricsError


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Reject duplicate keys so raw response bytes have one unambiguous meaning."""
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise MetricsError("Binance income-history JSON contains a duplicate object key")
        result[key] = value
    return result


@dataclass(frozen=True)
class BinanceTransferImport:
    """Transfer facts and provenance from one exact Binance income-history payload."""

    facts: tuple[dict[str, Any], ...]
    source_id: str
    source_sha256: str
    event_time_range_ns: tuple[int, int] | None
    limitations: tuple[str, ...]
    query_time_range_ms: tuple[int, int] | None = None
    page_sha256: tuple[str, ...] = ()


@dataclass(frozen=True)
class BinanceIncomePage:
    """One archived inclusive time-range query and its exact response bytes."""

    start_time_ms: int
    end_time_ms: int
    limit: int
    payload: bytes


def import_binance_usdm_transfer_income(
    payload: bytes | str,
    *,
    account_scope_id: str,
    currency: str,
    sample_barriers_ns: Sequence[int],
) -> BinanceTransferImport:
    """Convert an archived ``GET /fapi/v1/income`` TRANSFER response to ledger facts.

    The first barrier is the event-free initial snapshot. Each exchange event is
    assigned to the first sample barrier at or after its millisecond timestamp.
    This makes the attribution explicit while preserving the native ledger's
    requirement that flows be after the initial snapshot and on a unique barrier.
    The exact input bytes are hashed; this does not authenticate the exporter or
    certify that the three-month Binance endpoint supplied a complete history.
    """
    if not isinstance(account_scope_id, str) or not account_scope_id.strip():
        raise MetricsError("Binance transfer import requires an account scope ID")
    if not isinstance(currency, str) or not currency.strip():
        raise MetricsError("Binance transfer import requires an account currency")
    if not isinstance(payload, (bytes, str)):
        raise MetricsError("Binance transfer payload must preserve exact source bytes")
    raw_bytes = payload if isinstance(payload, bytes) else payload.encode("utf-8")
    try:
        document = json.loads(raw_bytes, object_pairs_hook=_unique_json_object)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MetricsError("invalid Binance income-history JSON payload") from exc
    if not isinstance(document, list):
        raise MetricsError("Binance income-history payload must be a JSON array")
    if not sample_barriers_ns or any(type(value) is not int or value < 0 for value in sample_barriers_ns):
        raise MetricsError("sample barriers must be nonnegative integer timestamps")
    barriers = list(sample_barriers_ns)
    if barriers != sorted(barriers) or len(set(barriers)) != len(barriers):
        raise MetricsError("sample barriers must be strictly increasing")
    if len(barriers) < 2:
        raise MetricsError("at least an initial and one later sample barrier are required")

    digest = hashlib.sha256(raw_bytes).hexdigest()
    source_id = f"binance-usdm-income:{account_scope_id.strip()}:{digest}"
    event_by_id: dict[str, tuple[tuple[Any, ...], dict[str, Any]]] = {}
    for row in document:
        if not isinstance(row, Mapping):
            raise MetricsError("Binance income-history row must be an object")
        if row.get("incomeType") != "TRANSFER":
            raise MetricsError("Binance income-history payload contains a non-TRANSFER row")
        asset = row.get("asset")
        if asset != currency:
            raise MetricsError("Binance transfer currency does not match account currency")
        tran_id = row.get("tranId")
        if type(tran_id) is not int or tran_id < 0:
            raise MetricsError("Binance transfer requires a nonnegative integer tranId")
        event_id = f"binance-usdm:{account_scope_id.strip()}:TRANSFER:{tran_id}"
        timestamp_ms = row.get("time")
        if type(timestamp_ms) is not int or timestamp_ms < 0:
            raise MetricsError("Binance transfer requires a nonnegative integer time")
        try:
            amount = Decimal(str(row.get("income")))
        except (InvalidOperation, ValueError, TypeError) as exc:
            raise MetricsError("invalid Binance transfer amount") from exc
        if not amount.is_finite() or amount == 0:
            raise MetricsError("Binance transfer amount must be finite and nonzero")
        event_time_ns = timestamp_ms * 1_000_000
        barrier_index = bisect_left(barriers, event_time_ns)
        if barrier_index == 0 or barrier_index == len(barriers):
            raise MetricsError("Binance transfer time falls outside post-initial sample barriers")
        fact = {
            "event_id": event_id,
            "decision_ts_ns": barriers[barrier_index],
            "currency": currency,
            "kind": "DEPOSIT" if amount > 0 else "WITHDRAWAL",
            "amount": str(abs(amount)),
            "source_id": source_id,
            "source_sha256": digest,
        }
        identity = (timestamp_ms, asset, str(amount), row.get("incomeType"), tran_id)
        previous = event_by_id.get(event_id)
        if previous is not None and previous[0] != identity:
            raise MetricsError("Binance transfer tranId has conflicting facts")
        event_by_id[event_id] = (identity, fact)

    facts = tuple(item[1] for item in sorted(
        event_by_id.values(), key=lambda item: (item[0][0], item[0][4])
    ))
    event_times = [item[0][0] * 1_000_000 for item in event_by_id.values()]
    return BinanceTransferImport(
        facts=facts,
        source_id=source_id,
        source_sha256=digest,
        event_time_range_ns=(min(event_times), max(event_times)) if event_times else None,
        limitations=(
            "Binance USDⓈ-M income history retains only the latest three months.",
            "A response payload hash identifies bytes but does not authenticate its origin or prove query completeness.",
            "The caller must verify the signed query scope, pagination/window coverage, and account identity.",
            "Events are attributed to the first sample barrier at or after the exchange event time.",
        ),
    )


def import_binance_usdm_transfer_income_pages(
    pages: Sequence[BinanceIncomePage],
    *,
    query_start_ms: int,
    query_end_ms: int,
    account_scope_id: str,
    currency: str,
    sample_barriers_ns: Sequence[int],
) -> BinanceTransferImport:
    """Import archived, non-overlapping income pages covering one query range.

    Binance income query bounds are inclusive. Each page must therefore start
    exactly one millisecond after the previous page ends. A response containing
    ``limit`` rows is rejected because it may have been truncated; callers must
    split that range further. This checks the supplied archive and declared
    query plan only. It does not authenticate that Binance returned the bytes or
    that the request was signed for the stated account and parameters.
    """
    if (type(query_start_ms) is not int or query_start_ms < 0
            or type(query_end_ms) is not int or query_end_ms < query_start_ms):
        raise MetricsError("income query range must be a nonnegative inclusive interval")
    if not pages:
        raise MetricsError("income query requires archived response pages")
    if not isinstance(account_scope_id, str) or not account_scope_id.strip():
        raise MetricsError("Binance transfer import requires an account scope ID")
    if not isinstance(currency, str) or not currency.strip():
        raise MetricsError("Binance transfer import requires an account currency")

    expected_start = query_start_ms
    page_hashes: list[str] = []
    rows: list[dict[str, Any]] = []
    page_manifest: list[dict[str, Any]] = []
    for page in pages:
        if not isinstance(page, BinanceIncomePage):
            raise MetricsError("income pages must use BinanceIncomePage records")
        if (type(page.start_time_ms) is not int or type(page.end_time_ms) is not int
                or page.start_time_ms != expected_start or page.end_time_ms < page.start_time_ms
                or page.end_time_ms > query_end_ms):
            raise MetricsError("income page ranges must contiguously cover the declared query")
        if type(page.limit) is not int or not 1 <= page.limit <= 1000:
            raise MetricsError("income page limit must be between 1 and Binance's maximum of 1000")
        if not isinstance(page.payload, bytes):
            raise MetricsError("income page must preserve exact response bytes")
        try:
            page_rows = json.loads(page.payload, object_pairs_hook=_unique_json_object)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise MetricsError("invalid Binance income-history page JSON") from exc
        if not isinstance(page_rows, list):
            raise MetricsError("Binance income-history page must be a JSON array")
        if len(page_rows) >= page.limit:
            raise MetricsError("income page reached its limit and may be truncated; split the range")
        for row in page_rows:
            if not isinstance(row, Mapping):
                raise MetricsError("Binance income-history row must be an object")
            timestamp_ms = row.get("time")
            if (type(timestamp_ms) is not int or not page.start_time_ms <= timestamp_ms
                    <= page.end_time_ms):
                raise MetricsError("income row falls outside its declared page range")
            rows.append(dict(row))
        digest = hashlib.sha256(page.payload).hexdigest()
        page_hashes.append(digest)
        page_manifest.append({
            "start_time_ms": page.start_time_ms,
            "end_time_ms": page.end_time_ms,
            "limit": page.limit,
            "row_count": len(page_rows),
            "payload_sha256": digest,
        })
        expected_start = page.end_time_ms + 1
    if expected_start != query_end_ms + 1:
        raise MetricsError("income page ranges do not reach the declared query end")

    merged_payload = json.dumps(rows, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    imported = import_binance_usdm_transfer_income(
        merged_payload,
        account_scope_id=account_scope_id,
        currency=currency,
        sample_barriers_ns=sample_barriers_ns,
    )
    manifest = json.dumps({
        "schema": "rles.binance-usdm-income-query.v1",
        "endpoint": "/fapi/v1/income",
        "income_type": "TRANSFER",
        "account_scope_id": account_scope_id.strip(),
        "currency": currency,
        "query_start_ms": query_start_ms,
        "query_end_ms": query_end_ms,
        "pages": page_manifest,
    }, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    manifest_sha256 = hashlib.sha256(manifest).hexdigest()
    source_id = f"binance-usdm-income-query:{account_scope_id.strip()}:{manifest_sha256}"
    facts = tuple({**fact, "source_id": source_id, "source_sha256": manifest_sha256}
                  for fact in imported.facts)
    return BinanceTransferImport(
        facts=facts,
        source_id=source_id,
        source_sha256=manifest_sha256,
        event_time_range_ns=imported.event_time_range_ns,
        limitations=(
            *imported.limitations,
            "Page hashes and contiguous declared ranges do not authenticate Binance responses or signed account scope.",
            "A response with exactly its requested limit is rejected as potentially truncated.",
        ),
        query_time_range_ms=(query_start_ms, query_end_ms),
        page_sha256=tuple(page_hashes),
    )
