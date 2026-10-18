"""Map detached native snapshots into regular economic metric increments."""
from __future__ import annotations

from decimal import Decimal, InvalidOperation
from collections.abc import Iterator, Sequence as SequenceABC
from itertools import groupby
from typing import Any, Mapping, Sequence

from .metrics import AccountFactsAccumulator, MetricsError

NATIVE_LIQUIDATION_COUNTS_AVAILABLE = True

_LIQUIDATION_ORDER_STATE_FIELDS = frozenset({
    "avg_px", "filled_qty", "is_canceled", "is_closed", "is_emulated", "is_expired",
    "is_filled", "is_inflight", "is_open", "is_pending_cancel", "is_pending_update",
    "is_rejected", "is_triggered", "is_working", "leaves_qty", "slippage", "status",
    "ts_last",
})


def _money(value: Any, currency: str) -> Decimal:
    parts = str(value).split()
    if len(parts) != 2 or parts[1] != currency:
        raise MetricsError("native amount has unexpected currency")
    return _decimal(parts[0], "ledger amount")


def _decimal(value: Any, label: str) -> Decimal:
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError) as exc:
        raise MetricsError(f"invalid native {label}") from exc
    if not result.is_finite():
        raise MetricsError(f"nonfinite native {label}")
    return result


def _ordered_snapshots(snapshots: Sequence[Any]) -> list[Any]:
    if (not isinstance(snapshots, SequenceABC)
            or isinstance(snapshots, (str, bytes, bytearray)) or not snapshots):
        raise MetricsError("native account snapshots required")
    try:
        currency = snapshots[0].currency
    except (AttributeError, TypeError) as exc:
        raise MetricsError("native account snapshot lacks currency") from exc
    if not isinstance(currency, str) or not currency:
        raise MetricsError("native account currency required")
    for snapshot in snapshots:
        try:
            timestamp = snapshot.decision_ts_ns
            snapshot_currency = snapshot.currency
            equity = snapshot.equity
        except (AttributeError, TypeError) as exc:
            raise MetricsError("native account snapshot lacks timestamp, currency, or equity") from exc
        if type(timestamp) is not int or timestamp < 0:
            raise MetricsError("native snapshot timestamp must be a nonnegative integer")
        if snapshot_currency != currency:
            raise MetricsError("native snapshots have conflicting currencies")
        _decimal(equity, "equity")
    # Input order is the barrier sequence within a timestamp (e.g. decision,
    # then terminal flatten). Stable sorting must not reorder those barriers.
    return sorted(snapshots, key=lambda item: item.decision_ts_ns)


def _snapshot_records(snapshot: Any, field: str, label: str) -> Sequence[Any]:
    try:
        records = getattr(snapshot, field)
    except (AttributeError, TypeError) as exc:
        raise MetricsError(f"native snapshot lacks {label} history") from exc
    if (not isinstance(records, SequenceABC)
            or isinstance(records, (str, bytes, bytearray))):
        raise MetricsError(f"native {label} history must be a sequence")
    return records


def _new_facts(records: Any, seen: dict[str, dict], label: str) -> Iterator[dict]:
    for record in records:
        if not isinstance(record, Mapping):
            raise MetricsError(f"native {label} history fact must be a mapping")
        identity = record.get("event_id")
        if not isinstance(identity, str) or not identity or identity != identity.strip():
            raise MetricsError(f"native {label} history fact lacks identity")
        required = ({"client_order_id", "order_side", "last_qty", "last_px", "commission"}
                    if label == "fill" else {"pnl_change"} if label == "funding" else set())
        if not required.issubset(record):
            raise MetricsError(f"native {label} history fact lacks required fields")
        if label == "fill" and (
                not isinstance(record["client_order_id"], str)
                or not record["client_order_id"]
                or record["client_order_id"] != record["client_order_id"].strip()):
            raise MetricsError("native fill client order identity is invalid")
        normalized = dict(record)
        previous = seen.get(identity)
        if previous is not None:
            if previous != normalized:
                raise MetricsError(f"native {label} identity has conflicting facts")
            continue
        seen[identity] = normalized
        yield normalized


def _liquidation_order_facts(snapshots: Sequence[Any]) -> dict[str, dict]:
    """Collect each liquidation order once, allowing only native state updates."""
    facts: dict[str, dict] = {}
    identities: dict[str, dict] = {}
    filled_quantities: dict[str, Decimal] = {}
    for snapshot in snapshots:
        for raw_order in _snapshot_records(snapshot, "liquidation_orders", "liquidation order"):
            if not isinstance(raw_order, Mapping):
                raise MetricsError("native liquidation order must be a mapping")
            identity = raw_order.get("client_order_id")
            if not isinstance(identity, str) or not identity or identity != identity.strip():
                raise MetricsError("native liquidation order lacks identity")
            order = dict(raw_order)
            immutable = {key: value for key, value in order.items()
                         if key not in _LIQUIDATION_ORDER_STATE_FIELDS}
            previous = identities.get(identity)
            if previous is not None and previous != immutable:
                raise MetricsError("native liquidation order identity has conflicting facts")
            identities[identity] = immutable
            if "filled_qty" in order:
                filled_qty = _decimal(order["filled_qty"], "liquidation order filled quantity")
                if filled_qty < 0:
                    raise MetricsError("native liquidation order filled quantity is negative")
                if "quantity" in order:
                    quantity = _decimal(order["quantity"], "liquidation order quantity")
                    if quantity <= 0 or filled_qty > quantity:
                        raise MetricsError("native liquidation order filled quantity exceeds order quantity")
                previous_filled = filled_quantities.get(identity)
                if previous_filled is not None and filled_qty < previous_filled:
                    raise MetricsError("native liquidation order cumulative fill quantity regressed")
                filled_quantities[identity] = filled_qty
            # Native order rows are cumulative snapshots. Keep the latest lifecycle state
            # after confirming that its stable order facts still match.
            facts[identity] = order
    return facts


def _external_cashflows_by_time(cashflows: Sequence[Mapping[str, Any]],
                                snapshots: Sequence[Any]) -> tuple[dict[int, Decimal], list[dict]]:
    """Validate explicitly sourced transfers and align them to unique sample barriers.

    The native AccountState API does not identify deposits or withdrawals. This
    interface therefore accepts only caller-supplied, source-identified facts;
    it never infers transfers from balance changes or ``reported`` snapshots.
    """
    if (not isinstance(cashflows, SequenceABC)
            or isinstance(cashflows, (str, bytes, bytearray))):
        raise MetricsError("external cashflows must be a sequence of sourced facts")
    ordered = _ordered_snapshots(snapshots)
    times: dict[int, int] = {}
    for snapshot in ordered:
        times[snapshot.decision_ts_ns] = times.get(snapshot.decision_ts_ns, 0) + 1
    allowed_times = set(times)
    first_time = ordered[0].decision_ts_ns
    currency = ordered[0].currency
    normalized: list[dict] = []
    for raw in cashflows:
        if not isinstance(raw, Mapping):
            raise MetricsError("external cashflow must be an explicitly sourced mapping")
        normalized.append(dict(raw))

    seen: dict[str, dict] = {}
    by_time: dict[int, Decimal] = {}
    unique: list[dict] = []
    for fact in _new_facts(normalized, seen, "external cashflow"):
        timestamp = fact.get("decision_ts_ns")
        source = fact.get("source_id")
        source_hash = fact.get("source_sha256")
        kind = fact.get("kind")
        if (type(timestamp) is not int or timestamp <= first_time or timestamp not in allowed_times):
            raise MetricsError("external cashflow must align to a sample after the initial barrier")
        if times[timestamp] != 1:
            raise MetricsError("external cashflow cannot align to duplicate native barriers")
        if fact.get("currency") != currency:
            raise MetricsError("external cashflow currency differs from native account")
        if not isinstance(kind, str) or kind not in {"DEPOSIT", "WITHDRAWAL"}:
            raise MetricsError("external cashflow kind must be DEPOSIT or WITHDRAWAL")
        amount = _decimal(fact.get("amount"), "external cashflow amount")
        if amount <= 0:
            raise MetricsError("external cashflow amount must be positive")
        if not isinstance(source, str) or not source.strip():
            raise MetricsError("external cashflow source identity required")
        if (not isinstance(source_hash, str) or len(source_hash) != 64
                or any(character not in "0123456789abcdef" for character in source_hash)):
            raise MetricsError("external cashflow source SHA-256 required")
        # Identity is checked by _new_facts before semantic normalization. Store
        # the original typed fact and apply direction only in the ledger sum.
        signed = amount if kind == "DEPOSIT" else -amount
        by_time[timestamp] = by_time.get(timestamp, Decimal(0)) + signed
        unique.append(fact)
    return by_time, unique


def _fill_values(fill: dict) -> tuple[Decimal, Decimal]:
    side = fill.get("order_side")
    if not isinstance(side, str) or side not in {"BUY", "SELL"}:
        raise MetricsError("native fill side must be BUY or SELL")
    quantity = _decimal(fill.get("last_qty"), "fill quantity")
    price = _decimal(fill.get("last_px"), "fill price")
    if quantity <= 0 or price <= 0:
        raise MetricsError("native fill quantity and price must be positive")
    return quantity, price


def iter_account_rows(snapshots: Sequence[Any], *, fill_references: Mapping[str, Decimal],
                      external_cashflows: Sequence[Mapping[str, Any]] = (),
                      allow_price_improvement: bool = False) -> Iterator[dict]:
    """Yield regular native account rows with explicit sample-boundary flows.

    Callers must split history at a transfer barrier. A source hash records the
    supplied provenance identity; callers remain responsible for source trust.
    """
    if type(allow_price_improvement) is not bool:
        raise MetricsError("allow_price_improvement must be Boolean")
    if not isinstance(fill_references, Mapping):
        raise MetricsError("native fill references must be a mapping")
    fill_facts: dict[str, dict] = {}
    funding_facts: dict[str, dict] = {}
    # Multiple native barriers can share a timestamp (decision, then terminal
    # flatten). Stable sorting preserves their event order. Merge their facts
    # and use the last barrier's account state for the one sampled equity row.
    ordered = _ordered_snapshots(snapshots)
    for item in ordered:
        _snapshot_records(item, "fills", "fill")
        _snapshot_records(item, "funding", "funding")
    cashflows_by_time, _ = _external_cashflows_by_time(external_cashflows, ordered)
    for timestamp, group in groupby(ordered, key=lambda item: item.decision_ts_ns):
        same_time = tuple(group)
        snapshot = same_time[-1]
        fee = funding = turnover = slippage = Decimal(0)
        for fill in _new_facts((fill for item in same_time
                                for fill in _snapshot_records(item, "fills", "fill")),
                               fill_facts, "fill"):
            quantity, price = _fill_values(fill)
            reference = fill_references.get(fill["client_order_id"])
            if reference is None:
                raise MetricsError("fill has no native simulation reference for slippage accounting")
            reference = _decimal(reference, "fill reference")
            if reference <= 0:
                raise MetricsError("native fill reference must be positive")
            sign = 1 if fill["order_side"] == "BUY" else -1
            cost = (price - reference) * quantity * sign
            if cost < 0 and not allow_price_improvement:
                raise MetricsError("conservative model produced favorable slippage")
            slippage += cost
            turnover += quantity * price
            fee += _money(fill["commission"], snapshot.currency)
        for adjustment in _new_facts((adjustment for item in same_time
                                     for adjustment in _snapshot_records(item, "funding", "funding")),
                                     funding_facts, "funding"):
            funding += _money(adjustment["pnl_change"], snapshot.currency)
        yield {"ts_ns": timestamp, "equity": float(snapshot.equity),
               "external_cashflow": float(cashflows_by_time.get(timestamp, Decimal(0))),
               "turnover_notional": float(turnover), "fee": float(fee), "funding": float(funding),
               "slippage_cost": float(slippage)}


def account_rows(snapshots: Sequence[Any], *, fill_references: Mapping[str, Decimal],
                 external_cashflows: Sequence[Mapping[str, Any]] = (),
                 allow_price_improvement: bool = False) -> list[dict]:
    """Materialize account rows, including only caller-supplied transfer facts."""
    return list(iter_account_rows(snapshots, fill_references=fill_references,
                                  external_cashflows=external_cashflows,
                                  allow_price_improvement=allow_price_improvement))


def reconcile_linear_equity(snapshots: Sequence[Any], *,
                            external_cashflows: Sequence[Mapping[str, Any]] = (),
                            rounding_per_event: Decimal = Decimal("0.00000001"),
                            include_rows: bool = True) -> dict:
    """Independently reconcile quote-settled, unit-multiplier linear accounts.

    Requires a flat, event-free starting barrier. External cashflows must be
    supplied as explicit source-identified facts aligned to unique sample barriers.
    The caller is responsible for verifying the source; the recorded SHA-256 only
    identifies the supplied source artifact.
    Check every flat barrier and, when the native snapshot contains a complete
    unit-multiplier linear position plus its authoritative mark, check open
    barriers using unrealized price PnL. Actual fill prices already include
    execution slippage. Inverse contracts, multi-asset accounts and external
    deposits/withdrawals without independent typed source facts are outside this contract.
    """
    if (not isinstance(rounding_per_event, Decimal) or not rounding_per_event.is_finite()
            or rounding_per_event < 0 or type(include_rows) is not bool):
        raise MetricsError("finite nonnegative Decimal rounding and Boolean rows required")
    ordered = _ordered_snapshots(snapshots)
    for item in ordered:
        _snapshot_records(item, "positions", "position")
        _snapshot_records(item, "orders", "order")
        _snapshot_records(item, "fills", "fill")
        _snapshot_records(item, "funding", "funding")
    cashflows_by_time, cashflow_facts = _external_cashflows_by_time(external_cashflows, ordered)
    first = ordered[0]
    if (_snapshot_records(first, "positions", "position")
            or _snapshot_records(first, "orders", "order")
            or _snapshot_records(first, "fills", "fill")
            or _snapshot_records(first, "funding", "funding")):
        raise MetricsError("linear reconciliation requires a flat event-free initial barrier")
    initial = _decimal(first.equity, "initial equity")
    if initial <= 0:
        raise MetricsError("linear reconciliation requires positive initial equity")
    quantity = cash = fee = funding = external_cashflow = max_residual = Decimal(0)
    fills: dict[str, dict] = {}
    adjustments: dict[str, dict] = {}
    checked = open_barriers = checked_open = unchecked_open = 0
    rows = []
    for index, snapshot in enumerate(ordered):
        external_cashflow += cashflows_by_time.get(snapshot.decision_ts_ns, Decimal(0))
        for fill in _new_facts(_snapshot_records(snapshot, "fills", "fill"), fills, "fill"):
            size, price = _fill_values(fill)
            signed = size if fill["order_side"] == "BUY" else -size
            quantity += signed
            cash -= signed * price
            fee += _money(fill["commission"], snapshot.currency)
        for adjustment in _new_facts(_snapshot_records(snapshot, "funding", "funding"),
                                     adjustments, "funding"):
            funding += _money(adjustment["pnl_change"], snapshot.currency)
        if quantity != 0:
            open_barriers += 1
            # Compact intermediate snapshots may omit a full position row. Only
            # certify an open barrier when its mark and linear contract metadata
            # are available and agree with the native fill ledger.
            mark_value = getattr(snapshot, "mark", getattr(snapshot, "mark_price", None))
            positions = _snapshot_records(snapshot, "positions", "position")
            if mark_value is not None and positions:
                if len(positions) != 1:
                    raise MetricsError("linear reconciliation requires one open native position")
                position = positions[0]
                if not isinstance(position, Mapping):
                    raise MetricsError("linear reconciliation requires native position mapping")
                if position.get("is_inverse") is True:
                    raise MetricsError("inverse position is outside linear equity reconciliation")
                if position.get("is_inverse") is not False:
                    raise MetricsError("native position inverse capability is unknown")
                multiplier = _decimal(position.get("multiplier"), "position multiplier")
                if multiplier != 1:
                    raise MetricsError("non-unit position multiplier is outside linear equity reconciliation")
                if position.get("settlement_currency") != snapshot.currency:
                    raise MetricsError("position settlement currency differs from account currency")
                side = position.get("side")
                native_size = _decimal(position.get("quantity"), "position quantity")
                if (native_size <= 0 or not isinstance(side, str)
                        or side not in {"LONG", "SHORT"}):
                    raise MetricsError("native linear position side and quantity are invalid")
                native_quantity = native_size if side == "LONG" else -native_size
                if native_quantity != quantity:
                    raise MetricsError("native position disagrees with cumulative fill quantity")
                mark = _decimal(mark_value, "mark price")
                if mark <= 0:
                    raise MetricsError("native mark price must be positive")
                expected = initial + quantity * mark + cash - fee + funding + external_cashflow
                actual = _decimal(snapshot.equity, "equity")
                residual = actual - expected
                tolerance = rounding_per_event * (len(fills) + len(adjustments) + 1)
                max_residual = max(max_residual, abs(residual))
                if abs(residual) > tolerance:
                    raise MetricsError(f"linear native equity mismatch at open barrier {index}: "
                                       f"residual={residual}, tolerance={tolerance}")
                checked_open += 1
                if include_rows:
                    rows.append({"barrier_index": index, "ts_ns": snapshot.decision_ts_ns,
                                 "native_equity": str(actual), "expected_equity": str(expected),
                                 "price_pnl_at_mark": str(cash + quantity * mark),
                                 "mark_price": str(mark), "signed_quantity": str(quantity),
                                 "fee": str(fee), "funding": str(funding),
                                 "external_cashflow": str(external_cashflow),
                                 "residual": str(residual), "tolerance": str(tolerance)})
            else:
                unchecked_open += 1
            # The final barrier must retain authoritative terminal state.
            if index == len(ordered) - 1 and not snapshot.positions:
                raise MetricsError("terminal position disagrees with native fill quantity")
            continue
        if _snapshot_records(snapshot, "positions", "position"):
            raise MetricsError("flat fill quantity disagrees with native positions")
        expected = initial + cash - fee + funding + external_cashflow
        actual = _decimal(snapshot.equity, "equity")
        residual = actual - expected
        tolerance = rounding_per_event * (len(fills) + len(adjustments) + 1)
        max_residual = max(max_residual, abs(residual))
        if abs(residual) > tolerance:
            raise MetricsError(f"linear native equity mismatch at barrier {index}: "
                               f"residual={residual}, tolerance={tolerance}")
        checked += 1
        if include_rows:
            rows.append({"barrier_index": index, "ts_ns": snapshot.decision_ts_ns,
                         "native_equity": str(actual), "expected_equity": str(expected),
                         "realized_price_pnl": str(cash), "fee": str(fee),
                         "funding": str(funding), "external_cashflow": str(external_cashflow),
                         "residual": str(residual),
                         "tolerance": str(tolerance)})
    result: dict = {"scope": "flat_and_marked_open_barriers_quote_settled_unit_linear_with_explicit_cashflow_sources",
              "checked_flat_barriers": checked, "open_barriers": open_barriers,
              "unchecked_open_barriers": unchecked_open,
              "checked_open_barriers": checked_open,
              "terminal_checked": quantity == 0, "max_abs_residual": str(max_residual),
              "rounding_per_event": str(rounding_per_event),
              "unique_fills": len(fills), "unique_funding": len(adjustments),
              "unique_external_cashflows": len(cashflow_facts),
              "net_external_cashflow": str(external_cashflow)}
    if include_rows:
        result["rows"] = rows
    return result


def native_report(snapshots: Sequence[Any], *, fill_references: Mapping[str, Decimal],
                  sample_interval_ns: int, allow_price_improvement: bool = False,
                  include_account_rows: bool = True,
                  external_cashflows: Sequence[Mapping[str, Any]] = (),
                  linear_equity_rounding: Decimal | None = None) -> dict:
    """Build native economic metrics without inferring external account flows."""
    if type(include_account_rows) is not bool:
        raise MetricsError("include_account_rows must be Boolean")
    ordered = _ordered_snapshots(snapshots)
    for item in ordered:
        _snapshot_records(item, "positions", "position")
        _snapshot_records(item, "orders", "order")
    accumulator = AccountFactsAccumulator(sample_interval_ns=sample_interval_ns,
                                          allow_price_improvement=allow_price_improvement)
    rows: list[dict] | None = [] if include_account_rows else None
    for row in iter_account_rows(ordered, fill_references=fill_references,
                                 external_cashflows=external_cashflows,
                                 allow_price_improvement=allow_price_improvement):
        accumulator.add(row)
        if rows is not None:
            rows.append(row)
    metrics = accumulator.result(periods_per_year=365.25 * 86400 * 1e9 / sample_interval_ns)
    final = ordered[-1]
    fill_facts: dict[str, dict] = {}
    funding_facts: dict[str, dict] = {}
    for snapshot in ordered:
        for record, facts, label in ((_snapshot_records(snapshot, "fills", "fill"), fill_facts, "fill"),
                                     (_snapshot_records(snapshot, "funding", "funding"),
                                      funding_facts, "funding")):
            for item in record:
                identity = str(item.get("event_id", ""))
                if not identity:
                    raise MetricsError(f"native {label} history fact lacks identity")
                previous = facts.get(identity)
                if previous is not None and previous != item:
                    raise MetricsError(f"native {label} identity has conflicting facts")
                facts[identity] = dict(item)
    observed = all(
        getattr(snapshot, "liquidation_capability", False) is True
        and getattr(snapshot, "liquidation_orders", None) is not None
        for snapshot in ordered
    )
    liquidation_orders = _liquidation_order_facts(ordered) if observed else {}
    _, cashflow_facts = _external_cashflows_by_time(external_cashflows, ordered)
    final_positions = _snapshot_records(final, "positions", "position")
    final_orders = _snapshot_records(final, "orders", "order")
    report = {"metrics": metrics, "fills": list(fill_facts.values()),
            "funding": list(funding_facts.values()),
            "terminal_flat": not final_positions and not final_orders,
            "external_cashflows": cashflow_facts,
            "liquidations": len(liquidation_orders) if observed else None,
            "liquidations_observed": observed,
            "liquidation_orders": list(liquidation_orders.values()) if observed else None,
            "costs_already_in_native_equity": True}
    if rows is not None:
        report["account_rows"] = rows
    if linear_equity_rounding is not None:
        report["equity_reconciliation"] = reconcile_linear_equity(
            ordered, external_cashflows=external_cashflows,
            rounding_per_event=linear_equity_rounding, include_rows=include_account_rows,
        )
    return report
