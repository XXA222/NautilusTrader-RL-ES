"""Map detached native snapshots into regular economic metric increments."""
from __future__ import annotations

from decimal import Decimal
from typing import Any, Mapping, Sequence

from .metrics import MetricsError, from_account_facts

NATIVE_LIQUIDATION_COUNTS_AVAILABLE = True


def _money(value: Any, currency: str) -> Decimal:
    parts = str(value).replace(",", "").split()
    if len(parts) != 2 or parts[1] != currency:
        raise MetricsError("native amount has unexpected currency")
    result = Decimal(parts[0])
    if not result.is_finite():
        raise MetricsError("nonfinite native ledger amount")
    return result


def account_rows(snapshots: Sequence[Any], *, fill_references: Mapping[str, Decimal],
                 allow_price_improvement: bool = False) -> list[dict]:
    if type(allow_price_improvement) is not bool:
        raise MetricsError("allow_price_improvement must be Boolean")
    if not snapshots:
        raise MetricsError("native account snapshots required")
    by_time = {s.decision_ts_ns: s for s in snapshots}
    fill_facts: dict[str, dict] = {}
    funding_facts: dict[str, dict] = {}
    rows = []
    for timestamp, snapshot in sorted(by_time.items()):
        fee = funding = turnover = slippage = Decimal(0)
        for fill in snapshot.fills:
            identity = fill["event_id"]
            previous = fill_facts.get(identity)
            if previous is not None:
                if previous != fill:
                    raise MetricsError("native fill identity has conflicting facts")
                continue
            fill_facts[identity] = dict(fill)
            quantity, price = Decimal(fill["last_qty"]), Decimal(fill["last_px"])
            reference = fill_references.get(fill["client_order_id"])
            if reference is None:
                raise MetricsError("fill has no native simulation reference for slippage accounting")
            sign = 1 if fill["order_side"] == "BUY" else -1
            cost = (price - reference) * quantity * sign
            if cost < 0 and not allow_price_improvement:
                raise MetricsError("conservative model produced favorable slippage")
            slippage += cost
            turnover += quantity * price
            fee += _money(fill["commission"], snapshot.currency)
        for adjustment in snapshot.funding:
            identity = adjustment["event_id"]
            previous = funding_facts.get(identity)
            if previous is not None:
                if previous != adjustment:
                    raise MetricsError("native funding identity has conflicting facts")
                continue
            funding_facts[identity] = dict(adjustment)
            funding += _money(adjustment["pnl_change"], snapshot.currency)
        rows.append({"ts_ns": timestamp, "equity": float(snapshot.equity), "external_cashflow": 0.0,
                     "turnover_notional": float(turnover), "fee": float(fee), "funding": float(funding),
                     "slippage_cost": float(slippage)})
    return rows


def native_report(snapshots: Sequence[Any], *, fill_references: Mapping[str, Decimal],
                  sample_interval_ns: int, allow_price_improvement: bool = False) -> dict:
    rows = account_rows(snapshots, fill_references=fill_references,
                        allow_price_improvement=allow_price_improvement)
    metrics = from_account_facts(rows, periods_per_year=365.25 * 86400 * 1e9 / sample_interval_ns,
                                 sample_interval_ns=sample_interval_ns,
                                 allow_price_improvement=allow_price_improvement)
    final = max(snapshots, key=lambda s: s.decision_ts_ns)
    fill_facts: dict[str, dict] = {}
    funding_facts: dict[str, dict] = {}
    for snapshot in snapshots:
        for record, facts, label in ((snapshot.fills, fill_facts, "fill"),
                                     (snapshot.funding, funding_facts, "funding")):
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
        for snapshot in snapshots
    )
    liquidation_orders: dict[str, dict] = {}
    if observed:
        for snapshot in snapshots:
            for order in snapshot.liquidation_orders:
                identity = str(order.get("client_order_id", ""))
                if not identity:
                    raise MetricsError("native liquidation order lacks identity")
                liquidation_orders[identity] = order
    return {"metrics": metrics, "account_rows": rows, "fills": list(fill_facts.values()),
            "funding": list(funding_facts.values()), "terminal_flat": not final.positions and not final.orders,
            "liquidations": len(liquidation_orders) if observed else None,
            "liquidations_observed": observed,
            "liquidation_orders": list(liquidation_orders.values()) if observed else None,
            "costs_already_in_native_equity": True}
