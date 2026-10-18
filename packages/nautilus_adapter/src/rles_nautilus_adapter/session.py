"""Official Nautilus backtest public-API session with strict timestamp barriers.

L1 supplies neutral DTO factories. L2 supplies ordered, visible market events.
No native object escapes through those DTOs; no synthetic account state exists.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from hashlib import sha256
import importlib.metadata
import re
import sys
from threading import Lock
from typing import Any, Callable

from rles_contracts.registry import ProviderDescriptor

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_NATIVE_SESSION_LOCK = Lock()
_NATIVE_SESSION_OWNER: Any = None


class EngineBoundaryError(RuntimeError):
    """Execution is blocked because the event/account protocol is not proven."""


@dataclass(frozen=True, slots=True)
class VisibleBatch:
    timestamp_ns: int
    events: tuple[Any, ...]
    complete_timestamp: bool

    def __post_init__(self) -> None:
        if (type(self.timestamp_ns) is not int or self.timestamp_ns <= 0
                or not isinstance(self.events, tuple) or not self.events
                or self.complete_timestamp is not True):
            raise ValueError("non-empty, complete immutable timestamp batch required")
        for event in self.events:
            event_ts = getattr(event, "ts_event_ns", None)
            init_ts = getattr(event, "ts_init_ns", None)
            available = getattr(event, "available_at_ns", None)
            if (type(event_ts) is not int or type(init_ts) is not int or
                    init_ts != self.timestamp_ns or event_ts < 0):
                raise EngineBoundaryError("mixed replay timestamps in one batch")
            if (type(available) is not int or available < max(event_ts, init_ts)
                    or available > self.timestamp_ns):
                raise EngineBoundaryError("market data not proven visible at replay timestamp")
            event_id = getattr(event, "event_id", None)
            source_hash = getattr(event, "source_hash", None)
            if (not isinstance(event_id, str) or not event_id or
                    not isinstance(source_hash, str) or _SHA256_RE.fullmatch(source_hash) is None):
                raise EngineBoundaryError("market event provenance missing")


@dataclass(frozen=True, slots=True)
class NativeAccountFacts:
    """Detached facts read from authoritative Nautilus ledger/cache.

    Compact readers expose fills and funding adjustments as newly observed facts;
    non-compact funding remains a full cache-derived snapshot.
    """
    equity: Decimal
    currency: str
    positions: tuple[dict, ...]
    orders: tuple[dict, ...]
    account: dict
    account_events: tuple[dict, ...]
    fills: tuple[dict, ...]
    funding_adjustments: tuple[dict, ...]
    orders_all: tuple[dict, ...]
    liquidation_orders: tuple[dict, ...] | None
    mark_price: Decimal | None
    margins: tuple[dict, ...]
    mark_stale: bool
    liquidation_capability: bool
    timestamp_ns: int


class NativeFactsReader:
    """Read public portfolio/cache queries. Never infer fees or funding from prices."""

    def __init__(self, *, account_id: Any, instrument_id: Any, currency: str,
                 liquidation_capability: bool = False,
                 compact_history: bool = False) -> None:
        if account_id is None or instrument_id is None or not currency:
            raise ValueError("native account/instrument/base currency required")
        if type(liquidation_capability) is not bool:
            raise TypeError("liquidation capability must be certified Boolean")
        if type(compact_history) is not bool:
            raise TypeError("native cache history compaction must be Boolean")
        self.account_id, self.instrument_id, self.currency = account_id, instrument_id, currency
        self.liquidation_capability = liquidation_capability
        self.compact_history = compact_history
        self._latest_account_event: dict | None = None
        self._funding_initialized = False
        self._position_last_updates: dict[tuple[str, int], int] = {}
        self._funding_facts: dict[str, dict] = {}

    def read(self, engine: Any, timestamp_ns: int) -> NativeAccountFacts:
        native = engine.portfolio.build_snapshot(self.account_id)
        if native is None or native.is_stale or native.unpriced_instruments:
            raise EngineBoundaryError("unpriced or stale native equity is not scoreable")
        value = native.base_currency_equity
        if value is None:
            raise EngineBoundaryError("base-currency native equity unavailable")
        if str(value.currency) != self.currency:
            raise EngineBoundaryError("native equity currency differs from frozen account currency")
        equity = Decimal(str(value.as_decimal()))
        if not equity.is_finite():
            raise EngineBoundaryError("non-finite native account equity")
        account = engine.portfolio.account(account_id=self.account_id)
        if account is None:
            raise EngineBoundaryError("native account not found")
        if str(account.base_currency) != self.currency:
            raise EngineBoundaryError("native account base currency differs from episode currency")
        cache = engine.cache
        scope = {"instrument_id": self.instrument_id, "account_id": self.account_id}
        positions_all = tuple(cache.positions(**scope))
        if self.compact_history:
            # Do not call Position.to_dict() here: it materializes every fill,
            # adjustment, commission, venue order ID, and trade ID before the
            # caller can discard them. Long episodes would repeatedly allocate
            # the entire open-position history at every decision timestamp.
            positions = tuple(_compact_position_record(p)
                              for p in cache.positions_open(**scope))
        else:
            positions = tuple(p.to_dict() for p in cache.positions_open(**scope))
        # Pending commands may be in-flight even when not yet classified open.
        # Union is by native client_order_id, never by array position.
        pending_by_id: dict[str, dict] = {}
        for order in (*cache.orders_open(**scope), *cache.orders_inflight(**scope)):
            record = order.to_dict()
            client_id = str(record.get("client_order_id", ""))
            if not client_id:
                raise EngineBoundaryError("native pending order missing client identity")
            try:
                quantity = Decimal(str(record.get("quantity")))
                filled_qty = Decimal(str(record.get("filled_qty")))
            except (InvalidOperation, TypeError, ValueError) as exc:
                raise EngineBoundaryError("native pending order quantity facts are invalid") from exc
            if (not quantity.is_finite() or not filled_qty.is_finite()
                    or quantity <= 0 or filled_qty < 0 or filled_qty > quantity):
                raise EngineBoundaryError("native pending order filled quantity is outside order size")
            if record.get("side") not in {"BUY", "SELL"}:
                raise EngineBoundaryError("native pending order side is invalid")
            # Nautilus Order.to_dict() exposes cumulative quantity fields but
            # not leaves_qty. Derive the pending amount from those native facts
            # so the planner sees partial fills without inventing an order size.
            record["leaves_qty"] = str(quantity - filled_qty)
            existing = pending_by_id.get(client_id)
            if existing is not None and existing != record:
                raise EngineBoundaryError("inconsistent native open/inflight order state")
            pending_by_id[client_id] = record
        orders = tuple(pending_by_id[k] for k in sorted(pending_by_id))
        native_orders = tuple(sorted(cache.orders(**scope), key=lambda o: str(o.client_order_id)))
        all_orders = tuple(o.to_dict() for o in native_orders)
        from nautilus_trader.model import OrderFilled, PositionAdjustmentType
        # NETTING reuses a position ID after closing. Its current Position.events
        # then contains only the new cycle; the native order event histories
        # retain every fill, including the archived round trips.
        fills = self._unique_events(ev for order in native_orders for ev in order.events()
                                    if isinstance(ev, OrderFilled))
        liquidation_orders = None
        if self.liquidation_capability:
            venue = str(self.instrument_id.venue)
            order_prefix = f"LIQUIDATION-{venue}-"
            tag = f"LIQUIDATION_{venue}_CLOSE"
            identified = []
            for row in all_orders:
                client_id = str(row.get("client_order_id", ""))
                tags = row.get("tags") or ()
                tagged = tag in tags
                prefixed = client_id.startswith(order_prefix)
                if tagged != prefixed:
                    raise EngineBoundaryError("native liquidation order marker is inconsistent")
                if tagged:
                    identified.append(row)
            liquidation_orders = tuple(identified)
        if self.compact_history:
            funding = self._compact_funding_updates(positions_all, cache,
                                                    PositionAdjustmentType.FUNDING)
        else:
            archived = tuple(p for p in cache.position_snapshots(account_id=self.account_id)
                             if p.instrument_id == self.instrument_id)
            # Snapshots can overlap the current position; deduplicate by native event
            # identity, rejecting contradictory records rather than counting twice.
            funding = self._unique_events(adj for p in (*positions_all, *archived)
                                          for adj in p.adjustments()
                                          if adj.adjustment_type == PositionAdjustmentType.FUNDING)
        events = tuple(ev.to_dict() for ev in account.events)
        if self.compact_history:
            if events:
                self._latest_account_event = events[-1]
            if self._latest_account_event is None:
                raise EngineBoundaryError("native account balance event unavailable for compact history")
            # Observation assembly needs the latest authoritative wallet state,
            # not the account's unbounded event log. Keep that one immutable
            # ledger fact before asking Nautilus to prune the native history.
            events = (self._latest_account_event,)
        native_margins = account.margins()
        margin_values = native_margins.values() if isinstance(native_margins, Mapping) else native_margins
        margins = tuple(m.to_dict() for m in margin_values)
        from nautilus_trader.model import PriceType
        raw_mark = cache.price(self.instrument_id, PriceType.MARK)
        mark = Decimal(str(raw_mark.as_decimal())) if raw_mark is not None else None
        if mark is not None and (not mark.is_finite() or mark <= 0):
            raise EngineBoundaryError("invalid native mark")
        facts = NativeAccountFacts(
            equity, self.currency, positions, orders, account.to_dict(), events, fills,
            funding, all_orders, liquidation_orders, mark, margins, bool(native.is_stale),
            self.liquidation_capability, timestamp_ns,
        )
        if self.compact_history:
            # Closed orders/positions and old account events have already been
            # detached into this read. HistoricalEnvironmentFactory journals
            # fills, funding, liquidation identity, and cumulative quantities
            # before the next barrier. Open/inflight state is never purged.
            cache.purge_closed_orders(ts_now=timestamp_ns, buffer_secs=0)
            cache.purge_closed_positions(ts_now=timestamp_ns, buffer_secs=0)
            cache.purge_account_events(ts_now=timestamp_ns, lookback_secs=0)
        return facts

    def _compact_funding_updates(self, positions: tuple[Any, ...], cache: Any,
                                 funding_type: Any) -> tuple[dict, ...]:
        """Read funding only when a Position timestamp changes; return new facts."""
        candidates = list(positions)
        if not self._funding_initialized:
            candidates.extend(
                position for position in cache.position_snapshots(account_id=self.account_id)
                if position.instrument_id == self.instrument_id
            )
            self._funding_initialized = True
        updates: dict[str, dict] = {}
        for position in candidates:
            position_key = (str(position.id), int(position.ts_opened))
            last_update = int(position.ts_last)
            if self._position_last_updates.get(position_key) == last_update:
                continue
            self._position_last_updates[position_key] = last_update
            for adjustment in position.adjustments():
                if adjustment.adjustment_type != funding_type:
                    continue
                record = adjustment.to_dict()
                identity = str(record.get("event_id", ""))
                if not identity:
                    raise EngineBoundaryError("native funding event identity missing")
                previous = self._funding_facts.get(identity)
                if previous is not None:
                    if previous != record:
                        raise EngineBoundaryError("native funding identity has conflicting facts")
                    continue
                self._funding_facts[identity] = record
                updates[identity] = record
        return tuple(sorted(updates.values(),
                            key=lambda row: (row["ts_init"], row["ts_event"], row["event_id"])))

    @staticmethod
    def _unique_events(events: Iterable[Any]) -> tuple[dict, ...]:
        by_id: dict[str, dict] = {}
        for event in events:
            record = event.to_dict()
            key = str(record.get("event_id", ""))
            if not key:
                raise EngineBoundaryError("native ledger event identity missing")
            if key in by_id and by_id[key] != record:
                raise EngineBoundaryError("native ledger event identity has conflicting facts")
            by_id[key] = record
        return tuple(sorted(by_id.values(), key=lambda r: (r["ts_init"], r["ts_event"], r["event_id"])))


def _compact_position_record(position: Any) -> dict:
    """Serialize only the stable scalar Position fields used by observations."""
    base_currency = position.base_currency
    quote_currency = position.quote_currency
    settlement_currency = position.settlement_currency
    realized_pnl = position.realized_pnl
    return {
        "type": "Position",
        "trader_id": str(position.trader_id),
        "strategy_id": str(position.strategy_id),
        "instrument_id": str(position.instrument_id),
        "position_id": str(position.id),
        "account_id": str(position.account_id),
        "opening_order_id": str(position.opening_order_id),
        "closing_order_id": (str(position.closing_order_id)
                              if position.closing_order_id is not None else None),
        "entry": str(position.entry),
        "side": str(position.side),
        "signed_qty": position.signed_qty,
        "quantity": str(position.quantity),
        "peak_qty": str(position.peak_qty),
        "price_precision": position.price_precision,
        "size_precision": position.size_precision,
        "multiplier": str(position.multiplier),
        "is_inverse": position.is_inverse,
        "base_currency": base_currency.code if base_currency is not None else None,
        "quote_currency": quote_currency.code if quote_currency is not None else None,
        "settlement_currency": (settlement_currency.code
                                if settlement_currency is not None else None),
        "ts_init": position.ts_init,
        "ts_opened": position.ts_opened,
        "ts_last": position.ts_last,
        "ts_closed": position.ts_closed,
        "duration_ns": position.duration_ns,
        "avg_px_open": position.avg_px_open,
        "avg_px_close": position.avg_px_close,
        "realized_return": position.realized_return,
        "realized_pnl": str(realized_pnl) if realized_pnl is not None else None,
    }

class EngineSession:
    """One immutable fold -> one native engine. No reset across folds/candidates."""
    BARRIER_PROTOCOL = "nt2rc6_stream_complete_timestamp_v1"

    def __init__(self, *, build_native: Callable[[Any], Any],
                 submit_plan: Callable[[Any, Any], None],
                 decode_event: Callable[[Any], Any],
                 facts_reader: NativeFactsReader,
                 contract_snapshot_factory: Callable[..., Any],
                 certified_barrier_protocol: str,
                 prepare_submit: Callable[[Any, Any, int], None] | None = None) -> None:
        if certified_barrier_protocol != self.BARRIER_PROTOCOL:
            raise EngineBoundaryError("native barrier protocol not approved")
        for callback in (build_native, submit_plan, decode_event, contract_snapshot_factory):
            if not callable(callback):
                raise TypeError("reviewed native bindings and L1 snapshot constructor required")
        if not isinstance(facts_reader, NativeFactsReader):
            raise TypeError("native facts reader required")
        if prepare_submit is not None and not callable(prepare_submit):
            raise TypeError("optional native submit preparation callback must be callable")
        self._build, self._submit, self._decode = build_native, submit_plan, decode_event
        self._read, self._make_snapshot = facts_reader, contract_snapshot_factory
        self._prepare_submit = prepare_submit
        self._engine: Any | None = None
        self._state = "CREATED"
        self._last_ts = 0
        self._action_ids: set[str] = set()
        self._event_ids: set[str] = set()
        self._fingerprint: str | None = None

    @property
    def state(self) -> str:
        return self._state

    @property
    def fingerprint(self) -> str:
        if self._fingerprint is None:
            raise EngineBoundaryError("engine not open")
        return self._fingerprint

    def open(self, episode: Any) -> str:
        global _NATIVE_SESSION_OWNER
        if self._state != "CREATED":
            raise EngineBoundaryError("single open per native EngineSession")
        if sys.version_info[:3] != (3, 14, 7):
            raise EngineBoundaryError("research runtime must use frozen CPython 3.14.7")
        version = importlib.metadata.version("nautilus_trader")
        if version != "2.0.0rc6":
            raise EngineBoundaryError(f"unapproved NautilusTrader version: {version}")
        from nautilus_trader.backtest import BacktestEngine
        try:
            dataset_hash = getattr(episode, "dataset_hash", None)
            fold_hash = getattr(episode, "fold_hash", None)
            config_hash = getattr(episode, "config_hash", None)
            if (not isinstance(dataset_hash, str) or _SHA256_RE.fullmatch(dataset_hash) is None
                or not isinstance(fold_hash, str) or _SHA256_RE.fullmatch(fold_hash) is None
                or not isinstance(config_hash, str) or _SHA256_RE.fullmatch(config_hash) is None):
                raise EngineBoundaryError("episode must bind lowercase SHA-256 data/fold/config hashes")
            inputs = (dataset_hash, fold_hash, config_hash)
            # rc6 uses process/thread-global component and message registries.
            # Two live engines can route orders into the wrong native account.
            # Parallel episodes therefore belong in isolated worker processes.
            with _NATIVE_SESSION_LOCK:
                if _NATIVE_SESSION_OWNER is not None:
                    raise EngineBoundaryError("one live native EngineSession per process; use isolated workers")
                _NATIVE_SESSION_OWNER = self
            engine = self._build(episode)
            if not isinstance(engine, BacktestEngine):
                raise EngineBoundaryError("engine factory must return official BacktestEngine")
            self._engine = engine
            # Not the ephemeral engine.instance_id: identical runs require stable
            # hashes, while the native instance ID is only execution provenance.
            self._fingerprint = sha256(("|".join((*inputs, version, self.BARRIER_PROTOCOL))).encode()).hexdigest()
            self._state = "READY"
            return self._fingerprint
        except BaseException:
            self.close()
            raise

    def _preflight(self, batches: Iterable[VisibleBatch], *, end_ts_ns: int
                   ) -> tuple[tuple[VisibleBatch, tuple[Any, ...]], ...]:
        if self._engine is None or type(end_ts_ns) is not int or end_ts_ns <= self._last_ts:
            raise EngineBoundaryError("invalid native engine or advancing barrier")
        prepared: list[tuple[VisibleBatch, tuple[Any, ...]]] = []
        ts = self._last_ts
        seen: set[str] = set()
        for batch in batches:
            if not isinstance(batch, VisibleBatch) or batch.timestamp_ns <= ts or batch.timestamp_ns > end_ts_ns:
                raise EngineBoundaryError("unordered or incomplete timestamp batch")
            ts = batch.timestamp_ns
            decoded: list[Any] = []
            for event in batch.events:
                eid = str(event.event_id)
                if eid in seen or eid in self._event_ids:
                    raise EngineBoundaryError("duplicate native source event identity")
                seen.add(eid)
                native = self._decode(event)
                if native is None or type(getattr(native, "ts_init", None)) is not int \
                        or native.ts_init != batch.timestamp_ns:
                    raise EngineBoundaryError("decoded native event replay timestamp differs")
                decoded.append(native)
            prepared.append((batch, tuple(decoded)))
        if not prepared or ts != end_ts_ns:
            raise EngineBoundaryError("next decision clock must have one complete final timestamp batch")
        return tuple(prepared)

    def _replay(self, prepared: tuple[tuple[VisibleBatch, tuple[Any, ...]], ...]) -> None:
        if self._engine is None:
            raise EngineBoundaryError("native engine missing")
        # Every batch was preflighted as a complete timestamp. The native engine
        # still processes each event on its own clocks, but one streaming run
        # per decision interval avoids tens of thousands of runner restarts for
        # second-cadence quote histories. No account snapshot is exposed between
        # these batches; the stable boundary remains the interval end.
        replay_events = [event for _batch, events in prepared for event in events]
        self._engine.add_data(replay_events)
        self._engine.run(streaming=True)
        self._engine.clear_data()
        for batch, _native_events in prepared:
            self._last_ts = batch.timestamp_ns
            self._event_ids.update(str(e.event_id) for e in batch.events)

    def _native_facts(self) -> NativeAccountFacts:
        if self._engine is None:
            raise EngineBoundaryError("native engine missing")
        return self._read.read(self._engine, self._last_ts)

    def _snapshot_of(self, facts: NativeAccountFacts) -> Any:
        return self._make_snapshot(
            equity=facts.equity, currency=facts.currency, positions=facts.positions,
            orders=facts.orders, account=facts.account,
            account_events=facts.account_events, fills=facts.fills,
            funding=facts.funding_adjustments, all_orders=facts.orders_all,
            liquidation_orders=facts.liquidation_orders,
            margins=facts.margins, mark=facts.mark_price,
            liquidation_capability=facts.liquidation_capability,
            decision_ts_ns=facts.timestamp_ns,
        )

    def bootstrap(self, batches: Iterable[VisibleBatch], *, decision_ts_ns: int) -> Any:
        if self._state != "READY":
            raise EngineBoundaryError("bootstrap requires READY")
        try:
            prepared = self._preflight(batches, end_ts_ns=decision_ts_ns)
            self._replay(prepared)
            self._state = "RUNNING"
            return self.snapshot()
        except BaseException:
            self.close()
            raise

    def _plan_action(self, plan: Any, *, allow_cancel: bool = True) -> str:
        from rles_contracts.dto import ExecutionPlan, OrderType, TimeInForce

        if type(plan) is not ExecutionPlan:
            raise EngineBoundaryError("native submit requires frozen L1 ExecutionPlan")
        for required in ("decision_id", "latency_model_id", "execution_spec_hash"):
            if not isinstance(getattr(plan, required, None), str) or not getattr(plan, required):
                raise EngineBoundaryError(f"reviewed ExecutionPlan.{required} required")
        if plan.order_type is not OrderType.MARKET_LIKE or plan.tif is not TimeInForce.GTC:
            raise EngineBoundaryError("unsupported native order type/TIF; execution spec mismatch")
        action = getattr(plan, "action", None)
        if action not in {"SUBMIT", "CANCEL", "WAIT", "NOOP"}:
            raise EngineBoundaryError("unrecognized execution plan action")
        key = getattr(plan, "idempotency_key", None)
        if not isinstance(key, str) or not key or key in self._action_ids:
            raise EngineBoundaryError("duplicate or missing action idempotency identity")
        if action == "SUBMIT":
            qty = getattr(plan, "delta_qty", None)
            if isinstance(qty, float) or qty is None:
                raise EngineBoundaryError("native order quantity must be exact Decimal")
            quantity = Decimal(str(qty))
            if not quantity.is_finite() or quantity == 0:
                raise EngineBoundaryError("submit quantity must be finite nonzero")
            if getattr(plan, "cancel_order_ids", ()) or getattr(plan, "requires_cancel_ack", False):
                raise EngineBoundaryError("submit and cancel/ack in same step forbidden")
            if getattr(plan, "cancel_policy", None) != "none":
                raise EngineBoundaryError("submit plan must not imply cancellation")
        elif action == "CANCEL":
            order_ids = getattr(plan, "cancel_order_ids", ())
            if not allow_cancel or not getattr(plan, "requires_cancel_ack", False) or not order_ids:
                raise EngineBoundaryError("cancel requires explicit native ACK protocol")
            if len(set(order_ids)) != len(order_ids) or getattr(plan, "delta_qty", None) != 0:
                raise EngineBoundaryError("cancel must contain distinct IDs and zero order quantity")
        elif getattr(plan, "delta_qty", Decimal(0)) != 0:
            raise EngineBoundaryError("non-trading plan cannot carry positive order quantity")
        return action

    def advance(self, plan: Any, *, batches: Iterable[VisibleBatch], next_decision_ts_ns: int) -> Any:
        if self._state != "RUNNING":
            raise EngineBoundaryError("advance requires RUNNING state")
        try:
            prepared = self._preflight(batches, end_ts_ns=next_decision_ts_ns)
            action = self._plan_action(plan)
            live = self._native_facts().orders
            if action == "SUBMIT" and live:
                raise EngineBoundaryError("native pending orders must resolve before new submit")
            if action == "CANCEL":
                requested = set(plan.cancel_order_ids)
                # Upstream must prove that every cancel ID belongs to a currently
                # open native order; otherwise failure cannot be mistaken for ACK.
                live_ids = {str(d.get("client_order_id", "")) for d in live}
                if requested != live_ids:
                    raise EngineBoundaryError("cancel list must cover native open and in-flight orders")
            key = plan.idempotency_key
            self._action_ids.add(key)  # reserve before any side effect
            if action in {"SUBMIT", "CANCEL"}:
                if action == "SUBMIT" and self._prepare_submit is not None:
                    self._prepare_submit(self._engine, plan, self._last_ts)
                self._submit(self._engine, plan)
            self._replay(prepared)
            return self.snapshot()
        except BaseException:
            self.close()
            raise

    def snapshot(self) -> Any:
        if self._state not in {"RUNNING", "DRAINING"}:
            raise EngineBoundaryError("native snapshot only at stable decision barrier")
        return self._snapshot_of(self._native_facts())

    def drain_and_close(self, terminal_policy: str, *, flatten_plan: Any,
                        drain_batches: Iterable[VisibleBatch], terminal_ts_ns: int,
                        cancel_plan: Any | None = None,
                        cancel_batches: Iterable[VisibleBatch] | None = None,
                        cancel_ts_ns: int | None = None) -> Any:
        if self._state != "RUNNING" or terminal_policy != "force_flatten_with_cost":
            raise EngineBoundaryError("native terminal close protocol invalid")
        self._state = "DRAINING"
        try:
            pending = self._native_facts().orders
            if pending:
                if cancel_plan is None or cancel_batches is None or cancel_ts_ns is None:
                    raise EngineBoundaryError("terminal pending orders require explicit native cancel+ACK")
                prepared_cancel = self._preflight(cancel_batches, end_ts_ns=cancel_ts_ns)
                cancellation = self._plan_action(cancel_plan)
                if cancellation == "CANCEL":
                    pending_ids = {str(o.get("client_order_id", "")) for o in pending}
                    if set(cancel_plan.cancel_order_ids) != pending_ids:
                        raise EngineBoundaryError("terminal cancel must cover every native pending order")
                if cancellation not in {"CANCEL", "WAIT"}:
                    raise EngineBoundaryError("terminal pending orders require CANCEL or existing ACK wait")
                self._action_ids.add(cancel_plan.idempotency_key)
                if cancellation == "CANCEL":
                    self._submit(self._engine, cancel_plan)
                self._replay(prepared_cancel)
                if self._native_facts().orders:
                    raise EngineBoundaryError("terminal cancellation not acknowledged")
            elif cancel_plan is not None or cancel_batches is not None or cancel_ts_ns is not None:
                raise EngineBoundaryError("unexpected terminal cancel artifacts")
            # A pending order can fill before its cancel ACK. Replan the exit
            # from the post-ACK native account, rather than using stale quantity.
            if callable(flatten_plan):
                flatten_plan = flatten_plan(self._snapshot_of(self._native_facts()))
            prepared_close = self._preflight(drain_batches, end_ts_ns=terminal_ts_ns)
            terminal_action = self._plan_action(flatten_plan, allow_cancel=False)
            if terminal_action != "SUBMIT" and (terminal_action != "NOOP" or self._native_facts().positions):
                raise EngineBoundaryError("terminal exit must submit close or prove flat account")
            if flatten_plan.idempotency_key in self._action_ids:
                raise EngineBoundaryError("terminal action idempotency key reused")
            self._action_ids.add(flatten_plan.idempotency_key)
            if flatten_plan.action == "SUBMIT":
                if self._prepare_submit is not None:
                    self._prepare_submit(self._engine, flatten_plan, self._last_ts)
                from .submitter import NativePlanSubmitter
                if isinstance(self._submit, NativePlanSubmitter):
                    self._submit.submit_flatten(self._engine, flatten_plan)
                else:
                    self._submit(self._engine, flatten_plan)
            self._replay(prepared_close)
            if self._engine is None:
                raise EngineBoundaryError("native engine disappeared before terminal shutdown")
            self._engine.end()  # Official native shutdown/funding/fill drain
            facts = self._native_facts()
            if facts.positions or facts.orders:
                raise EngineBoundaryError("unsettled terminal orders or positions; score BLOCKED")
            return self._snapshot_of(facts)
        finally:
            self.close()

    def close(self) -> None:
        global _NATIVE_SESSION_OWNER
        try:
            if self._engine is not None:
                self._engine.dispose()
        finally:
            self._engine = None
            self._state = "CLOSED"
            with _NATIVE_SESSION_LOCK:
                if _NATIVE_SESSION_OWNER is self:
                    _NATIVE_SESSION_OWNER = None


def _create_engine(settings: Mapping[str, object]) -> EngineSession:
    required = {
        "build_native", "submit_plan", "decode_event", "facts_reader",
        "contract_snapshot_factory", "certified_barrier_protocol",
    }
    optional = {"prepare_submit"}
    if not required <= set(settings) or set(settings) - required - optional:
        raise ValueError("engine provider settings must match the frozen constructor fields")
    return EngineSession(**settings)  # type: ignore[arg-type]


provider = ProviderDescriptor(
    kind="engine",
    contract_version="0.4.0",
    factory=_create_engine,
)
