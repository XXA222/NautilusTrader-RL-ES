"""Exact L1 plans submitted through the pinned native Strategy public API."""
from __future__ import annotations

from decimal import Decimal
from hashlib import sha256
import re
from typing import Any

from rles_contracts.dto import ExecutionPlan, OrderType, TimeInForce

from .session import EngineBoundaryError


class NativePlanSubmitter:
    """One engine/strategy binding, with immutable execution contract identity.

    Construct together with PublicBacktestBuilder for each episode. The native
    engine exposes no public strategy lookup in rc6; the builder binds the actual
    registered strategy. No order matching or account arithmetic lives here.
    """

    def __init__(self, *, instrument_id: str, quantity_unit: str,
                 quantity_increment: Decimal, execution_spec_hash: str,
                 latency_model_id: str) -> None:
        if not instrument_id or not quantity_unit or not latency_model_id:
            raise ValueError("frozen instrument/unit/latency identity required")
        if type(quantity_increment) is not Decimal or not quantity_increment.is_finite() or quantity_increment <= 0:
            raise ValueError("exact positive native quantity increment required")
        if not isinstance(execution_spec_hash, str) or re.fullmatch(r"[0-9a-f]{64}", execution_spec_hash) is None:
            raise ValueError("frozen execution spec SHA-256 required")
        self._identity = (instrument_id, quantity_unit, quantity_increment,
                          execution_spec_hash, latency_model_id)
        self._engine: Any = None
        self._strategy: Any = None
        self._used: set[str] = set()

    def bind(self, engine: Any, strategy: Any) -> None:
        from nautilus_trader.backtest import BacktestEngine
        from nautilus_trader.model import InstrumentId
        from nautilus_trader.trading import Strategy

        if self._engine is not None:
            raise EngineBoundaryError("native submitter cannot be rebound across episodes")
        if not isinstance(engine, BacktestEngine) or not isinstance(strategy, Strategy):
            raise EngineBoundaryError("official native engine and strategy required")
        instrument = engine.cache.instrument(InstrumentId.from_str(self._identity[0]))
        if instrument is None:
            raise EngineBoundaryError("frozen instrument absent from native cache")
        if (str(instrument.base_currency) != self._identity[1]
                or instrument.size_increment.as_decimal() != self._identity[2]):
            raise EngineBoundaryError("native quantity unit/increment differs from frozen contract")
        if strategy.trader_id != engine.trader_id:
            raise EngineBoundaryError("strategy is not registered with native trader")
        self._engine, self._strategy = engine, strategy

    def __call__(self, engine: Any, plan: ExecutionPlan) -> None:
        self._execute(engine, plan, flatten=False)

    def submit_flatten(self, engine: Any, plan: ExecutionPlan) -> None:
        """Close exactly the sole owned native position with reduce-only."""
        self._execute(engine, plan, flatten=True)

    def _execute(self, engine: Any, plan: ExecutionPlan, *, flatten: bool) -> None:
        from nautilus_trader.model import ClientOrderId, InstrumentId, OrderSide, Quantity
        from nautilus_trader.model import PositionSide
        from nautilus_trader.model import TimeInForce as NativeTimeInForce

        if self._engine is None or engine is not self._engine or not self._strategy.is_running():
            raise EngineBoundaryError("native submit requires its bound running episode")
        if type(plan) is not ExecutionPlan:
            raise EngineBoundaryError("native submit requires frozen L1 ExecutionPlan")
        if (plan.instrument_id, plan.quantity_unit, plan.quantity_increment,
                plan.execution_spec_hash, plan.latency_model_id) != self._identity:
            raise EngineBoundaryError("plan differs from frozen native execution contract")
        if plan.order_type is not OrderType.MARKET_LIKE or plan.tif is not TimeInForce.GTC:
            raise EngineBoundaryError("only native market GTC execution supported")
        if plan.idempotency_key in self._used:
            raise EngineBoundaryError("native action idempotency key already used")
        instrument_id = InstrumentId.from_str(plan.instrument_id)
        policy_close = plan.action == "SUBMIT" and plan.reason == "policy_flat"
        risk_reduction = plan.action == "SUBMIT" and plan.reason in {"hard_risk_reduce", "policy_flat"}
        if flatten or risk_reduction:
            positions = tuple(engine.cache.positions_open(instrument_id=instrument_id))
            if (plan.action != "SUBMIT" or len(positions) != 1
                    or positions[0].strategy_id != self._strategy.strategy_id):
                raise EngineBoundaryError("reduce-only action requires one owned native position")
            position = positions[0]
            quantity = position.quantity.as_decimal()
            if position.side == PositionSide.LONG:
                expected = quantity.copy_negate()
            elif position.side == PositionSide.SHORT:
                expected = quantity
            else:
                raise EngineBoundaryError("terminal native position side is not closeable")
            if (flatten or policy_close) and plan.delta_qty != expected:
                raise EngineBoundaryError("terminal flatten quantity differs from native position")
            if risk_reduction and (plan.delta_qty * expected <= 0 or abs(plan.delta_qty) > quantity):
                raise EngineBoundaryError("hard-risk reduction must reduce the owned native position")
        if plan.action == "SUBMIT":
            instrument = engine.cache.instrument(instrument_id)
            try:
                quantity = Quantity.from_decimal_dp(plan.delta_qty.copy_abs(), instrument.size_precision)
            except (ValueError, OverflowError) as exc:
                raise EngineBoundaryError("quantity outside exact native representation") from exc
            if quantity.as_decimal() != plan.delta_qty.copy_abs():
                raise EngineBoundaryError("native quantity conversion would round execution plan")
            client_id = ClientOrderId("RLES-" + sha256(plan.idempotency_key.encode("utf-8")).hexdigest())
            if engine.cache.order(client_id) is not None:
                raise EngineBoundaryError("native client order identity collision")
            order = self._strategy.order_factory.market(
                instrument_id=instrument_id,
                order_side=OrderSide.BUY if plan.delta_qty > 0 else OrderSide.SELL,
                quantity=quantity, time_in_force=NativeTimeInForce.GTC,
                client_order_id=client_id, reduce_only=flatten or risk_reduction,
            )
            self._used.add(plan.idempotency_key)
            self._strategy.submit_order(order)
        elif plan.action == "CANCEL":
            ids = tuple(ClientOrderId(value) for value in plan.cancel_order_ids)
            # Validate the complete list before issuing its first command.
            for client_id in ids:
                order = engine.cache.order(client_id)
                if (order is None or order.instrument_id != instrument_id
                        or order.strategy_id != self._strategy.strategy_id
                        or order.is_closed):
                    raise EngineBoundaryError("cancel target is not this strategy's live native order")
            self._used.add(plan.idempotency_key)
            for client_id in ids:
                self._strategy.cancel_order(client_id)
        elif plan.action in {"NOOP", "WAIT"}:
            self._used.add(plan.idempotency_key)
        else:
            raise EngineBoundaryError("unsupported native execution plan action")
