from __future__ import annotations

from decimal import Decimal
from dataclasses import replace

import pytest

from rles_contracts.dto import ExecutionPlan, PendingOrderPolicy
from rles_execution import (
    FillProfile,
    InstrumentRules,
    LatencyProfile,
    PendingOrder,
    PlanningBlocked,
    ResolvedExecutionConfig,
    TargetQuantityPlanner,
    to_contract_plan,
    resolve_scenario,
)
from rles_nautilus_adapter.session import EngineSession
from rles_risk import HardRiskConfig, HardRiskGuard


pytestmark = pytest.mark.contracts


def _planner() -> TargetQuantityPlanner:
    config = ResolvedExecutionConfig(
        schema_version="0.4.0", deadband=Decimal("0"),
        pending_order_policy="cancel_replace", order_type="MARKET", tif="GTC",
        terminal_policy="force_flatten_with_cost",
        base_latency=LatencyProfile("base", 0, 0, 0),
        stress_latency=LatencyProfile("stress", 1, 1, 1),
        base_fill=FillProfile("base", Decimal("1")),
        stress_fill=FillProfile("stress", Decimal("2")),
    )
    return TargetQuantityPlanner(config)


def _decision(*, target: str, current: str = "0",
              pending: tuple[PendingOrder, ...] = ()):
    return _planner().plan(
        decision_id="decision-1",
        instrument=InstrumentRules("BTC/USD.SIM", Decimal("0.0001"),
                                   Decimal("0.0001"), Decimal("5"), Decimal("1")),
        target_exposure=Decimal(target), risk_budget_notional=Decimal("1000"),
        equity=Decimal("1000"), mark_price=Decimal("100"),
        current_qty=Decimal(current), pending=pending,
    )


@pytest.mark.parametrize("sign", [-1, 1])
def test_explicit_zero_target_exits_small_position_without_deadband(sign):
    legacy = replace(_planner().config, deadband=Decimal("0.10"))
    rules = InstrumentRules("BTC/USD.SIM", Decimal("0.0001"), Decimal("0.0001"),
                            Decimal("5"), Decimal("1"))
    args = dict(decision_id="voluntary-flat", instrument=rules, target_exposure=Decimal(0),
                risk_budget_notional=Decimal("1000"), equity=Decimal("1000"),
                mark_price=Decimal("100"), current_qty=Decimal("0.5") * sign, pending=())
    assert TargetQuantityPlanner(legacy).plan(**args).reason == "deadband"
    explicit = replace(legacy, zero_target_policy="close")
    decision = TargetQuantityPlanner(explicit).plan(**args)
    assert (decision.action, decision.reason, decision.delta_qty) == ("SUBMIT", "policy_flat", -args["current_qty"])
    assert legacy.fingerprint() != explicit.fingerprint()


def test_explicit_flat_cancels_pending_orders_and_respects_dated_minimums():
    planner = TargetQuantityPlanner(replace(_planner().config, deadband=Decimal("0.10"),
                                            zero_target_policy="close", pending_order_policy="wait"))
    rules = InstrumentRules("BTC/USD.SIM", Decimal("0.0001"), Decimal("0.0001"),
                            Decimal("20"), Decimal("1"))
    args = dict(decision_id="flat-dust", instrument=rules, target_exposure=Decimal(0),
                risk_budget_notional=Decimal("1000"), equity=Decimal("1000"),
                mark_price=Decimal("100"), current_qty=Decimal("0.1"), pending=())
    blocked = planner.plan(**args)
    assert blocked.action == "NOOP" and "historical minimum" in blocked.reason
    allowed = planner.plan(**{**args, "instrument": replace(rules, reduce_only_min_notional_exempt=True)})
    assert allowed.action == "SUBMIT" and allowed.reason == "policy_flat"
    pending = (PendingOrder("native-1", Decimal("1"), "ACCEPTED"),)
    assert planner.plan(**{**args, "current_qty": Decimal(0), "pending": pending}).action == "CANCEL"
    assert planner.plan(**{**args, "pending": (replace(pending[0], status="PENDING_CANCEL"),)}).action == "WAIT"


@pytest.mark.parametrize(("target", "pending", "action", "policy"), (
    ("0.5", (), "SUBMIT", PendingOrderPolicy.NONE),
    ("0", (), "NOOP", PendingOrderPolicy.NONE),
    ("0.5", (PendingOrder("native-1", Decimal("1"), "ACCEPTED"),),
     "CANCEL", PendingOrderPolicy.CANCEL_REPLACE),
))
def test_l3_decision_constructs_frozen_l1_plan(target, pending, action, policy) -> None:
    decision = _decision(target=target, pending=pending)
    plan = to_contract_plan(decision, ExecutionPlan,
                            latency_model_id="base", quantity_unit="BTC")
    assert type(plan) is ExecutionPlan
    assert plan.action == action
    assert plan.cancel_policy is policy
    assert plan.quantity_increment == Decimal("0.0001")
    assert plan.execution_spec_hash == _planner().config.fingerprint()
    assert plan.idempotency_key == decision.idempotency_key
    assert plan.cancel_order_ids == decision.cancel_order_ids
    session = object.__new__(EngineSession)
    session._action_ids = set()
    assert session._plan_action(plan) == action


def test_l1_plan_rejects_cancel_without_ack() -> None:
    decision = _decision(
        target="0.5", pending=(PendingOrder("native-1", Decimal("1"), "ACCEPTED"),))
    plan = to_contract_plan(decision, ExecutionPlan,
                            latency_model_id="base", quantity_unit="BTC")
    from dataclasses import replace
    with pytest.raises(ValueError, match="cancel requires"):
        replace(plan, requires_cancel_ack=False)


def test_quote_execution_requires_zero_synthetic_slippage() -> None:
    with pytest.raises(ValueError, match="native displayed depth"):
        FillProfile("quote-slipped", Decimal("1"), "quote")


def test_fee_stress_is_hashed_and_only_increases_positive_fees() -> None:
    config = replace(_planner().config, stress_fill=FillProfile("stress", Decimal("2"), fee_multiplier=Decimal("2")))
    assert config.base_fill.fee_multiplier == 1
    assert config.fingerprint() != _planner().config.fingerprint()
    scenario = resolve_scenario(config, "stress")
    assert scenario.fee_multiplier == "2"
    assert resolve_scenario(config, "base").fee_multiplier == "1"
    from rles_nautilus_adapter.perpetual import _effective_fee_rates

    rule = {"maker_fee": "0.0002", "taker_fee": "0.0005"}
    assert _effective_fee_rates(rule, config, "base") == (Decimal("0.0002"), Decimal("0.0005"))
    assert _effective_fee_rates(rule, config, "stress") == (Decimal("0.0004"), Decimal("0.0010"))
    rebate = {"maker_fee": "-0.0001", "taker_fee": "0.0005"}
    assert _effective_fee_rates(rebate, config, "stress") == (Decimal("-0.0001"), Decimal("0.0010"))
    assert rule["maker_fee"] == "0.0002"  # source facts remain untouched


def test_partial_depth_stress_is_explicit_and_bound_to_execution_identity() -> None:
    config = replace(
        _planner().config,
        stress_fill=FillProfile("depth-stress", Decimal("2"),
                                simulated_depth_fraction=Decimal("0.4")),
    )
    assert config.fingerprint() != _planner().config.fingerprint()
    scenario = resolve_scenario(config, "stress")
    assert scenario.simulated_depth_fraction == "0.4"
    assert scenario.scenario_hash != resolve_scenario(_planner().config, "stress").scenario_hash
    assert FillProfile("base", Decimal("1")).simulated_depth_fraction == Decimal(1)


def test_partial_depth_is_bounded_and_quote_depth_cannot_be_synthetic() -> None:
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        FillProfile("invalid-depth", Decimal("2"), simulated_depth_fraction=Decimal("1.1"))
    with pytest.raises(ValueError, match="base execution must retain"):
        replace(_planner().config,
                base_fill=FillProfile("partial-base", Decimal("1"),
                                      simulated_depth_fraction=Decimal("0.5")))
    with pytest.raises(ValueError, match="native displayed depth"):
        FillProfile("synthetic-quote-cap", Decimal(0), "quote",
                    simulated_depth_fraction=Decimal("0.5"))


def test_default_fee_multipliers_preserve_legacy_execution_fingerprint() -> None:
    assert _planner().config.fingerprint() == (
        "5c45e562d7fd7171d3aaa40cc25a19e3f65980ada7e000bade7ac751c5da0434"
    )


def test_fee_multiplier_must_be_at_least_one() -> None:
    with pytest.raises(ValueError, match="more optimistic"):
        FillProfile("optimistic-fee", Decimal("1"), fee_multiplier=Decimal("0.9"))


def test_subminimum_eth_target_is_flattened_instead_of_leaving_dust() -> None:
    planner = _planner()
    instrument = InstrumentRules(
        "ETHUSDT-PERP.SIM", Decimal("0.001"), Decimal("0.001"), Decimal("20"), Decimal("1"),
    )
    decision = planner.plan(
        decision_id="eth-dust-prevention", instrument=instrument,
        target_exposure=Decimal("0.01"), risk_budget_notional=Decimal("1000"),
        equity=Decimal("1000"), mark_price=Decimal("2000"), current_qty=Decimal("0.02"),
        pending=(),
    )
    assert decision.action == "SUBMIT"
    assert decision.target_qty == 0
    assert decision.delta_qty == Decimal("-0.02")
    assert abs(decision.delta_qty * Decimal("2000")) >= instrument.min_notional


def test_subminimum_eth_open_target_stays_flat() -> None:
    planner = _planner()
    instrument = InstrumentRules(
        "ETHUSDT-PERP.SIM", Decimal("0.001"), Decimal("0.001"), Decimal("20"), Decimal("1"),
    )
    decision = planner.plan(
        decision_id="eth-subminimum-open", instrument=instrument,
        target_exposure=Decimal("0.01"), risk_budget_notional=Decimal("1000"),
        equity=Decimal("1000"), mark_price=Decimal("2000"), current_qty=Decimal("0"),
        pending=(),
    )
    assert decision.action == "NOOP"
    assert decision.target_qty == 0


def test_terminal_reduce_only_close_uses_explicit_min_notional_exception() -> None:
    planner = _planner()
    instrument = InstrumentRules(
        "ETHUSDT-PERP.SIM", Decimal("0.001"), Decimal("0.001"), Decimal("20"),
        Decimal("1"), reduce_only_min_notional_exempt=True,
    )
    decision = planner.plan(
        decision_id="terminal-reduce-only-dust", instrument=instrument,
        target_exposure=Decimal("0"), risk_budget_notional=Decimal("1000"),
        equity=Decimal("1000"), mark_price=Decimal("2000"), current_qty=Decimal("0.005"),
        pending=(), terminal=True,
    )
    assert decision.action == "SUBMIT"
    assert decision.reason == "terminal_close"
    assert decision.delta_qty == Decimal("-0.005")


def test_terminal_close_min_notional_exception_stays_opt_in() -> None:
    planner = _planner()
    instrument = InstrumentRules(
        "ETHUSDT-PERP.SIM", Decimal("0.001"), Decimal("0.001"), Decimal("20"), Decimal("1"),
    )
    with pytest.raises(PlanningBlocked, match="terminal close violates"):
        planner.plan(
            decision_id="terminal-close-no-exception", instrument=instrument,
            target_exposure=Decimal("0"), risk_budget_notional=Decimal("1000"),
            equity=Decimal("1000"), mark_price=Decimal("2000"), current_qty=Decimal("0.005"),
            pending=(), terminal=True,
        )


@pytest.mark.parametrize("sign", [1, -1])
@pytest.mark.parametrize("requested", ["1", "0.5"])
def test_hard_cap_reduction_bypasses_deadband(sign, requested) -> None:
    guard = HardRiskGuard(HardRiskConfig(
        Decimal("500"), Decimal("1"), Decimal("0.1"), Decimal("1"), Decimal("1"),
    ))
    risk = guard.apply(
        Decimal(requested) * sign, budget_notional=Decimal("1000"), equity=Decimal("10000"),
        free_margin=Decimal("10000"), current_notional=Decimal("520") * sign,
        mark_available=True, funding_available=True, liquidation_capability=True, data_quality_grade="DQ-B",
    )
    assert risk.adjusted == Decimal("0.5") * sign
    assert risk.force_reduce
    planner = TargetQuantityPlanner(replace(_planner().config, deadband=Decimal("0.05")))
    kwargs = dict(
        decision_id="hard-cap", instrument=InstrumentRules(
            "ETHUSDT-PERP.SIM", Decimal("0.001"), Decimal("0.001"), Decimal("20"), Decimal("1")),
        target_exposure=risk.adjusted, risk_budget_notional=Decimal("1000"),
        equity=Decimal("10000"), mark_price=Decimal("2000"),
        current_qty=Decimal("0.26") * sign, pending=(),
    )
    assert planner.plan(**kwargs).reason == "deadband"
    decision = planner.plan(**kwargs, force_reduce=risk.force_reduce)
    assert decision.action == "SUBMIT"
    assert decision.reason == "hard_risk_reduce"
    assert decision.delta_qty == Decimal("-0.01") * sign


@pytest.mark.parametrize("requested", ["0", "0.02"])
@pytest.mark.parametrize("equity", ["0", "1"])
def test_equity_halt_requires_reduction_even_when_policy_requests_flat(requested, equity) -> None:
    risk = HardRiskGuard(HardRiskConfig(
        Decimal("500"), Decimal("1"), Decimal("0.1"), Decimal("1"), Decimal("1"),
    )).apply(
        Decimal(requested), budget_notional=Decimal("1000"), equity=Decimal(equity),
        free_margin=Decimal("0"), current_notional=Decimal("20"),
        mark_available=True, funding_available=True, liquidation_capability=True, data_quality_grade="DQ-B",
    )
    assert risk.adjusted == 0 and risk.force_reduce
    planner = TargetQuantityPlanner(replace(_planner().config, deadband=Decimal("0.05")))
    decision = planner.plan(
        decision_id="equity-halt", instrument=InstrumentRules(
            "ETHUSDT-PERP.SIM", Decimal("0.001"), Decimal("0.001"), Decimal("20"), Decimal("1")),
        target_exposure=risk.adjusted, risk_budget_notional=Decimal("1000"),
        equity=Decimal(equity), mark_price=Decimal("2000"), current_qty=Decimal("0.01"),
        pending=(), force_reduce=risk.force_reduce,
    )
    assert decision.action == "SUBMIT" and decision.delta_qty == Decimal("-0.01")


@pytest.mark.parametrize("exempt", [False, True])
def test_hard_reduction_minimum_exception_is_explicit_and_cancels_pending(exempt) -> None:
    planner = TargetQuantityPlanner(replace(_planner().config, deadband=Decimal("0.05"), pending_order_policy="wait"))
    kwargs = dict(
        decision_id="small-hard-reduction", instrument=InstrumentRules(
            "ETHUSDT-PERP.SIM", Decimal("0.001"), Decimal("0.001"), Decimal("20"), Decimal("1"), exempt),
        target_exposure=Decimal("0.5"), risk_budget_notional=Decimal("1000"),
        equity=Decimal("10000"), mark_price=Decimal("2000"), current_qty=Decimal("0.251"),
        force_reduce=True,
    )
    cancellation = planner.plan(**kwargs, pending=(PendingOrder("live", Decimal("0.1"), "ACCEPTED"),))
    assert cancellation.action == "CANCEL" and cancellation.requires_cancel_ack
    if exempt:
        reduction = planner.plan(**kwargs, pending=())
        assert reduction.delta_qty == Decimal("-0.001")
        assert reduction.reason == "hard_risk_reduce"
    else:
        with pytest.raises(PlanningBlocked, match="hard-risk reduction violates"):
            planner.plan(**kwargs, pending=())


@pytest.mark.parametrize("current,target", [("0", "0.5"), ("0.25", "0.6"), ("0.25", "-0.5")])
def test_forced_reduction_cannot_open_expand_or_reverse(current, target) -> None:
    with pytest.raises(PlanningBlocked, match="strictly reduce"):
        _planner().plan(
            decision_id="invalid-reduction", instrument=InstrumentRules(
                "ETHUSDT-PERP.SIM", Decimal("0.001"), Decimal("0.001"), Decimal("1"), Decimal("1")),
            target_exposure=Decimal(target), risk_budget_notional=Decimal("1000"),
            equity=Decimal("10000"), mark_price=Decimal("2000"), current_qty=Decimal(current),
            pending=(), force_reduce=True,
        )
