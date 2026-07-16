"""Offline tests for trade_validator.validate_trade_plan — pure arithmetic,
no network, no AI calls. Covers the LONG/SHORT/WAIT rules from section 7 of
the product spec: stop/TP ordering, distance-to-price and distance-to-stop
sanity checks (in ATR multiples), R:R vs the mode's minimum floor, and WAIT's
"must carry conditions" requirement.
"""

from ai_schema import EntryZone, TakeProfit, TradePlan
from trade_validator import ValidationContext, validate_trade_plan


def _base_plan(**overrides) -> TradePlan:
    data = dict(
        signal="LONG",
        entry_status="ENTER_NOW",
        confidence=65,
        market_regime="TREND_UP",
        entry=EntryZone(type="LIMIT_ZONE", from_=1776.80, to=1777.20, trigger="x"),
        stop_loss=1774.80,
        take_profits=[
            TakeProfit(label="TP1", price=1779.20, close_percent=40),
            TakeProfit(label="TP2", price=1781.00, close_percent=35),
            TakeProfit(label="TP3", price=1785.00, close_percent=25),
        ],
        time_horizon_minutes=30,
        valid_for_minutes=15,
        reasons=["r"],
        risks=["r"],
        invalidation_conditions=["c"],
        wait_conditions=[],
        summary="s",
    )
    data.update(overrides)
    return TradePlan(**data)


def _ctx(**overrides) -> ValidationContext:
    data = dict(current_price=1777.0, atr=2.0, spread=0.1)
    data.update(overrides)
    return ValidationContext(**data)


def test_valid_long_plan_passes():
    result = validate_trade_plan(_base_plan(), "scalping", _ctx())
    assert result.status == "valid"


def test_long_tp_below_entry_rejected():
    plan = _base_plan(take_profits=[TakeProfit(label="TP1", price=1770.0, close_percent=100)])
    result = validate_trade_plan(plan, "scalping", _ctx())
    assert result.status == "rejected"
    assert any(i.code == "TP1_WRONG_SIDE" for i in result.issues)


def test_long_stop_above_entry_rejected():
    plan = _base_plan(stop_loss=1778.0)
    result = validate_trade_plan(plan, "scalping", _ctx())
    assert result.status == "rejected"
    assert any(i.code == "SL_WRONG_SIDE" for i in result.issues)


def test_short_stop_below_entry_rejected():
    plan = _base_plan(
        signal="SHORT",
        entry=EntryZone(type="LIMIT_ZONE", from_=1777.20, to=1776.80, trigger="x"),
        stop_loss=1770.0,
        take_profits=[
            TakeProfit(label="TP1", price=1774.0, close_percent=40),
            TakeProfit(label="TP2", price=1772.0, close_percent=35),
            TakeProfit(label="TP3", price=1768.0, close_percent=25),
        ],
    )
    result = validate_trade_plan(plan, "scalping", _ctx())
    assert result.status == "rejected"
    assert any(i.code == "SL_WRONG_SIDE" for i in result.issues)


def test_tp_out_of_order_rejected():
    plan = _base_plan(
        take_profits=[
            TakeProfit(label="TP1", price=1785.0, close_percent=40),
            TakeProfit(label="TP2", price=1779.0, close_percent=35),
            TakeProfit(label="TP3", price=1790.0, close_percent=25),
        ]
    )
    result = validate_trade_plan(plan, "scalping", _ctx())
    assert result.status == "rejected"
    assert any(i.code == "TP_ORDER" for i in result.issues)


def test_stop_too_tight_warns():
    plan = _base_plan(stop_loss=1776.79)  # ~0.1 ATR from entry mid with atr=2.0
    result = validate_trade_plan(plan, "scalping", _ctx(atr=2.0))
    assert any(i.code == "STOP_TOO_TIGHT" for i in result.issues)


def test_entry_too_far_warns():
    plan = _base_plan(entry=EntryZone(type="LIMIT_ZONE", from_=1800.0, to=1801.0, trigger="x"))
    result = validate_trade_plan(plan, "scalping", _ctx(atr=2.0, current_price=1777.0))
    assert any(i.code == "ENTRY_TOO_FAR" for i in result.issues)


def test_rr_below_mode_floor_rejected():
    plan = _base_plan(
        take_profits=[TakeProfit(label="TP1", price=1778.0, close_percent=100)],
    )
    result = validate_trade_plan(plan, "swing", _ctx())
    assert result.status == "rejected"
    assert any(i.code == "RR_TOO_LOW" for i in result.issues)


def test_wait_without_conditions_rejected():
    plan = _base_plan(signal="WAIT", wait_conditions=[], invalidation_conditions=[])
    result = validate_trade_plan(plan, "scalping", _ctx())
    assert result.status == "rejected"
    assert any(i.code == "WAIT_NO_CONDITIONS" for i in result.issues)


def test_wait_with_conditions_valid():
    plan = _base_plan(signal="WAIT", wait_conditions=["Закрепление выше 1780"])
    result = validate_trade_plan(plan, "scalping", _ctx())
    assert result.status == "valid"


def test_contradictions_and_missing_context_do_not_affect_validity():
    plan = _base_plan(
        contradictions=["Модель считает режим RANGE, хотя передан TREND_UP"],
        missing_context=["Не хватает данных по ликвидациям"],
    )
    result = validate_trade_plan(plan, "scalping", _ctx())
    assert result.status == "valid"
