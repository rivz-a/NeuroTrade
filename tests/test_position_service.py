"""Offline tests for position_service.calculate_active_position — the
orchestration that bridges consensus_engine's SelectedPlan into
risk_manager's PositionCalculator and decides whether "open LONG/SHORT" is
actually warranted right now.

No network, no AI, no mocking needed: `calculate_active_position` itself
never touches the network (only `resolve_instrument_rules`, tested
separately, does) — every test here builds an `InstrumentRules` by hand and
passes it in directly, which is itself evidence this function never calls
out to BingX or an AI model.
"""

from decimal import Decimal

from ai_client import AIAnalysisResult
from ai_schema import EntryZone, TakeProfit, TradePlan
from position_service import calculate_active_position
from risk_manager import (
    EntryPriceMode,
    InstrumentRules,
    MarginMode,
    PositionStatus,
    RiskSettings,
)
from trade_validator import ValidationResult

NOW = 1_000_000.0


def _plan(
    signal="LONG",
    entry_from=100.0,
    entry_to=101.0,
    stop_loss=95.0,
    tp1=115.0,
    entry_status="ENTER_NOW",
    entry_type="LIMIT",
) -> TradePlan:
    return TradePlan(
        signal=signal,
        entry_status=entry_status,
        confidence=70,
        market_regime="TREND_UP",
        entry=EntryZone(type=entry_type, from_=entry_from, to=entry_to, trigger="x"),
        stop_loss=stop_loss,
        take_profits=[TakeProfit(label="TP1", price=tp1, close_percent=100)],
        time_horizon_minutes=60,
        valid_for_minutes=120,
        reasons=["r"],
        risks=["k"],
        invalidation_conditions=["c"],
        wait_conditions=[],
        summary="s",
    )


def _wait_plan() -> TradePlan:
    return TradePlan(
        signal="WAIT",
        entry_status="NO_TRADE",
        confidence=40,
        market_regime="RANGE",
        entry=EntryZone(type="NONE", from_=0.0, to=0.0, trigger=""),
        stop_loss=0.0,
        take_profits=[],
        time_horizon_minutes=60,
        valid_for_minutes=120,
        reasons=[],
        risks=[],
        invalidation_conditions=[],
        wait_conditions=["wait for range breakout"],
        summary="s",
    )


def _votes(plan: TradePlan, count: int = 3) -> list[AIAnalysisResult]:
    labels = ["A", "B", "C"][:count]
    return [
        AIAnalysisResult(
            label=label, model=label.lower(), content="{}", error=None,
            latency_seconds=1.0, created_at=NOW, trade_plan=plan,
            validation=ValidationResult(status="valid", issues=[]),
        )
        for label in labels
    ]


def _settings(**overrides) -> RiskSettings:
    base = dict(
        account_balance_usdt=Decimal("1000"),
        risk_percent=Decimal("1"),
        leverage=5,
        margin_mode=MarginMode.ISOLATED,
        max_margin_percent=Decimal("30"),
        maker_fee_percent=Decimal("0.02"),
        taker_fee_percent=Decimal("0.05"),
        slippage_percent=Decimal("0.02"),
        min_risk_reward=Decimal("1.5"),
        entry_price_mode=EntryPriceMode.MIDPOINT,
        quantity_step=Decimal("0.001"),
        price_step=Decimal("0.01"),
        minimum_order_notional_usdt=Decimal("2"),
    )
    base.update(overrides)
    return RiskSettings(**base)


RULES = InstrumentRules(
    symbol="ETH-USDT",
    price_step=Decimal("0.01"),
    quantity_step=Decimal("0.001"),
    minimum_quantity=Decimal("0"),
    minimum_notional_usdt=Decimal("2"),
    maximum_leverage=0,
    source="FALLBACK",
)


def test_wait_consensus_is_not_applicable():
    votes = _votes(_wait_plan())
    result = calculate_active_position(
        votes, "scalping", total_models=3, current_price=100.5,
        settings=_settings(), instrument_rules=RULES, now=NOW,
    )
    assert result.applicable is False
    assert result.calculation is None
    assert result.bingx_fields is None


def test_wait_pullback_is_reference_only_and_cannot_open():
    votes = _votes(_plan(entry_status="WAIT_PULLBACK"))
    result = calculate_active_position(
        votes, "scalping", total_models=3, current_price=100.5,
        settings=_settings(), instrument_rules=RULES, now=NOW,
    )
    assert result.applicable is True
    assert result.reference_only is True
    assert result.can_open is False
    assert result.display_status == "WAITING_TRIGGER"


def test_price_already_past_stop_is_stop_already_breached():
    plan = _plan(signal="LONG", entry_from=100.0, entry_to=101.0, stop_loss=95.0, tp1=115.0)
    votes = _votes(plan)
    result = calculate_active_position(
        votes, "scalping", total_models=3, current_price=90.0,  # below stop_loss
        settings=_settings(), instrument_rules=RULES, now=NOW,
    )
    assert result.applicable is True
    assert result.reference_only is True
    assert result.can_open is False
    assert result.display_status == "STOP_ALREADY_BREACHED"


def test_price_already_past_tp1_is_tp_already_reached():
    plan = _plan(signal="LONG", entry_from=100.0, entry_to=101.0, stop_loss=95.0, tp1=115.0)
    votes = _votes(plan)
    result = calculate_active_position(
        votes, "scalping", total_models=3, current_price=120.0,  # above tp1
        settings=_settings(), instrument_rules=RULES, now=NOW,
    )
    assert result.applicable is True
    assert result.reference_only is True
    assert result.can_open is False
    assert result.display_status == "TP_ALREADY_REACHED"


def test_short_stop_already_breached_direction_is_inverted():
    plan = _plan(signal="SHORT", entry_from=100.0, entry_to=101.0, stop_loss=105.0, tp1=90.0)
    votes = _votes(plan)
    result = calculate_active_position(
        votes, "scalping", total_models=3, current_price=110.0,  # above stop_loss for a SHORT
        settings=_settings(), instrument_rules=RULES, now=NOW,
    )
    assert result.display_status == "STOP_ALREADY_BREACHED"


def test_allowed_and_valid_can_open():
    plan = _plan(signal="LONG", entry_from=100.0, entry_to=101.0, stop_loss=95.0, tp1=115.0)
    votes = _votes(plan)
    result = calculate_active_position(
        votes, "scalping", total_models=3, current_price=100.5,  # inside entry zone
        settings=_settings(), instrument_rules=RULES, now=NOW,
    )
    assert result.applicable is True
    assert result.reference_only is False
    assert result.consensus.trade_permission == "ALLOWED"
    assert result.calculation is not None
    assert result.calculation.status == PositionStatus.VALID
    assert result.can_open is True
    assert result.display_status == "VALID"
    assert result.bingx_fields is not None
    assert result.bingx_fields.side == "LONG"


def test_changing_only_settings_changes_the_calculation_deterministically():
    plan = _plan(signal="LONG", entry_from=100.0, entry_to=101.0, stop_loss=95.0, tp1=115.0)
    votes = _votes(plan)

    low_risk = calculate_active_position(
        votes, "scalping", total_models=3, current_price=100.5,
        settings=_settings(risk_percent=Decimal("1")), instrument_rules=RULES, now=NOW,
    )
    high_risk = calculate_active_position(
        votes, "scalping", total_models=3, current_price=100.5,
        settings=_settings(risk_percent=Decimal("5")), instrument_rules=RULES, now=NOW,
    )
    assert low_risk.calculation is not None and high_risk.calculation is not None
    assert low_risk.calculation.position_size_coin_rounded != high_risk.calculation.position_size_coin_rounded
    assert high_risk.calculation.position_size_coin_rounded > low_risk.calculation.position_size_coin_rounded


def test_margin_limited_shows_distinct_status_but_stays_openable():
    plan = _plan(signal="LONG", entry_from=100.0, entry_to=101.0, stop_loss=95.0, tp1=115.0)
    votes = _votes(plan)
    result = calculate_active_position(
        votes, "scalping", total_models=3, current_price=100.5,
        settings=_settings(max_margin_percent=Decimal("1"), leverage=2), instrument_rules=RULES, now=NOW,
    )
    assert result.calculation is not None
    assert result.calculation.limited_by == "MARGIN"
    assert result.calculation.status == PositionStatus.VALID
    assert result.display_status == "MARGIN_LIMIT"
    assert result.reference_only is False
    assert result.can_open is True


def test_risk_limited_normal_case_stays_plain_valid():
    # PositionCalculator always sizes by either risk-budget or margin —
    # RISK-limited is the default/expected outcome for an ordinary trade
    # (see test_allowed_and_valid_can_open's settings, generous margin/
    # leverage), so it must NOT get promoted to a distinct "RISK_LIMIT"
    # badge — that would fire on nearly every successful trade.
    plan = _plan(signal="LONG", entry_from=100.0, entry_to=101.0, stop_loss=95.0, tp1=115.0)
    votes = _votes(plan)
    result = calculate_active_position(
        votes, "scalping", total_models=3, current_price=100.5,
        settings=_settings(), instrument_rules=RULES, now=NOW,
    )
    assert result.calculation is not None
    assert result.calculation.limited_by == "RISK"
    assert result.display_status == "VALID"


def test_no_agreeing_plan_is_not_applicable():
    votes = [
        AIAnalysisResult(
            label="A", model="a", content="{}", error="rejected",
            latency_seconds=1.0, created_at=NOW, trade_plan=_plan(),
        )
    ]
    result = calculate_active_position(
        votes, "scalping", total_models=3, current_price=100.5,
        settings=_settings(), instrument_rules=RULES, now=NOW,
    )
    assert result.applicable is False
