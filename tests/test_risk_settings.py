"""Offline tests for risk_manager.RiskSettings / DEFAULT_RISK_SETTINGS /
validate_scenario_and_settings' settings-level checks. No network, no AI.
"""

from decimal import Decimal

from risk_manager import (
    DEFAULT_RISK_SETTINGS,
    BingXInputMode,
    EntryPriceMode,
    MarginMode,
    PositionStatus,
    RiskSettings,
    TakeProfitTarget,
    TradeScenario,
    validate_scenario_and_settings,
)


def _valid_scenario() -> TradeScenario:
    return TradeScenario(
        signal="LONG",
        entry_from=Decimal("100.0"),
        entry_to=Decimal("101.0"),
        stop_loss=Decimal("98.0"),
        take_profits=[TakeProfitTarget("TP1", Decimal("105.0"), Decimal("100"))],
    )


def test_default_settings_match_spec_example():
    s = DEFAULT_RISK_SETTINGS
    assert s.account_balance_usdt == Decimal("100.0")
    assert s.risk_percent == Decimal("1.0")
    assert s.leverage == 5
    assert s.margin_mode == MarginMode.ISOLATED
    assert s.max_margin_percent == Decimal("30.0")
    assert s.maker_fee_percent == Decimal("0.02")
    assert s.taker_fee_percent == Decimal("0.05")
    assert s.slippage_percent == Decimal("0.02")
    assert s.min_risk_reward == Decimal("1.5")
    assert s.entry_price_mode == EntryPriceMode.MIDPOINT
    assert s.quantity_step == Decimal("0.001")
    assert s.price_step == Decimal("0.01")
    assert s.minimum_order_notional_usdt == Decimal("2.0")
    assert s.bingx_order_input_mode == BingXInputMode.MARGIN_USDT


def test_from_raw_uses_string_conversion_not_binary_float():
    # 0.1 + 0.2 style traps: str(0.1) round-trips cleanly through Decimal,
    # Decimal(0.1) directly would carry binary floating-point noise.
    s = RiskSettings.from_raw(
        account_balance_usdt=100.1,
        risk_percent=1.0,
        leverage=5,
        margin_mode="ISOLATED",
        max_margin_percent=30.0,
        maker_fee_percent=0.02,
        taker_fee_percent=0.05,
        slippage_percent=0.02,
        min_risk_reward=1.5,
        entry_price_mode="MIDPOINT",
        quantity_step=0.001,
        price_step=0.01,
        minimum_order_notional_usdt=2.0,
    )
    assert s.account_balance_usdt == Decimal("100.1")


def _settings(**overrides) -> RiskSettings:
    base = dict(
        account_balance_usdt=Decimal("100"),
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


def test_zero_leverage_is_invalid_settings():
    result = validate_scenario_and_settings(_valid_scenario(), _settings(leverage=0))
    assert result.ok is False
    assert result.status == PositionStatus.INVALID_SETTINGS


def test_zero_balance_is_invalid_settings():
    result = validate_scenario_and_settings(_valid_scenario(), _settings(account_balance_usdt=Decimal("0")))
    assert result.ok is False
    assert result.status == PositionStatus.INVALID_SETTINGS


def test_risk_percent_over_100_is_invalid_settings():
    result = validate_scenario_and_settings(_valid_scenario(), _settings(risk_percent=Decimal("150")))
    assert result.ok is False
    assert result.status == PositionStatus.INVALID_SETTINGS


def test_risk_percent_zero_is_valid_settings():
    # Degenerate but not nonsensical — should flow through to a zero-size
    # position (BELOW_MINIMUM_ORDER), not be rejected as invalid settings.
    result = validate_scenario_and_settings(_valid_scenario(), _settings(risk_percent=Decimal("0")))
    assert result.ok is True


def test_negative_fees_are_invalid_settings():
    result = validate_scenario_and_settings(_valid_scenario(), _settings(taker_fee_percent=Decimal("-0.05")))
    assert result.ok is False
    assert result.status == PositionStatus.INVALID_SETTINGS


def test_negative_slippage_is_invalid_settings():
    result = validate_scenario_and_settings(_valid_scenario(), _settings(slippage_percent=Decimal("-0.01")))
    assert result.ok is False
    assert result.status == PositionStatus.INVALID_SETTINGS


def test_nonpositive_quantity_step_is_invalid_settings():
    result = validate_scenario_and_settings(_valid_scenario(), _settings(quantity_step=Decimal("0")))
    assert result.ok is False
    assert result.status == PositionStatus.INVALID_SETTINGS


def test_nonpositive_price_step_is_invalid_settings():
    result = validate_scenario_and_settings(_valid_scenario(), _settings(price_step=Decimal("0")))
    assert result.ok is False
    assert result.status == PositionStatus.INVALID_SETTINGS


def test_max_margin_percent_over_100_is_invalid_settings():
    result = validate_scenario_and_settings(_valid_scenario(), _settings(max_margin_percent=Decimal("150")))
    assert result.ok is False
    assert result.status == PositionStatus.INVALID_SETTINGS


def test_min_risk_reward_zero_is_invalid_settings():
    result = validate_scenario_and_settings(_valid_scenario(), _settings(min_risk_reward=Decimal("0")))
    assert result.ok is False
    assert result.status == PositionStatus.INVALID_SETTINGS
