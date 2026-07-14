"""Offline tests for risk_manager.normalize_entry_order_type and
build_bingx_manual_fields — pure functions, no network/AI.
"""

from decimal import Decimal

from risk_manager import (
    BingXInputMode,
    EntryPriceMode,
    MarginMode,
    PositionCalculator,
    PositionStatus,
    RiskSettings,
    TakeProfitTarget,
    TradeScenario,
    build_bingx_manual_fields,
    normalize_entry_order_type,
)


def test_normalize_exact_matches():
    assert normalize_entry_order_type("market") == "MARKET"
    assert normalize_entry_order_type("Market") == "MARKET"
    assert normalize_entry_order_type("limit") == "LIMIT"
    assert normalize_entry_order_type("trigger") == "TRIGGER"


def test_normalize_limit_short_via_substring_defaults_to_limit():
    # "Limit Short" isn't an exact key; it doesn't contain "market" or
    # "trigger"/"breakout" either, so it falls through to the LIMIT default.
    assert normalize_entry_order_type("Limit Short") == "LIMIT"


def test_normalize_breakout_and_trigger_substrings():
    assert normalize_entry_order_type("breakout") == "TRIGGER"
    assert normalize_entry_order_type("pullback_breakout_confirm") == "TRIGGER"
    assert normalize_entry_order_type("stop_trigger_entry") == "TRIGGER"


def test_normalize_unrecognized_or_blank_defaults_to_limit():
    assert normalize_entry_order_type("pullback_to_ema") == "LIMIT"
    assert normalize_entry_order_type("") == "LIMIT"
    assert normalize_entry_order_type(None) == "LIMIT"


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
        bingx_order_input_mode=BingXInputMode.MARGIN_USDT,
    )
    base.update(overrides)
    return RiskSettings(**base)


def _scenario(**overrides) -> TradeScenario:
    base = dict(
        signal="LONG",
        entry_from=Decimal("100"),
        entry_to=Decimal("100"),
        stop_loss=Decimal("98"),
        take_profits=[TakeProfitTarget("TP1", Decimal("105"), Decimal("100"))],
    )
    base.update(overrides)
    return TradeScenario(**base)


def test_build_fields_for_long_valid_position():
    settings = _settings()
    calc = PositionCalculator(settings).calculate(_scenario())
    assert calc.status == PositionStatus.VALID

    fields = build_bingx_manual_fields(calc, "LIMIT", settings)
    assert fields.side == "LONG"
    assert fields.margin_mode == "ISOLATED"
    assert fields.order_type == "LIMIT"
    assert fields.leverage == 5
    assert fields.price == calc.entry_price
    assert fields.stop_loss == calc.stop_loss
    assert fields.take_profit == calc.take_profit_results[0].price
    assert fields.selected_input_mode == "MARGIN_USDT"
    assert fields.selected_input_value == calc.bingx_fields["MARGIN_USDT"]


def test_build_fields_for_short_position():
    settings = _settings()
    scenario = _scenario(
        signal="SHORT",
        entry_from=Decimal("100"),
        entry_to=Decimal("100"),
        stop_loss=Decimal("102"),
        take_profits=[TakeProfitTarget("TP1", Decimal("94"), Decimal("100"))],
    )
    calc = PositionCalculator(settings).calculate(scenario)
    assert calc.status == PositionStatus.VALID

    fields = build_bingx_manual_fields(calc, "MARKET", settings)
    assert fields.side == "SHORT"
    assert fields.order_type == "MARKET"


def test_selected_input_mode_notional():
    settings = _settings(bingx_order_input_mode=BingXInputMode.NOTIONAL_USDT)
    calc = PositionCalculator(settings).calculate(_scenario())
    fields = build_bingx_manual_fields(calc, "LIMIT", settings)
    assert fields.selected_input_mode == "NOTIONAL_USDT"
    assert fields.selected_input_value == calc.bingx_fields["NOTIONAL_USDT"]
    assert fields.notional_usdt == calc.bingx_fields["NOTIONAL_USDT"]


def test_selected_input_mode_coin_quantity():
    settings = _settings(bingx_order_input_mode=BingXInputMode.COIN_QUANTITY)
    calc = PositionCalculator(settings).calculate(_scenario())
    fields = build_bingx_manual_fields(calc, "LIMIT", settings)
    assert fields.selected_input_mode == "COIN_QUANTITY"
    assert fields.selected_input_value == calc.bingx_fields["COIN_QUANTITY"]
    assert fields.coin_quantity == calc.position_size_coin_rounded
