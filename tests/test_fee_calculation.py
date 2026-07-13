"""Offline tests for risk_manager's fee/slippage assumptions and the
analytical (non-iterative) risk-budget solve. No network, no AI.
"""

from decimal import Decimal

from risk_manager import (
    EntryPriceMode,
    MarginMode,
    PositionCalculator,
    RiskSettings,
    TakeProfitTarget,
    TradeScenario,
)


def _settings(**overrides) -> RiskSettings:
    base = dict(
        account_balance_usdt=Decimal("100"),
        risk_percent=Decimal("1"),
        leverage=10,
        margin_mode=MarginMode.ISOLATED,
        max_margin_percent=Decimal("100"),  # generous, so margin never binds in these tests
        maker_fee_percent=Decimal("0.02"),
        taker_fee_percent=Decimal("0.05"),
        slippage_percent=Decimal("0.02"),
        min_risk_reward=Decimal("0.1"),  # generous, TP status isn't under test here
        entry_price_mode=EntryPriceMode.MIDPOINT,
        quantity_step=Decimal("0.00000001"),  # ~no rounding distortion
        price_step=Decimal("0.00000001"),
        minimum_order_notional_usdt=Decimal("0"),
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
        entry_order_type="LIMIT",
        assume_maker_entry=False,
    )
    base.update(overrides)
    return TradeScenario(**base)


def test_analytical_solve_hits_risk_budget_exactly_before_rounding_distortion():
    settings = _settings()
    calc = PositionCalculator(settings)
    result = calc.calculate(_scenario())
    risk_budget = settings.account_balance_usdt * settings.risk_percent / Decimal("100")
    # Rounding to an 8-decimal step leaves only tiny residual error.
    assert abs(result.total_expected_loss_usdt - risk_budget) < Decimal("0.01")
    assert result.limited_by == "RISK"


def test_maker_entry_is_cheaper_than_taker_entry():
    settings = _settings()
    calc = PositionCalculator(settings)
    maker_result = calc.calculate(_scenario(assume_maker_entry=True, entry_order_type="LIMIT"))
    taker_result = calc.calculate(_scenario(assume_maker_entry=False, entry_order_type="LIMIT"))
    assert maker_result.entry_fee_usdt < taker_result.entry_fee_usdt


def test_market_entry_is_always_taker_even_if_maker_assumed():
    settings = _settings()
    calc = PositionCalculator(settings)
    market_result = calc.calculate(_scenario(assume_maker_entry=True, entry_order_type="MARKET"))
    taker_result = calc.calculate(_scenario(assume_maker_entry=False, entry_order_type="LIMIT"))
    # Same taker rate applies -> same entry fee (both scenarios have identical size/price).
    assert market_result.entry_fee_usdt == taker_result.entry_fee_usdt


def test_trigger_entry_is_always_taker():
    settings = _settings()
    calc = PositionCalculator(settings)
    trigger_result = calc.calculate(_scenario(assume_maker_entry=True, entry_order_type="TRIGGER"))
    taker_result = calc.calculate(_scenario(assume_maker_entry=False, entry_order_type="LIMIT"))
    assert trigger_result.entry_fee_usdt == taker_result.entry_fee_usdt


def test_stop_exit_fee_uses_taker_rate_regardless_of_entry_type():
    settings = _settings()
    calc = PositionCalculator(settings)
    maker_entry_result = calc.calculate(_scenario(assume_maker_entry=True, entry_order_type="LIMIT"))
    # stop_exit_fee_usdt = size * stop_loss * taker_rate -- verify against the taker rate directly.
    expected = maker_entry_result.position_size_coin_rounded * Decimal("98") * (Decimal("0.05") / Decimal("100"))
    assert abs(maker_entry_result.stop_exit_fee_usdt - expected) < Decimal("0.0000001")


def test_higher_slippage_increases_total_expected_loss():
    low = PositionCalculator(_settings(slippage_percent=Decimal("0"))).calculate(_scenario())
    high = PositionCalculator(_settings(slippage_percent=Decimal("0.5"))).calculate(_scenario())
    assert high.total_expected_loss_usdt > low.total_expected_loss_usdt
    assert high.slippage_estimate_usdt > low.slippage_estimate_usdt
    assert low.slippage_estimate_usdt == Decimal("0")


def test_zero_fees_and_slippage_loss_equals_pure_price_risk():
    settings = _settings(maker_fee_percent=Decimal("0"), taker_fee_percent=Decimal("0"), slippage_percent=Decimal("0"))
    result = PositionCalculator(settings).calculate(_scenario())
    risk_budget = settings.account_balance_usdt * settings.risk_percent / Decimal("100")
    assert abs(result.total_expected_loss_usdt - risk_budget) < Decimal("0.0001")
    assert result.entry_fee_usdt == Decimal("0")
    assert result.stop_exit_fee_usdt == Decimal("0")
    assert result.slippage_estimate_usdt == Decimal("0")
