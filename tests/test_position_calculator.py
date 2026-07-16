"""End-to-end offline tests for risk_manager.PositionCalculator — the 25
scenarios from the product spec (section 21), adapted to this module's
architecture. No network, no AI calls; every input/output is exact Decimal
arithmetic checked against hand- or formula-derived expectations.

Scenarios 18 ("цена вышла из зоны входа") and 19 ("сигнал устарел") are not
covered here by design — they need a live current price / signal timestamp,
which `PositionCalculator` deliberately does not take as input (that's the
dashboard integration layer's job in a later iteration, per the plan).
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
)


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


# 1. LONG с обычным stop loss.
def test_long_basic():
    result = PositionCalculator(_settings()).calculate(_scenario())
    assert result.status == PositionStatus.VALID
    assert result.entry_price == Decimal("100")
    assert result.stop_distance == Decimal("2")
    assert result.position_size_coin_rounded > 0


# 2. SHORT с обычным stop loss.
def test_short_basic():
    scenario = _scenario(
        signal="SHORT",
        entry_from=Decimal("100"),
        entry_to=Decimal("100"),
        stop_loss=Decimal("102"),
        take_profits=[TakeProfitTarget("TP1", Decimal("95"), Decimal("100"))],
    )
    result = PositionCalculator(_settings()).calculate(scenario)
    assert result.status == PositionStatus.VALID
    assert result.stop_distance == Decimal("2")
    assert result.take_profit_results[0].gross_profit_usdt > 0


# 3. Вход диапазоном — все 4 режима.
def test_entry_price_mode_midpoint():
    scenario = _scenario(entry_from=Decimal("100"), entry_to=Decimal("102"))
    result = PositionCalculator(_settings(entry_price_mode=EntryPriceMode.MIDPOINT)).calculate(scenario)
    assert result.entry_price == Decimal("101")


def test_entry_price_mode_conservative_long_uses_upper_bound():
    scenario = _scenario(signal="LONG", entry_from=Decimal("100"), entry_to=Decimal("102"), stop_loss=Decimal("95"))
    result = PositionCalculator(_settings(entry_price_mode=EntryPriceMode.CONSERVATIVE)).calculate(scenario)
    assert result.entry_price == Decimal("102")


def test_entry_price_mode_best_case_long_uses_lower_bound():
    scenario = _scenario(signal="LONG", entry_from=Decimal("100"), entry_to=Decimal("102"), stop_loss=Decimal("95"))
    result = PositionCalculator(_settings(entry_price_mode=EntryPriceMode.BEST_CASE)).calculate(scenario)
    assert result.entry_price == Decimal("100")


def test_entry_price_mode_manual_uses_given_price():
    scenario = _scenario(entry_from=Decimal("100"), entry_to=Decimal("102"), manual_entry_price=Decimal("101.4"))
    result = PositionCalculator(_settings(entry_price_mode=EntryPriceMode.MANUAL)).calculate(scenario)
    assert result.entry_price == Decimal("101.4")


# 4. Позиция ограничена риском (margin cap generous enough not to bind).
def test_position_limited_by_risk():
    settings = _settings(max_margin_percent=Decimal("100"), leverage=50)
    result = PositionCalculator(settings).calculate(_scenario())
    assert result.limited_by == "RISK"


# 5. Позиция ограничена маржой.
def test_position_limited_by_margin():
    settings = _settings(max_margin_percent=Decimal("1"), leverage=2, risk_percent=Decimal("50"))
    result = PositionCalculator(settings).calculate(_scenario())
    assert result.limited_by == "MARGIN"
    assert any("маржой" in w.lower() or "маржа" in w.lower() for w in result.warnings)


# 6. Комиссии уменьшают размер позиции.
def test_fees_reduce_position_size():
    no_fee_settings = _settings(
        maker_fee_percent=Decimal("0"), taker_fee_percent=Decimal("0"), slippage_percent=Decimal("0"),
        max_margin_percent=Decimal("100"), leverage=50,
    )
    with_fee_settings = _settings(max_margin_percent=Decimal("100"), leverage=50)
    no_fee = PositionCalculator(no_fee_settings).calculate(_scenario())
    with_fee = PositionCalculator(with_fee_settings).calculate(_scenario())
    assert with_fee.position_size_coin < no_fee.position_size_coin


# 7. TP1 после комиссий (или просто по цене) невыгоден, TP2 годный -> план не отклоняется целиком.
def test_weak_tp1_strong_tp2_does_not_reject_whole_plan():
    scenario = _scenario(
        take_profits=[
            TakeProfitTarget("TP1", Decimal("100.5"), Decimal("50")),
            TakeProfitTarget("TP2", Decimal("110"), Decimal("50")),
        ]
    )
    settings = _settings(
        account_balance_usdt=Decimal("1000"), risk_percent=Decimal("1"), min_risk_reward=Decimal("1.5"),
        max_margin_percent=Decimal("100"), leverage=20,
    )
    result = PositionCalculator(settings).calculate(scenario)
    assert result.status == PositionStatus.VALID
    assert result.primary_target_label == "TP2"
    assert any("TP1" in w for w in result.warnings)


# Neither TP clears even the GROSS R:R bar -> LOW_RISK_REWARD (never was viable).
def test_no_tp_meets_gross_rr_gives_low_risk_reward():
    scenario = _scenario(take_profits=[TakeProfitTarget("TP1", Decimal("100.2"), Decimal("100"))])
    settings = _settings(min_risk_reward=Decimal("1.5"), max_margin_percent=Decimal("100"), leverage=20)
    result = PositionCalculator(settings).calculate(scenario)
    assert result.status == PositionStatus.LOW_RISK_REWARD


# Gross R:R clears the bar but fees/slippage push net below it -> FEES_TOO_HIGH.
def test_gross_ok_net_fails_gives_fees_too_high():
    scenario = _scenario(
        signal="LONG",
        entry_from=Decimal("100"), entry_to=Decimal("100"),
        stop_loss=Decimal("90"),
        take_profits=[TakeProfitTarget("TP1", Decimal("105"), Decimal("100"))],
    )
    settings = _settings(
        account_balance_usdt=Decimal("1000"), risk_percent=Decimal("10"),
        maker_fee_percent=Decimal("1"), taker_fee_percent=Decimal("1"), slippage_percent=Decimal("1"),
        min_risk_reward=Decimal("0.4"), max_margin_percent=Decimal("100"), leverage=20,
        quantity_step=Decimal("0.0001"),
    )
    result = PositionCalculator(settings).calculate(scenario)
    tp1 = result.take_profit_results[0]
    assert tp1.gross_rr >= settings.min_risk_reward  # would have looked fine before fees
    assert tp1.net_rr is not None and tp1.net_rr < settings.min_risk_reward
    assert result.status == PositionStatus.FEES_TOO_HIGH


# 8. Размер ниже минимального ордера.
def test_below_minimum_order():
    settings = _settings(account_balance_usdt=Decimal("10"), risk_percent=Decimal("0.01"), minimum_order_notional_usdt=Decimal("5"))
    result = PositionCalculator(settings).calculate(_scenario())
    assert result.status == PositionStatus.POSITION_TOO_SMALL


# 9. Неверный stop loss для LONG (стоп выше входа).
def test_invalid_stop_for_long():
    scenario = _scenario(signal="LONG", stop_loss=Decimal("101"))  # above entry — wrong side
    result = PositionCalculator(_settings()).calculate(scenario)
    assert result.status == PositionStatus.INVALID_SCENARIO


# 10. Неверный stop loss для SHORT (стоп ниже входа).
def test_invalid_stop_for_short():
    scenario = _scenario(
        signal="SHORT", entry_from=Decimal("100"), entry_to=Decimal("100"),
        stop_loss=Decimal("99"),  # below entry — wrong side for SHORT
        take_profits=[TakeProfitTarget("TP1", Decimal("95"), Decimal("100"))],
    )
    result = PositionCalculator(_settings()).calculate(scenario)
    assert result.status == PositionStatus.INVALID_SCENARIO


# 11. Плечо равно нулю.
def test_zero_leverage_end_to_end():
    result = PositionCalculator(_settings(leverage=0)).calculate(_scenario())
    assert result.status == PositionStatus.INVALID_SETTINGS


# 12. Баланс равен нулю.
def test_zero_balance_end_to_end():
    result = PositionCalculator(_settings(account_balance_usdt=Decimal("0"))).calculate(_scenario())
    assert result.status == PositionStatus.INVALID_SETTINGS


# 13. Риск больше 100%.
def test_risk_over_100_end_to_end():
    result = PositionCalculator(_settings(risk_percent=Decimal("150"))).calculate(_scenario())
    assert result.status == PositionStatus.INVALID_SETTINGS


# 14. Риск равен нулю.
def test_risk_zero_end_to_end():
    result = PositionCalculator(_settings(risk_percent=Decimal("0"))).calculate(_scenario())
    assert result.status == PositionStatus.POSITION_TOO_SMALL
    assert result.position_size_coin_rounded == Decimal("0")


# 15. Очень близкий stop loss -> large naive size, margin cap should bind.
def test_very_tight_stop_gets_margin_limited():
    scenario = _scenario(stop_loss=Decimal("99.99"))  # stop_distance = 0.01
    settings = _settings(max_margin_percent=Decimal("30"), leverage=5)
    result = PositionCalculator(settings).calculate(scenario)
    assert result.limited_by == "MARGIN"


# 16. Очень далёкий stop loss -> tiny size, likely below minimum order.
def test_very_wide_stop_falls_below_minimum():
    scenario = _scenario(entry_from=Decimal("100"), entry_to=Decimal("100"), stop_loss=Decimal("50"))
    settings = _settings(account_balance_usdt=Decimal("50"), risk_percent=Decimal("1"))
    result = PositionCalculator(settings).calculate(scenario)
    assert result.status == PositionStatus.POSITION_TOO_SMALL


# 17. Округление количества вниз.
def test_quantity_rounds_down_to_step():
    settings = _settings(quantity_step=Decimal("0.01"), max_margin_percent=Decimal("100"), leverage=50)
    result = PositionCalculator(settings).calculate(_scenario())
    assert result.position_size_coin_rounded <= result.position_size_coin
    remainder = result.position_size_coin_rounded / Decimal("0.01")
    assert remainder == remainder.to_integral_value()


# 20. WAIT не рассчитывает позицию.
def test_wait_signal_not_calculated():
    scenario = TradeScenario(
        signal="WAIT",  # bypasses the Literal type hint at runtime, exactly what we're guarding against
        entry_from=Decimal("100"), entry_to=Decimal("100"), stop_loss=Decimal("98"),
    )
    result = PositionCalculator(_settings()).calculate(scenario)
    assert result.status == PositionStatus.WAIT
    assert result.position_size_coin_rounded == Decimal("0")


# 21. Несколько TP с частичным закрытием.
def test_multiple_tp_partial_close_portions():
    scenario = _scenario(
        take_profits=[
            TakeProfitTarget("TP1", Decimal("102"), Decimal("50")),
            TakeProfitTarget("TP2", Decimal("104"), Decimal("30")),
            TakeProfitTarget("TP3", Decimal("106"), Decimal("20")),
        ]
    )
    settings = _settings(max_margin_percent=Decimal("100"), leverage=20, min_risk_reward=Decimal("0.5"))
    result = PositionCalculator(settings).calculate(scenario)
    assert len(result.take_profit_results) == 3
    full_size = result.position_size_coin_rounded
    tp1, tp2, tp3 = result.take_profit_results
    # gross_profit_usdt is proportional to close_percent of the full size.
    assert tp1.gross_profit_usdt == full_size * Decimal("0.50") * (Decimal("102") - result.entry_price)
    assert tp2.gross_profit_usdt == full_size * Decimal("0.30") * (Decimal("104") - result.entry_price)
    assert tp3.gross_profit_usdt == full_size * Decimal("0.20") * (Decimal("106") - result.entry_price)


# 22. Разные режимы BingX input mode — все три значения всегда присутствуют.
def test_bingx_fields_always_include_all_three_modes():
    for mode in (BingXInputMode.MARGIN_USDT, BingXInputMode.NOTIONAL_USDT, BingXInputMode.COIN_QUANTITY):
        settings = _settings(bingx_order_input_mode=mode)
        result = PositionCalculator(settings).calculate(_scenario())
        assert set(result.bingx_fields.keys()) == {"MARGIN_USDT", "NOTIONAL_USDT", "COIN_QUANTITY"}
        assert result.bingx_fields["MARGIN_USDT"] == result.required_margin_usdt
        assert result.bingx_fields["NOTIONAL_USDT"] == result.position_notional_usdt
        assert result.bingx_fields["COIN_QUANTITY"] == result.position_size_coin_rounded


# 23. CROSS и ISOLATED.
def test_cross_margin_mode_warns_isolated_does_not():
    cross_result = PositionCalculator(_settings(margin_mode=MarginMode.CROSS)).calculate(_scenario())
    isolated_result = PositionCalculator(_settings(margin_mode=MarginMode.ISOLATED)).calculate(_scenario())
    assert any("CROSS" in w for w in cross_result.warnings)
    assert not any("CROSS" in w for w in isolated_result.warnings)


# 24. Изменение настроек без нового AI-запроса — тот же сценарий, другие настройки.
def test_recalculate_with_new_settings_same_scenario_no_ai_involved():
    scenario = _scenario()  # built once, reused — nothing here talks to AI/network
    low_risk = PositionCalculator(_settings(risk_percent=Decimal("0.5"))).calculate(scenario)
    high_risk = PositionCalculator(_settings(risk_percent=Decimal("2"))).calculate(scenario)
    assert high_risk.position_size_coin > low_risk.position_size_coin
    assert high_risk.risk_budget_usdt == low_risk.risk_budget_usdt * 4


# 25. Отрицательная чистая прибыль после комиссий.
def test_negative_net_profit_after_fees():
    scenario = _scenario(
        entry_from=Decimal("100"), entry_to=Decimal("100"), stop_loss=Decimal("99"),
        take_profits=[TakeProfitTarget("TP1", Decimal("100.05"), Decimal("100"))],
    )
    settings = _settings(
        maker_fee_percent=Decimal("1"), taker_fee_percent=Decimal("1"), slippage_percent=Decimal("1"),
        min_risk_reward=Decimal("0.01"), max_margin_percent=Decimal("100"), leverage=20,
    )
    result = PositionCalculator(settings).calculate(scenario)
    assert result.take_profit_results[0].net_profit_usdt < 0


# --- Этап 1.1: blended-результат при частичном закрытии (TP1/TP2/TP3) ---


# 26. Три цели 50/30/20% — blended это точная сумма трёх TakeProfitResult.
def test_blended_result_three_targets_sums_to_hundred():
    scenario = _scenario(
        take_profits=[
            TakeProfitTarget("TP1", Decimal("102"), Decimal("50")),
            TakeProfitTarget("TP2", Decimal("104"), Decimal("30")),
            TakeProfitTarget("TP3", Decimal("106"), Decimal("20")),
        ]
    )
    settings = _settings(max_margin_percent=Decimal("100"), leverage=20, min_risk_reward=Decimal("0.5"))
    result = PositionCalculator(settings).calculate(scenario)
    tp1, tp2, tp3 = result.take_profit_results
    blended = result.blended
    assert blended is not None
    assert blended.total_close_percent == Decimal("100")
    assert blended.gross_profit_usdt == tp1.gross_profit_usdt + tp2.gross_profit_usdt + tp3.gross_profit_usdt
    assert blended.net_profit_usdt == tp1.net_profit_usdt + tp2.net_profit_usdt + tp3.net_profit_usdt
    assert blended.fees_usdt == tp1.fees_usdt + tp2.fees_usdt + tp3.fees_usdt
    price_loss = result.position_size_coin_rounded * result.stop_distance
    assert blended.gross_rr == blended.gross_profit_usdt / price_loss
    assert blended.net_rr == blended.net_profit_usdt / result.total_expected_loss_usdt
    assert not any("close_percent" in w for w in result.warnings)


# 27. Две цели 60/40% — сумма ровно 100%, предупреждения о расхождении нет.
def test_blended_result_two_targets_no_coverage_warning():
    scenario = _scenario(
        take_profits=[
            TakeProfitTarget("TP1", Decimal("103"), Decimal("60")),
            TakeProfitTarget("TP2", Decimal("106"), Decimal("40")),
        ]
    )
    result = PositionCalculator(_settings()).calculate(scenario)
    assert result.blended is not None
    assert result.blended.total_close_percent == Decimal("100")
    assert not any("%" in w and "остаток" in w for w in result.warnings)
    assert not any("некорректно" in w for w in result.warnings)


# 28. Одна цель 100% — blended численно совпадает с единственным TakeProfitResult.
def test_blended_result_single_target_mirrors_the_one_result():
    result = PositionCalculator(_settings()).calculate(_scenario())
    tp1 = result.take_profit_results[0]
    blended = result.blended
    assert blended is not None
    assert blended.total_close_percent == Decimal("100")
    assert blended.gross_profit_usdt == tp1.gross_profit_usdt
    assert blended.net_profit_usdt == tp1.net_profit_usdt
    assert blended.net_rr == tp1.net_rr


# 29. Две цели, сумма 99.4% — validate_scenario_and_settings принимает любую
# сумму в диапазоне 99-101% (risk_manager.py:294-297), но наш допуск на
# точное совпадение с 100% для blended-предупреждения строже (0.5%), так что
# этот легитимный, прошедший валидацию план всё равно получает свою пометку.
def test_blended_result_undercovered_but_valid_plan_warns():
    scenario = _scenario(
        take_profits=[
            TakeProfitTarget("TP1", Decimal("102"), Decimal("50")),
            TakeProfitTarget("TP2", Decimal("104"), Decimal("49.4")),
        ]
    )
    result = PositionCalculator(_settings()).calculate(scenario)
    assert result.status not in (PositionStatus.INVALID_SCENARIO, PositionStatus.INVALID_SETTINGS)
    assert result.blended is not None
    assert result.blended.total_close_percent == Decimal("99.4")
    assert any("99.4" in w and "остаток" in w for w in result.warnings)


# 30. Две цели, сумма 100.6% — тоже внутри допустимого для валидации диапазона
# (99-101%), но выходит за 0.5%-порог blended-предупреждения о превышении.
def test_blended_result_overcovered_but_valid_plan_warns():
    scenario = _scenario(
        take_profits=[
            TakeProfitTarget("TP1", Decimal("102"), Decimal("50")),
            TakeProfitTarget("TP2", Decimal("104"), Decimal("50.6")),
        ]
    )
    result = PositionCalculator(_settings()).calculate(scenario)
    assert result.status not in (PositionStatus.INVALID_SCENARIO, PositionStatus.INVALID_SETTINGS)
    assert result.blended is not None
    assert result.blended.total_close_percent == Decimal("100.6")
    assert any("100.6" in w and "больше 100%" in w for w in result.warnings)


# 30b. Сумма явно за пределами 99-101% (например 80%) отклоняется ещё на
# уровне validate_scenario_and_settings — до нашего кода blended вообще не
# доходит, calculate() возвращает пустой результат с issues, а не warnings.
def test_grossly_invalid_close_percent_sum_rejected_before_blended():
    scenario = _scenario(
        take_profits=[
            TakeProfitTarget("TP1", Decimal("102"), Decimal("50")),
            TakeProfitTarget("TP2", Decimal("104"), Decimal("30")),
        ]
    )
    result = PositionCalculator(_settings()).calculate(scenario)
    assert result.status == PositionStatus.INVALID_SCENARIO
    assert result.blended is None
    assert any("close_percent" in i and "100%" in i for i in result.issues)


# 31. Невалидный сценарий (нет take_profits) — blended отсутствует.
def test_blended_result_is_none_without_take_profits():
    scenario = _scenario(take_profits=[])
    result = PositionCalculator(_settings()).calculate(scenario)
    assert result.blended is None
