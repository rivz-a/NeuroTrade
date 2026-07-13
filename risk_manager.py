"""Deterministic, code-only position sizing and risk management.

No AI model computes anything here — position size, margin, fees, and
Risk/Reward are pure arithmetic on numbers the user configured
(`RiskSettings`) plus the trade idea a model proposed (`TradeScenario`:
signal, entry zone, stop loss, take profits — nothing else). Same inputs
always produce the same result, regardless of which AI model (or none)
supplied the trade idea.

This module is intentionally independent of `ai_schema.TradePlan` — the
dashboard is responsible for building a `TradeScenario` out of a validated
`TradePlan` later. That keeps this module testable and usable without
touching the AI prompt/schema at all.

All money math uses `Decimal`, never `float` — see `RiskSettings.from_raw`
for the one sanctioned entry point that converts plain numbers (as you'd
get from JSON) into `Decimal` via `Decimal(str(x))`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import ROUND_DOWN, ROUND_HALF_UP, Decimal
from enum import Enum
from typing import Literal


class MarginMode(str, Enum):
    ISOLATED = "ISOLATED"
    CROSS = "CROSS"


class EntryPriceMode(str, Enum):
    MIDPOINT = "MIDPOINT"
    CONSERVATIVE = "CONSERVATIVE"
    BEST_CASE = "BEST_CASE"
    MANUAL = "MANUAL"


class BingXInputMode(str, Enum):
    MARGIN_USDT = "MARGIN_USDT"
    NOTIONAL_USDT = "NOTIONAL_USDT"
    COIN_QUANTITY = "COIN_QUANTITY"


class PositionStatus(str, Enum):
    VALID = "VALID"
    WAIT = "WAIT"
    INVALID_SCENARIO = "INVALID_SCENARIO"
    STALE_SIGNAL = "STALE_SIGNAL"
    PRICE_OUTSIDE_ENTRY_ZONE = "PRICE_OUTSIDE_ENTRY_ZONE"
    RISK_LIMIT = "RISK_LIMIT"
    MARGIN_LIMIT = "MARGIN_LIMIT"
    BELOW_MINIMUM_ORDER = "BELOW_MINIMUM_ORDER"
    LOW_RISK_REWARD = "LOW_RISK_REWARD"
    FEES_TOO_HIGH = "FEES_TOO_HIGH"
    MISSING_MARKET_RULES = "MISSING_MARKET_RULES"
    INVALID_SETTINGS = "INVALID_SETTINGS"


# Russian, user-facing messages for every status — dashboard/API just look
# these up rather than inventing wording per call site.
STATUS_MESSAGES: dict[PositionStatus, str] = {
    PositionStatus.VALID: "Расчёт позиции корректен.",
    PositionStatus.WAIT: "Сигнал WAIT — размер позиции не рассчитывается.",
    PositionStatus.INVALID_SCENARIO: "Торговый сценарий некорректен — расчёт позиции невозможен.",
    PositionStatus.STALE_SIGNAL: "Сигнал устарел.",
    PositionStatus.PRICE_OUTSIDE_ENTRY_ZONE: "Текущая цена вышла из расчётной зоны входа.",
    PositionStatus.RISK_LIMIT: "Размер позиции ограничен допустимым риском на сделку.",
    PositionStatus.MARGIN_LIMIT: "Размер позиции ограничен максимальной разрешённой маржой.",
    PositionStatus.BELOW_MINIMUM_ORDER: (
        "Рассчитанный размер позиции меньше минимального размера ордера BingX. "
        "Не повышайте риск автоматически."
    ),
    PositionStatus.LOW_RISK_REWARD: "Соотношение риска и прибыли ниже минимального. Сделка не рекомендована.",
    PositionStatus.FEES_TOO_HIGH: (
        "Комиссии и ожидаемое проскальзывание поглощают большую часть потенциальной прибыли по TP1."
    ),
    PositionStatus.MISSING_MARKET_RULES: "Параметры торговой пары недоступны — используются значения из настроек.",
    PositionStatus.INVALID_SETTINGS: "Настройки риск-менеджмента некорректны.",
}


def _dec(value) -> Decimal:
    """The one sanctioned float/int/str -> Decimal conversion in this module —
    always via str(), never Decimal(float) directly (which reproduces binary
    floating-point rounding error inside a Decimal).
    """
    if isinstance(value, Decimal):
        return value
    return Decimal(str(value))


@dataclass(frozen=True)
class RiskSettings:
    account_balance_usdt: Decimal
    risk_percent: Decimal
    leverage: int
    margin_mode: MarginMode
    max_margin_percent: Decimal
    maker_fee_percent: Decimal
    taker_fee_percent: Decimal
    slippage_percent: Decimal
    min_risk_reward: Decimal
    entry_price_mode: EntryPriceMode
    quantity_step: Decimal
    price_step: Decimal
    minimum_order_notional_usdt: Decimal
    minimum_quantity: Decimal = Decimal("0")
    # None means "assume no other open positions" -> falls back to account_balance_usdt.
    available_balance_usdt: Decimal | None = None
    bingx_order_input_mode: BingXInputMode = BingXInputMode.MARGIN_USDT

    @classmethod
    def from_raw(
        cls,
        *,
        account_balance_usdt,
        risk_percent,
        leverage,
        margin_mode,
        max_margin_percent,
        maker_fee_percent,
        taker_fee_percent,
        slippage_percent,
        min_risk_reward,
        entry_price_mode,
        quantity_step,
        price_step,
        minimum_order_notional_usdt,
        minimum_quantity=0,
        available_balance_usdt=None,
        bingx_order_input_mode=BingXInputMode.MARGIN_USDT,
    ) -> "RiskSettings":
        """Builds RiskSettings from plain Python/JSON values (int/float/str) —
        the only place raw numbers get promoted to Decimal in this module.
        """
        return cls(
            account_balance_usdt=_dec(account_balance_usdt),
            risk_percent=_dec(risk_percent),
            leverage=int(leverage),
            margin_mode=MarginMode(margin_mode),
            max_margin_percent=_dec(max_margin_percent),
            maker_fee_percent=_dec(maker_fee_percent),
            taker_fee_percent=_dec(taker_fee_percent),
            slippage_percent=_dec(slippage_percent),
            min_risk_reward=_dec(min_risk_reward),
            entry_price_mode=EntryPriceMode(entry_price_mode),
            quantity_step=_dec(quantity_step),
            price_step=_dec(price_step),
            minimum_order_notional_usdt=_dec(minimum_order_notional_usdt),
            minimum_quantity=_dec(minimum_quantity),
            available_balance_usdt=_dec(available_balance_usdt) if available_balance_usdt is not None else None,
            bingx_order_input_mode=BingXInputMode(bingx_order_input_mode),
        )


DEFAULT_RISK_SETTINGS = RiskSettings.from_raw(
    account_balance_usdt=100.0,
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


@dataclass(frozen=True)
class TakeProfitTarget:
    label: str
    price: Decimal
    close_percent: Decimal


@dataclass(frozen=True)
class TradeScenario:
    signal: Literal["LONG", "SHORT"]
    entry_from: Decimal
    entry_to: Decimal
    stop_loss: Decimal
    take_profits: list[TakeProfitTarget] = field(default_factory=list)
    entry_order_type: Literal["MARKET", "LIMIT", "TRIGGER"] = "LIMIT"
    # True only when the caller can guarantee a passive (non-crossing) limit
    # fill — otherwise entry is costed at the conservative taker rate, same
    # as every other assumption in this module.
    assume_maker_entry: bool = False
    manual_entry_price: Decimal | None = None


@dataclass(frozen=True)
class RiskValidationResult:
    ok: bool
    status: PositionStatus | None
    issues: list[str]


def validate_scenario_and_settings(scenario: TradeScenario, settings: RiskSettings) -> RiskValidationResult:
    """Sanity-checks the calculator's OWN inputs (not the AI's raw text —
    that's `trade_validator.py`'s job, on `ai_schema.TradePlan`, before a
    `TradeScenario` is even built). Runs strictly before any arithmetic so a
    NaN/negative/inverted input can never silently produce a bogus position.
    """
    settings_issues: list[str] = []
    if settings.account_balance_usdt <= 0:
        settings_issues.append("Баланс счёта должен быть положительным.")
    if settings.leverage < 1:
        settings_issues.append("Плечо должно быть не меньше 1.")
    if not (Decimal("0") <= settings.risk_percent <= Decimal("100")):
        settings_issues.append("Риск на сделку должен быть в диапазоне 0-100%.")
    if not (Decimal("0") < settings.max_margin_percent <= Decimal("100")):
        settings_issues.append("Максимальная доля маржи должна быть в диапазоне (0, 100]%.")
    if settings.maker_fee_percent < 0 or settings.taker_fee_percent < 0:
        settings_issues.append("Комиссии не могут быть отрицательными.")
    if settings.slippage_percent < 0:
        settings_issues.append("Проскальзывание не может быть отрицательным.")
    if settings.quantity_step <= 0:
        settings_issues.append("Шаг количества должен быть положительным.")
    if settings.price_step <= 0:
        settings_issues.append("Шаг цены должен быть положительным.")
    if settings.minimum_order_notional_usdt < 0:
        settings_issues.append("Минимальный номинал ордера не может быть отрицательным.")
    if settings.min_risk_reward <= 0:
        settings_issues.append("Минимальный Risk/Reward должен быть положительным.")

    if settings_issues:
        return RiskValidationResult(ok=False, status=PositionStatus.INVALID_SETTINGS, issues=settings_issues)

    if scenario.signal not in ("LONG", "SHORT"):
        return RiskValidationResult(
            ok=False, status=PositionStatus.WAIT, issues=["WAIT/неизвестный сигнал не рассчитывается."]
        )

    issues: list[str] = []

    for field_name, value in (
        ("entry_from", scenario.entry_from),
        ("entry_to", scenario.entry_to),
        ("stop_loss", scenario.stop_loss),
    ):
        if not value.is_finite():
            issues.append(f"{field_name} должно быть конечным числом (не NaN/Infinity).")
        elif value <= 0:
            issues.append(f"{field_name} должно быть положительным.")

    for tp in scenario.take_profits:
        if not tp.price.is_finite() or tp.price <= 0:
            issues.append(f"Цена {tp.label} должна быть положительным конечным числом.")
        if not tp.close_percent.is_finite() or tp.close_percent <= 0:
            issues.append(f"close_percent для {tp.label} должен быть положительным.")

    if issues:
        return RiskValidationResult(ok=False, status=PositionStatus.INVALID_SCENARIO, issues=issues)

    if scenario.entry_from > scenario.entry_to:
        issues.append("Зона входа инвертирована (entry_from > entry_to).")

    is_long = scenario.signal == "LONG"
    if is_long and scenario.stop_loss >= scenario.entry_from:
        issues.append("Stop loss должен быть ниже зоны входа для LONG.")
    if not is_long and scenario.stop_loss <= scenario.entry_to:
        issues.append("Stop loss должен быть выше зоны входа для SHORT.")

    for tp in scenario.take_profits:
        if is_long and tp.price <= scenario.entry_to:
            issues.append(f"{tp.label} должен быть выше зоны входа для LONG.")
        if not is_long and tp.price >= scenario.entry_from:
            issues.append(f"{tp.label} должен быть ниже зоны входа для SHORT.")

    prices = [tp.price for tp in scenario.take_profits]
    for i in range(1, len(prices)):
        if is_long and prices[i] <= prices[i - 1]:
            issues.append("Take profits должны идти по возрастанию для LONG (TP1 < TP2 < TP3).")
            break
        if not is_long and prices[i] >= prices[i - 1]:
            issues.append("Take profits должны идти по убыванию для SHORT (TP1 > TP2 > TP3).")
            break

    if scenario.take_profits:
        close_percent_sum = sum((tp.close_percent for tp in scenario.take_profits), Decimal("0"))
        if not (Decimal("99") <= close_percent_sum <= Decimal("101")):
            issues.append(f"Сумма close_percent должна быть около 100% (сейчас {close_percent_sum}%).")

    if scenario.manual_entry_price is not None:
        if not scenario.manual_entry_price.is_finite() or scenario.manual_entry_price <= 0:
            issues.append("manual_entry_price должна быть положительным конечным числом.")
        elif not (scenario.entry_from <= scenario.manual_entry_price <= scenario.entry_to):
            issues.append("manual_entry_price должна быть внутри зоны входа.")

    if settings.entry_price_mode == EntryPriceMode.MANUAL and scenario.manual_entry_price is None:
        issues.append("Режим MANUAL требует указания manual_entry_price.")

    if issues:
        return RiskValidationResult(ok=False, status=PositionStatus.INVALID_SCENARIO, issues=issues)

    return RiskValidationResult(ok=True, status=None, issues=[])


def resolve_entry_price(scenario: TradeScenario, mode: EntryPriceMode) -> Decimal:
    lo, hi = scenario.entry_from, scenario.entry_to
    if lo > hi:
        lo, hi = hi, lo

    if mode == EntryPriceMode.MANUAL:
        if scenario.manual_entry_price is None:
            raise ValueError("MANUAL entry_price_mode requires manual_entry_price")
        return scenario.manual_entry_price
    if mode == EntryPriceMode.MIDPOINT:
        return (lo + hi) / 2
    if mode == EntryPriceMode.CONSERVATIVE:
        return hi if scenario.signal == "LONG" else lo
    if mode == EntryPriceMode.BEST_CASE:
        return lo if scenario.signal == "LONG" else hi
    raise ValueError(f"Unknown entry_price_mode: {mode}")


def round_down_to_step(value: Decimal, step: Decimal) -> Decimal:
    """Never rounds up — rounding a position's quantity up would silently
    increase its risk above what was budgeted.
    """
    if step <= 0:
        return value
    return (value / step).to_integral_value(rounding=ROUND_DOWN) * step


def round_to_step(value: Decimal, step: Decimal) -> Decimal:
    """Nearest-step rounding — used for prices (display/BingX tick size),
    where rounding direction doesn't change how much is at risk.
    """
    if step <= 0:
        return value
    return (value / step).to_integral_value(rounding=ROUND_HALF_UP) * step


@dataclass(frozen=True)
class TakeProfitResult:
    label: str
    price: Decimal
    distance: Decimal
    distance_percent: Decimal
    gross_profit_usdt: Decimal
    fees_usdt: Decimal
    net_profit_usdt: Decimal
    gross_rr: Decimal
    net_rr: Decimal | None


@dataclass(frozen=True)
class PositionCalculation:
    status: PositionStatus
    limited_by: Literal["RISK", "MARGIN"] | None
    signal: str
    entry_price: Decimal
    entry_zone: tuple[Decimal, Decimal]
    entry_price_mode: EntryPriceMode
    stop_loss: Decimal
    stop_distance: Decimal
    account_balance_usdt: Decimal
    risk_percent: Decimal
    risk_budget_usdt: Decimal
    leverage: int
    margin_mode: MarginMode
    position_size_coin: Decimal
    position_size_coin_rounded: Decimal
    position_notional_usdt: Decimal
    required_margin_usdt: Decimal
    max_allowed_margin_usdt: Decimal
    entry_fee_usdt: Decimal
    stop_exit_fee_usdt: Decimal
    slippage_estimate_usdt: Decimal
    total_expected_loss_usdt: Decimal
    actual_risk_percent: Decimal
    take_profit_results: list[TakeProfitResult]
    primary_target_label: str | None
    free_balance_after_open_usdt: Decimal
    bingx_fields: dict[str, Decimal]
    warnings: list[str]
    issues: list[str]

    @property
    def ok(self) -> bool:
        return self.status == PositionStatus.VALID


_ZERO = Decimal("0")


def _blank_calculation(status: PositionStatus, issues: list[str], scenario: TradeScenario | None = None) -> PositionCalculation:
    """Used when validation fails before any arithmetic is possible — only
    `status`/`issues` are meaningful on the returned object, every numeric
    field is a zeroed placeholder (never a fabricated number).
    """
    return PositionCalculation(
        status=status,
        limited_by=None,
        signal=scenario.signal if scenario else "",
        entry_price=_ZERO,
        entry_zone=(_ZERO, _ZERO),
        entry_price_mode=EntryPriceMode.MIDPOINT,
        stop_loss=_ZERO,
        stop_distance=_ZERO,
        account_balance_usdt=_ZERO,
        risk_percent=_ZERO,
        risk_budget_usdt=_ZERO,
        leverage=0,
        margin_mode=MarginMode.ISOLATED,
        position_size_coin=_ZERO,
        position_size_coin_rounded=_ZERO,
        position_notional_usdt=_ZERO,
        required_margin_usdt=_ZERO,
        max_allowed_margin_usdt=_ZERO,
        entry_fee_usdt=_ZERO,
        stop_exit_fee_usdt=_ZERO,
        slippage_estimate_usdt=_ZERO,
        total_expected_loss_usdt=_ZERO,
        actual_risk_percent=_ZERO,
        take_profit_results=[],
        primary_target_label=None,
        free_balance_after_open_usdt=_ZERO,
        bingx_fields={},
        warnings=[],
        issues=issues,
    )


class PositionCalculator:
    """Bind once to a `RiskSettings`, then `calculate()` as many
    `TradeScenario`s as you like — changing only settings (e.g. the user
    edits risk % in the UI) never requires a new AI call, just a new
    `PositionCalculator(new_settings).calculate(same_scenario)`.
    """

    def __init__(self, settings: RiskSettings) -> None:
        self.settings = settings

    def calculate(self, scenario: TradeScenario) -> PositionCalculation:
        settings = self.settings
        validation = validate_scenario_and_settings(scenario, settings)
        if not validation.ok:
            return _blank_calculation(validation.status, validation.issues, scenario)

        entry_price = resolve_entry_price(scenario, settings.entry_price_mode)
        entry_price = round_to_step(entry_price, settings.price_step)

        is_long = scenario.signal == "LONG"
        stop_distance = (entry_price - scenario.stop_loss) if is_long else (scenario.stop_loss - entry_price)

        risk_budget_usdt = settings.account_balance_usdt * settings.risk_percent / Decimal("100")

        entry_fee_rate = (
            settings.maker_fee_percent / Decimal("100")
            if (scenario.entry_order_type == "LIMIT" and scenario.assume_maker_entry)
            else settings.taker_fee_percent / Decimal("100")
        )
        stop_exit_fee_rate = settings.taker_fee_percent / Decimal("100")
        tp_exit_fee_rate = settings.taker_fee_percent / Decimal("100")
        slippage_rate = settings.slippage_percent / Decimal("100")

        per_unit_loss = (
            stop_distance
            + entry_price * entry_fee_rate
            + scenario.stop_loss * stop_exit_fee_rate
            + entry_price * slippage_rate
        )
        size_by_risk = risk_budget_usdt / per_unit_loss if per_unit_loss > 0 else _ZERO

        available_balance = settings.available_balance_usdt
        if available_balance is None:
            available_balance = settings.account_balance_usdt
        max_allowed_margin_usdt = min(
            settings.account_balance_usdt * settings.max_margin_percent / Decimal("100"),
            available_balance,
        )
        size_by_margin = (max_allowed_margin_usdt * settings.leverage) / entry_price

        warnings: list[str] = []
        if size_by_margin < size_by_risk:
            limited_by: Literal["RISK", "MARGIN"] | None = "MARGIN"
            warnings.append(STATUS_MESSAGES[PositionStatus.MARGIN_LIMIT])
        else:
            limited_by = "RISK"
        position_size_coin = min(size_by_risk, size_by_margin)

        position_size_coin_rounded = round_down_to_step(position_size_coin, settings.quantity_step)
        if position_size_coin_rounded < settings.minimum_quantity:
            position_size_coin_rounded = _ZERO

        if settings.margin_mode == MarginMode.CROSS:
            warnings.append(
                "Используется CROSS вместо ISOLATED — потенциальный убыток не ограничен только маржой этой позиции."
            )

        # Recompute everything from the ROUNDED size, per spec — never trust
        # the pre-rounding numbers for the final result.
        position_notional_usdt = position_size_coin_rounded * entry_price
        required_margin_usdt = (
            (position_notional_usdt / settings.leverage) if settings.leverage else _ZERO
        )
        entry_fee_usdt = position_notional_usdt * entry_fee_rate
        stop_exit_notional = position_size_coin_rounded * scenario.stop_loss
        stop_exit_fee_usdt = stop_exit_notional * stop_exit_fee_rate
        slippage_estimate_usdt = position_notional_usdt * slippage_rate
        price_loss = position_size_coin_rounded * stop_distance
        total_expected_loss_usdt = price_loss + entry_fee_usdt + stop_exit_fee_usdt + slippage_estimate_usdt
        actual_risk_percent = (
            (total_expected_loss_usdt / settings.account_balance_usdt * Decimal("100"))
            if settings.account_balance_usdt > 0
            else _ZERO
        )
        free_balance_after_open_usdt = settings.account_balance_usdt - required_margin_usdt

        issues: list[str] = []
        status = PositionStatus.VALID
        if (
            position_size_coin_rounded <= 0
            or position_notional_usdt < settings.minimum_order_notional_usdt
            or position_size_coin_rounded < settings.minimum_quantity
        ):
            status = PositionStatus.BELOW_MINIMUM_ORDER
            issues.append(STATUS_MESSAGES[status])

        take_profit_results: list[TakeProfitResult] = []
        primary_target_label: str | None = None
        any_gross_ok = False
        for tp in scenario.take_profits:
            portion = position_size_coin_rounded * tp.close_percent / Decimal("100")
            reward_per_unit = (tp.price - entry_price) if is_long else (entry_price - tp.price)
            gross_profit_usdt = portion * reward_per_unit
            entry_fee_share = entry_fee_usdt * tp.close_percent / Decimal("100")
            tp_exit_notional = portion * tp.price
            tp_exit_fee = tp_exit_notional * tp_exit_fee_rate
            fees_usdt = entry_fee_share + tp_exit_fee
            net_profit_usdt = gross_profit_usdt - fees_usdt
            distance = abs(tp.price - entry_price)
            distance_percent = (distance / entry_price * Decimal("100")) if entry_price > 0 else _ZERO
            gross_rr = (reward_per_unit / stop_distance) if stop_distance > 0 else _ZERO
            net_rr = (net_profit_usdt / total_expected_loss_usdt) if total_expected_loss_usdt > 0 else None

            if gross_rr >= settings.min_risk_reward:
                any_gross_ok = True
            if primary_target_label is None and net_rr is not None and net_rr >= settings.min_risk_reward:
                primary_target_label = tp.label

            take_profit_results.append(
                TakeProfitResult(
                    label=tp.label,
                    price=tp.price,
                    distance=distance,
                    distance_percent=distance_percent,
                    gross_profit_usdt=gross_profit_usdt,
                    fees_usdt=fees_usdt,
                    net_profit_usdt=net_profit_usdt,
                    gross_rr=gross_rr,
                    net_rr=net_rr,
                )
            )

        if status == PositionStatus.VALID and take_profit_results:
            if primary_target_label is None:
                status = PositionStatus.FEES_TOO_HIGH if any_gross_ok else PositionStatus.LOW_RISK_REWARD
                issues.append(STATUS_MESSAGES[status])
            else:
                for tp_result in take_profit_results:
                    if tp_result.label != primary_target_label and (
                        tp_result.net_rr is None or tp_result.net_rr < settings.min_risk_reward
                    ):
                        warnings.append(
                            f"{tp_result.label} имеет чистый R:R ниже минимального — основная цель: {primary_target_label}."
                        )

        bingx_fields = {
            BingXInputMode.MARGIN_USDT.value: required_margin_usdt,
            BingXInputMode.NOTIONAL_USDT.value: position_notional_usdt,
            BingXInputMode.COIN_QUANTITY.value: position_size_coin_rounded,
        }

        return PositionCalculation(
            status=status,
            limited_by=limited_by,
            signal=scenario.signal,
            entry_price=entry_price,
            entry_zone=(scenario.entry_from, scenario.entry_to),
            entry_price_mode=settings.entry_price_mode,
            stop_loss=scenario.stop_loss,
            stop_distance=stop_distance,
            account_balance_usdt=settings.account_balance_usdt,
            risk_percent=settings.risk_percent,
            risk_budget_usdt=risk_budget_usdt,
            leverage=settings.leverage,
            margin_mode=settings.margin_mode,
            position_size_coin=position_size_coin,
            position_size_coin_rounded=position_size_coin_rounded,
            position_notional_usdt=position_notional_usdt,
            required_margin_usdt=required_margin_usdt,
            max_allowed_margin_usdt=max_allowed_margin_usdt,
            entry_fee_usdt=entry_fee_usdt,
            stop_exit_fee_usdt=stop_exit_fee_usdt,
            slippage_estimate_usdt=slippage_estimate_usdt,
            total_expected_loss_usdt=total_expected_loss_usdt,
            actual_risk_percent=actual_risk_percent,
            take_profit_results=take_profit_results,
            primary_target_label=primary_target_label,
            free_balance_after_open_usdt=free_balance_after_open_usdt,
            bingx_fields=bingx_fields,
            warnings=warnings,
            issues=issues,
        )
