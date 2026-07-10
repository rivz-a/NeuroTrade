"""Programmatic validation of an AI-produced `TradePlan` (see `ai_schema.py`).

No LLM does this check — it's pure arithmetic on numbers the model already
committed to (entry zone, stop loss, take profits) plus market context
(current price, ATR, spread) that this app already computed. A plan that
fails validation is never shown as a ready-to-trade signal.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from ai_schema import TradePlan

Severity = Literal["reject", "warning"]

# Reused from the existing swing system prompt's own stated policy (R:R not
# worse than 1:1.5) — scalping gets a lighter floor since stops are tight
# and held for minutes, not hours.
MIN_RISK_REWARD = {"scalping": 1.0, "swing": 1.5}

# How far (in ATR multiples) an entry may sit from the current price before
# it's considered stale/unreachable, and how tight/wide a stop may be.
MAX_ENTRY_DISTANCE_ATR = 3.0
MIN_STOP_DISTANCE_ATR = 0.15
MAX_STOP_DISTANCE_ATR = 6.0
RISK_REWARD_TOLERANCE = 0.2  # 20% — beyond this, declared vs computed R:R disagree


@dataclass(frozen=True)
class ValidationIssue:
    severity: Severity
    code: str
    message: str


@dataclass(frozen=True)
class ValidationResult:
    status: Literal["valid", "warning", "rejected"]
    issues: list[ValidationIssue] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.status != "rejected"


@dataclass(frozen=True)
class ValidationContext:
    current_price: float
    atr: float | None
    spread: float | None


def build_validation_context(snapshot: dict, mode: str) -> ValidationContext:
    """Pick the ATR timeframe that matches the mode's own stated horizon —
    5m for scalping (same timeframe the scalping prompt leans on), 1h for
    swing (same as the swing prompt's stop-placement policy).
    """
    atr_tf = "5m" if mode == "scalping" else "1h"
    atr = None
    timeframes = snapshot.get("timeframes") or {}
    tf_data = timeframes.get(atr_tf)
    if tf_data:
        atr = tf_data.get("indicators", {}).get("atr14")

    spread = None
    orderbook = snapshot.get("orderbook")
    if orderbook and orderbook.get("bids") and orderbook.get("asks"):
        spread = orderbook["asks"][0][0] - orderbook["bids"][0][0]

    return ValidationContext(current_price=snapshot["current_price"], atr=atr, spread=spread)


def _entry_mid(plan: TradePlan) -> float:
    return (plan.entry.from_ + plan.entry.to) / 2


def _reject(code: str, message: str) -> ValidationIssue:
    return ValidationIssue("reject", code, message)


def _warn(code: str, message: str) -> ValidationIssue:
    return ValidationIssue("warning", code, message)


def _validate_directional(plan: TradePlan, mode: str, ctx: ValidationContext) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    is_long = plan.signal == "LONG"
    entry_mid = _entry_mid(plan)

    # --- Stop loss on the correct side of the entry zone ---
    if is_long:
        if plan.stop_loss >= plan.entry.from_:
            issues.append(_reject("SL_WRONG_SIDE", "Stop loss не ниже зоны входа для LONG."))
    else:
        if plan.stop_loss <= plan.entry.to:
            issues.append(_reject("SL_WRONG_SIDE", "Stop loss не выше зоны входа для SHORT."))

    if plan.stop_loss == entry_mid:
        issues.append(_reject("SL_EQUALS_ENTRY", "Stop loss совпадает со входом."))

    # --- Take profit ordering ---
    tps = plan.take_profits
    if not tps:
        issues.append(_reject("NO_TAKE_PROFITS", "Не указано ни одного take profit."))
    else:
        tp1 = tps[0].price
        if is_long and tp1 <= plan.entry.to:
            issues.append(_reject("TP1_WRONG_SIDE", "TP1 не выше зоны входа для LONG."))
        if not is_long and tp1 >= plan.entry.from_:
            issues.append(_reject("TP1_WRONG_SIDE", "TP1 не ниже зоны входа для SHORT."))

        for prev, cur in zip(tps, tps[1:]):
            if is_long and cur.price <= prev.price:
                issues.append(_reject("TP_ORDER", f"{cur.label} не выше {prev.label} для LONG."))
            if not is_long and cur.price >= prev.price:
                issues.append(_reject("TP_ORDER", f"{cur.label} не ниже {prev.label} для SHORT."))

    # --- Distance checks (need ATR) ---
    if ctx.atr and ctx.atr > 0:
        entry_distance_atr = abs(entry_mid - ctx.current_price) / ctx.atr
        if entry_distance_atr > MAX_ENTRY_DISTANCE_ATR:
            issues.append(
                _warn(
                    "ENTRY_TOO_FAR",
                    f"Вход на {entry_distance_atr:.1f} ATR от текущей цены — вероятно, недостижим.",
                )
            )

        stop_distance_atr = abs(entry_mid - plan.stop_loss) / ctx.atr
        if stop_distance_atr < MIN_STOP_DISTANCE_ATR:
            issues.append(
                _warn("STOP_TOO_TIGHT", f"Stop loss всего {stop_distance_atr:.2f} ATR от входа — слишком узкий.")
            )
        elif stop_distance_atr > MAX_STOP_DISTANCE_ATR:
            issues.append(
                _warn("STOP_TOO_WIDE", f"Stop loss в {stop_distance_atr:.1f} ATR от входа — слишком широкий.")
            )

    # --- Stop not inside the current spread around price ---
    if ctx.spread and ctx.spread > 0:
        half_spread = ctx.spread / 2
        if abs(plan.stop_loss - ctx.current_price) < half_spread:
            issues.append(_reject("STOP_INSIDE_SPREAD", "Stop loss находится внутри текущего спреда."))

    # --- Risk/Reward: computed vs declared, and vs mode floor ---
    if tps:
        risk = abs(entry_mid - plan.stop_loss)
        reward_tp1 = abs(tps[0].price - entry_mid)
        if risk > 0:
            computed_rr = reward_tp1 / risk
            declared_rr = plan.risk_reward.tp1
            if declared_rr is not None and declared_rr > 0:
                diff_ratio = abs(computed_rr - declared_rr) / declared_rr
                if diff_ratio > RISK_REWARD_TOLERANCE:
                    issues.append(
                        _warn(
                            "RR_MISMATCH",
                            f"Заявленный R:R до TP1 ({declared_rr:.2f}) расходится с расчётным "
                            f"({computed_rr:.2f}).",
                        )
                    )
            min_rr = MIN_RISK_REWARD.get(mode, MIN_RISK_REWARD["scalping"])
            if computed_rr < min_rr:
                issues.append(
                    _reject(
                        "RR_TOO_LOW",
                        f"R:R до TP1 ({computed_rr:.2f}) ниже минимального для режима ({min_rr:.1f}).",
                    )
                )

    return issues


def _validate_wait(plan: TradePlan) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    if not plan.wait_conditions and not plan.invalidation_conditions:
        issues.append(
            _reject(
                "WAIT_NO_CONDITIONS",
                "WAIT без wait_conditions/invalidation_conditions — непонятно, что должно измениться.",
            )
        )
    return issues


def validate_trade_plan(plan: TradePlan, mode: str, ctx: ValidationContext) -> ValidationResult:
    issues: list[ValidationIssue] = []

    if plan.signal in ("LONG", "SHORT"):
        issues.extend(_validate_directional(plan, mode, ctx))
    else:
        issues.extend(_validate_wait(plan))

    if plan.valid_for_minutes <= 0:
        issues.append(_reject("BAD_VALIDITY", "Срок действия сигнала должен быть положительным."))

    if any(i.severity == "reject" for i in issues):
        status = "rejected"
    elif issues:
        status = "warning"
    else:
        status = "valid"

    return ValidationResult(status=status, issues=issues)
