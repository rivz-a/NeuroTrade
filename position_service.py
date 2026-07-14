"""Orchestrates a full position calculation for the currently active trade
plan of one mode. Bridges `consensus_engine.py`'s already-computed
`SelectedPlan` into `risk_manager.py`'s `TradeScenario`, merges live BingX
instrument rules (when available) into the user's `RiskSettings`, runs
`PositionCalculator`, and decides whether an "open LONG/SHORT" instruction
is actually warranted right now.

`calculate_active_position` itself never touches the network or AI — it
takes an already-resolved `InstrumentRules` as a plain argument. The one
network call in this module is `resolve_instrument_rules`, a free/public
BingX lookup (not AI) with a graceful FALLBACK when it fails, kept as a
separate function so the core calculation stays a pure, always-testable
function.

Reusing this on a settings change (a user editing risk % in the dashboard)
never needs a new AI call: the same cached `AIAnalysisResult` list already
holds everything `compute_consensus` needs, so recalculating is just
calling `calculate_active_position` again with a different `RiskSettings`.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from decimal import Decimal

import bingx_client
from ai_client import AIAnalysisResult
from consensus_engine import ConsensusResult, compute_consensus
from risk_manager import (
    STATUS_MESSAGES,
    BingXManualFields,
    InstrumentRules,
    PositionCalculation,
    PositionCalculator,
    PositionStatus,
    RiskSettings,
    TakeProfitTarget,
    TradeScenario,
    build_bingx_manual_fields,
    normalize_entry_order_type,
)

_DISPLAY_MESSAGES = {
    "STOP_ALREADY_BREACHED": (
        "Цена уже пересекла stop loss с момента формирования сценария — не входить, обновите анализ."
    ),
    "TP_ALREADY_REACHED": (
        "Цена уже достигла TP1 с момента формирования сценария — новый вход по этому плану не имеет смысла, "
        "обновите анализ."
    ),
}


@dataclass(frozen=True)
class PositionServiceResult:
    applicable: bool  # False = nothing to show (WAIT consensus, or no plan at all)
    reference_only: bool  # True = calculated for reference only, not an active instruction
    can_open: bool  # True only when it's actually warranted to show "open LONG/SHORT"
    display_status: str
    display_message: str
    consensus: ConsensusResult
    calculation: PositionCalculation | None
    bingx_fields: BingXManualFields | None
    instrument_rules: InstrumentRules


def resolve_instrument_rules(symbol: str, settings: RiskSettings) -> InstrumentRules:
    """Live BingX contract rules when the free `/quote/contracts` lookup
    succeeds, else a FALLBACK built from the user's own settings. Never
    raises — a BingX hiccup degrades to FALLBACK, it never blocks or
    crashes a dashboard render.
    """
    try:
        live = bingx_client.get_instrument_rules(symbol)
    except Exception:
        live = None

    if live is not None:
        return InstrumentRules(
            symbol=symbol,
            price_step=live["price_step"],
            quantity_step=live["quantity_step"],
            minimum_quantity=live["minimum_quantity"],
            minimum_notional_usdt=live["minimum_notional_usdt"],
            maximum_leverage=0,  # not exposed by BingX's public contracts endpoint
            source="BINGX_API",
        )
    return InstrumentRules(
        symbol=symbol,
        price_step=settings.price_step,
        quantity_step=settings.quantity_step,
        minimum_quantity=settings.minimum_quantity,
        minimum_notional_usdt=settings.minimum_order_notional_usdt,
        maximum_leverage=0,
        source="FALLBACK",
    )


def merge_instrument_rules(settings: RiskSettings, instrument_rules: InstrumentRules) -> RiskSettings:
    """Live BingX rules (when fetched successfully) override the user's own
    rounding/minimum-order settings — those are exchange facts, not
    preferences. A FALLBACK `InstrumentRules` (BingX unavailable) changes
    nothing; the user's own settings are already the fallback.
    """
    if instrument_rules.source != "BINGX_API":
        return settings
    return dataclasses.replace(
        settings,
        quantity_step=instrument_rules.quantity_step,
        price_step=instrument_rules.price_step,
        minimum_quantity=instrument_rules.minimum_quantity,
        minimum_order_notional_usdt=instrument_rules.minimum_notional_usdt,
    )


def calculate_active_position(
    results: list[AIAnalysisResult],
    mode: str,
    total_models: int,
    current_price: float,
    settings: RiskSettings,
    instrument_rules: InstrumentRules,
    now: float | None = None,
) -> PositionServiceResult:
    consensus = compute_consensus(results, mode, total_models, now=now, current_price=current_price)

    if consensus.overall_signal == "WAIT" or consensus.plan is None:
        return PositionServiceResult(
            applicable=False,
            reference_only=False,
            can_open=False,
            display_status=consensus.trade_permission,
            display_message=consensus.trade_permission_reason,
            consensus=consensus,
            calculation=None,
            bingx_fields=None,
            instrument_rules=instrument_rules,
        )

    plan = consensus.plan
    is_long = consensus.overall_signal == "LONG"

    # Checks consensus_engine can't do itself (it only knows the entry
    # zone, not stop/TP1 vs the live price): has the moment already passed?
    stop_already_breached = False
    tp_already_reached = False
    if plan.take_profits:
        tp1_price = plan.take_profits[0][1]
        if is_long:
            stop_already_breached = current_price <= plan.stop_loss
            tp_already_reached = current_price >= tp1_price
        else:
            stop_already_breached = current_price >= plan.stop_loss
            tp_already_reached = current_price <= tp1_price

    effective_settings = merge_instrument_rules(settings, instrument_rules)

    scenario = TradeScenario(
        signal=consensus.overall_signal,
        entry_from=Decimal(str(plan.entry_from)),
        entry_to=Decimal(str(plan.entry_to)),
        stop_loss=Decimal(str(plan.stop_loss)),
        take_profits=[
            TakeProfitTarget(label=label, price=Decimal(str(price)), close_percent=Decimal(str(close_percent)))
            for label, price, close_percent in plan.take_profits
        ],
        entry_order_type=normalize_entry_order_type(plan.entry_type),
    )

    calculation = PositionCalculator(effective_settings).calculate(scenario)
    bingx_fields = build_bingx_manual_fields(calculation, scenario.entry_order_type, effective_settings)

    if stop_already_breached:
        display_status, display_message = "STOP_ALREADY_BREACHED", _DISPLAY_MESSAGES["STOP_ALREADY_BREACHED"]
    elif tp_already_reached:
        display_status, display_message = "TP_ALREADY_REACHED", _DISPLAY_MESSAGES["TP_ALREADY_REACHED"]
    elif consensus.trade_permission != "ALLOWED":
        display_status, display_message = consensus.trade_permission, consensus.trade_permission_reason
    else:
        display_status = calculation.status.value
        display_message = STATUS_MESSAGES.get(calculation.status, "")

    reference_only = stop_already_breached or tp_already_reached or consensus.trade_permission != "ALLOWED"
    can_open = (not reference_only) and calculation.status == PositionStatus.VALID

    return PositionServiceResult(
        applicable=True,
        reference_only=reference_only,
        can_open=can_open,
        display_status=display_status,
        display_message=display_message,
        consensus=consensus,
        calculation=calculation,
        bingx_fields=bingx_fields,
        instrument_rules=instrument_rules,
    )


def decimalize(value):
    """Recursively converts `Decimal -> str` (never through `float`) inside
    a dataclass/dict/list structure, for JSON API responses. `str`-subclass
    Enums (MarginMode, EntryPriceMode, ...) already serialize fine as-is.
    """
    if isinstance(value, Decimal):
        return str(value)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return decimalize(dataclasses.asdict(value))
    if isinstance(value, dict):
        return {k: decimalize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [decimalize(v) for v in value]
    return value
