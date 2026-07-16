"""Real-order lifecycle: pre-flight safety checks -> entry/SL/TP placement
-> journal_db bookkeeping, for the semi-auto "Confirm and send to BingX"
flow (Stage 11). Callers other than a test or a future dashboard button are
`execution_engine.confirm_and_execute`, which runs the global (kill switch/
daily-limit/cooldown) gates BEFORE ever calling `place_entry_order` here.

`place_entry_order` re-derives every check that could have gone stale since
the trade_plans row was written (price, staleness, position size, balance)
rather than trusting the row's own `trade_permission`/`position_status` at
face value — those were computed once, at signal time, and the user may
click "confirm" minutes later.

Checks run cheapest/local-first, network only once every local check has
passed (see the Stage 11 plan for the exact ordering and rationale). The
FIRST real_orders row (the entry) is written to journal_db BEFORE any
BingX-mutating call — that write is this app's own idempotency guarantee,
independent of whatever BingX's real client-order-id parameter turns out to
be (see bingx_private_client.py's module docstring).

In `config.EXECUTION_DRY_RUN` (the hard default), every `bingx_private_client`
call raises `DryRunNotSent` instead of touching the network — this module
catches that at each call site and continues exactly as it would after a
real success, so the full pipeline (checks, DB writes, ordering) is
exercised end-to-end without ever sending a live request. The one
externally-visible difference is `OrderResult.status`: "DRY_RUN" instead of
"PLACED", and no real `exchange_order_id`s.
"""

from __future__ import annotations

import dataclasses
import time
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Literal

import bingx_client
import bingx_private_client
import config
import journal_db
import risk_settings_store
from risk_manager import (
    PositionCalculator,
    PositionStatus,
    TakeProfitTarget,
    TradeScenario,
    normalize_entry_order_type,
    round_down_to_step,
)


class _OrderPlacementFailed(Exception):
    """Internal-only — unwinds place_entry_order's leverage/margin-mode
    setup on a genuine (non-dry-run) BingX error."""


@dataclass(frozen=True)
class OrderResult:
    status: Literal["PLACED", "REJECTED", "ALREADY_EXISTS", "DRY_RUN"]
    reason: str | None
    real_order_ids: list[int]
    exchange_order_ids: list[str]


@dataclass(frozen=True)
class CloseAllResult:
    symbol: str
    orders_cancelled: list[int]
    position_closed: bool
    errors: list[str]


# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------


def _actionable_trade_plan_strict(trade_plan: dict | None) -> tuple[bool, str | None]:
    """Mirrors paper_trading._actionable_trade_plan but STRICTER: also
    requires trade_permission == 'ALLOWED'. A real, immediate order needs
    price in the entry zone RIGHT NOW — unlike a resting paper order, which
    can sit out WAITING_TRIGGER / PRICE_OUTSIDE_ENTRY_ZONE.
    """
    if trade_plan is None:
        return False, "trade plan not found"
    if trade_plan.get("overall_signal") not in ("LONG", "SHORT"):
        return False, f"signal is {trade_plan.get('overall_signal')!r}, not LONG/SHORT"
    if trade_plan.get("entry_from") is None or trade_plan.get("entry_to") is None:
        return False, "no entry zone on trade plan"
    if trade_plan.get("stop_loss_calc") is None and trade_plan.get("stop_loss") is None:
        return False, "no stop loss on trade plan"
    if trade_plan.get("position_status") != "VALID":
        return False, f"position_status is {trade_plan.get('position_status')!r}, not VALID"
    qty = trade_plan.get("position_size_coin_rounded")
    if qty is None or qty <= 0:
        return False, "position_size_coin_rounded is missing or non-positive"
    if trade_plan.get("trade_permission") != "ALLOWED":
        return False, f"trade_permission is {trade_plan.get('trade_permission')!r}, not ALLOWED"
    return True, None


def _is_expired(trade_plan: dict, now: float) -> bool:
    anchor = trade_plan.get("formed_at") or trade_plan.get("timestamp")
    valid_for_minutes = trade_plan.get("valid_for_minutes")
    if anchor is None or valid_for_minutes is None:
        return False
    return now >= anchor + valid_for_minutes * 60


def _price_still_valid(trade_plan: dict, current_price: float) -> tuple[bool, str | None]:
    """Re-derives consensus_engine's price-zone tolerance check and
    position_service's stop/TP1-already-crossed check against the price at
    EXECUTION time — trade_plans.trade_permission was computed once, at
    signal time, and can be stale by the time the user clicks confirm.
    """
    entry_from, entry_to = trade_plan["entry_from"], trade_plan["entry_to"]
    lo, hi = min(entry_from, entry_to), max(entry_from, entry_to)
    tolerance = hi * config.EXECUTION_PRICE_ZONE_TOLERANCE_FRACTION
    if current_price < lo - tolerance:
        return False, f"current price {current_price} is below the entry zone ({lo}-{hi})"
    if current_price > hi + tolerance:
        return False, f"current price {current_price} is above the entry zone ({lo}-{hi})"

    is_long = trade_plan["overall_signal"] == "LONG"
    stop_loss = trade_plan.get("stop_loss_calc")
    stop_loss = float(stop_loss) if stop_loss is not None else trade_plan.get("stop_loss")
    take_profits = trade_plan.get("take_profits") or []

    if stop_loss is not None:
        breached = current_price <= stop_loss if is_long else current_price >= stop_loss
        if breached:
            return False, f"current price {current_price} has already crossed the stop loss ({stop_loss})"
    if take_profits:
        tp1_price = take_profits[0][1]
        reached = current_price >= tp1_price if is_long else current_price <= tp1_price
        if reached:
            return False, f"current price {current_price} has already reached TP1 ({tp1_price})"
    return True, None


def _scenario_from_trade_plan(trade_plan: dict) -> TradeScenario:
    stop_loss_calc = trade_plan.get("stop_loss_calc")
    stop_loss = stop_loss_calc if stop_loss_calc is not None else Decimal(str(trade_plan["stop_loss"]))
    take_profits = trade_plan.get("take_profits") or []
    return TradeScenario(
        signal=trade_plan["overall_signal"],
        entry_from=Decimal(str(trade_plan["entry_from"])),
        entry_to=Decimal(str(trade_plan["entry_to"])),
        stop_loss=stop_loss,
        take_profits=[
            TakeProfitTarget(label=label, price=Decimal(str(price)), close_percent=Decimal(str(close_percent)))
            for label, price, close_percent in take_profits
        ],
        entry_order_type=normalize_entry_order_type(trade_plan.get("entry_type") or ""),
    )


def _parse_balance(data: Any) -> Decimal | None:
    """Best-effort parse of BingX's balance response — the exact shape is
    unconfirmed (see bingx_private_client.py's module docstring), so this
    tries a few plausible keys and returns None (never a fabricated number)
    if none match, which the caller treats as a hard rejection.
    """
    if not isinstance(data, dict):
        return None
    candidate = data.get("balance") if isinstance(data.get("balance"), dict) else data
    for key in ("balance", "availableMargin", "equity"):
        value = candidate.get(key)
        if value is not None:
            try:
                return Decimal(str(value))
            except Exception:
                continue
    return None


def _extract_order_id(data: Any) -> str | None:
    if not isinstance(data, dict):
        return None
    order = data.get("order") if isinstance(data.get("order"), dict) else data
    for key in ("orderId", "orderID", "order_id"):
        if key in order:
            return str(order[key])
    return None


def _opening_bingx_side(position_side: str) -> str:
    return "BUY" if position_side == "LONG" else "SELL"


def _closing_bingx_side(position_side: str) -> str:
    return "SELL" if position_side == "LONG" else "BUY"


def _call_private_or_raise(fn, *args, **kwargs) -> Any:
    try:
        return fn(*args, **kwargs)
    except bingx_private_client.DryRunNotSent:
        return None
    except bingx_client.BingXError as exc:
        raise _OrderPlacementFailed(str(exc)) from exc


def _rejected(reason: str) -> OrderResult:
    return OrderResult(status="REJECTED", reason=reason, real_order_ids=[], exchange_order_ids=[])


# ---------------------------------------------------------------------------
# place_entry_order
# ---------------------------------------------------------------------------


def _place_exit_order(
    conn,
    *,
    symbol: str,
    position_side: str,
    bingx_order_type: Literal["STOP_MARKET", "TAKE_PROFIT_MARKET"],
    trigger_price: Decimal,
    quantity: Decimal,
    trade_plan_id: int,
    leverage: int,
    margin_mode: str,
    now: float,
    label: str | None = None,
) -> tuple[int, str | None, str | None]:
    """Inserts a PENDING real_orders row for one resting stop-loss/take-
    profit order, then attempts to place it. Returns (real_order_id,
    exchange_order_id_or_None, error_or_None) — never raises; a failed SL
    needs different handling (critical) than a failed TP (a warning), so the
    caller decides what to do with the error.
    """
    order_id = journal_db.insert_real_order(
        conn,
        symbol=symbol,
        side=position_side,
        order_type="TRIGGER",
        quantity=quantity,
        trigger_price=trigger_price,
        stop_loss=(trigger_price if bingx_order_type == "STOP_MARKET" else None),
        take_profit=(trigger_price if bingx_order_type == "TAKE_PROFIT_MARKET" else None),
        leverage=leverage,
        margin_mode=margin_mode,
        trade_plan_id=trade_plan_id,
        status="PENDING",
        now=now,
        notes=("DRY_RUN: signed request built, not sent" if config.EXECUTION_DRY_RUN else None),
        label=label,
        is_dry_run=config.EXECUTION_DRY_RUN,
    )
    try:
        data = bingx_private_client.place_order(
            symbol,
            _closing_bingx_side(position_side),
            position_side,
            bingx_order_type,
            str(quantity),
            stop_price=str(trigger_price),
            client_order_id=f"neurotrade-{order_id}",
        )
        exchange_order_id = _extract_order_id(data)
        if exchange_order_id is not None:
            journal_db.set_real_order_exchange_id(conn, order_id, exchange_order_id, status="OPEN", now=now)
        else:
            journal_db.update_order_status(conn, "real_orders", order_id, "OPEN", now=now)
        return order_id, exchange_order_id, None
    except bingx_private_client.DryRunNotSent:
        return order_id, None, None
    except bingx_client.BingXError as exc:
        journal_db.update_order_status(conn, "real_orders", order_id, "REJECTED", now=now)
        return order_id, None, str(exc)


def place_entry_order(conn, trade_plan_id: int, *, now: float | None = None) -> OrderResult:
    now = now if now is not None else time.time()

    existing = journal_db.get_real_order_by_trade_plan_id(conn, trade_plan_id)
    if existing is not None:
        exchange_ids = [existing["exchange_order_id"]] if existing.get("exchange_order_id") else []
        return OrderResult(
            status="ALREADY_EXISTS",
            reason="a real order already exists for this trade_plan_id",
            real_order_ids=[existing["id"]],
            exchange_order_ids=exchange_ids,
        )

    trade_plan = journal_db.get_trade_plan(conn, trade_plan_id)
    ok, reason = _actionable_trade_plan_strict(trade_plan)
    if not ok:
        return _rejected(reason)

    if _is_expired(trade_plan, now):
        return _rejected("signal expired (past valid_for_minutes since formed_at)")

    margin_mode = trade_plan.get("margin_mode")
    if margin_mode != "ISOLATED":
        return _rejected(f"margin_mode is {margin_mode!r} — only ISOLATED is allowed at this stage")

    symbol = trade_plan["symbol"]

    try:
        current_price = bingx_client.get_price(config.to_bingx_symbol(symbol))
    except bingx_client.BingXError as exc:
        return _rejected(f"failed to fetch current price: {exc}")

    zone_ok, zone_reason = _price_still_valid(trade_plan, current_price)
    if not zone_ok:
        return _rejected(zone_reason)

    real_balance: Decimal | None
    try:
        balance_data = bingx_private_client.get_balance()
        real_balance = _parse_balance(balance_data)
        if real_balance is None:
            return _rejected("could not parse BingX balance response")
    except bingx_private_client.DryRunNotSent:
        real_balance = None  # DRY_RUN: no live balance to check — fall back to configured settings below.
    except bingx_client.BingXError as exc:
        return _rejected(f"failed to fetch balance: {exc}")

    settings = risk_settings_store.load()
    if real_balance is not None:
        settings = dataclasses.replace(settings, account_balance_usdt=real_balance, available_balance_usdt=None)

    scenario = _scenario_from_trade_plan(trade_plan)
    calculation = PositionCalculator(settings).calculate(scenario)
    if calculation.status != PositionStatus.VALID:
        return _rejected(f"position size recompute is not VALID: {calculation.status.value}")

    if journal_db.get_open_positions(conn, symbol, source="REAL"):
        return _rejected("an open REAL position already exists for this symbol (max one per instrument, no averaging)")

    if journal_db.get_pending_real_orders(conn, symbol):
        return _rejected("a pending REAL order already exists for this symbol")

    side = trade_plan["overall_signal"]
    quantity = calculation.position_size_coin_rounded
    stop_loss = calculation.stop_loss
    entry_order_type = normalize_entry_order_type(trade_plan.get("entry_type") or "")
    take_profits = [tuple(tp) for tp in (trade_plan.get("take_profits") or [])]

    # Deliberately no stop_loss/take_profit on this row — those fields are
    # reserved for the SL/TP rows below, so position_manager.py can tell an
    # entry order apart from an exit order by "stop_loss/take_profit is
    # NULL", without a dedicated purpose/role column (see the Stage 11 plan:
    # no DDL changes this stage).
    entry_order_id = journal_db.insert_real_order(
        conn,
        symbol=symbol,
        side=side,
        order_type=entry_order_type,
        quantity=quantity,
        leverage=settings.leverage,
        margin_mode=margin_mode,
        trade_plan_id=trade_plan_id,
        status="PENDING",
        now=now,
        notes=("DRY_RUN: signed request built, not sent" if config.EXECUTION_DRY_RUN else None),
        is_dry_run=config.EXECUTION_DRY_RUN,
    )
    real_order_ids = [entry_order_id]

    try:
        _call_private_or_raise(bingx_private_client.set_leverage, symbol, side, settings.leverage)
        _call_private_or_raise(bingx_private_client.set_margin_type, symbol, "ISOLATED")
    except _OrderPlacementFailed as exc:
        journal_db.update_order_status(conn, "real_orders", entry_order_id, "REJECTED", now=now)
        return OrderResult(
            status="REJECTED",
            reason=f"leverage/margin-mode setup failed: {exc}",
            real_order_ids=real_order_ids,
            exchange_order_ids=[],
        )

    bingx_entry_order_type = "MARKET" if entry_order_type == "MARKET" else "LIMIT"
    entry_price = None if entry_order_type == "MARKET" else calculation.entry_price
    try:
        data = bingx_private_client.place_order(
            symbol,
            _opening_bingx_side(side),
            side,
            bingx_entry_order_type,
            str(quantity),
            price=(str(entry_price) if entry_price is not None else None),
            client_order_id=f"neurotrade-{entry_order_id}",
        )
        entry_exchange_id = _extract_order_id(data)
    except bingx_private_client.DryRunNotSent:
        entry_exchange_id = None
    except bingx_client.BingXError as exc:
        journal_db.update_order_status(conn, "real_orders", entry_order_id, "REJECTED", now=now)
        return OrderResult(
            status="REJECTED",
            reason=f"entry order placement failed: {exc}",
            real_order_ids=real_order_ids,
            exchange_order_ids=[],
        )

    exchange_order_ids: list[str] = []
    if entry_exchange_id is not None:
        journal_db.set_real_order_exchange_id(conn, entry_order_id, entry_exchange_id, status="OPEN", now=now)
        exchange_order_ids.append(entry_exchange_id)
    elif not config.EXECUTION_DRY_RUN:
        journal_db.update_order_status(conn, "real_orders", entry_order_id, "OPEN", now=now)

    sl_order_id, sl_exchange_id, sl_error = _place_exit_order(
        conn,
        symbol=symbol,
        position_side=side,
        bingx_order_type="STOP_MARKET",
        trigger_price=stop_loss,
        quantity=quantity,
        trade_plan_id=trade_plan_id,
        leverage=settings.leverage,
        margin_mode=margin_mode,
        now=now,
    )
    real_order_ids.append(sl_order_id)
    if sl_exchange_id is not None:
        exchange_order_ids.append(sl_exchange_id)
    if sl_error is not None:
        journal_db.log_event(
            conn,
            level="CRITICAL",
            event_code="REAL_STOP_LOSS_PLACEMENT_FAILED",
            source_module="order_manager",
            symbol=symbol,
            message=(
                f"Entry order placed but stop loss failed: {sl_error}. A real position may now be open "
                "WITHOUT a stop loss — manual intervention required."
            ),
            trade_plan_id=trade_plan_id,
        )
        return OrderResult(
            status="REJECTED",
            reason=(
                f"entry placed but stop loss failed: {sl_error} — CHECK BINGX MANUALLY, "
                "position may be unprotected"
            ),
            real_order_ids=real_order_ids,
            exchange_order_ids=exchange_order_ids,
        )

    for label, price, close_percent in take_profits:
        tp_qty = round_down_to_step(quantity * Decimal(str(close_percent)) / Decimal("100"), settings.quantity_step)
        if tp_qty <= 0:
            continue
        tp_order_id, tp_exchange_id, tp_error = _place_exit_order(
            conn,
            symbol=symbol,
            position_side=side,
            bingx_order_type="TAKE_PROFIT_MARKET",
            trigger_price=Decimal(str(price)),
            quantity=tp_qty,
            trade_plan_id=trade_plan_id,
            leverage=settings.leverage,
            margin_mode=margin_mode,
            now=now,
            label=label,
        )
        real_order_ids.append(tp_order_id)
        if tp_exchange_id is not None:
            exchange_order_ids.append(tp_exchange_id)
        if tp_error is not None:
            # A missing TP is not as dangerous as a missing SL (the position
            # is still protected) — logged, not fatal to the overall flow.
            journal_db.log_event(
                conn,
                level="WARNING",
                event_code="REAL_TAKE_PROFIT_PLACEMENT_FAILED",
                source_module="order_manager",
                symbol=symbol,
                message=f"Take profit {label} failed to place: {tp_error}.",
                trade_plan_id=trade_plan_id,
            )

    # DRY_RUN never gets a live fill confirmation at all, so it always
    # creates the position immediately (the only way to exercise the rest
    # of the pipeline locally). In LIVE mode, a MARKET entry fills
    # essentially instantly, so creating it here is accurate — but a
    # LIMIT/TRIGGER entry may sit unfilled for a while; creating a position
    # for it now would be a guess, not a fact. position_manager.check_entry_fill
    # (Stage 12) creates it later, only once BingX actually confirms the fill.
    if config.EXECUTION_DRY_RUN or bingx_entry_order_type == "MARKET":
        position_id = journal_db.insert_position(
            conn,
            symbol=symbol,
            side=side,
            source="REAL",
            entry_price=calculation.entry_price,
            quantity=quantity,
            leverage=settings.leverage,
            margin_mode=margin_mode,
            stop_loss=stop_loss,
            take_profit=(Decimal(str(take_profits[0][1])) if take_profits else None),
            trade_plan_id=trade_plan_id,
            real_order_id=entry_order_id,
            status="OPEN",
            now=now,
            is_dry_run=config.EXECUTION_DRY_RUN,
        )
        journal_db.insert_fill(
            conn,
            position_id=position_id,
            symbol=symbol,
            side=side,
            fill_type="ENTRY",
            price=calculation.entry_price,
            quantity=quantity,
            real_order_id=entry_order_id,
            now=now,
        )

    return OrderResult(
        status="DRY_RUN" if config.EXECUTION_DRY_RUN else "PLACED",
        reason=None,
        real_order_ids=real_order_ids,
        exchange_order_ids=exchange_order_ids,
    )


# ---------------------------------------------------------------------------
# close_all
# ---------------------------------------------------------------------------


def close_all(conn, symbol: str) -> CloseAllResult:
    """Manual 'close everything' safeguard — cancels every pending REAL
    order and market-closes any open REAL position for the symbol. Ready to
    be wired to a future dashboard button; this stage stays backend-only.
    """
    now = time.time()
    errors: list[str] = []
    cancelled: list[int] = []

    for order in journal_db.get_pending_real_orders(conn, symbol):
        exchange_id = order.get("exchange_order_id")
        try:
            if exchange_id is not None:
                bingx_private_client.cancel_order(symbol, exchange_id)
            journal_db.update_order_status(conn, "real_orders", order["id"], "CANCELLED", now=now)
            cancelled.append(order["id"])
        except bingx_private_client.DryRunNotSent:
            journal_db.update_order_status(conn, "real_orders", order["id"], "CANCELLED", now=now)
            cancelled.append(order["id"])
        except bingx_client.BingXError as exc:
            errors.append(f"failed to cancel real_order {order['id']}: {exc}")

    position_closed = False
    for position in journal_db.get_open_positions(conn, symbol, source="REAL"):
        try:
            current_price = bingx_client.get_price(config.to_bingx_symbol(symbol))
        except bingx_client.BingXError as exc:
            errors.append(f"failed to fetch price to close position {position['id']}: {exc}")
            continue

        try:
            bingx_private_client.place_order(
                symbol,
                _closing_bingx_side(position["side"]),
                position["side"],
                "MARKET",
                str(position["quantity"]),
                client_order_id=f"neurotrade-close-{position['id']}",
            )
        except bingx_private_client.DryRunNotSent:
            pass
        except bingx_client.BingXError as exc:
            errors.append(f"failed to market-close position {position['id']}: {exc}")
            continue

        exit_price = Decimal(str(current_price))
        journal_db.close_position(conn, position["id"], exit_price=exit_price, now=now)
        journal_db.insert_fill(
            conn,
            position_id=position["id"],
            symbol=symbol,
            side=position["side"],
            fill_type="MANUAL_CLOSE",
            price=exit_price,
            quantity=position["quantity"],
            now=now,
        )
        position_closed = True

    return CloseAllResult(symbol=symbol, orders_cancelled=cancelled, position_closed=position_closed, errors=errors)
