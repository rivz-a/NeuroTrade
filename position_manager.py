"""Real-position accompaniment engine (Stage 11 laid the reconciliation
foundation; Stage 12 completes it): RECONCILIATION plus active follow-up,
not simulation. Stop-loss and take-profit orders for a real position are
resting orders on BingX itself; BingX's own matching engine fills them, not
this app's code. Contrast with `paper_trading.py`, which simulates fills
against tick prices because there is no real exchange behind a paper order.

This module's jobs:
  - Confirm a LIMIT/TRIGGER entry actually filled before trusting it
    (`check_entry_fill`) or give up on it once the signal's own validity
    window has elapsed (`cancel_stale_entry_order`) — see order_manager.py's
    Stage 12 fix: only MARKET/DRY_RUN entries get an immediate local
    position; everything else waits for one of these two to resolve it.
  - Notice when a locally-OPEN real position's size on BingX has shrunk —
    fully (SL or the last TP) or partially (an earlier TP filled but the
    position stays open) — and keep journal_db in sync (`sync_position_status`).
  - Manage the resting stop as take-profits fill: move to breakeven after
    the first TP, then a real trailing stop after a second
    (`manage_stop_loss`) — the one case besides time/regime/data-quality
    where this app initiates a real mutation instead of just reconciling.
  - Get out immediately if the position's own thesis stops holding: past
    its time limit (`check_time_based_close`), the market regime reversed
    against it (`check_regime_close`), or the live data can no longer be
    trusted at all (`check_data_quality_close`).

Nothing here is wired into a scheduler; like `paper_trading.process_tick`,
these functions are meant to be called repeatedly by something external
(a script, a future cron/loop) — see execution_engine.monitor() for the
orchestration order.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

import bingx_client
import bingx_private_client
import config
import feature_engine
import journal_db
import market_data_engine
import market_regime

# Regime transitions that invalidate an open position's own thesis — a
# directional-reversal table, not "any regime change." REVERSAL_RISK applies
# to both sides (it signals an imminent reversal regardless of which way the
# position is already facing); RANGE/VOLATILITY_*/UNSTABLE are left alone —
# see the Stage 12 plan for why this table and not a broader one.
_LONG_INVALIDATING_REGIMES = frozenset({"TREND_DOWN", "BREAKOUT_DOWN", "REVERSAL_RISK"})
_SHORT_INVALIDATING_REGIMES = frozenset({"TREND_UP", "BREAKOUT_UP", "REVERSAL_RISK"})

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SyncResult:
    symbol: str
    positions_closed: list[int]
    positions_partially_closed: list[int]
    discrepancies: list[str]


@dataclass(frozen=True)
class ReconciliationReport:
    symbol: str
    local_open_not_on_exchange: list[int]
    exchange_open_not_local: list[str]
    errors: list[str]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _closing_bingx_side(position_side: str) -> str:
    return "SELL" if position_side == "LONG" else "BUY"


def _extract_order_id(data: Any) -> str | None:
    if not isinstance(data, dict):
        return None
    order = data.get("order") if isinstance(data.get("order"), dict) else data
    for key in ("orderId", "orderID", "order_id"):
        if key in order:
            return str(order[key])
    return None


def _live_position_quantity(live_positions: list[dict], symbol: str, side: str) -> Decimal:
    """Best-effort parse of BingX's positions response (unconfirmed exact
    shape — see bingx_private_client.py) — abs(positionAmt) for the
    matching symbol+positionSide, or 0 if absent/unparseable.
    """
    for p in live_positions:
        if p.get("symbol") != symbol or p.get("positionSide") != side:
            continue
        amt = p.get("positionAmt")
        if amt is None:
            continue
        try:
            return abs(Decimal(str(amt)))
        except Exception:
            continue
    return Decimal("0")


def _live_position_entry_price(live_positions: list[dict], symbol: str, side: str) -> Decimal | None:
    for p in live_positions:
        if p.get("symbol") != symbol or p.get("positionSide") != side:
            continue
        price = p.get("avgPrice") or p.get("entryPrice")
        if price is None:
            continue
        try:
            return Decimal(str(price))
        except Exception:
            continue
    return None


def _live_position_open(live_positions: list[dict], symbol: str, side: str) -> bool:
    return _live_position_quantity(live_positions, symbol, side) > 0


def _original_stop_for_position(conn, position: dict) -> Decimal | None:
    """Mirrors paper_trading._original_stop_for_position — positions.stop_loss
    itself gets mutated in place as the stop is moved (breakeven, trailing),
    so the ORIGINAL risk distance must come from the trade_plan's own
    (never-mutated) stop, not from the position row.
    """
    trade_plan_id = position.get("trade_plan_id")
    if trade_plan_id is not None:
        trade_plan = journal_db.get_trade_plan(conn, trade_plan_id)
        if trade_plan is not None:
            stop = trade_plan.get("stop_loss_calc")
            if stop is not None:
                return stop
            if trade_plan.get("stop_loss") is not None:
                return Decimal(str(trade_plan["stop_loss"]))
    return position.get("stop_loss")


def _live_regime(symbol: str) -> market_regime.RegimeResult | None:
    try:
        snapshot = market_data_engine.collect_snapshot(symbol)
    except bingx_client.BingXError:
        return None
    features = feature_engine.compute_features(snapshot)
    return market_regime.classify_regime(features)


def _live_data_quality(symbol: str) -> str | None:
    try:
        snapshot = market_data_engine.collect_snapshot(symbol)
    except bingx_client.BingXError:
        return None
    return snapshot.data_quality


def _market_close_position(conn, position: dict, *, fill_type: str, now: float) -> bool:
    """Shared by check_time_based_close/check_regime_close/
    check_data_quality_close — the three "something external says get out
    now" triggers, which only differ in WHY, not HOW: cancel every resting
    real order still tied to this position's trade_plan (its remaining
    SL/TPs), then market-close whatever quantity is left (already reduced
    by sync_position_status if earlier TPs had partially closed it).
    """
    symbol, side = position["symbol"], position["side"]

    for order in journal_db.get_pending_real_orders(conn, symbol):
        if order.get("trade_plan_id") != position.get("trade_plan_id"):
            continue
        try:
            if order.get("exchange_order_id") is not None:
                bingx_private_client.cancel_order(symbol, order["exchange_order_id"])
            journal_db.update_order_status(conn, "real_orders", order["id"], "CANCELLED", now=now)
        except bingx_private_client.DryRunNotSent:
            journal_db.update_order_status(conn, "real_orders", order["id"], "CANCELLED", now=now)
        except bingx_client.BingXError:
            return False

    try:
        current_price = bingx_client.get_price(config.to_bingx_symbol(symbol))
    except bingx_client.BingXError:
        current_price = None

    try:
        bingx_private_client.place_order(
            symbol,
            _closing_bingx_side(side),
            side,
            "MARKET",
            str(position["quantity"]),
            client_order_id=f"neurotrade-close-{position['id']}-{int(now)}",
        )
    except bingx_private_client.DryRunNotSent:
        pass
    except bingx_client.BingXError:
        return False

    exit_price = Decimal(str(current_price)) if current_price is not None else position["entry_price"]
    journal_db.close_position(conn, position["id"], exit_price=exit_price, now=now)
    journal_db.insert_fill(
        conn,
        position_id=position["id"],
        symbol=symbol,
        side=side,
        fill_type=fill_type,
        price=exit_price,
        quantity=position["quantity"],
        now=now,
    )
    return True


# ---------------------------------------------------------------------------
# check_entry_fill / cancel_stale_entry_order
# ---------------------------------------------------------------------------


def check_entry_fill(conn, trade_plan_id: int, *, now: float | None = None) -> bool:
    """"Контролировать исполнение": for a trade_plan whose ENTRY real_order
    is a LIMIT/TRIGGER order still PENDING/OPEN (see order_manager.py's
    Stage 12 fix — MARKET and DRY_RUN entries already have their position
    created immediately and this is a no-op for those), polls BingX and,
    once the entry has actually filled there, creates the local
    positions/fills(ENTRY) rows only now. Returns True if a position was
    newly created.
    """
    now = now if now is not None else time.time()

    entry_order = journal_db.get_entry_real_order_for_trade_plan(conn, trade_plan_id)
    if entry_order is None or entry_order["status"] not in ("PENDING", "OPEN"):
        return False
    if journal_db.get_position_by_trade_plan_id(conn, trade_plan_id) is not None:
        return False

    symbol, side = entry_order["symbol"], entry_order["side"]
    try:
        live_positions = bingx_private_client.get_positions(symbol) or []
    except bingx_private_client.DryRunNotSent:
        return False
    except bingx_client.BingXError:
        return False
    live_positions = live_positions if isinstance(live_positions, list) else []

    filled_qty = _live_position_quantity(live_positions, symbol, side)
    if filled_qty <= 0:
        return False  # still not filled

    fill_price = _live_position_entry_price(live_positions, symbol, side)
    if fill_price is None:
        fill_price = entry_order.get("price")
    if fill_price is None:
        try:
            fill_price = Decimal(str(bingx_client.get_price(config.to_bingx_symbol(symbol))))
        except bingx_client.BingXError:
            fill_price = None
    if fill_price is None:
        return False  # can't determine a fill price -> don't fabricate a position

    journal_db.update_order_status(conn, "real_orders", entry_order["id"], "FILLED", filled_quantity=filled_qty, now=now)

    stop_loss = None
    for order in journal_db.get_pending_real_orders(conn, symbol):
        if order.get("trade_plan_id") == trade_plan_id and order.get("stop_loss") is not None:
            stop_loss = order["stop_loss"]
            break

    position_id = journal_db.insert_position(
        conn,
        symbol=symbol,
        side=side,
        source="REAL",
        entry_price=fill_price,
        quantity=filled_qty,
        leverage=entry_order.get("leverage"),
        margin_mode=entry_order.get("margin_mode"),
        stop_loss=stop_loss,
        trade_plan_id=trade_plan_id,
        real_order_id=entry_order["id"],
        status="OPEN",
        now=now,
    )
    journal_db.insert_fill(
        conn,
        position_id=position_id,
        symbol=symbol,
        side=side,
        fill_type="ENTRY",
        price=fill_price,
        quantity=filled_qty,
        real_order_id=entry_order["id"],
        now=now,
    )
    return True


def cancel_stale_entry_order(conn, trade_plan_id: int, *, now: float | None = None) -> bool:
    """"Отменять неисполненный ордер": same staleness formula as
    paper_trading._expire_order_if_due (anchor=formed_at, valid_for_minutes)
    — a LIMIT/TRIGGER entry that never filled within the signal's own
    validity window means no position was ever opened; cancel it rather
    than leave it resting indefinitely.
    """
    now = now if now is not None else time.time()

    entry_order = journal_db.get_entry_real_order_for_trade_plan(conn, trade_plan_id)
    if entry_order is None or entry_order["status"] not in ("PENDING", "OPEN"):
        return False
    if journal_db.get_position_by_trade_plan_id(conn, trade_plan_id) is not None:
        return False  # it filled after all -> not stale, nothing to cancel

    trade_plan = journal_db.get_trade_plan(conn, trade_plan_id)
    if trade_plan is None:
        return False
    anchor = trade_plan.get("formed_at") or trade_plan.get("timestamp")
    valid_for_minutes = trade_plan.get("valid_for_minutes")
    if anchor is None or valid_for_minutes is None or now < anchor + valid_for_minutes * 60:
        return False

    try:
        if entry_order.get("exchange_order_id") is not None:
            bingx_private_client.cancel_order(entry_order["symbol"], entry_order["exchange_order_id"])
        journal_db.update_order_status(conn, "real_orders", entry_order["id"], "CANCELLED", now=now)
        return True
    except bingx_private_client.DryRunNotSent:
        journal_db.update_order_status(conn, "real_orders", entry_order["id"], "CANCELLED", now=now)
        return True
    except bingx_client.BingXError:
        return False


# ---------------------------------------------------------------------------
# sync_position_status
# ---------------------------------------------------------------------------


def sync_position_status(conn, symbol: str) -> SyncResult:
    """Polls BingX for this symbol's live positions/open orders and detects
    when a locally-OPEN real position's size has shrunk on the exchange —
    fully (0 remaining -> closed) or partially (some remaining -> a take
    profit fired, position stays OPEN with a smaller quantity). Which
    resting order disappeared from BingX's open-order list determines the
    recorded exit reason — a heuristic, not a guarantee, given the
    unconfirmed response shape (see bingx_private_client.py); any shrink
    where the exact cause can't be determined is still recorded (with a
    best-effort exit price) and flagged in `discrepancies` rather than left
    silently unaccounted for.
    """
    now = time.time()

    try:
        live_positions = bingx_private_client.get_positions(symbol) or []
        live_open_orders = bingx_private_client.get_open_orders(symbol) or []
    except bingx_private_client.DryRunNotSent:
        return SyncResult(symbol=symbol, positions_closed=[], positions_partially_closed=[], discrepancies=[])
    except bingx_client.BingXError as exc:
        return SyncResult(
            symbol=symbol, positions_closed=[], positions_partially_closed=[],
            discrepancies=[f"failed to fetch live state: {exc}"],
        )

    live_positions = live_positions if isinstance(live_positions, list) else []
    live_open_orders = live_open_orders if isinstance(live_open_orders, list) else []
    live_open_ids = {str(o.get("orderId")) for o in live_open_orders if o.get("orderId") is not None}

    positions_closed: list[int] = []
    positions_partially_closed: list[int] = []
    discrepancies: list[str] = []

    for position in journal_db.get_open_positions(conn, symbol, source="REAL"):
        local_qty = position["quantity"]
        live_qty = _live_position_quantity(live_positions, symbol, position["side"])
        if live_qty >= local_qty:
            continue  # unchanged -> nothing to reconcile

        closed_qty = local_qty - live_qty

        related_orders = [
            o
            for o in journal_db.get_pending_real_orders(conn, symbol)
            if o.get("trade_plan_id") == position.get("trade_plan_id")
        ]
        exit_fill_type, exit_label, exit_price = "STOP_LOSS", None, None
        for order in related_orders:
            exchange_id = order.get("exchange_order_id")
            if exchange_id is None or exchange_id in live_open_ids:
                continue  # still resting, or was never confirmed -> not the one that fired
            if order.get("take_profit") is not None:
                exit_fill_type, exit_label, exit_price = "TAKE_PROFIT", order.get("label"), order["take_profit"]
            elif order.get("stop_loss") is not None:
                exit_fill_type, exit_price = "STOP_LOSS", order["stop_loss"]
            journal_db.update_order_status(conn, "real_orders", order["id"], "FILLED", now=now)

        if exit_price is None:
            exit_price = position.get("stop_loss") or position["entry_price"]
            discrepancies.append(
                f"position {position['id']}: size shrank on BingX but the exact exit fill could not be "
                "determined — recorded with a best-effort exit price"
            )

        journal_db.insert_fill(
            conn,
            position_id=position["id"],
            symbol=symbol,
            side=position["side"],
            fill_type=exit_fill_type,
            label=exit_label,
            price=exit_price,
            quantity=closed_qty,
            now=now,
        )

        if live_qty > 0:
            journal_db.reduce_position_quantity(conn, position["id"], live_qty, now=now)
            positions_partially_closed.append(position["id"])
        else:
            journal_db.close_position(conn, position["id"], exit_price=exit_price, now=now)
            positions_closed.append(position["id"])

    return SyncResult(
        symbol=symbol,
        positions_closed=positions_closed,
        positions_partially_closed=positions_partially_closed,
        discrepancies=discrepancies,
    )


# ---------------------------------------------------------------------------
# manage_stop_loss
# ---------------------------------------------------------------------------


def manage_stop_loss(conn, position_id: int, *, now: float | None = None) -> str | None:
    """Replaces Stage 11's apply_trailing_stop with a state machine keyed
    off how many take-profit levels have actually filled (journal_db's
    fills table — not a raw R-multiple threshold): after the FIRST TP
    fills, moves the stop to breakeven ("перевести в безубыток"); after a
    SECOND TP fills, switches to a real trailing stop for the remainder
    ("использовать trailing stop") that ratchets toward price by
    config.EXECUTION_TRAILING_STOP_R_MULTIPLE and never gives ground back.
    Both stages use the same cancel-old-SL/place-new-SL mechanism, so one
    function — splitting it in two would just duplicate that mechanism.
    Returns "BREAKEVEN", "TRAILING", or None (nothing to do yet, or the
    candidate stop wasn't actually better than the current one).
    """
    position = journal_db.get_position(conn, position_id)
    if position is None or position["status"] != "OPEN":
        return None

    entry_price = position["entry_price"]
    original_stop = _original_stop_for_position(conn, position)
    if original_stop is None:
        return None
    risk = abs(entry_price - original_stop)
    if risk <= 0:
        return None

    tp_fills = sum(1 for f in journal_db.get_position_fills(conn, position_id) if f["fill_type"] == "TAKE_PROFIT")
    if tp_fills == 0:
        return None

    is_long = position["side"] == "LONG"

    if tp_fills == 1:
        candidate_stop = entry_price
        stage = "BREAKEVEN"
    else:
        try:
            current_price = bingx_client.get_price(config.to_bingx_symbol(position["symbol"]))
        except bingx_client.BingXError:
            return None
        current_price_d = Decimal(str(current_price))
        trail_distance = risk * Decimal(str(config.EXECUTION_TRAILING_STOP_R_MULTIPLE))
        candidate_stop = (current_price_d - trail_distance) if is_long else (current_price_d + trail_distance)
        stage = "TRAILING"

    current_stop = position.get("stop_loss")
    already_better_or_equal = current_stop is not None and (
        (is_long and candidate_stop <= current_stop) or (not is_long and candidate_stop >= current_stop)
    )
    if already_better_or_equal:
        return None

    now = now if now is not None else time.time()
    sl_order = None
    if position.get("trade_plan_id") is not None:
        for order in journal_db.get_pending_real_orders(conn, position["symbol"]):
            if order.get("trade_plan_id") == position["trade_plan_id"] and order.get("stop_loss") is not None:
                sl_order = order
                break

    try:
        if sl_order is not None and sl_order.get("exchange_order_id") is not None:
            bingx_private_client.cancel_order(position["symbol"], sl_order["exchange_order_id"])
        data = bingx_private_client.place_order(
            position["symbol"],
            _closing_bingx_side(position["side"]),
            position["side"],
            "STOP_MARKET",
            str(position["quantity"]),
            stop_price=str(candidate_stop),
            client_order_id=f"neurotrade-stop-{position_id}-{int(now)}",
        )
        new_exchange_id = _extract_order_id(data)
    except bingx_private_client.DryRunNotSent:
        new_exchange_id = None
    except bingx_client.BingXError:
        return None

    if sl_order is not None:
        journal_db.update_order_status(conn, "real_orders", sl_order["id"], "CANCELLED", now=now)
    journal_db.insert_real_order(
        conn,
        symbol=position["symbol"],
        side=position["side"],
        order_type="TRIGGER",
        quantity=position["quantity"],
        trigger_price=candidate_stop,
        stop_loss=candidate_stop,
        trade_plan_id=position.get("trade_plan_id"),
        leverage=position.get("leverage"),
        margin_mode=position.get("margin_mode"),
        status="OPEN",
        now=now,
        exchange_order_id=new_exchange_id,
    )
    journal_db.update_position_stop_loss(conn, position_id, candidate_stop, now=now)
    return stage


# ---------------------------------------------------------------------------
# check_regime_close / check_data_quality_close / check_time_based_close
# ---------------------------------------------------------------------------


def check_regime_close(
    conn, position_id: int, *, current_regime: market_regime.RegimeResult | None = None, now: float | None = None
) -> bool:
    """"Закрывать при изменении режима рынка" — a directional-invalidation
    table (see module docstring), not "close on any regime change." If
    `current_regime` isn't injected, computed live the same way
    report_builder.build_ai_context does: market_data_engine.collect_snapshot
    -> feature_engine.compute_features -> market_regime.classify_regime.
    """
    now = now if now is not None else time.time()
    position = journal_db.get_position(conn, position_id)
    if position is None or position["status"] != "OPEN":
        return False

    if current_regime is None:
        current_regime = _live_regime(position["symbol"])
    if current_regime is None:
        return False  # couldn't determine -> don't act

    invalidating = _LONG_INVALIDATING_REGIMES if position["side"] == "LONG" else _SHORT_INVALIDATING_REGIMES
    if current_regime.regime not in invalidating:
        return False

    return _market_close_position(conn, position, fill_type="REGIME_CHANGE", now=now)


def check_data_quality_close(
    conn, position_id: int, *, current_data_quality: str | None = None, now: float | None = None
) -> bool:
    """"Закрывать при критической ошибке данных" — data_quality == "NO_TRADE"
    (the same threshold market_data_engine/market_regime already use to
    mean "can't trust this read at all"). If not injected, fetched live via
    market_data_engine.collect_snapshot.
    """
    now = now if now is not None else time.time()
    position = journal_db.get_position(conn, position_id)
    if position is None or position["status"] != "OPEN":
        return False

    if current_data_quality is None:
        current_data_quality = _live_data_quality(position["symbol"])
    if current_data_quality != "NO_TRADE":
        return False

    return _market_close_position(conn, position, fill_type="DATA_QUALITY", now=now)


def check_time_based_close(conn, position_id: int, *, now: float | None = None) -> bool:
    """Same config.PAPER_TRADING_MAX_HOLD_SECONDS[mode] policy as
    paper_trading.py — delegates the actual cancel+close mechanics to
    _market_close_position, shared with the regime/data-quality checks.
    """
    now = now if now is not None else time.time()
    position = journal_db.get_position(conn, position_id)
    if position is None or position["status"] != "OPEN":
        return False

    trade_plan = journal_db.get_trade_plan(conn, position["trade_plan_id"]) if position.get("trade_plan_id") else None
    mode = trade_plan.get("mode") if trade_plan else None
    max_hold = config.PAPER_TRADING_MAX_HOLD_SECONDS.get(mode)
    if max_hold is None or now - position["opened_at"] < max_hold:
        return False

    return _market_close_position(conn, position, fill_type="TIMEOUT", now=now)


# ---------------------------------------------------------------------------
# reconcile_state / recover_after_restart
# ---------------------------------------------------------------------------


def reconcile_state(conn, symbol: str) -> ReconciliationReport:
    """Report-only comparison of journal_db's local REAL state against what
    BingX actually reports — no automatic correction. An automatic 'fix'
    carries its own risk of doing the wrong thing with real funds; a report
    the user/system can act on is the safer choice for this stage. Covers
    both directions: a local OPEN position/pending order the exchange no
    longer shows, and a live exchange order this app has no local record of
    (placed outside this app, or lost in a crash).
    """
    try:
        live_positions = bingx_private_client.get_positions(symbol) or []
        live_open_orders = bingx_private_client.get_open_orders(symbol) or []
    except bingx_private_client.DryRunNotSent:
        return ReconciliationReport(
            symbol=symbol, local_open_not_on_exchange=[], exchange_open_not_local=[],
            errors=["DRY_RUN — no live state to reconcile against"],
        )
    except bingx_client.BingXError as exc:
        return ReconciliationReport(
            symbol=symbol, local_open_not_on_exchange=[], exchange_open_not_local=[],
            errors=[f"failed to fetch live state: {exc}"],
        )

    live_positions = live_positions if isinstance(live_positions, list) else []
    live_open_orders = live_open_orders if isinstance(live_open_orders, list) else []
    live_order_ids = {str(o.get("orderId")) for o in live_open_orders if o.get("orderId") is not None}

    local_open_not_on_exchange = [
        position["id"]
        for position in journal_db.get_open_positions(conn, symbol, source="REAL")
        if not _live_position_open(live_positions, symbol, position["side"])
    ]

    local_order_ids = {
        order["exchange_order_id"]
        for order in journal_db.get_pending_real_orders(conn, symbol)
        if order.get("exchange_order_id") is not None
    }
    exchange_open_not_local = sorted(live_order_ids - local_order_ids)

    return ReconciliationReport(
        symbol=symbol,
        local_open_not_on_exchange=local_open_not_on_exchange,
        exchange_open_not_local=exchange_open_not_local,
        errors=[],
    )


def recover_after_restart(conn, symbol: str) -> ReconciliationReport:
    """'Восстановление после перезапуска' = reconcile_state() invoked at
    startup — a specialization of the same reconciliation, not separate
    logic."""
    return reconcile_state(conn, symbol)
