"""Paper Trading Engine — a live, tick-driven simulation of order fills and
position exits against real BingX book-ticker prices, using the user's own
risk_manager.RiskSettings (maker/taker fees, slippage%) rather than a second,
disconnected fee assumption.

A virtual order is opened FROM an existing `trade_plans` row already in
journal_db (`open_virtual_order`) — never from raw ad-hoc arguments, never
tied to a live AI call. `process_tick()` is the resumable state machine:
all state lives in journal_db (paper_orders/positions/fills/trade_outcomes),
nothing is cached in memory between calls, so it can be invoked any number
of times from anywhere (a script, a future scheduler, a dashboard refresh
hook) with the same result as long as it's fed honest prices. This module
does NOT add that continuous loop itself — see the Stage 8 plan.

Simplifications, stated plainly rather than silently absorbed:
- One bid/ask pair per tick is a POINT, not an OHLC range — AMBIGUOUS
  (both SL and a TP touched "in the same candle", as outcome_simulator.py
  has to handle) cannot occur here: SL and TP always sit on opposite sides
  of entry, so a single price cannot satisfy both.
- Spread IS modeled: entries/exits always cross the real bid/ask from
  bingx_client.get_book_ticker (buy at ask, sell at bid) — there is no
  separate "spread%" parameter.
- Partial fills are capped by the real quantity available at the best
  bid/ask (from the same book-ticker call) — not a full orderbook-depth
  simulation, but grounded in real (if shallow) liquidity data rather than
  invented.
- MFE/MAE are tick-sampled running maxima (see
  journal_db.update_trade_outcome_running_extremes), bounded by how often
  process_tick is actually called — NOT OHLC-exact like outcome_simulator.
- Take-profit fills execute at the limit price exactly (no slippage);
  stop-loss and timeout fills are market-ish (slippage applies). Every
  exit is taker fee; entry is always taker too (this engine always crosses
  the spread to fill, so `assume_maker_entry=True` would be a false
  assumption per risk_manager's own contract).
- Trailing stop: exactly one policy — move to breakeven (the position's own
  filled entry price, no fee adjustment) once price has moved
  `config.PAPER_TRADING_BREAKEVEN_TRIGGER_R` in favor, measured against the
  ORIGINAL stop distance (trade_plans.stop_loss_calc / stop_loss), never the
  already-trailed positions.stop_loss.
- `trade_outcomes.slippage_usdt` is left NULL — its economic effect is
  already fully embedded in realized_pnl_usdt and each fill's stored price;
  isolating it would need a "pre-slippage reference price" column on
  `fills` that doesn't exist and wasn't requested.
- No margin/liquidation modeling — positions never reach `LIQUIDATED`.
- `positions.stop_loss` is never NULL for a position this engine creates
  (`open_virtual_order` requires a resolved stop before it will ever open
  an order) — `_check_stop_loss_exit` still guards defensively for any
  future manually-created position outside this engine.
"""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass
from decimal import Decimal
from typing import Literal

import bingx_client
import config
import journal_db
import risk_settings_store
from outcome_simulator import SimulatedOutcome
from risk_manager import RiskSettings, compute_liquidation_price, compute_maintenance_margin_usdt

_EXIT_REASONS = ("SL", "TP1", "TP2", "TP3", "TIMEOUT", "AMBIGUOUS", "LIQUIDATION")


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OpenOrderResult:
    order_id: int | None
    status: Literal["CREATED", "ALREADY_EXISTS", "SKIPPED_NOT_ACTIONABLE"]
    reason: str | None = None


@dataclass(frozen=True)
class TickResult:
    symbol: str
    now: float
    bid: Decimal | None
    ask: Decimal | None
    orders_filled: list[int]
    orders_partially_filled: list[int]
    orders_expired: list[int]
    orders_force_closed_partial: list[int]
    positions_opened: list[int]
    positions_closed: list[int]
    stops_trailed: list[int]
    fills_created: list[int]
    errors: list[str]


@dataclass(frozen=True)
class BucketStats:
    total: int
    evaluated: int
    wins: int
    win_rate: float | None
    expectancy_r: float | None
    median_r: float | None
    profit_factor: float | None
    profit_factor_undefined: bool
    max_drawdown_r: float | None
    avg_mfe_r: float | None
    avg_mae_r: float | None
    avg_duration_seconds: float | None
    exit_reason_counts: dict[str, int]
    low_sample: bool


@dataclass(frozen=True)
class PaperTradingStatistics:
    mode: str
    window_start: float
    window_end: float
    by_model: dict[str, BucketStats]
    by_strategy: dict[tuple[str, str | None], BucketStats]
    by_regime: dict[str, BucketStats]


@dataclass(frozen=True)
class StatsRefreshResult:
    model_rows_inserted: int
    strategy_rows_inserted: int


# ---------------------------------------------------------------------------
# open_virtual_order
# ---------------------------------------------------------------------------


def _original_stop(trade_plan: dict) -> Decimal:
    stop = trade_plan.get("stop_loss_calc")
    if stop is not None:
        return stop
    return Decimal(str(trade_plan["stop_loss"]))


def _actionable_trade_plan(trade_plan: dict | None) -> tuple[bool, str | None]:
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
    return True, None


def _normalize_order_type(entry_type: str | None) -> Literal["MARKET", "LIMIT", "TRIGGER"]:
    t = (entry_type or "").upper()
    if "MARKET" in t:
        return "MARKET"
    if "BREAKOUT" in t:
        return "TRIGGER"
    return "LIMIT"


def open_virtual_order(conn, trade_plan_id: int, *, now: float | None = None) -> OpenOrderResult:
    """Creates a PENDING paper_orders row FROM an existing trade_plans row.
    Never touches price/book-ticker — filling happens on a later
    process_tick() call. Idempotent per trade_plan_id: calling twice for the
    same trade_plan_id returns the existing order, never a duplicate.

    Deliberately NOT gated on trade_permission text (WAITING_TRIGGER /
    PRICE_OUTSIDE_ENTRY_ZONE both still carry a fully risk-manager-validated
    plan — they just mean price isn't right AT SIGNAL TIME, which is exactly
    what a paper order waits out).
    """
    now = now if now is not None else time.time()

    existing = journal_db.get_paper_order_by_trade_plan_id(conn, trade_plan_id)
    if existing is not None:
        return OpenOrderResult(order_id=existing["id"], status="ALREADY_EXISTS")

    trade_plan = journal_db.get_trade_plan(conn, trade_plan_id)
    ok, reason = _actionable_trade_plan(trade_plan)
    if not ok:
        return OpenOrderResult(order_id=None, status="SKIPPED_NOT_ACTIONABLE", reason=reason)

    side = trade_plan["overall_signal"]
    order_type = _normalize_order_type(trade_plan.get("entry_type"))
    stop_loss = _original_stop(trade_plan)
    quantity = trade_plan["position_size_coin_rounded"]
    take_profits = trade_plan.get("take_profits") or []
    take_profit_single = Decimal(str(take_profits[0][1])) if take_profits else None

    order_id = journal_db.insert_paper_order(
        conn,
        symbol=trade_plan["symbol"],
        side=side,
        order_type=order_type,
        quantity=quantity,
        entry_from=trade_plan["entry_from"],
        entry_to=trade_plan["entry_to"],
        stop_loss=stop_loss,
        take_profit=take_profit_single,
        take_profits=[tuple(tp) for tp in take_profits],
        leverage=trade_plan.get("leverage"),
        margin_mode=trade_plan.get("margin_mode"),
        trade_plan_id=trade_plan_id,
        status="PENDING",
        now=now,
    )
    return OpenOrderResult(order_id=order_id, status="CREATED")


# ---------------------------------------------------------------------------
# process_tick
# ---------------------------------------------------------------------------


def _resolve_settings(settings: RiskSettings | None) -> RiskSettings:
    return settings if settings is not None else risk_settings_store.load()


def process_tick(
    conn,
    symbol: str,
    *,
    now: float | None = None,
    bid: float | None = None,
    ask: float | None = None,
    bid_qty: float | None = None,
    ask_qty: float | None = None,
    settings: RiskSettings | None = None,
) -> TickResult:
    """One discrete-time step for one symbol. Order within one call
    (deterministic, id-ascending): first advance every PENDING/
    PARTIALLY_FILLED paper order (may open/grow positions), then advance
    every OPEN position (checkpoints/extremes -> TP farthest-first -> SL ->
    trailing-stop -> time-based close).
    """
    now = now if now is not None else time.time()

    pending_orders = journal_db.get_pending_paper_orders(conn, symbol)
    open_positions = journal_db.get_open_positions(conn, symbol)

    r: dict = dict(
        symbol=symbol, now=now, bid=None, ask=None,
        orders_filled=[], orders_partially_filled=[], orders_expired=[],
        orders_force_closed_partial=[], positions_opened=[], positions_closed=[],
        stops_trailed=[], fills_created=[], errors=[],
    )

    if not pending_orders and not open_positions:
        return TickResult(**r)

    need_live = bid is None or ask is None or bid_qty is None or ask_qty is None
    if need_live:
        ticker = bingx_client.get_book_ticker(config.to_bingx_symbol(symbol))
        if ticker is None:
            r["errors"].append(f"book ticker unavailable for {symbol}")
            return TickResult(**r)
        bid, ask, bid_qty, ask_qty = ticker["bid_price"], ticker["ask_price"], ticker["bid_qty"], ticker["ask_qty"]

    bid_d, ask_d = Decimal(str(bid)), Decimal(str(ask))
    bid_qty_d, ask_qty_d = Decimal(str(bid_qty)), Decimal(str(ask_qty))
    r["bid"], r["ask"] = bid_d, ask_d

    settings = _resolve_settings(settings)

    for order in pending_orders:
        _advance_pending_order(conn, order, now=now, bid=bid_d, ask=ask_d, bid_qty=bid_qty_d, ask_qty=ask_qty_d, settings=settings, r=r)

    for position in journal_db.get_open_positions(conn, symbol):
        _advance_open_position(conn, position, now=now, bid=bid_d, ask=ask_d, settings=settings, r=r)

    return TickResult(**r)


def _advance_pending_order(conn, order: dict, *, now, bid, ask, bid_qty, ask_qty, settings, r: dict) -> None:
    fill_status = _try_fill_order(conn, order, now=now, bid=bid, ask=ask, bid_qty=bid_qty, ask_qty=ask_qty, settings=settings, r=r)
    if fill_status in ("FILLED", "PARTIALLY_FILLED"):
        return
    trade_plan = journal_db.get_trade_plan(conn, order["trade_plan_id"]) if order.get("trade_plan_id") else None
    _expire_order_if_due(conn, order, trade_plan, now=now, r=r)


def _try_fill_order(
    conn, order: dict, *, now, bid, ask, bid_qty, ask_qty, settings, r: dict
) -> Literal["FILLED", "PARTIALLY_FILLED", "NO_LIQUIDITY", "NOT_TOUCHED"]:
    side = order["side"]
    order_type = order["order_type"]

    if order_type == "MARKET":
        touched = True
    elif order["entry_from"] is None:
        touched = False
    elif order_type == "TRIGGER":
        # A breakout/breakdown trigger fires the OPPOSITE way from a limit
        # zone: price must clear the far edge, not retrace into it. LONG
        # needs ask to break ABOVE the zone's high; SHORT needs bid to break
        # BELOW the zone's low. Using the LIMIT comparator here (as an
        # earlier version of this function did) would fill a LONG breakout
        # order on a pullback DOWN into the zone — backwards for what a
        # breakout entry means.
        lo, hi = min(order["entry_from"], order["entry_to"]), max(order["entry_from"], order["entry_to"])
        touched = (ask >= Decimal(str(hi))) if side == "LONG" else (bid <= Decimal(str(lo)))
    else:  # LIMIT
        lo, hi = min(order["entry_from"], order["entry_to"]), max(order["entry_from"], order["entry_to"])
        touched = (ask <= Decimal(str(hi))) if side == "LONG" else (bid >= Decimal(str(lo)))

    if not touched:
        return "NOT_TOUCHED"

    remaining = order["quantity"] - order["filled_quantity"]
    if remaining <= 0:
        return "NOT_TOUCHED"

    slippage_rate = settings.slippage_percent / Decimal("100")
    if side == "LONG":
        fill_price = ask * (1 + slippage_rate)
        available_qty = ask_qty
    else:
        fill_price = bid * (1 - slippage_rate)
        available_qty = bid_qty

    if available_qty <= 0:
        return "NO_LIQUIDITY"

    fill_qty = min(remaining, available_qty)
    new_filled = order["filled_quantity"] + fill_qty
    is_full = new_filled >= order["quantity"]

    fee_rate = settings.taker_fee_percent / Decimal("100")
    fee_usdt = fill_price * fill_qty * fee_rate

    position_id, created_new = _get_or_grow_position_for_order(conn, order, side=side, fill_price=fill_price, fill_qty=fill_qty, now=now)
    fill_id = journal_db.insert_fill(
        conn, position_id=position_id, symbol=order["symbol"], side=side, fill_type="ENTRY",
        price=fill_price, quantity=fill_qty, fee_usdt=fee_usdt, paper_order_id=order["id"], now=now,
    )
    r["fills_created"].append(fill_id)
    if created_new:
        r["positions_opened"].append(position_id)

    new_status = "FILLED" if is_full else "PARTIALLY_FILLED"
    journal_db.update_order_status(conn, "paper_orders", order["id"], new_status, filled_quantity=new_filled, now=now)
    (r["orders_filled"] if is_full else r["orders_partially_filled"]).append(order["id"])
    return new_status


def _get_or_grow_position_for_order(conn, order: dict, *, side: str, fill_price: Decimal, fill_qty: Decimal, now: float) -> tuple[int, bool]:
    existing = journal_db.get_position_by_paper_order_id(conn, order["id"])
    if existing is not None:
        journal_db.update_position_add_fill(conn, existing["id"], additional_quantity=fill_qty, fill_price=fill_price, now=now)
        return existing["id"], False

    leverage = order.get("leverage")
    initial_margin_usdt = maintenance_margin_usdt = liquidation_price = None
    if leverage:
        mmr = Decimal(str(config.PAPER_TRADING_MAINTENANCE_MARGIN_RATE))
        notional = fill_price * fill_qty
        try:
            liquidation_price = compute_liquidation_price(fill_price, leverage, side, mmr)
            initial_margin_usdt = notional / leverage
            maintenance_margin_usdt = compute_maintenance_margin_usdt(notional, mmr)
        except ValueError:
            # leverage too extreme relative to maintenance_margin_rate for the
            # formula to represent a valid position (see compute_liquidation_price's
            # docstring) — best-effort: skip margin modeling for this position
            # rather than failing the whole tick loop over an edge case.
            pass

    position_id = journal_db.insert_position(
        conn, symbol=order["symbol"], side=side, source="PAPER",
        entry_price=fill_price, quantity=fill_qty,
        leverage=leverage, margin_mode=order.get("margin_mode"),
        stop_loss=order["stop_loss"], take_profit=order["take_profit"],
        trade_plan_id=order.get("trade_plan_id"), paper_order_id=order["id"], status="OPEN", now=now,
        initial_margin_usdt=initial_margin_usdt, maintenance_margin_usdt=maintenance_margin_usdt,
        liquidation_price=liquidation_price,
    )
    trade_plan_id = order.get("trade_plan_id")
    if trade_plan_id is not None:
        trade_plan = journal_db.get_trade_plan(conn, trade_plan_id)
        journal_db.insert_trade_outcome_pending(
            conn, trade_plan_id=trade_plan_id, symbol=order["symbol"], mode=trade_plan["mode"],
            entry_price=float(fill_price), now=now,
        )
    return position_id, True


def _expire_order_if_due(conn, order: dict, trade_plan: dict | None, *, now: float, r: dict) -> None:
    if trade_plan is None:
        return
    anchor = trade_plan.get("formed_at")
    if anchor is None:
        anchor = trade_plan.get("timestamp")
    valid_for_minutes = trade_plan.get("valid_for_minutes")
    if anchor is None or valid_for_minutes is None:
        return
    if now < anchor + valid_for_minutes * 60:
        return

    if order["filled_quantity"] <= 0:
        journal_db.update_order_status(conn, "paper_orders", order["id"], "EXPIRED", now=now)
        r["orders_expired"].append(order["id"])
    else:
        # Already-achieved quantity is at risk in a real position — bookkeep
        # the order as FILLED-as-is, no position-side action needed (the
        # position already reflects exactly what was filled).
        journal_db.update_order_status(conn, "paper_orders", order["id"], "FILLED", now=now)
        r["orders_force_closed_partial"].append(order["id"])


# ---------------------------------------------------------------------------
# Advancing open positions
# ---------------------------------------------------------------------------


def _original_stop_for_position(conn, position: dict) -> Decimal:
    trade_plan_id = position.get("trade_plan_id")
    if trade_plan_id is not None:
        trade_plan = journal_db.get_trade_plan(conn, trade_plan_id)
        if trade_plan is not None:
            return _original_stop(trade_plan)
    return position["stop_loss"]


def _remaining_quantity(conn, position: dict) -> Decimal:
    fills = journal_db.get_position_fills(conn, position["id"])
    closed_qty = sum((f["quantity"] for f in fills if f["fill_type"] != "ENTRY"), start=Decimal("0"))
    return position["quantity"] - closed_qty


def _tp_already_filled(conn, position_id: int, label: str) -> bool:
    fills = journal_db.get_position_fills(conn, position_id)
    return any(f["fill_type"] == "TAKE_PROFIT" and f["label"] == label for f in fills)


def _entry_fee_usdt_for_position(conn, position: dict) -> Decimal:
    fills = journal_db.get_position_fills(conn, position["id"])
    total = Decimal("0")
    for f in fills:
        if f["fill_type"] == "ENTRY" and f["fee_usdt"] is not None:
            total += f["fee_usdt"]
    return total


def _advance_open_position(conn, position: dict, *, now, bid, ask, settings, r: dict) -> None:
    trade_plan_id = position.get("trade_plan_id")
    trade_plan = journal_db.get_trade_plan(conn, trade_plan_id) if trade_plan_id else None
    trade_outcome = journal_db.get_trade_outcome(conn, trade_plan_id) if trade_plan_id else None

    if trade_outcome is not None:
        _update_checkpoints_and_extremes(conn, position, trade_outcome, now=now, bid=bid, ask=ask)
    _update_unrealized_pnl(conn, position, bid=bid, ask=ask)

    # Liquidation is checked first: it's the exchange's own forced action,
    # which in reality preempts/cancels resting SL/TP orders — a stop loss
    # is just a reserved order, not a guarantee it fires before liquidation
    # at high leverage.
    if _check_liquidation_exit(conn, position, now=now, bid=bid, ask=ask, r=r):
        return
    if _check_take_profit_exits(conn, position, trade_plan, now=now, bid=bid, ask=ask, settings=settings, r=r):
        return
    if _check_stop_loss_exit(conn, position, now=now, bid=bid, ask=ask, settings=settings, r=r):
        return
    _apply_trailing_stop(conn, position, now=now, bid=bid, ask=ask, r=r)
    _check_time_based_close(conn, position, trade_plan, now=now, bid=bid, ask=ask, settings=settings, r=r)


def _update_unrealized_pnl(conn, position: dict, *, bid: Decimal, ask: Decimal) -> None:
    """Gross mark-to-market PnL (no fee deduction, matching what an
    exchange UI typically shows) — recorded every tick for any OPEN
    position, independent of whether it has a trade_plan_id/trade_outcome.
    """
    is_long = position["side"] == "LONG"
    mark_price = bid if is_long else ask
    gross = (mark_price - position["entry_price"]) * position["quantity"] if is_long else (position["entry_price"] - mark_price) * position["quantity"]
    journal_db.update_position_unrealized_pnl(conn, position["id"], gross)


def _check_liquidation_exit(conn, position: dict, *, now, bid, ask, r: dict) -> bool:
    """Forced full close at the position's precomputed liquidation_price
    (see risk_manager.compute_liquidation_price) — a position with no
    leverage on record has no liquidation_price and this is a no-op.
    Losing the ENTIRE margin allocated to whatever quantity is still open
    (not a price-delta P&L like a normal exit) is the defining behavior of
    isolated-margin liquidation: the loss is capped at the margin put up,
    prorated to the remaining fraction if earlier take-profits already
    reduced the position (paper positions never physically shrink
    `quantity` on a partial close — see _remaining_quantity).
    """
    liq_price = position.get("liquidation_price")
    if liq_price is None:
        return False
    is_long = position["side"] == "LONG"
    triggered = (bid <= liq_price) if is_long else (ask >= liq_price)
    if not triggered:
        return False

    # This engine's tick model is a single bid/ask POINT, not an OHLC range,
    # so true same-candle ambiguity (outcome_simulator.py's AMBIGUOUS) can't
    # happen between TP and liquidation/SL (opposite sides of entry) — but
    # IS possible between liquidation and stop_loss if one tick's price move
    # is large enough to cross both. Liquidation always wins (it's the
    # exchange's forced action); intrabar_ambiguous records when that
    # actually mattered, so downstream stats don't read this tick as a
    # cleaner signal than a single point-sample can really support.
    stop_loss = position.get("stop_loss")
    sl_also_triggered = stop_loss is not None and ((bid <= stop_loss) if is_long else (ask >= stop_loss))

    remaining = _remaining_quantity(conn, position)
    if remaining > 0:
        total_qty = position["quantity"]
        initial_margin = position.get("initial_margin_usdt") or Decimal("0")
        remaining_fraction = (remaining / total_qty) if total_qty else Decimal("0")
        pnl = -(initial_margin * remaining_fraction)
        fill_id = journal_db.insert_fill(
            conn, position_id=position["id"], symbol=position["symbol"], side=position["side"],
            fill_type="LIQUIDATION", price=liq_price, quantity=remaining, realized_pnl_usdt=pnl, now=now,
            intrabar_ambiguous=sl_also_triggered, resolution_policy="LIQUIDATION_FIRST",
        )
        r["fills_created"].append(fill_id)

    return _finalize_if_closed(conn, position, now=now, r=r, status="LIQUIDATED")


def _update_checkpoints_and_extremes(conn, position: dict, trade_outcome: dict, *, now, bid, ask) -> None:
    opened_at = position["opened_at"]
    elapsed = now - opened_at
    mid = (bid + ask) / 2
    for minute in (1, 5, 15, 30):
        if elapsed >= minute * 60:
            journal_db.update_trade_outcome_price_checkpoint(conn, trade_outcome["id"], minute=minute, price=float(mid))

    is_long = position["side"] == "LONG"
    exit_price_now = bid if is_long else ask
    original_stop = _original_stop_for_position(conn, position)
    risk = abs(position["entry_price"] - original_stop)
    if risk <= 0:
        return
    if is_long:
        favorable_r = float((exit_price_now - position["entry_price"]) / risk)
        adverse_r = float((position["entry_price"] - exit_price_now) / risk)
    else:
        favorable_r = float((position["entry_price"] - exit_price_now) / risk)
        adverse_r = float((exit_price_now - position["entry_price"]) / risk)
    journal_db.update_trade_outcome_running_extremes(conn, trade_outcome["id"], favorable_r=favorable_r, adverse_r=adverse_r)


def _check_take_profit_exits(conn, position: dict, trade_plan: dict | None, *, now, bid, ask, settings, r: dict) -> bool:
    """Margin-model policy (deliberate choice, not an oversight): a partial
    TP close does NOT recompute `positions.liquidation_price` — it stays
    fixed at the value computed from the original entry. Only the LOSS at
    liquidation time is prorated (initial_margin_usdt * remaining_fraction,
    see _check_liquidation_exit) to reflect that isolated margin frees up
    proportionally to closed size at a fixed leverage. This is simple and
    deterministic, at the cost of not modeling any actual margin/leverage
    adjustment a trader might make after a partial close. Contrast with
    `update_position_add_fill`, which DOES recompute liquidation_price —
    there the average entry price itself changes, so the old value would be
    outright wrong, not just a simplification.
    """
    if trade_plan is not None:
        take_profits = trade_plan.get("take_profits") or []
        if take_profits:
            is_long = position["side"] == "LONG"
            sorted_tps = sorted(take_profits, key=lambda tp: tp[1], reverse=is_long)
            remaining = _remaining_quantity(conn, position)
            total_qty = position["quantity"]
            fee_rate = settings.taker_fee_percent / Decimal("100")

            for label, price, close_percent in sorted_tps:
                if remaining <= 0:
                    break
                price_d = Decimal(str(price))
                crossed = (bid >= price_d) if is_long else (ask <= price_d)
                if not crossed or _tp_already_filled(conn, position["id"], label):
                    continue
                portion = min(total_qty * Decimal(str(close_percent)) / Decimal("100"), remaining)
                if portion <= 0:
                    continue
                fee_usdt = price_d * portion * fee_rate
                _close_out_position(conn, position, fill_type="TAKE_PROFIT", exit_price=price_d, quantity=portion, fee_usdt=fee_usdt, now=now, r=r, label=label)
                remaining -= portion

    return _finalize_if_closed(conn, position, now=now, r=r)


def _check_stop_loss_exit(conn, position: dict, *, now, bid, ask, settings, r: dict) -> bool:
    stop_loss = position.get("stop_loss")
    if stop_loss is None:
        return False
    is_long = position["side"] == "LONG"
    triggered = (bid <= stop_loss) if is_long else (ask >= stop_loss)
    if not triggered:
        return False

    remaining = _remaining_quantity(conn, position)
    if remaining > 0:
        slippage_rate = settings.slippage_percent / Decimal("100")
        exit_price = bid * (1 - slippage_rate) if is_long else ask * (1 + slippage_rate)
        fee_rate = settings.taker_fee_percent / Decimal("100")
        fee_usdt = exit_price * remaining * fee_rate
        _close_out_position(conn, position, fill_type="STOP_LOSS", exit_price=exit_price, quantity=remaining, fee_usdt=fee_usdt, now=now, r=r)

    return _finalize_if_closed(conn, position, now=now, r=r)


def _apply_trailing_stop(conn, position: dict, *, now, bid, ask, r: dict) -> bool:
    if position.get("trade_plan_id") is None:
        return False
    entry_price = position["entry_price"]
    original_stop = _original_stop_for_position(conn, position)
    risk = abs(entry_price - original_stop)
    if risk <= 0:
        return False

    is_long = position["side"] == "LONG"
    current_price = bid if is_long else ask
    current_r = (current_price - entry_price) / risk if is_long else (entry_price - current_price) / risk
    if current_r < Decimal(str(config.PAPER_TRADING_BREAKEVEN_TRIGGER_R)):
        return False

    current_stop = position.get("stop_loss")
    already_at_or_better = current_stop is not None and (
        (is_long and current_stop >= entry_price) or (not is_long and current_stop <= entry_price)
    )
    if already_at_or_better:
        return False

    journal_db.update_position_stop_loss(conn, position["id"], entry_price, now=now)
    r["stops_trailed"].append(position["id"])
    return True


def _check_time_based_close(conn, position: dict, trade_plan: dict | None, *, now, bid, ask, settings, r: dict) -> bool:
    if trade_plan is None:
        return False
    max_hold = config.PAPER_TRADING_MAX_HOLD_SECONDS.get(trade_plan.get("mode"))
    if max_hold is None or now - position["opened_at"] < max_hold:
        return False

    remaining = _remaining_quantity(conn, position)
    if remaining > 0:
        is_long = position["side"] == "LONG"
        slippage_rate = settings.slippage_percent / Decimal("100")
        exit_price = bid * (1 - slippage_rate) if is_long else ask * (1 + slippage_rate)
        fee_rate = settings.taker_fee_percent / Decimal("100")
        fee_usdt = exit_price * remaining * fee_rate
        _close_out_position(conn, position, fill_type="TIMEOUT", exit_price=exit_price, quantity=remaining, fee_usdt=fee_usdt, now=now, r=r)

    return _finalize_if_closed(conn, position, now=now, r=r)


def _close_out_position(conn, position: dict, *, fill_type: str, exit_price: Decimal, quantity: Decimal, fee_usdt: Decimal, now: float, r: dict, label: str | None = None) -> int:
    entry_fee_usdt = _entry_fee_usdt_for_position(conn, position)
    total_qty = position["quantity"]
    entry_fee_share = (entry_fee_usdt * quantity / total_qty) if total_qty else Decimal("0")
    is_long = position["side"] == "LONG"
    gross = (exit_price - position["entry_price"]) * quantity if is_long else (position["entry_price"] - exit_price) * quantity
    pnl = gross - fee_usdt - entry_fee_share

    fill_id = journal_db.insert_fill(
        conn, position_id=position["id"], symbol=position["symbol"], side=position["side"],
        fill_type=fill_type, label=label, price=exit_price, quantity=quantity,
        fee_usdt=fee_usdt, realized_pnl_usdt=pnl, now=now,
    )
    r["fills_created"].append(fill_id)
    return fill_id


def _map_fill_to_exit_reason(fill: dict) -> str:
    if fill["fill_type"] == "STOP_LOSS":
        return "SL"
    if fill["fill_type"] == "TIMEOUT":
        return "TIMEOUT"
    if fill["fill_type"] == "LIQUIDATION":
        return "LIQUIDATION"
    if fill["fill_type"] == "TAKE_PROFIT":
        return fill["label"]
    return "TIMEOUT"


def _finalize_if_closed(conn, position: dict, *, now: float, r: dict, status: Literal["CLOSED", "LIQUIDATED"] = "CLOSED") -> bool:
    if _remaining_quantity(conn, position) > 0:
        return False

    fills = journal_db.get_position_fills(conn, position["id"])
    closing_fills = [f for f in fills if f["fill_type"] != "ENTRY"]
    if not closing_fills:
        return False

    last_fill = closing_fills[-1]
    total_fees = sum((f["fee_usdt"] or Decimal("0")) for f in fills)
    total_pnl = sum((f["realized_pnl_usdt"] or Decimal("0")) for f in closing_fills)

    journal_db.close_position(
        conn, position["id"], exit_price=last_fill["price"], realized_pnl_usdt=total_pnl,
        fees_paid_usdt=total_fees, now=now, status=status,
    )
    r["positions_closed"].append(position["id"])

    trade_plan_id = position.get("trade_plan_id")
    if trade_plan_id is not None:
        trade_outcome = journal_db.get_trade_outcome(conn, trade_plan_id)
        trade_plan = journal_db.get_trade_plan(conn, trade_plan_id)
        if trade_outcome is not None and trade_plan is not None:
            original_stop = _original_stop(trade_plan)
            initial_risk = abs(position["entry_price"] - original_stop) * position["quantity"]
            r_multiple = float(total_pnl / initial_risk) if initial_risk > 0 else 0.0
            outcome = SimulatedOutcome(
                exit_reason=_map_fill_to_exit_reason(last_fill),
                r_multiple=r_multiple,
                mfe_r=trade_outcome.get("mfe_r") or 0.0,
                mae_r=trade_outcome.get("mae_r") or 0.0,
                exit_price=float(last_fill["price"]),
                exit_time=now,
                duration_seconds=now - position["opened_at"],
            )
            journal_db.finalize_trade_outcome(
                conn, trade_outcome["id"], outcome, commission_usdt=float(total_fees), realized_pnl_usdt=float(total_pnl), now=now,
            )
    return True


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def _max_drawdown(r_values: list[float]) -> float | None:
    """Peak-to-trough of CUMULATIVE r_multiple (running sum) — mirrors
    prediction_tracker._max_drawdown exactly (small deliberate duplication,
    not an import of a private cross-module name, per the Stage 8 plan).
    """
    if not r_values:
        return None
    cumulative = 0.0
    peak = 0.0
    max_dd = 0.0
    for r_val in r_values:
        cumulative += r_val
        peak = max(peak, cumulative)
        max_dd = max(max_dd, peak - cumulative)
    return max_dd


def _bucket_stats(total: int, evaluated_records: list[dict]) -> BucketStats:
    r_values = [rec["r_multiple"] for rec in evaluated_records]
    n = len(r_values)
    wins = sum(1 for rv in r_values if rv > 0)
    gains = sum(rv for rv in r_values if rv > 0)
    losses = sum(-rv for rv in r_values if rv < 0)
    profit_factor = (gains / losses) if losses > 0 else None
    profit_factor_undefined = losses == 0 and gains > 0

    exit_reason_counts = {reason: 0 for reason in _EXIT_REASONS}
    for rec in evaluated_records:
        reason = rec.get("exit_reason")
        if reason in exit_reason_counts:
            exit_reason_counts[reason] += 1

    mfe_values = [rec.get("mfe_r") or 0.0 for rec in evaluated_records]
    mae_values = [rec.get("mae_r") or 0.0 for rec in evaluated_records]
    durations = [rec.get("duration_seconds") or 0.0 for rec in evaluated_records]

    return BucketStats(
        total=total,
        evaluated=n,
        wins=wins,
        win_rate=(wins / n * 100) if n else None,
        expectancy_r=(sum(r_values) / n) if n else None,
        median_r=statistics.median(r_values) if r_values else None,
        profit_factor=profit_factor,
        profit_factor_undefined=profit_factor_undefined,
        max_drawdown_r=_max_drawdown(r_values),
        avg_mfe_r=(sum(mfe_values) / n) if n else None,
        avg_mae_r=(sum(mae_values) / n) if n else None,
        avg_duration_seconds=(sum(durations) / n) if n else None,
        exit_reason_counts=exit_reason_counts,
        low_sample=n < config.MIN_SAMPLE_FOR_STATS,
    )


def compute_paper_trading_statistics(conn, *, mode: str, window_start: float, window_end: float) -> PaperTradingStatistics:
    """Windowed on paper_orders.created_at (not trade_outcomes.evaluated_at)
    so trades still open/pending in the window count toward `total` without
    counting toward `evaluated` — mirrors prediction_tracker's
    total = pending + skipped + evaluated split.
    """
    rows = conn.execute(
        """
        SELECT
            po.id AS paper_order_id, po.created_at AS order_created_at, po.status AS order_status,
            tp.id AS trade_plan_id, tp.source_label,
            ss.ruleset_version, ss.decision, ss.regime,
            tout.status AS outcome_status, tout.r_multiple, tout.mfe_r, tout.mae_r,
            tout.duration_seconds, tout.exit_reason
        FROM paper_orders po
        JOIN trade_plans tp ON tp.id = po.trade_plan_id
        JOIN strategy_scores ss ON ss.id = tp.strategy_score_id
        LEFT JOIN trade_outcomes tout ON tout.trade_plan_id = tp.id
        WHERE tp.mode = ? AND po.created_at >= ? AND po.created_at < ?
        """,
        (mode, window_start, window_end),
    ).fetchall()

    model_totals: dict[str, int] = {}
    model_records: dict[str, list[dict]] = {}
    strategy_totals: dict[tuple[str, str | None], int] = {}
    strategy_records: dict[tuple[str, str | None], list[dict]] = {}
    regime_totals: dict[str, int] = {}
    regime_records: dict[str, list[dict]] = {}

    for row in rows:
        # An order that expired without ever filling never became a real
        # trade attempt (EXPIRED is only set when filled_quantity <= 0 — see
        # _expire_order_if_due) — same treatment as a WAIT signal that never
        # got an order at all, so it must not inflate "still open/in
        # progress" (total - evaluated) forever.
        if row["order_status"] == "EXPIRED":
            continue
        model_key = row["source_label"]
        strategy_key = (row["ruleset_version"], row["decision"])
        regime_key = row["regime"]

        model_totals[model_key] = model_totals.get(model_key, 0) + 1
        strategy_totals[strategy_key] = strategy_totals.get(strategy_key, 0) + 1
        regime_totals[regime_key] = regime_totals.get(regime_key, 0) + 1

        if row["outcome_status"] == "EVALUATED" and row["r_multiple"] is not None:
            record = {
                "r_multiple": row["r_multiple"],
                "mfe_r": row["mfe_r"],
                "mae_r": row["mae_r"],
                "duration_seconds": row["duration_seconds"],
                "exit_reason": row["exit_reason"],
            }
            model_records.setdefault(model_key, []).append(record)
            strategy_records.setdefault(strategy_key, []).append(record)
            regime_records.setdefault(regime_key, []).append(record)

    by_model = {k: _bucket_stats(v, model_records.get(k, [])) for k, v in model_totals.items()}
    by_strategy = {k: _bucket_stats(v, strategy_records.get(k, [])) for k, v in strategy_totals.items()}
    by_regime = {k: _bucket_stats(v, regime_records.get(k, [])) for k, v in regime_totals.items()}

    return PaperTradingStatistics(mode=mode, window_start=window_start, window_end=window_end, by_model=by_model, by_strategy=by_strategy, by_regime=by_regime)


def refresh_statistics(conn, *, mode: str, window_start: float, window_end: float, computed_at: float | None = None) -> StatsRefreshResult:
    """Persists compute_paper_trading_statistics()'s output into
    model_statistics (one row per model) and strategy_statistics (one row
    per (ruleset_version, decision) bucket with regime=NULL, plus one row
    per regime bucket with decision=NULL, ruleset_version=
    config.STRATEGY_RULESET_VERSION). Does not delete/supersede prior rows —
    each call adds a new computed_at-stamped snapshot; readers take the
    latest via get_model_statistics/get_strategy_statistics ORDER BY
    computed_at DESC.
    """
    computed_at = computed_at if computed_at is not None else time.time()
    stats = compute_paper_trading_statistics(conn, mode=mode, window_start=window_start, window_end=window_end)

    model_rows = 0
    for label, bucket in stats.by_model.items():
        journal_db.insert_model_statistics(
            conn, model_label=label, mode=mode, window_start=window_start, window_end=window_end,
            total_predictions=bucket.total, evaluated_count=bucket.evaluated, wins=bucket.wins,
            win_rate=bucket.win_rate, expectancy_r=bucket.expectancy_r, median_r=bucket.median_r,
            profit_factor=bucket.profit_factor, profit_factor_undefined=bucket.profit_factor_undefined,
            max_drawdown_r=bucket.max_drawdown_r, avg_mfe_r=bucket.avg_mfe_r, avg_mae_r=bucket.avg_mae_r,
            avg_duration_seconds=bucket.avg_duration_seconds, exit_reason_counts=bucket.exit_reason_counts,
            low_sample=bucket.low_sample, computed_at=computed_at,
        )
        model_rows += 1

    strategy_rows = 0
    for (ruleset_version, decision), bucket in stats.by_strategy.items():
        journal_db.insert_strategy_statistics(
            conn, ruleset_version=ruleset_version, mode=mode, decision=decision, regime=None,
            window_start=window_start, window_end=window_end, total_signals=bucket.total,
            evaluated_count=bucket.evaluated, wins=bucket.wins, win_rate=bucket.win_rate,
            expectancy_r=bucket.expectancy_r, median_r=bucket.median_r, profit_factor=bucket.profit_factor,
            low_sample=bucket.low_sample, computed_at=computed_at,
        )
        strategy_rows += 1

    for regime, bucket in stats.by_regime.items():
        journal_db.insert_strategy_statistics(
            conn, ruleset_version=config.STRATEGY_RULESET_VERSION, mode=mode, decision=None, regime=regime,
            window_start=window_start, window_end=window_end, total_signals=bucket.total,
            evaluated_count=bucket.evaluated, wins=bucket.wins, win_rate=bucket.win_rate,
            expectancy_r=bucket.expectancy_r, median_r=bucket.median_r, profit_factor=bucket.profit_factor,
            low_sample=bucket.low_sample, computed_at=computed_at,
        )
        strategy_rows += 1

    return StatsRefreshResult(model_rows_inserted=model_rows, strategy_rows_inserted=strategy_rows)
