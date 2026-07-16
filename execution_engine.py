"""Top-level orchestrator for Stage 11's semi-auto execution: the global,
account-wide safety gates (kill switch, daily loss limit, max trades/day,
cooldown after a stop) run here, BEFORE `order_manager.place_entry_order`
ever runs its own per-trade checks. `confirm_and_execute` is the one
function a future "Confirm and send to BingX" dashboard button would call
(this stage stays backend-only — see the Stage 11 plan).

Global gates are derived entirely from journal_db's own audit trail (SUM of
today's REAL realized PnL, COUNT of today's REAL entry orders, MAX of the
last STOP_LOSS fill's timestamp) rather than a separate counter/state store
— the same data order_manager/position_manager already write, just
aggregated, so there is no second source of truth that could drift out of
sync with the journal.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal

import config
import journal_db
import order_manager
import position_manager


def _rejected(reason: str) -> order_manager.OrderResult:
    return order_manager.OrderResult(status="REJECTED", reason=reason, real_order_ids=[], exchange_order_ids=[])


def kill_switch_engaged() -> bool:
    """A file, not a .env flag, specifically so it can be toggled in an
    emergency WITHOUT restarting the process."""
    return config.EXECUTION_KILL_SWITCH_FILE.exists()


def _today_bounds(now: float) -> tuple[float, float]:
    start_of_day = datetime.fromtimestamp(now, tz=timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    start = start_of_day.timestamp()
    return start, start + 86400


def _today_real_stats(conn, symbol: str, now: float) -> tuple[int, Decimal, float | None]:
    """Returns (entry orders placed today, sum of today's realized REAL PnL,
    epoch of the most recent REAL stop-loss fill or None). An entry order is
    identified by stop_loss/take_profit both being NULL on its real_orders
    row — see order_manager.place_entry_order's comment on why those fields
    are reserved for SL/TP rows only.
    """
    start, end = _today_bounds(now)

    trade_count_row = conn.execute(
        """
        SELECT COUNT(*) AS c FROM real_orders
        WHERE symbol = ? AND stop_loss IS NULL AND take_profit IS NULL
          AND status != 'REJECTED' AND created_at >= ? AND created_at < ?
        """,
        (symbol, start, end),
    ).fetchone()
    trade_count = trade_count_row["c"] if trade_count_row else 0

    pnl_row = conn.execute(
        """
        SELECT COALESCE(SUM(CAST(f.realized_pnl_usdt AS REAL)), 0) AS total
        FROM fills f JOIN positions p ON p.id = f.position_id
        WHERE p.source = 'REAL' AND f.symbol = ? AND f.fill_type != 'ENTRY'
          AND f.filled_at >= ? AND f.filled_at < ?
        """,
        (symbol, start, end),
    ).fetchone()
    daily_pnl = Decimal(str(pnl_row["total"])) if pnl_row else Decimal("0")

    last_stop_row = conn.execute(
        """
        SELECT MAX(f.filled_at) AS t FROM fills f JOIN positions p ON p.id = f.position_id
        WHERE p.source = 'REAL' AND f.symbol = ? AND f.fill_type = 'STOP_LOSS'
        """,
        (symbol,),
    ).fetchone()
    last_stop_time = last_stop_row["t"] if last_stop_row and last_stop_row["t"] is not None else None

    return trade_count, daily_pnl, last_stop_time


def confirm_and_execute(conn, trade_plan_id: int, *, now: float | None = None) -> order_manager.OrderResult:
    """The "Подтвердить и отправить в BingX" entrypoint. Runs the global
    gates first (kill switch -> daily loss limit -> max trades/day ->
    cooldown after a stop), each an immediate REJECTED without ever calling
    order_manager, then defers to order_manager.place_entry_order for every
    per-trade check and the actual placement.
    """
    now = now if now is not None else time.time()

    if kill_switch_engaged():
        return _rejected("kill switch is engaged (kill_switch.flag present) — all execution blocked")

    trade_plan = journal_db.get_trade_plan(conn, trade_plan_id)
    if trade_plan is None:
        return _rejected("trade plan not found")
    symbol = trade_plan["symbol"]

    trade_count, daily_pnl, last_stop_time = _today_real_stats(conn, symbol, now)

    if daily_pnl <= -Decimal(str(config.EXECUTION_DAILY_LOSS_LIMIT_USDT)):
        return _rejected(
            f"daily loss limit reached: {daily_pnl} USDT <= -{config.EXECUTION_DAILY_LOSS_LIMIT_USDT} USDT"
        )

    if trade_count >= config.EXECUTION_MAX_TRADES_PER_DAY:
        return _rejected(
            f"max trades per day reached: {trade_count} >= {config.EXECUTION_MAX_TRADES_PER_DAY}"
        )

    if last_stop_time is not None and now - last_stop_time < config.EXECUTION_COOLDOWN_AFTER_STOP_SECONDS:
        remaining = config.EXECUTION_COOLDOWN_AFTER_STOP_SECONDS - (now - last_stop_time)
        return _rejected(f"cooldown after stop loss still active — {remaining:.0f}s remaining")

    return order_manager.place_entry_order(conn, trade_plan_id, now=now)


def monitor(conn, symbol: str) -> None:
    """Reconcile actual state first, then confirm/give up on still-unfilled
    entries, then — for every position still OPEN — manage its stop and
    check the exit triggers, cheapest/most-local last (time-based close
    only needs the clock; regime/data-quality checks may need a live
    fetch). Called from outside (a script, a future scheduler) — like
    paper_trading.process_tick, this module never runs its own loop.
    """
    position_manager.sync_position_status(conn, symbol)

    for order in journal_db.get_pending_real_orders(conn, symbol):
        if order.get("stop_loss") is not None or order.get("take_profit") is not None:
            continue  # an SL/TP row, not an entry
        trade_plan_id = order.get("trade_plan_id")
        if trade_plan_id is None:
            continue
        if not position_manager.check_entry_fill(conn, trade_plan_id):
            position_manager.cancel_stale_entry_order(conn, trade_plan_id)

    for position in journal_db.get_open_positions(conn, symbol, source="REAL"):
        position_manager.manage_stop_loss(conn, position["id"])
        position_manager.check_regime_close(conn, position["id"])
        position_manager.check_data_quality_close(conn, position["id"])
        position_manager.check_time_based_close(conn, position["id"])


def close_all(conn, symbol: str) -> order_manager.CloseAllResult:
    """Прямой проброс — the single entrypoint a future 'close everything'
    button would call."""
    return order_manager.close_all(conn, symbol)
