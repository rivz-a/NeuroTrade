"""TradingRuntime — the scheduler that actually calls paper_trading and
execution_engine on a repeating cadence, closing the "must be called from
outside" gap every engine before this stage deliberately left open (see
paper_trading.py's own module docstring, Stage 8).

A plain synchronous loop, not asyncio: every other module in this codebase
(bingx_client, journal_db/sqlite3, paper_trading, execution_engine) is
synchronous, and this app trades one instrument on one local machine — the
concurrency asyncio buys isn't needed here, and would mean either wrapping
every sqlite3 call in run_in_executor or switching to aiosqlite for no real
benefit.

Three cadences, tracked by elapsed wall-clock time (not an iteration
counter, which would drift if a tick ever takes longer than usual):
  - FAST (config.RUNTIME_FAST_INTERVAL_SECONDS, default 2s): one market
    data fetch, then paper_trading.process_tick (RUNTIME_MODE="PAPER" only).
  - MEDIUM (config.RUNTIME_MEDIUM_INTERVAL_SECONDS, default 10s):
    execution_engine.monitor for already-open REAL positions. Deliberately
    NOT on the fast cadence — position_manager's checks still fetch their
    own price internally (see the plan for why a full price-injection
    refactor across Stages 11-12 was out of scope here); running them less
    often keeps the extra BingX calls low without touching that code.
  - AI_CYCLE: runtime/ai_cycle.py, cycled independently for EVERY trading
    mode (scalping AND swing) on its own cadence — the configured primary
    mode (config.TRADING_MODE) uses config.RUNTIME_AI_CYCLE_INTERVAL_SECONDS
    (its own prediction horizon, or an env override); any other mode uses
    config.RUNTIME_SECONDARY_MODE_CYCLE_INTERVAL_SECONDS (2h default) rather
    than its own (much longer) prediction horizon, so a setup that develops
    between checks isn't missed just because the last signal's horizon
    hasn't elapsed — a rejected/WAIT signal has nothing "in flight" to wait
    out. Each mode's cycle: fetch fresh data, ask the AI, and open a paper
    position if
    the consensus is actionable. RUNTIME_MODE="PAPER" only, and every mode
    is skipped (without consuming/advancing its own interval) while a
    pending/open PAPER order already exists for the symbol — the "one order
    at a time" invariant is shared ACROSS modes, not per mode, so this app
    never stacks more than one paper position at a time on the same symbol
    the same way the plan for eventual real trading insists on ("один ордер
    одновременно"), even if e.g. scalping and swing both have an actionable
    signal in the same tick (whichever is checked first, per
    sorted(VALID_TRADING_MODES), claims the slot; the rest wait for it to
    close). Also fires immediately, interval or not, for every mode at once
    when config.RUNTIME_AI_CYCLE_TRIGGER_FILE exists — server.py's dashboard
    "Обновить" button touches that file instead of paying for its own
    separate AI call, so the runtime (which already pays for this analysis
    on its own schedule) stays the single source of truth instead of two
    independent paid AI-call paths existing side by side.

execution_engine.confirm_and_execute (placing a NEW real order) is never
called from here — that stays a separate, manual action. This runtime only
ever monitors positions/orders that already exist.
"""

from __future__ import annotations

import signal
import time
from dataclasses import dataclass, field

import bingx_client
import config
import execution_engine
import journal_db
import market_data_engine
import paper_trading
import position_manager
from runtime import ai_cycle, heartbeat, locking


class _ShutdownRequested(Exception):
    """Raised from the SIGTERM handler so `systemctl stop` unwinds the run()
    loop through the same clean-shutdown path as Ctrl+C (KeyboardInterrupt),
    instead of Python's default SIGTERM behavior of killing the process
    immediately without running any `finally` block.
    """


def _raise_shutdown(signum, frame) -> None:
    raise _ShutdownRequested()


@dataclass(frozen=True)
class RuntimeTickResult:
    market_data_quality: str | None
    paper_tick: paper_trading.TickResult | None
    real_monitor_ran: bool
    ai_cycle_results: dict[str, ai_cycle.AICycleResult] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)


class TradingRuntime:
    def __init__(self, symbol: str, *, conn=None) -> None:
        self.symbol = symbol
        self._conn = conn if conn is not None else journal_db.init_db()
        self._lock_handle = None
        self._last_medium_run: float = 0.0
        # Per mode: each trading mode (scalping/swing) is cycled on its own
        # cadence -- scalping every RUNTIME_AI_CYCLE_INTERVAL_SECONDS (its
        # own prediction horizon, or an env override), swing every
        # PREDICTION_HORIZON_SECONDS["swing"] (8h by default) -- so asking
        # about a multi-hour swing setup every 30 minutes doesn't burn AI
        # calls for a plan that hasn't even had time to resolve yet.
        self._last_ai_cycle_run: dict[str, float] = {}

    # -----------------------------------------------------------------
    # Startup recovery
    # -----------------------------------------------------------------

    def _startup_recovery(self) -> None:
        """Reuses position_manager.recover_after_restart (Stage 11) for
        REAL state — no new reconciliation logic. Paper positions need no
        special recovery step: paper_trading.py is fully DB-resumable by
        design (see its own module docstring), so the next process_tick
        call just continues from wherever journal_db left off.
        """
        report = position_manager.recover_after_restart(self._conn, self.symbol)
        if report.local_open_not_on_exchange or report.exchange_open_not_local:
            journal_db.log_event(
                self._conn,
                level="WARNING",
                event_code="RUNTIME_STARTUP_RECONCILIATION_DISCREPANCY",
                source_module="trading_runtime",
                symbol=self.symbol,
                message=(
                    f"Startup reconciliation found discrepancies: "
                    f"local_open_not_on_exchange={report.local_open_not_on_exchange}, "
                    f"exchange_open_not_local={report.exchange_open_not_local}"
                ),
            )
        self._conn.commit()

        open_paper = journal_db.get_open_positions(self._conn, self.symbol, source="PAPER")
        journal_db.log_event(
            self._conn,
            level="INFO",
            event_code="RUNTIME_STARTUP",
            source_module="trading_runtime",
            symbol=self.symbol,
            message=f"trading_runtime starting for {self.symbol} — {len(open_paper)} open PAPER position(s) resumed.",
        )
        self._conn.commit()

    # -----------------------------------------------------------------
    # One tick
    # -----------------------------------------------------------------

    def run_once(self, *, now: float | None = None) -> RuntimeTickResult:
        now = now if now is not None else time.time()
        errors: list[str] = []

        try:
            snapshot = market_data_engine.collect_snapshot(self.symbol, now=now)
            data_quality = snapshot.data_quality
        except bingx_client.BingXError as exc:
            errors.append(f"collect_snapshot failed: {exc}")
            data_quality = None

        paper_tick_result: paper_trading.TickResult | None = None
        if data_quality == "NO_TRADE":
            errors.append("skipped paper tick: market data quality is NO_TRADE")
        elif data_quality is not None and config.RUNTIME_MODE == "PAPER":
            try:
                paper_tick_result = paper_trading.process_tick(self._conn, self.symbol, now=now)
                self._conn.commit()
            except Exception as exc:  # best-effort: one bad tick must not kill the runtime
                errors.append(f"process_tick failed: {exc}")

        # PAPER mode must stay entirely inert with respect to REAL state —
        # execution_engine.monitor touches real_orders/positions (source=
        # "REAL") even when EXECUTION_DRY_RUN short-circuits the actual
        # network calls, so it only runs in a mode that's explicitly meant
        # to watch REAL positions.
        real_monitor_ran = False
        if config.RUNTIME_MODE != "PAPER" and now - self._last_medium_run >= config.RUNTIME_MEDIUM_INTERVAL_SECONDS:
            try:
                execution_engine.monitor(self._conn, self.symbol)
                self._conn.commit()
                real_monitor_ran = True
            except Exception as exc:  # best-effort, same as above
                errors.append(f"execution_engine.monitor failed: {exc}")
            self._last_medium_run = now

        open_paper = journal_db.get_open_positions(self._conn, self.symbol, source="PAPER")
        open_real = journal_db.get_open_positions(self._conn, self.symbol, source="REAL")

        ai_cycle_results: dict[str, ai_cycle.AICycleResult] = {}
        triggered = config.RUNTIME_MODE == "PAPER" and config.RUNTIME_AI_CYCLE_TRIGGER_FILE.exists()
        trigger_consumed = False
        if config.RUNTIME_MODE == "PAPER":
            for mode in sorted(config.VALID_TRADING_MODES):
                # The configured primary mode keeps its existing, overridable
                # cadence (RUNTIME_AI_CYCLE_INTERVAL_SECONDS); any other mode
                # cycled alongside it uses RUNTIME_SECONDARY_MODE_CYCLE_
                # INTERVAL_SECONDS -- deliberately NOT that mode's own (much
                # longer) prediction horizon, so a setup that develops between
                # checks isn't missed just because the last signal's horizon
                # hasn't elapsed yet (a rejected/WAIT signal has nothing "in
                # flight" to wait out).
                interval = (
                    config.RUNTIME_AI_CYCLE_INTERVAL_SECONDS
                    if mode == config.TRADING_MODE
                    else config.RUNTIME_SECONDARY_MODE_CYCLE_INTERVAL_SECONDS
                )
                interval_elapsed = now - self._last_ai_cycle_run.get(mode, 0.0) >= interval
                if not (triggered or interval_elapsed):
                    continue
                pending_paper_orders = journal_db.get_pending_paper_orders(self._conn, self.symbol)
                if pending_paper_orders or open_paper:
                    # The one-position-per-symbol slot is taken (by this tick's
                    # own earlier mode, or from before) — every remaining mode
                    # waits, same as the original single-mode gate. A pending
                    # trigger stays queued (not consumed) until a slot is free.
                    break
                if triggered and not trigger_consumed:
                    try:
                        config.RUNTIME_AI_CYCLE_TRIGGER_FILE.unlink()
                    except OSError:
                        pass
                    trigger_consumed = True
                self._last_ai_cycle_run[mode] = now
                # An AI cycle can block for up to ~AI_REQUEST_TIMEOUT seconds
                # (all models run in parallel, but that's still far past
                # RUNTIME_HEARTBEAT_STALE_SECONDS) — write a heartbeat NOW,
                # tagged "ai_cycle", so heartbeat_status() applies the wider
                # busy-stale budget instead of flagging this as a hang.
                heartbeat.write_heartbeat(
                    config.RUNTIME_HEARTBEAT_FILE,
                    pid=None,
                    mode=config.RUNTIME_MODE,
                    symbol=self.symbol,
                    last_tick_at=now,
                    market_data_quality=data_quality,
                    open_paper_positions=len(open_paper),
                    open_real_positions=len(open_real),
                    last_error=errors[-1] if errors else None,
                    activity="ai_cycle",
                )
                try:
                    ai_cycle_results[mode] = ai_cycle.run_ai_cycle(self._conn, self.symbol, mode, now=now)
                    self._conn.commit()
                except Exception as exc:  # best-effort, same as paper tick/monitor above
                    errors.append(f"ai_cycle failed ({mode}): {exc}")
                # A just-opened paper position changes both counts below, and
                # must be visible to the next mode's gate check in this loop.
                open_paper = journal_db.get_open_positions(self._conn, self.symbol, source="PAPER")
                open_real = journal_db.get_open_positions(self._conn, self.symbol, source="REAL")

        heartbeat.write_heartbeat(
            config.RUNTIME_HEARTBEAT_FILE,
            pid=None,
            mode=config.RUNTIME_MODE,
            symbol=self.symbol,
            last_tick_at=now,
            market_data_quality=data_quality,
            open_paper_positions=len(open_paper),
            open_real_positions=len(open_real),
            last_error=errors[-1] if errors else None,
        )

        return RuntimeTickResult(
            market_data_quality=data_quality,
            paper_tick=paper_tick_result,
            real_monitor_ran=real_monitor_ran,
            ai_cycle_results=ai_cycle_results,
            errors=errors,
        )

    # -----------------------------------------------------------------
    # The actual long-lived loop
    # -----------------------------------------------------------------

    def run(self) -> None:
        previous_sigterm_handler = signal.signal(signal.SIGTERM, _raise_shutdown)
        self._lock_handle = locking.acquire_lock(config.RUNTIME_LOCK_FILE)
        try:
            self._startup_recovery()
            while True:
                self.run_once()
                time.sleep(config.RUNTIME_FAST_INTERVAL_SECONDS)
        except (KeyboardInterrupt, _ShutdownRequested) as exc:
            reason = "KeyboardInterrupt" if isinstance(exc, KeyboardInterrupt) else "SIGTERM"
            journal_db.log_event(
                self._conn, level="INFO", event_code="RUNTIME_SHUTDOWN", source_module="trading_runtime",
                symbol=self.symbol, message=f"trading_runtime stopped ({reason}).",
            )
            self._conn.commit()
        finally:
            signal.signal(signal.SIGTERM, previous_sigterm_handler)
            locking.release_lock(self._lock_handle)
            self._conn.close()
