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

Two cadences, tracked by elapsed wall-clock time (not an iteration
counter, which would drift if a tick ever takes longer than usual):
  - FAST (config.RUNTIME_FAST_INTERVAL_SECONDS, default 2s): one market
    data fetch, then paper_trading.process_tick (RUNTIME_MODE="PAPER" only).
  - MEDIUM (config.RUNTIME_MEDIUM_INTERVAL_SECONDS, default 10s):
    execution_engine.monitor for already-open REAL positions. Deliberately
    NOT on the fast cadence — position_manager's checks still fetch their
    own price internally (see the plan for why a full price-injection
    refactor across Stages 11-12 was out of scope here); running them less
    often keeps the extra BingX calls low without touching that code.

execution_engine.confirm_and_execute (placing a NEW real order) is never
called from here — that stays a separate, manual action. This runtime only
ever monitors positions/orders that already exist.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import bingx_client
import config
import execution_engine
import journal_db
import market_data_engine
import paper_trading
import position_manager
from runtime import heartbeat, locking


@dataclass(frozen=True)
class RuntimeTickResult:
    market_data_quality: str | None
    paper_tick: paper_trading.TickResult | None
    real_monitor_ran: bool
    errors: list[str] = field(default_factory=list)


class TradingRuntime:
    def __init__(self, symbol: str, *, conn=None) -> None:
        self.symbol = symbol
        self._conn = conn if conn is not None else journal_db.init_db()
        self._lock_handle = None
        self._last_medium_run: float = 0.0

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

        real_monitor_ran = False
        if now - self._last_medium_run >= config.RUNTIME_MEDIUM_INTERVAL_SECONDS:
            try:
                execution_engine.monitor(self._conn, self.symbol)
                self._conn.commit()
                real_monitor_ran = True
            except Exception as exc:  # best-effort, same as above
                errors.append(f"execution_engine.monitor failed: {exc}")
            self._last_medium_run = now

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
            errors=errors,
        )

    # -----------------------------------------------------------------
    # The actual long-lived loop
    # -----------------------------------------------------------------

    def run(self) -> None:
        self._lock_handle = locking.acquire_lock(config.RUNTIME_LOCK_FILE)
        try:
            self._startup_recovery()
            while True:
                self.run_once()
                time.sleep(config.RUNTIME_FAST_INTERVAL_SECONDS)
        except KeyboardInterrupt:
            journal_db.log_event(
                self._conn, level="INFO", event_code="RUNTIME_SHUTDOWN", source_module="trading_runtime",
                symbol=self.symbol, message="trading_runtime stopped (KeyboardInterrupt).",
            )
            self._conn.commit()
        finally:
            locking.release_lock(self._lock_handle)
            self._conn.close()
