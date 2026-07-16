"""Offline tests for runtime/service.py's TradingRuntime — no network. Every
external call (market_data_engine, paper_trading, execution_engine,
position_manager) is monkeypatched; only journal_db (against a tmp_path
SQLite file) is real.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import bingx_client
import config
import execution_engine
import journal_db
import market_data_engine
import paper_trading
import position_manager
from runtime.service import TradingRuntime

NOW = 1_700_000_000.0


@pytest.fixture(autouse=True)
def _use_tmp_paths(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "JOURNAL_DB_FILE", tmp_path / "runtime_test.db")
    monkeypatch.setattr(config, "RUNTIME_HEARTBEAT_FILE", tmp_path / "heartbeat.json")
    monkeypatch.setattr(config, "RUNTIME_MODE", "PAPER")
    monkeypatch.setattr(config, "RUNTIME_MEDIUM_INTERVAL_SECONDS", 10.0)


@pytest.fixture
def conn(_use_tmp_paths):
    c = journal_db.init_db()
    yield c
    c.close()


def _snapshot(data_quality="GOOD"):
    return SimpleNamespace(data_quality=data_quality)


def _no_op_reconciliation_report(symbol):
    return position_manager.ReconciliationReport(
        symbol=symbol, local_open_not_on_exchange=[], exchange_open_not_local=[], errors=[]
    )


def test_run_once_calls_paper_tick_in_paper_mode(conn, monkeypatch):
    monkeypatch.setattr(market_data_engine, "collect_snapshot", lambda symbol, now=None: _snapshot("GOOD"))
    calls = []
    monkeypatch.setattr(
        paper_trading, "process_tick",
        lambda c, symbol, now=None: calls.append(("paper_tick", symbol)) or paper_trading.TickResult(
            symbol=symbol, now=now, bid=None, ask=None, orders_filled=[], orders_partially_filled=[],
            orders_expired=[], orders_force_closed_partial=[], positions_opened=[], positions_closed=[],
            stops_trailed=[], fills_created=[], errors=[],
        ),
    )

    runtime = TradingRuntime("ETHUSDT", conn=conn)
    result = runtime.run_once(now=NOW)

    assert ("paper_tick", "ETHUSDT") in calls
    assert result.market_data_quality == "GOOD"
    assert result.paper_tick is not None
    assert result.errors == []


def test_run_once_skips_paper_tick_when_data_quality_no_trade(conn, monkeypatch):
    monkeypatch.setattr(market_data_engine, "collect_snapshot", lambda symbol, now=None: _snapshot("NO_TRADE"))
    monkeypatch.setattr(
        paper_trading, "process_tick",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("process_tick should not be called on NO_TRADE data")),
    )

    runtime = TradingRuntime("ETHUSDT", conn=conn)
    result = runtime.run_once(now=NOW)

    assert result.paper_tick is None
    assert any("NO_TRADE" in e for e in result.errors)


def test_run_once_does_not_call_paper_tick_in_monitor_only_mode(conn, monkeypatch):
    monkeypatch.setattr(config, "RUNTIME_MODE", "MONITOR_ONLY")
    monkeypatch.setattr(market_data_engine, "collect_snapshot", lambda symbol, now=None: _snapshot("GOOD"))
    monkeypatch.setattr(
        paper_trading, "process_tick",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("process_tick should not run in MONITOR_ONLY")),
    )

    runtime = TradingRuntime("ETHUSDT", conn=conn)
    result = runtime.run_once(now=NOW)
    assert result.paper_tick is None
    assert result.errors == []


def test_run_once_calls_monitor_only_when_medium_interval_elapsed(conn, monkeypatch):
    monkeypatch.setattr(market_data_engine, "collect_snapshot", lambda symbol, now=None: _snapshot("GOOD"))
    monkeypatch.setattr(paper_trading, "process_tick", lambda c, symbol, now=None: None)
    monitor_calls = []
    monkeypatch.setattr(execution_engine, "monitor", lambda c, symbol: monitor_calls.append(symbol))

    runtime = TradingRuntime("ETHUSDT", conn=conn)
    result1 = runtime.run_once(now=NOW)
    assert result1.real_monitor_ran is True  # first tick always runs it (elapsed since epoch 0 is huge)
    assert monitor_calls == ["ETHUSDT"]

    result2 = runtime.run_once(now=NOW + 2)  # well within the 10s medium interval
    assert result2.real_monitor_ran is False
    assert monitor_calls == ["ETHUSDT"]  # not called again

    result3 = runtime.run_once(now=NOW + 11)  # past the medium interval
    assert result3.real_monitor_ran is True
    assert monitor_calls == ["ETHUSDT", "ETHUSDT"]


def test_run_once_survives_collect_snapshot_failure(conn, monkeypatch):
    monkeypatch.setattr(
        market_data_engine, "collect_snapshot",
        lambda symbol, now=None: (_ for _ in ()).throw(bingx_client.NetworkError("down")),
    )
    monkeypatch.setattr(execution_engine, "monitor", lambda c, symbol: None)

    runtime = TradingRuntime("ETHUSDT", conn=conn)
    result = runtime.run_once(now=NOW)

    assert result.market_data_quality is None
    assert result.paper_tick is None
    assert any("collect_snapshot failed" in e for e in result.errors)

    from runtime.heartbeat import read_heartbeat
    hb = read_heartbeat(config.RUNTIME_HEARTBEAT_FILE)
    assert hb is not None  # heartbeat still written despite the failure


def test_run_once_survives_process_tick_exception(conn, monkeypatch):
    monkeypatch.setattr(market_data_engine, "collect_snapshot", lambda symbol, now=None: _snapshot("GOOD"))
    monkeypatch.setattr(
        paper_trading, "process_tick",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    monkeypatch.setattr(execution_engine, "monitor", lambda c, symbol: None)

    runtime = TradingRuntime("ETHUSDT", conn=conn)
    result = runtime.run_once(now=NOW)
    assert any("process_tick failed" in e for e in result.errors)


def test_startup_recovery_logs_warning_on_discrepancy(conn, monkeypatch):
    report = position_manager.ReconciliationReport(
        symbol="ETHUSDT", local_open_not_on_exchange=[42], exchange_open_not_local=[], errors=[]
    )
    monkeypatch.setattr(position_manager, "recover_after_restart", lambda c, symbol: report)

    runtime = TradingRuntime("ETHUSDT", conn=conn)
    runtime._startup_recovery()

    events = journal_db.get_recent_system_events(conn, event_code="RUNTIME_STARTUP_RECONCILIATION_DISCREPANCY")
    assert len(events) == 1


def test_startup_recovery_no_warning_when_clean(conn, monkeypatch):
    monkeypatch.setattr(position_manager, "recover_after_restart", lambda c, symbol: _no_op_reconciliation_report(symbol))

    runtime = TradingRuntime("ETHUSDT", conn=conn)
    runtime._startup_recovery()

    events = journal_db.get_recent_system_events(conn, event_code="RUNTIME_STARTUP_RECONCILIATION_DISCREPANCY")
    assert events == []
    startup_events = journal_db.get_recent_system_events(conn, event_code="RUNTIME_STARTUP")
    assert len(startup_events) == 1
