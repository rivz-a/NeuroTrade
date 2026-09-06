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
from runtime import ai_cycle
from runtime.service import TradingRuntime

NOW = 1_700_000_000.0


def _noop_ai_cycle(conn, symbol, mode, *, now=None):
    return ai_cycle.AICycleResult(ran=False, reason="test-default-noop")


@pytest.fixture(autouse=True)
def _use_tmp_paths(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "JOURNAL_DB_FILE", tmp_path / "runtime_test.db")
    monkeypatch.setattr(config, "RUNTIME_HEARTBEAT_FILE", tmp_path / "heartbeat.json")
    monkeypatch.setattr(config, "RUNTIME_MODE", "PAPER")
    monkeypatch.setattr(config, "RUNTIME_MEDIUM_INTERVAL_SECONDS", 10.0)
    monkeypatch.setattr(config, "RUNTIME_AI_CYCLE_INTERVAL_SECONDS", 1800.0)
    monkeypatch.setattr(config, "RUNTIME_AI_CYCLE_TRIGGER_FILE", tmp_path / "ai_cycle_trigger.flag")
    # Every test gets a harmless no-op by default -- tests targeting the AI
    # cycle itself override this explicitly, same convention as the other
    # external calls (market_data_engine, paper_trading, execution_engine)
    # this file always monkeypatches per-test.
    monkeypatch.setattr(ai_cycle, "run_ai_cycle", _noop_ai_cycle)


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
    monkeypatch.setattr(config, "RUNTIME_MODE", "MONITOR_ONLY")
    monkeypatch.setattr(market_data_engine, "collect_snapshot", lambda symbol, now=None: _snapshot("GOOD"))
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


def test_run_once_never_calls_monitor_in_paper_mode(conn, monkeypatch):
    # PAPER mode must stay entirely inert with respect to REAL state, even
    # though the medium interval has long since elapsed.
    monkeypatch.setattr(market_data_engine, "collect_snapshot", lambda symbol, now=None: _snapshot("GOOD"))
    monkeypatch.setattr(paper_trading, "process_tick", lambda c, symbol, now=None: None)
    monkeypatch.setattr(
        execution_engine, "monitor",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("monitor should never run in PAPER mode")),
    )

    runtime = TradingRuntime("ETHUSDT", conn=conn)
    result = runtime.run_once(now=NOW)
    assert result.real_monitor_ran is False


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


def test_run_once_runs_ai_cycle_on_first_tick_when_nothing_open(conn, monkeypatch):
    # Every trading mode is cycled independently -- on a cold start (nothing
    # ever ran, nothing open), both scalping and swing are due in the same
    # tick, sorted alphabetically.
    monkeypatch.setattr(market_data_engine, "collect_snapshot", lambda symbol, now=None: _snapshot("GOOD"))
    monkeypatch.setattr(paper_trading, "process_tick", lambda c, symbol, now=None: None)
    calls = []

    def _tracked(c, symbol, mode, *, now=None):
        calls.append((symbol, mode))
        return ai_cycle.AICycleResult(ran=True, trade_plan_id=1, paper_order_status="SKIPPED_NOT_ACTIONABLE")

    monkeypatch.setattr(ai_cycle, "run_ai_cycle", _tracked)

    runtime = TradingRuntime("ETHUSDT", conn=conn)
    result = runtime.run_once(now=NOW)

    assert calls == [("ETHUSDT", "scalping"), ("ETHUSDT", "swing")]
    assert result.ai_cycle_results.keys() == {"scalping", "swing"}
    assert result.ai_cycle_results["scalping"].ran is True
    assert result.ai_cycle_results["scalping"].trade_plan_id == 1


def test_run_once_skips_ai_cycle_when_open_paper_position_exists(conn, monkeypatch):
    monkeypatch.setattr(market_data_engine, "collect_snapshot", lambda symbol, now=None: _snapshot("GOOD"))
    monkeypatch.setattr(paper_trading, "process_tick", lambda c, symbol, now=None: None)
    monkeypatch.setattr(
        ai_cycle, "run_ai_cycle",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("ai_cycle should not run with an open PAPER position")),
    )
    journal_db.insert_position(
        conn, symbol="ETHUSDT", side="LONG", source="PAPER", entry_price="100", quantity="1", now=NOW
    )
    conn.commit()

    runtime = TradingRuntime("ETHUSDT", conn=conn)
    result = runtime.run_once(now=NOW)
    assert result.ai_cycle_results == {}
    assert result.errors == []


def test_run_once_skips_ai_cycle_when_pending_paper_order_exists(conn, monkeypatch):
    monkeypatch.setattr(market_data_engine, "collect_snapshot", lambda symbol, now=None: _snapshot("GOOD"))
    monkeypatch.setattr(paper_trading, "process_tick", lambda c, symbol, now=None: None)
    monkeypatch.setattr(
        ai_cycle, "run_ai_cycle",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("ai_cycle should not run with a pending paper order")),
    )
    journal_db.insert_paper_order(
        conn, symbol="ETHUSDT", side="LONG", order_type="LIMIT", quantity="1",
        entry_from=100, entry_to=101, stop_loss=95, status="PENDING", now=NOW,
    )
    conn.commit()

    runtime = TradingRuntime("ETHUSDT", conn=conn)
    result = runtime.run_once(now=NOW)
    assert result.ai_cycle_results == {}


def test_run_once_does_not_run_ai_cycle_in_monitor_only_mode(conn, monkeypatch):
    monkeypatch.setattr(config, "RUNTIME_MODE", "MONITOR_ONLY")
    monkeypatch.setattr(market_data_engine, "collect_snapshot", lambda symbol, now=None: _snapshot("GOOD"))
    monkeypatch.setattr(execution_engine, "monitor", lambda c, symbol: None)
    monkeypatch.setattr(
        ai_cycle, "run_ai_cycle",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("ai_cycle should never run outside PAPER mode")),
    )

    runtime = TradingRuntime("ETHUSDT", conn=conn)
    result = runtime.run_once(now=NOW)
    assert result.ai_cycle_results == {}


def test_run_once_respects_ai_cycle_interval(conn, monkeypatch):
    # swing's own (much longer) horizon means it only fires on the cold-start
    # tick in this test -- the interval under test here is scalping's.
    monkeypatch.setattr(config, "RUNTIME_AI_CYCLE_INTERVAL_SECONDS", 1800.0)
    monkeypatch.setattr(market_data_engine, "collect_snapshot", lambda symbol, now=None: _snapshot("GOOD"))
    monkeypatch.setattr(paper_trading, "process_tick", lambda c, symbol, now=None: None)
    calls = []
    monkeypatch.setattr(
        ai_cycle, "run_ai_cycle",
        lambda c, symbol, mode, *, now=None: calls.append((mode, now)) or ai_cycle.AICycleResult(ran=True),
    )

    runtime = TradingRuntime("ETHUSDT", conn=conn)
    runtime.run_once(now=NOW)  # first tick always runs every mode (elapsed since epoch 0 is huge)
    scalping_calls = [c for c in calls if c[0] == "scalping"]
    assert scalping_calls == [("scalping", NOW)]

    runtime.run_once(now=NOW + 60)  # well within the 1800s interval
    assert [c for c in calls if c[0] == "scalping"] == [("scalping", NOW)]  # not called again

    runtime.run_once(now=NOW + 1801)  # past the interval
    assert [c for c in calls if c[0] == "scalping"] == [("scalping", NOW), ("scalping", NOW + 1801)]


def test_run_once_ai_cycle_writes_busy_heartbeat_before_running(conn, monkeypatch):
    from runtime.heartbeat import read_heartbeat

    monkeypatch.setattr(market_data_engine, "collect_snapshot", lambda symbol, now=None: _snapshot("GOOD"))
    monkeypatch.setattr(paper_trading, "process_tick", lambda c, symbol, now=None: None)
    seen_activity = []

    def _check_heartbeat_mid_cycle(c, symbol, mode, *, now=None):
        hb = read_heartbeat(config.RUNTIME_HEARTBEAT_FILE)
        seen_activity.append(hb.get("activity") if hb else None)
        return ai_cycle.AICycleResult(ran=True)

    monkeypatch.setattr(ai_cycle, "run_ai_cycle", _check_heartbeat_mid_cycle)

    runtime = TradingRuntime("ETHUSDT", conn=conn)
    runtime.run_once(now=NOW)

    # both modes fire on this cold-start tick, each writing its own
    # in-flight "ai_cycle" heartbeat before the (mocked) call.
    assert seen_activity == ["ai_cycle", "ai_cycle"]
    # the FINAL heartbeat (written after the cycle completes) no longer
    # carries the in-flight marker
    final_hb = read_heartbeat(config.RUNTIME_HEARTBEAT_FILE)
    assert final_hb.get("activity") is None


def test_run_once_survives_ai_cycle_exception(conn, monkeypatch):
    monkeypatch.setattr(market_data_engine, "collect_snapshot", lambda symbol, now=None: _snapshot("GOOD"))
    monkeypatch.setattr(paper_trading, "process_tick", lambda c, symbol, now=None: None)
    monkeypatch.setattr(
        ai_cycle, "run_ai_cycle", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom"))
    )

    runtime = TradingRuntime("ETHUSDT", conn=conn)
    result = runtime.run_once(now=NOW)
    # one exception per mode -- a bad cycle for one mode must not stop the
    # others from being attempted.
    assert sum("ai_cycle failed" in e for e in result.errors) == 2
    assert result.ai_cycle_results == {}


def test_run_once_trigger_file_bypasses_the_interval(conn, monkeypatch):
    # swing's own horizon is far longer than what's exercised here, so this
    # test tracks scalping's calls specifically (same convention as
    # test_run_once_respects_ai_cycle_interval above).
    monkeypatch.setattr(config, "RUNTIME_AI_CYCLE_INTERVAL_SECONDS", 1800.0)
    monkeypatch.setattr(market_data_engine, "collect_snapshot", lambda symbol, now=None: _snapshot("GOOD"))
    monkeypatch.setattr(paper_trading, "process_tick", lambda c, symbol, now=None: None)
    calls = []
    monkeypatch.setattr(
        ai_cycle, "run_ai_cycle",
        lambda c, symbol, mode, *, now=None: calls.append((mode, now)) or ai_cycle.AICycleResult(ran=True),
    )

    runtime = TradingRuntime("ETHUSDT", conn=conn)
    runtime.run_once(now=NOW)  # first tick always runs every mode regardless
    assert [c for c in calls if c[0] == "scalping"] == [("scalping", NOW)]

    runtime.run_once(now=NOW + 60)  # well within the 1800s interval, no trigger
    assert [c for c in calls if c[0] == "scalping"] == [("scalping", NOW)]

    config.RUNTIME_AI_CYCLE_TRIGGER_FILE.touch()
    runtime.run_once(now=NOW + 120)  # still within the interval, but triggered
    assert [c for c in calls if c[0] == "scalping"] == [("scalping", NOW), ("scalping", NOW + 120)]


def test_run_once_trigger_file_is_consumed_after_use(conn, monkeypatch):
    monkeypatch.setattr(market_data_engine, "collect_snapshot", lambda symbol, now=None: _snapshot("GOOD"))
    monkeypatch.setattr(paper_trading, "process_tick", lambda c, symbol, now=None: None)
    monkeypatch.setattr(ai_cycle, "run_ai_cycle", lambda *a, **k: ai_cycle.AICycleResult(ran=True))

    config.RUNTIME_AI_CYCLE_TRIGGER_FILE.touch()
    runtime = TradingRuntime("ETHUSDT", conn=conn)
    runtime.run_once(now=NOW)

    assert not config.RUNTIME_AI_CYCLE_TRIGGER_FILE.exists()


def test_run_once_trigger_file_left_alone_when_position_open(conn, monkeypatch):
    monkeypatch.setattr(market_data_engine, "collect_snapshot", lambda symbol, now=None: _snapshot("GOOD"))
    monkeypatch.setattr(paper_trading, "process_tick", lambda c, symbol, now=None: None)
    monkeypatch.setattr(
        ai_cycle, "run_ai_cycle",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("ai_cycle should not run with an open PAPER position")),
    )
    journal_db.insert_position(
        conn, symbol="ETHUSDT", side="LONG", source="PAPER", entry_price="100", quantity="1", now=NOW
    )
    conn.commit()

    config.RUNTIME_AI_CYCLE_TRIGGER_FILE.touch()
    runtime = TradingRuntime("ETHUSDT", conn=conn)
    runtime.run_once(now=NOW)

    # still queued -- the request stays pending until a position frees up,
    # it isn't silently dropped just because now wasn't a good moment.
    assert config.RUNTIME_AI_CYCLE_TRIGGER_FILE.exists()


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
