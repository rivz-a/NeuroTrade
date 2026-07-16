"""Offline tests for execution_engine.py — no network. Focused on the
global gates (kill switch, daily loss limit, max trades/day, cooldown)
that run BEFORE order_manager.place_entry_order is ever called; the
per-trade checks themselves are covered by test_order_manager.py.
"""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

import pytest

import bingx_client
import config
import execution_engine
import journal_db
import order_manager
import position_manager

NOW = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
NOW_EPOCH = NOW.timestamp()


@pytest.fixture(autouse=True)
def _use_tmp_db(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "JOURNAL_DB_FILE", tmp_path / "execution_engine_test.db")
    monkeypatch.setattr(config, "EXECUTION_KILL_SWITCH_FILE", tmp_path / "kill_switch.flag")


@pytest.fixture(autouse=True)
def _credentials(monkeypatch):
    monkeypatch.setattr(config, "BINGX_API_KEY", "test-api-key")
    monkeypatch.setattr(config, "BINGX_API_SECRET", "test-api-secret")
    monkeypatch.setattr(config, "EXECUTION_DRY_RUN", True)


@pytest.fixture(autouse=True)
def _current_price(monkeypatch):
    monkeypatch.setattr(bingx_client, "get_price", lambda symbol: 100.5)


@pytest.fixture
def conn(_use_tmp_db):
    c = journal_db.init_db()
    yield c
    c.close()


def _minimal_trade_plan(conn, *, symbol="ETHUSDT", trade_permission="ALLOWED") -> int:
    import pandas as pd

    from feature_engine import (
        FeatureSet, FuturesFeatures, MomentumFeatures, OrderbookFeatures,
        TradeFlowFeatures, TrendFeatures, VolatilityFeatures, VolumeFeatures,
    )
    from market_data_engine import MarketDataSnapshot
    from market_regime import RegimeResult
    from strategy_engine import Contribution, ScoreResult
    from consensus_engine import ConsensusResult, SelectedPlan
    from risk_manager import DEFAULT_RISK_SETTINGS, PositionCalculator, TakeProfitTarget, TradeScenario

    df = pd.DataFrame([{"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5, "volume": 10.0, "time": NOW}])
    snapshot = MarketDataSnapshot(
        timestamp=NOW, symbol=symbol, price=100.5, bid=100.4, ask=100.6, spread=0.2, spread_percent=0.2,
        timeframes={"1m": {"candles": [], "has_gap": False, "is_stale": False, "zero_volume": False, "dataframe": df}},
        funding_rate=0.0001, funding_history=[], open_interest=123456.0, open_interest_history=[],
        orderbook=None, orderbook_history=[], volume_24h=999.0, recent_trades=[], instrument_rules=None,
        data_quality="GOOD", quality_issues=[],
    )
    features = FeatureSet(
        timestamp=NOW, symbol=symbol, data_quality="GOOD",
        trend={"1m": TrendFeatures(0.5, 1.0, 2.0, 0.1, 0.3, 0.4, 25.0, "UP", "BULLISH", True, True, False, False, False, False, "BULLISH")},
        momentum={"1m": MomentumFeatures(55.0, 1.0, 0.2, 0.01, 0.5, 1.2, False, False)},
        volatility={"1m": VolatilityFeatures(2.0, 60.0, 1.5, 0.8, False, False, 0.9)},
        volume={"1m": VolumeFeatures(1.1, False, "INCREASING")},
        trade_flow=TradeFlowFeatures(5.0, 3.0, 2.0, 2.0),
        futures=FuturesFeatures(0.0001, 0.00001, 123456.0, 100.0, 0.08, "NEW_LONGS"),
        orderbook=OrderbookFeatures(0.1, 0.12, 0.2, 0.01, 100.51, False),
        timeframe_alignment=0.75, distance_to_support_atr=0.5, distance_to_resistance_atr=1.2,
    )
    regime = RegimeResult(
        timestamp=NOW, symbol=symbol, regime="TREND_UP",
        regime_by_timeframe={"1m": "TREND_UP"}, reasons=["1m: TREND_UP"], strategy_hint="hint",
    )
    score = ScoreResult(
        timestamp=NOW, symbol=symbol, mode="scalping", ruleset_version="v1",
        long_score=68.0, short_score=31.0, no_trade_score=44.0, decision="LONG_BIAS", quality="B",
        contributions=[Contribution("timeframe_alignment", 12.0)],
    )
    snap_id = journal_db.insert_market_snapshot(conn, snapshot)
    fs_id = journal_db.insert_feature_snapshot(conn, features, market_snapshot_id=snap_id)
    score_id = journal_db.insert_strategy_score(conn, score, regime, feature_snapshot_id=fs_id)

    entry_from, entry_to, stop_loss = 100.0, 101.0, 98.0
    take_profits = [("TP1", 105.0, 100.0)]
    scenario = TradeScenario(
        signal="LONG", entry_from=Decimal(str(entry_from)), entry_to=Decimal(str(entry_to)),
        stop_loss=Decimal(str(stop_loss)),
        take_profits=[TakeProfitTarget(label, Decimal(str(price)), Decimal(str(pct))) for label, price, pct in take_profits],
    )
    calculation = PositionCalculator(DEFAULT_RISK_SETTINGS).calculate(scenario)
    plan = SelectedPlan(
        source_label="GPT-4o mini", entry_status="ENTER_NOW", entry_type="LIMIT_ZONE",
        entry_from=entry_from, entry_to=entry_to, stop_loss=stop_loss, take_profits=take_profits,
        risk_reward_tp1=2.0, time_horizon_minutes=30, valid_for_minutes=30, formed_at=NOW_EPOCH,
    )
    consensus = ConsensusResult(
        mode="scalping", overall_signal="LONG", state="strong", agreement_fraction=1.0, agreeing_count=3,
        vote_count=3, total_models=3, avg_confidence=70.0, plan=plan, trade_permission=trade_permission,
        trade_permission_reason="ok", reasons=["r"], risks=["k"], wait_or_invalidation=[],
    )
    return journal_db.insert_trade_plan(
        conn, consensus, strategy_score_id=score_id, symbol=symbol, timestamp=NOW_EPOCH, calculation=calculation,
    )


# ---------------------------------------------------------------------------
# kill switch
# ---------------------------------------------------------------------------


def test_kill_switch_engaged_false_when_file_absent():
    assert execution_engine.kill_switch_engaged() is False


def test_kill_switch_engaged_true_when_file_present():
    config.EXECUTION_KILL_SWITCH_FILE.parent.mkdir(parents=True, exist_ok=True)
    config.EXECUTION_KILL_SWITCH_FILE.write_text("stop")
    assert execution_engine.kill_switch_engaged() is True


def test_confirm_and_execute_blocked_by_kill_switch(conn):
    plan_id = _minimal_trade_plan(conn)
    config.EXECUTION_KILL_SWITCH_FILE.parent.mkdir(parents=True, exist_ok=True)
    config.EXECUTION_KILL_SWITCH_FILE.write_text("stop")

    result = execution_engine.confirm_and_execute(conn, plan_id, now=NOW_EPOCH)
    assert result.status == "REJECTED"
    assert "kill switch" in result.reason
    assert journal_db.get_real_order_by_trade_plan_id(conn, plan_id) is None


# ---------------------------------------------------------------------------
# daily loss limit / max trades / cooldown
# ---------------------------------------------------------------------------


def test_confirm_and_execute_blocked_by_daily_loss_limit(conn, monkeypatch):
    monkeypatch.setattr(config, "EXECUTION_DAILY_LOSS_LIMIT_USDT", 5.0)
    plan_id = _minimal_trade_plan(conn)

    position_id = journal_db.insert_position(
        conn, symbol="ETHUSDT", side="LONG", source="REAL", entry_price=Decimal("100"), quantity=Decimal("0.1"), status="CLOSED",
    )
    journal_db.insert_fill(
        conn, position_id=position_id, symbol="ETHUSDT", side="LONG", fill_type="STOP_LOSS",
        price=Decimal("98"), quantity=Decimal("0.1"), realized_pnl_usdt=Decimal("-6.0"), now=NOW_EPOCH,
    )

    result = execution_engine.confirm_and_execute(conn, plan_id, now=NOW_EPOCH)
    assert result.status == "REJECTED"
    assert "daily loss limit" in result.reason


def test_confirm_and_execute_blocked_by_max_trades_per_day(conn, monkeypatch):
    monkeypatch.setattr(config, "EXECUTION_MAX_TRADES_PER_DAY", 1)
    plan_id = _minimal_trade_plan(conn)
    other_plan_id = _minimal_trade_plan(conn)

    journal_db.insert_real_order(
        conn, symbol="ETHUSDT", side="LONG", order_type="MARKET", quantity="0.01",
        trade_plan_id=other_plan_id, status="OPEN", now=NOW_EPOCH,
    )

    result = execution_engine.confirm_and_execute(conn, plan_id, now=NOW_EPOCH)
    assert result.status == "REJECTED"
    assert "max trades" in result.reason


def test_confirm_and_execute_blocked_by_cooldown_after_stop(conn, monkeypatch):
    monkeypatch.setattr(config, "EXECUTION_COOLDOWN_AFTER_STOP_SECONDS", 1800)
    plan_id = _minimal_trade_plan(conn)

    position_id = journal_db.insert_position(
        conn, symbol="ETHUSDT", side="LONG", source="REAL", entry_price=Decimal("100"), quantity=Decimal("0.1"), status="CLOSED",
    )
    journal_db.insert_fill(
        conn, position_id=position_id, symbol="ETHUSDT", side="LONG", fill_type="STOP_LOSS",
        price=Decimal("98"), quantity=Decimal("0.1"), realized_pnl_usdt=Decimal("-1.0"), now=NOW_EPOCH - 60,
    )

    result = execution_engine.confirm_and_execute(conn, plan_id, now=NOW_EPOCH)
    assert result.status == "REJECTED"
    assert "cooldown" in result.reason


def test_confirm_and_execute_cooldown_expired_allows_execution(conn, monkeypatch):
    monkeypatch.setattr(config, "EXECUTION_COOLDOWN_AFTER_STOP_SECONDS", 1800)
    plan_id = _minimal_trade_plan(conn)

    position_id = journal_db.insert_position(
        conn, symbol="ETHUSDT", side="LONG", source="REAL", entry_price=Decimal("100"), quantity=Decimal("0.1"), status="CLOSED",
    )
    journal_db.insert_fill(
        conn, position_id=position_id, symbol="ETHUSDT", side="LONG", fill_type="STOP_LOSS",
        price=Decimal("98"), quantity=Decimal("0.1"), realized_pnl_usdt=Decimal("-1.0"), now=NOW_EPOCH - 3600,
    )

    result = execution_engine.confirm_and_execute(conn, plan_id, now=NOW_EPOCH)
    assert result.status == "DRY_RUN"


def test_confirm_and_execute_happy_path_reaches_order_manager(conn):
    plan_id = _minimal_trade_plan(conn)
    result = execution_engine.confirm_and_execute(conn, plan_id, now=NOW_EPOCH)
    assert result.status == "DRY_RUN"
    assert journal_db.get_real_order_by_trade_plan_id(conn, plan_id) is not None


def test_confirm_and_execute_rejects_unknown_trade_plan(conn):
    result = execution_engine.confirm_and_execute(conn, 999999, now=NOW_EPOCH)
    assert result.status == "REJECTED"
    assert "not found" in result.reason


# ---------------------------------------------------------------------------
# monitor / close_all delegation
# ---------------------------------------------------------------------------


def test_monitor_calls_sync_and_position_checks_for_each_open_position(conn, monkeypatch):
    calls = []
    monkeypatch.setattr(position_manager, "sync_position_status", lambda conn, symbol: calls.append(("sync", symbol)))
    monkeypatch.setattr(position_manager, "manage_stop_loss", lambda conn, pid: calls.append(("stop", pid)))
    monkeypatch.setattr(position_manager, "check_regime_close", lambda conn, pid: calls.append(("regime", pid)))
    monkeypatch.setattr(position_manager, "check_data_quality_close", lambda conn, pid: calls.append(("data", pid)))
    monkeypatch.setattr(position_manager, "check_time_based_close", lambda conn, pid: calls.append(("close", pid)))

    position_id = journal_db.insert_position(
        conn, symbol="ETHUSDT", side="LONG", source="REAL", entry_price=Decimal("100"), quantity=Decimal("0.1"), status="OPEN",
    )

    execution_engine.monitor(conn, "ETHUSDT")
    assert ("sync", "ETHUSDT") in calls
    assert ("stop", position_id) in calls
    assert ("regime", position_id) in calls
    assert ("data", position_id) in calls
    assert ("close", position_id) in calls


def test_monitor_checks_entry_fill_for_pending_entry_orders(conn, monkeypatch):
    monkeypatch.setattr(position_manager, "sync_position_status", lambda conn, symbol: None)
    calls = []
    monkeypatch.setattr(position_manager, "check_entry_fill", lambda conn, plan_id: calls.append(("check", plan_id)) or False)
    monkeypatch.setattr(position_manager, "cancel_stale_entry_order", lambda conn, plan_id: calls.append(("cancel", plan_id)))

    plan_id = _minimal_trade_plan(conn)
    journal_db.insert_real_order(
        conn, symbol="ETHUSDT", side="LONG", order_type="LIMIT", quantity="0.1", trade_plan_id=plan_id, status="OPEN",
    )

    execution_engine.monitor(conn, "ETHUSDT")
    assert ("check", plan_id) in calls
    assert ("cancel", plan_id) in calls  # check_entry_fill returned False -> cancel is attempted


def test_monitor_skips_cancel_when_entry_fill_confirmed(conn, monkeypatch):
    monkeypatch.setattr(position_manager, "sync_position_status", lambda conn, symbol: None)
    calls = []
    monkeypatch.setattr(position_manager, "check_entry_fill", lambda conn, plan_id: calls.append(("check", plan_id)) or True)
    monkeypatch.setattr(position_manager, "cancel_stale_entry_order", lambda conn, plan_id: calls.append(("cancel", plan_id)))

    plan_id = _minimal_trade_plan(conn)
    journal_db.insert_real_order(
        conn, symbol="ETHUSDT", side="LONG", order_type="LIMIT", quantity="0.1", trade_plan_id=plan_id, status="OPEN",
    )

    execution_engine.monitor(conn, "ETHUSDT")
    assert ("check", plan_id) in calls
    assert ("cancel", plan_id) not in calls


def test_close_all_delegates_to_order_manager(conn, monkeypatch):
    sentinel = order_manager.CloseAllResult(symbol="ETHUSDT", orders_cancelled=[1, 2], position_closed=True, errors=[])
    monkeypatch.setattr(order_manager, "close_all", lambda conn, symbol: sentinel)
    result = execution_engine.close_all(conn, "ETHUSDT")
    assert result is sentinel
