"""Offline tests for position_manager.py — no network. bingx_client.get_price
and every bingx_private_client function are monkeypatched per test.
"""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

import pytest

import bingx_client
import bingx_private_client
import config
import journal_db
import position_manager

NOW = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
NOW_EPOCH = NOW.timestamp()


@pytest.fixture(autouse=True)
def _use_tmp_db(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "JOURNAL_DB_FILE", tmp_path / "position_manager_test.db")


@pytest.fixture(autouse=True)
def _credentials(monkeypatch):
    monkeypatch.setattr(config, "BINGX_API_KEY", "test-api-key")
    monkeypatch.setattr(config, "BINGX_API_SECRET", "test-api-secret")
    monkeypatch.setattr(config, "EXECUTION_DRY_RUN", False)  # tests mock bingx_private_client directly


@pytest.fixture
def conn(_use_tmp_db):
    c = journal_db.init_db()
    yield c
    c.close()


def _open_position(
    conn, *, side="LONG", entry_price="100.0", stop_loss="98.0", quantity="0.1",
    trade_plan_id=None, real_order_id=None, opened_at=NOW_EPOCH,
) -> int:
    return journal_db.insert_position(
        conn, symbol="ETHUSDT", side=side, source="REAL",
        entry_price=Decimal(entry_price), quantity=Decimal(quantity), stop_loss=Decimal(stop_loss),
        leverage=5, margin_mode="ISOLATED", trade_plan_id=trade_plan_id, real_order_id=real_order_id,
        status="OPEN", now=opened_at,
    )


# ---------------------------------------------------------------------------
# sync_position_status
# ---------------------------------------------------------------------------


def test_sync_position_status_leaves_open_position_when_still_open_on_exchange(conn, monkeypatch):
    position_id = _open_position(conn)
    monkeypatch.setattr(bingx_private_client, "get_positions", lambda symbol=None: [{"symbol": "ETHUSDT", "positionSide": "LONG", "positionAmt": "0.1"}])
    monkeypatch.setattr(bingx_private_client, "get_open_orders", lambda symbol=None: [])

    result = position_manager.sync_position_status(conn, "ETHUSDT")
    assert result.positions_closed == []
    assert journal_db.get_position(conn, position_id)["status"] == "OPEN"


def test_sync_position_status_detects_stop_loss_close(conn, monkeypatch):
    plan_id = _minimal_trade_plan(conn, mode="scalping")
    sl_order_id = journal_db.insert_real_order(
        conn, symbol="ETHUSDT", side="LONG", order_type="TRIGGER", quantity="0.1",
        stop_loss=Decimal("98.0"), trade_plan_id=plan_id, status="OPEN", exchange_order_id="sl-1",
    )
    tp_order_id = journal_db.insert_real_order(
        conn, symbol="ETHUSDT", side="LONG", order_type="TRIGGER", quantity="0.1",
        take_profit=Decimal("105.0"), trade_plan_id=plan_id, status="OPEN", exchange_order_id="tp-1",
    )
    position_id = _open_position(conn, trade_plan_id=plan_id)

    monkeypatch.setattr(bingx_private_client, "get_positions", lambda symbol=None: [])
    # SL's exchange id is gone from open orders (it filled); TP's is still there.
    monkeypatch.setattr(bingx_private_client, "get_open_orders", lambda symbol=None: [{"orderId": "tp-1"}])

    result = position_manager.sync_position_status(conn, "ETHUSDT")
    assert result.positions_closed == [position_id]
    position = journal_db.get_position(conn, position_id)
    assert position["status"] == "CLOSED"

    fills = journal_db.get_position_fills(conn, position_id)
    assert any(f["fill_type"] == "STOP_LOSS" for f in fills)
    assert journal_db.update_order_status  # sanity: module imported correctly


def test_sync_position_status_detects_take_profit_close(conn, monkeypatch):
    plan_id = _minimal_trade_plan(conn, mode="scalping")
    journal_db.insert_real_order(
        conn, symbol="ETHUSDT", side="LONG", order_type="TRIGGER", quantity="0.1",
        stop_loss=Decimal("98.0"), trade_plan_id=plan_id, status="OPEN", exchange_order_id="sl-2",
    )
    journal_db.insert_real_order(
        conn, symbol="ETHUSDT", side="LONG", order_type="TRIGGER", quantity="0.1",
        take_profit=Decimal("105.0"), trade_plan_id=plan_id, status="OPEN", exchange_order_id="tp-2",
    )
    position_id = _open_position(conn, trade_plan_id=plan_id)

    monkeypatch.setattr(bingx_private_client, "get_positions", lambda symbol=None: [])
    # TP's exchange id is gone (it filled); SL's is still resting.
    monkeypatch.setattr(bingx_private_client, "get_open_orders", lambda symbol=None: [{"orderId": "sl-2"}])

    result = position_manager.sync_position_status(conn, "ETHUSDT")
    assert result.positions_closed == [position_id]
    fills = journal_db.get_position_fills(conn, position_id)
    assert any(f["fill_type"] == "TAKE_PROFIT" for f in fills)


def test_sync_position_status_detects_partial_take_profit_close(conn, monkeypatch):
    plan_id = _minimal_trade_plan(conn, mode="scalping")
    journal_db.insert_real_order(
        conn, symbol="ETHUSDT", side="LONG", order_type="TRIGGER", quantity="0.06",
        stop_loss=Decimal("98.0"), trade_plan_id=plan_id, status="OPEN", exchange_order_id="sl-3",
    )
    journal_db.insert_real_order(
        conn, symbol="ETHUSDT", side="LONG", order_type="TRIGGER", quantity="0.04",
        take_profit=Decimal("104.0"), trade_plan_id=plan_id, status="OPEN", exchange_order_id="tp-3",
        label="TP1",
    )
    position_id = _open_position(conn, trade_plan_id=plan_id, quantity="0.1")

    # TP1 (0.04) filled and dropped off BingX's open orders; 0.06 remains open on the exchange.
    monkeypatch.setattr(bingx_private_client, "get_positions", lambda symbol=None: [{"symbol": "ETHUSDT", "positionSide": "LONG", "positionAmt": "0.06"}])
    monkeypatch.setattr(bingx_private_client, "get_open_orders", lambda symbol=None: [{"orderId": "sl-3"}])

    result = position_manager.sync_position_status(conn, "ETHUSDT")
    assert result.positions_closed == []
    assert result.positions_partially_closed == [position_id]

    position = journal_db.get_position(conn, position_id)
    assert position["status"] == "OPEN"
    assert position["quantity"] == Decimal("0.06")

    fills = journal_db.get_position_fills(conn, position_id)
    tp_fills = [f for f in fills if f["fill_type"] == "TAKE_PROFIT"]
    assert len(tp_fills) == 1
    assert tp_fills[0]["quantity"] == Decimal("0.04")
    assert tp_fills[0]["label"] == "TP1"


def test_sync_position_status_reports_error_on_bingx_failure(conn, monkeypatch):
    _open_position(conn)
    monkeypatch.setattr(
        bingx_private_client, "get_positions",
        lambda symbol=None: (_ for _ in ()).throw(bingx_client.NetworkError("down")),
    )
    result = position_manager.sync_position_status(conn, "ETHUSDT")
    assert result.discrepancies
    assert journal_db.get_open_positions(conn, "ETHUSDT", source="REAL")  # untouched


# ---------------------------------------------------------------------------
# manage_stop_loss
# ---------------------------------------------------------------------------


def _add_tp_fill(conn, position_id, *, symbol="ETHUSDT", side="LONG", label="TP1", price="104.0", quantity="0.04"):
    journal_db.insert_fill(
        conn, position_id=position_id, symbol=symbol, side=side, fill_type="TAKE_PROFIT",
        label=label, price=Decimal(price), quantity=Decimal(quantity),
    )


def test_manage_stop_loss_no_op_when_no_tp_filled_yet(conn, monkeypatch):
    position_id = _open_position(conn, side="LONG", entry_price="100.0", stop_loss="98.0")
    monkeypatch.setattr(bingx_client, "get_price", lambda symbol: 105.0)
    result = position_manager.manage_stop_loss(conn, position_id)
    assert result is None
    assert journal_db.get_position(conn, position_id)["stop_loss"] == Decimal("98.0")


def test_manage_stop_loss_moves_to_breakeven_after_first_tp(conn, monkeypatch):
    position_id = _open_position(conn, side="LONG", entry_price="100.0", stop_loss="98.0")
    _add_tp_fill(conn, position_id)
    monkeypatch.setattr(bingx_private_client, "cancel_order", lambda *a, **k: {})
    monkeypatch.setattr(bingx_private_client, "place_order", lambda *a, **k: {"orderId": "sl-new"})

    result = position_manager.manage_stop_loss(conn, position_id)
    assert result == "BREAKEVEN"
    assert journal_db.get_position(conn, position_id)["stop_loss"] == Decimal("100.0")


def test_manage_stop_loss_no_op_when_breakeven_already_applied(conn):
    position_id = _open_position(conn, side="LONG", entry_price="100.0", stop_loss="100.0")
    _add_tp_fill(conn, position_id)
    result = position_manager.manage_stop_loss(conn, position_id)
    assert result is None


def test_manage_stop_loss_trails_after_second_tp(conn, monkeypatch):
    # trade_plan_id anchors the ORIGINAL stop (98.0) for the risk calculation —
    # positions.stop_loss itself is already the (mutated) breakeven value.
    plan_id = _actionable_trade_plan(conn)
    position_id = _open_position(conn, side="LONG", entry_price="100.0", stop_loss="100.0", trade_plan_id=plan_id)
    _add_tp_fill(conn, position_id, label="TP1", price="104.0", quantity="0.04")
    _add_tp_fill(conn, position_id, label="TP2", price="107.0", quantity="0.03")
    monkeypatch.setattr(bingx_client, "get_price", lambda symbol: 108.0)  # risk=2.0 (entry 100 vs original stop 98)
    monkeypatch.setattr(bingx_private_client, "cancel_order", lambda *a, **k: {})
    monkeypatch.setattr(bingx_private_client, "place_order", lambda *a, **k: {"orderId": "sl-trail"})

    result = position_manager.manage_stop_loss(conn, position_id)
    assert result == "TRAILING"
    # candidate = 108 - (risk=2.0 * EXECUTION_TRAILING_STOP_R_MULTIPLE=1.0) = 106.0, better than 100.0
    assert journal_db.get_position(conn, position_id)["stop_loss"] == Decimal("106.0")


def test_manage_stop_loss_trailing_no_op_when_candidate_not_better(conn, monkeypatch):
    plan_id = _actionable_trade_plan(conn)
    position_id = _open_position(conn, side="LONG", entry_price="100.0", stop_loss="106.0", trade_plan_id=plan_id)  # already trailed high
    _add_tp_fill(conn, position_id, label="TP1")
    _add_tp_fill(conn, position_id, label="TP2")
    monkeypatch.setattr(bingx_client, "get_price", lambda symbol: 107.5)  # candidate = 107.5 - risk(2.0) = 105.5 < 106.0
    result = position_manager.manage_stop_loss(conn, position_id)
    assert result is None
    assert journal_db.get_position(conn, position_id)["stop_loss"] == Decimal("106.0")


def test_manage_stop_loss_no_op_for_closed_position(conn):
    position_id = _open_position(conn)
    journal_db.close_position(conn, position_id, exit_price=Decimal("99.0"))
    assert position_manager.manage_stop_loss(conn, position_id) is None


def test_manage_stop_loss_short_side_breakeven(conn, monkeypatch):
    position_id = _open_position(conn, side="SHORT", entry_price="100.0", stop_loss="102.0")
    _add_tp_fill(conn, position_id, side="SHORT", label="TP1", price="96.0", quantity="0.04")
    monkeypatch.setattr(bingx_private_client, "cancel_order", lambda *a, **k: {})
    monkeypatch.setattr(bingx_private_client, "place_order", lambda *a, **k: {"orderId": "sl-new"})

    result = position_manager.manage_stop_loss(conn, position_id)
    assert result == "BREAKEVEN"
    assert journal_db.get_position(conn, position_id)["stop_loss"] == Decimal("100.0")


# ---------------------------------------------------------------------------
# check_time_based_close
# ---------------------------------------------------------------------------


def test_check_time_based_close_no_op_before_max_hold(conn):
    snap_id = None
    # Build a minimal trade_plan so mode is resolvable.
    plan_id = _minimal_trade_plan(conn, mode="scalping")
    position_id = _open_position(conn, trade_plan_id=plan_id, opened_at=NOW_EPOCH)
    result = position_manager.check_time_based_close(conn, position_id)
    assert result is False


def test_check_time_based_close_closes_after_max_hold(conn, monkeypatch):
    plan_id = _minimal_trade_plan(conn, mode="scalping")
    opened_at = NOW_EPOCH - config.PAPER_TRADING_MAX_HOLD_SECONDS["scalping"] - 10
    position_id = _open_position(conn, trade_plan_id=plan_id, opened_at=opened_at)

    monkeypatch.setattr(bingx_client, "get_price", lambda symbol: 101.0)
    monkeypatch.setattr(bingx_private_client, "place_order", lambda *a, **k: {"orderId": "close-1"})

    # Freeze "now" as seen by time.time() inside the module under test.
    monkeypatch.setattr(position_manager.time, "time", lambda: NOW_EPOCH)

    result = position_manager.check_time_based_close(conn, position_id)
    assert result is True
    assert journal_db.get_position(conn, position_id)["status"] == "CLOSED"
    fills = journal_db.get_position_fills(conn, position_id)
    assert any(f["fill_type"] == "TIMEOUT" for f in fills)


# ---------------------------------------------------------------------------
# check_regime_close
# ---------------------------------------------------------------------------


def _regime(name: str):
    from market_regime import RegimeResult

    return RegimeResult(
        timestamp=NOW, symbol="ETHUSDT", regime=name,
        regime_by_timeframe={"1m": name}, reasons=[f"1m: {name}"], strategy_hint="hint",
    )


def test_check_regime_close_long_closes_on_trend_down(conn, monkeypatch):
    position_id = _open_position(conn, side="LONG")
    monkeypatch.setattr(bingx_client, "get_price", lambda symbol: 99.0)
    monkeypatch.setattr(bingx_private_client, "place_order", lambda *a, **k: {"orderId": "close-1"})

    result = position_manager.check_regime_close(conn, position_id, current_regime=_regime("TREND_DOWN"))
    assert result is True
    assert journal_db.get_position(conn, position_id)["status"] == "CLOSED"
    fills = journal_db.get_position_fills(conn, position_id)
    assert any(f["fill_type"] == "REGIME_CHANGE" for f in fills)


def test_check_regime_close_long_ignores_range(conn):
    position_id = _open_position(conn, side="LONG")
    result = position_manager.check_regime_close(conn, position_id, current_regime=_regime("RANGE"))
    assert result is False
    assert journal_db.get_position(conn, position_id)["status"] == "OPEN"


def test_check_regime_close_short_closes_on_trend_up(conn, monkeypatch):
    position_id = _open_position(conn, side="SHORT")
    monkeypatch.setattr(bingx_client, "get_price", lambda symbol: 101.0)
    monkeypatch.setattr(bingx_private_client, "place_order", lambda *a, **k: {"orderId": "close-1"})

    result = position_manager.check_regime_close(conn, position_id, current_regime=_regime("TREND_UP"))
    assert result is True


def test_check_regime_close_reversal_risk_closes_either_side(conn, monkeypatch):
    long_id = _open_position(conn, side="LONG")
    monkeypatch.setattr(bingx_client, "get_price", lambda symbol: 100.0)
    monkeypatch.setattr(bingx_private_client, "place_order", lambda *a, **k: {"orderId": "close-1"})
    assert position_manager.check_regime_close(conn, long_id, current_regime=_regime("REVERSAL_RISK")) is True


def test_check_regime_close_no_op_for_closed_position(conn):
    position_id = _open_position(conn)
    journal_db.close_position(conn, position_id, exit_price=Decimal("99.0"))
    assert position_manager.check_regime_close(conn, position_id, current_regime=_regime("TREND_DOWN")) is False


# ---------------------------------------------------------------------------
# check_data_quality_close
# ---------------------------------------------------------------------------


def test_check_data_quality_close_closes_on_no_trade(conn, monkeypatch):
    position_id = _open_position(conn)
    monkeypatch.setattr(bingx_client, "get_price", lambda symbol: 99.0)
    monkeypatch.setattr(bingx_private_client, "place_order", lambda *a, **k: {"orderId": "close-1"})

    result = position_manager.check_data_quality_close(conn, position_id, current_data_quality="NO_TRADE")
    assert result is True
    fills = journal_db.get_position_fills(conn, position_id)
    assert any(f["fill_type"] == "DATA_QUALITY" for f in fills)


def test_check_data_quality_close_no_op_on_good(conn):
    position_id = _open_position(conn)
    result = position_manager.check_data_quality_close(conn, position_id, current_data_quality="GOOD")
    assert result is False


def test_check_data_quality_close_no_op_on_degraded(conn):
    position_id = _open_position(conn)
    result = position_manager.check_data_quality_close(conn, position_id, current_data_quality="DEGRADED")
    assert result is False


# ---------------------------------------------------------------------------
# check_entry_fill / cancel_stale_entry_order
# ---------------------------------------------------------------------------


def _actionable_trade_plan(conn, *, side="LONG", valid_for_minutes=30, formed_at=NOW_EPOCH) -> int:
    import pandas as pd

    from consensus_engine import ConsensusResult, SelectedPlan
    from feature_engine import (
        FeatureSet, FuturesFeatures, MomentumFeatures, OrderbookFeatures,
        TradeFlowFeatures, TrendFeatures, VolatilityFeatures, VolumeFeatures,
    )
    from market_data_engine import MarketDataSnapshot
    from market_regime import RegimeResult
    from risk_manager import DEFAULT_RISK_SETTINGS, PositionCalculator, TakeProfitTarget, TradeScenario
    from strategy_engine import Contribution, ScoreResult

    df = pd.DataFrame([{"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5, "volume": 10.0, "time": NOW}])
    snapshot = MarketDataSnapshot(
        timestamp=NOW, symbol="ETHUSDT", price=100.5, bid=100.4, ask=100.6, spread=0.2, spread_percent=0.2,
        timeframes={"1m": {"candles": [], "has_gap": False, "is_stale": False, "zero_volume": False, "dataframe": df}},
        funding_rate=0.0001, funding_history=[], open_interest=123456.0, open_interest_history=[],
        orderbook=None, orderbook_history=[], volume_24h=999.0, recent_trades=[], instrument_rules=None,
        data_quality="GOOD", quality_issues=[],
    )
    features = FeatureSet(
        timestamp=NOW, symbol="ETHUSDT", data_quality="GOOD",
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
        timestamp=NOW, symbol="ETHUSDT", regime="TREND_UP",
        regime_by_timeframe={"1m": "TREND_UP"}, reasons=["1m: TREND_UP"], strategy_hint="hint",
    )
    score = ScoreResult(
        timestamp=NOW, symbol="ETHUSDT", mode="scalping", ruleset_version="v1",
        long_score=68.0, short_score=31.0, no_trade_score=44.0, decision="LONG_BIAS", quality="B",
        contributions=[Contribution("timeframe_alignment", 12.0)],
    )
    snap_id = journal_db.insert_market_snapshot(conn, snapshot)
    fs_id = journal_db.insert_feature_snapshot(conn, features, market_snapshot_id=snap_id)
    score_id = journal_db.insert_strategy_score(conn, score, regime, feature_snapshot_id=fs_id)

    entry_from, entry_to, stop_loss = (100.0, 101.0, 98.0) if side == "LONG" else (100.0, 101.0, 103.0)
    tp_price = 105.0 if side == "LONG" else 96.0
    take_profits = [("TP1", tp_price, 100.0)]
    scenario = TradeScenario(
        signal=side, entry_from=Decimal(str(entry_from)), entry_to=Decimal(str(entry_to)),
        stop_loss=Decimal(str(stop_loss)),
        take_profits=[TakeProfitTarget(label, Decimal(str(price)), Decimal(str(pct))) for label, price, pct in take_profits],
    )
    calculation = PositionCalculator(DEFAULT_RISK_SETTINGS).calculate(scenario)
    plan = SelectedPlan(
        source_label="GPT-4o mini", entry_status="ENTER_NOW", entry_type="LIMIT_ZONE",
        entry_from=entry_from, entry_to=entry_to, stop_loss=stop_loss, take_profits=take_profits,
        risk_reward_tp1=2.0, time_horizon_minutes=30, valid_for_minutes=valid_for_minutes, formed_at=formed_at,
    )
    consensus = ConsensusResult(
        mode="scalping", overall_signal=side, state="strong", agreement_fraction=1.0, agreeing_count=3,
        vote_count=3, total_models=3, avg_confidence=70.0, plan=plan, trade_permission="ALLOWED",
        trade_permission_reason="ok", reasons=["r"], risks=["k"], wait_or_invalidation=[],
    )
    return journal_db.insert_trade_plan(
        conn, consensus, strategy_score_id=score_id, symbol="ETHUSDT", timestamp=formed_at, calculation=calculation,
    )


def test_check_entry_fill_creates_position_when_confirmed(conn, monkeypatch):
    plan_id = _actionable_trade_plan(conn)
    entry_order_id = journal_db.insert_real_order(
        conn, symbol="ETHUSDT", side="LONG", order_type="LIMIT", quantity="0.1",
        trade_plan_id=plan_id, status="OPEN", now=NOW_EPOCH,
    )
    journal_db.insert_real_order(
        conn, symbol="ETHUSDT", side="LONG", order_type="TRIGGER", quantity="0.1",
        stop_loss=Decimal("98.0"), trade_plan_id=plan_id, status="OPEN", now=NOW_EPOCH,
    )
    monkeypatch.setattr(
        bingx_private_client, "get_positions",
        lambda symbol=None: [{"symbol": "ETHUSDT", "positionSide": "LONG", "positionAmt": "0.1", "avgPrice": "100.4"}],
    )

    created = position_manager.check_entry_fill(conn, plan_id)
    assert created is True

    position = journal_db.get_position_by_trade_plan_id(conn, plan_id)
    assert position is not None
    assert position["entry_price"] == Decimal("100.4")
    assert position["quantity"] == Decimal("0.1")
    assert position["stop_loss"] == Decimal("98.0")

    entry_row = journal_db.get_entry_real_order_for_trade_plan(conn, plan_id)
    assert entry_row["status"] == "FILLED"


def test_check_entry_fill_no_op_when_still_unfilled(conn, monkeypatch):
    plan_id = _actionable_trade_plan(conn)
    journal_db.insert_real_order(
        conn, symbol="ETHUSDT", side="LONG", order_type="LIMIT", quantity="0.1",
        trade_plan_id=plan_id, status="OPEN", now=NOW_EPOCH,
    )
    monkeypatch.setattr(bingx_private_client, "get_positions", lambda symbol=None: [])
    assert position_manager.check_entry_fill(conn, plan_id) is False
    assert journal_db.get_position_by_trade_plan_id(conn, plan_id) is None


def test_check_entry_fill_no_op_when_position_already_exists(conn, monkeypatch):
    plan_id = _actionable_trade_plan(conn)
    journal_db.insert_real_order(
        conn, symbol="ETHUSDT", side="LONG", order_type="LIMIT", quantity="0.1",
        trade_plan_id=plan_id, status="OPEN", now=NOW_EPOCH,
    )
    journal_db.insert_position(
        conn, symbol="ETHUSDT", side="LONG", source="REAL", entry_price=Decimal("100.4"),
        quantity=Decimal("0.1"), trade_plan_id=plan_id, status="OPEN", now=NOW_EPOCH,
    )
    called = {"n": 0}
    monkeypatch.setattr(bingx_private_client, "get_positions", lambda symbol=None: (called.__setitem__("n", called["n"] + 1), [])[1])
    assert position_manager.check_entry_fill(conn, plan_id) is False
    assert called["n"] == 0  # short-circuited before ever calling BingX


def test_cancel_stale_entry_order_cancels_after_expiry(conn, monkeypatch):
    plan_id = _actionable_trade_plan(conn, valid_for_minutes=5, formed_at=NOW_EPOCH)
    journal_db.insert_real_order(
        conn, symbol="ETHUSDT", side="LONG", order_type="LIMIT", quantity="0.1",
        trade_plan_id=plan_id, status="OPEN", now=NOW_EPOCH, exchange_order_id="entry-1",
    )
    monkeypatch.setattr(bingx_private_client, "cancel_order", lambda *a, **k: {})

    result = position_manager.cancel_stale_entry_order(conn, plan_id, now=NOW_EPOCH + 10 * 60)
    assert result is True
    entry_row = journal_db.get_entry_real_order_for_trade_plan(conn, plan_id)
    assert entry_row["status"] == "CANCELLED"


def test_cancel_stale_entry_order_no_op_before_expiry(conn):
    plan_id = _actionable_trade_plan(conn, valid_for_minutes=30, formed_at=NOW_EPOCH)
    journal_db.insert_real_order(
        conn, symbol="ETHUSDT", side="LONG", order_type="LIMIT", quantity="0.1",
        trade_plan_id=plan_id, status="OPEN", now=NOW_EPOCH,
    )
    result = position_manager.cancel_stale_entry_order(conn, plan_id, now=NOW_EPOCH + 60)
    assert result is False


def test_cancel_stale_entry_order_no_op_when_position_exists(conn):
    plan_id = _actionable_trade_plan(conn, valid_for_minutes=5, formed_at=NOW_EPOCH)
    journal_db.insert_real_order(
        conn, symbol="ETHUSDT", side="LONG", order_type="LIMIT", quantity="0.1",
        trade_plan_id=plan_id, status="OPEN", now=NOW_EPOCH,
    )
    journal_db.insert_position(
        conn, symbol="ETHUSDT", side="LONG", source="REAL", entry_price=Decimal("100.4"),
        quantity=Decimal("0.1"), trade_plan_id=plan_id, status="OPEN", now=NOW_EPOCH,
    )
    result = position_manager.cancel_stale_entry_order(conn, plan_id, now=NOW_EPOCH + 10 * 60)
    assert result is False


def _minimal_trade_plan(conn, *, mode: str) -> int:
    import pandas as pd

    from feature_engine import (
        FeatureSet, FuturesFeatures, MomentumFeatures, OrderbookFeatures,
        TradeFlowFeatures, TrendFeatures, VolatilityFeatures, VolumeFeatures,
    )
    from market_data_engine import MarketDataSnapshot
    from market_regime import RegimeResult
    from strategy_engine import Contribution, ScoreResult
    from consensus_engine import ConsensusResult

    df = pd.DataFrame([{"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5, "volume": 10.0, "time": NOW}])
    snapshot = MarketDataSnapshot(
        timestamp=NOW, symbol="ETHUSDT", price=100.5, bid=100.4, ask=100.6, spread=0.2, spread_percent=0.2,
        timeframes={"1m": {"candles": [], "has_gap": False, "is_stale": False, "zero_volume": False, "dataframe": df}},
        funding_rate=0.0001, funding_history=[], open_interest=123456.0, open_interest_history=[],
        orderbook=None, orderbook_history=[], volume_24h=999.0, recent_trades=[], instrument_rules=None,
        data_quality="GOOD", quality_issues=[],
    )
    features = FeatureSet(
        timestamp=NOW, symbol="ETHUSDT", data_quality="GOOD",
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
        timestamp=NOW, symbol="ETHUSDT", regime="TREND_UP",
        regime_by_timeframe={"1m": "TREND_UP"}, reasons=["1m: TREND_UP"], strategy_hint="hint",
    )
    score = ScoreResult(
        timestamp=NOW, symbol="ETHUSDT", mode=mode, ruleset_version="v1",
        long_score=68.0, short_score=31.0, no_trade_score=44.0, decision="LONG_BIAS", quality="B",
        contributions=[Contribution("timeframe_alignment", 12.0)],
    )
    snap_id = journal_db.insert_market_snapshot(conn, snapshot)
    fs_id = journal_db.insert_feature_snapshot(conn, features, market_snapshot_id=snap_id)
    score_id = journal_db.insert_strategy_score(conn, score, regime, feature_snapshot_id=fs_id)
    consensus = ConsensusResult(
        mode=mode, overall_signal="WAIT", state="strong", agreement_fraction=1.0, agreeing_count=3,
        vote_count=3, total_models=3, avg_confidence=70.0, plan=None, trade_permission="WAIT",
        trade_permission_reason="ok", reasons=["r"], risks=["k"], wait_or_invalidation=[],
    )
    return journal_db.insert_trade_plan(conn, consensus, strategy_score_id=score_id, symbol="ETHUSDT", timestamp=NOW_EPOCH)


# ---------------------------------------------------------------------------
# reconcile_state / recover_after_restart
# ---------------------------------------------------------------------------


def test_reconcile_state_finds_local_position_missing_on_exchange(conn, monkeypatch):
    position_id = _open_position(conn)
    monkeypatch.setattr(bingx_private_client, "get_positions", lambda symbol=None: [])
    monkeypatch.setattr(bingx_private_client, "get_open_orders", lambda symbol=None: [])

    report = position_manager.reconcile_state(conn, "ETHUSDT")
    assert report.local_open_not_on_exchange == [position_id]
    assert report.exchange_open_not_local == []


def test_reconcile_state_finds_exchange_order_missing_locally(conn, monkeypatch):
    monkeypatch.setattr(bingx_private_client, "get_positions", lambda symbol=None: [])
    monkeypatch.setattr(bingx_private_client, "get_open_orders", lambda symbol=None: [{"orderId": "ghost-1"}])

    report = position_manager.reconcile_state(conn, "ETHUSDT")
    assert report.exchange_open_not_local == ["ghost-1"]


def test_reconcile_state_clean_when_everything_matches(conn, monkeypatch):
    position_id = _open_position(conn)
    monkeypatch.setattr(bingx_private_client, "get_positions", lambda symbol=None: [{"symbol": "ETHUSDT", "positionSide": "LONG", "positionAmt": "0.1"}])
    monkeypatch.setattr(bingx_private_client, "get_open_orders", lambda symbol=None: [])

    report = position_manager.reconcile_state(conn, "ETHUSDT")
    assert report.local_open_not_on_exchange == []
    assert report.exchange_open_not_local == []


def test_reconcile_state_reports_error_on_bingx_failure(conn, monkeypatch):
    monkeypatch.setattr(
        bingx_private_client, "get_positions",
        lambda symbol=None: (_ for _ in ()).throw(bingx_client.APIError("boom")),
    )
    report = position_manager.reconcile_state(conn, "ETHUSDT")
    assert report.errors


def test_recover_after_restart_delegates_to_reconcile_state(conn, monkeypatch):
    monkeypatch.setattr(bingx_private_client, "get_positions", lambda symbol=None: [])
    monkeypatch.setattr(bingx_private_client, "get_open_orders", lambda symbol=None: [])
    report = position_manager.recover_after_restart(conn, "ETHUSDT")
    assert report.symbol == "ETHUSDT"
