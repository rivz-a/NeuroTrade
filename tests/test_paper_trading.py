"""Offline tests for paper_trading.py — no network, no AI. bid/ask/bid_qty/
ask_qty/settings are always injected explicitly into process_tick (its own
dependency-injection design), so bingx_client/risk_settings_store never need
mocking. All I/O redirected to a pytest tmp_path via config.JOURNAL_DB_FILE.

Self-contained fixture factories (no cross-import from test_journal_db.py —
matches this repo's per-file-duplication test convention).
"""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

import pandas as pd
import pytest

import config
import journal_db
import paper_trading
from consensus_engine import ConsensusResult, SelectedPlan
from feature_engine import (
    FeatureSet,
    FuturesFeatures,
    MomentumFeatures,
    OrderbookFeatures,
    TradeFlowFeatures,
    TrendFeatures,
    VolatilityFeatures,
    VolumeFeatures,
)
from market_data_engine import MarketDataSnapshot
from market_regime import RegimeResult
from outcome_simulator import SimulatedOutcome
from risk_manager import DEFAULT_RISK_SETTINGS, PositionCalculator, TakeProfitTarget, TradeScenario
from strategy_engine import Contribution, ScoreResult

NOW = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
NOW_EPOCH = NOW.timestamp()


@pytest.fixture(autouse=True)
def _use_tmp_db(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "JOURNAL_DB_FILE", tmp_path / "paper_trading_test.db")


@pytest.fixture
def conn(_use_tmp_db):
    c = journal_db.init_db()
    yield c
    c.close()


# ---------------------------------------------------------------------------
# Fixture factories
# ---------------------------------------------------------------------------


def _market_snapshot() -> MarketDataSnapshot:
    df = pd.DataFrame([{"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5, "volume": 10.0, "time": NOW}])
    timeframes = {
        "1m": {
            "candles": [{"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5, "volume": 10.0, "time": NOW}],
            "has_gap": False, "is_stale": False, "zero_volume": False, "dataframe": df,
        }
    }
    return MarketDataSnapshot(
        timestamp=NOW, symbol="ETHUSDT", price=100.5, bid=100.4, ask=100.6, spread=0.2, spread_percent=0.2,
        timeframes=timeframes, funding_rate=0.0001, funding_history=[], open_interest=123456.0,
        open_interest_history=[], orderbook=None, orderbook_history=[], volume_24h=999.0, recent_trades=[],
        instrument_rules=None, data_quality="GOOD", quality_issues=[],
    )


def _feature_set() -> FeatureSet:
    trend = {"1m": TrendFeatures(0.5, 1.0, 2.0, 0.1, 0.3, 0.4, 25.0, "UP", "BULLISH", True, True, False, False, False, False, "BULLISH")}
    momentum = {"1m": MomentumFeatures(55.0, 1.0, 0.2, 0.01, 0.5, 1.2, False, False)}
    volatility = {"1m": VolatilityFeatures(2.0, 60.0, 1.5, 0.8, False, False, 0.9)}
    volume = {"1m": VolumeFeatures(1.1, False, "INCREASING")}
    return FeatureSet(
        timestamp=NOW, symbol="ETHUSDT", data_quality="GOOD", trend=trend, momentum=momentum,
        volatility=volatility, volume=volume, trade_flow=TradeFlowFeatures(5.0, 3.0, 2.0, 2.0),
        futures=FuturesFeatures(0.0001, 0.00001, 123456.0, 100.0, 0.08, "NEW_LONGS"),
        orderbook=OrderbookFeatures(0.1, 0.12, 0.2, 0.01, 100.51, False),
        timeframe_alignment=0.75, distance_to_support_atr=0.5, distance_to_resistance_atr=1.2,
    )


def _regime_result(**overrides) -> RegimeResult:
    defaults = dict(
        timestamp=NOW, symbol="ETHUSDT", regime="TREND_UP",
        regime_by_timeframe={"1m": "TREND_UP"}, reasons=["1m: TREND_UP"], strategy_hint="hint",
    )
    defaults.update(overrides)
    return RegimeResult(**defaults)


def _score_result(**overrides) -> ScoreResult:
    defaults = dict(
        timestamp=NOW, symbol="ETHUSDT", mode="scalping", ruleset_version="v1",
        long_score=68.0, short_score=31.0, no_trade_score=44.0, decision="LONG_BIAS", quality="B",
        contributions=[Contribution("timeframe_alignment", 12.0)],
    )
    defaults.update(overrides)
    return ScoreResult(**defaults)


def _open_trade_plan(
    conn,
    *,
    signal: str = "LONG",
    entry_from: float = 100.0,
    entry_to: float = 101.0,
    stop_loss: float = 98.0,
    take_profits: list[tuple[str, float, float]] | None = None,
    valid_for_minutes: int = 15,
    formed_at: float | None = None,
    mode: str = "scalping",
) -> int:
    """Full market_snapshot -> feature_snapshot -> strategy_score -> trade_plan
    chain, with a REAL risk_manager.PositionCalculator computing position
    sizing/status (matching this repo's established test pattern) so
    trade_plans.position_status/position_size_coin_rounded/stop_loss_calc
    etc. are all realistic, not hand-waved.
    """
    if take_profits is None:
        take_profits = [("TP1", 105.0, 100.0)]
    formed_at = formed_at if formed_at is not None else NOW_EPOCH

    snap_id = journal_db.insert_market_snapshot(conn, _market_snapshot())
    fs_id = journal_db.insert_feature_snapshot(conn, _feature_set(), market_snapshot_id=snap_id)
    score_id = journal_db.insert_strategy_score(conn, _score_result(mode=mode), _regime_result(), feature_snapshot_id=fs_id)

    calculation = None
    plan = None
    if signal in ("LONG", "SHORT"):
        scenario = TradeScenario(
            signal=signal,
            entry_from=Decimal(str(entry_from)),
            entry_to=Decimal(str(entry_to)),
            stop_loss=Decimal(str(stop_loss)),
            take_profits=[TakeProfitTarget(label, Decimal(str(price)), Decimal(str(pct))) for label, price, pct in take_profits],
        )
        calculation = PositionCalculator(DEFAULT_RISK_SETTINGS).calculate(scenario)
        plan = SelectedPlan(
            source_label="GPT-4o mini", entry_status="ENTER_NOW", entry_type="LIMIT_ZONE",
            entry_from=entry_from, entry_to=entry_to, stop_loss=stop_loss,
            take_profits=list(take_profits), risk_reward_tp1=2.0,
            time_horizon_minutes=30, valid_for_minutes=valid_for_minutes, formed_at=formed_at,
        )

    consensus = ConsensusResult(
        mode=mode, overall_signal=signal, state="strong" if signal != "WAIT" else "strong",
        agreement_fraction=1.0, agreeing_count=3, vote_count=3, total_models=3, avg_confidence=70.0,
        plan=plan, trade_permission="ALLOWED" if signal in ("LONG", "SHORT") else "WAIT",
        trade_permission_reason="ok", reasons=["r"], risks=["k"], wait_or_invalidation=[],
    )

    return journal_db.insert_trade_plan(
        conn, consensus, strategy_score_id=score_id, symbol="ETHUSDT", timestamp=formed_at, calculation=calculation,
    )


def _tick(conn, *, bid, ask, bid_qty=1000.0, ask_qty=1000.0, now=NOW_EPOCH, settings=DEFAULT_RISK_SETTINGS):
    return paper_trading.process_tick(
        conn, "ETHUSDT", now=now, bid=bid, ask=ask, bid_qty=bid_qty, ask_qty=ask_qty, settings=settings
    )


# ---------------------------------------------------------------------------
# open_virtual_order
# ---------------------------------------------------------------------------


def test_open_virtual_order_creates_long_order(conn):
    plan_id = _open_trade_plan(conn, signal="LONG")
    result = paper_trading.open_virtual_order(conn, plan_id)
    assert result.status == "CREATED"
    order = journal_db.get_paper_order_by_trade_plan_id(conn, plan_id)
    assert order["side"] == "LONG"
    assert order["order_type"] == "LIMIT"
    assert order["entry_from"] == 100.0
    assert order["entry_to"] == 101.0
    assert order["stop_loss"] == Decimal("98.0")
    assert order["status"] == "PENDING"
    assert order["quantity"] > 0


def test_open_virtual_order_creates_short_order(conn):
    plan_id = _open_trade_plan(conn, signal="SHORT", entry_from=100.0, entry_to=101.0, stop_loss=103.0, take_profits=[("TP1", 95.0, 100.0)])
    result = paper_trading.open_virtual_order(conn, plan_id)
    assert result.status == "CREATED"
    order = journal_db.get_paper_order_by_trade_plan_id(conn, plan_id)
    assert order["side"] == "SHORT"
    assert order["stop_loss"] == Decimal("103.0")


def test_open_virtual_order_wait_signal_skipped(conn):
    plan_id = _open_trade_plan(conn, signal="WAIT")
    result = paper_trading.open_virtual_order(conn, plan_id)
    assert result.status == "SKIPPED_NOT_ACTIONABLE"
    assert result.order_id is None
    assert journal_db.get_paper_order_by_trade_plan_id(conn, plan_id) is None


def test_open_virtual_order_invalid_position_status_skipped(conn):
    plan_id = _open_trade_plan(conn, signal="LONG")
    conn.execute("UPDATE trade_plans SET position_status = 'POSITION_TOO_SMALL' WHERE id = ?", (plan_id,))
    result = paper_trading.open_virtual_order(conn, plan_id)
    assert result.status == "SKIPPED_NOT_ACTIONABLE"
    assert "POSITION_TOO_SMALL" in result.reason
    assert journal_db.get_paper_order_by_trade_plan_id(conn, plan_id) is None


def test_open_virtual_order_plan_not_found(conn):
    result = paper_trading.open_virtual_order(conn, 999999)
    assert result.status == "SKIPPED_NOT_ACTIONABLE"
    assert result.order_id is None


def test_open_virtual_order_idempotent(conn):
    plan_id = _open_trade_plan(conn, signal="LONG")
    first = paper_trading.open_virtual_order(conn, plan_id)
    second = paper_trading.open_virtual_order(conn, plan_id)
    assert first.status == "CREATED"
    assert second.status == "ALREADY_EXISTS"
    assert second.order_id == first.order_id
    count = conn.execute("SELECT COUNT(*) FROM paper_orders WHERE trade_plan_id = ?", (plan_id,)).fetchone()[0]
    assert count == 1


def test_open_virtual_order_waiting_trigger_still_creates_order(conn):
    plan_id = _open_trade_plan(conn, signal="LONG")
    conn.execute("UPDATE trade_plans SET trade_permission = 'WAITING_TRIGGER' WHERE id = ?", (plan_id,))
    result = paper_trading.open_virtual_order(conn, plan_id)
    assert result.status == "CREATED"


def test_open_virtual_order_price_outside_entry_zone_still_creates_order(conn):
    plan_id = _open_trade_plan(conn, signal="LONG")
    conn.execute("UPDATE trade_plans SET trade_permission = 'PRICE_OUTSIDE_ENTRY_ZONE' WHERE id = ?", (plan_id,))
    result = paper_trading.open_virtual_order(conn, plan_id)
    assert result.status == "CREATED"


def test_normalize_order_type():
    assert paper_trading._normalize_order_type("MARKET") == "MARKET"
    assert paper_trading._normalize_order_type("market") == "MARKET"
    assert paper_trading._normalize_order_type("WAIT_BREAKOUT") == "TRIGGER"
    assert paper_trading._normalize_order_type("LIMIT_ZONE") == "LIMIT"
    assert paper_trading._normalize_order_type(None) == "LIMIT"


# ---------------------------------------------------------------------------
# Entry fill
# ---------------------------------------------------------------------------


def test_long_entry_fills_when_ask_enters_zone(conn):
    plan_id = _open_trade_plan(conn, signal="LONG", entry_from=100.0, entry_to=101.0)
    paper_trading.open_virtual_order(conn, plan_id)
    result = _tick(conn, bid=100.4, ask=100.5)
    order = journal_db.get_paper_order_by_trade_plan_id(conn, plan_id)
    assert order["status"] == "FILLED"
    assert result.orders_filled == [order["id"]]
    positions = journal_db.get_open_positions(conn, "ETHUSDT")
    assert len(positions) == 1
    expected = Decimal("100.5") * (1 + DEFAULT_RISK_SETTINGS.slippage_percent / Decimal("100"))
    assert positions[0]["entry_price"] == expected


def test_short_entry_fills_when_bid_enters_zone(conn):
    plan_id = _open_trade_plan(conn, signal="SHORT", entry_from=100.0, entry_to=101.0, stop_loss=103.0, take_profits=[("TP1", 95.0, 100.0)])
    paper_trading.open_virtual_order(conn, plan_id)
    result = _tick(conn, bid=100.5, ask=100.6)
    order = journal_db.get_paper_order_by_trade_plan_id(conn, plan_id)
    assert order["status"] == "FILLED"
    positions = journal_db.get_open_positions(conn, "ETHUSDT")
    expected = Decimal("100.5") * (1 - DEFAULT_RISK_SETTINGS.slippage_percent / Decimal("100"))
    assert positions[0]["entry_price"] == expected


def test_long_entry_not_touched_stays_pending(conn):
    plan_id = _open_trade_plan(conn, signal="LONG", entry_from=100.0, entry_to=101.0)
    paper_trading.open_virtual_order(conn, plan_id)
    _tick(conn, bid=104.0, ask=105.0)  # price above the whole zone
    order = journal_db.get_paper_order_by_trade_plan_id(conn, plan_id)
    assert order["status"] == "PENDING"
    assert journal_db.get_open_positions(conn, "ETHUSDT") == []


def test_long_entry_partial_fill_by_liquidity(conn):
    plan_id = _open_trade_plan(conn, signal="LONG", entry_from=100.0, entry_to=101.0)
    paper_trading.open_virtual_order(conn, plan_id)
    order_before = journal_db.get_paper_order_by_trade_plan_id(conn, plan_id)
    small_qty = order_before["quantity"] / 2
    _tick(conn, bid=100.4, ask=100.5, ask_qty=float(small_qty))
    order = journal_db.get_paper_order_by_trade_plan_id(conn, plan_id)
    assert order["status"] == "PARTIALLY_FILLED"
    assert order["filled_quantity"] == small_qty
    positions = journal_db.get_open_positions(conn, "ETHUSDT")
    assert len(positions) == 1
    assert positions[0]["quantity"] == small_qty


def test_long_entry_partial_fill_across_two_ticks_weighted_average(conn):
    plan_id = _open_trade_plan(conn, signal="LONG", entry_from=100.0, entry_to=101.0)
    paper_trading.open_virtual_order(conn, plan_id)
    order_before = journal_db.get_paper_order_by_trade_plan_id(conn, plan_id)
    half = order_before["quantity"] / 2

    _tick(conn, bid=100.4, ask=100.0, ask_qty=float(half), now=NOW_EPOCH)
    _tick(conn, bid=100.4, ask=100.6, ask_qty=float(half), now=NOW_EPOCH + 1)

    order = journal_db.get_paper_order_by_trade_plan_id(conn, plan_id)
    assert order["status"] == "FILLED"
    positions = journal_db.get_open_positions(conn, "ETHUSDT")
    assert len(positions) == 1
    slip = DEFAULT_RISK_SETTINGS.slippage_percent / Decimal("100")
    price1 = Decimal("100.0") * (1 + slip)
    price2 = Decimal("100.6") * (1 + slip)
    expected_avg = (price1 * half + price2 * half) / (half + half)
    assert positions[0]["entry_price"] == expected_avg
    assert positions[0]["quantity"] == half + half

    outcomes = conn.execute("SELECT COUNT(*) FROM trade_outcomes WHERE trade_plan_id = ?", (plan_id,)).fetchone()[0]
    assert outcomes == 1


def test_long_entry_zero_liquidity_no_fill(conn):
    plan_id = _open_trade_plan(conn, signal="LONG", entry_from=100.0, entry_to=101.0)
    paper_trading.open_virtual_order(conn, plan_id)
    _tick(conn, bid=100.4, ask=100.5, ask_qty=0.0)
    order = journal_db.get_paper_order_by_trade_plan_id(conn, plan_id)
    assert order["status"] == "PENDING"
    assert order["filled_quantity"] == 0
    assert journal_db.get_open_positions(conn, "ETHUSDT") == []


# ---------------------------------------------------------------------------
# Expiry
# ---------------------------------------------------------------------------


def test_zero_fill_order_expires_past_valid_for_minutes(conn):
    plan_id = _open_trade_plan(conn, signal="LONG", entry_from=100.0, entry_to=101.0, valid_for_minutes=15)
    paper_trading.open_virtual_order(conn, plan_id)
    result = _tick(conn, bid=104.0, ask=105.0, now=NOW_EPOCH + 20 * 60)  # 20 min later, price never touched zone
    order = journal_db.get_paper_order_by_trade_plan_id(conn, plan_id)
    assert order["status"] == "EXPIRED"
    assert result.orders_expired == [order["id"]]


def test_order_not_yet_expired_stays_pending(conn):
    plan_id = _open_trade_plan(conn, signal="LONG", entry_from=100.0, entry_to=101.0, valid_for_minutes=15)
    paper_trading.open_virtual_order(conn, plan_id)
    _tick(conn, bid=104.0, ask=105.0, now=NOW_EPOCH + 5 * 60)
    order = journal_db.get_paper_order_by_trade_plan_id(conn, plan_id)
    assert order["status"] == "PENDING"


def test_partially_filled_order_past_expiry_becomes_filled_as_is(conn):
    plan_id = _open_trade_plan(conn, signal="LONG", entry_from=100.0, entry_to=101.0, valid_for_minutes=15)
    paper_trading.open_virtual_order(conn, plan_id)
    order_before = journal_db.get_paper_order_by_trade_plan_id(conn, plan_id)
    small_qty = order_before["quantity"] / 4
    _tick(conn, bid=100.4, ask=100.5, ask_qty=float(small_qty), now=NOW_EPOCH)
    result = _tick(conn, bid=104.0, ask=105.0, now=NOW_EPOCH + 20 * 60)
    order = journal_db.get_paper_order_by_trade_plan_id(conn, plan_id)
    assert order["status"] == "FILLED"
    assert order["filled_quantity"] == small_qty
    assert result.orders_force_closed_partial == [order["id"]]
    positions = journal_db.get_open_positions(conn, "ETHUSDT")
    assert positions[0]["quantity"] == small_qty


def test_order_without_trade_plan_never_expires(conn):
    order_id = journal_db.insert_paper_order(conn, symbol="ETHUSDT", side="LONG", order_type="LIMIT", quantity="0.5")
    _tick(conn, bid=100.0, ask=100.1, now=NOW_EPOCH + 999999)
    order = conn.execute("SELECT status FROM paper_orders WHERE id = ?", (order_id,)).fetchone()
    assert order["status"] == "PENDING"


# ---------------------------------------------------------------------------
# Stop-loss exits
# ---------------------------------------------------------------------------


def _open_and_fill_long(conn, *, stop_loss=98.0, take_profits=None, valid_for_minutes=15):
    plan_id = _open_trade_plan(conn, signal="LONG", entry_from=100.0, entry_to=101.0, stop_loss=stop_loss, take_profits=take_profits, valid_for_minutes=valid_for_minutes)
    paper_trading.open_virtual_order(conn, plan_id)
    _tick(conn, bid=100.4, ask=100.5, now=NOW_EPOCH)
    return plan_id, journal_db.get_open_positions(conn, "ETHUSDT")[0]


def _open_and_fill_short(conn, *, stop_loss=103.0, take_profits=None, valid_for_minutes=15):
    if take_profits is None:
        take_profits = [("TP1", 95.0, 100.0)]
    plan_id = _open_trade_plan(conn, signal="SHORT", entry_from=100.0, entry_to=101.0, stop_loss=stop_loss, take_profits=take_profits, valid_for_minutes=valid_for_minutes)
    paper_trading.open_virtual_order(conn, plan_id)
    _tick(conn, bid=100.5, ask=100.6, now=NOW_EPOCH)
    return plan_id, journal_db.get_open_positions(conn, "ETHUSDT")[0]


def test_long_stop_loss_exit(conn):
    plan_id, position = _open_and_fill_long(conn, stop_loss=98.0)
    result = _tick(conn, bid=97.5, ask=97.6, now=NOW_EPOCH + 60)
    assert result.positions_closed == [position["id"]]
    closed = conn.execute("SELECT status, exit_price FROM positions WHERE id = ?", (position["id"],)).fetchone()
    assert closed["status"] == "CLOSED"
    expected_exit = Decimal("97.5") * (1 - DEFAULT_RISK_SETTINGS.slippage_percent / Decimal("100"))
    assert Decimal(closed["exit_price"]) == expected_exit
    outcome = journal_db.get_trade_outcome(conn, plan_id)
    assert outcome["status"] == "EVALUATED"
    assert outcome["exit_reason"] == "SL"


def test_short_stop_loss_exit(conn):
    plan_id, position = _open_and_fill_short(conn, stop_loss=103.0)
    result = _tick(conn, bid=103.4, ask=103.5, now=NOW_EPOCH + 60)
    assert result.positions_closed == [position["id"]]
    outcome = journal_db.get_trade_outcome(conn, plan_id)
    assert outcome["exit_reason"] == "SL"
    closed = conn.execute("SELECT exit_price FROM positions WHERE id = ?", (position["id"],)).fetchone()
    expected_exit = Decimal("103.5") * (1 + DEFAULT_RISK_SETTINGS.slippage_percent / Decimal("100"))
    assert Decimal(closed["exit_price"]) == expected_exit


# ---------------------------------------------------------------------------
# Take-profit exits
# ---------------------------------------------------------------------------


def test_long_single_tp_partial_close(conn):
    plan_id, position = _open_and_fill_long(conn, take_profits=[("TP1", 105.0, 50.0), ("TP2", 110.0, 50.0)])
    result = _tick(conn, bid=105.5, ask=105.6, now=NOW_EPOCH + 60)
    assert result.positions_closed == []  # still open, only TP1 (50%) crossed
    remaining = paper_trading._remaining_quantity(conn, journal_db.get_open_positions(conn, "ETHUSDT")[0])
    assert remaining == position["quantity"] / 2
    fills = journal_db.get_position_fills(conn, position["id"])
    tp_fills = [f for f in fills if f["fill_type"] == "TAKE_PROFIT"]
    assert len(tp_fills) == 1
    assert tp_fills[0]["label"] == "TP1"
    assert tp_fills[0]["price"] == Decimal("105.0")  # exact limit price, no slippage


def test_short_single_tp_partial_close(conn):
    plan_id, position = _open_and_fill_short(conn, take_profits=[("TP1", 95.0, 50.0), ("TP2", 90.0, 50.0)])
    result = _tick(conn, bid=94.4, ask=94.5, now=NOW_EPOCH + 60)
    assert result.positions_closed == []
    fills = journal_db.get_position_fills(conn, position["id"])
    tp_fills = [f for f in fills if f["fill_type"] == "TAKE_PROFIT"]
    assert len(tp_fills) == 1
    assert tp_fills[0]["price"] == Decimal("95.0")


def test_tp_crossed_exactly_exhausting_remaining_closes_position(conn):
    plan_id, position = _open_and_fill_long(conn, take_profits=[("TP1", 105.0, 100.0)])
    result = _tick(conn, bid=105.5, ask=105.6, now=NOW_EPOCH + 60)
    assert result.positions_closed == [position["id"]]
    outcome = journal_db.get_trade_outcome(conn, plan_id)
    assert outcome["status"] == "EVALUATED"
    assert outcome["exit_reason"] == "TP1"


def test_multi_tp_crossed_same_tick_farthest_first(conn):
    plan_id, position = _open_and_fill_long(conn, take_profits=[("TP1", 105.0, 50.0), ("TP2", 110.0, 50.0)])
    result = _tick(conn, bid=111.0, ask=111.1, now=NOW_EPOCH + 60)
    fills = journal_db.get_position_fills(conn, position["id"])
    tp_fills = [f for f in fills if f["fill_type"] == "TAKE_PROFIT"]
    assert len(tp_fills) == 2
    assert [f["label"] for f in tp_fills] == ["TP2", "TP1"]  # farthest processed first
    assert result.positions_closed == [position["id"]]
    outcome = journal_db.get_trade_outcome(conn, plan_id)
    # Exit reason reflects the LAST fill that zeroed the remainder (the
    # nearer TP, since farthest-first processing consumes TP2 first) —
    # deliberate, locked in by this test per the Stage 8 plan.
    assert outcome["exit_reason"] == "TP1"


def test_tp_does_not_refire_on_subsequent_tick(conn):
    plan_id, position = _open_and_fill_long(conn, take_profits=[("TP1", 105.0, 50.0), ("TP2", 110.0, 50.0)])
    _tick(conn, bid=105.5, ask=105.6, now=NOW_EPOCH + 60)
    _tick(conn, bid=105.5, ask=105.6, now=NOW_EPOCH + 120)
    fills = journal_db.get_position_fills(conn, position["id"])
    tp_fills = [f for f in fills if f["fill_type"] == "TAKE_PROFIT"]
    assert len(tp_fills) == 1


# ---------------------------------------------------------------------------
# Trailing stop / breakeven
# ---------------------------------------------------------------------------


def test_trailing_stop_moves_to_breakeven_long(conn):
    plan_id, position = _open_and_fill_long(conn, stop_loss=98.0)
    entry_price = position["entry_price"]
    # risk = entry - 98 ~= 2.4ish; move price 1R above entry to trigger breakeven
    risk = entry_price - Decimal("98.0")
    trigger_price = entry_price + risk
    result = _tick(conn, bid=float(trigger_price), ask=float(trigger_price) + 0.1, now=NOW_EPOCH + 60)
    assert result.stops_trailed == [position["id"]]
    updated = conn.execute("SELECT stop_loss FROM positions WHERE id = ?", (position["id"],)).fetchone()
    assert Decimal(updated["stop_loss"]) == entry_price


def test_trailing_stop_does_not_move_again_once_at_breakeven(conn):
    plan_id, position = _open_and_fill_long(conn, stop_loss=98.0)
    entry_price = position["entry_price"]
    risk = entry_price - Decimal("98.0")
    trigger_price = entry_price + risk
    _tick(conn, bid=float(trigger_price), ask=float(trigger_price) + 0.1, now=NOW_EPOCH + 60)
    result = _tick(conn, bid=float(trigger_price) + 0.5, ask=float(trigger_price) + 0.6, now=NOW_EPOCH + 120)
    assert result.stops_trailed == []


def test_trailing_stop_does_not_move_before_trigger(conn):
    plan_id, position = _open_and_fill_long(conn, stop_loss=98.0)
    result = _tick(conn, bid=float(position["entry_price"]) + 0.05, ask=float(position["entry_price"]) + 0.15, now=NOW_EPOCH + 60)
    assert result.stops_trailed == []
    updated = conn.execute("SELECT stop_loss FROM positions WHERE id = ?", (position["id"],)).fetchone()
    assert Decimal(updated["stop_loss"]) == Decimal("98.0")


def test_trailing_stop_short_mirrors_long(conn):
    plan_id, position = _open_and_fill_short(conn, stop_loss=103.0)
    entry_price = position["entry_price"]
    risk = Decimal("103.0") - entry_price
    trigger_price = entry_price - risk
    result = _tick(conn, bid=float(trigger_price) - 0.1, ask=float(trigger_price), now=NOW_EPOCH + 60)
    assert result.stops_trailed == [position["id"]]
    updated = conn.execute("SELECT stop_loss FROM positions WHERE id = ?", (position["id"],)).fetchone()
    assert Decimal(updated["stop_loss"]) == entry_price


# ---------------------------------------------------------------------------
# Time-based close
# ---------------------------------------------------------------------------


def test_scalping_time_based_close(conn):
    plan_id, position = _open_and_fill_long(conn, stop_loss=95.0, take_profits=[("TP1", 112.0, 100.0)])  # far stop so it never triggers first
    max_hold = config.PAPER_TRADING_MAX_HOLD_SECONDS["scalping"]
    result = _tick(conn, bid=100.45, ask=100.55, now=NOW_EPOCH + max_hold + 1)
    assert result.positions_closed == [position["id"]]
    outcome = journal_db.get_trade_outcome(conn, plan_id)
    assert outcome["exit_reason"] == "TIMEOUT"


def test_swing_time_based_close_uses_different_threshold(conn):
    plan_id, position = _open_and_fill_long(conn, stop_loss=95.0, take_profits=[("TP1", 112.0, 100.0)])
    conn.execute("UPDATE trade_plans SET mode = 'swing' WHERE id = ?", (plan_id,))
    scalping_hold = config.PAPER_TRADING_MAX_HOLD_SECONDS["scalping"]
    swing_hold = config.PAPER_TRADING_MAX_HOLD_SECONDS["swing"]
    assert swing_hold > scalping_hold
    result = _tick(conn, bid=100.45, ask=100.55, now=NOW_EPOCH + scalping_hold + 1)
    assert result.positions_closed == []  # not yet due under swing's longer threshold
    result2 = _tick(conn, bid=100.45, ask=100.55, now=NOW_EPOCH + swing_hold + 1)
    assert result2.positions_closed == [position["id"]]


def test_order_without_trade_plan_position_never_time_closed(conn):
    order_id = journal_db.insert_paper_order(conn, symbol="ETHUSDT", side="LONG", order_type="MARKET", quantity="0.5", stop_loss="90.0")
    _tick(conn, bid=100.0, ask=100.1, now=NOW_EPOCH)  # MARKET fills unconditionally
    position = journal_db.get_open_positions(conn, "ETHUSDT")[0]
    result = _tick(conn, bid=100.05, ask=100.15, now=NOW_EPOCH + 999999)
    assert result.positions_closed == []
    still_open = conn.execute("SELECT status FROM positions WHERE id = ?", (position["id"],)).fetchone()
    assert still_open["status"] == "OPEN"


# ---------------------------------------------------------------------------
# Full lifecycle integration
# ---------------------------------------------------------------------------


def test_full_lifecycle_partial_tp_trail_then_timeout(conn):
    plan_id = _open_trade_plan(
        conn, signal="LONG", entry_from=100.0, entry_to=101.0, stop_loss=98.0,
        take_profits=[("TP1", 105.0, 50.0), ("TP2", 110.0, 50.0)], valid_for_minutes=15,
    )
    paper_trading.open_virtual_order(conn, plan_id)

    # Tick 1: entry fills
    _tick(conn, bid=100.4, ask=100.5, now=NOW_EPOCH)
    position = journal_db.get_open_positions(conn, "ETHUSDT")[0]
    entry_price = position["entry_price"]
    original_qty = position["quantity"]

    # Tick 2: price rallies past breakeven trigger (but below TP1) -> stop trails
    risk = entry_price - Decimal("98.0")
    breakeven_trigger_price = float(entry_price + risk)
    result2 = _tick(conn, bid=breakeven_trigger_price, ask=breakeven_trigger_price + 0.1, now=NOW_EPOCH + 60)
    assert result2.stops_trailed == [position["id"]]

    # Tick 3: TP1 crosses, 50% closes
    result3 = _tick(conn, bid=105.5, ask=105.6, now=NOW_EPOCH + 120)
    assert result3.positions_closed == []

    # Tick 4: force-closed by time (never reaches TP2)
    max_hold = config.PAPER_TRADING_MAX_HOLD_SECONDS["scalping"]
    result4 = _tick(conn, bid=106.0, ask=106.1, now=NOW_EPOCH + max_hold + 1)
    assert result4.positions_closed == [position["id"]]

    outcome = journal_db.get_trade_outcome(conn, plan_id)
    assert outcome["status"] == "EVALUATED"
    assert outcome["exit_reason"] == "TIMEOUT"
    assert outcome["mfe_r"] is not None and outcome["mfe_r"] > 0
    closed_position = conn.execute("SELECT status, realized_pnl_usdt FROM positions WHERE id = ?", (position["id"],)).fetchone()
    assert closed_position["status"] == "CLOSED"

    fills = journal_db.get_position_fills(conn, position["id"])
    closing_fills = [f for f in fills if f["fill_type"] != "ENTRY"]
    total_pnl = sum(f["realized_pnl_usdt"] for f in closing_fills)
    assert Decimal(closed_position["realized_pnl_usdt"]) == total_pnl
    total_closed_qty = sum(f["quantity"] for f in closing_fills)
    assert total_closed_qty == original_qty
    # Hand-checked R multiple
    initial_risk = abs(entry_price - Decimal("98.0")) * original_qty
    assert outcome["r_multiple"] == pytest.approx(float(total_pnl / initial_risk))


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def _finalize_synthetic_outcome(conn, *, r_multiple, mfe_r=0.5, mae_r=-0.2, exit_reason="TP1", source_label="GPT-4o mini", regime="TREND_UP", ruleset_version="v1", decision="LONG_BIAS", mode="scalping", order_created_offset=0.0):
    """Builds a full chain and a CLOSED-equivalent trade_outcome with a
    hand-specified r_multiple, for statistics tests that need known numbers
    rather than numbers derived from price-path arithmetic.
    """
    plan_id = _open_trade_plan(conn, signal="LONG", mode=mode)
    conn.execute(
        "UPDATE strategy_scores SET regime = ?, ruleset_version = ?, decision = ? "
        "WHERE id = (SELECT strategy_score_id FROM trade_plans WHERE id = ?)",
        (regime, ruleset_version, decision, plan_id),
    )
    conn.execute("UPDATE trade_plans SET source_label = ? WHERE id = ?", (source_label, plan_id))

    order_created_at = NOW_EPOCH + order_created_offset
    journal_db.insert_paper_order(
        conn, symbol="ETHUSDT", side="LONG", order_type="LIMIT", quantity="1.0",
        trade_plan_id=plan_id, status="FILLED", now=order_created_at,
    )

    outcome_id = journal_db.insert_trade_outcome_pending(conn, trade_plan_id=plan_id, symbol="ETHUSDT", mode=mode, now=order_created_at)
    outcome = SimulatedOutcome(exit_reason=exit_reason, r_multiple=r_multiple, mfe_r=mfe_r, mae_r=mae_r, exit_price=105.0, exit_time=order_created_at + 60, duration_seconds=60.0)
    journal_db.finalize_trade_outcome(conn, outcome_id, outcome, now=order_created_at + 60)
    return plan_id


def test_bucket_stats_win_rate_expectancy_median():
    records = [{"r_multiple": 1.0, "mfe_r": 1.0, "mae_r": -0.1, "duration_seconds": 60.0, "exit_reason": "TP1"},
               {"r_multiple": -0.5, "mfe_r": 0.2, "mae_r": -0.5, "duration_seconds": 30.0, "exit_reason": "SL"},
               {"r_multiple": 2.0, "mfe_r": 2.0, "mae_r": 0.0, "duration_seconds": 90.0, "exit_reason": "TP2"}]
    bucket = paper_trading._bucket_stats(3, records)
    assert bucket.total == 3
    assert bucket.evaluated == 3
    assert bucket.wins == 2
    assert bucket.win_rate == pytest.approx(200 / 3)
    assert bucket.expectancy_r == pytest.approx((1.0 - 0.5 + 2.0) / 3)
    assert bucket.median_r == 1.0


def test_bucket_stats_profit_factor_normal():
    records = [{"r_multiple": 2.0, "mfe_r": 2.0, "mae_r": 0.0, "duration_seconds": 1.0, "exit_reason": "TP1"},
               {"r_multiple": -1.0, "mfe_r": 0.1, "mae_r": -1.0, "duration_seconds": 1.0, "exit_reason": "SL"}]
    bucket = paper_trading._bucket_stats(2, records)
    assert bucket.profit_factor == pytest.approx(2.0)
    assert bucket.profit_factor_undefined is False


def test_bucket_stats_profit_factor_undefined_when_no_losses():
    records = [{"r_multiple": 1.0, "mfe_r": 1.0, "mae_r": 0.0, "duration_seconds": 1.0, "exit_reason": "TP1"}]
    bucket = paper_trading._bucket_stats(1, records)
    assert bucket.profit_factor is None
    assert bucket.profit_factor_undefined is True


def test_max_drawdown_known_sequence():
    # cumulative: 1, 3, 1.5, -0.5, 2 -> peak 3, trough after peak -0.5 -> dd 3.5
    r_values = [1.0, 2.0, -1.5, -2.0, 2.5]
    assert paper_trading._max_drawdown(r_values) == pytest.approx(3.5)


def test_low_sample_flag():
    records = [{"r_multiple": 1.0, "mfe_r": 1.0, "mae_r": 0.0, "duration_seconds": 1.0, "exit_reason": "TP1"}]
    bucket = paper_trading._bucket_stats(1, records)
    assert bucket.low_sample is True
    many = [{"r_multiple": 1.0, "mfe_r": 1.0, "mae_r": 0.0, "duration_seconds": 1.0, "exit_reason": "TP1"}] * config.MIN_SAMPLE_FOR_STATS
    assert paper_trading._bucket_stats(len(many), many).low_sample is False


def test_compute_statistics_by_model_buckets_on_source_label(conn):
    _finalize_synthetic_outcome(conn, r_multiple=1.0, source_label="GPT-4o mini")
    _finalize_synthetic_outcome(conn, r_multiple=-0.5, source_label="Gemini 2.5 Pro")
    stats = paper_trading.compute_paper_trading_statistics(conn, mode="scalping", window_start=NOW_EPOCH - 10, window_end=NOW_EPOCH + 3600)
    assert set(stats.by_model.keys()) == {"GPT-4o mini", "Gemini 2.5 Pro"}
    assert stats.by_model["GPT-4o mini"].evaluated == 1
    assert stats.by_model["Gemini 2.5 Pro"].evaluated == 1


def test_compute_statistics_by_strategy_buckets_on_ruleset_and_decision(conn):
    _finalize_synthetic_outcome(conn, r_multiple=1.0, ruleset_version="v1", decision="LONG_BIAS")
    _finalize_synthetic_outcome(conn, r_multiple=0.5, ruleset_version="v1", decision="SHORT_BIAS")
    stats = paper_trading.compute_paper_trading_statistics(conn, mode="scalping", window_start=NOW_EPOCH - 10, window_end=NOW_EPOCH + 3600)
    assert ("v1", "LONG_BIAS") in stats.by_strategy
    assert ("v1", "SHORT_BIAS") in stats.by_strategy


def test_compute_statistics_by_regime(conn):
    _finalize_synthetic_outcome(conn, r_multiple=1.0, regime="TREND_UP")
    _finalize_synthetic_outcome(conn, r_multiple=-1.0, regime="RANGE")
    stats = paper_trading.compute_paper_trading_statistics(conn, mode="scalping", window_start=NOW_EPOCH - 10, window_end=NOW_EPOCH + 3600)
    assert stats.by_regime["TREND_UP"].evaluated == 1
    assert stats.by_regime["RANGE"].evaluated == 1


def test_compute_statistics_mode_segmentation(conn):
    _finalize_synthetic_outcome(conn, r_multiple=1.0, mode="scalping")
    _finalize_synthetic_outcome(conn, r_multiple=1.0, mode="swing")
    scalping_stats = paper_trading.compute_paper_trading_statistics(conn, mode="scalping", window_start=NOW_EPOCH - 10, window_end=NOW_EPOCH + 3600)
    total_scalping = sum(b.total for b in scalping_stats.by_model.values())
    assert total_scalping == 1


def test_compute_statistics_pending_trade_counts_in_total_not_evaluated(conn):
    plan_id = _open_trade_plan(conn, signal="LONG")
    paper_trading.open_virtual_order(conn, plan_id, now=NOW_EPOCH)
    stats = paper_trading.compute_paper_trading_statistics(conn, mode="scalping", window_start=NOW_EPOCH - 10, window_end=NOW_EPOCH + 3600)
    total = sum(b.total for b in stats.by_model.values())
    evaluated = sum(b.evaluated for b in stats.by_model.values())
    assert total == 1
    assert evaluated == 0


def test_refresh_statistics_persists_rows_with_correct_regime_decision_split(conn):
    _finalize_synthetic_outcome(conn, r_multiple=1.0, regime="TREND_UP", ruleset_version="v1", decision="LONG_BIAS")
    result = paper_trading.refresh_statistics(conn, mode="scalping", window_start=NOW_EPOCH - 10, window_end=NOW_EPOCH + 3600)
    assert result.model_rows_inserted == 1
    assert result.strategy_rows_inserted == 2  # one by-strategy row + one by-regime row

    by_strategy_rows = journal_db.get_strategy_statistics(conn, ruleset_version="v1", mode="scalping")
    strategy_row = next(r for r in by_strategy_rows if r["decision"] is not None)
    assert strategy_row["decision"] == "LONG_BIAS"
    assert strategy_row["regime"] is None

    by_regime_rows = journal_db.get_strategy_statistics(conn, regime="TREND_UP")
    regime_row = by_regime_rows[0]
    assert regime_row["regime"] == "TREND_UP"
    assert regime_row["decision"] is None
