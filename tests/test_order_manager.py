"""Offline tests for order_manager.py — no network. bingx_client.get_price
and every bingx_private_client function are monkeypatched per test;
config.EXECUTION_DRY_RUN defaults to True (fake credentials set) so the
happy-path exercises the real DryRunNotSent short-circuit end to end.

Fixture factories mirror tests/test_paper_trading.py's own duplicated
pattern (this repo's established per-file-duplication test convention) —
`_open_trade_plan` additionally accepts `trade_permission`/`margin_mode`
overrides since order_manager's checks are stricter than paper_trading's.
"""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

import pandas as pd
import pytest

import bingx_client
import bingx_private_client
import config
import journal_db
import order_manager
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
from risk_manager import DEFAULT_RISK_SETTINGS, PositionCalculator, TakeProfitTarget, TradeScenario
from strategy_engine import Contribution, ScoreResult

NOW = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
NOW_EPOCH = NOW.timestamp()


@pytest.fixture(autouse=True)
def _use_tmp_db(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "JOURNAL_DB_FILE", tmp_path / "order_manager_test.db")


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


# ---------------------------------------------------------------------------
# Fixture factories (self-contained, mirrors test_paper_trading.py)
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
    trade_permission: str = "ALLOWED",
    margin_mode_override: str | None = None,
) -> int:
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
        mode=mode, overall_signal=signal, state="strong",
        agreement_fraction=1.0, agreeing_count=3, vote_count=3, total_models=3, avg_confidence=70.0,
        plan=plan, trade_permission=trade_permission if signal in ("LONG", "SHORT") else "WAIT",
        trade_permission_reason="ok", reasons=["r"], risks=["k"], wait_or_invalidation=[],
    )

    plan_id = journal_db.insert_trade_plan(
        conn, consensus, strategy_score_id=score_id, symbol="ETHUSDT", timestamp=formed_at, calculation=calculation,
    )
    if margin_mode_override is not None:
        conn.execute("UPDATE trade_plans SET margin_mode = ? WHERE id = ?", (margin_mode_override, plan_id))
    return plan_id


# ---------------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------------


def test_place_entry_order_already_exists(conn):
    plan_id = _open_trade_plan(conn)
    journal_db.insert_real_order(conn, symbol="ETHUSDT", side="LONG", order_type="MARKET", quantity="0.01", trade_plan_id=plan_id)
    result = order_manager.place_entry_order(conn, plan_id, now=NOW_EPOCH)
    assert result.status == "ALREADY_EXISTS"


# ---------------------------------------------------------------------------
# Trade plan validity / staleness / margin mode
# ---------------------------------------------------------------------------


def test_place_entry_order_rejects_wait_signal(conn):
    plan_id = _open_trade_plan(conn, signal="WAIT")
    result = order_manager.place_entry_order(conn, plan_id, now=NOW_EPOCH)
    assert result.status == "REJECTED"
    assert "signal" in result.reason


def test_place_entry_order_rejects_non_allowed_permission(conn):
    plan_id = _open_trade_plan(conn, trade_permission="WAITING_TRIGGER")
    result = order_manager.place_entry_order(conn, plan_id, now=NOW_EPOCH)
    assert result.status == "REJECTED"
    assert "trade_permission" in result.reason


def test_place_entry_order_rejects_expired_signal(conn):
    plan_id = _open_trade_plan(conn, valid_for_minutes=5, formed_at=NOW_EPOCH)
    result = order_manager.place_entry_order(conn, plan_id, now=NOW_EPOCH + 10 * 60)
    assert result.status == "REJECTED"
    assert "expired" in result.reason


def test_place_entry_order_rejects_non_isolated_margin(conn):
    plan_id = _open_trade_plan(conn, margin_mode_override="CROSS")
    result = order_manager.place_entry_order(conn, plan_id, now=NOW_EPOCH)
    assert result.status == "REJECTED"
    assert "ISOLATED" in result.reason


# ---------------------------------------------------------------------------
# Price re-check
# ---------------------------------------------------------------------------


def test_place_entry_order_rejects_price_outside_zone(conn, monkeypatch):
    plan_id = _open_trade_plan(conn, entry_from=100.0, entry_to=101.0)
    monkeypatch.setattr(bingx_client, "get_price", lambda symbol: 150.0)
    result = order_manager.place_entry_order(conn, plan_id, now=NOW_EPOCH)
    assert result.status == "REJECTED"
    assert "entry zone" in result.reason


def test_place_entry_order_rejects_stop_already_breached(conn, monkeypatch):
    plan_id = _open_trade_plan(conn, signal="LONG", entry_from=100.0, entry_to=101.0, stop_loss=98.0)
    monkeypatch.setattr(bingx_client, "get_price", lambda symbol: 100.5)
    conn.execute("UPDATE trade_plans SET stop_loss_calc = '101.0' WHERE id = ?", (plan_id,))
    result = order_manager.place_entry_order(conn, plan_id, now=NOW_EPOCH)
    assert result.status == "REJECTED"
    assert "stop loss" in result.reason


def test_place_entry_order_rejects_when_price_fetch_fails(conn, monkeypatch):
    plan_id = _open_trade_plan(conn)
    monkeypatch.setattr(bingx_client, "get_price", lambda symbol: (_ for _ in ()).throw(bingx_client.NetworkError("down")))
    result = order_manager.place_entry_order(conn, plan_id, now=NOW_EPOCH)
    assert result.status == "REJECTED"
    assert "current price" in result.reason


# ---------------------------------------------------------------------------
# Balance
# ---------------------------------------------------------------------------


def test_place_entry_order_rejects_insufficient_balance(conn, monkeypatch):
    plan_id = _open_trade_plan(conn)
    monkeypatch.setattr(config, "EXECUTION_DRY_RUN", False)
    monkeypatch.setattr(bingx_private_client, "get_balance", lambda: {"balance": {"balance": "0.5"}})
    result = order_manager.place_entry_order(conn, plan_id, now=NOW_EPOCH)
    assert result.status == "REJECTED"
    assert "recompute" in result.reason or "VALID" in result.reason


def test_place_entry_order_rejects_unparseable_balance(conn, monkeypatch):
    plan_id = _open_trade_plan(conn)
    monkeypatch.setattr(config, "EXECUTION_DRY_RUN", False)
    monkeypatch.setattr(bingx_private_client, "get_balance", lambda: {"nonsense": True})
    result = order_manager.place_entry_order(conn, plan_id, now=NOW_EPOCH)
    assert result.status == "REJECTED"
    assert "balance" in result.reason


# ---------------------------------------------------------------------------
# Duplicate position / order checks
# ---------------------------------------------------------------------------


def test_place_entry_order_rejects_existing_open_real_position(conn):
    plan_id = _open_trade_plan(conn)
    journal_db.insert_position(
        conn, symbol="ETHUSDT", side="LONG", source="REAL", entry_price=Decimal("100"), quantity=Decimal("0.01"), status="OPEN",
    )
    result = order_manager.place_entry_order(conn, plan_id, now=NOW_EPOCH)
    assert result.status == "REJECTED"
    assert "already exists" in result.reason


def test_place_entry_order_rejects_pending_real_order_same_symbol(conn):
    plan_id = _open_trade_plan(conn)
    other_plan_id = _open_trade_plan(conn)
    journal_db.insert_real_order(conn, symbol="ETHUSDT", side="LONG", order_type="MARKET", quantity="0.01", trade_plan_id=other_plan_id, status="PENDING")
    result = order_manager.place_entry_order(conn, plan_id, now=NOW_EPOCH)
    assert result.status == "REJECTED"
    assert "pending" in result.reason


# ---------------------------------------------------------------------------
# Happy path — DRY_RUN
# ---------------------------------------------------------------------------


def test_place_entry_order_dry_run_happy_path(conn):
    plan_id = _open_trade_plan(conn, signal="LONG", entry_from=100.0, entry_to=101.0, stop_loss=98.0, take_profits=[("TP1", 105.0, 100.0)])
    result = order_manager.place_entry_order(conn, plan_id, now=NOW_EPOCH)
    assert result.status == "DRY_RUN"
    assert result.reason is None
    assert len(result.real_order_ids) == 3  # entry + SL + TP1
    assert result.exchange_order_ids == []

    entry_row = journal_db.get_real_order_by_trade_plan_id(conn, plan_id)
    assert entry_row is not None
    assert entry_row["stop_loss"] is None  # entry row never carries stop_loss/take_profit
    assert entry_row["take_profit"] is None

    position = journal_db.get_open_positions(conn, "ETHUSDT", source="REAL")
    assert len(position) == 1
    assert position[0]["side"] == "LONG"
    assert position[0]["quantity"] > 0

    fills = journal_db.get_position_fills(conn, position[0]["id"])
    assert len(fills) == 1
    assert fills[0]["fill_type"] == "ENTRY"


def test_place_entry_order_dry_run_places_sl_and_all_tp_rows(conn):
    # Mirrors config.BACKTEST_TP_LEVELS' tuning: a 40/60 split needs each
    # INDIVIDUAL target's net R:R (not just the blended average) to clear
    # DEFAULT_RISK_SETTINGS.min_risk_reward=1.5, or risk_manager rejects the
    # whole scenario as FEES_TOO_HIGH — see the Stage 9 plan's note on this.
    plan_id = _open_trade_plan(
        conn, entry_from=100.0, entry_to=101.0, stop_loss=98.0,
        take_profits=[("TP1", 104.25, 40.0), ("TP2", 108.0, 60.0)],
    )
    result = order_manager.place_entry_order(conn, plan_id, now=NOW_EPOCH)
    assert result.status == "DRY_RUN"
    # entry + SL + TP1 + TP2 = 4 real_orders rows
    assert len(result.real_order_ids) == 4

    rows = conn.execute("SELECT * FROM real_orders WHERE trade_plan_id = ?", (plan_id,)).fetchall()
    assert len(rows) == 4
    sl_rows = [r for r in rows if r["stop_loss"] is not None]
    tp_rows = [r for r in rows if r["take_profit"] is not None]
    assert len(sl_rows) == 1
    assert len(tp_rows) == 2


def test_place_entry_order_dry_run_notes_marked(conn):
    plan_id = _open_trade_plan(conn)
    order_manager.place_entry_order(conn, plan_id, now=NOW_EPOCH)
    entry_row = journal_db.get_real_order_by_trade_plan_id(conn, plan_id)
    assert "DRY_RUN" in (entry_row.get("notes") or "")


# ---------------------------------------------------------------------------
# Leverage/margin-mode setup failure (live mode)
# ---------------------------------------------------------------------------


def test_place_entry_order_leverage_failure_marks_entry_rejected(conn, monkeypatch):
    plan_id = _open_trade_plan(conn)
    monkeypatch.setattr(config, "EXECUTION_DRY_RUN", False)
    monkeypatch.setattr(bingx_private_client, "get_balance", lambda: {"balance": {"balance": "1000"}})
    monkeypatch.setattr(
        bingx_private_client, "set_leverage",
        lambda *a, **k: (_ for _ in ()).throw(bingx_client.APIError("leverage rejected")),
    )
    result = order_manager.place_entry_order(conn, plan_id, now=NOW_EPOCH)
    assert result.status == "REJECTED"
    assert "leverage" in result.reason
    entry_row = journal_db.get_real_order_by_trade_plan_id(conn, plan_id)
    assert entry_row["status"] == "REJECTED"


def test_place_entry_order_entry_placement_failure(conn, monkeypatch):
    plan_id = _open_trade_plan(conn)
    monkeypatch.setattr(config, "EXECUTION_DRY_RUN", False)
    monkeypatch.setattr(bingx_private_client, "get_balance", lambda: {"balance": {"balance": "1000"}})
    monkeypatch.setattr(bingx_private_client, "set_leverage", lambda *a, **k: {})
    monkeypatch.setattr(bingx_private_client, "set_margin_type", lambda *a, **k: {})
    monkeypatch.setattr(
        bingx_private_client, "place_order",
        lambda *a, **k: (_ for _ in ()).throw(bingx_client.APIError("insufficient margin")),
    )
    result = order_manager.place_entry_order(conn, plan_id, now=NOW_EPOCH)
    assert result.status == "REJECTED"
    assert "entry order placement failed" in result.reason


def test_place_entry_order_sl_failure_is_critical_but_records_entry(conn, monkeypatch):
    plan_id = _open_trade_plan(conn)
    monkeypatch.setattr(config, "EXECUTION_DRY_RUN", False)
    monkeypatch.setattr(bingx_private_client, "get_balance", lambda: {"balance": {"balance": "1000"}})
    monkeypatch.setattr(bingx_private_client, "set_leverage", lambda *a, **k: {})
    monkeypatch.setattr(bingx_private_client, "set_margin_type", lambda *a, **k: {})

    calls = {"n": 0}

    def _place_order(symbol, side, position_side, order_type, quantity, **kwargs):
        calls["n"] += 1
        if order_type == "STOP_MARKET":
            raise bingx_client.APIError("SL rejected")
        return {"orderId": f"ex-{calls['n']}"}

    monkeypatch.setattr(bingx_private_client, "place_order", _place_order)
    result = order_manager.place_entry_order(conn, plan_id, now=NOW_EPOCH)
    assert result.status == "REJECTED"
    assert "stop loss" in result.reason
    assert "ex-1" in result.exchange_order_ids  # entry did go through

    events = journal_db.get_recent_system_events(conn, level="CRITICAL")
    assert any(e["event_code"] == "REAL_STOP_LOSS_PLACEMENT_FAILED" for e in events)


def test_place_entry_order_live_limit_entry_defers_position_creation(conn, monkeypatch):
    # Stage 12: a LIMIT/TRIGGER entry in LIVE mode isn't necessarily filled
    # the instant it's placed — order_manager must not assume it is.
    # position_manager.check_entry_fill (Stage 12) creates the position
    # later, only once BingX confirms the fill.
    plan_id = _open_trade_plan(conn)  # entry_type="LIMIT_ZONE" normalizes to LIMIT
    monkeypatch.setattr(config, "EXECUTION_DRY_RUN", False)
    monkeypatch.setattr(bingx_private_client, "get_balance", lambda: {"balance": {"balance": "1000"}})
    monkeypatch.setattr(bingx_private_client, "set_leverage", lambda *a, **k: {})
    monkeypatch.setattr(bingx_private_client, "set_margin_type", lambda *a, **k: {})
    calls = {"n": 0}

    def _place_order(symbol, side, position_side, order_type, quantity, **kwargs):
        calls["n"] += 1
        return {"orderId": f"ex-{calls['n']}"}

    monkeypatch.setattr(bingx_private_client, "place_order", _place_order)

    result = order_manager.place_entry_order(conn, plan_id, now=NOW_EPOCH)
    assert result.status == "PLACED"
    assert journal_db.get_open_positions(conn, "ETHUSDT", source="REAL") == []
    assert journal_db.get_position_by_trade_plan_id(conn, plan_id) is None

    entry_row = journal_db.get_entry_real_order_for_trade_plan(conn, plan_id)
    assert entry_row["status"] == "OPEN"
    assert entry_row["exchange_order_id"] == "ex-1"


# ---------------------------------------------------------------------------
# close_all
# ---------------------------------------------------------------------------


def test_close_all_dry_run_cancels_and_closes(conn):
    plan_id = _open_trade_plan(conn)
    order_manager.place_entry_order(conn, plan_id, now=NOW_EPOCH)  # opens a DRY_RUN position + pending SL/TP orders

    result = order_manager.close_all(conn, "ETHUSDT")
    assert result.position_closed is True
    assert len(result.orders_cancelled) >= 1
    assert journal_db.get_open_positions(conn, "ETHUSDT", source="REAL") == []
    assert journal_db.get_pending_real_orders(conn, "ETHUSDT") == []


def test_close_all_no_positions_is_a_no_op(conn):
    result = order_manager.close_all(conn, "ETHUSDT")
    assert result.position_closed is False
    assert result.orders_cancelled == []
    assert result.errors == []
