"""Offline tests for journal_db.py / journal_schema.py — no network, no AI.
All I/O redirected to a pytest tmp_path via config.JOURNAL_DB_FILE. Dataclass
fixtures are built by hand (or, for PositionCalculation/BingXManualFields,
via the real risk_manager.PositionCalculator/build_bingx_manual_fields, since
hand-building ~30 Decimal fields is error-prone and this already has a
proven test pattern in tests/test_risk_settings.py).
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from decimal import Decimal

import pandas as pd
import pytest

import config
import journal_db
from ai_client import AIAnalysisResult
from ai_schema import EntryZone, TakeProfit, TradePlan
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
from risk_manager import (
    DEFAULT_RISK_SETTINGS,
    PositionCalculator,
    TakeProfitTarget,
    TradeScenario,
    build_bingx_manual_fields,
)
from strategy_engine import Contribution, ScoreResult
from trade_validator import ValidationIssue, ValidationResult

NOW = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
NOW_EPOCH = NOW.timestamp()


@pytest.fixture(autouse=True)
def _use_tmp_db(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "JOURNAL_DB_FILE", tmp_path / "journal_test.db")


@pytest.fixture
def conn(_use_tmp_db):
    c = journal_db.init_db()
    yield c
    c.close()


# ---------------------------------------------------------------------------
# Fixture factories
# ---------------------------------------------------------------------------


def _market_snapshot(**overrides) -> MarketDataSnapshot:
    df = pd.DataFrame(
        [{"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5, "volume": 10.0, "time": NOW}]
    )
    timeframes = {
        "1m": {
            "candles": [{"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5, "volume": 10.0, "time": NOW}],
            "has_gap": False,
            "is_stale": False,
            "zero_volume": False,
            "dataframe": df,
        }
    }
    defaults = dict(
        timestamp=NOW,
        symbol="ETHUSDT",
        price=100.5,
        bid=100.4,
        ask=100.6,
        spread=0.2,
        spread_percent=0.2,
        timeframes=timeframes,
        funding_rate=0.0001,
        funding_history=[{"rate": 0.0001, "time": int(NOW_EPOCH * 1000)}],
        open_interest=123456.0,
        open_interest_history=[{"open_interest": 123000.0, "time": int(NOW_EPOCH * 1000)}],
        orderbook={"bids": [[100.4, 1.0]], "asks": [[100.6, 1.0]], "timestamp": NOW_EPOCH, "age_seconds": 0.5},
        orderbook_history=[{"imbalance": 0.1, "time": int(NOW_EPOCH * 1000)}],
        volume_24h=999.0,
        recent_trades=[{"price": 100.5, "qty": 1.0, "time": int(NOW_EPOCH * 1000), "is_buyer_maker": True}],
        instrument_rules={"quantity_step": "0.01", "price_step": "0.01"},
        data_quality="GOOD",
        quality_issues=[],
    )
    defaults.update(overrides)
    return MarketDataSnapshot(**defaults)


def _feature_set(**overrides) -> FeatureSet:
    trend = {
        "1m": TrendFeatures(
            price_vs_ema20_pct=0.5,
            price_vs_ema50_pct=1.0,
            price_vs_ema200_pct=2.0,
            ema20_slope_pct=0.1,
            ema20_ema50_distance_pct=0.3,
            ema50_ema200_distance_pct=0.4,
            adx=25.0,
            supertrend_direction="UP",
            structure_direction="BULLISH",
            higher_high=True,
            higher_low=True,
            lower_high=False,
            lower_low=False,
            break_of_structure=False,
            change_of_character=False,
            trend_state="BULLISH",
        )
    }
    momentum = {
        "1m": MomentumFeatures(
            rsi14=55.0, rsi_roc=1.0, macd_hist=0.2, macd_hist_change=0.01, roc=0.5, momentum=1.2,
            bullish_divergence=False, bearish_divergence=False,
        )
    }
    volatility = {
        "1m": VolatilityFeatures(
            atr14=2.0, atr_percentile=60.0, bollinger_width_pct=1.5, realized_volatility_pct=0.8,
            range_compression=False, volatility_expansion=False, last_candle_range_atr_ratio=0.9,
        )
    }
    volume = {"1m": VolumeFeatures(volume_ratio=1.1, volume_spike=False, volume_trend="INCREASING")}
    defaults = dict(
        timestamp=NOW,
        symbol="ETHUSDT",
        data_quality="GOOD",
        trend=trend,
        momentum=momentum,
        volatility=volatility,
        volume=volume,
        trade_flow=TradeFlowFeatures(buy_volume=5.0, sell_volume=3.0, delta=2.0, cumulative_volume_delta=2.0),
        futures=FuturesFeatures(
            funding_rate=0.0001, funding_change=0.00001, open_interest=123456.0,
            open_interest_change=100.0, open_interest_change_pct=0.08, oi_price_regime="NEW_LONGS",
        ),
        orderbook=OrderbookFeatures(
            imbalance=0.1, imbalance_avg_30_120s=0.12, spread=0.2, spread_change=0.01,
            microprice=100.51, has_large_wall=False,
        ),
        timeframe_alignment=0.75,
        distance_to_support_atr=0.5,
        distance_to_resistance_atr=1.2,
    )
    defaults.update(overrides)
    return FeatureSet(**defaults)


def _empty_feature_set() -> FeatureSet:
    return FeatureSet(
        timestamp=NOW,
        symbol="ETHUSDT",
        data_quality="NO_TRADE",
        trend={},
        momentum={},
        volatility={},
        volume={},
        trade_flow=TradeFlowFeatures(0.0, 0.0, 0.0, 0.0),
        futures=FuturesFeatures(None, None, None, None, None, None),
        orderbook=OrderbookFeatures(None, None, None, None, None, None),
        timeframe_alignment=0.0,
        distance_to_support_atr=None,
        distance_to_resistance_atr=None,
    )


def _regime_result(**overrides) -> RegimeResult:
    defaults = dict(
        timestamp=NOW,
        symbol="ETHUSDT",
        regime="TREND_UP",
        regime_by_timeframe={"1m": "TREND_UP"},
        reasons=["1m: TREND_UP"],
        strategy_hint="hint",
    )
    defaults.update(overrides)
    return RegimeResult(**defaults)


def _score_result(**overrides) -> ScoreResult:
    defaults = dict(
        timestamp=NOW,
        symbol="ETHUSDT",
        mode="scalping",
        ruleset_version="v1",
        long_score=68.0,
        short_score=31.0,
        no_trade_score=44.0,
        decision="LONG_BIAS",
        quality="B",
        contributions=[Contribution("timeframe_alignment", 12.0), Contribution("low_volume", -7.0)],
    )
    defaults.update(overrides)
    return ScoreResult(**defaults)


def _trade_plan(**overrides) -> TradePlan:
    defaults = dict(
        signal="LONG",
        entry_status="ENTER_NOW",
        confidence=70,
        market_regime="TREND_UP",
        entry=EntryZone(type="LIMIT_ZONE", from_=100.0, to=101.0, trigger="x"),
        stop_loss=98.0,
        take_profits=[TakeProfit(label="TP1", price=105.0, close_percent=100)],
        time_horizon_minutes=30,
        valid_for_minutes=15,
        reasons=["r"],
        risks=["k"],
        invalidation_conditions=["c"],
        wait_conditions=[],
        contradictions=[],
        missing_context=[],
        summary="s",
    )
    defaults.update(overrides)
    return TradePlan(**defaults)


def _ai_result(**overrides) -> AIAnalysisResult:
    defaults = dict(
        label="GPT-4o mini",
        model="gpt-4o-mini",
        content="{}",
        error=None,
        latency_seconds=1.2,
        created_at=NOW_EPOCH,
        trade_plan=_trade_plan(),
        validation=ValidationResult(status="valid", issues=[]),
        error_code=None,
        repaired=False,
    )
    defaults.update(overrides)
    return AIAnalysisResult(**defaults)


def _consensus_result(with_plan: bool, **overrides) -> ConsensusResult:
    plan = (
        SelectedPlan(
            source_label="GPT-4o mini",
            entry_status="ENTER_NOW",
            entry_type="LIMIT_ZONE",
            entry_from=100.0,
            entry_to=101.0,
            stop_loss=98.0,
            take_profits=[("TP1", 105.0, 100.0)],
            risk_reward_tp1=2.5,
            time_horizon_minutes=30,
            valid_for_minutes=15,
            formed_at=NOW_EPOCH,
        )
        if with_plan
        else None
    )
    defaults = dict(
        mode="scalping",
        overall_signal="LONG" if with_plan else "WAIT",
        state="strong",
        agreement_fraction=1.0,
        agreeing_count=3,
        vote_count=3,
        total_models=3,
        avg_confidence=70.0,
        plan=plan,
        trade_permission="ALLOWED" if with_plan else "WAIT",
        trade_permission_reason="ok",
        reasons=["r"],
        risks=["k"],
        wait_or_invalidation=[],
    )
    defaults.update(overrides)
    return ConsensusResult(**defaults)


def _position_calculation():
    scenario = TradeScenario(
        signal="LONG",
        entry_from=Decimal("100.0"),
        entry_to=Decimal("101.0"),
        stop_loss=Decimal("98.0"),
        take_profits=[TakeProfitTarget("TP1", Decimal("105.0"), Decimal("100"))],
    )
    return PositionCalculator(DEFAULT_RISK_SETTINGS).calculate(scenario)


def _bingx_fields(calculation):
    return build_bingx_manual_fields(calculation, "LIMIT", DEFAULT_RISK_SETTINGS)


def _simulated_outcome(**overrides) -> SimulatedOutcome:
    defaults = dict(
        exit_reason="TP1",
        r_multiple=2.3,
        mfe_r=2.5,
        mae_r=-0.3,
        exit_price=105.0,
        exit_time=NOW_EPOCH + 600,
        duration_seconds=600.0,
    )
    defaults.update(overrides)
    return SimulatedOutcome(**defaults)


def _score_chain(conn) -> int:
    """market_snapshot -> feature_snapshot -> strategy_score, for tests that
    only care about what's attached below strategy_scores."""
    snap_id = journal_db.insert_market_snapshot(conn, _market_snapshot())
    fs_id = journal_db.insert_feature_snapshot(conn, _feature_set(), market_snapshot_id=snap_id)
    return journal_db.insert_strategy_score(conn, _score_result(), _regime_result(), feature_snapshot_id=fs_id)


def _plan_id_for_outcome(conn) -> int:
    score_id = _score_chain(conn)
    return journal_db.insert_trade_plan(
        conn, _consensus_result(with_plan=True), strategy_score_id=score_id, symbol="ETHUSDT", timestamp=NOW_EPOCH
    )


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


def test_create_schema_creates_all_13_tables(conn):
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    expected = {
        "market_snapshots", "feature_snapshots", "strategy_scores", "ai_predictions", "trade_plans",
        "paper_orders", "real_orders", "positions", "fills", "trade_outcomes",
        "model_statistics", "strategy_statistics", "system_events",
    }
    assert expected <= tables


def test_init_db_twice_is_idempotent(conn):
    snap_id = journal_db.insert_market_snapshot(conn, _market_snapshot())
    journal_db.init_db(conn)
    assert journal_db.get_market_snapshot(conn, snap_id) is not None


def test_foreign_keys_enforced(conn):
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO feature_snapshots (market_snapshot_id, symbol, timestamp, data_quality, "
            "timeframe_alignment, trade_flow_buy_volume, trade_flow_sell_volume, trade_flow_delta, "
            "trade_flow_cvd, created_at) VALUES (999999, 'ETHUSDT', ?, 'GOOD', 0.5, 0, 0, 0, 0, ?)",
            (NOW_EPOCH, NOW_EPOCH),
        )


# ---------------------------------------------------------------------------
# market_snapshots
# ---------------------------------------------------------------------------


def test_market_snapshot_round_trip_full(conn):
    snap = _market_snapshot()
    snap_id = journal_db.insert_market_snapshot(conn, snap)
    row = journal_db.get_market_snapshot(conn, snap_id)
    assert row["symbol"] == "ETHUSDT"
    assert row["price"] == 100.5
    assert row["data_quality"] == "GOOD"
    assert row["funding_history"] == snap.funding_history
    assert row["orderbook"] == snap.orderbook
    assert row["recent_trades"] == snap.recent_trades
    assert row["instrument_rules"] == snap.instrument_rules


def test_market_snapshot_round_trip_none_optionals(conn):
    snap = _market_snapshot(
        price=None, bid=None, ask=None, spread=None, spread_percent=None,
        funding_rate=None, open_interest=None, volume_24h=None, orderbook=None, instrument_rules=None,
    )
    snap_id = journal_db.insert_market_snapshot(conn, snap)
    row = journal_db.get_market_snapshot(conn, snap_id)
    assert row["price"] is None
    assert row["orderbook"] is None
    assert row["instrument_rules"] is None


def test_market_snapshot_round_trip_empty_lists(conn):
    snap = _market_snapshot(funding_history=[], recent_trades=[], open_interest_history=[], orderbook_history=[])
    snap_id = journal_db.insert_market_snapshot(conn, snap)
    row = journal_db.get_market_snapshot(conn, snap_id)
    assert row["funding_history"] == []
    assert row["recent_trades"] == []


def test_market_snapshot_timeframes_drops_dataframe_keeps_flags(conn):
    snap_id = journal_db.insert_market_snapshot(conn, _market_snapshot())
    row = journal_db.get_market_snapshot(conn, snap_id)
    tf = row["timeframes"]["1m"]
    assert "dataframe" not in tf
    assert tf["has_gap"] is False
    assert tf["is_stale"] is False
    assert tf["zero_volume"] is False
    assert len(tf["candles"]) == 1


def test_get_recent_market_snapshots_orders_newest_first_and_respects_limit(conn):
    for i in range(5):
        ts = datetime.fromtimestamp(NOW_EPOCH + i, tz=timezone.utc)
        journal_db.insert_market_snapshot(conn, _market_snapshot(price=100.0 + i, timestamp=ts))
    rows = journal_db.get_recent_market_snapshots(conn, "ETHUSDT", limit=3)
    assert len(rows) == 3
    assert [r["price"] for r in rows] == [104.0, 103.0, 102.0]


# ---------------------------------------------------------------------------
# feature_snapshots
# ---------------------------------------------------------------------------


def test_feature_snapshot_round_trip_full(conn):
    snap_id = journal_db.insert_market_snapshot(conn, _market_snapshot())
    fs_id = journal_db.insert_feature_snapshot(conn, _feature_set(), market_snapshot_id=snap_id)
    row = journal_db.get_feature_snapshot(conn, fs_id)
    assert row["market_snapshot_id"] == snap_id
    assert row["data_quality"] == "GOOD"
    assert row["trend"]["1m"]["adx"] == 25.0
    assert row["momentum"]["1m"]["rsi14"] == 55.0
    assert row["trade_flow_buy_volume"] == 5.0
    assert row["oi_price_regime"] == "NEW_LONGS"
    assert row["orderbook_imbalance"] == 0.1
    assert row["orderbook_has_large_wall"] is False


def test_feature_snapshot_round_trip_empty_no_trade(conn):
    snap_id = journal_db.insert_market_snapshot(conn, _market_snapshot(data_quality="NO_TRADE"))
    fs_id = journal_db.insert_feature_snapshot(conn, _empty_feature_set(), market_snapshot_id=snap_id)
    row = journal_db.get_feature_snapshot(conn, fs_id)
    assert row["trend"] == {}
    assert row["trade_flow_buy_volume"] == 0.0
    assert row["oi_price_regime"] is None
    assert row["orderbook_has_large_wall"] is None


def test_feature_snapshot_links_to_market_snapshot(conn):
    snap_id = journal_db.insert_market_snapshot(conn, _market_snapshot())
    fs_id = journal_db.insert_feature_snapshot(conn, _feature_set(), market_snapshot_id=snap_id)
    row = journal_db.get_feature_snapshot(conn, fs_id)
    assert row["market_snapshot_id"] == snap_id


def test_feature_snapshot_bad_fk_raises(conn):
    with pytest.raises(sqlite3.IntegrityError):
        journal_db.insert_feature_snapshot(conn, _feature_set(), market_snapshot_id=999999)


# ---------------------------------------------------------------------------
# strategy_scores
# ---------------------------------------------------------------------------


def test_strategy_score_round_trip_with_regime(conn):
    snap_id = journal_db.insert_market_snapshot(conn, _market_snapshot())
    fs_id = journal_db.insert_feature_snapshot(conn, _feature_set(), market_snapshot_id=snap_id)
    score_id = journal_db.insert_strategy_score(conn, _score_result(), _regime_result(), feature_snapshot_id=fs_id)
    row = journal_db.get_strategy_score(conn, score_id)
    assert row["regime"] == "TREND_UP"
    assert row["regime_by_timeframe"] == {"1m": "TREND_UP"}
    assert row["regime_strategy_hint"] == "hint"
    assert row["decision"] == "LONG_BIAS"


def test_strategy_score_contributions_round_trip_including_empty(conn):
    snap_id = journal_db.insert_market_snapshot(conn, _market_snapshot())
    fs_id = journal_db.insert_feature_snapshot(conn, _feature_set(), market_snapshot_id=snap_id)
    score_id = journal_db.insert_strategy_score(conn, _score_result(), _regime_result(), feature_snapshot_id=fs_id)
    row = journal_db.get_strategy_score(conn, score_id)
    assert row["contributions"] == [
        {"feature": "timeframe_alignment", "score": 12.0},
        {"feature": "low_volume", "score": -7.0},
    ]

    fs_id2 = journal_db.insert_feature_snapshot(conn, _empty_feature_set(), market_snapshot_id=snap_id)
    score_id2 = journal_db.insert_strategy_score(
        conn,
        _score_result(contributions=[], decision="NO_TRADE", quality="D"),
        _regime_result(regime="NO_DATA", regime_by_timeframe={}, reasons=[]),
        feature_snapshot_id=fs_id2,
    )
    row2 = journal_db.get_strategy_score(conn, score_id2)
    assert row2["contributions"] == []


def test_strategy_score_no_data_result_round_trip(conn):
    snap_id = journal_db.insert_market_snapshot(conn, _market_snapshot(data_quality="NO_TRADE"))
    fs_id = journal_db.insert_feature_snapshot(conn, _empty_feature_set(), market_snapshot_id=snap_id)
    score_id = journal_db.insert_strategy_score(
        conn,
        _score_result(contributions=[], long_score=0, short_score=0, no_trade_score=100, decision="NO_TRADE", quality="D"),
        _regime_result(regime="NO_DATA", regime_by_timeframe={}, reasons=[], strategy_hint=""),
        feature_snapshot_id=fs_id,
    )
    row = journal_db.get_strategy_score(conn, score_id)
    assert row["regime_by_timeframe"] == {}
    assert row["no_trade_score"] == 100


def test_strategy_score_check_constraint_rejects_invalid_mode(conn):
    snap_id = journal_db.insert_market_snapshot(conn, _market_snapshot())
    fs_id = journal_db.insert_feature_snapshot(conn, _feature_set(), market_snapshot_id=snap_id)
    with pytest.raises(sqlite3.IntegrityError):
        journal_db.insert_strategy_score(
            conn, _score_result(mode="invalid_mode"), _regime_result(), feature_snapshot_id=fs_id
        )


# ---------------------------------------------------------------------------
# ai_predictions
# ---------------------------------------------------------------------------


def test_ai_prediction_round_trip_full_plan(conn):
    score_id = _score_chain(conn)
    result = _ai_result()
    pred_id = journal_db.insert_ai_prediction(conn, result, strategy_score_id=score_id, symbol="ETHUSDT", mode="scalping")
    rows = journal_db.get_ai_predictions_for_score(conn, score_id)
    assert len(rows) == 1
    row = rows[0]
    assert row["id"] == pred_id
    assert row["signal"] == "LONG"
    assert row["entry_from"] == 100.0
    assert row["take_profits"] == [{"label": "TP1", "price": 105.0, "close_percent": 100.0}]
    assert row["reasons"] == ["r"]
    assert row["ok"] is True
    assert row["validation_status"] == "valid"


def test_ai_prediction_round_trip_wait_no_plan(conn):
    score_id = _score_chain(conn)
    result = _ai_result(trade_plan=None, error="timeout", validation=None)
    journal_db.insert_ai_prediction(conn, result, strategy_score_id=score_id, symbol="ETHUSDT", mode="scalping")
    row = journal_db.get_ai_predictions_for_score(conn, score_id)[0]
    assert row["signal"] is None
    assert row["entry_from"] is None
    assert row["take_profits"] == []
    assert row["ok"] is False
    assert row["validation_status"] is None


def test_ai_prediction_validation_issues_round_trip(conn):
    score_id = _score_chain(conn)
    validation = ValidationResult(
        status="warning", issues=[ValidationIssue(severity="warning", code="STOP_TOO_TIGHT", message="m")]
    )
    result = _ai_result(validation=validation)
    journal_db.insert_ai_prediction(conn, result, strategy_score_id=score_id, symbol="ETHUSDT", mode="scalping")
    row = journal_db.get_ai_predictions_for_score(conn, score_id)[0]
    assert row["validation_status"] == "warning"
    assert row["validation_issues"] == [{"severity": "warning", "code": "STOP_TOO_TIGHT", "message": "m"}]


def test_get_ai_predictions_for_score_returns_all_in_order(conn):
    score_id = _score_chain(conn)
    for label in ("A", "B", "C"):
        journal_db.insert_ai_prediction(
            conn, _ai_result(label=label), strategy_score_id=score_id, symbol="ETHUSDT", mode="scalping"
        )
    rows = journal_db.get_ai_predictions_for_score(conn, score_id)
    assert [r["label"] for r in rows] == ["A", "B", "C"]


def test_ai_prediction_repaired_and_error_code_round_trip(conn):
    score_id = _score_chain(conn)
    result = _ai_result(repaired=True, error_code="JSON_REPAIRED")
    journal_db.insert_ai_prediction(conn, result, strategy_score_id=score_id, symbol="ETHUSDT", mode="scalping")
    row = journal_db.get_ai_predictions_for_score(conn, score_id)[0]
    assert row["repaired"] is True
    assert row["error_code"] == "JSON_REPAIRED"


# ---------------------------------------------------------------------------
# trade_plans
# ---------------------------------------------------------------------------


def test_trade_plan_wait_consensus_no_plan(conn):
    score_id = _score_chain(conn)
    plan_id = journal_db.insert_trade_plan(
        conn, _consensus_result(with_plan=False), strategy_score_id=score_id, symbol="ETHUSDT", timestamp=NOW_EPOCH
    )
    row = journal_db.get_trade_plan(conn, plan_id)
    assert row["overall_signal"] == "WAIT"
    assert row["entry_from"] is None
    assert row["position_status"] is None


def test_trade_plan_long_full_with_calculation_decimal_exact(conn):
    score_id = _score_chain(conn)
    calculation = _position_calculation()
    plan_id = journal_db.insert_trade_plan(
        conn, _consensus_result(with_plan=True), strategy_score_id=score_id, symbol="ETHUSDT",
        timestamp=NOW_EPOCH, calculation=calculation,
    )
    row = journal_db.get_trade_plan(conn, plan_id)
    assert row["overall_signal"] == "LONG"
    assert row["entry_from"] == 100.0
    assert row["position_status"] == calculation.status.value
    assert row["position_size_coin"] == calculation.position_size_coin
    assert row["required_margin_usdt"] == calculation.required_margin_usdt
    assert row["entry_price_calc"] == calculation.entry_price
    assert isinstance(row["position_size_coin"], Decimal)


def test_trade_plan_calculation_none_position_columns_null(conn):
    score_id = _score_chain(conn)
    plan_id = journal_db.insert_trade_plan(
        conn, _consensus_result(with_plan=True), strategy_score_id=score_id, symbol="ETHUSDT", timestamp=NOW_EPOCH
    )
    row = journal_db.get_trade_plan(conn, plan_id)
    assert row["overall_signal"] == "LONG"
    assert row["position_size_coin"] is None
    assert row["position_status"] is None


def test_trade_plan_bingx_fields_round_trip(conn):
    score_id = _score_chain(conn)
    calculation = _position_calculation()
    bingx_fields = _bingx_fields(calculation)
    plan_id = journal_db.insert_trade_plan(
        conn, _consensus_result(with_plan=True), strategy_score_id=score_id, symbol="ETHUSDT",
        timestamp=NOW_EPOCH, calculation=calculation, bingx_fields=bingx_fields,
    )
    row = journal_db.get_trade_plan(conn, plan_id)
    assert row["bingx_fields"]["side"] == bingx_fields.side
    assert row["bingx_fields"]["margin_usdt"] == str(bingx_fields.margin_usdt)


def test_trade_plan_unique_strategy_score_id(conn):
    score_id = _score_chain(conn)
    journal_db.insert_trade_plan(
        conn, _consensus_result(with_plan=False), strategy_score_id=score_id, symbol="ETHUSDT", timestamp=NOW_EPOCH
    )
    with pytest.raises(sqlite3.IntegrityError):
        journal_db.insert_trade_plan(
            conn, _consensus_result(with_plan=False), strategy_score_id=score_id, symbol="ETHUSDT", timestamp=NOW_EPOCH
        )


def test_trade_plan_blended_json_populated_and_none(conn):
    score_id = _score_chain(conn)
    calculation = _position_calculation()
    plan_id = journal_db.insert_trade_plan(
        conn, _consensus_result(with_plan=True), strategy_score_id=score_id, symbol="ETHUSDT",
        timestamp=NOW_EPOCH, calculation=calculation,
    )
    row = journal_db.get_trade_plan(conn, plan_id)
    if calculation.blended is not None:
        assert row["blended"] is not None
    else:
        assert row["blended"] is None

    score_id2 = _score_chain(conn)
    plan_id2 = journal_db.insert_trade_plan(
        conn, _consensus_result(with_plan=False), strategy_score_id=score_id2, symbol="ETHUSDT", timestamp=NOW_EPOCH
    )
    row2 = journal_db.get_trade_plan(conn, plan_id2)
    assert row2["blended"] is None


# ---------------------------------------------------------------------------
# paper_orders / real_orders / positions / fills
# ---------------------------------------------------------------------------


def test_paper_order_round_trip_decimal_fields(conn):
    order_id = journal_db.insert_paper_order(
        conn, symbol="ETHUSDT", side="LONG", order_type="LIMIT", quantity=Decimal("0.5"),
        price=Decimal("100.25"), stop_loss=Decimal("98.0"), take_profit=Decimal("105.0"),
    )
    row = conn.execute("SELECT * FROM paper_orders WHERE id = ?", (order_id,)).fetchone()
    assert Decimal(row["quantity"]) == Decimal("0.5")
    assert Decimal(row["price"]) == Decimal("100.25")
    assert row["status"] == "PENDING"


def test_real_order_exchange_order_id_uniqueness(conn):
    journal_db.insert_real_order(
        conn, symbol="ETHUSDT", side="LONG", order_type="MARKET", quantity="0.5", exchange_order_id="ORD-1"
    )
    with pytest.raises(sqlite3.IntegrityError):
        journal_db.insert_real_order(
            conn, symbol="ETHUSDT", side="LONG", order_type="MARKET", quantity="0.5", exchange_order_id="ORD-1"
        )


def test_update_order_status(conn):
    order_id = journal_db.insert_paper_order(conn, symbol="ETHUSDT", side="LONG", order_type="LIMIT", quantity="0.5")
    journal_db.update_order_status(conn, "paper_orders", order_id, "FILLED", filled_quantity="0.5")
    row = conn.execute("SELECT status, filled_quantity FROM paper_orders WHERE id = ?", (order_id,)).fetchone()
    assert row["status"] == "FILLED"
    assert Decimal(row["filled_quantity"]) == Decimal("0.5")


def test_position_linked_via_paper_order_id(conn):
    order_id = journal_db.insert_paper_order(conn, symbol="ETHUSDT", side="LONG", order_type="MARKET", quantity="0.5")
    position_id = journal_db.insert_position(
        conn, symbol="ETHUSDT", side="LONG", source="PAPER", entry_price="100.0", quantity="0.5", paper_order_id=order_id
    )
    row = conn.execute("SELECT * FROM positions WHERE id = ?", (position_id,)).fetchone()
    assert row["paper_order_id"] == order_id
    assert row["real_order_id"] is None
    assert row["status"] == "OPEN"


def test_close_position(conn):
    position_id = journal_db.insert_position(
        conn, symbol="ETHUSDT", side="LONG", source="PAPER", entry_price="100.0", quantity="0.5"
    )
    journal_db.close_position(conn, position_id, exit_price="105.0", realized_pnl_usdt="2.5", fees_paid_usdt="0.1")
    row = conn.execute("SELECT * FROM positions WHERE id = ?", (position_id,)).fetchone()
    assert row["status"] == "CLOSED"
    assert Decimal(row["exit_price"]) == Decimal("105.0")
    assert row["closed_at"] is not None


def test_fill_linked_to_position(conn):
    position_id = journal_db.insert_position(
        conn, symbol="ETHUSDT", side="LONG", source="PAPER", entry_price="100.0", quantity="0.5"
    )
    fill_id = journal_db.insert_fill(
        conn, position_id=position_id, symbol="ETHUSDT", side="LONG", fill_type="ENTRY", price="100.0", quantity="0.5"
    )
    row = conn.execute("SELECT * FROM fills WHERE id = ?", (fill_id,)).fetchone()
    assert row["position_id"] == position_id


def test_fill_bad_position_fk_raises(conn):
    with pytest.raises(sqlite3.IntegrityError):
        journal_db.insert_fill(
            conn, position_id=999999, symbol="ETHUSDT", side="LONG", fill_type="ENTRY", price="100.0", quantity="0.5"
        )


# ---------------------------------------------------------------------------
# trade_outcomes
# ---------------------------------------------------------------------------


def test_insert_trade_outcome_pending(conn):
    plan_id = _plan_id_for_outcome(conn)
    outcome_id = journal_db.insert_trade_outcome_pending(
        conn, trade_plan_id=plan_id, symbol="ETHUSDT", mode="scalping", entry_price=100.5
    )
    row = journal_db.get_trade_outcome(conn, plan_id)
    assert row["id"] == outcome_id
    assert row["status"] == "PENDING"
    assert row["exit_reason"] is None
    assert row["price_after_1m"] is None


def test_price_checkpoints_independent(conn):
    plan_id = _plan_id_for_outcome(conn)
    outcome_id = journal_db.insert_trade_outcome_pending(conn, trade_plan_id=plan_id, symbol="ETHUSDT", mode="scalping")
    journal_db.update_trade_outcome_price_checkpoint(conn, outcome_id, minute=1, price=101.0)
    journal_db.update_trade_outcome_price_checkpoint(conn, outcome_id, minute=5, price=102.0)
    row = journal_db.get_trade_outcome(conn, plan_id)
    assert row["price_after_1m"] == 101.0
    assert row["price_after_5m"] == 102.0
    assert row["price_after_15m"] is None
    assert row["price_after_30m"] is None


def test_finalize_trade_outcome_round_trip(conn):
    plan_id = _plan_id_for_outcome(conn)
    outcome_id = journal_db.insert_trade_outcome_pending(conn, trade_plan_id=plan_id, symbol="ETHUSDT", mode="scalping")
    outcome = _simulated_outcome()
    journal_db.finalize_trade_outcome(
        conn, outcome_id, outcome, commission_usdt=0.5, slippage_usdt=0.1, realized_pnl_usdt=12.5
    )
    row = journal_db.get_trade_outcome(conn, plan_id)
    assert row["status"] == "EVALUATED"
    assert row["exit_reason"] == "TP1"
    assert row["r_multiple"] == 2.3
    assert row["mfe_r"] == 2.5
    assert row["mae_r"] == -0.3
    assert row["commission_usdt"] == 0.5
    assert row["realized_pnl_usdt"] == 12.5
    assert row["evaluated_at"] is not None


def test_trade_outcome_unique_trade_plan_id(conn):
    plan_id = _plan_id_for_outcome(conn)
    journal_db.insert_trade_outcome_pending(conn, trade_plan_id=plan_id, symbol="ETHUSDT", mode="scalping")
    with pytest.raises(sqlite3.IntegrityError):
        journal_db.insert_trade_outcome_pending(conn, trade_plan_id=plan_id, symbol="ETHUSDT", mode="scalping")


def test_trade_outcome_exit_reason_check_constraint(conn):
    plan_id = _plan_id_for_outcome(conn)
    outcome_id = journal_db.insert_trade_outcome_pending(conn, trade_plan_id=plan_id, symbol="ETHUSDT", mode="scalping")
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("UPDATE trade_outcomes SET exit_reason = 'BOGUS' WHERE id = ?", (outcome_id,))


# ---------------------------------------------------------------------------
# model_statistics / strategy_statistics
# ---------------------------------------------------------------------------


def test_model_statistics_round_trip(conn):
    stat_id = journal_db.insert_model_statistics(
        conn, model_label="GPT-4o mini", mode="scalping", window_start=NOW_EPOCH - 86400, window_end=NOW_EPOCH,
        total_predictions=20, evaluated_count=15, wins=9, win_rate=0.6, expectancy_r=0.4, median_r=0.3,
        profit_factor=1.8, profit_factor_undefined=False, max_drawdown_r=-1.2, avg_mfe_r=1.5, avg_mae_r=-0.5,
        avg_duration_seconds=600.0, exit_reason_counts={"SL": 5, "TP1": 9, "TIMEOUT": 1}, low_sample=False,
    )
    rows = journal_db.get_model_statistics(conn, model_label="GPT-4o mini", mode="scalping")
    assert len(rows) == 1
    row = rows[0]
    assert row["id"] == stat_id
    assert row["exit_reason_counts"] == {"SL": 5, "TP1": 9, "TIMEOUT": 1}
    assert row["profit_factor_undefined"] is False


def test_model_statistics_undefined_profit_factor_round_trip(conn):
    journal_db.insert_model_statistics(
        conn, model_label="Gemini 2.5 Pro", mode="swing", window_start=NOW_EPOCH - 86400, window_end=NOW_EPOCH,
        total_predictions=5, evaluated_count=3, wins=3, win_rate=1.0, expectancy_r=1.0, median_r=1.0,
        profit_factor=None, profit_factor_undefined=True, max_drawdown_r=0.0, avg_mfe_r=1.0, avg_mae_r=0.0,
        avg_duration_seconds=300.0, exit_reason_counts={}, low_sample=True,
    )
    rows = journal_db.get_model_statistics(conn, model_label="Gemini 2.5 Pro")
    row = rows[0]
    assert row["profit_factor"] is None
    assert row["profit_factor_undefined"] is True
    assert row["low_sample"] is True


def test_get_model_statistics_filters_by_label_and_mode(conn):
    common = dict(
        window_start=0, window_end=1, total_predictions=1, evaluated_count=1, wins=1, win_rate=1.0,
        expectancy_r=1.0, median_r=1.0, profit_factor=1.0, profit_factor_undefined=False,
        max_drawdown_r=0.0, avg_mfe_r=0.0, avg_mae_r=0.0, avg_duration_seconds=1.0,
        exit_reason_counts={}, low_sample=False,
    )
    journal_db.insert_model_statistics(conn, model_label="A", mode="scalping", **common)
    journal_db.insert_model_statistics(conn, model_label="B", mode="swing", **common)
    rows = journal_db.get_model_statistics(conn, model_label="A")
    assert len(rows) == 1
    assert rows[0]["model_label"] == "A"


def test_strategy_statistics_round_trip(conn):
    stat_id = journal_db.insert_strategy_statistics(
        conn, ruleset_version="v1", mode="scalping", decision="LONG_BIAS", window_start=NOW_EPOCH - 86400,
        window_end=NOW_EPOCH, total_signals=10, evaluated_count=8, wins=5, win_rate=0.625, expectancy_r=0.3,
        median_r=0.2, profit_factor=1.5, low_sample=False,
    )
    rows = journal_db.get_strategy_statistics(conn, ruleset_version="v1", mode="scalping")
    assert len(rows) == 1
    assert rows[0]["id"] == stat_id
    assert rows[0]["decision"] == "LONG_BIAS"


# ---------------------------------------------------------------------------
# system_events
# ---------------------------------------------------------------------------


def test_system_event_round_trip_with_fk_columns(conn):
    snap_id = journal_db.insert_market_snapshot(conn, _market_snapshot())
    event_id = journal_db.log_event(
        conn, level="WARNING", event_code="DATA_DEGRADED", source_module="market_data_engine",
        message="stale orderbook", symbol="ETHUSDT", details={"age_seconds": 45}, market_snapshot_id=snap_id,
    )
    rows = journal_db.get_recent_system_events(conn)
    row = next(r for r in rows if r["id"] == event_id)
    assert row["market_snapshot_id"] == snap_id
    assert row["details"] == {"age_seconds": 45}


def test_system_event_round_trip_all_fk_null(conn):
    event_id = journal_db.log_event(conn, level="INFO", event_code="STARTUP", source_module="server", message="started")
    rows = journal_db.get_recent_system_events(conn)
    row = next(r for r in rows if r["id"] == event_id)
    assert row["market_snapshot_id"] is None
    assert row["trade_plan_id"] is None
    assert row["details"] == {}


def test_get_recent_system_events_filters_by_level(conn):
    journal_db.log_event(conn, level="ERROR", event_code="X", source_module="m", message="err")
    journal_db.log_event(conn, level="INFO", event_code="Y", source_module="m", message="info")
    rows = journal_db.get_recent_system_events(conn, level="ERROR")
    assert all(r["level"] == "ERROR" for r in rows)
    assert len(rows) == 1


# ---------------------------------------------------------------------------
# Chaining / integration
# ---------------------------------------------------------------------------


def test_full_pipeline_chain_ids_threaded_correctly(conn):
    snap_id = journal_db.insert_market_snapshot(conn, _market_snapshot())
    fs_id = journal_db.insert_feature_snapshot(conn, _feature_set(), market_snapshot_id=snap_id)
    score_id = journal_db.insert_strategy_score(conn, _score_result(), _regime_result(), feature_snapshot_id=fs_id)
    pred_id_a = journal_db.insert_ai_prediction(
        conn, _ai_result(label="A"), strategy_score_id=score_id, symbol="ETHUSDT", mode="scalping"
    )
    pred_id_b = journal_db.insert_ai_prediction(
        conn, _ai_result(label="B"), strategy_score_id=score_id, symbol="ETHUSDT", mode="scalping"
    )
    plan_id = journal_db.insert_trade_plan(
        conn, _consensus_result(with_plan=True), strategy_score_id=score_id, symbol="ETHUSDT",
        timestamp=NOW_EPOCH, calculation=_position_calculation(),
    )
    outcome_id = journal_db.insert_trade_outcome_pending(conn, trade_plan_id=plan_id, symbol="ETHUSDT", mode="scalping")
    journal_db.finalize_trade_outcome(conn, outcome_id, _simulated_outcome())

    assert journal_db.get_feature_snapshot(conn, fs_id)["market_snapshot_id"] == snap_id
    assert journal_db.get_strategy_score(conn, score_id)["feature_snapshot_id"] == fs_id
    preds = journal_db.get_ai_predictions_for_score(conn, score_id)
    assert {p["id"] for p in preds} == {pred_id_a, pred_id_b}
    assert journal_db.get_trade_plan(conn, plan_id)["strategy_score_id"] == score_id
    assert journal_db.get_trade_outcome(conn, plan_id)["status"] == "EVALUATED"


def test_get_full_signal_reconstructs_chain(conn):
    snap_id = journal_db.insert_market_snapshot(conn, _market_snapshot())
    fs_id = journal_db.insert_feature_snapshot(conn, _feature_set(), market_snapshot_id=snap_id)
    score_id = journal_db.insert_strategy_score(conn, _score_result(), _regime_result(), feature_snapshot_id=fs_id)
    journal_db.insert_ai_prediction(conn, _ai_result(), strategy_score_id=score_id, symbol="ETHUSDT", mode="scalping")
    plan_id = journal_db.insert_trade_plan(
        conn, _consensus_result(with_plan=True), strategy_score_id=score_id, symbol="ETHUSDT", timestamp=NOW_EPOCH
    )
    outcome_id = journal_db.insert_trade_outcome_pending(conn, trade_plan_id=plan_id, symbol="ETHUSDT", mode="scalping")
    journal_db.finalize_trade_outcome(conn, outcome_id, _simulated_outcome())

    full = journal_db.get_full_signal(conn, score_id)
    assert full["market_snapshot"]["id"] == snap_id
    assert full["feature_snapshot"]["id"] == fs_id
    assert full["strategy_score"]["id"] == score_id
    assert len(full["ai_predictions"]) == 1
    assert full["trade_plan"]["id"] == plan_id
    assert full["trade_outcome"]["status"] == "EVALUATED"


def test_connect_rolls_back_on_exception(tmp_path):
    db_path = tmp_path / "rollback_test.db"
    try:
        with journal_db.connect(db_path=db_path) as c:
            journal_db.insert_market_snapshot(c, _market_snapshot())
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    check_conn = journal_db.get_connection(db_path=db_path)
    count = check_conn.execute("SELECT COUNT(*) FROM market_snapshots").fetchone()[0]
    check_conn.close()
    assert count == 0


# ---------------------------------------------------------------------------
# Decimal / datetime helpers
# ---------------------------------------------------------------------------


def test_dec_text_round_trip_precise():
    value = Decimal("0.123456789012345")
    assert journal_db._text_to_dec(journal_db._dec_to_text(value)) == value


def test_dt_epoch_round_trip():
    dt = datetime(2026, 3, 5, 8, 30, 0, tzinfo=timezone.utc)
    epoch = journal_db._dt_to_epoch(dt)
    back = journal_db._epoch_to_dt(epoch)
    assert back == dt


# ---------------------------------------------------------------------------
# Stage 8 schema tweaks: paper_orders entry_from/to + UNIQUE, fills TIMEOUT +
# label, strategy_statistics.regime
# ---------------------------------------------------------------------------


def test_paper_order_entry_from_to_round_trip(conn):
    order_id = journal_db.insert_paper_order(
        conn, symbol="ETHUSDT", side="LONG", order_type="LIMIT", quantity="0.5",
        entry_from=100.0, entry_to=101.0,
    )
    row = conn.execute("SELECT entry_from, entry_to FROM paper_orders WHERE id = ?", (order_id,)).fetchone()
    assert row["entry_from"] == 100.0
    assert row["entry_to"] == 101.0


def test_paper_order_trade_plan_id_unique(conn):
    plan_id = _plan_id_for_outcome(conn)
    journal_db.insert_paper_order(conn, symbol="ETHUSDT", side="LONG", order_type="LIMIT", quantity="0.5", trade_plan_id=plan_id)
    with pytest.raises(sqlite3.IntegrityError):
        journal_db.insert_paper_order(conn, symbol="ETHUSDT", side="LONG", order_type="LIMIT", quantity="0.5", trade_plan_id=plan_id)


def test_paper_order_trade_plan_id_null_allows_multiple(conn):
    journal_db.insert_paper_order(conn, symbol="ETHUSDT", side="LONG", order_type="LIMIT", quantity="0.5")
    journal_db.insert_paper_order(conn, symbol="ETHUSDT", side="LONG", order_type="LIMIT", quantity="0.5")
    count = conn.execute("SELECT COUNT(*) FROM paper_orders WHERE trade_plan_id IS NULL").fetchone()[0]
    assert count == 2


def test_fill_type_timeout_accepted_and_label_round_trip(conn):
    position_id = journal_db.insert_position(conn, symbol="ETHUSDT", side="LONG", source="PAPER", entry_price="100.0", quantity="0.5")
    fill_id = journal_db.insert_fill(
        conn, position_id=position_id, symbol="ETHUSDT", side="LONG", fill_type="TIMEOUT", price="99.0", quantity="0.5"
    )
    row = conn.execute("SELECT fill_type, label FROM fills WHERE id = ?", (fill_id,)).fetchone()
    assert row["fill_type"] == "TIMEOUT"
    assert row["label"] is None


def test_fill_label_round_trip_for_take_profit(conn):
    position_id = journal_db.insert_position(conn, symbol="ETHUSDT", side="LONG", source="PAPER", entry_price="100.0", quantity="0.5")
    fill_id = journal_db.insert_fill(
        conn, position_id=position_id, symbol="ETHUSDT", side="LONG", fill_type="TAKE_PROFIT",
        price="105.0", quantity="0.25", label="TP1",
    )
    row = conn.execute("SELECT label FROM fills WHERE id = ?", (fill_id,)).fetchone()
    assert row["label"] == "TP1"


def test_fill_type_invalid_still_rejected(conn):
    position_id = journal_db.insert_position(conn, symbol="ETHUSDT", side="LONG", source="PAPER", entry_price="100.0", quantity="0.5")
    with pytest.raises(sqlite3.IntegrityError):
        journal_db.insert_fill(conn, position_id=position_id, symbol="ETHUSDT", side="LONG", fill_type="BOGUS", price="100.0", quantity="0.5")


def test_strategy_statistics_regime_round_trip(conn):
    stat_id = journal_db.insert_strategy_statistics(
        conn, ruleset_version="v1", mode="scalping", decision=None, regime="TREND_UP",
        window_start=0, window_end=1, total_signals=5, evaluated_count=5, wins=3,
        win_rate=60.0, expectancy_r=0.2, median_r=0.1, profit_factor=1.5, low_sample=False,
    )
    rows = journal_db.get_strategy_statistics(conn, regime="TREND_UP")
    assert len(rows) == 1
    assert rows[0]["id"] == stat_id
    assert rows[0]["decision"] is None
    assert rows[0]["regime"] == "TREND_UP"


def test_strategy_statistics_regime_defaults_null(conn):
    journal_db.insert_strategy_statistics(
        conn, ruleset_version="v1", mode="scalping", decision="LONG_BIAS",
        window_start=0, window_end=1, total_signals=1, evaluated_count=1, wins=1,
        win_rate=100.0, expectancy_r=1.0, median_r=1.0, profit_factor=None, low_sample=True,
    )
    rows = journal_db.get_strategy_statistics(conn, ruleset_version="v1", mode="scalping")
    assert rows[0]["regime"] is None


def test_strategy_statistics_regime_check_constraint(conn):
    with pytest.raises(sqlite3.IntegrityError):
        journal_db.insert_strategy_statistics(
            conn, ruleset_version="v1", mode="scalping", decision=None, regime="NOT_A_REGIME",
            window_start=0, window_end=1, total_signals=1, evaluated_count=0, wins=0,
            win_rate=None, expectancy_r=None, median_r=None, profit_factor=None, low_sample=True,
        )


# ---------------------------------------------------------------------------
# Stage 8 new journal_db.py functions
# ---------------------------------------------------------------------------


def test_price_checkpoint_idempotent_on_repeated_call(conn):
    plan_id = _plan_id_for_outcome(conn)
    outcome_id = journal_db.insert_trade_outcome_pending(conn, trade_plan_id=plan_id, symbol="ETHUSDT", mode="scalping")
    journal_db.update_trade_outcome_price_checkpoint(conn, outcome_id, minute=1, price=101.0)
    journal_db.update_trade_outcome_price_checkpoint(conn, outcome_id, minute=1, price=999.0)
    row = journal_db.get_trade_outcome(conn, plan_id)
    assert row["price_after_1m"] == 101.0


def test_update_trade_outcome_running_extremes_monotonic_max(conn):
    plan_id = _plan_id_for_outcome(conn)
    outcome_id = journal_db.insert_trade_outcome_pending(conn, trade_plan_id=plan_id, symbol="ETHUSDT", mode="scalping")
    journal_db.update_trade_outcome_running_extremes(conn, outcome_id, favorable_r=0.5, adverse_r=0.1)
    journal_db.update_trade_outcome_running_extremes(conn, outcome_id, favorable_r=1.2, adverse_r=0.05)
    journal_db.update_trade_outcome_running_extremes(conn, outcome_id, favorable_r=0.8, adverse_r=0.3)
    row = journal_db.get_trade_outcome(conn, plan_id)
    assert row["mfe_r"] == 1.2
    assert row["mae_r"] == 0.3


def test_update_trade_outcome_running_extremes_handles_negative_first_reading(conn):
    plan_id = _plan_id_for_outcome(conn)
    outcome_id = journal_db.insert_trade_outcome_pending(conn, trade_plan_id=plan_id, symbol="ETHUSDT", mode="scalping")
    journal_db.update_trade_outcome_running_extremes(conn, outcome_id, favorable_r=-0.2, adverse_r=0.4)
    row = journal_db.get_trade_outcome(conn, plan_id)
    assert row["mfe_r"] == -0.2
    assert row["mae_r"] == 0.4


def test_update_position_stop_loss(conn):
    position_id = journal_db.insert_position(conn, symbol="ETHUSDT", side="LONG", source="PAPER", entry_price="100.0", quantity="0.5", stop_loss="98.0")
    journal_db.update_position_stop_loss(conn, position_id, Decimal("100.0"))
    row = conn.execute("SELECT stop_loss FROM positions WHERE id = ?", (position_id,)).fetchone()
    assert Decimal(row["stop_loss"]) == Decimal("100.0")


def test_update_position_add_fill_weighted_average(conn):
    position_id = journal_db.insert_position(conn, symbol="ETHUSDT", side="LONG", source="PAPER", entry_price="100.0", quantity="1.0")
    journal_db.update_position_add_fill(conn, position_id, additional_quantity=Decimal("1.0"), fill_price=Decimal("102.0"))
    row = conn.execute("SELECT entry_price, quantity FROM positions WHERE id = ?", (position_id,)).fetchone()
    # (100*1 + 102*1) / 2 = 101 exactly
    assert Decimal(row["entry_price"]) == Decimal("101")
    assert Decimal(row["quantity"]) == Decimal("2.0")


def test_get_pending_paper_orders_filters_status(conn):
    a = journal_db.insert_paper_order(conn, symbol="ETHUSDT", side="LONG", order_type="LIMIT", quantity="0.5", status="PENDING")
    b = journal_db.insert_paper_order(conn, symbol="ETHUSDT", side="LONG", order_type="LIMIT", quantity="0.5")
    journal_db.update_order_status(conn, "paper_orders", b, "PARTIALLY_FILLED", filled_quantity="0.1")
    c = journal_db.insert_paper_order(conn, symbol="ETHUSDT", side="LONG", order_type="LIMIT", quantity="0.5")
    journal_db.update_order_status(conn, "paper_orders", c, "FILLED", filled_quantity="0.5")
    rows = journal_db.get_pending_paper_orders(conn, "ETHUSDT")
    assert {r["id"] for r in rows} == {a, b}


def test_get_open_positions_filters_status_and_source(conn):
    open_id = journal_db.insert_position(conn, symbol="ETHUSDT", side="LONG", source="PAPER", entry_price="100.0", quantity="0.5")
    closed_id = journal_db.insert_position(conn, symbol="ETHUSDT", side="LONG", source="PAPER", entry_price="100.0", quantity="0.5")
    journal_db.close_position(conn, closed_id, exit_price="101.0")
    real_id = journal_db.insert_position(conn, symbol="ETHUSDT", side="LONG", source="REAL", entry_price="100.0", quantity="0.5")
    rows = journal_db.get_open_positions(conn, "ETHUSDT")
    assert {r["id"] for r in rows} == {open_id}
    assert closed_id not in {r["id"] for r in rows}
    assert real_id not in {r["id"] for r in rows}


def test_get_position_fills_chronological(conn):
    position_id = journal_db.insert_position(conn, symbol="ETHUSDT", side="LONG", source="PAPER", entry_price="100.0", quantity="0.5")
    f1 = journal_db.insert_fill(conn, position_id=position_id, symbol="ETHUSDT", side="LONG", fill_type="ENTRY", price="100.0", quantity="0.5", now=1000.0)
    f2 = journal_db.insert_fill(conn, position_id=position_id, symbol="ETHUSDT", side="LONG", fill_type="TAKE_PROFIT", price="105.0", quantity="0.25", label="TP1", now=1010.0)
    rows = journal_db.get_position_fills(conn, position_id)
    assert [r["id"] for r in rows] == [f1, f2]


def test_get_paper_order_by_trade_plan_id(conn):
    plan_id = _plan_id_for_outcome(conn)
    order_id = journal_db.insert_paper_order(conn, symbol="ETHUSDT", side="LONG", order_type="LIMIT", quantity="0.5", trade_plan_id=plan_id)
    row = journal_db.get_paper_order_by_trade_plan_id(conn, plan_id)
    assert row["id"] == order_id
    assert journal_db.get_paper_order_by_trade_plan_id(conn, plan_id + 999) is None


def test_get_position_by_paper_order_id(conn):
    order_id = journal_db.insert_paper_order(conn, symbol="ETHUSDT", side="LONG", order_type="LIMIT", quantity="0.5")
    position_id = journal_db.insert_position(conn, symbol="ETHUSDT", side="LONG", source="PAPER", entry_price="100.0", quantity="0.5", paper_order_id=order_id)
    row = journal_db.get_position_by_paper_order_id(conn, order_id)
    assert row["id"] == position_id
    journal_db.close_position(conn, position_id, exit_price="101.0")
    assert journal_db.get_position_by_paper_order_id(conn, order_id) is None
