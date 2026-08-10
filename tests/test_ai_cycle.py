"""Offline tests for runtime/ai_cycle.py -- the bridge from a scheduled AI
analysis to an actual paper trade (market_snapshots -> feature_snapshots ->
strategy_scores -> trade_plans -> paper_orders).

Every network/AI/report-building call is monkeypatched; journal_db runs for
real against a tmp_path SQLite file, same convention as
test_trading_runtime.py. `position_service.calculate_active_position` and
`paper_trading.open_virtual_order` are deliberately NOT mocked -- proving
those wire together correctly (a real, actionable consensus actually opens
a paper order; a WAIT one doesn't) is the whole point of this module.
"""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

import pytest

import ai_client
import config
import feature_engine
import journal_db
import market_data_engine
import market_regime
import position_service
import risk_settings_store
import strategy_engine
from ai_client import AIAnalysisResult
from ai_schema import EntryZone, TakeProfit, TradePlan
from feature_engine import (
    FeatureSet,
    FuturesFeatures,
    OrderbookFeatures,
    TradeFlowFeatures,
)
from market_data_engine import MarketDataSnapshot
from market_regime import RegimeResult
from risk_manager import EntryPriceMode, InstrumentRules, MarginMode, RiskSettings
from runtime import ai_cycle
from strategy_engine import Contribution, ScoreResult
from trade_validator import ValidationResult

NOW = 1_700_000_000.0
NOW_DT = datetime.fromtimestamp(NOW, tz=timezone.utc)

RULES = InstrumentRules(
    symbol="ETH-USDT",
    price_step=Decimal("0.01"),
    quantity_step=Decimal("0.001"),
    minimum_quantity=Decimal("0"),
    minimum_notional_usdt=Decimal("2"),
    maximum_leverage=0,
    source="FALLBACK",
)


@pytest.fixture(autouse=True)
def _use_tmp_paths(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "JOURNAL_DB_FILE", tmp_path / "ai_cycle_test.db")
    monkeypatch.setattr(config, "PREDICTION_HISTORY_FILE", tmp_path / "predictions.jsonl")


@pytest.fixture
def conn(_use_tmp_paths):
    c = journal_db.init_db()
    yield c
    c.close()


def _market_snapshot() -> MarketDataSnapshot:
    return MarketDataSnapshot(
        timestamp=NOW_DT,
        symbol="ETHUSDT",
        price=100.5,
        bid=100.4,
        ask=100.6,
        spread=0.2,
        spread_percent=0.2,
        timeframes={},
        funding_rate=0.0001,
        funding_history=[],
        open_interest=123456.0,
        open_interest_history=[],
        orderbook=None,
        orderbook_history=[],
        volume_24h=999.0,
        recent_trades=[],
        instrument_rules=None,
        data_quality="GOOD",
        quality_issues=[],
    )


def _empty_feature_set() -> FeatureSet:
    return FeatureSet(
        timestamp=NOW_DT,
        symbol="ETHUSDT",
        data_quality="GOOD",
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


def _regime_result() -> RegimeResult:
    return RegimeResult(
        timestamp=NOW_DT, symbol="ETHUSDT", regime="TREND_UP",
        regime_by_timeframe={"1m": "TREND_UP"}, reasons=["1m: TREND_UP"], strategy_hint="hint",
    )


def _score_result() -> ScoreResult:
    return ScoreResult(
        timestamp=NOW_DT, symbol="ETHUSDT", mode="scalping", ruleset_version="v1",
        long_score=68.0, short_score=31.0, no_trade_score=44.0, decision="LONG_BIAS", quality="B",
        contributions=[Contribution("timeframe_alignment", 12.0)],
    )


def _plan(signal="LONG", entry_from=100.0, entry_to=101.0, stop_loss=95.0, tp1=115.0, entry_status="ENTER_NOW") -> TradePlan:
    return TradePlan(
        signal=signal, entry_status=entry_status, confidence=70, market_regime="TREND_UP",
        entry=EntryZone(type="LIMIT", from_=entry_from, to=entry_to, trigger="x"),
        stop_loss=stop_loss, take_profits=[TakeProfit(label="TP1", price=tp1, close_percent=100)],
        time_horizon_minutes=60, valid_for_minutes=120, reasons=["r"], risks=["k"],
        invalidation_conditions=["c"], wait_conditions=[], summary="s",
    )


def _wait_plan() -> TradePlan:
    return TradePlan(
        signal="WAIT", entry_status="NO_TRADE", confidence=40, market_regime="RANGE",
        entry=EntryZone(type="NONE", from_=0.0, to=0.0, trigger=""), stop_loss=0.0, take_profits=[],
        time_horizon_minutes=60, valid_for_minutes=120, reasons=[], risks=[],
        invalidation_conditions=[], wait_conditions=["wait"], summary="s",
    )


def _votes(plan: TradePlan, count: int = 3) -> list[AIAnalysisResult]:
    labels = ["A", "B", "C"][:count]
    return [
        AIAnalysisResult(
            label=label, model=label.lower(), content="{}", error=None, latency_seconds=1.0,
            created_at=NOW, trade_plan=plan, validation=ValidationResult(status="valid", issues=[]),
        )
        for label in labels
    ]


def _settings(**overrides) -> RiskSettings:
    base = dict(
        account_balance_usdt=Decimal("1000"), risk_percent=Decimal("1"), leverage=5,
        margin_mode=MarginMode.ISOLATED, max_margin_percent=Decimal("30"), maker_fee_percent=Decimal("0.02"),
        taker_fee_percent=Decimal("0.05"), slippage_percent=Decimal("0.02"), min_risk_reward=Decimal("1.5"),
        entry_price_mode=EntryPriceMode.MIDPOINT, quantity_step=Decimal("0.001"), price_step=Decimal("0.01"),
        minimum_order_notional_usdt=Decimal("2"),
    )
    base.update(overrides)
    return RiskSettings(**base)


def _patch_common(monkeypatch, *, results: list[AIAnalysisResult]):
    """Wires every external boundary to deterministic fixtures, leaving
    journal_db, calculate_active_position and open_virtual_order real.
    """
    monkeypatch.setattr(market_data_engine, "collect_snapshot", lambda symbol, now=None: _market_snapshot())
    monkeypatch.setattr(feature_engine, "compute_features", lambda snapshot: _empty_feature_set())
    monkeypatch.setattr(market_regime, "classify_regime", lambda features: _regime_result())
    monkeypatch.setattr(strategy_engine, "score_strategy", lambda features, regime, mode: _score_result())
    monkeypatch.setattr(ai_cycle, "fetch_snapshot", lambda symbol: {"symbol": symbol, "current_price": 100.5})
    monkeypatch.setattr(ai_cycle, "build_report", lambda snapshot, ctx: "report")
    monkeypatch.setattr(ai_cycle, "build_validation_context", lambda snapshot, mode: object())
    monkeypatch.setattr(ai_client, "analyze_with_all", lambda *a, **k: results)
    monkeypatch.setattr(risk_settings_store, "load", lambda: _settings())
    monkeypatch.setattr(position_service, "resolve_instrument_rules", lambda symbol, settings: RULES)
    monkeypatch.setattr(config, "AI_MODELS", [object(), object(), object()])


def test_actionable_long_consensus_opens_a_paper_order(conn, monkeypatch):
    _patch_common(monkeypatch, results=_votes(_plan()))

    result = ai_cycle.run_ai_cycle(conn, "ETHUSDT", "scalping", now=NOW)

    assert result.ran is True
    assert result.trade_plan_id is not None
    assert result.paper_order_status == "CREATED"

    plan_row = journal_db.get_trade_plan(conn, result.trade_plan_id)
    assert plan_row["overall_signal"] == "LONG"
    assert plan_row["position_status"] == "VALID"

    order = journal_db.get_paper_order_by_trade_plan_id(conn, result.trade_plan_id)
    assert order is not None
    assert order["status"] == "PENDING"


def test_wait_consensus_records_decision_without_opening_an_order(conn, monkeypatch):
    _patch_common(monkeypatch, results=_votes(_wait_plan()))

    result = ai_cycle.run_ai_cycle(conn, "ETHUSDT", "scalping", now=NOW)

    assert result.ran is True
    assert result.trade_plan_id is not None
    assert result.paper_order_status == "SKIPPED_NOT_ACTIONABLE"

    plan_row = journal_db.get_trade_plan(conn, result.trade_plan_id)
    assert plan_row["overall_signal"] == "WAIT"
    assert journal_db.get_paper_order_by_trade_plan_id(conn, result.trade_plan_id) is None


def test_calling_twice_for_the_same_cycle_never_duplicates_the_order(conn, monkeypatch):
    _patch_common(monkeypatch, results=_votes(_plan()))
    result = ai_cycle.run_ai_cycle(conn, "ETHUSDT", "scalping", now=NOW)

    # Simulate a second, independent cycle immediately after -- a fresh
    # trade_plan row is still created (it's a new decision each time), but
    # open_virtual_order's own idempotency is exercised on the FIRST plan
    # id if called again directly.
    from paper_trading import open_virtual_order

    repeat = open_virtual_order(conn, result.trade_plan_id, now=NOW)
    assert repeat.status == "ALREADY_EXISTS"
    assert repeat.order_id == journal_db.get_paper_order_by_trade_plan_id(conn, result.trade_plan_id)["id"]


def test_feature_pipeline_failure_skips_the_cycle_cleanly(conn, monkeypatch):
    monkeypatch.setattr(
        market_data_engine, "collect_snapshot",
        lambda symbol, now=None: (_ for _ in ()).throw(RuntimeError("bingx down")),
    )
    result = ai_cycle.run_ai_cycle(conn, "ETHUSDT", "scalping", now=NOW)
    assert result.ran is False
    assert "feature/regime/score pipeline failed" in result.reason


def test_fetch_snapshot_failure_skips_the_cycle_cleanly(conn, monkeypatch):
    monkeypatch.setattr(market_data_engine, "collect_snapshot", lambda symbol, now=None: _market_snapshot())
    monkeypatch.setattr(feature_engine, "compute_features", lambda snapshot: _empty_feature_set())
    monkeypatch.setattr(market_regime, "classify_regime", lambda features: _regime_result())
    monkeypatch.setattr(strategy_engine, "score_strategy", lambda features, regime, mode: _score_result())
    monkeypatch.setattr(risk_settings_store, "load", lambda: _settings())
    monkeypatch.setattr(
        ai_cycle, "fetch_snapshot", lambda symbol: (_ for _ in ()).throw(RuntimeError("network down"))
    )

    result = ai_cycle.run_ai_cycle(conn, "ETHUSDT", "scalping", now=NOW)
    assert result.ran is False
    assert "fetch_snapshot failed" in result.reason
    # nothing should have been persisted -- the failure was before any AI call
    assert conn.execute("SELECT COUNT(*) FROM trade_plans").fetchone()[0] == 0


def test_no_ai_key_configured_skips_the_cycle_cleanly(conn, monkeypatch):
    monkeypatch.setattr(market_data_engine, "collect_snapshot", lambda symbol, now=None: _market_snapshot())
    monkeypatch.setattr(feature_engine, "compute_features", lambda snapshot: _empty_feature_set())
    monkeypatch.setattr(market_regime, "classify_regime", lambda features: _regime_result())
    monkeypatch.setattr(strategy_engine, "score_strategy", lambda features, regime, mode: _score_result())
    monkeypatch.setattr(risk_settings_store, "load", lambda: _settings())
    monkeypatch.setattr(ai_cycle, "fetch_snapshot", lambda symbol: {"symbol": symbol, "current_price": 100.5})
    monkeypatch.setattr(ai_cycle, "build_report", lambda snapshot, ctx: "report")
    monkeypatch.setattr(ai_cycle, "build_validation_context", lambda snapshot, mode: object())
    monkeypatch.setattr(
        ai_client, "analyze_with_all",
        lambda *a, **k: (_ for _ in ()).throw(ai_client.AIConfigError("no key configured")),
    )

    result = ai_cycle.run_ai_cycle(conn, "ETHUSDT", "scalping", now=NOW)
    assert result.ran is False
    assert "no key configured" in result.reason
