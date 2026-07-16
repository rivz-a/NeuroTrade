"""Offline tests for report_builder.py's Stage 6 additions (AIContext,
build_ai_context, _ai_context_section, and build_report's new optional
parameter) — no network, no AI calls. market_data_engine is mocked for
build_ai_context; RegimeResult/ScoreResult/FeatureSet are built directly
by hand for _ai_context_section, same style as the Stage 3-5 test files.
"""

from datetime import datetime, timezone
from unittest.mock import patch

import config
import report_builder
from feature_engine import FeatureSet, FuturesFeatures, OrderbookFeatures, TradeFlowFeatures
from market_regime import RegimeResult
from report_builder import AIContext, build_ai_context, build_report
from risk_manager import DEFAULT_RISK_SETTINGS
from strategy_engine import Contribution, ScoreResult

NOW = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)


def _position():
    return config.PositionConfig(position="NONE", entry_price=None, leverage=None, stop_loss=None, take_profit=None)


def _snapshot(mode_key="scalping"):
    empty_tf = {"price": 100.0, "indicators": {}, "candles": []}
    return {
        "symbol": "ETHUSDT",
        "exchange": "BingX",
        "timestamp": NOW,
        "current_price": 100.0,
        "timeframes": {"1m": empty_tf, "5m": empty_tf, "15m": empty_tf, "1h": empty_tf},
        "change_15m": 0.1,
        "change_1h": 0.2,
        "change_24h": 0.3,
        "funding_rate": 0.0001,
        "funding_history": [],
        "open_interest": None,
        "oi_trend": None,
        "orderbook": None,
        "levels": {"support": [95.0], "resistance": [105.0]},
        "position": _position(),
        "mode_key": mode_key,
    }


def _feature_set() -> FeatureSet:
    return FeatureSet(
        timestamp=NOW,
        symbol="ETHUSDT",
        data_quality="GOOD",
        trend={},
        momentum={},
        volatility={},
        volume={},
        trade_flow=TradeFlowFeatures(1.0, 0.5, 0.5, 0.5),
        futures=FuturesFeatures(0.0001, 0.0, 1_000_000.0, 0.0, 0.0, "NEW_LONGS"),
        orderbook=OrderbookFeatures(0.1, None, 0.2, 0.0, 100.1, False),
        timeframe_alignment=0.75,
        distance_to_support_atr=0.5,
        distance_to_resistance_atr=1.2,
    )


def _regime_result() -> RegimeResult:
    return RegimeResult(
        timestamp=NOW,
        symbol="ETHUSDT",
        regime="TREND_UP",
        regime_by_timeframe={"5m": "TREND_UP", "1h": "RANGE"},
        reasons=["5m: TREND_UP (...)"],
        strategy_hint="Искать вход в LONG на откате.",
    )


def _score_result(mode="scalping") -> ScoreResult:
    return ScoreResult(
        timestamp=NOW,
        symbol="ETHUSDT",
        mode=mode,
        ruleset_version="v1",
        long_score=68.0,
        short_score=31.0,
        no_trade_score=44.0,
        decision="LONG_BIAS",
        quality="B",
        contributions=[Contribution("timeframe_alignment", 12.0), Contribution("low_volume", -7.0)],
    )


def _ai_context(mode="scalping") -> AIContext:
    return AIContext(
        regime=_regime_result(), score=_score_result(mode), features=_feature_set(), risk_settings=DEFAULT_RISK_SETTINGS
    )


def test_build_report_without_ai_context_has_no_new_section():
    text = build_report(_snapshot())
    assert "Программный анализ" not in text
    assert "LONG_SCORE" not in text
    # existing sections are unaffected
    assert "Пара: ETHUSDT" in text
    assert "Задача для ChatGPT:" in text


def test_build_report_with_ai_context_has_new_section():
    text = build_report(_snapshot(), _ai_context())
    assert "Программный анализ" in text
    assert "Режим рынка (определён кодом): TREND_UP" in text
    assert "LONG_SCORE: 68" in text
    assert "SHORT_SCORE: 31" in text
    assert "NO_TRADE_SCORE: 44" in text
    assert "LONG_BIAS" in text and "качество B" in text
    assert "Риск-ограничения" in text
    assert "не рассчитывай размер позиции" in text
    # the new section still comes before the task instruction
    assert text.index("Программный анализ") < text.index("Задача для ChatGPT:")


def test_ai_context_section_includes_contributions():
    lines = report_builder._ai_context_section(_ai_context())
    text = "\n".join(lines)
    assert "timeframe_alignment +12" in text
    assert "low_volume -7" in text


def test_ai_context_section_reads_correct_primary_timeframe_per_mode():
    scalp_lines = "\n".join(report_builder._ai_context_section(_ai_context("scalping")))
    swing_lines = "\n".join(report_builder._ai_context_section(_ai_context("swing")))
    assert config.STRATEGY_SCALPING_PRIMARY_TIMEFRAME in scalp_lines
    assert config.STRATEGY_SWING_PRIMARY_TIMEFRAME in swing_lines


def test_ai_context_section_includes_risk_settings():
    lines = "\n".join(report_builder._ai_context_section(_ai_context()))
    assert f"{DEFAULT_RISK_SETTINGS.risk_percent}%" in lines
    assert f"{DEFAULT_RISK_SETTINGS.min_risk_reward}" in lines


def test_build_ai_context_returns_none_on_failure():
    with patch("market_data_engine.collect_snapshot", side_effect=RuntimeError("boom")):
        assert build_ai_context("ETH-USDT", "scalping") is None


def test_build_ai_context_composes_all_four_engines():
    fake_snapshot = object()
    with (
        patch("market_data_engine.collect_snapshot", return_value=fake_snapshot) as mock_collect,
        patch("feature_engine.compute_features", return_value=_feature_set()) as mock_features,
        patch("market_regime.classify_regime", return_value=_regime_result()) as mock_regime,
        patch("strategy_engine.score_strategy", return_value=_score_result("swing")) as mock_score,
    ):
        ctx = build_ai_context("ETH-USDT", "swing")

    assert ctx is not None
    mock_collect.assert_called_once_with("ETH-USDT")
    mock_features.assert_called_once_with(fake_snapshot)
    mock_regime.assert_called_once()
    mock_score.assert_called_once()
    assert ctx.score.mode == "swing"
    assert ctx.regime.regime == "TREND_UP"
    assert ctx.risk_settings == report_builder.risk_settings_store.load()
