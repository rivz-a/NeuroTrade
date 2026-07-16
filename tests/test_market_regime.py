"""Offline tests for market_regime.classify_regime. TrendFeatures/
MomentumFeatures/VolatilityFeatures/VolumeFeatures/FuturesFeatures are
built directly by hand (plain dataclasses from feature_engine.py) — no
synthetic candles or network needed, unlike test_feature_engine.py.
"""

from datetime import datetime, timezone

import config
import market_regime
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

NOW = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

NEUTRAL_FUTURES = FuturesFeatures(
    funding_rate=0.0001,
    funding_change=0.0,
    open_interest=1_000_000.0,
    open_interest_change=0.0,
    open_interest_change_pct=0.0,
    oi_price_regime=None,
)


def _trend(**overrides) -> TrendFeatures:
    base = dict(
        price_vs_ema20_pct=0.5,
        price_vs_ema50_pct=1.0,
        price_vs_ema200_pct=2.0,
        ema20_slope_pct=0.2,
        ema20_ema50_distance_pct=0.5,
        ema50_ema200_distance_pct=1.0,
        adx=30.0,
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
    base.update(overrides)
    return TrendFeatures(**base)


def _momentum(**overrides) -> MomentumFeatures:
    base = dict(
        rsi14=55.0,
        rsi_roc=1.0,
        macd_hist=0.5,
        macd_hist_change=0.1,
        roc=1.0,
        momentum=1.0,
        bullish_divergence=False,
        bearish_divergence=False,
    )
    base.update(overrides)
    return MomentumFeatures(**base)


def _volatility(**overrides) -> VolatilityFeatures:
    base = dict(
        atr14=10.0,
        atr_percentile=50.0,
        bollinger_width_pct=2.0,
        realized_volatility_pct=0.5,
        range_compression=False,
        volatility_expansion=False,
        last_candle_range_atr_ratio=1.0,
    )
    base.update(overrides)
    return VolatilityFeatures(**base)


def _volume(**overrides) -> VolumeFeatures:
    base = dict(volume_ratio=1.0, volume_spike=False, volume_trend="FLAT")
    base.update(overrides)
    return VolumeFeatures(**base)


def _feature_set(
    trend: dict[str, TrendFeatures],
    momentum: dict[str, MomentumFeatures] | None = None,
    volatility: dict[str, VolatilityFeatures] | None = None,
    volume: dict[str, VolumeFeatures] | None = None,
    futures: FuturesFeatures = NEUTRAL_FUTURES,
    data_quality="GOOD",
) -> FeatureSet:
    tfs = list(trend.keys())
    momentum = momentum or {tf: _momentum() for tf in tfs}
    volatility = volatility or {tf: _volatility() for tf in tfs}
    volume = volume or {tf: _volume() for tf in tfs}
    return FeatureSet(
        timestamp=NOW,
        symbol="ETHUSDT",
        data_quality=data_quality,
        trend=trend,
        momentum=momentum,
        volatility=volatility,
        volume=volume,
        trade_flow=TradeFlowFeatures(0.0, 0.0, 0.0, 0.0),
        futures=futures,
        orderbook=OrderbookFeatures(None, None, None, None, None, None),
        timeframe_alignment=1.0,
        distance_to_support_atr=None,
        distance_to_resistance_atr=None,
    )


def _single_tf_result(trend, momentum=None, volatility=None, volume=None, futures=NEUTRAL_FUTURES):
    fs = _feature_set(
        trend={"1m": trend},
        momentum={"1m": momentum} if momentum else None,
        volatility={"1m": volatility} if volatility else None,
        volume={"1m": volume} if volume else None,
        futures=futures,
    )
    return market_regime.classify_regime(fs)


# --- one test per regime ---


def test_trend_up():
    r = _single_tf_result(_trend(adx=30.0, trend_state="BULLISH", supertrend_direction="UP"))
    assert r.regime == "TREND_UP"


def test_trend_down():
    r = _single_tf_result(_trend(adx=30.0, trend_state="BEARISH", supertrend_direction="DOWN", structure_direction="BEARISH"))
    assert r.regime == "TREND_DOWN"


def test_range():
    r = _single_tf_result(_trend(adx=10.0, trend_state="NEUTRAL", structure_direction="NEUTRAL"))
    assert r.regime == "RANGE"


def test_breakout_up():
    r = _single_tf_result(
        _trend(break_of_structure=True, structure_direction="BULLISH"),
        volume=_volume(volume_spike=True),
    )
    assert r.regime == "BREAKOUT_UP"


def test_breakout_down():
    r = _single_tf_result(
        _trend(break_of_structure=True, structure_direction="BEARISH"),
        volume=_volume(volume_spike=True),
    )
    assert r.regime == "BREAKOUT_DOWN"


def test_volatility_compression():
    r = _single_tf_result(_trend(adx=10.0), volatility=_volatility(range_compression=True))
    assert r.regime == "VOLATILITY_COMPRESSION"


def test_volatility_expansion():
    r = _single_tf_result(_trend(adx=10.0, trend_state="NEUTRAL"), volatility=_volatility(volatility_expansion=True))
    assert r.regime == "VOLATILITY_EXPANSION"


def test_reversal_risk_from_change_of_character():
    r = _single_tf_result(_trend(adx=30.0, change_of_character=True))
    assert r.regime == "REVERSAL_RISK"


def test_reversal_risk_from_divergence():
    r = _single_tf_result(
        _trend(adx=30.0, trend_state="BULLISH"),
        momentum=_momentum(bearish_divergence=True),
    )
    assert r.regime == "REVERSAL_RISK"


def test_reversal_risk_from_funding_and_oi_unwind():
    futures = FuturesFeatures(
        funding_rate=0.001,  # well above REGIME_FUNDING_EXTREME_THRESHOLD default 0.0005
        funding_change=0.0005,
        open_interest=1_000_000.0,
        open_interest_change=-50_000.0,
        open_interest_change_pct=-5.0,
        oi_price_regime="SHORT_COVERING",
    )
    # deliberately NOT trending / no CHoCH / no divergence, so this test
    # isolates the funding+OI path specifically.
    r = _single_tf_result(_trend(adx=15.0, trend_state="NEUTRAL"), futures=futures)
    assert r.regime == "REVERSAL_RISK"
    assert any("funding" in reason for reason in r.reasons)


def test_unstable_when_nothing_matches_cleanly():
    # mid ADX (neither trending nor weak), structure contradicts trend_state
    r = _single_tf_result(_trend(adx=21.0, trend_state="BULLISH", structure_direction="BEARISH", supertrend_direction="DOWN"))
    assert r.regime == "UNSTABLE"


def test_no_data_from_missing_adx():
    r = _single_tf_result(_trend(adx=None))
    assert r.regime == "NO_DATA"
    assert r.regime_by_timeframe["1m"] == "NO_DATA"


# --- aggregation across timeframes ---


def test_aggregate_prefers_higher_priority_over_majority():
    fs = _feature_set(
        trend={
            "1m": _trend(adx=30.0, trend_state="BULLISH"),
            "5m": _trend(adx=30.0, trend_state="BULLISH"),
            "15m": _trend(break_of_structure=True, structure_direction="BULLISH"),
        },
        volume={
            "1m": _volume(),
            "5m": _volume(),
            "15m": _volume(volume_spike=True),
        },
    )
    r = market_regime.classify_regime(fs)
    # two TREND_UP timeframes vs one BREAKOUT_UP -> BREAKOUT_UP still wins (higher priority)
    assert r.regime == "BREAKOUT_UP"
    assert r.regime_by_timeframe["15m"] == "BREAKOUT_UP"


def test_aggregate_no_data_only_when_all_timeframes_lack_data():
    fs = _feature_set(
        trend={
            "1m": _trend(adx=None),
            "5m": _trend(adx=30.0, trend_state="BULLISH"),
        }
    )
    r = market_regime.classify_regime(fs)
    assert r.regime != "NO_DATA"
    assert r.regime_by_timeframe["1m"] == "NO_DATA"
    assert r.regime_by_timeframe["5m"] == "TREND_UP"


def test_aggregate_no_data_when_every_timeframe_lacks_data():
    fs = _feature_set(trend={"1m": _trend(adx=None), "5m": _trend(adx=None)})
    r = market_regime.classify_regime(fs)
    assert r.regime == "NO_DATA"


# --- NO_TRADE input propagation ---


def test_no_trade_snapshot_short_circuits():
    fs = _feature_set(trend={}, momentum={}, volatility={}, volume={}, data_quality="NO_TRADE")
    r = market_regime.classify_regime(fs)
    assert r.regime == "NO_DATA"
    assert r.regime_by_timeframe == {}


# --- strategy hint completeness ---


def test_every_regime_has_a_strategy_hint():
    for regime in market_regime.Regime.__args__:
        assert regime in market_regime._STRATEGY_HINTS
        assert market_regime._STRATEGY_HINTS[regime].strip()
