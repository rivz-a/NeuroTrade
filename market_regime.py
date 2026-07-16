"""Market Regime Engine — classifies market state programmatically from a
`feature_engine.FeatureSet` into one of 10 regimes (BingX -> Market Data
Engine -> Feature Engine -> **Market Regime Engine** -> Strategy/Score
Engine -> ...). Pure decision tree over already-computed features — no
new indicator math here, that's Feature Engine's job.

Not wired into the AI pipeline yet — same standalone-first approach as
Stages 2-3.

Naming note: `ai_schema.MarketRegime` (the AI's own self-reported regime
field on every TradePlan) already uses 7 of these 10 names verbatim
(TREND_UP, TREND_DOWN, RANGE, VOLATILITY_EXPANSION, VOLATILITY_COMPRESSION,
REVERSAL_RISK, UNSTABLE — only its "UNKNOWN" vs this module's "NO_DATA",
plus BREAKOUT_UP/BREAKOUT_DOWN being new here, differ). That's very
unlikely to be a coincidence, but reconciling the two (e.g. having the AI
prompt receive this module's regime instead of guessing its own) is a
separate, later decision — this module intentionally does not import from
or modify `ai_schema.py`, which is a live AI-facing contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal

import config
from feature_engine import FeatureSet, FuturesFeatures, MomentumFeatures, TrendFeatures, VolatilityFeatures, VolumeFeatures

Regime = Literal[
    "TREND_UP",
    "TREND_DOWN",
    "RANGE",
    "BREAKOUT_UP",
    "BREAKOUT_DOWN",
    "VOLATILITY_COMPRESSION",
    "VOLATILITY_EXPANSION",
    "REVERSAL_RISK",
    "UNSTABLE",
    "NO_DATA",
]

# First match wins when reducing several timeframes' regimes to one overall
# call — a rare-but-important state (a confirmed breakout, a reversal
# warning) on even a single timeframe shouldn't be diluted by a plain
# majority vote against several TREND/RANGE timeframes.
_AGGREGATE_PRIORITY: tuple[Regime, ...] = (
    "BREAKOUT_UP",
    "BREAKOUT_DOWN",
    "REVERSAL_RISK",
    "VOLATILITY_EXPANSION",
    "VOLATILITY_COMPRESSION",
    "TREND_UP",
    "TREND_DOWN",
    "RANGE",
    "UNSTABLE",
)

_STRATEGY_HINTS: dict[Regime, str] = {
    "TREND_UP": "Искать вход в LONG на откате (pullback), не покупать на пиках импульса.",
    "TREND_DOWN": "Искать вход в SHORT на откате (pullback), не продавать на дне импульса.",
    "RANGE": "Вход только от границ диапазона. Запрет входа в середине диапазона.",
    "BREAKOUT_UP": "Подтверждённый пробой вверх — вход по направлению пробоя, стоп за уровень пробоя.",
    "BREAKOUT_DOWN": "Подтверждённый пробой вниз — вход по направлению пробоя, стоп за уровень пробоя.",
    "VOLATILITY_COMPRESSION": "Диапазон сжимается — ждать подтверждённый пробой, не входить внутри сжатия.",
    "VOLATILITY_EXPANSION": "Волатильность растёт без подтверждённого направления — снижать размер позиции.",
    "REVERSAL_RISK": "Признаки истощения тренда/разворота — уменьшать риск, не наращивать позицию по тренду.",
    "UNSTABLE": "Противоречивые сигналы — воздержаться от входа до прояснения структуры.",
    "NO_DATA": "Недостаточно данных для определения режима — сделки не рассматривать.",
}


@dataclass(frozen=True)
class RegimeResult:
    timestamp: datetime
    symbol: str
    regime: Regime
    regime_by_timeframe: dict[str, Regime]
    reasons: list[str]
    strategy_hint: str


def _classify_timeframe(
    trend: TrendFeatures,
    momentum: MomentumFeatures,
    volatility: VolatilityFeatures,
    volume: VolumeFeatures,
    futures: FuturesFeatures,
) -> tuple[Regime, str]:
    if trend.adx is None or volatility.atr_percentile is None:
        return "NO_DATA", "insufficient history for ADX/ATR percentile"

    adx = trend.adx
    is_trending = adx >= config.REGIME_ADX_TREND_THRESHOLD
    is_weak = adx <= config.REGIME_ADX_RANGE_THRESHOLD

    if trend.break_of_structure and (volume.volume_spike or volatility.volatility_expansion):
        if trend.structure_direction == "BULLISH":
            return "BREAKOUT_UP", "break_of_structure + volume_spike/volatility_expansion (bullish)"
        if trend.structure_direction == "BEARISH":
            return "BREAKOUT_DOWN", "break_of_structure + volume_spike/volatility_expansion (bearish)"

    choch_reversal = trend.change_of_character and is_trending
    divergence_reversal = (trend.trend_state == "BULLISH" and momentum.bearish_divergence) or (
        trend.trend_state == "BEARISH" and momentum.bullish_divergence
    )
    funding_extreme = futures.funding_rate is not None and abs(futures.funding_rate) >= config.REGIME_FUNDING_EXTREME_THRESHOLD
    unwind_regime = futures.oi_price_regime in ("SHORT_COVERING", "LONG_COVERING")

    if choch_reversal:
        return "REVERSAL_RISK", "change_of_character against an established trend"
    if divergence_reversal:
        return "REVERSAL_RISK", "momentum divergence against the prevailing trend"
    if funding_extreme and unwind_regime:
        return "REVERSAL_RISK", f"extreme funding ({futures.funding_rate}) with {futures.oi_price_regime}"

    if volatility.range_compression and is_weak:
        return "VOLATILITY_COMPRESSION", "range_compression + weak ADX"

    if volatility.volatility_expansion and not is_trending:
        return "VOLATILITY_EXPANSION", "volatility_expansion without a confirmed trend"

    if is_trending and trend.trend_state == "BULLISH" and trend.supertrend_direction != "DOWN":
        return "TREND_UP", "ADX trending + bullish EMA alignment + SuperTrend agrees"
    if is_trending and trend.trend_state == "BEARISH" and trend.supertrend_direction != "UP":
        return "TREND_DOWN", "ADX trending + bearish EMA alignment + SuperTrend agrees"

    if is_weak and trend.structure_direction == "NEUTRAL":
        return "RANGE", "weak ADX + neutral structure"

    return "UNSTABLE", "no condition matched cleanly (mixed/contradictory signals)"


def _aggregate(per_tf: dict[str, Regime]) -> Regime:
    non_no_data = [r for r in per_tf.values() if r != "NO_DATA"]
    if not non_no_data:
        return "NO_DATA"
    for candidate in _AGGREGATE_PRIORITY:
        if candidate in non_no_data:
            return candidate
    return "UNSTABLE"


def classify_regime(features: FeatureSet) -> RegimeResult:
    if features.data_quality == "NO_TRADE":
        return RegimeResult(
            timestamp=features.timestamp,
            symbol=features.symbol,
            regime="NO_DATA",
            regime_by_timeframe={},
            reasons=["market data quality is NO_TRADE — regime not evaluated"],
            strategy_hint=_STRATEGY_HINTS["NO_DATA"],
        )

    regime_by_timeframe: dict[str, Regime] = {}
    reasons: list[str] = []
    for interval, trend in features.trend.items():
        momentum = features.momentum.get(interval)
        volatility = features.volatility.get(interval)
        volume = features.volume.get(interval)
        if momentum is None or volatility is None or volume is None:
            regime_by_timeframe[interval] = "NO_DATA"
            reasons.append(f"{interval}: NO_DATA (missing feature group)")
            continue
        regime, reason = _classify_timeframe(trend, momentum, volatility, volume, features.futures)
        regime_by_timeframe[interval] = regime
        reasons.append(f"{interval}: {regime} ({reason})")

    overall = _aggregate(regime_by_timeframe)

    return RegimeResult(
        timestamp=features.timestamp,
        symbol=features.symbol,
        regime=overall,
        regime_by_timeframe=regime_by_timeframe,
        reasons=reasons,
        strategy_hint=_STRATEGY_HINTS[overall],
    )
