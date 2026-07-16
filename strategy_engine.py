"""Strategy Score Engine — LLM-independent LONG/SHORT/NO_TRADE scoring
from a `feature_engine.FeatureSet` + `market_regime.RegimeResult` (BingX
-> Market Data Engine -> Feature Engine -> Market Regime Engine ->
**Strategy Score Engine** -> AI Analysis Engine -> ...).

Rules are DATA (a tuple of `Rule` objects), not logic scattered across
if/elif branches — that's what makes the ruleset versionable (swap/extend
the tuple, bump `config.STRATEGY_RULESET_VERSION`), testable (each rule's
`condition`/`effect` can be exercised directly against a hand-built
`RuleContext`), and configurable (every threshold lives in `config.py`).

`gross_rr` is optional: Strategy Score Engine runs BEFORE AI Analysis
Engine in the pipeline, so there is no trade plan (entry/stop/TP) yet to
compute a real R:R from. The R:R rule simply doesn't fire when it isn't
supplied — this module never fabricates a risk/reward number. Callers
that DO have a proposed plan's R:R (e.g. re-scoring an AI plan later) can
pass it in.

Not wired into the AI pipeline yet — same standalone-first approach as
Stages 2-4.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Literal

import config
from feature_engine import FeatureSet, MomentumFeatures, TrendFeatures, VolatilityFeatures, VolumeFeatures
from market_regime import RegimeResult

Mode = Literal["scalping", "swing"]
Decision = Literal["LONG_BIAS", "SHORT_BIAS", "NO_TRADE", "NEUTRAL"]
Quality = Literal["A", "B", "C", "D"]


@dataclass(frozen=True)
class RuleContext:
    features: FeatureSet
    regime: RegimeResult
    mode: Mode
    primary_tf: str
    gross_rr: float | None


@dataclass(frozen=True)
class RuleEffect:
    long: float = 0.0
    short: float = 0.0
    no_trade: float = 0.0


@dataclass(frozen=True)
class Rule:
    rule_id: str
    modes: tuple[Mode, ...] | None  # None = applies to both modes
    condition: Callable[[RuleContext], bool]
    effect: Callable[[RuleContext], RuleEffect]


@dataclass(frozen=True)
class Contribution:
    feature: str
    score: float


@dataclass(frozen=True)
class ScoreResult:
    timestamp: datetime
    symbol: str
    mode: Mode
    ruleset_version: str
    long_score: float
    short_score: float
    no_trade_score: float
    decision: Decision
    quality: Quality
    contributions: list[Contribution] = field(default_factory=list)


def _primary_trend(ctx: RuleContext) -> TrendFeatures | None:
    return ctx.features.trend.get(ctx.primary_tf)


def _primary_momentum(ctx: RuleContext) -> MomentumFeatures | None:
    return ctx.features.momentum.get(ctx.primary_tf)


def _primary_volatility(ctx: RuleContext) -> VolatilityFeatures | None:
    return ctx.features.volatility.get(ctx.primary_tf)


def _primary_volume(ctx: RuleContext) -> VolumeFeatures | None:
    return ctx.features.volume.get(ctx.primary_tf)


def _funding_weight(ctx: RuleContext) -> float:
    return config.STRATEGY_SWING_FUNDING_WEIGHT_MULTIPLIER if ctx.mode == "swing" else 1.0


# --- rule conditions/effects -------------------------------------------------


def _price_above_emas_cond(ctx: RuleContext) -> bool:
    t = _primary_trend(ctx)
    return t is not None and t.price_vs_ema20_pct is not None and t.price_vs_ema50_pct is not None


def _price_above_emas_effect(ctx: RuleContext) -> RuleEffect:
    t = _primary_trend(ctx)
    if t.price_vs_ema20_pct > 0 and t.price_vs_ema50_pct > 0:
        return RuleEffect(long=8)
    if t.price_vs_ema20_pct < 0 and t.price_vs_ema50_pct < 0:
        return RuleEffect(short=8)
    return RuleEffect()


def _ema20_above_ema50_cond(ctx: RuleContext) -> bool:
    t = _primary_trend(ctx)
    return t is not None and t.ema20_ema50_distance_pct is not None and t.ema20_ema50_distance_pct != 0


def _ema20_above_ema50_effect(ctx: RuleContext) -> RuleEffect:
    t = _primary_trend(ctx)
    return RuleEffect(long=6) if t.ema20_ema50_distance_pct > 0 else RuleEffect(short=6)


def _structure_direction_cond(ctx: RuleContext) -> bool:
    t = _primary_trend(ctx)
    return t is not None and t.structure_direction in ("BULLISH", "BEARISH")


def _structure_direction_effect(ctx: RuleContext) -> RuleEffect:
    t = _primary_trend(ctx)
    return RuleEffect(long=8) if t.structure_direction == "BULLISH" else RuleEffect(short=8)


def _timeframe_alignment_cond(ctx: RuleContext) -> bool:
    return (
        ctx.features.timeframe_alignment >= config.STRATEGY_TIMEFRAME_ALIGNMENT_THRESHOLD
        and ctx.regime.regime in ("TREND_UP", "TREND_DOWN", "BREAKOUT_UP", "BREAKOUT_DOWN")
    )


def _timeframe_alignment_effect(ctx: RuleContext) -> RuleEffect:
    if ctx.regime.regime in ("TREND_UP", "BREAKOUT_UP"):
        return RuleEffect(long=12)
    return RuleEffect(short=12)


def _rsi_extreme_cond(ctx: RuleContext) -> bool:
    m = _primary_momentum(ctx)
    return m is not None and m.rsi14 is not None and (
        m.rsi14 >= config.STRATEGY_RSI_OVERBOUGHT or m.rsi14 <= config.STRATEGY_RSI_OVERSOLD
    )


def _rsi_extreme_effect(ctx: RuleContext) -> RuleEffect:
    m = _primary_momentum(ctx)
    if m.rsi14 >= config.STRATEGY_RSI_OVERBOUGHT:
        return RuleEffect(long=-5, short=3)
    return RuleEffect(short=-5, long=3)


def _macd_hist_direction_cond(ctx: RuleContext) -> bool:
    m = _primary_momentum(ctx)
    return m is not None and m.macd_hist is not None and m.macd_hist_change is not None


def _macd_hist_direction_effect(ctx: RuleContext) -> RuleEffect:
    m = _primary_momentum(ctx)
    if m.macd_hist > 0 and m.macd_hist_change > 0:
        return RuleEffect(long=5)
    if m.macd_hist < 0 and m.macd_hist_change < 0:
        return RuleEffect(short=5)
    return RuleEffect()


def _divergence_cond(ctx: RuleContext) -> bool:
    m = _primary_momentum(ctx)
    return m is not None and (m.bullish_divergence or m.bearish_divergence)


def _divergence_effect(ctx: RuleContext) -> RuleEffect:
    m = _primary_momentum(ctx)
    if m.bullish_divergence:
        return RuleEffect(long=6, short=-3)
    return RuleEffect(short=6, long=-3)


def _low_volume_cond(ctx: RuleContext) -> bool:
    v = _primary_volume(ctx)
    return v is not None and v.volume_ratio is not None and v.volume_ratio < config.STRATEGY_LOW_VOLUME_RATIO


def _low_volume_effect(ctx: RuleContext) -> RuleEffect:
    return RuleEffect(no_trade=10)


def _regime_is(*regimes: str) -> Callable[[RuleContext], bool]:
    return lambda ctx: ctx.regime.regime in regimes


def _oi_price_trend_cond(ctx: RuleContext) -> bool:
    return ctx.features.futures.oi_price_regime in ("NEW_LONGS", "NEW_SHORTS")


def _oi_price_trend_effect(ctx: RuleContext) -> RuleEffect:
    if ctx.features.futures.oi_price_regime == "NEW_LONGS":
        return RuleEffect(long=10)
    return RuleEffect(short=10)


def _funding_extreme_cond(ctx: RuleContext) -> bool:
    rate = ctx.features.futures.funding_rate
    return rate is not None and abs(rate) >= config.STRATEGY_FUNDING_EXTREME_THRESHOLD


def _funding_extreme_effect(ctx: RuleContext) -> RuleEffect:
    weight = _funding_weight(ctx)
    rate = ctx.features.futures.funding_rate
    if rate >= config.STRATEGY_FUNDING_EXTREME_THRESHOLD:
        return RuleEffect(long=-8 * weight, short=4 * weight)
    return RuleEffect(short=-8 * weight, long=4 * weight)


def _orderbook_imbalance_cond(ctx: RuleContext) -> bool:
    imb = ctx.features.orderbook.imbalance
    return imb is not None and abs(imb) >= config.STRATEGY_ORDERBOOK_IMBALANCE_THRESHOLD


def _orderbook_imbalance_effect(ctx: RuleContext) -> RuleEffect:
    imb = ctx.features.orderbook.imbalance
    return RuleEffect(long=4) if imb >= config.STRATEGY_ORDERBOOK_IMBALANCE_THRESHOLD else RuleEffect(short=4)


def _near_resistance_cond(ctx: RuleContext) -> bool:
    d = ctx.features.distance_to_resistance_atr
    return d is not None and d < config.STRATEGY_NEAR_LEVEL_ATR_THRESHOLD


def _near_resistance_effect(ctx: RuleContext) -> RuleEffect:
    return RuleEffect(long=-12)


def _near_support_cond(ctx: RuleContext) -> bool:
    d = ctx.features.distance_to_support_atr
    return d is not None and d < config.STRATEGY_NEAR_LEVEL_ATR_THRESHOLD


def _near_support_effect(ctx: RuleContext) -> RuleEffect:
    return RuleEffect(short=-12)


def _poor_risk_reward_cond(ctx: RuleContext) -> bool:
    return ctx.gross_rr is not None and ctx.gross_rr < config.STRATEGY_MIN_RISK_REWARD


def _poor_risk_reward_effect(ctx: RuleContext) -> RuleEffect:
    return RuleEffect(no_trade=30)


def _regime_effect(**kwargs: float) -> Callable[[RuleContext], RuleEffect]:
    return lambda ctx: RuleEffect(**kwargs)


def _regime_breakout_effect(ctx: RuleContext) -> RuleEffect:
    return RuleEffect(long=10) if ctx.regime.regime == "BREAKOUT_UP" else RuleEffect(short=10)


RULES: tuple[Rule, ...] = (
    # Trend / EMA (primary_tf)
    Rule("price_above_emas", None, _price_above_emas_cond, _price_above_emas_effect),
    Rule("ema20_above_ema50", None, _ema20_above_ema50_cond, _ema20_above_ema50_effect),
    Rule("structure_direction", None, _structure_direction_cond, _structure_direction_effect),
    Rule("timeframe_alignment", None, _timeframe_alignment_cond, _timeframe_alignment_effect),
    # Momentum (primary_tf)
    Rule("rsi_extreme", None, _rsi_extreme_cond, _rsi_extreme_effect),
    Rule("macd_hist_direction", None, _macd_hist_direction_cond, _macd_hist_direction_effect),
    Rule("divergence", None, _divergence_cond, _divergence_effect),
    # Volume (primary_tf)
    Rule("low_volume", None, _low_volume_cond, _low_volume_effect),
    # Market regime (from RegimeResult.regime, not raw flags — avoids double-counting)
    Rule("regime_reversal_risk", None, _regime_is("REVERSAL_RISK"), _regime_effect(no_trade=15, long=-5, short=-5)),
    Rule("regime_range", None, _regime_is("RANGE"), _regime_effect(no_trade=8)),
    Rule(
        "regime_volatility_compression",
        None,
        _regime_is("VOLATILITY_COMPRESSION"),
        _regime_effect(no_trade=15),
    ),
    Rule("regime_volatility_expansion", None, _regime_is("VOLATILITY_EXPANSION"), _regime_effect(no_trade=10)),
    Rule("regime_breakout", None, _regime_is("BREAKOUT_UP", "BREAKOUT_DOWN"), _regime_breakout_effect),
    Rule("regime_unstable", None, _regime_is("UNSTABLE"), _regime_effect(no_trade=10)),
    # Futures (symbol-level; funding weighted x2 for swing)
    Rule("oi_price_trend", None, _oi_price_trend_cond, _oi_price_trend_effect),
    Rule("funding_extreme", None, _funding_extreme_cond, _funding_extreme_effect),
    # Order book — scalping only, microstructure doesn't matter for an 8h swing hold
    Rule("orderbook_imbalance", ("scalping",), _orderbook_imbalance_cond, _orderbook_imbalance_effect),
    # Levels
    Rule("near_resistance", None, _near_resistance_cond, _near_resistance_effect),
    Rule("near_support", None, _near_support_cond, _near_support_effect),
    # Risk/reward — only fires if a caller supplied gross_rr
    Rule("poor_risk_reward", None, _poor_risk_reward_cond, _poor_risk_reward_effect),
)


def _quality_for(score: float) -> Quality:
    if score >= 80:
        return "A"
    if score >= 65:
        return "B"
    if score >= 50:
        return "C"
    return "D"


def _no_data_result(features: FeatureSet, mode: Mode) -> ScoreResult:
    return ScoreResult(
        timestamp=features.timestamp,
        symbol=features.symbol,
        mode=mode,
        ruleset_version=config.STRATEGY_RULESET_VERSION,
        long_score=0.0,
        short_score=0.0,
        no_trade_score=100.0,
        decision="NO_TRADE",
        quality="D",
        contributions=[],
    )


def score_strategy(
    features: FeatureSet,
    regime: RegimeResult,
    mode: Mode,
    gross_rr: float | None = None,
) -> ScoreResult:
    if features.data_quality == "NO_TRADE" or regime.regime == "NO_DATA":
        return _no_data_result(features, mode)

    primary_tf = config.STRATEGY_SCALPING_PRIMARY_TIMEFRAME if mode == "scalping" else config.STRATEGY_SWING_PRIMARY_TIMEFRAME
    ctx = RuleContext(features=features, regime=regime, mode=mode, primary_tf=primary_tf, gross_rr=gross_rr)

    long_score = config.STRATEGY_BASELINE_SCORE
    short_score = config.STRATEGY_BASELINE_SCORE
    no_trade_score = config.STRATEGY_BASELINE_SCORE
    contributions: list[Contribution] = []

    for rule in RULES:
        if rule.modes is not None and mode not in rule.modes:
            continue
        if not rule.condition(ctx):
            continue
        effect = rule.effect(ctx)
        active_axes = [(axis, delta) for axis, delta in (("long", effect.long), ("short", effect.short), ("no_trade", effect.no_trade)) if delta]
        for axis, delta in active_axes:
            feature_name = rule.rule_id if len(active_axes) == 1 else f"{rule.rule_id}:{axis}"
            contributions.append(Contribution(feature=feature_name, score=delta))
            if axis == "long":
                long_score += delta
            elif axis == "short":
                short_score += delta
            else:
                no_trade_score += delta

    long_score = max(0.0, min(100.0, long_score))
    short_score = max(0.0, min(100.0, short_score))
    no_trade_score = max(0.0, min(100.0, no_trade_score))

    if no_trade_score >= max(long_score, short_score) and no_trade_score >= config.STRATEGY_NO_TRADE_DOMINANT_SCORE:
        decision: Decision = "NO_TRADE"
        quality = _quality_for(no_trade_score)
    elif long_score - short_score >= config.STRATEGY_BIAS_MARGIN and long_score >= config.STRATEGY_BIAS_MIN_SCORE:
        decision = "LONG_BIAS"
        quality = _quality_for(long_score)
    elif short_score - long_score >= config.STRATEGY_BIAS_MARGIN and short_score >= config.STRATEGY_BIAS_MIN_SCORE:
        decision = "SHORT_BIAS"
        quality = _quality_for(short_score)
    else:
        decision = "NEUTRAL"
        quality = _quality_for(max(long_score, short_score, no_trade_score))

    return ScoreResult(
        timestamp=features.timestamp,
        symbol=features.symbol,
        mode=mode,
        ruleset_version=config.STRATEGY_RULESET_VERSION,
        long_score=long_score,
        short_score=short_score,
        no_trade_score=no_trade_score,
        decision=decision,
        quality=quality,
        contributions=contributions,
    )
