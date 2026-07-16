"""Feature Engine — computes trend/momentum/volatility/volume/futures/
order-book features programmatically from a
`market_data_engine.MarketDataSnapshot`, so an AI model is handed
already-interpreted numbers instead of being asked to eyeball raw candles
itself (BingX -> Market Data Engine -> **Feature Engine** -> ...).

Not wired into the AI pipeline in this iteration — same standalone-first
approach as `market_data_engine.py` in Stage 2: a fully tested module on
its own, integration is a separate later decision.

Two honest limitations, documented where they bite (not silently faked):
- `cumulative_volume_delta` only accumulates over the trades already
  present in `snapshot.recent_trades` (the last `MARKET_DATA_TRADES_LIMIT`
  trades) — a real session-long CVD would need its own persisted
  accumulator across many snapshot collections, which doesn't exist yet.
- `orderbook.imbalance_avg_30_120s` and `has_large_wall` are bounded by
  what `market_data_history.jsonl` actually stores (top-of-book only, at
  whatever cadence the app happens to be run at — there's no fixed-tick
  background poller anywhere in this codebase). Both return `None`/best-
  effort rather than a number extrapolated from too little data.
"""

from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

import numpy as np
import pandas as pd
from ta.momentum import ROCIndicator, RSIIndicator
from ta.trend import ADXIndicator, EMAIndicator, MACD
from ta.volatility import AverageTrueRange

import config
import indicators
import levels
import market_data_history
from market_data_engine import MarketDataSnapshot, compute_orderbook_imbalance

DataQuality = Literal["GOOD", "DEGRADED", "NO_TRADE"]
TrendState = Literal["BULLISH", "BEARISH", "NEUTRAL"]


@dataclass(frozen=True)
class TrendFeatures:
    price_vs_ema20_pct: float | None
    price_vs_ema50_pct: float | None
    price_vs_ema200_pct: float | None
    ema20_slope_pct: float | None
    ema20_ema50_distance_pct: float | None
    ema50_ema200_distance_pct: float | None
    adx: float | None
    supertrend_direction: Literal["UP", "DOWN"] | None
    structure_direction: TrendState
    higher_high: bool | None
    higher_low: bool | None
    lower_high: bool | None
    lower_low: bool | None
    break_of_structure: bool
    change_of_character: bool
    trend_state: TrendState


@dataclass(frozen=True)
class MomentumFeatures:
    rsi14: float | None
    rsi_roc: float | None
    macd_hist: float | None
    macd_hist_change: float | None
    roc: float | None
    momentum: float | None
    bullish_divergence: bool
    bearish_divergence: bool


@dataclass(frozen=True)
class VolatilityFeatures:
    atr14: float | None
    atr_percentile: float | None
    bollinger_width_pct: float | None
    realized_volatility_pct: float | None
    range_compression: bool
    volatility_expansion: bool
    last_candle_range_atr_ratio: float | None


@dataclass(frozen=True)
class VolumeFeatures:
    """Per-timeframe — derived purely from that timeframe's own candles.
    Buy/sell/delta/CVD are NOT here: they come from `recent_trades`, which
    is one flat list on the snapshot, not per-timeframe — see
    `TradeFlowFeatures`.
    """

    volume_ratio: float | None
    volume_spike: bool
    volume_trend: Literal["INCREASING", "DECREASING", "FLAT"] | None


@dataclass(frozen=True)
class TradeFlowFeatures:
    """Symbol-level, from `snapshot.recent_trades` — see the module
    docstring for why `cumulative_volume_delta` is bounded to that window
    (numerically it always equals `delta` within one window; the field is
    kept distinct because a future session-persisted CVD would diverge
    from a single-window delta).
    """

    buy_volume: float
    sell_volume: float
    delta: float
    cumulative_volume_delta: float


@dataclass(frozen=True)
class FuturesFeatures:
    funding_rate: float | None
    funding_change: float | None
    open_interest: float | None
    open_interest_change: float | None
    open_interest_change_pct: float | None
    oi_price_regime: Literal["NEW_LONGS", "SHORT_COVERING", "NEW_SHORTS", "LONG_COVERING"] | None


@dataclass(frozen=True)
class OrderbookFeatures:
    imbalance: float | None
    imbalance_avg_30_120s: float | None
    spread: float | None
    spread_change: float | None
    microprice: float | None
    has_large_wall: bool | None


@dataclass(frozen=True)
class FeatureSet:
    timestamp: datetime
    symbol: str
    data_quality: DataQuality
    trend: dict[str, TrendFeatures]
    momentum: dict[str, MomentumFeatures]
    volatility: dict[str, VolatilityFeatures]
    volume: dict[str, VolumeFeatures]
    trade_flow: TradeFlowFeatures
    futures: FuturesFeatures
    orderbook: OrderbookFeatures
    timeframe_alignment: float
    distance_to_support_atr: float | None
    distance_to_resistance_atr: float | None

    def to_flat_dict(self) -> dict:
        """Flat, prefix-by-timeframe representation in the spirit of the
        product spec's example (`trend_1m`, `volume_ratio_5m`, ...) — the
        typed structure above is the source of truth, this is a
        convenience view for logging/prompts.
        """
        flat: dict = {
            "timestamp": self.timestamp.isoformat(),
            "symbol": self.symbol,
            "data_quality": self.data_quality,
            "timeframe_alignment": self.timeframe_alignment,
            "distance_to_support_atr": self.distance_to_support_atr,
            "distance_to_resistance_atr": self.distance_to_resistance_atr,
            "funding_rate": self.futures.funding_rate,
            "funding_change": self.futures.funding_change,
            "open_interest": self.futures.open_interest,
            "open_interest_change_pct": self.futures.open_interest_change_pct,
            "oi_price_regime": self.futures.oi_price_regime,
            "orderbook_imbalance": self.orderbook.imbalance,
            "orderbook_imbalance_avg_30_120s": self.orderbook.imbalance_avg_30_120s,
            "spread_change": self.orderbook.spread_change,
            "microprice": self.orderbook.microprice,
            "has_large_wall": self.orderbook.has_large_wall,
            "buy_volume": self.trade_flow.buy_volume,
            "sell_volume": self.trade_flow.sell_volume,
            "delta": self.trade_flow.delta,
            "cumulative_volume_delta": self.trade_flow.cumulative_volume_delta,
        }
        for tf, f in self.trend.items():
            flat[f"trend_{tf}"] = f.trend_state
            flat[f"adx_{tf}"] = f.adx
            flat[f"supertrend_{tf}"] = f.supertrend_direction
            flat[f"structure_{tf}"] = f.structure_direction
            flat[f"break_of_structure_{tf}"] = f.break_of_structure
            flat[f"change_of_character_{tf}"] = f.change_of_character
        for tf, f in self.momentum.items():
            flat[f"rsi_{tf}"] = f.rsi14
            flat[f"macd_hist_{tf}"] = f.macd_hist
            flat[f"bullish_divergence_{tf}"] = f.bullish_divergence
            flat[f"bearish_divergence_{tf}"] = f.bearish_divergence
        for tf, f in self.volatility.items():
            flat[f"atr_percentile_{tf}"] = f.atr_percentile
            flat[f"range_compression_{tf}"] = f.range_compression
            flat[f"volatility_expansion_{tf}"] = f.volatility_expansion
        for tf, f in self.volume.items():
            flat[f"volume_ratio_{tf}"] = f.volume_ratio
            flat[f"volume_spike_{tf}"] = f.volume_spike
        return flat


def _pct_diff(a: float | None, b: float | None) -> float | None:
    if a is None or b is None or b == 0:
        return None
    return (a - b) / b * 100


def _series_change(close: pd.Series, series_fn, lookback: int, min_len: int) -> float | None:
    if len(close) < min_len + lookback:
        return None
    series = series_fn(close).dropna()
    if len(series) <= lookback:
        return None
    return float(series.iloc[-1] - series.iloc[-1 - lookback])


def _ema_slope_pct(close: pd.Series, window: int, lookback: int) -> float | None:
    if len(close) < window + lookback:
        return None
    series = EMAIndicator(close, window=window).ema_indicator().dropna()
    if len(series) <= lookback:
        return None
    past = series.iloc[-1 - lookback]
    if past == 0:
        return None
    return float((series.iloc[-1] - past) / past * 100)


def _adx(df: pd.DataFrame, period: int) -> float | None:
    if len(df) < period * 2:
        return None
    series = ADXIndicator(df["high"], df["low"], df["close"], window=period).adx().dropna()
    return float(series.iloc[-1]) if not series.empty else None


def _supertrend_direction(df: pd.DataFrame, period: int, multiplier: float) -> Literal["UP", "DOWN"] | None:
    if len(df) < period + 2:
        return None
    atr = AverageTrueRange(df["high"], df["low"], df["close"], window=period).average_true_range().to_numpy()
    valid = ~np.isnan(atr)
    if not valid.any():
        return None
    start = int(np.argmax(valid))
    if len(atr) - start < 2:
        return None

    high = df["high"].to_numpy()
    low = df["low"].to_numpy()
    close = df["close"].to_numpy()
    hl2 = (high + low) / 2
    upper = hl2 + multiplier * atr
    lower = hl2 - multiplier * atr

    final_upper = np.copy(upper)
    final_lower = np.copy(lower)
    n = len(df)
    trend_up = np.ones(n, dtype=bool)
    trend_up[start] = close[start] >= lower[start]

    for i in range(start + 1, n):
        final_upper[i] = min(upper[i], final_upper[i - 1]) if close[i - 1] <= final_upper[i - 1] else upper[i]
        final_lower[i] = max(lower[i], final_lower[i - 1]) if close[i - 1] >= final_lower[i - 1] else lower[i]
        if close[i] > final_upper[i - 1]:
            trend_up[i] = True
        elif close[i] < final_lower[i - 1]:
            trend_up[i] = False
        else:
            trend_up[i] = trend_up[i - 1]

    return "UP" if trend_up[-1] else "DOWN"


def _find_swings(df: pd.DataFrame, window: int) -> list[tuple[int, float, str]]:
    """Ordered sequence of (index, price, "high"|"low") fractal swing
    points — same fractal technique as `levels._swing_points`, but kept
    separate: that function clusters swings into nearest support/
    resistance levels, while structure analysis needs the ORDERED
    sequence of recent swings to classify HH/HL/LH/LL, not clusters.
    """
    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()
    n = len(df)
    swings: list[tuple[int, float, str]] = []
    for i in range(window, n - window):
        local_high = highs[i - window : i + window + 1]
        local_low = lows[i - window : i + window + 1]
        # A plateau (two+ adjacent candles sharing the exact same extreme,
        # e.g. the transition candle between an up-leg and a pullback)
        # satisfies the fractal check on consecutive indices — collapse
        # those into one swing point, otherwise the most recent two
        # "swings" compared by _structure_features/_divergence can be an
        # identical value tied against itself, always failing both the
        # higher-than and lower-than check.
        if highs[i] == local_high.max() and not (
            swings and swings[-1][2] == "high" and swings[-1][1] == float(highs[i])
        ):
            swings.append((i, float(highs[i]), "high"))
        if lows[i] == local_low.min() and not (
            swings and swings[-1][2] == "low" and swings[-1][1] == float(lows[i])
        ):
            swings.append((i, float(lows[i]), "low"))
    return swings


def _structure_features(df: pd.DataFrame, window: int) -> dict:
    if len(df) < window * 2 + 1:
        return {
            "direction": "NEUTRAL",
            "higher_high": None,
            "higher_low": None,
            "lower_high": None,
            "lower_low": None,
            "bos": False,
            "choch": False,
        }

    swings = _find_swings(df, window)
    swing_highs = [s for s in swings if s[2] == "high"]
    swing_lows = [s for s in swings if s[2] == "low"]

    higher_high = higher_low = lower_high = lower_low = None
    if len(swing_highs) >= 2:
        higher_high = swing_highs[-1][1] > swing_highs[-2][1]
        lower_high = swing_highs[-1][1] < swing_highs[-2][1]
    if len(swing_lows) >= 2:
        higher_low = swing_lows[-1][1] > swing_lows[-2][1]
        lower_low = swing_lows[-1][1] < swing_lows[-2][1]

    if higher_high and higher_low:
        direction: TrendState = "BULLISH"
    elif lower_high and lower_low:
        direction = "BEARISH"
    else:
        direction = "NEUTRAL"

    last_close = float(df["close"].iloc[-1])
    last_swing_high = swing_highs[-1][1] if swing_highs else None
    last_swing_low = swing_lows[-1][1] if swing_lows else None

    bos = False
    if direction == "BULLISH" and last_swing_high is not None:
        bos = last_close > last_swing_high
    elif direction == "BEARISH" and last_swing_low is not None:
        bos = last_close < last_swing_low

    choch = False
    if direction == "BULLISH" and last_swing_low is not None:
        choch = last_close < last_swing_low
    elif direction == "BEARISH" and last_swing_high is not None:
        choch = last_close > last_swing_high

    return {
        "direction": direction,
        "higher_high": higher_high,
        "higher_low": higher_low,
        "lower_high": lower_high,
        "lower_low": lower_low,
        "bos": bos,
        "choch": choch,
    }


def _divergence(df: pd.DataFrame, rsi_series: pd.Series, window: int) -> tuple[bool, bool]:
    if len(df) < window * 2 + 1:
        return False, False
    swings = _find_swings(df, window)
    lows = [s for s in swings if s[2] == "low"]
    highs = [s for s in swings if s[2] == "high"]

    bullish = False
    if len(lows) >= 2:
        (i1, p1, _), (i2, p2, _) = lows[-2], lows[-1]
        if i1 < len(rsi_series) and i2 < len(rsi_series):
            r1, r2 = rsi_series.iloc[i1], rsi_series.iloc[i2]
            if pd.notna(r1) and pd.notna(r2) and p2 < p1 and r2 > r1:
                bullish = True

    bearish = False
    if len(highs) >= 2:
        (i1, p1, _), (i2, p2, _) = highs[-2], highs[-1]
        if i1 < len(rsi_series) and i2 < len(rsi_series):
            r1, r2 = rsi_series.iloc[i1], rsi_series.iloc[i2]
            if pd.notna(r1) and pd.notna(r2) and p2 > p1 and r2 < r1:
                bearish = True

    return bullish, bearish


def _atr_percentile(df: pd.DataFrame, period: int, lookback: int) -> float | None:
    if len(df) < period + 2:
        return None
    atr_series = AverageTrueRange(df["high"], df["low"], df["close"], window=period).average_true_range().dropna()
    if atr_series.empty:
        return None
    window = atr_series.tail(lookback)
    current = window.iloc[-1]
    return float((window <= current).mean() * 100)


def _realized_volatility_pct(df: pd.DataFrame, window: int) -> float | None:
    if len(df) < window + 1:
        return None
    closes = df["close"].tail(window + 1)
    log_returns = np.log(closes / closes.shift(1)).dropna()
    if log_returns.empty:
        return None
    return float(log_returns.std() * 100)


def _trend_features(df: pd.DataFrame) -> TrendFeatures:
    ind = indicators.compute_indicators(df)
    price = ind["last_price"]
    ema20, ema50, ema200 = ind["ema20"], ind["ema50"], ind["ema200"]

    price_vs_ema20 = _pct_diff(price, ema20)
    price_vs_ema50 = _pct_diff(price, ema50)
    price_vs_ema200 = _pct_diff(price, ema200)
    ema20_slope = _ema_slope_pct(df["close"], 20, config.FEATURE_EMA_SLOPE_LOOKBACK)
    ema20_50_dist = _pct_diff(ema20, ema50)
    ema50_200_dist = _pct_diff(ema50, ema200)
    adx = _adx(df, config.FEATURE_ADX_PERIOD)
    supertrend_dir = _supertrend_direction(df, config.FEATURE_SUPERTREND_PERIOD, config.FEATURE_SUPERTREND_MULTIPLIER)
    structure = _structure_features(df, config.FEATURE_STRUCTURE_SWING_WINDOW)

    votes = [v for v in (price_vs_ema20, price_vs_ema50, price_vs_ema200) if v is not None]
    if not votes:
        trend_state: TrendState = "NEUTRAL"
    elif all(v > 0 for v in votes):
        trend_state = "BULLISH"
    elif all(v < 0 for v in votes):
        trend_state = "BEARISH"
    else:
        trend_state = "NEUTRAL"

    return TrendFeatures(
        price_vs_ema20_pct=price_vs_ema20,
        price_vs_ema50_pct=price_vs_ema50,
        price_vs_ema200_pct=price_vs_ema200,
        ema20_slope_pct=ema20_slope,
        ema20_ema50_distance_pct=ema20_50_dist,
        ema50_ema200_distance_pct=ema50_200_dist,
        adx=adx,
        supertrend_direction=supertrend_dir,
        structure_direction=structure["direction"],
        higher_high=structure["higher_high"],
        higher_low=structure["higher_low"],
        lower_high=structure["lower_high"],
        lower_low=structure["lower_low"],
        break_of_structure=structure["bos"],
        change_of_character=structure["choch"],
        trend_state=trend_state,
    )


def _momentum_features(df: pd.DataFrame) -> MomentumFeatures:
    ind = indicators.compute_indicators(df)
    close = df["close"]
    lookback = config.FEATURE_RATE_OF_CHANGE_LOOKBACK

    rsi_roc = _series_change(close, lambda c: RSIIndicator(c, window=14).rsi(), lookback, min_len=15)
    macd_hist_change = _series_change(
        close, lambda c: MACD(c, window_slow=26, window_fast=12, window_sign=9).macd_diff(), lookback, min_len=26
    )

    roc = None
    if len(df) >= lookback + 1:
        roc_series = ROCIndicator(close, window=lookback).roc().dropna()
        roc = float(roc_series.iloc[-1]) if not roc_series.empty else None

    momentum = None
    if len(df) > lookback:
        momentum = float(close.iloc[-1] - close.iloc[-1 - lookback])

    bullish_div = bearish_div = False
    if len(df) >= 15:
        rsi_series = RSIIndicator(close, window=14).rsi()
        bullish_div, bearish_div = _divergence(df, rsi_series, config.FEATURE_STRUCTURE_SWING_WINDOW)

    return MomentumFeatures(
        rsi14=ind["rsi14"],
        rsi_roc=rsi_roc,
        macd_hist=ind["macd_hist"],
        macd_hist_change=macd_hist_change,
        roc=roc,
        momentum=momentum,
        bullish_divergence=bullish_div,
        bearish_divergence=bearish_div,
    )


def _volatility_features(df: pd.DataFrame) -> VolatilityFeatures:
    ind = indicators.compute_indicators(df)
    atr = ind["atr14"]
    atr_pct = _atr_percentile(df, 14, config.FEATURE_ATR_PERCENTILE_LOOKBACK)
    bb_upper, bb_mid, bb_lower = ind["bb_upper"], ind["bb_mid"], ind["bb_lower"]

    bb_width = None
    if bb_upper is not None and bb_lower is not None and bb_mid:
        bb_width = (bb_upper - bb_lower) / bb_mid * 100

    realized_vol = _realized_volatility_pct(df, config.FEATURE_REALIZED_VOL_WINDOW)
    range_compression = atr_pct is not None and atr_pct <= config.FEATURE_RANGE_COMPRESSION_PERCENTILE
    volatility_expansion = atr_pct is not None and atr_pct >= config.FEATURE_VOLATILITY_EXPANSION_PERCENTILE

    last_range_ratio = None
    if atr:
        last_range = float(df["high"].iloc[-1] - df["low"].iloc[-1])
        last_range_ratio = last_range / atr

    return VolatilityFeatures(
        atr14=atr,
        atr_percentile=atr_pct,
        bollinger_width_pct=bb_width,
        realized_volatility_pct=realized_vol,
        range_compression=range_compression,
        volatility_expansion=volatility_expansion,
        last_candle_range_atr_ratio=last_range_ratio,
    )


def _volume_features(df: pd.DataFrame) -> VolumeFeatures:
    volume = df["volume"]
    last_volume = float(volume.iloc[-1])

    ratio = None
    if len(volume) >= config.FEATURE_VOLUME_SMA_WINDOW:
        sma = float(volume.tail(config.FEATURE_VOLUME_SMA_WINDOW).mean())
        ratio = (last_volume / sma) if sma else None
    spike = ratio is not None and ratio >= config.FEATURE_VOLUME_SPIKE_MULTIPLIER

    trend = None
    if len(volume) >= config.FEATURE_VOLUME_TREND_SLOW_WINDOW:
        fast = float(volume.tail(config.FEATURE_VOLUME_TREND_FAST_WINDOW).mean())
        slow = float(volume.tail(config.FEATURE_VOLUME_TREND_SLOW_WINDOW).mean())
        if slow == 0:
            trend = None
        elif fast > slow * 1.05:
            trend = "INCREASING"
        elif fast < slow * 0.95:
            trend = "DECREASING"
        else:
            trend = "FLAT"

    return VolumeFeatures(volume_ratio=ratio, volume_spike=spike, volume_trend=trend)


def _trade_flow_features(trades: list[dict]) -> TradeFlowFeatures:
    # isBuyerMaker=True means the taker was a seller (matched against a
    # resting buy order) -> sell-initiated; False -> buy-initiated.
    buy_volume = sum(t["qty"] for t in trades if not t["is_buyer_maker"])
    sell_volume = sum(t["qty"] for t in trades if t["is_buyer_maker"])
    delta = buy_volume - sell_volume
    cumulative = 0.0
    for t in trades:  # already sorted oldest -> newest by bingx_client.get_recent_trades
        cumulative += t["qty"] if not t["is_buyer_maker"] else -t["qty"]
    return TradeFlowFeatures(
        buy_volume=buy_volume, sell_volume=sell_volume, delta=delta, cumulative_volume_delta=cumulative
    )


def _oi_price_regime(
    price_change: float | None, oi_change: float | None
) -> Literal["NEW_LONGS", "SHORT_COVERING", "NEW_SHORTS", "LONG_COVERING"] | None:
    if not price_change or not oi_change:
        return None
    if price_change > 0 and oi_change > 0:
        return "NEW_LONGS"
    if price_change > 0 and oi_change < 0:
        return "SHORT_COVERING"
    if price_change < 0 and oi_change > 0:
        return "NEW_SHORTS"
    if price_change < 0 and oi_change < 0:
        return "LONG_COVERING"
    return None


def _futures_features(snapshot: MarketDataSnapshot) -> FuturesFeatures:
    funding_change = None
    if snapshot.funding_rate is not None and snapshot.funding_history:
        funding_change = snapshot.funding_rate - snapshot.funding_history[-1]["rate"]

    oi_change = oi_change_pct = price_change = regime = None
    if snapshot.open_interest is not None and snapshot.open_interest_history:
        prev = snapshot.open_interest_history[-1]
        prev_oi = prev.get("open_interest")
        prev_price = prev.get("price")
        if prev_oi is not None:
            oi_change = snapshot.open_interest - prev_oi
            oi_change_pct = (oi_change / prev_oi * 100) if prev_oi else None
        if prev_price is not None and snapshot.price is not None:
            price_change = snapshot.price - prev_price
        regime = _oi_price_regime(price_change, oi_change)

    return FuturesFeatures(
        funding_rate=snapshot.funding_rate,
        funding_change=funding_change,
        open_interest=snapshot.open_interest,
        open_interest_change=oi_change,
        open_interest_change_pct=oi_change_pct,
        oi_price_regime=regime,
    )


def _orderbook_features(snapshot: MarketDataSnapshot, now: float) -> OrderbookFeatures:
    orderbook = snapshot.orderbook
    imbalance = compute_orderbook_imbalance(orderbook)

    spread_change = None
    if snapshot.orderbook_history and snapshot.spread is not None:
        prev_spread = snapshot.orderbook_history[-1].get("spread")
        if prev_spread is not None:
            spread_change = snapshot.spread - prev_spread

    microprice = None
    if orderbook and orderbook.get("bids") and orderbook.get("asks"):
        bid_price, bid_qty = orderbook["bids"][0]
        ask_price, ask_qty = orderbook["asks"][0]
        total = bid_qty + ask_qty
        if total:
            microprice = (bid_price * ask_qty + ask_price * bid_qty) / total

    has_large_wall = None
    if orderbook:
        sizes = [q for _, q in orderbook.get("bids", [])] + [q for _, q in orderbook.get("asks", [])]
        if sizes:
            avg_size = sum(sizes) / len(sizes)
            has_large_wall = avg_size > 0 and max(sizes) >= avg_size * config.FEATURE_LARGE_WALL_MULTIPLIER

    imbalance_avg = None
    try:
        bingx_symbol = config.to_bingx_symbol(snapshot.symbol)
    except ValueError:
        bingx_symbol = None
    if bingx_symbol is not None:
        recent = market_data_history.load_recent(
            bingx_symbol,
            config.FEATURE_ORDERBOOK_IMBALANCE_MIN_WINDOW_SECONDS,
            config.FEATURE_ORDERBOOK_IMBALANCE_MAX_WINDOW_SECONDS,
            now,
        )
        values = [s["orderbook_imbalance"] for s in recent if s.get("orderbook_imbalance") is not None]
        if values:
            imbalance_avg = sum(values) / len(values)

    return OrderbookFeatures(
        imbalance=imbalance,
        imbalance_avg_30_120s=imbalance_avg,
        spread=snapshot.spread,
        spread_change=spread_change,
        microprice=microprice,
        has_large_wall=has_large_wall,
    )


def _empty_feature_set(snapshot: MarketDataSnapshot) -> FeatureSet:
    return FeatureSet(
        timestamp=snapshot.timestamp,
        symbol=snapshot.symbol,
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


def compute_features(snapshot: MarketDataSnapshot, now: float | None = None) -> FeatureSet:
    if snapshot.data_quality == "NO_TRADE":
        return _empty_feature_set(snapshot)

    now = now if now is not None else time.time()

    trend: dict[str, TrendFeatures] = {}
    momentum: dict[str, MomentumFeatures] = {}
    volatility: dict[str, VolatilityFeatures] = {}
    volume: dict[str, VolumeFeatures] = {}
    for interval, tf_data in snapshot.timeframes.items():
        df = tf_data.get("dataframe")
        if df is None or df.empty:
            continue
        trend[interval] = _trend_features(df)
        momentum[interval] = _momentum_features(df)
        volatility[interval] = _volatility_features(df)
        volume[interval] = _volume_features(df)

    trade_flow = _trade_flow_features(snapshot.recent_trades)
    futures = _futures_features(snapshot)
    orderbook = _orderbook_features(snapshot, now)

    distance_support_atr = distance_resistance_atr = None
    levels_tf_data = snapshot.timeframes.get("1h")
    levels_df = levels_tf_data.get("dataframe") if levels_tf_data else None
    atr_1h = volatility["1h"].atr14 if "1h" in volatility else None
    if levels_df is not None and not levels_df.empty and snapshot.price is not None and atr_1h:
        nearby = levels.find_support_resistance(levels_df, snapshot.price)
        if nearby["support"]:
            distance_support_atr = abs(snapshot.price - nearby["support"][0]) / atr_1h
        if nearby["resistance"]:
            distance_resistance_atr = abs(snapshot.price - nearby["resistance"][0]) / atr_1h

    trend_states = [f.trend_state for f in trend.values()]
    if trend_states:
        _, top_count = Counter(trend_states).most_common(1)[0]
        timeframe_alignment = top_count / len(trend_states)
    else:
        timeframe_alignment = 0.0

    return FeatureSet(
        timestamp=snapshot.timestamp,
        symbol=snapshot.symbol,
        data_quality=snapshot.data_quality,
        trend=trend,
        momentum=momentum,
        volatility=volatility,
        volume=volume,
        trade_flow=trade_flow,
        futures=futures,
        orderbook=orderbook,
        timeframe_alignment=timeframe_alignment,
        distance_to_support_atr=distance_support_atr,
        distance_to_resistance_atr=distance_resistance_atr,
    )
