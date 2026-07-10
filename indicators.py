"""Technical indicator calculations built on top of the `ta` library.

All functions are pure: they take a pandas DataFrame of OHLCV candles
(columns: open, high, low, close, volume, time) and return plain
floats/dicts. No network calls happen here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator, MACD
from ta.volatility import AverageTrueRange, BollingerBands

MIN_CANDLES_FOR_FULL_INDICATORS = 200


def _last_valid(series: pd.Series) -> float | None:
    series = series.dropna()
    if series.empty:
        return None
    value = series.iloc[-1]
    return float(value) if np.isfinite(value) else None


def compute_indicators(df: pd.DataFrame) -> dict:
    """Compute the latest value of each indicator for the given candle set.

    Gracefully returns None for indicators that need more history than the
    dataframe currently has (e.g. EMA200 on a short candle set) instead of
    raising, since a partial report is still useful for a scalping snapshot.
    """
    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    result: dict = {
        "last_price": float(close.iloc[-1]),
        "last_volume": float(volume.iloc[-1]),
        "avg_volume": float(volume.mean()),
    }

    result["ema20"] = _last_valid(EMAIndicator(close, window=20).ema_indicator()) if len(df) >= 20 else None
    result["ema50"] = _last_valid(EMAIndicator(close, window=50).ema_indicator()) if len(df) >= 50 else None
    result["ema200"] = _last_valid(EMAIndicator(close, window=200).ema_indicator()) if len(df) >= 200 else None

    result["rsi14"] = _last_valid(RSIIndicator(close, window=14).rsi()) if len(df) >= 15 else None

    if len(df) >= 26:
        macd_ind = MACD(close, window_slow=26, window_fast=12, window_sign=9)
        result["macd"] = _last_valid(macd_ind.macd())
        result["macd_signal"] = _last_valid(macd_ind.macd_signal())
        result["macd_hist"] = _last_valid(macd_ind.macd_diff())
    else:
        result["macd"] = result["macd_signal"] = result["macd_hist"] = None

    result["atr14"] = (
        _last_valid(AverageTrueRange(high, low, close, window=14).average_true_range())
        if len(df) >= 15
        else None
    )

    if len(df) >= 20:
        bb = BollingerBands(close, window=20, window_dev=2)
        result["bb_upper"] = _last_valid(bb.bollinger_hband())
        result["bb_mid"] = _last_valid(bb.bollinger_mavg())
        result["bb_lower"] = _last_valid(bb.bollinger_lband())
    else:
        result["bb_upper"] = result["bb_mid"] = result["bb_lower"] = None

    return result


def price_change_pct(df: pd.DataFrame, periods: int) -> float | None:
    """% change between the last close and the close `periods` candles ago."""
    if len(df) <= periods:
        return None
    last_close = df["close"].iloc[-1]
    past_close = df["close"].iloc[-1 - periods]
    if past_close == 0:
        return None
    return float((last_close - past_close) / past_close * 100)


def last_n_candles(df: pd.DataFrame, n: int = 5) -> list[dict]:
    tail = df.tail(n)
    candles = []
    for _, row in tail.iterrows():
        candles.append(
            {
                "time": row["time"],
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": float(row["volume"]),
            }
        )
    return candles
