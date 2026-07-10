"""Simple local support/resistance detection from OHLCV candles.

Uses a fractal-style swing high/low search (a candle whose high/low is the
extreme within a surrounding window) and clusters nearby swing points into
single levels so the report doesn't list near-duplicate prices.
"""

from __future__ import annotations

import pandas as pd


def _swing_points(df: pd.DataFrame, window: int = 3) -> tuple[list[float], list[float]]:
    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()
    n = len(df)

    swing_highs: list[float] = []
    swing_lows: list[float] = []

    for i in range(window, n - window):
        local_high_slice = highs[i - window : i + window + 1]
        local_low_slice = lows[i - window : i + window + 1]
        if highs[i] == local_high_slice.max():
            swing_highs.append(float(highs[i]))
        if lows[i] == local_low_slice.min():
            swing_lows.append(float(lows[i]))

    return swing_highs, swing_lows


def _cluster(levels: list[float], tolerance_pct: float) -> list[float]:
    if not levels:
        return []
    levels = sorted(levels)
    clusters: list[list[float]] = [[levels[0]]]
    for level in levels[1:]:
        cluster_avg = sum(clusters[-1]) / len(clusters[-1])
        if abs(level - cluster_avg) / cluster_avg <= tolerance_pct:
            clusters[-1].append(level)
        else:
            clusters.append([level])
    return [sum(c) / len(c) for c in clusters]


def find_support_resistance(
    df: pd.DataFrame,
    current_price: float,
    window: int = 3,
    tolerance_pct: float = 0.0015,
    num_levels: int = 3,
) -> dict[str, list[float]]:
    """Return the nearest `num_levels` support and resistance prices."""
    if len(df) < window * 2 + 1:
        return {"support": [], "resistance": []}

    swing_highs, swing_lows = _swing_points(df, window=window)
    resistance_candidates = _cluster(swing_highs, tolerance_pct)
    support_candidates = _cluster(swing_lows, tolerance_pct)

    resistance = sorted(
        [lvl for lvl in resistance_candidates if lvl > current_price],
        key=lambda lvl: lvl - current_price,
    )[:num_levels]
    support = sorted(
        [lvl for lvl in support_candidates if lvl < current_price],
        key=lambda lvl: current_price - lvl,
    )[:num_levels]

    return {
        "support": sorted(support, reverse=True),
        "resistance": sorted(resistance),
    }
