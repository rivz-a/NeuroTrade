"""Walks the real BingX candle path after a prediction to see what actually
would have happened — stop loss hit first, a take profit hit first, or
neither (timeout) — instead of just comparing the price at a fixed deadline.

Pure function, no network/AI — the caller (`prediction_tracker.py`) is
responsible for fetching the candles (`bingx_client.get_klines_range`) for
the exact [predicted_at, predicted_at + horizon] window.

Simplifications (see the Stage 3 plan): entry is assumed to fill instantly
at the given `entry_price` (already the model's entry-zone midpoint) rather
than waiting for price to actually reach a pullback/breakout zone; a candle
that touches both stop loss and a take profit in the same bar can't be
resolved from OHLC alone, so it's marked "AMBIGUOUS" and scored as if the
stop loss hit — the conservative assumption.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd

ExitReason = Literal["SL", "TP1", "TP2", "TP3", "TIMEOUT", "AMBIGUOUS"]


@dataclass(frozen=True)
class SimulatedOutcome:
    exit_reason: ExitReason
    r_multiple: float  # net of round-trip commission, in units of initial risk
    mfe_r: float  # max favorable excursion reached along the whole path, in R
    mae_r: float  # max adverse excursion reached along the whole path, in R
    exit_price: float
    exit_time: float  # epoch seconds
    duration_seconds: float


def simulate_outcome(
    signal: Literal["LONG", "SHORT"],
    entry_price: float,
    stop_loss: float,
    take_profits: list[tuple[str, float]],
    candles: pd.DataFrame,
    commission_pct: float,
    slippage_pct: float,
) -> SimulatedOutcome | None:
    """None when there's nothing to simulate — no candles, or a degenerate
    (zero-risk) plan where entry and stop coincide after slippage.
    """
    if candles.empty:
        return None

    is_long = signal == "LONG"
    filled_entry = entry_price * (1 + slippage_pct) if is_long else entry_price * (1 - slippage_pct)
    risk = abs(filled_entry - stop_loss)
    if risk <= 0:
        return None

    # Farthest-target-first, so that if price swept through several targets
    # in one bar, we credit the best one actually reached rather than the
    # nearest (arbitrary) one.
    sorted_tps = sorted(take_profits, key=lambda t: t[1], reverse=is_long)

    entry_time = candles.iloc[0]["time"]
    mfe_r = 0.0
    mae_r = 0.0
    exit_reason: ExitReason | None = None
    exit_price: float | None = None
    exit_time = None

    for _, row in candles.iterrows():
        high = row["high"]
        low = row["low"]

        if is_long:
            favorable_r = (high - filled_entry) / risk
            adverse_r = (filled_entry - low) / risk
        else:
            favorable_r = (filled_entry - low) / risk
            adverse_r = (high - filled_entry) / risk
        mfe_r = max(mfe_r, favorable_r)
        mae_r = max(mae_r, adverse_r)

        sl_hit = (low <= stop_loss) if is_long else (high >= stop_loss)
        hit_tp = None
        for label, price in sorted_tps:
            reached = (high >= price) if is_long else (low <= price)
            if reached:
                hit_tp = (label, price)
                break

        if sl_hit and hit_tp:
            exit_reason, exit_price, exit_time = "AMBIGUOUS", stop_loss, row["time"]
            break
        if sl_hit:
            exit_reason, exit_price, exit_time = "SL", stop_loss, row["time"]
            break
        if hit_tp:
            exit_reason, exit_price, exit_time = hit_tp[0], hit_tp[1], row["time"]
            break

    if exit_reason is None:
        last_row = candles.iloc[-1]
        exit_reason, exit_price, exit_time = "TIMEOUT", last_row["close"], last_row["time"]

    raw_r = ((exit_price - filled_entry) / risk) if is_long else ((filled_entry - exit_price) / risk)
    commission_r = (2 * commission_pct * filled_entry) / risk
    r_multiple = raw_r - commission_r

    def _to_epoch(value) -> float:
        return value.timestamp() if hasattr(value, "timestamp") else float(value)

    exit_epoch = _to_epoch(exit_time)
    entry_epoch = _to_epoch(entry_time)

    return SimulatedOutcome(
        exit_reason=exit_reason,
        r_multiple=r_multiple,
        mfe_r=mfe_r,
        mae_r=mae_r,
        exit_price=exit_price,
        exit_time=exit_epoch,
        duration_seconds=exit_epoch - entry_epoch,
    )
