"""Backtesting Engine — walks historical BingX candles bar-by-bar WITHOUT
look-ahead, calling the EXACT SAME feature_engine.compute_features /
market_regime.classify_regime / strategy_engine.score_strategy functions the
live pipeline uses. This is the whole point: one strategy engine, two
callers (live and backtest), never two competing implementations of "the
rules."

No AI involved — score_strategy's rule-based decision IS "the rules" being
tested here; the AI layer (Stage 6) is a separate advisory stage, out of
scope for a historical replay (also non-reproducible and far too slow/
expensive to call per historical bar). Results are in-memory only
(BacktestResult) — no journal_db writes per bar (would be far too slow/
heavy for a realistic backtest window: one full market_snapshot ->
feature_snapshot -> strategy_score INSERT chain per bar, even bars with no
trade).

Honest limitation, not a bug: BingX's public API has no historical
open-interest, orderbook, or trade-tape endpoint (confirmed by direct audit
of bingx_client.py / oi_history.py / market_data_history.py docstrings —
those local logs are only real if the app happened to be running and
polling at that historical moment, which it wasn't for any backtest
period). So during backtesting:
  - TradeFlowFeatures is always zeroed (built only from recent_trades,
    which has no historical endpoint).
  - FuturesFeatures.open_interest_change/oi_price_regime is always None
    (needs a prior LOCAL sample that can't exist for a backtest period);
    funding_rate/funding_change are also None (BingX's funding-history
    endpoint has no date-range params, shallow/recent-only window).
  - OrderbookFeatures is always None (built only from orderbook/
    orderbook_history, both current-only on BingX's public API).
strategy_engine.RULES entries keyed off these fields simply never fire
during backtesting — they already guard on `is not None`. This is the SAME
code path receiving structurally-unavailable-during-replay inputs, not a
different code path. The trend/momentum/volatility/volume-based rules (the
majority of RULES) are fully live throughout, since those are built purely
from candles, which BingX can supply historically without any gaps.

Since strategy_engine only produces a score/decision, not price levels
(those come from the AI's TradePlan live, which this backtest deliberately
excludes), this engine synthesizes a simple ATR/RR-based bracket order from
the same FeatureSet just computed (see `_synthesize_bracket`) — an
explicit, honest substitute for "what the AI would have proposed," not a
claim of matching it. The backtest validates the score engine's
directional/timing judgment against a fixed, reasonable execution policy,
not any specific AI-generated price levels.

Portfolio model: one position at a time (sequential trades), compounding
equity — each closed trade's realized PnL updates a running balance, the
NEXT trade is sized against the UPDATED balance via the exact same
risk_manager.PositionCalculator paper_trading.py already uses live.
"""

from __future__ import annotations

import dataclasses
import statistics
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from typing import Literal

import pandas as pd

import bingx_client
import config
import feature_engine
import indicators
import market_data_engine
import market_regime
import risk_settings_store
import strategy_engine
from market_data_engine import MarketDataSnapshot
from risk_manager import PositionCalculator, PositionStatus, RiskSettings, TakeProfitTarget, TradeScenario

_EXIT_REASONS = ("SL", "TP1", "TP2", "TP3", "TIMEOUT", "AMBIGUOUS")


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Fill:
    fill_type: Literal["ENTRY", "TAKE_PROFIT", "STOP_LOSS", "TIMEOUT", "AMBIGUOUS"]
    label: str | None
    price: Decimal
    quantity: Decimal
    fee_usdt: Decimal
    time: float


@dataclass(frozen=True)
class BacktestTrade:
    side: Literal["LONG", "SHORT"]
    entry_time: float
    entry_price: Decimal
    stop_loss: Decimal
    take_profits: list[tuple[str, Decimal, Decimal]]
    quantity: Decimal
    exit_reason: str
    exit_price: Decimal
    exit_time: float
    duration_seconds: float
    r_multiple: float
    mfe_r: float
    mae_r: float
    realized_pnl_usdt: Decimal
    fees_usdt: Decimal
    fills: list[Fill]
    regime: str
    decision: str
    long_score: float
    short_score: float
    no_trade_score: float


@dataclass(frozen=True)
class EquityPoint:
    time: float
    equity_usdt: Decimal


@dataclass(frozen=True)
class BacktestStats:
    total_trades: int
    wins: int
    win_rate: float | None
    expectancy_r: float | None
    median_r: float | None
    profit_factor: float | None
    profit_factor_undefined: bool
    max_drawdown_r: float | None
    max_drawdown_usdt: Decimal | None
    max_drawdown_pct: float | None
    avg_mfe_r: float | None
    avg_mae_r: float | None
    avg_duration_seconds: float | None
    exit_reason_counts: dict[str, int]
    low_sample: bool
    final_equity_usdt: Decimal
    total_return_pct: float


@dataclass(frozen=True)
class BacktestResult:
    symbol: str
    mode: str
    start: float
    end: float
    trades: list[BacktestTrade]
    equity_curve: list[EquityPoint]
    stats: BacktestStats
    bars_evaluated: int


@dataclass(frozen=True)
class _ExitOutcome:
    exit_reason: str
    exit_price: Decimal
    exit_time: float
    duration_seconds: float
    r_multiple: float
    mfe_r: float
    mae_r: float
    realized_pnl_usdt: Decimal
    fees_usdt: Decimal
    fills: list[Fill]


# ---------------------------------------------------------------------------
# Historical data loading (paginated)
# ---------------------------------------------------------------------------


def _load_full_history(symbol: str, interval: str, start_ms: int, end_ms: int, page_limit: int | None = None) -> pd.DataFrame:
    """Paginates bingx_client.get_klines_range (single-page-only) to cover
    an arbitrary [start_ms, end_ms] range. Loops advancing past the last
    returned candle's time; stops when a page returns fewer than
    `page_limit` rows (proxy for "reached the end of available data"), the
    cursor stops making forward progress, or BingX signals no data left
    for the remaining window (`bingx_client.NoDataError`) — that specific
    exception means "nothing more to fetch here," not a real failure;
    any other BingXError subclass (network/rate-limit) propagates normally.
    """
    page_limit = page_limit if page_limit is not None else config.BACKTEST_KLINE_PAGE_LIMIT
    interval_ms = int(market_data_engine._interval_seconds(interval) * 1000)
    frames: list[pd.DataFrame] = []
    cursor = start_ms

    while cursor < end_ms:
        try:
            page = bingx_client.get_klines_range(symbol, interval, cursor, end_ms, limit=page_limit)
        except bingx_client.NoDataError:
            break
        if page.empty:
            break
        frames.append(page)
        last_time_ms = int(page["time"].max().timestamp() * 1000)
        next_cursor = last_time_ms + interval_ms
        if next_cursor <= cursor:
            break
        cursor = next_cursor
        if len(page) < page_limit:
            break

    if not frames:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume", "time"])
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(subset="time").sort_values("time").reset_index(drop=True)
    return combined


# ---------------------------------------------------------------------------
# No-look-ahead snapshot builder
# ---------------------------------------------------------------------------


def _has_gap(window: pd.DataFrame, interval_s: float, lookback: int = 10) -> bool:
    tail = window.tail(lookback)
    if len(tail) < 2:
        return False
    deltas = tail["time"].diff().dropna().dt.total_seconds()
    return bool((deltas > interval_s * config.MARKET_DATA_MAX_CANDLE_GAP_MULTIPLIER).any())


def _build_backtest_snapshot(
    symbol: str, full_history: dict[str, pd.DataFrame], now: float, current_price: float, kline_limit: int
) -> MarketDataSnapshot:
    """A candle is visible at `now` only once it has FULLY closed:
    open_time + interval_seconds(tf) <= now. `df["time"]` holds each
    candle's OPEN time (confirmed shape from bingx_client._parse_klines).
    This single filter, applied identically to every timeframe including
    the one the walk loop steps on, is the entire no-look-ahead mechanism —
    for higher timeframes it automatically excludes a still-forming bar
    even though the loop has already advanced past that bar's open time.
    """
    now_ts = pd.Timestamp(now, unit="s", tz="UTC")
    timeframes: dict[str, dict] = {}
    quality_issues: list[str] = []
    insufficient_history = False

    for tf, df in full_history.items():
        interval_s = market_data_engine._interval_seconds(tf)
        visible = df[df["time"] + pd.Timedelta(seconds=interval_s) <= now_ts]
        if visible.empty:
            insufficient_history = True
            continue
        window = visible.tail(kline_limit).reset_index(drop=True)
        if len(window) < kline_limit:
            insufficient_history = True
        has_gap = _has_gap(window, interval_s)
        candles = indicators.last_n_candles(window, 5)
        zero_volume = any(c["volume"] == 0 for c in candles[-3:]) if candles else False
        if has_gap:
            quality_issues.append(f"{tf}: пропуск свечи в истории")
        if zero_volume:
            quality_issues.append(f"{tf}: нулевой объём в последних свечах")
        timeframes[tf] = {
            "candles": candles,
            "has_gap": has_gap,
            "is_stale": False,  # staleness is a live wall-clock concept, meaningless in replay
            "zero_volume": zero_volume,
            "dataframe": window,
        }

    if not timeframes:
        data_quality: str = "NO_TRADE"
    elif insufficient_history:
        data_quality = "DEGRADED"
        quality_issues.append("недостаточно истории для разогрева индикаторов на одном или более таймфреймов")
    elif quality_issues:
        data_quality = "DEGRADED"
    else:
        data_quality = "GOOD"

    return MarketDataSnapshot(
        timestamp=datetime.fromtimestamp(now, tz=timezone.utc),
        symbol=symbol,
        price=current_price,
        bid=None,
        ask=None,
        spread=None,
        spread_percent=None,
        timeframes=timeframes,
        funding_rate=None,
        funding_history=[],
        open_interest=None,
        open_interest_history=[],
        orderbook=None,
        orderbook_history=[],
        volume_24h=None,
        recent_trades=[],
        instrument_rules=None,
        data_quality=data_quality,
        quality_issues=quality_issues,
    )


# ---------------------------------------------------------------------------
# Bracket synthesis (entry/stop/take-profit, in the AI's absence)
# ---------------------------------------------------------------------------


def _synthesize_bracket(
    entry_price: Decimal, atr14: Decimal, signal: Literal["LONG", "SHORT"], min_risk_reward: Decimal
) -> tuple[Decimal, list[TakeProfitTarget]]:
    stop_distance = atr14 * Decimal(str(config.BACKTEST_STOP_ATR_MULTIPLIER))
    stop_loss = entry_price - stop_distance if signal == "LONG" else entry_price + stop_distance

    take_profits: list[TakeProfitTarget] = []
    for i, (rr_multiple, close_percent) in enumerate(config.BACKTEST_TP_LEVELS, start=1):
        target_distance = stop_distance * min_risk_reward * Decimal(str(rr_multiple))
        price = entry_price + target_distance if signal == "LONG" else entry_price - target_distance
        take_profits.append(TakeProfitTarget(f"TP{i}", price, Decimal(str(close_percent))))
    return stop_loss, take_profits


# ---------------------------------------------------------------------------
# Candle-bar exit simulation
# ---------------------------------------------------------------------------


def _simulate_exit(
    entry_time: float,
    entry_price: Decimal,
    quantity: Decimal,
    side: Literal["LONG", "SHORT"],
    stop_loss: Decimal,
    take_profits: list[TakeProfitTarget],
    future_bars: pd.DataFrame,
    settings: RiskSettings,
    mode: str,
    entry_fee_usdt: Decimal,
) -> tuple[_ExitOutcome, int]:
    """Walks `future_bars` (primary-timeframe bars strictly after entry, in
    order) looking for SL/TP/trailing/time-based exits, using each bar's
    high/low (not just close). Returns the outcome plus the positional
    index (within `future_bars`) of the bar the trade closed on, so the
    caller can fast-forward its own walk loop past it without
    re-evaluating bars a position was already open through.
    """
    is_long = side == "LONG"
    risk = abs(entry_price - stop_loss)
    slippage_rate = settings.slippage_percent / Decimal("100")
    fee_rate = settings.taker_fee_percent / Decimal("100")
    max_hold = config.PAPER_TRADING_MAX_HOLD_SECONDS.get(mode, float("inf"))
    breakeven_trigger = Decimal(str(config.PAPER_TRADING_BREAKEVEN_TRIGGER_R))

    remaining = quantity
    stop = stop_loss
    breakeven_moved = False
    mfe_r = 0.0
    mae_r = 0.0
    fills: list[Fill] = [Fill("ENTRY", None, entry_price, quantity, entry_fee_usdt, entry_time)]
    sorted_tps = sorted(take_profits, key=lambda tp: tp.price, reverse=is_long)
    filled_labels: set[str] = set()

    last_bar_idx = len(future_bars) - 1
    exit_bar_idx = last_bar_idx

    for pos, (_, bar) in enumerate(future_bars.iterrows()):
        bar_time = bar["time"].timestamp()
        high, low = Decimal(str(bar["high"])), Decimal(str(bar["low"]))

        favorable = (high - entry_price) / risk if is_long else (entry_price - low) / risk
        adverse = (entry_price - low) / risk if is_long else (high - entry_price) / risk
        mfe_r = max(mfe_r, float(favorable))
        mae_r = max(mae_r, float(adverse))

        sl_touched = (low <= stop) if is_long else (high >= stop)
        touched_tps = [
            tp for tp in sorted_tps
            if tp.label not in filled_labels and ((high >= tp.price) if is_long else (low <= tp.price))
        ]

        if sl_touched and touched_tps:
            exit_price = stop * (1 - slippage_rate) if is_long else stop * (1 + slippage_rate)
            fee = exit_price * remaining * fee_rate
            fills.append(Fill("AMBIGUOUS", None, exit_price, remaining, fee, bar_time))
            remaining = Decimal("0")
            exit_bar_idx = pos
            break

        for tp in touched_tps:
            if remaining <= 0:
                break
            portion = min(quantity * tp.close_percent / Decimal("100"), remaining)
            if portion <= 0:
                continue
            fee = tp.price * portion * fee_rate
            fills.append(Fill("TAKE_PROFIT", tp.label, tp.price, portion, fee, bar_time))
            filled_labels.add(tp.label)
            remaining -= portion

        if remaining <= 0:
            exit_bar_idx = pos
            break

        if sl_touched:
            exit_price = stop * (1 - slippage_rate) if is_long else stop * (1 + slippage_rate)
            fee = exit_price * remaining * fee_rate
            fills.append(Fill("STOP_LOSS", None, exit_price, remaining, fee, bar_time))
            remaining = Decimal("0")
            exit_bar_idx = pos
            break

        if not breakeven_moved and favorable >= breakeven_trigger:
            stop = entry_price
            breakeven_moved = True

        if bar_time - entry_time >= max_hold:
            close_price = Decimal(str(bar["close"]))
            exit_price = close_price * (1 - slippage_rate) if is_long else close_price * (1 + slippage_rate)
            fee = exit_price * remaining * fee_rate
            fills.append(Fill("TIMEOUT", None, exit_price, remaining, fee, bar_time))
            remaining = Decimal("0")
            exit_bar_idx = pos
            break

    if remaining > 0 and last_bar_idx >= 0:
        last_bar = future_bars.iloc[last_bar_idx]
        close_price = Decimal(str(last_bar["close"]))
        exit_price = close_price * (1 - slippage_rate) if is_long else close_price * (1 + slippage_rate)
        fee = exit_price * remaining * fee_rate
        fills.append(Fill("TIMEOUT", None, exit_price, remaining, fee, last_bar["time"].timestamp()))
        exit_bar_idx = last_bar_idx

    closing_fills = [f for f in fills if f.fill_type != "ENTRY"]
    last_fill = closing_fills[-1]
    entry_fee_total = fills[0].fee_usdt
    total_fees = entry_fee_total + sum((f.fee_usdt for f in closing_fills), Decimal("0"))

    total_pnl = Decimal("0")
    for f in closing_fills:
        entry_fee_share = entry_fee_total * f.quantity / quantity if quantity else Decimal("0")
        gross = (f.price - entry_price) * f.quantity if is_long else (entry_price - f.price) * f.quantity
        total_pnl += gross - f.fee_usdt - entry_fee_share

    initial_risk_usdt = risk * quantity
    r_multiple = float(total_pnl / initial_risk_usdt) if initial_risk_usdt > 0 else 0.0

    if last_fill.fill_type == "STOP_LOSS":
        exit_reason = "SL"
    elif last_fill.fill_type == "TAKE_PROFIT":
        exit_reason = last_fill.label
    else:
        exit_reason = last_fill.fill_type  # TIMEOUT or AMBIGUOUS

    outcome = _ExitOutcome(
        exit_reason=exit_reason,
        exit_price=last_fill.price,
        exit_time=last_fill.time,
        duration_seconds=last_fill.time - entry_time,
        r_multiple=r_multiple,
        mfe_r=mfe_r,
        mae_r=mae_r,
        realized_pnl_usdt=total_pnl,
        fees_usdt=total_fees,
        fills=fills,
    )
    return outcome, exit_bar_idx


# ---------------------------------------------------------------------------
# Main walk-forward loop
# ---------------------------------------------------------------------------


def _primary_timeframe(mode: str) -> str:
    return config.STRATEGY_SCALPING_PRIMARY_TIMEFRAME if mode == "scalping" else config.STRATEGY_SWING_PRIMARY_TIMEFRAME


def run_backtest(
    symbol: str, mode: Literal["scalping", "swing"], start: float, end: float, settings: RiskSettings | None = None
) -> BacktestResult:
    settings = settings if settings is not None else risk_settings_store.load()
    primary_tf = _primary_timeframe(mode)
    bingx_symbol = config.to_bingx_symbol(symbol)
    primary_interval_s = market_data_engine._interval_seconds(primary_tf)

    full_history: dict[str, pd.DataFrame] = {}
    for tf in config.TIMEFRAMES:
        interval_s = market_data_engine._interval_seconds(tf)
        warm_up_start_ms = int((start - config.KLINE_LIMIT * interval_s * 1.2) * 1000)
        full_history[tf] = _load_full_history(bingx_symbol, tf, warm_up_start_ms, int(end * 1000))

    primary_df = full_history[primary_tf]
    close_times = primary_df["time"] + pd.Timedelta(seconds=primary_interval_s)
    start_ts, end_ts = pd.Timestamp(start, unit="s", tz="UTC"), pd.Timestamp(end, unit="s", tz="UTC")
    step_indices = primary_df.index[(close_times >= start_ts) & (close_times <= end_ts)].tolist()

    equity = settings.account_balance_usdt
    trades: list[BacktestTrade] = []
    equity_curve: list[EquityPoint] = [EquityPoint(start, equity)]
    bars_evaluated = 0

    i = 0
    while i < len(step_indices):
        idx = step_indices[i]
        row = primary_df.loc[idx]
        now = (row["time"] + pd.Timedelta(seconds=primary_interval_s)).timestamp()
        bars_evaluated += 1

        snapshot = _build_backtest_snapshot(bingx_symbol, full_history, now, float(row["close"]), config.KLINE_LIMIT)
        features = feature_engine.compute_features(snapshot, now=now)
        regime = market_regime.classify_regime(features)
        score = strategy_engine.score_strategy(features, regime, mode)

        signal: Literal["LONG", "SHORT"] | None
        if score.decision == "LONG_BIAS":
            signal = "LONG"
        elif score.decision == "SHORT_BIAS":
            signal = "SHORT"
        else:
            signal = None

        traded = False
        if signal is not None:
            vol = features.volatility.get(primary_tf)
            if vol is not None and vol.atr14 is not None:
                entry_price = Decimal(str(row["close"]))
                atr = Decimal(str(vol.atr14))
                stop_loss, take_profits = _synthesize_bracket(entry_price, atr, signal, settings.min_risk_reward)

                scenario = TradeScenario(
                    signal=signal, entry_from=entry_price, entry_to=entry_price, stop_loss=stop_loss,
                    take_profits=take_profits, entry_order_type="MARKET",
                )
                sizing_settings = dataclasses.replace(settings, account_balance_usdt=equity, available_balance_usdt=None)
                calc = PositionCalculator(sizing_settings).calculate(scenario)

                if calc.status == PositionStatus.VALID:
                    quantity = calc.position_size_coin_rounded
                    slippage_rate = settings.slippage_percent / Decimal("100")
                    filled_entry = entry_price * (1 + slippage_rate) if signal == "LONG" else entry_price * (1 - slippage_rate)
                    fee_rate = settings.taker_fee_percent / Decimal("100")
                    entry_fee = filled_entry * quantity * fee_rate

                    future_bars = primary_df.loc[idx + 1:]
                    outcome, exit_bar_idx = _simulate_exit(
                        now, filled_entry, quantity, signal, stop_loss, take_profits, future_bars,
                        settings, mode, entry_fee,
                    )

                    equity += outcome.realized_pnl_usdt
                    trades.append(BacktestTrade(
                        side=signal, entry_time=now, entry_price=filled_entry, stop_loss=stop_loss,
                        take_profits=[(tp.label, tp.price, tp.close_percent) for tp in take_profits],
                        quantity=quantity, exit_reason=outcome.exit_reason, exit_price=outcome.exit_price,
                        exit_time=outcome.exit_time, duration_seconds=outcome.duration_seconds,
                        r_multiple=outcome.r_multiple, mfe_r=outcome.mfe_r, mae_r=outcome.mae_r,
                        realized_pnl_usdt=outcome.realized_pnl_usdt, fees_usdt=outcome.fees_usdt,
                        fills=outcome.fills, regime=regime.regime, decision=score.decision,
                        long_score=score.long_score, short_score=score.short_score, no_trade_score=score.no_trade_score,
                    ))
                    equity_curve.append(EquityPoint(outcome.exit_time, equity))

                    exit_primary_idx = idx + 1 + exit_bar_idx
                    next_i = i + 1
                    while next_i < len(step_indices) and step_indices[next_i] <= exit_primary_idx:
                        next_i += 1
                    i = next_i
                    traded = True

        if not traded:
            i += 1

    stats = _compute_stats(trades, equity_curve, settings.account_balance_usdt)
    return BacktestResult(
        symbol=symbol, mode=mode, start=start, end=end, trades=trades,
        equity_curve=equity_curve, stats=stats, bars_evaluated=bars_evaluated,
    )


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def _max_drawdown(r_values: list[float]) -> float | None:
    """Peak-to-trough of CUMULATIVE r_multiple — third deliberate
    duplication of the same formula already in paper_trading.py /
    prediction_tracker.py, per the established Stage 8 pattern.
    """
    if not r_values:
        return None
    cumulative = 0.0
    peak = 0.0
    max_dd = 0.0
    for r_val in r_values:
        cumulative += r_val
        peak = max(peak, cumulative)
        max_dd = max(max_dd, peak - cumulative)
    return max_dd


def _equity_drawdown(equity_curve: list[EquityPoint]) -> tuple[Decimal, float]:
    """Peak-to-trough of the equity curve ITSELF, in USDT and in percent of
    the peak — a currency-denominated complement to the R-based
    max_drawdown_r above (the user explicitly asked for "equity curve"/
    "drawdown" as currency concepts, not only the R-based statistic).
    """
    if not equity_curve:
        return Decimal("0"), 0.0
    peak = equity_curve[0].equity_usdt
    max_dd_usdt = Decimal("0")
    max_dd_pct = 0.0
    for point in equity_curve:
        peak = max(peak, point.equity_usdt)
        dd_usdt = peak - point.equity_usdt
        max_dd_usdt = max(max_dd_usdt, dd_usdt)
        if peak > 0:
            max_dd_pct = max(max_dd_pct, float(dd_usdt / peak * 100))
    return max_dd_usdt, max_dd_pct


def _compute_stats(trades: list[BacktestTrade], equity_curve: list[EquityPoint], starting_balance: Decimal) -> BacktestStats:
    r_values = [t.r_multiple for t in trades]
    n = len(r_values)
    wins = sum(1 for r in r_values if r > 0)
    gains = sum(r for r in r_values if r > 0)
    losses = sum(-r for r in r_values if r < 0)
    profit_factor = (gains / losses) if losses > 0 else None
    profit_factor_undefined = losses == 0 and gains > 0

    exit_reason_counts = {reason: 0 for reason in _EXIT_REASONS}
    for t in trades:
        if t.exit_reason in exit_reason_counts:
            exit_reason_counts[t.exit_reason] += 1

    mfe_values = [t.mfe_r for t in trades]
    mae_values = [t.mae_r for t in trades]
    durations = [t.duration_seconds for t in trades]

    max_dd_usdt, max_dd_pct = _equity_drawdown(equity_curve)
    final_equity = equity_curve[-1].equity_usdt if equity_curve else starting_balance
    total_return_pct = float((final_equity - starting_balance) / starting_balance * 100) if starting_balance > 0 else 0.0

    return BacktestStats(
        total_trades=n,
        wins=wins,
        win_rate=(wins / n * 100) if n else None,
        expectancy_r=(sum(r_values) / n) if n else None,
        median_r=statistics.median(r_values) if r_values else None,
        profit_factor=profit_factor,
        profit_factor_undefined=profit_factor_undefined,
        max_drawdown_r=_max_drawdown(r_values),
        max_drawdown_usdt=max_dd_usdt,
        max_drawdown_pct=max_dd_pct,
        avg_mfe_r=(sum(mfe_values) / n) if n else None,
        avg_mae_r=(sum(mae_values) / n) if n else None,
        avg_duration_seconds=(sum(durations) / n) if n else None,
        exit_reason_counts=exit_reason_counts,
        low_sample=n < config.MIN_SAMPLE_FOR_STATS,
        final_equity_usdt=final_equity,
        total_return_pct=total_return_pct,
    )
