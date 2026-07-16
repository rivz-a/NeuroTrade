"""Offline tests for backtest_engine.py — no network, no AI. Pure/synthetic
pd.DataFrame candle fixtures (matching tests/test_feature_engine.py's style)
for the core walk/exit tests; bingx_client.get_klines_range is mocked only
for the pagination-loader tests.
"""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

import pandas as pd
import pytest

import backtest_engine
import bingx_client
import config
import strategy_engine
from backtest_engine import BacktestTrade, EquityPoint
from risk_manager import DEFAULT_RISK_SETTINGS, TakeProfitTarget

NOW = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
NOW_EPOCH = NOW.timestamp()


def _df(n=250, start_price=1000.0, step=0.0, interval_seconds=60, volume=100.0, end_time=NOW_EPOCH, wiggle=0.05) -> pd.DataFrame:
    times = [pd.Timestamp(end_time - (n - 1 - i) * interval_seconds, unit="s", tz="UTC") for i in range(n)]
    rows = []
    close_prev = start_price
    for i, t in enumerate(times):
        close = start_price + i * step
        open_ = close_prev
        high = max(open_, close) + abs(step) * 0.5 + wiggle
        low = min(open_, close) - abs(step) * 0.5 - wiggle
        rows.append({"open": open_, "high": high, "low": low, "close": close, "volume": volume, "time": t})
        close_prev = close
    return pd.DataFrame(rows)


def _explicit_df(times_epoch: list[float], price: float = 100.0, volume: float = 10.0) -> pd.DataFrame:
    rows = [
        {"open": price, "high": price + 1, "low": price - 1, "close": price, "volume": volume, "time": pd.Timestamp(t, unit="s", tz="UTC")}
        for t in times_epoch
    ]
    return pd.DataFrame(rows)


def _bars(rows: list[tuple[float, float, float, float, float, float]]) -> pd.DataFrame:
    """Each row: (time_offset_seconds_from_NOW_EPOCH, open, high, low, close, volume)."""
    return pd.DataFrame(
        [
            {"time": pd.Timestamp(NOW_EPOCH + t, unit="s", tz="UTC"), "open": o, "high": h, "low": l, "close": c, "volume": v}
            for t, o, h, l, c, v in rows
        ]
    )


def _trade(r_multiple=1.0, mfe_r=1.0, mae_r=-0.1, duration_seconds=60.0, exit_reason="TP1", pnl=Decimal("1")) -> BacktestTrade:
    return BacktestTrade(
        side="LONG", entry_time=NOW_EPOCH, entry_price=Decimal("100"), stop_loss=Decimal("95"),
        take_profits=[("TP1", Decimal("105"), Decimal("100"))], quantity=Decimal("1"),
        exit_reason=exit_reason, exit_price=Decimal("105"), exit_time=NOW_EPOCH + duration_seconds,
        duration_seconds=duration_seconds, r_multiple=r_multiple, mfe_r=mfe_r, mae_r=mae_r,
        realized_pnl_usdt=pnl, fees_usdt=Decimal("0.1"), fills=[], regime="TREND_UP", decision="LONG_BIAS",
        long_score=70.0, short_score=20.0, no_trade_score=30.0,
    )


# ---------------------------------------------------------------------------
# Pagination
# ---------------------------------------------------------------------------


def test_load_full_history_single_page(monkeypatch):
    df = _df(n=50, interval_seconds=60, end_time=NOW_EPOCH)
    monkeypatch.setattr(bingx_client, "get_klines_range", lambda symbol, interval, start_ms, end_ms, limit=1000: df)
    result = backtest_engine._load_full_history("ETH-USDT", "1m", 0, 10**12, page_limit=1000)
    assert len(result) == 50


def test_load_full_history_multiple_pages_concatenated(monkeypatch):
    page1 = _df(n=5, interval_seconds=60, end_time=NOW_EPOCH - 300)
    page2 = _df(n=5, interval_seconds=60, end_time=NOW_EPOCH)
    calls = []

    def fake(symbol, interval, start_ms, end_ms, limit=1000):
        calls.append(start_ms)
        return page1 if len(calls) == 1 else page2

    monkeypatch.setattr(bingx_client, "get_klines_range", fake)
    result = backtest_engine._load_full_history(
        "ETH-USDT", "1m", int((NOW_EPOCH - 300 - 5 * 60) * 1000), int(NOW_EPOCH * 1000), page_limit=5
    )
    assert len(calls) == 2
    assert len(result) == 10


def test_load_full_history_stops_on_short_page(monkeypatch):
    page = _df(n=3, interval_seconds=60, end_time=NOW_EPOCH)
    calls = []

    def fake(symbol, interval, start_ms, end_ms, limit=1000):
        calls.append(1)
        return page

    monkeypatch.setattr(bingx_client, "get_klines_range", fake)
    result = backtest_engine._load_full_history("ETH-USDT", "1m", 0, int(NOW_EPOCH * 1000), page_limit=10)
    assert len(calls) == 1
    assert len(result) == 3


def test_load_full_history_no_data_error_stops_pagination(monkeypatch):
    def fake(symbol, interval, start_ms, end_ms, limit=1000):
        raise bingx_client.NoDataError("no data")

    monkeypatch.setattr(bingx_client, "get_klines_range", fake)
    result = backtest_engine._load_full_history("ETH-USDT", "1m", 0, int(NOW_EPOCH * 1000), page_limit=10)
    assert result.empty


# ---------------------------------------------------------------------------
# No-look-ahead snapshot builder
# ---------------------------------------------------------------------------


def test_has_gap_pure_function():
    df = _df(n=10, interval_seconds=60, end_time=NOW_EPOCH)
    assert backtest_engine._has_gap(df, interval_s=60) is False
    df2 = df.copy()
    df2.loc[df2.index[-1], "time"] = df2["time"].iloc[-1] + pd.Timedelta(seconds=600)
    assert backtest_engine._has_gap(df2, interval_s=60) is True


def test_bar_visible_exactly_at_own_close_not_before():
    hourly = _explicit_df([NOW_EPOCH - 7200, NOW_EPOCH - 3600, NOW_EPOCH])
    snapshot = backtest_engine._build_backtest_snapshot("ETHUSDT", {"1h": hourly}, now=NOW_EPOCH, current_price=100.0, kline_limit=220)
    visible_times = list(snapshot.timeframes["1h"]["dataframe"]["time"])
    assert pd.Timestamp(NOW_EPOCH - 3600, unit="s", tz="UTC") in visible_times
    assert pd.Timestamp(NOW_EPOCH, unit="s", tz="UTC") not in visible_times


def test_kline_limit_window_takes_last_n_visible():
    df = _df(n=300, interval_seconds=60, end_time=NOW_EPOCH)
    snapshot = backtest_engine._build_backtest_snapshot("ETHUSDT", {"1m": df}, now=NOW_EPOCH, current_price=100.0, kline_limit=220)
    assert len(snapshot.timeframes["1m"]["dataframe"]) == 220


def test_data_quality_no_trade_when_no_visible_candles():
    history = {"1m": _explicit_df([NOW_EPOCH + 100])}
    snapshot = backtest_engine._build_backtest_snapshot("ETHUSDT", history, now=NOW_EPOCH, current_price=100.0, kline_limit=220)
    assert snapshot.data_quality == "NO_TRADE"


def test_data_quality_degraded_on_insufficient_history():
    df = _df(n=50, interval_seconds=60, end_time=NOW_EPOCH)
    snapshot = backtest_engine._build_backtest_snapshot("ETHUSDT", {"1m": df}, now=NOW_EPOCH, current_price=100.0, kline_limit=220)
    assert snapshot.data_quality == "DEGRADED"


def test_data_quality_good_with_full_clean_history():
    df = _df(n=250, interval_seconds=60, end_time=NOW_EPOCH, volume=10.0)
    snapshot = backtest_engine._build_backtest_snapshot("ETHUSDT", {"1m": df}, now=NOW_EPOCH, current_price=100.0, kline_limit=220)
    assert snapshot.data_quality == "GOOD"


def test_zero_volume_detected():
    df = _df(n=250, interval_seconds=60, end_time=NOW_EPOCH, volume=10.0)
    # df's last row opens exactly at NOW_EPOCH (closes at NOW_EPOCH+60) so it
    # is still forming and excluded from visibility at now=NOW_EPOCH — zero
    # the second-to-last (already-closed) candle instead.
    df.loc[df.index[-2], "volume"] = 0.0
    snapshot = backtest_engine._build_backtest_snapshot("ETHUSDT", {"1m": df}, now=NOW_EPOCH, current_price=100.0, kline_limit=220)
    assert snapshot.timeframes["1m"]["zero_volume"] is True
    assert snapshot.data_quality == "DEGRADED"


# ---------------------------------------------------------------------------
# Bracket synthesis
# ---------------------------------------------------------------------------


def test_synthesize_bracket_long():
    entry, atr = Decimal("100"), Decimal("2")
    stop, tps = backtest_engine._synthesize_bracket(entry, atr, "LONG", Decimal("1.5"))
    expected_stop_distance = atr * Decimal(str(config.BACKTEST_STOP_ATR_MULTIPLIER))
    assert stop == entry - expected_stop_distance
    assert len(tps) == 2
    assert tps[0].label == "TP1"
    assert tps[0].price > entry
    assert tps[1].price > tps[0].price


def test_synthesize_bracket_short():
    entry, atr = Decimal("100"), Decimal("2")
    stop, tps = backtest_engine._synthesize_bracket(entry, atr, "SHORT", Decimal("1.5"))
    expected_stop_distance = atr * Decimal(str(config.BACKTEST_STOP_ATR_MULTIPLIER))
    assert stop == entry + expected_stop_distance
    assert tps[0].price < entry
    assert tps[1].price < tps[0].price


def test_synthesize_bracket_tp_levels_scale_with_min_risk_reward():
    entry, atr = Decimal("100"), Decimal("2")
    stop_distance = atr * Decimal(str(config.BACKTEST_STOP_ATR_MULTIPLIER))
    _, tps = backtest_engine._synthesize_bracket(entry, atr, "LONG", Decimal("2.0"))
    expected_tp1 = entry + stop_distance * Decimal("2.0") * Decimal(str(config.BACKTEST_TP_LEVELS[0][0]))
    assert tps[0].price == expected_tp1


def test_synthesize_bracket_close_percents_match_config():
    entry, atr = Decimal("100"), Decimal("2")
    _, tps = backtest_engine._synthesize_bracket(entry, atr, "LONG", Decimal("1.5"))
    for tp, (_, close_pct) in zip(tps, config.BACKTEST_TP_LEVELS):
        assert tp.close_percent == Decimal(str(close_pct))


# ---------------------------------------------------------------------------
# Candle-bar exit simulation
# ---------------------------------------------------------------------------


def test_simulate_exit_long_stop_loss():
    future = _bars([(60, 99, 99.5, 94.0, 95.0, 10)])
    outcome, exit_idx = backtest_engine._simulate_exit(
        NOW_EPOCH, Decimal("100"), Decimal("1"), "LONG", Decimal("95"),
        [TakeProfitTarget("TP1", Decimal("110"), Decimal("100"))], future, DEFAULT_RISK_SETTINGS, "scalping", Decimal("0"),
    )
    assert outcome.exit_reason == "SL"
    slip = DEFAULT_RISK_SETTINGS.slippage_percent / Decimal("100")
    assert outcome.exit_price == Decimal("95") * (1 - slip)
    assert exit_idx == 0


def test_simulate_exit_short_stop_loss():
    future = _bars([(60, 101, 106.0, 100.5, 105.0, 10)])
    outcome, exit_idx = backtest_engine._simulate_exit(
        NOW_EPOCH, Decimal("100"), Decimal("1"), "SHORT", Decimal("105"),
        [TakeProfitTarget("TP1", Decimal("90"), Decimal("100"))], future, DEFAULT_RISK_SETTINGS, "scalping", Decimal("0"),
    )
    assert outcome.exit_reason == "SL"
    slip = DEFAULT_RISK_SETTINGS.slippage_percent / Decimal("100")
    assert outcome.exit_price == Decimal("105") * (1 + slip)


def test_simulate_exit_single_tp_no_slippage():
    future = _bars([(60, 100, 105.5, 99.5, 105.0, 10)])
    outcome, exit_idx = backtest_engine._simulate_exit(
        NOW_EPOCH, Decimal("100"), Decimal("1"), "LONG", Decimal("95"),
        [TakeProfitTarget("TP1", Decimal("105"), Decimal("100"))], future, DEFAULT_RISK_SETTINGS, "scalping", Decimal("0"),
    )
    assert outcome.exit_reason == "TP1"
    assert outcome.exit_price == Decimal("105")


def test_simulate_exit_multi_tp_same_bar_farthest_first():
    future = _bars([(60, 100, 111.0, 99.5, 110.0, 10)])
    outcome, exit_idx = backtest_engine._simulate_exit(
        NOW_EPOCH, Decimal("100"), Decimal("2"), "LONG", Decimal("95"),
        [TakeProfitTarget("TP1", Decimal("105"), Decimal("50")), TakeProfitTarget("TP2", Decimal("110"), Decimal("50"))],
        future, DEFAULT_RISK_SETTINGS, "scalping", Decimal("0"),
    )
    tp_fills = [f for f in outcome.fills if f.fill_type == "TAKE_PROFIT"]
    assert [f.label for f in tp_fills] == ["TP2", "TP1"]
    assert outcome.exit_reason == "TP1"


def test_simulate_exit_partial_tp_then_stop_loss_for_remainder():
    future = _bars([
        (60, 100, 105.5, 99.5, 105.0, 10),
        (120, 105, 105.5, 94.0, 95.0, 10),
    ])
    outcome, exit_idx = backtest_engine._simulate_exit(
        NOW_EPOCH, Decimal("100"), Decimal("2"), "LONG", Decimal("95"),
        [TakeProfitTarget("TP1", Decimal("105"), Decimal("50")), TakeProfitTarget("TP2", Decimal("120"), Decimal("50"))],
        future, DEFAULT_RISK_SETTINGS, "scalping", Decimal("0"),
    )
    tp_fills = [f for f in outcome.fills if f.fill_type == "TAKE_PROFIT"]
    sl_fills = [f for f in outcome.fills if f.fill_type == "STOP_LOSS"]
    assert len(tp_fills) == 1 and tp_fills[0].quantity == Decimal("1")
    assert len(sl_fills) == 1 and sl_fills[0].quantity == Decimal("1")
    assert outcome.exit_reason == "SL"


def test_simulate_exit_ambiguous_same_bar():
    future = _bars([(60, 100, 106.0, 94.0, 100.0, 10)])
    outcome, exit_idx = backtest_engine._simulate_exit(
        NOW_EPOCH, Decimal("100"), Decimal("1"), "LONG", Decimal("95"),
        [TakeProfitTarget("TP1", Decimal("105"), Decimal("100"))], future, DEFAULT_RISK_SETTINGS, "scalping", Decimal("0"),
    )
    assert outcome.exit_reason == "AMBIGUOUS"
    slip = DEFAULT_RISK_SETTINGS.slippage_percent / Decimal("100")
    assert outcome.exit_price == Decimal("95") * (1 - slip)


def test_simulate_exit_trailing_stop_to_breakeven():
    future = _bars([
        (60, 100, 105.5, 99.5, 105.0, 10),
        (120, 105, 105.5, 99.0, 100.0, 10),
    ])
    outcome, exit_idx = backtest_engine._simulate_exit(
        NOW_EPOCH, Decimal("100"), Decimal("1"), "LONG", Decimal("95"),
        [TakeProfitTarget("TP1", Decimal("120"), Decimal("100"))], future, DEFAULT_RISK_SETTINGS, "scalping", Decimal("0"),
    )
    assert outcome.exit_reason == "SL"
    slip = DEFAULT_RISK_SETTINGS.slippage_percent / Decimal("100")
    assert outcome.exit_price == Decimal("100") * (1 - slip)


def test_simulate_exit_time_based_close():
    max_hold = config.PAPER_TRADING_MAX_HOLD_SECONDS["scalping"]
    future = _bars([(max_hold + 60, 100, 100.5, 99.5, 100.2, 10)])
    outcome, exit_idx = backtest_engine._simulate_exit(
        NOW_EPOCH, Decimal("100"), Decimal("1"), "LONG", Decimal("90"),
        [TakeProfitTarget("TP1", Decimal("150"), Decimal("100"))], future, DEFAULT_RISK_SETTINGS, "scalping", Decimal("0"),
    )
    assert outcome.exit_reason == "TIMEOUT"
    slip = DEFAULT_RISK_SETTINGS.slippage_percent / Decimal("100")
    assert outcome.exit_price == Decimal("100.2") * (1 - slip)  # closing a LONG = selling, adverse = lower


def test_simulate_exit_exhausts_future_bars_closes_at_last_close():
    future = _bars([
        (60, 100, 100.5, 99.5, 100.2, 10),
        (120, 100.2, 100.7, 99.9, 100.5, 10),
    ])
    outcome, exit_idx = backtest_engine._simulate_exit(
        NOW_EPOCH, Decimal("100"), Decimal("1"), "LONG", Decimal("50"),
        [TakeProfitTarget("TP1", Decimal("500"), Decimal("100"))], future, DEFAULT_RISK_SETTINGS, "swing", Decimal("0"),
    )
    assert outcome.exit_reason == "TIMEOUT"
    slip = DEFAULT_RISK_SETTINGS.slippage_percent / Decimal("100")
    assert outcome.exit_price == Decimal("100.5") * (1 - slip)  # closing a LONG = selling, adverse = lower
    assert exit_idx == 1


# ---------------------------------------------------------------------------
# run_backtest integration
# ---------------------------------------------------------------------------


def _synced_history(interval_seconds_map: dict[str, int], window_start: float, window_end: float, volume: float = 10.0) -> dict[str, pd.DataFrame]:
    # wiggle=5.0 on a start_price=1000.0 series gives a ~1% ATR — realistic
    # enough that risk_manager.PositionCalculator's per-target net-RR-after-
    # fees gate (see config.BACKTEST_TP_LEVELS' comment) can actually clear.
    result = {}
    for tf, secs in interval_seconds_map.items():
        n = int((window_end - window_start) / secs) + 5
        result[tf] = _df(n=n, interval_seconds=secs, end_time=window_end, volume=volume, wiggle=5.0)
    return result


def _install_fake_history(monkeypatch, window_start: float, window_end: float):
    interval_seconds = {"1m": 60, "5m": 300, "15m": 900, "1h": 3600}
    warm_start = window_start - config.KLINE_LIMIT * 3600 * 1.2
    histories = _synced_history(interval_seconds, warm_start, window_end)
    monkeypatch.setattr(
        backtest_engine, "_load_full_history",
        lambda symbol, interval, start_ms, end_ms, page_limit=None: histories[interval],
    )
    return histories


def _make_score(decision: str, mode: str = "scalping") -> strategy_engine.ScoreResult:
    return strategy_engine.ScoreResult(
        timestamp=NOW, symbol="ETHUSDT", mode=mode, ruleset_version="v1",
        long_score=70.0 if decision == "LONG_BIAS" else 30.0, short_score=20.0, no_trade_score=30.0,
        decision=decision, quality="B", contributions=[],
    )


def test_run_backtest_opens_one_trade_and_compounds_equity(monkeypatch):
    window_start, window_end = NOW_EPOCH, NOW_EPOCH + 300 * 60
    _install_fake_history(monkeypatch, window_start, window_end)

    call_count = {"n": 0}

    def fake_score(features, regime, mode, gross_rr=None):
        call_count["n"] += 1
        return _make_score("LONG_BIAS" if call_count["n"] == 1 else "NO_TRADE", mode)

    monkeypatch.setattr(strategy_engine, "score_strategy", fake_score)

    result = backtest_engine.run_backtest("ETHUSDT", "scalping", window_start, window_end, settings=DEFAULT_RISK_SETTINGS)
    assert len(result.trades) == 1
    assert result.equity_curve[-1].equity_usdt == DEFAULT_RISK_SETTINGS.account_balance_usdt + result.trades[0].realized_pnl_usdt
    assert result.bars_evaluated >= 1


def test_run_backtest_sequential_trades_never_overlap(monkeypatch):
    window_start, window_end = NOW_EPOCH, NOW_EPOCH + 300 * 200
    _install_fake_history(monkeypatch, window_start, window_end)
    monkeypatch.setattr(strategy_engine, "score_strategy", lambda features, regime, mode, gross_rr=None: _make_score("LONG_BIAS", mode))

    result = backtest_engine.run_backtest("ETHUSDT", "scalping", window_start, window_end, settings=DEFAULT_RISK_SETTINGS)
    assert len(result.trades) >= 1
    for a, b in zip(result.trades, result.trades[1:]):
        assert b.entry_time >= a.exit_time

    expected_equity = DEFAULT_RISK_SETTINGS.account_balance_usdt + sum((t.realized_pnl_usdt for t in result.trades), Decimal("0"))
    assert result.equity_curve[-1].equity_usdt == expected_equity


def test_run_backtest_atr_none_skips_signal(monkeypatch):
    window_start, window_end = NOW_EPOCH, NOW_EPOCH + 300 * 10
    _install_fake_history(monkeypatch, window_start, window_end)
    monkeypatch.setattr(strategy_engine, "score_strategy", lambda features, regime, mode, gross_rr=None: _make_score("LONG_BIAS", mode))

    import feature_engine as fe

    real_compute = fe.compute_features

    def fake_compute(snapshot, now=None):
        fs = real_compute(snapshot, now=now)
        import dataclasses as dc
        vol = dict(fs.volatility)
        primary_tf = config.STRATEGY_SCALPING_PRIMARY_TIMEFRAME
        if primary_tf in vol:
            vol[primary_tf] = dc.replace(vol[primary_tf], atr14=None)
        return dc.replace(fs, volatility=vol)

    monkeypatch.setattr(fe, "compute_features", fake_compute)
    monkeypatch.setattr(backtest_engine, "feature_engine", fe)

    result = backtest_engine.run_backtest("ETHUSDT", "scalping", window_start, window_end, settings=DEFAULT_RISK_SETTINGS)
    assert result.trades == []


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def test_compute_stats_basic_formulas():
    trades = [_trade(r_multiple=1.0, pnl=Decimal("10")), _trade(r_multiple=-0.5, pnl=Decimal("-5")), _trade(r_multiple=2.0, pnl=Decimal("20"))]
    equity_curve = [EquityPoint(NOW_EPOCH, Decimal("100")), EquityPoint(NOW_EPOCH + 1, Decimal("110")),
                     EquityPoint(NOW_EPOCH + 2, Decimal("105")), EquityPoint(NOW_EPOCH + 3, Decimal("125"))]
    stats = backtest_engine._compute_stats(trades, equity_curve, Decimal("100"))
    assert stats.total_trades == 3
    assert stats.wins == 2
    assert stats.win_rate == pytest.approx(200 / 3)
    assert stats.expectancy_r == pytest.approx((1.0 - 0.5 + 2.0) / 3)


def test_compute_stats_profit_factor_normal():
    trades = [_trade(r_multiple=2.0, pnl=Decimal("2")), _trade(r_multiple=-1.0, pnl=Decimal("-1"))]
    equity_curve = [EquityPoint(NOW_EPOCH, Decimal("100")), EquityPoint(NOW_EPOCH + 1, Decimal("101"))]
    stats = backtest_engine._compute_stats(trades, equity_curve, Decimal("100"))
    assert stats.profit_factor == pytest.approx(2.0)
    assert stats.profit_factor_undefined is False


def test_compute_stats_profit_factor_undefined_no_losses():
    trades = [_trade(r_multiple=1.0, pnl=Decimal("1"))]
    equity_curve = [EquityPoint(NOW_EPOCH, Decimal("100")), EquityPoint(NOW_EPOCH + 1, Decimal("101"))]
    stats = backtest_engine._compute_stats(trades, equity_curve, Decimal("100"))
    assert stats.profit_factor is None
    assert stats.profit_factor_undefined is True


def test_max_drawdown_r_known_sequence():
    assert backtest_engine._max_drawdown([1.0, 2.0, -1.5, -2.0, 2.5]) == pytest.approx(3.5)


def test_equity_drawdown_known_sequence():
    curve = [EquityPoint(0, Decimal("100")), EquityPoint(1, Decimal("120")), EquityPoint(2, Decimal("90")), EquityPoint(3, Decimal("130"))]
    dd_usdt, dd_pct = backtest_engine._equity_drawdown(curve)
    assert dd_usdt == Decimal("30")
    assert dd_pct == pytest.approx(25.0)


def test_low_sample_flag():
    trades = [_trade()]
    equity_curve = [EquityPoint(NOW_EPOCH, Decimal("100")), EquityPoint(NOW_EPOCH + 1, Decimal("101"))]
    stats = backtest_engine._compute_stats(trades, equity_curve, Decimal("100"))
    assert stats.low_sample is True


def test_total_return_and_final_equity():
    trades = [_trade(pnl=Decimal("10")), _trade(pnl=Decimal("-5"))]
    equity_curve = [EquityPoint(NOW_EPOCH, Decimal("100")), EquityPoint(NOW_EPOCH + 1, Decimal("110")), EquityPoint(NOW_EPOCH + 2, Decimal("105"))]
    stats = backtest_engine._compute_stats(trades, equity_curve, Decimal("100"))
    assert stats.final_equity_usdt == Decimal("105")
    assert stats.total_return_pct == pytest.approx(5.0)
