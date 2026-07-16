"""Offline tests for feature_engine.compute_features — synthetic
MarketDataSnapshot objects built by hand (no network, no bingx_client
calls of any kind; market_data_history is redirected to a pytest
tmp_path). Price series are constructed with KNOWN, deliberate behavior
(strictly rising/falling/flat) so the expected category (BULLISH/BEARISH/
NEUTRAL) is unambiguous rather than asserted against a black box.
"""

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

import config
import feature_engine
from market_data_engine import MarketDataSnapshot

NOW = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc).timestamp()


def _df(n=250, start_price=1000.0, step=1.0, interval_seconds=60, volume=100.0, end_time=NOW):
    """`n` candles ending at `end_time`, closes moving by `step` each
    candle (step=0 -> flat/oscillation-free series).
    """
    times = [datetime.fromtimestamp(end_time - (n - 1 - i) * interval_seconds, tz=timezone.utc) for i in range(n)]
    rows = []
    close_prev = start_price
    for i, t in enumerate(times):
        close = start_price + i * step
        open_ = close_prev
        high = max(open_, close) + abs(step) * 0.5 + 0.05
        low = min(open_, close) - abs(step) * 0.5 - 0.05
        rows.append({"open": open_, "high": high, "low": low, "close": close, "volume": volume, "time": t})
        close_prev = close
    return pd.DataFrame(rows)


def _staircase_df(direction=1, legs=6, leg_up=10, leg_pullback=4, step=3.0, start_price=1000.0, interval_seconds=60, volume=100.0, end_time=NOW):
    """A zigzag price path with a clear overall direction — unlike a
    perfectly straight line (which has NO local peaks/troughs at all, so
    the fractal swing detector legitimately finds nothing), this has real
    pullbacks between legs so higher_high/higher_low (or lower_high/
    lower_low) are actually computable, not None.
    """
    closes = [start_price]
    for _ in range(legs):
        for _ in range(leg_up):
            closes.append(closes[-1] + direction * step)
        for _ in range(leg_pullback):
            closes.append(closes[-1] - direction * step * 0.5)
    n = len(closes)
    times = [datetime.fromtimestamp(end_time - (n - 1 - i) * interval_seconds, tz=timezone.utc) for i in range(n)]
    rows = []
    prev = closes[0]
    for t, c in zip(times, closes):
        high = max(prev, c) + 0.05
        low = min(prev, c) - 0.05
        rows.append({"open": prev, "high": high, "low": low, "close": c, "volume": volume, "time": t})
        prev = c
    return pd.DataFrame(rows)


def _timeframes(df_by_interval: dict) -> dict:
    return {
        interval: {"candles": [], "has_gap": False, "is_stale": False, "zero_volume": False, "dataframe": df}
        for interval, df in df_by_interval.items()
    }


def _snapshot(
    timeframes=None,
    price=1249.0,
    data_quality="GOOD",
    funding_rate=None,
    funding_history=None,
    open_interest=None,
    open_interest_history=None,
    orderbook=None,
    orderbook_history=None,
    recent_trades=None,
    symbol="ETHUSDT",
) -> MarketDataSnapshot:
    if timeframes is None:
        timeframes = _timeframes({tf: _df() for tf in config.TIMEFRAMES})
    return MarketDataSnapshot(
        timestamp=datetime.fromtimestamp(NOW, tz=timezone.utc),
        symbol=symbol,
        price=price,
        bid=price - 0.1,
        ask=price + 0.1,
        spread=0.2,
        spread_percent=0.02,
        timeframes=timeframes,
        funding_rate=funding_rate,
        funding_history=funding_history or [],
        open_interest=open_interest,
        open_interest_history=open_interest_history or [],
        orderbook=orderbook,
        orderbook_history=orderbook_history or [],
        volume_24h=None,
        recent_trades=recent_trades or [],
        instrument_rules=None,
        data_quality=data_quality,
        quality_issues=[],
    )


def _use_tmp_history(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "MARKET_DATA_HISTORY_FILE", tmp_path / "market_data_history.jsonl")


# --- trend classification ---


def test_strictly_rising_price_is_bullish(monkeypatch, tmp_path):
    _use_tmp_history(monkeypatch, tmp_path)
    df = _df(step=1.0)
    snap = _snapshot(timeframes=_timeframes({"1m": df}), price=float(df["close"].iloc[-1]))
    fs = feature_engine.compute_features(snap, now=NOW)
    t = fs.trend["1m"]
    assert t.trend_state == "BULLISH"
    assert t.ema20_slope_pct is not None and t.ema20_slope_pct > 0
    assert t.supertrend_direction == "UP"


def test_zigzag_uptrend_gives_higher_high_and_higher_low(monkeypatch, tmp_path):
    _use_tmp_history(monkeypatch, tmp_path)
    df = _staircase_df(direction=1)
    snap = _snapshot(timeframes=_timeframes({"1m": df}), price=float(df["close"].iloc[-1]))
    fs = feature_engine.compute_features(snap, now=NOW)
    t = fs.trend["1m"]
    assert t.higher_high is True
    assert t.higher_low is True
    assert t.structure_direction == "BULLISH"


def test_strictly_falling_price_is_bearish(monkeypatch, tmp_path):
    _use_tmp_history(monkeypatch, tmp_path)
    df = _df(step=-1.0, start_price=2000.0)
    snap = _snapshot(timeframes=_timeframes({"1m": df}), price=float(df["close"].iloc[-1]))
    fs = feature_engine.compute_features(snap, now=NOW)
    t = fs.trend["1m"]
    assert t.trend_state == "BEARISH"
    assert t.ema20_slope_pct is not None and t.ema20_slope_pct < 0
    assert t.supertrend_direction == "DOWN"


def test_zigzag_downtrend_gives_lower_high_and_lower_low(monkeypatch, tmp_path):
    _use_tmp_history(monkeypatch, tmp_path)
    df = _staircase_df(direction=-1, start_price=2000.0)
    snap = _snapshot(timeframes=_timeframes({"1m": df}), price=float(df["close"].iloc[-1]))
    fs = feature_engine.compute_features(snap, now=NOW)
    t = fs.trend["1m"]
    assert t.lower_high is True
    assert t.lower_low is True
    assert t.structure_direction == "BEARISH"


def test_flat_price_is_neutral(monkeypatch, tmp_path):
    _use_tmp_history(monkeypatch, tmp_path)
    df = _df(step=0.0)
    snap = _snapshot(timeframes=_timeframes({"1m": df}), price=float(df["close"].iloc[-1]))
    fs = feature_engine.compute_features(snap, now=NOW)
    assert fs.trend["1m"].trend_state == "NEUTRAL"


def test_short_series_gives_none_not_exception(monkeypatch, tmp_path):
    _use_tmp_history(monkeypatch, tmp_path)
    df = _df(n=10, step=1.0)
    snap = _snapshot(timeframes=_timeframes({"1m": df}), price=float(df["close"].iloc[-1]))
    fs = feature_engine.compute_features(snap, now=NOW)
    t = fs.trend["1m"]
    assert t.price_vs_ema200_pct is None
    assert t.adx is None
    m = fs.momentum["1m"]
    assert m.rsi14 is None


def test_timeframe_alignment_all_agree_is_one(monkeypatch, tmp_path):
    _use_tmp_history(monkeypatch, tmp_path)
    df = _df(step=1.0)
    snap = _snapshot(timeframes=_timeframes({tf: df for tf in config.TIMEFRAMES}), price=float(df["close"].iloc[-1]))
    fs = feature_engine.compute_features(snap, now=NOW)
    assert fs.timeframe_alignment == pytest.approx(1.0)


# --- OI/price regime: all four quadrants ---


@pytest.mark.parametrize(
    "price, prev_price, oi, prev_oi, expected",
    [
        (110.0, 100.0, 1100.0, 1000.0, "NEW_LONGS"),
        (110.0, 100.0, 900.0, 1000.0, "SHORT_COVERING"),
        (90.0, 100.0, 1100.0, 1000.0, "NEW_SHORTS"),
        (90.0, 100.0, 900.0, 1000.0, "LONG_COVERING"),
    ],
)
def test_oi_price_regime_quadrants(monkeypatch, tmp_path, price, prev_price, oi, prev_oi, expected):
    _use_tmp_history(monkeypatch, tmp_path)
    snap = _snapshot(
        price=price,
        open_interest=oi,
        open_interest_history=[{"ts": NOW - 60, "symbol": "ETH-USDT", "open_interest": prev_oi, "price": prev_price}],
    )
    fs = feature_engine.compute_features(snap, now=NOW)
    assert fs.futures.oi_price_regime == expected
    assert fs.futures.open_interest_change == pytest.approx(oi - prev_oi)


# --- trade flow (buy/sell/delta/CVD) ---


def test_trade_flow_exact_arithmetic(monkeypatch, tmp_path):
    _use_tmp_history(monkeypatch, tmp_path)
    trades = [
        {"price": 100.0, "qty": 1.0, "time": 1, "is_buyer_maker": False},  # buy 1.0
        {"price": 100.0, "qty": 2.0, "time": 2, "is_buyer_maker": True},  # sell 2.0
        {"price": 100.0, "qty": 0.5, "time": 3, "is_buyer_maker": False},  # buy 0.5
    ]
    snap = _snapshot(recent_trades=trades)
    fs = feature_engine.compute_features(snap, now=NOW)
    assert fs.trade_flow.buy_volume == pytest.approx(1.5)
    assert fs.trade_flow.sell_volume == pytest.approx(2.0)
    assert fs.trade_flow.delta == pytest.approx(-0.5)
    assert fs.trade_flow.cumulative_volume_delta == pytest.approx(-0.5)


# --- orderbook: imbalance / microprice / spread change ---


def test_imbalance_and_microprice_exact_formula(monkeypatch, tmp_path):
    _use_tmp_history(monkeypatch, tmp_path)
    orderbook = {
        "bids": [[100.0, 3.0], [99.9, 1.0]],
        "asks": [[100.2, 1.0], [100.3, 1.0]],
        "timestamp": NOW,
        "age_seconds": 0.5,
    }
    snap = _snapshot(orderbook=orderbook)
    fs = feature_engine.compute_features(snap, now=NOW)
    # imbalance = (bid_qty - ask_qty) / (bid_qty + ask_qty) over ALL listed levels
    bid_qty, ask_qty = 4.0, 2.0
    expected_imbalance = (bid_qty - ask_qty) / (bid_qty + ask_qty)
    assert fs.orderbook.imbalance == pytest.approx(expected_imbalance)
    # microprice uses only the top level
    expected_microprice = (100.0 * 1.0 + 100.2 * 3.0) / (3.0 + 1.0)
    assert fs.orderbook.microprice == pytest.approx(expected_microprice)


def test_spread_change_vs_last_history_sample(monkeypatch, tmp_path):
    _use_tmp_history(monkeypatch, tmp_path)
    snap = _snapshot(orderbook_history=[{"ts": NOW - 30, "symbol": "ETH-USDT", "spread": 0.5}])
    fs = feature_engine.compute_features(snap, now=NOW)
    assert fs.orderbook.spread_change == pytest.approx(0.2 - 0.5)


def test_no_orderbook_gives_none_imbalance_and_microprice(monkeypatch, tmp_path):
    _use_tmp_history(monkeypatch, tmp_path)
    snap = _snapshot(orderbook=None)
    fs = feature_engine.compute_features(snap, now=NOW)
    assert fs.orderbook.imbalance is None
    assert fs.orderbook.microprice is None
    assert fs.orderbook.has_large_wall is None


def test_has_large_wall_detected(monkeypatch, tmp_path):
    _use_tmp_history(monkeypatch, tmp_path)
    # has_large_wall compares each level against the average of ALL levels
    # (including itself) — with only a handful of levels a single outlier
    # can't inflate the average past its own multiplier threshold no
    # matter how large it is, so this needs a realistic level count
    # (matching production's ORDERBOOK_DEPTH=20) for the wall to actually
    # clear config.FEATURE_LARGE_WALL_MULTIPLIER (default 5x).
    bids = [[100.0 - i * 0.01, 1.0] for i in range(10)]
    asks = [[100.2 + i * 0.01, 1.0] for i in range(9)] + [[100.2 + 9 * 0.01, 100.0]]
    orderbook = {"bids": bids, "asks": asks, "timestamp": NOW, "age_seconds": 0.5}
    snap = _snapshot(orderbook=orderbook)
    fs = feature_engine.compute_features(snap, now=NOW)
    assert fs.orderbook.has_large_wall is True


def test_no_large_wall_when_sizes_are_uniform(monkeypatch, tmp_path):
    _use_tmp_history(monkeypatch, tmp_path)
    bids = [[100.0 - i * 0.01, 1.0] for i in range(10)]
    asks = [[100.2 + i * 0.01, 1.0] for i in range(10)]
    orderbook = {"bids": bids, "asks": asks, "timestamp": NOW, "age_seconds": 0.5}
    snap = _snapshot(orderbook=orderbook)
    fs = feature_engine.compute_features(snap, now=NOW)
    assert fs.orderbook.has_large_wall is False


# --- 30-120s rolling imbalance average ---


def test_imbalance_avg_none_when_no_samples_in_window(monkeypatch, tmp_path):
    _use_tmp_history(monkeypatch, tmp_path)
    snap = _snapshot(symbol="ETHUSDT")
    fs = feature_engine.compute_features(snap, now=NOW)
    assert fs.orderbook.imbalance_avg_30_120s is None


def test_imbalance_avg_computed_from_history_window(monkeypatch, tmp_path):
    _use_tmp_history(monkeypatch, tmp_path)
    import market_data_history

    market_data_history.append_and_trim({"ts": NOW - 45, "symbol": "ETH-USDT", "orderbook_imbalance": 0.2})
    market_data_history.append_and_trim({"ts": NOW - 90, "symbol": "ETH-USDT", "orderbook_imbalance": 0.4})
    market_data_history.append_and_trim({"ts": NOW - 500, "symbol": "ETH-USDT", "orderbook_imbalance": 999.0})  # outside window
    snap = _snapshot(symbol="ETHUSDT")
    fs = feature_engine.compute_features(snap, now=NOW)
    assert fs.orderbook.imbalance_avg_30_120s == pytest.approx((0.2 + 0.4) / 2)


# --- NO_TRADE short-circuit ---


def test_no_trade_input_short_circuits_without_computation(monkeypatch, tmp_path):
    _use_tmp_history(monkeypatch, tmp_path)
    snap = _snapshot(data_quality="NO_TRADE")
    fs = feature_engine.compute_features(snap, now=NOW)
    assert fs.data_quality == "NO_TRADE"
    assert fs.trend == {}
    assert fs.momentum == {}
    assert fs.volatility == {}
    assert fs.volume == {}
    assert fs.distance_to_support_atr is None
    assert fs.timeframe_alignment == 0.0


# --- to_flat_dict ---


def test_to_flat_dict_contains_expected_keys(monkeypatch, tmp_path):
    _use_tmp_history(monkeypatch, tmp_path)
    df = _df(step=1.0)
    snap = _snapshot(timeframes=_timeframes({"1m": df}), price=float(df["close"].iloc[-1]))
    fs = feature_engine.compute_features(snap, now=NOW)
    flat = fs.to_flat_dict()
    assert flat["trend_1m"] == "BULLISH"
    assert "volume_ratio_1m" in flat
    assert "adx_1m" in flat
    assert flat["symbol"] == "ETHUSDT"
