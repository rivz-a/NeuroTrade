"""Fetches a full market snapshot from BingX (price, klines, indicators,
orderbook, funding/OI, support/resistance levels).

Shared by main.py (one-shot CLI report) and server.py (live dashboard server)
so there is a single source of truth for how a snapshot is built. Only public
BingX market-data endpoints are used — no API key, no trading.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Callable

import bingx_client
import config
import oi_history
import prediction_tracker
from indicators import compute_indicators, last_n_candles, price_change_pct
from levels import find_support_resistance

Logger = Callable[[str], None]


def fetch_snapshot(symbol_input: str, log: Logger = lambda msg: None) -> dict:
    bingx_symbol = config.to_bingx_symbol(symbol_input)

    log(f"Получаю текущую цену {bingx_symbol}...")
    current_price = bingx_client.get_price(bingx_symbol)

    # Score any past predictions whose horizon has elapsed using this fresh
    # price, before any new predictions get made this round.
    prediction_tracker.evaluate_due(current_price)

    timeframes: dict = {}
    klines_by_tf: dict = {}
    for tf in config.TIMEFRAMES:
        log(f"Загружаю свечи {tf}...")
        df = bingx_client.get_klines(bingx_symbol, tf, config.KLINE_LIMIT)
        klines_by_tf[tf] = df
        indicators = compute_indicators(df)
        timeframes[tf] = {
            "price": float(df["close"].iloc[-1]),
            "indicators": indicators,
            "candles": last_n_candles(df, 5),
        }

    change_15m = price_change_pct(klines_by_tf["1m"], 15)
    change_1h = price_change_pct(klines_by_tf["1m"], 60)

    ticker_24h = bingx_client.get_ticker_24h(bingx_symbol)
    if ticker_24h is not None:
        change_24h = ticker_24h["price_change_percent"]
    else:
        change_24h = price_change_pct(klines_by_tf["1h"], 24)

    log("Загружаю funding rate и open interest...")
    funding_rate = bingx_client.get_funding_rate(bingx_symbol)
    funding_history = bingx_client.get_funding_rate_history(bingx_symbol, config.FUNDING_HISTORY_LIMIT)
    open_interest = bingx_client.get_open_interest(bingx_symbol)
    oi_trend = oi_history.record_and_get_trend(bingx_symbol, open_interest, current_price)

    log("Загружаю стакан заявок...")
    try:
        orderbook = bingx_client.get_orderbook(bingx_symbol, config.ORDERBOOK_DEPTH)
    except bingx_client.BingXError as exc:
        log(f"Стакан заявок недоступен: {exc}")
        orderbook = None

    levels = find_support_resistance(klines_by_tf["1h"], current_price)

    position = config.load_position_config()

    return {
        "symbol": symbol_input.upper(),
        "exchange": config.EXCHANGE_NAME,
        "timestamp": datetime.now(timezone.utc),
        "current_price": current_price,
        "timeframes": timeframes,
        "change_15m": change_15m,
        "change_1h": change_1h,
        "change_24h": change_24h,
        "funding_rate": funding_rate,
        "funding_history": funding_history,
        "open_interest": open_interest,
        "oi_trend": oi_trend,
        "orderbook": orderbook,
        "levels": levels,
        "position": position,
    }
