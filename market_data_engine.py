"""Market Data Engine — a separate, normalized, quality-checked market-data
layer (Stage 2 of the roadmap: BingX -> Market Data Engine -> Feature
Engine -> ... -> AI Analysis Engine -> ...).

Deliberately independent of `market_data.py::fetch_snapshot`, the existing
snapshot builder that already feeds `report_builder.py` -> AI. That
pipeline hard-raises on a BingX failure for price/klines; this engine
never does — every failure degrades `data_quality` instead, down to
`"NO_TRADE"` when the data isn't good enough to reason about at all. Not
wired into the AI pipeline in this iteration; it's a standalone, fully
testable module first (same approach as `risk_manager.py` in Stage 1).

Reuses `bingx_client.py` for every BingX call (never re-implements a REST
wrapper) and `oi_history.py` as-is for open-interest history. Only
top-of-book/instrument-spec history is new (`market_data_history.py`).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Literal

import bingx_client
import config
import indicators
import market_data_history
import oi_history

DataQuality = Literal["GOOD", "DEGRADED", "NO_TRADE"]

_INTERVAL_SECONDS = {
    "1m": 60,
    "3m": 180,
    "5m": 300,
    "15m": 900,
    "30m": 1800,
    "1h": 3600,
    "2h": 7200,
    "4h": 14400,
    "1d": 86400,
}

_HISTORY_SAMPLES_RETURNED = 50


@dataclass(frozen=True)
class MarketDataSnapshot:
    timestamp: datetime
    symbol: str
    price: float | None
    bid: float | None
    ask: float | None
    spread: float | None
    spread_percent: float | None
    timeframes: dict[str, dict]
    funding_rate: float | None
    funding_history: list[dict]
    open_interest: float | None
    open_interest_history: list[dict]
    orderbook: dict | None
    orderbook_history: list[dict]
    volume_24h: float | None
    recent_trades: list[dict]
    instrument_rules: dict | None
    data_quality: DataQuality
    quality_issues: list[str]


def _interval_seconds(interval: str) -> float:
    return _INTERVAL_SECONDS.get(interval, 60.0)


def compute_orderbook_imbalance(orderbook: dict | None) -> float | None:
    """(bid_qty - ask_qty) / (bid_qty + ask_qty) over whatever depth the
    order book carries, in [-1, 1] — positive means more size resting on
    the bid side. Public (not `_`-prefixed): both `collect_snapshot`
    (persisted alongside top-of-book, for the rolling-window average) and
    `feature_engine.py` (the live per-snapshot feature) need the exact
    same formula, computed once, not reimplemented twice.
    """
    if not orderbook:
        return None
    bid_qty = sum(q for _, q in orderbook.get("bids", []))
    ask_qty = sum(q for _, q in orderbook.get("asks", []))
    total = bid_qty + ask_qty
    return ((bid_qty - ask_qty) / total) if total else None


def _checkpoint(now: float | None) -> float:
    """Wall-clock time "as of right now" for a staleness comparison. When
    `now` was explicitly injected (tests, or a caller with its own clock
    reference) it's used verbatim throughout, for determinism. When it
    wasn't (the normal production path), a fresh `time.time()` is taken at
    each checkpoint instead of reusing one captured at function entry —
    `collect_snapshot` makes several sequential blocking HTTP calls (four
    kline fetches, price, book ticker, ...) before reaching the order book,
    so a single "now" from the start is already stale by the time the
    order book's own timestamp needs comparing against it (this produced a
    real, live-observed negative age_seconds before this fix).
    """
    return now if now is not None else time.time()


def _normalize_timeframe(interval: str, df, now: float) -> dict:
    """Per-timeframe candle slice plus its own gap/staleness/zero-volume
    flags — computed for every configured timeframe (not just the base
    one used for the hard NO_TRADE gate), so callers can see the full
    picture, not just whichever timeframe happened to trip the gate.
    """
    candles = indicators.last_n_candles(df, 5)
    interval_s = _interval_seconds(interval)

    has_gap = False
    tail = indicators.last_n_candles(df, 10)
    for i in range(1, len(tail)):
        delta = (tail[i]["time"] - tail[i - 1]["time"]).total_seconds()
        if delta > interval_s * config.MARKET_DATA_MAX_CANDLE_GAP_MULTIPLIER:
            has_gap = True
            break

    is_stale = False
    if candles:
        age = now - candles[-1]["time"].timestamp()
        is_stale = age > interval_s * config.MARKET_DATA_MAX_CANDLE_GAP_MULTIPLIER

    zero_volume = any(c["volume"] == 0 for c in candles[-3:]) if candles else False

    return {
        "candles": candles,
        "has_gap": has_gap,
        "is_stale": is_stale,
        "zero_volume": zero_volume,
        # Full fetched history (config.KLINE_LIMIT candles), not just the
        # last 5 kept above for display/logging — feature_engine.py needs
        # this for EMA200/ADX/ATR-percentile/etc., which the compact
        # "candles" list is deliberately too short for. Additive: existing
        # readers (tests, quality checks) only use the four keys above.
        "dataframe": df,
    }


def _check_quality(
    price: float | None,
    timeframes: dict[str, dict],
    orderbook: dict | None,
    instrument_rules: dict | None,
    prev_sample: dict | None,
    now: float,
) -> tuple[DataQuality, list[str]]:
    issues: list[str] = []
    hard_fail = False

    if price is None:
        issues.append("Не удалось получить текущую цену.")
        hard_fail = True

    base_tf = config.TIMEFRAMES[0] if config.TIMEFRAMES else "1m"
    base_data = timeframes.get(base_tf)
    if base_data is None or not base_data["candles"]:
        issues.append(f"Нет свечей базового таймфрейма {base_tf}.")
        hard_fail = True
    else:
        if base_data["has_gap"]:
            issues.append(f"Пропуск свечи в базовом таймфрейме {base_tf}.")
            hard_fail = True
        if base_data["is_stale"]:
            issues.append(f"Последняя свеча {base_tf} устарела.")
            hard_fail = True
        if base_data["zero_volume"]:
            issues.append(f"Нулевой объём в последних свечах {base_tf}.")

    timestamps: dict[str, float] = {"now": now}
    if base_data and base_data["candles"]:
        timestamps["candle"] = base_data["candles"][-1]["time"].timestamp()
    if orderbook and orderbook.get("timestamp") is not None:
        timestamps["orderbook"] = orderbook["timestamp"]
    if len(timestamps) > 1:
        skew = max(timestamps.values()) - min(timestamps.values())
        if skew > config.MARKET_DATA_MAX_SOURCE_TIME_SKEW_SECONDS:
            issues.append(f"Источники данных рассинхронизированы на {skew:.0f} с.")

    if orderbook is None:
        issues.append("Стакан недоступен.")
    elif orderbook.get("age_seconds") is not None and orderbook["age_seconds"] > config.MARKET_DATA_MAX_ORDERBOOK_AGE_SECONDS:
        issues.append(f"Стакан устарел ({orderbook['age_seconds']:.0f} с).")

    if instrument_rules and prev_sample:
        spec_fields = ("quantity_step", "price_step", "minimum_quantity", "minimum_notional_usdt")
        changed = any(str(instrument_rules.get(field)) != str(prev_sample.get(field)) for field in spec_fields)
        if changed:
            issues.append("Спецификация инструмента изменилась с прошлого снимка.")

    if hard_fail:
        return "NO_TRADE", issues
    if issues:
        return "DEGRADED", issues
    return "GOOD", issues


def collect_snapshot(symbol: str, now: float | None = None) -> MarketDataSnapshot:
    # `now` stays exactly as the caller passed it (including None) for the
    # rest of the function — every `_checkpoint(now)` call below re-derives
    # "current time" from it, which only actually refreshes when `now` is
    # still None (the real production path; tests that inject a fixed `now`
    # get fully deterministic behavior instead). Overwriting `now` itself
    # here would freeze every later checkpoint to this one instant, which
    # is exactly the bug a live run against real BingX caught (a negative
    # order-book age, because several sequential HTTP calls happen between
    # this line and the order-book fetch).
    timestamp = datetime.fromtimestamp(_checkpoint(now), tz=timezone.utc)

    try:
        bingx_symbol = config.to_bingx_symbol(symbol)
    except ValueError as exc:
        return MarketDataSnapshot(
            timestamp=timestamp,
            symbol=symbol,
            price=None,
            bid=None,
            ask=None,
            spread=None,
            spread_percent=None,
            timeframes={},
            funding_rate=None,
            funding_history=[],
            open_interest=None,
            open_interest_history=[],
            orderbook=None,
            orderbook_history=[],
            volume_24h=None,
            recent_trades=[],
            instrument_rules=None,
            data_quality="NO_TRADE",
            quality_issues=[str(exc)],
        )

    issues: list[str] = []

    price: float | None = None
    try:
        price = bingx_client.get_price(bingx_symbol)
    except bingx_client.BingXError as exc:
        issues.append(f"Не удалось получить цену: {exc}")

    timeframes: dict[str, dict] = {}
    for interval in config.TIMEFRAMES:
        try:
            df = bingx_client.get_klines(bingx_symbol, interval, config.KLINE_LIMIT)
            timeframes[interval] = _normalize_timeframe(interval, df, _checkpoint(now))
        except bingx_client.BingXError as exc:
            issues.append(f"Не удалось получить свечи {interval}: {exc}")

    book_ticker = bingx_client.get_book_ticker(bingx_symbol)
    bid = ask = spread = spread_percent = None
    if book_ticker is not None:
        bid = book_ticker["bid_price"]
        ask = book_ticker["ask_price"]
        spread = ask - bid
        spread_percent = (spread / ask * 100) if ask else None

    orderbook: dict | None = None
    try:
        raw_orderbook = bingx_client.get_orderbook(bingx_symbol, config.ORDERBOOK_DEPTH)
        timestamp_ms = raw_orderbook.get("timestamp_ms")
        ob_timestamp = (timestamp_ms / 1000) if timestamp_ms is not None else None
        orderbook = {
            "bids": raw_orderbook.get("bids", []),
            "asks": raw_orderbook.get("asks", []),
            "timestamp": ob_timestamp,
            "age_seconds": (_checkpoint(now) - ob_timestamp) if ob_timestamp is not None else None,
        }
    except bingx_client.BingXError as exc:
        issues.append(f"Стакан недоступен: {exc}")

    funding_rate = bingx_client.get_funding_rate(bingx_symbol)
    funding_history = bingx_client.get_funding_rate_history(bingx_symbol, config.FUNDING_HISTORY_LIMIT)
    open_interest = bingx_client.get_open_interest(bingx_symbol)
    ticker_24h = bingx_client.get_ticker_24h(bingx_symbol)
    volume_24h = ticker_24h["volume"] if ticker_24h is not None else None
    recent_trades = bingx_client.get_recent_trades(bingx_symbol, config.MARKET_DATA_TRADES_LIMIT)
    recent_trades = recent_trades[-config.MARKET_DATA_TRADES_LIMIT :]
    instrument_rules_raw = bingx_client.get_instrument_rules(bingx_symbol)
    instrument_rules = (
        {k: str(v) for k, v in instrument_rules_raw.items()} if instrument_rules_raw is not None else None
    )

    # Read prior samples BEFORE recording this one, so "history" means
    # "before this snapshot" (matches market_data_history's read-before-
    # append convention below) rather than including itself.
    open_interest_history = oi_history.load_samples(bingx_symbol)[-_HISTORY_SAMPLES_RETURNED:]
    oi_history.record_and_get_trend(bingx_symbol, open_interest, price)

    prev_samples = market_data_history.load_samples(bingx_symbol)
    prev_sample = prev_samples[-1] if prev_samples else None
    orderbook_history = prev_samples[-_HISTORY_SAMPLES_RETURNED:]

    check_time = _checkpoint(now)
    data_quality, quality_issues = _check_quality(price, timeframes, orderbook, instrument_rules, prev_sample, check_time)
    issues.extend(quality_issues)

    market_data_history.append_and_trim(
        {
            "ts": check_time,
            "symbol": bingx_symbol,
            "best_bid": bid,
            "best_ask": ask,
            "spread": spread,
            "spread_percent": spread_percent,
            "orderbook_imbalance": compute_orderbook_imbalance(orderbook),
            "quantity_step": instrument_rules.get("quantity_step") if instrument_rules else None,
            "price_step": instrument_rules.get("price_step") if instrument_rules else None,
            "minimum_quantity": instrument_rules.get("minimum_quantity") if instrument_rules else None,
            "minimum_notional_usdt": instrument_rules.get("minimum_notional_usdt") if instrument_rules else None,
            "data_quality": data_quality,
        }
    )

    return MarketDataSnapshot(
        timestamp=timestamp,
        symbol=symbol,
        price=price,
        bid=bid,
        ask=ask,
        spread=spread,
        spread_percent=spread_percent,
        timeframes=timeframes,
        funding_rate=funding_rate,
        funding_history=funding_history,
        open_interest=open_interest,
        open_interest_history=open_interest_history,
        orderbook=orderbook,
        orderbook_history=orderbook_history,
        volume_24h=volume_24h,
        recent_trades=recent_trades,
        instrument_rules=instrument_rules,
        data_quality=data_quality,
        quality_issues=issues,
    )
