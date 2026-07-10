"""Builds the plain-text market snapshot report meant to be pasted into ChatGPT.

This module only formats data that was already fetched/computed elsewhere.
It does not call any network or trading API.
"""

from __future__ import annotations

from datetime import datetime, timezone

import config

NA = "н/д"

MODE_LABELS = {
    "scalping": "скальпинг",
    "swing": "свинг (удержание часы/сутки)",
}

_TASK_TEXTS = {
    "scalping": (
        "Проанализируй эти данные для скальпинга {symbol}. Дай сигнал LONG / SHORT / WAIT, "
        "уровни входа, stop loss, take profit 1/2/3, вероятность сценария и риски."
    ),
    "swing": (
        "Проанализируй эти данные для внутридневной свинг-сделки по {symbol} (трейдер держит "
        "позицию от нескольких часов до суток, а не минуты — это НЕ скальпинг). Дай сигнал "
        "LONG / SHORT / WAIT, уровень входа, stop loss (размещай его за структурными уровнями "
        "15m/1h — EMA20/EMA50, локальные экстремумы, а не в паре долларов от входа), take profit "
        "1/2/3 на более удалённых уровнях поддержки/сопротивления так, чтобы итоговое соотношение "
        "риск/прибыль было не хуже 1:1.5 (если ближайшие цели дают хуже — прямо скажи об этом и "
        "предложи более дальнюю цель или сигнал WAIT), вероятность сценария и риски."
    ),
}


def _fmt(value: float | None, digits: int = 2) -> str:
    if value is None:
        return NA
    return f"{value:.{digits}f}"


def _fmt_pct(value: float | None, digits: int = 2) -> str:
    if value is None:
        return NA
    sign = "+" if value > 0 else ""
    return f"{sign}{value:.{digits}f}%"


def _fmt_candle(candle: dict) -> str:
    ts = candle["time"]
    if hasattr(ts, "strftime"):
        ts_str = ts.strftime("%H:%M:%S")
    else:
        ts_str = str(ts)
    return (
        f"  {ts_str} | O:{candle['open']:.2f} H:{candle['high']:.2f} "
        f"L:{candle['low']:.2f} C:{candle['close']:.2f} V:{candle['volume']:.2f}"
    )


def _fmt_orderbook_side(levels: list[list[float]]) -> str:
    if not levels:
        return f"  {NA}"
    return "\n".join(f"  {price:.2f}  x  {qty:.4f}" for price, qty in levels)


def _fmt_age(seconds: float) -> str:
    if seconds < 90:
        return f"{seconds:.0f} с назад"
    minutes = seconds / 60
    if minutes < 90:
        return f"{minutes:.0f} мин назад"
    return f"{minutes / 60:.1f} ч назад"


def _fmt_funding_history(history: list[dict]) -> str:
    if not history:
        return f"  {NA}"
    lines = []
    for h in history:
        ts = datetime.fromtimestamp(h["time"] / 1000, tz=timezone.utc)
        lines.append(f"  {ts.strftime('%Y-%m-%d %H:%M')} UTC: {h['rate'] * 100:+.4f}%")
    return "\n".join(lines)


def _orderbook_imbalance(orderbook: dict) -> str:
    bids = orderbook.get("bids") or []
    asks = orderbook.get("asks") or []
    bid_qty = sum(qty for _, qty in bids)
    ask_qty = sum(qty for _, qty in asks)
    total = bid_qty + ask_qty
    if total <= 0:
        return NA
    bid_pct = bid_qty / total * 100
    return (
        f"bid {bid_qty:.4f} / ask {ask_qty:.4f} -> {bid_pct:.1f}% объёма на стороне покупателей "
        f"(по {len(bids)} bid- и {len(asks)} ask-уровням)"
    )


def _timeframe_block(name: str, tf_data: dict) -> str:
    ind = tf_data["indicators"]
    lines = [
        f"{name}:",
        f"- Цена: {_fmt(tf_data['price'])}",
        f"- EMA20: {_fmt(ind.get('ema20'))}",
        f"- EMA50: {_fmt(ind.get('ema50'))}",
        f"- EMA200: {_fmt(ind.get('ema200'))}",
        f"- RSI14: {_fmt(ind.get('rsi14'))}",
        f"- MACD: {_fmt(ind.get('macd'), 4)} (сигнал: {_fmt(ind.get('macd_signal'), 4)}, "
        f"гистограмма: {_fmt(ind.get('macd_hist'), 4)})",
        f"- ATR14: {_fmt(ind.get('atr14'))}",
        f"- Bollinger Bands: верх {_fmt(ind.get('bb_upper'))} / "
        f"центр {_fmt(ind.get('bb_mid'))} / низ {_fmt(ind.get('bb_lower'))}",
        f"- Объём (последняя свеча): {_fmt(tf_data['indicators'].get('last_volume'))}",
        f"- Средний объём: {_fmt(tf_data['indicators'].get('avg_volume'))}",
        "- Последние 5 свечей:",
    ]
    lines.extend(_fmt_candle(c) for c in tf_data["candles"])
    return "\n".join(lines)


def build_report(snapshot: dict) -> str:
    ts: datetime = snapshot["timestamp"]
    ts_str = ts.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    mode_key = snapshot.get("mode_key", "scalping")
    if mode_key not in MODE_LABELS:
        mode_key = "scalping"

    lines = [
        f"Пара: {snapshot['symbol']}",
        f"Биржа: {snapshot['exchange']}",
        f"Режим: {MODE_LABELS[mode_key]}",
        f"Дата/время: {ts_str}",
        f"Текущая цена: {_fmt(snapshot['current_price'])}",
        "",
        "Таймфреймы:",
        "",
    ]

    for tf_name in ("1m", "5m", "15m", "1h"):
        lines.append(_timeframe_block(tf_name, snapshot["timeframes"][tf_name]))
        lines.append("")

    lines.append(f"Изменение цены 15m: {_fmt_pct(snapshot['change_15m'])}")
    lines.append(f"Изменение цены 1h: {_fmt_pct(snapshot['change_1h'])}")
    lines.append(f"Изменение цены 24h: {_fmt_pct(snapshot['change_24h'])}")
    lines.append("")

    funding = snapshot.get("funding_rate")
    funding_str = f"{funding * 100:.4f}%" if funding is not None else NA
    lines.append(f"Funding rate (текущий/предсказанный): {funding_str}")
    funding_history = snapshot.get("funding_history") or []
    lines.append("История funding (последние settlement, ~8ч между ними, от старых к новым):")
    lines.append(_fmt_funding_history(funding_history))
    lines.append("")

    oi = snapshot.get("open_interest")
    lines.append(f"Open interest: {_fmt(oi, 2) if oi is not None else NA}")
    oi_trend = snapshot.get("oi_trend")
    if oi_trend:
        pct = oi_trend.get("pct")
        pct_str = f"{pct:+.2f}%" if pct is not None else NA
        lines.append(
            f"Изменение OI с прошлого снимка ({_fmt_age(oi_trend['age_seconds'])}): "
            f"{oi_trend['delta']:+.2f} ({pct_str})"
        )
    else:
        lines.append(
            "Изменение OI: н/д (нет предыдущего локального снимка для сравнения — "
            "появится после следующего обновления данных)"
        )
    lines.append("")

    orderbook = snapshot.get("orderbook")
    lines.append("Стакан заявок:")
    if orderbook:
        bids_display = orderbook["bids"][: config.ORDERBOOK_DISPLAY_DEPTH]
        asks_display = orderbook["asks"][: config.ORDERBOOK_DISPLAY_DEPTH]
        lines.append("- Лучшие bids:")
        lines.append(_fmt_orderbook_side(bids_display))
        lines.append("- Лучшие asks:")
        lines.append(_fmt_orderbook_side(asks_display))
        if orderbook["bids"] and orderbook["asks"]:
            spread = orderbook["asks"][0][0] - orderbook["bids"][0][0]
            spread_pct = spread / orderbook["asks"][0][0] * 100 if orderbook["asks"][0][0] else 0
            lines.append(f"- Спред: {spread:.2f} ({spread_pct:.4f}%)")
        else:
            lines.append(f"- Спред: {NA}")
        lines.append(f"- Дисбаланс объёма: {_orderbook_imbalance(orderbook)}")
    else:
        lines.append(f"- Лучшие bids: {NA}")
        lines.append(f"- Лучшие asks: {NA}")
        lines.append(f"- Спред: {NA}")
        lines.append(f"- Дисбаланс объёма: {NA}")
    lines.append("")

    levels = snapshot["levels"]
    lines.append("Уровни:")
    support = levels.get("support") or []
    resistance = levels.get("resistance") or []
    lines.append(f"- Поддержка: {', '.join(f'{p:.2f}' for p in support) if support else NA}")
    lines.append(f"- Сопротивление: {', '.join(f'{p:.2f}' for p in resistance) if resistance else NA}")
    lines.append("")

    pos = snapshot["position"]
    lines.append("Моя позиция:")
    lines.append(f"- {pos.position}")
    lines.append(f"- Цена входа: {_fmt(pos.entry_price) if pos.entry_price is not None else NA}")
    lines.append(f"- Плечо: {_fmt(pos.leverage, 1) if pos.leverage is not None else NA}")
    lines.append(f"- Stop loss: {_fmt(pos.stop_loss) if pos.stop_loss is not None else NA}")
    lines.append(f"- Take profit: {_fmt(pos.take_profit) if pos.take_profit is not None else NA}")
    lines.append("")

    lines.append("Задача для ChatGPT:")
    lines.append(_TASK_TEXTS[mode_key].format(symbol=snapshot["symbol"]))

    return "\n".join(lines)
