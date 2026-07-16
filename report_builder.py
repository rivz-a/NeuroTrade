"""Builds the plain-text market snapshot report meant to be pasted into ChatGPT.

This module only formats data that was already fetched/computed elsewhere.
It does not call any network or trading API.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

import config
import feature_engine
import market_data_engine
import market_regime
import risk_settings_store
import strategy_engine
from feature_engine import FeatureSet
from market_regime import RegimeResult
from risk_manager import RiskSettings
from strategy_engine import ScoreResult

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
        "1/2/3 на более удалённых уровнях поддержки/сопротивления. Минимально приемлемый R:R задан "
        "ниже в разделе риск-ограничений — сам его не пересчитывай, но если по факту предлагаемых "
        "тобой уровней он ниже заданного порога, прямо скажи об этом и предложи более дальнюю цель "
        "или сигнал WAIT — не подгоняй числа целей, чтобы формально пройти порог."
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


@dataclass(frozen=True)
class AIContext:
    """Everything Stage 6 adds to the prompt beyond the existing report —
    already-computed interpretation (regime, scores, derived features) and
    the risk restrictions the model must treat as fixed, not recompute.
    """

    regime: RegimeResult
    score: ScoreResult
    features: FeatureSet
    risk_settings: RiskSettings


def build_ai_context(symbol: str, mode: str) -> AIContext | None:
    """Best-effort: `None` on ANY failure (network, insufficient candle
    history, anything) so a problem in this second, independent pipeline
    never blocks the existing, working `fetch_snapshot`-based report — the
    caller just omits the new section and the report is exactly what it
    was before this stage. Runs a SECOND, separate round of BingX calls
    from `fetch_snapshot`'s own (via `market_data_engine.collect_snapshot`)
    — a known, accepted duplication for this iteration; merging the two
    data pipelines is a separate, later cleanup.
    """
    try:
        snapshot = market_data_engine.collect_snapshot(symbol)
        features = feature_engine.compute_features(snapshot)
        regime = market_regime.classify_regime(features)
        score = strategy_engine.score_strategy(features, regime, mode=mode)
        settings = risk_settings_store.load()
        return AIContext(regime=regime, score=score, features=features, risk_settings=settings)
    except Exception:
        return None


def _ai_context_section(ctx: AIContext) -> list[str]:
    lines = [
        "=== Программный анализ (Market Regime / Strategy Score Engine) ===",
        "Это уже посчитанная кодом интерпретация — используй как контекст, не пересчитывай.",
        "",
        f"Режим рынка (определён кодом): {ctx.regime.regime}",
        f"Рекомендация по режиму: {ctx.regime.strategy_hint}",
    ]
    if ctx.regime.regime_by_timeframe:
        tf_regimes = ", ".join(f"{tf}={r}" for tf, r in ctx.regime.regime_by_timeframe.items())
        lines.append(f"Режим по таймфреймам: {tf_regimes}")
    lines.append("")

    lines.append("Оценка сценариев движком (0-100, независимо от модели):")
    lines.append(f"- LONG_SCORE: {ctx.score.long_score:.0f}")
    lines.append(f"- SHORT_SCORE: {ctx.score.short_score:.0f}")
    lines.append(f"- NO_TRADE_SCORE: {ctx.score.no_trade_score:.0f}")
    lines.append(f"- Решение движка: {ctx.score.decision} (качество {ctx.score.quality})")
    if ctx.score.contributions:
        top = sorted(ctx.score.contributions, key=lambda c: abs(c.score), reverse=True)[:6]
        lines.append("- Основные вклады: " + ", ".join(f"{c.feature} {c.score:+.0f}" for c in top))
    lines.append("")

    primary_tf = (
        config.STRATEGY_SCALPING_PRIMARY_TIMEFRAME
        if ctx.score.mode == "scalping"
        else config.STRATEGY_SWING_PRIMARY_TIMEFRAME
    )
    lines.append(f"Признаки, которых нет в таймфреймах выше (опорный таймфрейм {primary_tf}):")
    trend = ctx.features.trend.get(primary_tf)
    if trend is not None:
        lines.append(
            f"- Тренд: {trend.trend_state}, ADX {_fmt(trend.adx, 1)}, "
            f"SuperTrend {trend.supertrend_direction or NA}, структура {trend.structure_direction} "
            f"(HH={trend.higher_high}, HL={trend.higher_low}, LH={trend.lower_high}, LL={trend.lower_low}, "
            f"break_of_structure={trend.break_of_structure}, change_of_character={trend.change_of_character})"
        )
    momentum = ctx.features.momentum.get(primary_tf)
    if momentum is not None:
        lines.append(
            f"- Моментум: bullish_divergence={momentum.bullish_divergence}, "
            f"bearish_divergence={momentum.bearish_divergence}"
        )
    volatility = ctx.features.volatility.get(primary_tf)
    if volatility is not None:
        lines.append(
            f"- Волатильность: ATR percentile {_fmt(volatility.atr_percentile, 0)}%, "
            f"сжатие диапазона={volatility.range_compression}, расширение={volatility.volatility_expansion}"
        )
    volume = ctx.features.volume.get(primary_tf)
    if volume is not None:
        lines.append(
            f"- Объём: ratio {_fmt(volume.volume_ratio, 2)}, spike={volume.volume_spike}, "
            f"тренд объёма={volume.volume_trend or NA}"
        )
    futures = ctx.features.futures
    funding_pct = futures.funding_rate * 100 if futures.funding_rate is not None else None
    lines.append(
        f"- Фьючерсы: funding {_fmt_pct(funding_pct)}, изменение OI "
        f"{_fmt_pct(futures.open_interest_change_pct)}, режим цена/OI: {futures.oi_price_regime or NA}"
    )
    ob = ctx.features.orderbook
    lines.append(f"- Стакан: imbalance {_fmt(ob.imbalance, 2)}, microprice {_fmt(ob.microprice)}")
    if ctx.features.distance_to_support_atr is not None:
        lines.append(f"- Расстояние до поддержки: {_fmt(ctx.features.distance_to_support_atr, 2)} ATR")
    if ctx.features.distance_to_resistance_atr is not None:
        lines.append(f"- Расстояние до сопротивления: {_fmt(ctx.features.distance_to_resistance_atr, 2)} ATR")
    lines.append("")

    rs = ctx.risk_settings
    lines.append("Риск-ограничения (заданы пользователем — НЕ пересчитывать и НЕ менять):")
    lines.append(f"- Риск на сделку: {rs.risk_percent}% от депозита")
    lines.append(f"- Максимальная маржа: {rs.max_margin_percent}%")
    lines.append(f"- Минимальный приемлемый R:R: {rs.min_risk_reward}")
    lines.append(f"- Плечо: {rs.leverage}x")
    lines.append("")

    lines.append(
        "ВАЖНО: не рассчитывай размер позиции, комиссии, маржу или R:R самостоятельно — это "
        "делает код детерминированно (Decimal-арифметика), не языковая модель. Не меняй указанные "
        "выше риск-ограничения. Сделка никогда не открывается автоматически — финальное решение "
        "принимает пользователь вручную, только после прохождения всех программных проверок "
        "(валидация плана, расчёт позиции, фильтры риска)."
    )
    return lines


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


def build_report(snapshot: dict, ai_context: AIContext | None = None) -> str:
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

    if ai_context is not None:
        lines.extend(_ai_context_section(ai_context))
        lines.append("")

    lines.append("Задача для ChatGPT:")
    lines.append(_TASK_TEXTS[mode_key].format(symbol=snapshot["symbol"]))

    return "\n".join(lines)
