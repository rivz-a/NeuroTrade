"""Builds a static, self-contained HTML dashboard comparing AI trading-signal
analyses from multiple models side by side.

No third-party JS and no cross-origin requests. The one exception is the
"Обновить" button's same-origin call to /api/refresh, which only works when
the file is served by server.py (`python main.py --serve` or
`python server.py`) — opened directly as a local file, the fetch simply
fails and the button shows a friendly hint instead of doing anything
destructive. All AI-generated text is HTML-escaped before embedding to
avoid any injection from untrusted model output.
"""

from __future__ import annotations

import html
import time
from datetime import datetime, timezone

import config
import journal_db
import paper_trading
import position_service
import prediction_tracker
import risk_settings_store
from ai_client import AIAnalysisResult
from ai_schema import TradePlan
from consensus_engine import ConsensusResult, compute_consensus
from position_service import PositionServiceResult
from report_builder import MODE_LABELS
from risk_manager import RiskSettings
from runtime.heartbeat import heartbeat_status
from signal_freshness import Freshness, compute_freshness, format_age, format_remaining
from trade_validator import ValidationIssue, ValidationResult

# Fixed categorical order (dataviz skill: never cycled, never reassigned by rank).
_SLOT_COLORS_LIGHT = ["#2a78d6", "#4a3aa7", "#1baf7a"]  # blue, violet, aqua
_SLOT_COLORS_DARK = ["#3987e5", "#9085e9", "#199e70"]

_STATUS = {
    "LONG": {"icon": "▲", "label": "LONG", "light": "#0ca30c", "dark": "#0ca30c"},
    "SHORT": {"icon": "▼", "label": "SHORT", "light": "#d03b3b", "dark": "#d03b3b"},
    "WAIT": {"icon": "●", "label": "WAIT", "light": "#fab219", "dark": "#fab219"},
}

_SIGNAL_VERB = {
    "LONG": "Открывай на LONG",
    "SHORT": "Открывай на SHORT",
    "WAIT": "Не входи (WAIT)",
}

_ENTRY_STATUS_LABELS = {
    "ENTER_NOW": "Входить сейчас",
    "WAIT_PULLBACK": "Ждать откат",
    "WAIT_BREAKOUT": "Ждать пробой",
    "WAIT_CONFIRMATION": "Ждать подтверждение",
    "LATE_ENTRY": "Поздний вход",
    "REJECTED": "Вход запрещён валидатором",
    "NO_TRADE": "Нет сделки",
}

# "Открывай на LONG/SHORT" is only warranted for ENTER_NOW — every other
# entry_status gets an "idea, not an instruction" phrasing, per the rule
# that a directional signal is not the same thing as permission to enter.
_ENTRY_STATUS_VERBS = {
    ("LONG", "ENTER_NOW"): "Открывай на LONG",
    ("SHORT", "ENTER_NOW"): "Открывай на SHORT",
    ("LONG", "WAIT_PULLBACK"): "Идея LONG — ждать откат",
    ("SHORT", "WAIT_PULLBACK"): "Идея SHORT — ждать откат",
    ("LONG", "WAIT_BREAKOUT"): "Идея LONG — ждать пробой",
    ("SHORT", "WAIT_BREAKOUT"): "Идея SHORT — ждать пробой",
    ("LONG", "WAIT_CONFIRMATION"): "Идея LONG — ждать подтверждение",
    ("SHORT", "WAIT_CONFIRMATION"): "Идея SHORT — ждать подтверждение",
    ("LONG", "LATE_ENTRY"): "Поздний вход — направление LONG",
    ("SHORT", "LATE_ENTRY"): "Поздний вход — направление SHORT",
    ("LONG", "REJECTED"): "Направление LONG, вход запрещён",
    ("SHORT", "REJECTED"): "Направление SHORT, вход запрещён",
    ("LONG", "NO_TRADE"): "Направление LONG, сделки нет",
    ("SHORT", "NO_TRADE"): "Направление SHORT, сделки нет",
}

_TRADE_PERMISSION_LABELS = {
    "ALLOWED": "Разрешено",
    "NOT_ALLOWED": "Не входить",
    "EXPIRED": "Сигнал устарел",
    "PRICE_OUTSIDE_ENTRY_ZONE": "Цена вне зоны входа",
    "WAITING_TRIGGER": "Ждать условие входа",
    "INVALID_PLAN": "Нет валидного плана",
    "WAIT": "WAIT",
}

# Reuses the existing LONG/WAIT/SHORT status colors as good/caution/blocked
# instead of a new palette — same convention as the accuracy panel's meters.
_TRADE_PERMISSION_SEVERITY = {
    "ALLOWED": "LONG",
    "NOT_ALLOWED": "SHORT",
    "EXPIRED": "SHORT",
    "PRICE_OUTSIDE_ENTRY_ZONE": "WAIT",
    "WAITING_TRIGGER": "WAIT",
    "INVALID_PLAN": "SHORT",
    "WAIT": "WAIT",
}


def _signal_verb(signal: str, entry_status: str) -> str:
    if signal == "WAIT":
        return _SIGNAL_VERB["WAIT"]
    return _ENTRY_STATUS_VERBS.get((signal, entry_status), _SIGNAL_VERB.get(signal, _SIGNAL_VERB["WAIT"]))


def _permission_badge(permission: str, reason: str) -> str:
    label = _TRADE_PERMISSION_LABELS.get(permission, permission)
    severity_key = _TRADE_PERMISSION_SEVERITY.get(permission, "WAIT")
    color = _STATUS[severity_key]["light"]
    return (
        f'<div class="trade-permission-badge" style="--status-color: {color}">'
        f'<span class="trade-permission-label">{html.escape(label)}</span>'
        f'<span class="trade-permission-reason">{html.escape(reason)}</span>'
        "</div>"
    )

_MARKET_REGIME_LABELS = {
    "TREND_UP": "восходящий тренд",
    "TREND_DOWN": "нисходящий тренд",
    "RANGE": "боковик",
    "VOLATILITY_EXPANSION": "расширение волатильности",
    "VOLATILITY_COMPRESSION": "сжатие волатильности",
    "REVERSAL_RISK": "риск разворота",
    "UNSTABLE": "нестабильный рынок",
    "UNKNOWN": "неизвестно",
}

_CONFIDENCE_TOOLTIP = "Это субъективная оценка модели, а не статистически подтверждённая вероятность успеха."

_CONSENSUS_STATE_LABELS = {
    "strong": "Сильный консенсус",
    "moderate": "Умеренный консенсус",
    "weak": "Слабый консенсус",
    "conflict": "Конфликт моделей",
    "insufficient_data": "Недостаточно данных",
}

_CONSENSUS_STATE_COLOR = {
    "strong": _STATUS["LONG"]["light"],
    "moderate": _STATUS["WAIT"]["light"],
    "weak": _STATUS["WAIT"]["light"],
    "conflict": _STATUS["SHORT"]["light"],
    "insufficient_data": "var(--text-muted)",
}


def _fmt(value: float | None, digits: int = 2) -> str:
    if value is None:
        return "н/д"
    return f"{value:.{digits}f}"


def _fmt_pct(value: float | None, digits: int = 2) -> str:
    if value is None:
        return "н/д"
    sign = "+" if value > 0 else ""
    return f"{sign}{value:.{digits}f}%"


def _fmt_updated_at(epoch_seconds: float | None) -> str:
    """Wall-clock time an individual card's analysis actually finished — not
    the page-level snapshot timestamp, which now only reflects whichever
    mode/model was refreshed most recently and can be misleading for the
    other (untouched) cards.
    """
    if epoch_seconds is None:
        return "н/д"
    dt = datetime.fromtimestamp(epoch_seconds, tz=timezone.utc)
    return dt.strftime("%Y-%m-%d %H:%M:%S UTC")


def _data_freshness_dot(age_seconds: float) -> str:
    """Green/amber/red dot next to the market header's timestamp — fresh
    (<2 мин), starting to age (2-10 мин), or stale/lost connection feel
    (>10 мин). Thresholds are generous relative to the scalping horizon
    (~30 мин) so a normal refresh cadence stays green.
    """
    if age_seconds < 120:
        color = _STATUS["LONG"]["light"]
    elif age_seconds < 600:
        color = _STATUS["WAIT"]["light"]
    else:
        color = _STATUS["SHORT"]["light"]
    return f'<span class="freshness-dot" style="--status-color: {color}" aria-hidden="true"></span>'


def _runtime_heartbeat_text(status: dict) -> str:
    """Plain-text label for the runtime-liveness badge — shared by the
    server-rendered initial state and the client-side poll loop (which
    receives the same `heartbeat_status()` shape as JSON from /api/status
    and must format it identically, so this exact wording also has a JS
    twin in the <script> block below; keep the two in sync if either
    changes).
    """
    state = status.get("state")
    if state == "ok":
        age = status.get("age_seconds") or 0
        mode = status.get("mode") or "?"
        paper = status.get("open_paper_positions")
        real = status.get("open_real_positions")
        label = "анализирует рынок" if status.get("activity") == "ai_cycle" else "активен"
        parts = [f"Runtime: {label} ({mode})"]
        if paper is not None or real is not None:
            parts.append(f"{paper if paper is not None else '?'} paper / {real if real is not None else '?'} real")
        parts.append(f"тик {int(age)}с назад")
        return " · ".join(parts)
    if state == "stale":
        age = status.get("age_seconds")
        age_str = f"{int(age)}с" if age is not None else "?"
        return f"Runtime: не отвечает ({age_str})"
    return "Runtime: нет данных"


def _runtime_heartbeat_badge(status: dict) -> str:
    state = status.get("state")
    color = {"ok": _STATUS["LONG"]["light"], "stale": _STATUS["SHORT"]["light"]}.get(state, "var(--text-muted)")
    text = html.escape(_runtime_heartbeat_text(status))
    return (
        f'<span id="runtime-heartbeat" class="runtime-heartbeat" data-state="{html.escape(state or "unknown")}">'
        f'<span class="freshness-dot" style="--status-color: {color}" aria-hidden="true"></span>'
        f'<span id="runtime-heartbeat-text">{text}</span>'
        f"</span>"
    )


def _freshness_badge(freshness: Freshness | None) -> str:
    """Age/expiry pill shown on a model card — 'Сигнал устарел' once the
    model's own `valid_for_minutes` has elapsed (see `signal_freshness.py`).
    A stale card is also excluded from the consensus vote (`consensus_engine.py`).
    """
    if freshness is None:
        return ""
    if freshness.is_stale:
        return '<div class="freshness-badge freshness-badge--stale">Сигнал устарел</div>'
    return f'<div class="freshness-badge">{html.escape(format_remaining(freshness.seconds_remaining))}</div>'


def _stat_tile(label: str, value: str) -> str:
    return (
        '<div class="stat-tile">'
        f'<div class="stat-label">{html.escape(label)}</div>'
        f'<div class="stat-value">{html.escape(value)}</div>'
        "</div>"
    )


def _verdict_line(plan: TradePlan) -> str:
    """One-line plain-language summary: 'Открывай на LONG, вход X, стоп Y, TP1 Z, уверенность P'.

    Values now come straight from the model's validated JSON plan — no more
    regex-scraping numbers out of free text. Each "label value" pair is
    wrapped in its own non-breaking span so the line wraps at the comma
    boundaries, never splitting a label from its value or a number.
    """
    status = _STATUS.get(plan.signal, _STATUS["WAIT"])
    verb = _signal_verb(plan.signal, plan.entry_status)
    if plan.signal == "WAIT":
        # WAIT still carries entry/stop_loss/take_profits values in the
        # schema (Pydantic requires them), but they're not a real trade —
        # showing them as numbers would look like a fabricated plan.
        entry = "—"
        stop_loss = "—"
        take_profit = "—"
    else:
        entry = (
            f"{plan.entry.from_:.2f}"
            if plan.entry.from_ == plan.entry.to
            else f"{plan.entry.from_:.2f}–{plan.entry.to:.2f}"
        )
        stop_loss = f"{plan.stop_loss:.2f}"
        take_profit = f"{plan.take_profits[0].price:.2f}" if plan.take_profits else "н/д"
    confidence = f"{plan.confidence}%"

    def pair(label: str, value: str) -> str:
        prefix = f"{html.escape(label)} " if label else ""
        return f'<span class="verdict-pair">{prefix}<strong>{html.escape(value)}</strong></span>'

    sentence = ", ".join(
        [
            pair("", verb),
            pair("вход", entry),
            pair("стоп", stop_loss),
            pair("TP1", take_profit),
            pair("уверенность модели", confidence),
        ]
    )
    tooltip = (
        f'<span class="confidence-hint" tabindex="0" title="{html.escape(_CONFIDENCE_TOOLTIP)}" '
        'aria-label="Пояснение про уверенность модели">&#9432;</span>'
    )
    return (
        f'<div class="verdict-line" style="--status-color: {status["light"]}">'
        f'<span class="verdict-icon" aria-hidden="true">{status["icon"]}</span> {sentence} {tooltip}'
        "</div>"
    )


def _list_block(title: str, items: list[str]) -> str:
    """Always-open list — used on the consensus/trade-plan cards, which are
    meant to be readable at a glance without an extra click.
    """
    if not items:
        return ""
    lis = "".join(f"<li>{html.escape(item)}</li>" for item in items)
    return f'<div class="plan-block"><h4>{html.escape(title)}</h4><ul>{lis}</ul></div>'


def _details_block(title: str, items: list[str]) -> str:
    """Collapsed-by-default toggle — used on individual model cards so three
    full walls of reasons/risks/invalidation text don't compete with the
    compact always-visible summary row (product spec section 12).
    """
    if not items:
        return ""
    lis = "".join(f"<li>{html.escape(item)}</li>" for item in items)
    return f'<details class="plan-details"><summary>{html.escape(title)}</summary><ul>{lis}</ul></details>'


def _validator_badge(validation: ValidationResult | None) -> str:
    if validation is None:
        return ""
    labels = {"valid": "Валидатор: OK", "warning": "Валидатор: предупреждение", "rejected": "Отклонено валидатором"}
    return f'<span class="validator-badge validator-badge--{validation.status}">{labels[validation.status]}</span>'


def _issues_html(issues: list[ValidationIssue]) -> str:
    if not issues:
        return ""
    items = "".join(f"<li>{html.escape(issue.message)}</li>" for issue in issues)
    return f'<ul class="validator-issues">{items}</ul>'


def _raw_response_details(content: str | None) -> str:
    if not content:
        return ""
    return (
        '<details class="raw-response"><summary>Показать сырой ответ модели</summary>'
        f"<pre>{html.escape(content)}</pre></details>"
    )


def _model_card(result: AIAnalysisResult, slot: int, mode: str) -> str:
    plan = result.trade_plan
    freshness = compute_freshness(result)
    is_stale = freshness is not None and freshness.is_stale
    freshness_html = _freshness_badge(freshness)

    if result.ok:
        # A usable, validator-approved plan (status "valid" or "warning").
        status = _STATUS.get(plan.signal, _STATUS["WAIT"])
        badge_html = (
            f'<span class="signal-badge" style="--status-color: {status["light"]}">'
            f'<span class="signal-icon">{status["icon"]}</span>'
            f'<span class="signal-text">{status["label"]}</span>'
            "</span>"
        )
        verdict_html = _verdict_line(plan)
        entry_status = _ENTRY_STATUS_LABELS.get(plan.entry_status, plan.entry_status)
        market_regime = _MARKET_REGIME_LABELS.get(plan.market_regime, plan.market_regime)
        # Collapsed-by-default toggles (spec section 12: "Причины / Риски /
        # Условия отмены / Полный анализ" — a compact card shouldn't open
        # with three full walls of text) — only the raw-JSON toggle stays
        # separate below since it's the "Полный анализ" button.
        body_html = (
            f'<div class="entry-status">{html.escape(entry_status)} &middot; '
            f"режим рынка: {html.escape(market_regime)}</div>"
            + _validator_badge(result.validation)
            + _issues_html(result.validation.issues if result.validation else [])
            + _details_block("Причины", plan.reasons)
            + _details_block("Риски", plan.risks)
            + _details_block("Условия отмены", plan.invalidation_conditions)
            + _raw_response_details(result.content)
        )
    elif plan is not None:
        # Parsed fine, but trade_validator rejected the plan — never shown as
        # a ready-to-trade signal, per the "не показывать невалидный план как
        # полноценный сигнал" rule.
        badge_html = (
            '<span class="signal-badge signal-badge--error">'
            '<span class="signal-icon">&#9940;</span><span class="signal-text">ОТКЛОНЕНО</span></span>'
        )
        verdict_html = ""
        issues = result.validation.issues if result.validation else []
        body_html = (
            '<p class="error-text">Сценарий отклонён валидатором:</p>'
            + _issues_html(issues)
            + _raw_response_details(result.content)
        )
    elif result.error is None:
        # Legacy AIAnalysisResult pickled before the JSON-schema migration —
        # no trade_plan field existed yet, so it defaults to None here too.
        badge_html = (
            '<span class="signal-badge signal-badge--neutral">'
            '<span class="signal-icon">&#8987;</span><span class="signal-text">СТАРЫЙ ФОРМАТ</span></span>'
        )
        verdict_html = ""
        body_html = (
            '<p class="legacy-text">Эта карточка из кеша ещё старой (текстовой) версии — '
            "нажмите &#8635;, чтобы получить структурированный план.</p>"
        )
    else:
        badge_html = (
            '<span class="signal-badge signal-badge--error">'
            '<span class="signal-icon">⚠</span>'
            '<span class="signal-text">ОШИБКА</span>'
            "</span>"
        )
        verdict_html = ""
        body_html = f'<p class="error-text">{html.escape(result.error)}</p>' + _raw_response_details(result.content)

    latency = f"{result.latency_seconds:.1f} с"
    updated_str = _fmt_updated_at(getattr(result, "created_at", None))

    refresh_btn_html = (
        '<button type="button" class="model-refresh-btn" '
        f'data-mode="{html.escape(mode)}" data-model="{html.escape(result.label)}" '
        'onclick="refreshModel(this)" '
        'title="Обновить только эту модель — остальные карточки не трогает">'
        '<span class="model-refresh-icon" aria-hidden="true">&#8635;</span>'
        "</button>"
    )

    card_class = "model-card model-card--stale" if is_stale else "model-card"

    return f"""
    <article class="{card_class}" style="--slot-index: {slot}">
      <header class="model-card-header">
        <span class="model-chip" aria-hidden="true"></span>
        <div class="model-header-text">
          <h3>{html.escape(result.label)}</h3>
          <div class="model-id">{html.escape(result.model)} &middot; {html.escape(latency)}</div>
          <div class="model-updated">Обновлено: {html.escape(updated_str)}</div>
          {freshness_html}
        </div>
        {badge_html}
        {refresh_btn_html}
      </header>
      <div class="model-refresh-status"></div>
      {verdict_html}
      <div class="model-card-body">
        {body_html}
      </div>
    </article>
    """


def _consensus_card(consensus: ConsensusResult) -> str:
    """"Итоговый вывод" — the answer to "what's happening and why" a reader
    should get in the first couple of seconds, computed entirely in code
    (`consensus_engine.py`) from the already-validated model plans — no
    extra AI call.
    """
    status = _STATUS.get(consensus.overall_signal, _STATUS["WAIT"])
    state_label = _CONSENSUS_STATE_LABELS.get(consensus.state, consensus.state)
    state_color = _CONSENSUS_STATE_COLOR.get(consensus.state, "var(--text-muted)")
    confidence_str = f"{consensus.avg_confidence:.0f}%" if consensus.avg_confidence is not None else "н/д"
    permission_html = _permission_badge(consensus.trade_permission, consensus.trade_permission_reason)

    return f"""
    <section class="consensus-card" style="--status-color: {status["light"]}">
      <div class="consensus-head">
        <span class="consensus-signal">
          <span class="signal-icon" aria-hidden="true">{status["icon"]}</span> {status["label"]}
        </span>
        <span class="consensus-state-badge" style="--state-color: {state_color}">{html.escape(state_label)}</span>
      </div>
      <div class="consensus-meta">
        Согласны с направлением: <strong>{consensus.agreeing_count} из {consensus.vote_count}</strong> валидных
        &middot; всего валидных ответов: <strong>{consensus.vote_count} из {consensus.total_models}</strong>
        &middot; средняя уверенность: <strong>{html.escape(confidence_str)}</strong>
      </div>
      {permission_html}
      {_list_block("Причины", consensus.reasons)}
      {_list_block("Риски", consensus.risks)}
    </section>
    """


def _trade_plan_card(consensus: ConsensusResult) -> str:
    """"Карточка торгового сценария" — entry/stop/targets synthesized (median)
    across the models agreeing with the consensus signal, or a WAIT block
    (spec section 14: WAIT is a full decision, not an empty result) when the
    consensus itself is WAIT.
    """
    permission_html = _permission_badge(consensus.trade_permission, consensus.trade_permission_reason)

    if consensus.overall_signal == "WAIT":
        conditions_html = _list_block("Условия для входа", consensus.wait_or_invalidation)
        empty_note = (
            ""
            if consensus.wait_or_invalidation
            else '<p class="wait-summary">Явных условий для входа модели не назвали — сверьтесь с '
            "полным анализом отдельных карточек.</p>"
        )
        return f"""
        <section class="trade-plan-card trade-plan-card--wait">
          <h2>Карточка сценария</h2>
          {permission_html}
          <p class="wait-summary">Нет преимущества для входа прямо сейчас.</p>
          {conditions_html}
          {empty_note}
        </section>
        """

    plan = consensus.plan
    if plan is None:
        return ""

    entry = (
        f"{plan.entry_from:.2f}" if plan.entry_from == plan.entry_to else f"{plan.entry_from:.2f}–{plan.entry_to:.2f}"
    )
    entry_status = _ENTRY_STATUS_LABELS.get(plan.entry_status, plan.entry_status)
    rr_str = f"{plan.risk_reward_tp1:.2f}" if plan.risk_reward_tp1 is not None else "н/д"

    figures = [
        ("Вход", entry),
        ("Stop loss", f"{plan.stop_loss:.2f}"),
    ]
    figures.extend((label, f"{price:.2f}") for label, price, _close_percent in plan.take_profits)
    figures.append(("R:R до TP1", rr_str))
    figures_html = "".join(
        f'<div class="plan-figure"><span class="plan-figure-label">{html.escape(label)}</span>'
        f'<span class="plan-figure-value">{html.escape(value)}</span></div>'
        for label, value in figures
    )

    price_zone_html = ""
    if consensus.trade_permission == "PRICE_OUTSIDE_ENTRY_ZONE":
        price_zone_html = (
            f'<p class="price-outside-zone-note">{html.escape(consensus.trade_permission_reason)} '
            "Не входить по рынку — ждите возврат в зону или обновите анализ.</p>"
        )

    freshness_html = ""
    if plan.formed_at is not None:
        now = time.time()
        expires_at = plan.formed_at + plan.valid_for_minutes * 60
        remaining = expires_at - now
        freshness_html = (
            '<div class="plan-freshness">'
            f"Сформирован {html.escape(format_age(now - plan.formed_at))} &middot; "
            f"{html.escape(format_remaining(remaining))}"
            "</div>"
        )

    return f"""
    <section class="trade-plan-card">
      <h2>Карточка сценария</h2>
      {permission_html}
      <div class="plan-source">Источник плана: {html.escape(plan.source_label)}</div>
      <div class="entry-status">{html.escape(entry_status)} &middot; горизонт ~{plan.time_horizon_minutes} мин</div>
      <div class="plan-figures">{figures_html}</div>
      {price_zone_html}
      {freshness_html}
      {_list_block("Причины входа", consensus.reasons)}
      {_list_block("Риски", consensus.risks)}
    </section>
    """


_POSITION_STATUS_LABELS = {
    "VALID": "Готово к открытию",
    "POSITION_TOO_SMALL": "Ниже минимального ордера",
    "LOW_RISK_REWARD": "R:R ниже минимума",
    "FEES_TOO_HIGH": "Комиссии съедают прибыль",
    "INVALID_SETTINGS": "Некорректные настройки риска",
    "INVALID_SCENARIO": "Некорректный сценарий",
    "MISSING_MARKET_RULES": "Правила инструмента недоступны",
    "RISK_LIMIT": "Ограничено риском",
    "MARGIN_LIMIT": "Ограничено маржой",
    "STALE_SIGNAL": "Сигнал устарел",
    "WAITING_TRIGGER": "Ждать условие входа",
    "PRICE_OUTSIDE_ENTRY_ZONE": "Цена вне зоны входа",
    "STOP_ALREADY_BREACHED": "Стоп уже пробит",
    "TP_ALREADY_REACHED": "TP1 уже достигнут",
    "WAIT": "WAIT",
    "INVALID_PLAN": "Нет валидного плана",
    "NOT_ALLOWED": "Не входить",
    "EXPIRED": "Сигнал устарел",
}

_MARGIN_MODE_LABELS_RU = {"ISOLATED": "Изолированная", "CROSS": "Кросс"}
_ORDER_TYPE_LABELS_RU = {"MARKET": "Рыночный", "LIMIT": "Лимитный", "TRIGGER": "Триггер"}
_ENTRY_PRICE_MODE_LABELS_RU = {
    "MIDPOINT": "середина диапазона",
    "CONSERVATIVE": "консервативный",
    "BEST_CASE": "лучший случай",
    "MANUAL": "вручную",
}
_BINGX_INPUT_MODE_LABELS_RU = {
    "MARGIN_USDT": "Стоимость / маржа",
    "NOTIONAL_USDT": "Номинал позиции",
    "COIN_QUANTITY": "Количество монеты",
}


def _fmt_decimal(value, digits: int = 2) -> str:
    if value is None:
        return "н/д"
    return f"{value:.{digits}f}"


def _position_status_severity(status: str) -> str:
    if status in ("VALID",):
        return "LONG"
    if status in ("WAITING_TRIGGER", "PRICE_OUTSIDE_ENTRY_ZONE", "RISK_LIMIT", "MARGIN_LIMIT", "MISSING_MARKET_RULES"):
        return "WAIT"
    return "SHORT"


def _stale_check_attrs(plan) -> str:
    """`data-*` attributes read by the client-side `checkStaleSignals()`
    ticker. Staleness has to be judged in the browser, not here: this HTML
    is built once (at refresh/settings-save time) and can sit open in a tab
    long past `valid_for_minutes` — the server has no way to know when a
    page is actually being looked at, so it can't decide "is this stale
    right now" for a viewer who opened the tab later.
    """
    if plan is None or plan.formed_at is None:
        return ""
    return f' data-formed-at="{plan.formed_at}" data-valid-minutes="{plan.valid_for_minutes}"'


def _position_card(service_result: PositionServiceResult | None) -> str:
    if service_result is None or not service_result.applicable or service_result.calculation is None:
        return ""
    calc = service_result.calculation

    status_label = _POSITION_STATUS_LABELS.get(service_result.display_status, service_result.display_status)
    status_color = _STATUS[_position_status_severity(service_result.display_status)]["light"]

    reference_html = ""
    if service_result.reference_only:
        reference_html = (
            '<div class="reference-only-note">Расчёт выполнен справочно — '
            f"{html.escape(service_result.display_message)} Не размещать ордер до выполнения условия входа.</div>"
        )

    limited_by_str = {"RISK": "по риску", "MARGIN": "по максимальной марже", None: "н/д"}.get(calc.limited_by, "н/д")
    entry_mode_str = _ENTRY_PRICE_MODE_LABELS_RU.get(calc.entry_price_mode.value, calc.entry_price_mode.value)
    rules_source_str = (
        "BingX API" if service_result.instrument_rules.source == "BINGX_API" else "локальные настройки (fallback)"
    )

    rows = [
        ("Источник сценария", html.escape(service_result.consensus.plan.source_label)),
        ("Баланс", f"{_fmt_decimal(calc.account_balance_usdt)} USDT"),
        ("Риск на сделку", f"{_fmt_decimal(calc.risk_percent)}%"),
        ("Риск-бюджет", f"{_fmt_decimal(calc.risk_budget_usdt)} USDT"),
        ("Зона входа", f"{_fmt_decimal(calc.entry_zone[0])}–{_fmt_decimal(calc.entry_zone[1])}"),
        ("Расчётная цена входа", _fmt_decimal(calc.entry_price)),
        ("Метод цены входа", entry_mode_str),
        ("Stop loss", _fmt_decimal(calc.stop_loss)),
        ("Расстояние до стопа", f"{_fmt_decimal(calc.stop_distance)} USDT"),
        ("Количество", _fmt_decimal(calc.position_size_coin_rounded, 4)),
        ("Номинал позиции", f"{_fmt_decimal(calc.position_notional_usdt)} USDT"),
        ("Требуемая маржа", f"{_fmt_decimal(calc.required_margin_usdt)} USDT"),
        ("Плечо", f"{calc.leverage}×"),
        ("Комиссия входа", f"{_fmt_decimal(calc.entry_fee_usdt, 4)} USDT"),
        ("Комиссия выхода по стопу", f"{_fmt_decimal(calc.stop_exit_fee_usdt, 4)} USDT"),
        ("Проскальзывание (оценка)", f"{_fmt_decimal(calc.slippage_estimate_usdt, 4)} USDT"),
        ("Ожидаемый убыток при стопе", f"{_fmt_decimal(calc.total_expected_loss_usdt)} USDT"),
        ("Фактический риск", f"{_fmt_decimal(calc.actual_risk_percent)}%"),
        ("Свободный баланс после открытия", f"{_fmt_decimal(calc.free_balance_after_open_usdt)} USDT"),
        ("Ограничение размера", limited_by_str),
        ("Источник правил инструмента", rules_source_str),
    ]
    rows_html = "".join(
        f'<div class="position-row"><span class="position-row-label">{html.escape(label)}</span>'
        f'<span class="position-row-value">{value}</span></div>'
        for label, value in rows
    )

    warnings_html = _list_block("Предупреждения", calc.warnings)
    issues_html = _list_block("Проблемы", calc.issues) if calc.issues else ""
    tp_table_html = _take_profit_table(service_result)
    stale_attrs_html = _stale_check_attrs(service_result.consensus.plan)

    return f"""
    <section class="position-card"{stale_attrs_html}>
      <h2>Расчёт позиции</h2>
      <span class="position-status-badge" style="--status-color: {status_color}">{html.escape(status_label)}</span>
      <div class="stale-signal-banner" style="display: none">Сигнал устарел — с момента формирования истекло
        заявленное окно действия. Обновите анализ, прежде чем действовать.</div>
      {reference_html}
      <div class="position-grid">{rows_html}</div>
      {warnings_html}
      {issues_html}
      {tp_table_html}
    </section>
    """


def _take_profit_table(service_result: PositionServiceResult) -> str:
    calc = service_result.calculation
    if calc is None or not calc.take_profit_results:
        return ""
    rows = []
    for tp in calc.take_profit_results:
        net_rr_str = _fmt_decimal(tp.net_rr) if tp.net_rr is not None else "н/д"
        primary_mark = " tp-row--primary" if tp.label == calc.primary_target_label else ""
        rows.append(
            f'<tr class="tp-row{primary_mark}">'
            f"<td>{html.escape(tp.label)}</td>"
            f"<td>{_fmt_decimal(tp.price)}</td>"
            f"<td>{_fmt_decimal(tp.close_percent)}%</td>"
            f"<td>{_fmt_decimal(tp.distance_percent)}%</td>"
            f"<td>{_fmt_decimal(tp.gross_profit_usdt, 4)}</td>"
            f"<td>{_fmt_decimal(tp.net_profit_usdt, 4)}</td>"
            f"<td>{_fmt_decimal(tp.gross_rr)}</td>"
            f"<td>{net_rr_str}</td>"
            "</tr>"
        )

    footer_html = ""
    blended = calc.blended
    if blended is not None and len(calc.take_profit_results) > 1:
        blended_net_rr_str = _fmt_decimal(blended.net_rr) if blended.net_rr is not None else "н/д"
        footer_html = f"""
        <tfoot>
          <tr class="tp-row tp-row--blended">
            <td colspan="2">Итог плана (все цели)</td>
            <td>{_fmt_decimal(blended.total_close_percent)}%</td>
            <td>&mdash;</td>
            <td>{_fmt_decimal(blended.gross_profit_usdt, 4)}</td>
            <td>{_fmt_decimal(blended.net_profit_usdt, 4)}</td>
            <td>{_fmt_decimal(blended.gross_rr)}</td>
            <td>{blended_net_rr_str}</td>
          </tr>
        </tfoot>
        """

    return f"""
    <div class="tp-table-wrap">
      <table class="tp-table">
        <thead>
          <tr><th>Цель</th><th>Цена</th><th>% закрытия</th><th>&Delta;%</th><th>Валовая, USDT</th>
          <th>Чистая, USDT</th><th>Gross R:R</th><th>Net R:R</th></tr>
        </thead>
        <tbody>{"".join(rows)}</tbody>
        {footer_html}
      </table>
    </div>
    """


def _bingx_card(service_result: PositionServiceResult | None) -> str:
    if service_result is None or not service_result.applicable or service_result.bingx_fields is None:
        return ""
    fields = service_result.bingx_fields

    status_label = _POSITION_STATUS_LABELS.get(service_result.display_status, service_result.display_status)
    status_color = _STATUS[_position_status_severity(service_result.display_status)]["light"]

    side_label = "Лонг" if fields.side == "LONG" else "Шорт"
    order_type_label = _ORDER_TYPE_LABELS_RU.get(fields.order_type, fields.order_type)
    margin_mode_label = _MARGIN_MODE_LABELS_RU.get(fields.margin_mode, fields.margin_mode)
    input_mode_label = _BINGX_INPUT_MODE_LABELS_RU.get(fields.selected_input_mode, fields.selected_input_mode)
    base_asset = service_result.instrument_rules.symbol.split("-")[0]

    if fields.selected_input_mode == "COIN_QUANTITY":
        selected_value_str = f"{_fmt_decimal(fields.selected_input_value, 4)} {base_asset}"
    else:
        selected_value_str = f"{_fmt_decimal(fields.selected_input_value)} USDT"

    if service_result.can_open:
        action_html = (
            '<div class="bingx-action bingx-action--allowed" data-original-text="'
            f'Финальное действие: нажать «Открыть {html.escape(side_label.lower())}»">'
            f"Финальное действие: нажать «Открыть {html.escape(side_label.lower())}»</div>"
        )
    else:
        action_html = (
            '<div class="bingx-action bingx-action--blocked">'
            f"{html.escape(service_result.display_message)} Не размещать ордер.</div>"
        )

    tp_count = len(service_result.consensus.plan.take_profits)
    tp_label = "TP1"
    if tp_count > 1:
        tp_label = f"TP1 (+{tp_count - 1} доп. цели — вручную)"

    rows = [
        ("Режим маржи", margin_mode_label),
        ("Плечо", f"{fields.leverage}×"),
        ("Направление", side_label.upper()),
        ("Тип ордера", order_type_label),
        ("Цена", _fmt_decimal(fields.price)),
        ("Количество", f"{_fmt_decimal(fields.coin_quantity, 4)} {base_asset}"),
        ("Номинал", f"{_fmt_decimal(fields.notional_usdt)} USDT"),
        ("Маржа", f"{_fmt_decimal(fields.margin_usdt)} USDT"),
        (tp_label, _fmt_decimal(fields.take_profit) if fields.take_profit is not None else "н/д"),
        ("Stop loss", _fmt_decimal(fields.stop_loss)),
        ("Статус размещения", html.escape(status_label)),
        ("Поле ввода в BingX", input_mode_label),
        ("Значение поля", selected_value_str),
    ]
    rows_html = "".join(
        f'<div class="bingx-field"><span class="bingx-field-label">{html.escape(label)}</span>'
        f'<span class="bingx-field-value">{value}</span></div>'
        for label, value in rows
    )

    tp_hint_html = ""
    if tp_count > 1:
        tp_hint_html = (
            '<p class="bingx-hint">В поле TP BingX уходит только первая цель (TP1) — '
            f"остальные {tp_count - 1} нужно выставить отдельными ручными ордерами после входа.</p>"
        )

    stale_attrs_html = _stale_check_attrs(service_result.consensus.plan)

    return f"""
    <section class="bingx-card"{stale_attrs_html}>
      <h2>Что заполнить в BingX</h2>
      <span class="position-status-badge" style="--status-color: {status_color}">{html.escape(status_label)}</span>
      <div class="stale-signal-banner" style="display: none">Сигнал устарел — с момента формирования истекло
        заявленное окно действия. Обновите анализ, прежде чем действовать.</div>
      <div class="bingx-fields">{rows_html}</div>
      <p class="bingx-hint">Проверьте, какой режим ввода выбран в форме ордера BingX — расчёт выполнен для режима
        «{html.escape(input_mode_label)}».</p>
      {tp_hint_html}
      {action_html}
    </section>
    """


def _risk_settings_button() -> str:
    return (
        '<button type="button" class="refresh-btn risk-settings-button" onclick="openRiskSettings()">'
        '<span aria-hidden="true">&#9881;</span> <span>Настройки риска</span></button>'
    )


def _risk_settings_modal(settings: RiskSettings) -> str:
    def field(label: str, input_id: str, value, step: str | None = None, suffix: str = "") -> str:
        step_attr = f' step="{step}"' if step else ""
        suffix_html = f'<span class="risk-field-suffix">{html.escape(suffix)}</span>' if suffix else ""
        return (
            f'<label class="risk-field"><span>{html.escape(label)}</span>'
            f'<span class="risk-field-input"><input type="number" id="{input_id}" '
            f'value="{html.escape(str(value))}"{step_attr}>{suffix_html}</span></label>'
        )

    def select(label: str, input_id: str, options: list[tuple[str, str]], current: str) -> str:
        opts_html = "".join(
            f'<option value="{html.escape(v)}"{" selected" if v == current else ""}>{html.escape(l)}</option>'
            for v, l in options
        )
        return f'<label class="risk-field"><span>{html.escape(label)}</span><select id="{input_id}">{opts_html}</select></label>'

    available_balance = (
        settings.available_balance_usdt if settings.available_balance_usdt is not None else settings.account_balance_usdt
    )

    fields_html = "".join(
        [
            field("Баланс счёта", "risk-account-balance", settings.account_balance_usdt, step="0.01", suffix="USDT"),
            field("Доступный баланс", "risk-available-balance", available_balance, step="0.01", suffix="USDT"),
            field("Риск на сделку", "risk-percent", settings.risk_percent, step="0.01", suffix="%"),
            field("Максимальная маржа", "risk-max-margin", settings.max_margin_percent, step="0.01", suffix="%"),
            field("Плечо", "risk-leverage", settings.leverage, step="1", suffix="×"),
            select(
                "Режим маржи",
                "risk-margin-mode",
                [("ISOLATED", "Изолированная"), ("CROSS", "Кросс")],
                settings.margin_mode.value,
            ),
            field("Комиссия maker", "risk-maker-fee", settings.maker_fee_percent, step="0.001", suffix="%"),
            field("Комиссия taker", "risk-taker-fee", settings.taker_fee_percent, step="0.001", suffix="%"),
            field("Проскальзывание", "risk-slippage", settings.slippage_percent, step="0.001", suffix="%"),
            field("Минимальный R:R", "risk-min-rr", settings.min_risk_reward, step="0.1"),
            select(
                "Цена входа",
                "risk-entry-mode",
                [
                    ("MIDPOINT", "Середина диапазона"),
                    ("CONSERVATIVE", "Консервативный"),
                    ("BEST_CASE", "Лучший случай"),
                    ("MANUAL", "Вручную"),
                ],
                settings.entry_price_mode.value,
            ),
            select(
                "Поле ввода в BingX",
                "risk-input-mode",
                [
                    ("MARGIN_USDT", "Стоимость / маржа"),
                    ("NOTIONAL_USDT", "Номинал"),
                    ("COIN_QUANTITY", "Количество монеты"),
                ],
                settings.bingx_order_input_mode.value,
            ),
        ]
    )

    return f"""
    <div id="risk-settings-overlay" class="risk-settings-overlay" style="display: none"
         data-quantity-step="{settings.quantity_step}" data-price-step="{settings.price_step}"
         data-min-notional="{settings.minimum_order_notional_usdt}" data-min-quantity="{settings.minimum_quantity}"
         onclick="if (event.target === this) closeRiskSettings()">
      <div class="risk-settings-modal" role="dialog" aria-modal="true" aria-label="Настройки риска">
        <h2>Настройки риска</h2>
        <p class="risk-settings-note">Все расчёты (размер позиции, маржа, комиссии) выполняются кодом
          детерминированно. AI никогда не участвует в этой математике и не вызывается при сохранении.</p>
        <div class="risk-fields-grid">{fields_html}</div>
        <div id="risk-settings-status" class="risk-settings-status"></div>
        <div class="risk-settings-actions">
          <button type="button" class="risk-btn risk-btn--primary" onclick="saveRiskSettings()">Сохранить</button>
          <button type="button" class="risk-btn" onclick="recalculateActiveScenario()">Пересчитать</button>
          <button type="button" class="risk-btn" onclick="resetRiskSettingsForm()">Сбросить</button>
          <button type="button" class="risk-btn" onclick="closeRiskSettings()">Отмена</button>
        </div>
      </div>
    </div>
    """


def _fmt_win_rate(win_rate: float | None) -> str:
    if win_rate is None:
        return "н/д"
    return f"{win_rate:.0f}%"


def _label_order() -> list[str]:
    return [m.label for m in config.AI_MODELS]


def _slot_for_label(label: str) -> int:
    order = _label_order()
    try:
        return order.index(label) % 3
    except ValueError:
        return 0


def _fmt_r(value: float | None, digits: int = 2) -> str:
    if value is None:
        return "н/д"
    sign = "+" if value > 0 else ""
    return f"{sign}{value:.{digits}f}R"


def _fmt_profit_factor(profit_factor: float | None, undefined: bool) -> str:
    if undefined:
        return "∞"  # no losing trades yet — ratio is not meaningful as a number
    if profit_factor is None:
        return "н/д"
    return f"{profit_factor:.2f}"


def _exit_reason_breakdown(counts: dict[str, int]) -> str:
    order = ["TP1", "TP2", "TP3", "SL", "TIMEOUT", "AMBIGUOUS"]
    parts = [f"{reason} {counts[reason]}" for reason in order if counts.get(reason)]
    return " &middot; ".join(parts)


def _win_rate_severity(win_rate: float | None, low_sample: bool) -> dict[str, str]:
    """Reuses the dashboard's existing LONG/WAIT/SHORT status colors as
    good/warning/serious (green/amber/red) instead of a new palette. Below
    `config.MIN_SAMPLE_FOR_STATS` evaluated predictions is treated as
    neutral — a rate computed from 1-2 samples is noise, not a signal worth
    coloring.
    """
    if win_rate is None or low_sample:
        return {"light": "var(--text-muted)", "dark": "var(--text-muted)"}
    if win_rate >= 60:
        return _STATUS["LONG"]
    if win_rate >= 40:
        return _STATUS["WAIT"]
    return _STATUS["SHORT"]


def _accuracy_row(label: str, s: dict) -> str:
    win_rate = s["win_rate"]
    n = s["evaluated"]
    severity = _win_rate_severity(win_rate, s["low_sample"])
    fill_pct = win_rate if win_rate is not None else 0

    if win_rate is None:
        rate_str = "н/д"
    elif s["low_sample"]:
        rate_str = f"{_fmt_win_rate(win_rate)} (мало данных)"
    else:
        rate_str = _fmt_win_rate(win_rate)

    # Spec section 19: never a bare percentage — always the sample size next to it.
    sample_note = f"{s['wins']} побед из {n}" if n else "ещё нет оценённых прогнозов"

    metrics_html = ""
    breakdown_html = ""
    if n:
        metrics_html = (
            '<div class="accuracy-row-metrics">'
            f"Expectancy: <strong>{_fmt_r(s['expectancy_r'])}</strong> &middot; "
            f"Profit factor: <strong>{html.escape(_fmt_profit_factor(s['profit_factor'], s['profit_factor_undefined']))}</strong> &middot; "
            f"Медиана: <strong>{_fmt_r(s['median_r'])}</strong> &middot; "
            f"Max DD: <strong>{_fmt_r(s['max_drawdown_r'])}</strong>"
            "</div>"
        )
        breakdown = _exit_reason_breakdown(s["exit_reason_counts"])
        if breakdown:
            breakdown_html = f'<div class="accuracy-row-breakdown">{breakdown}</div>'

    slot = _slot_for_label(label)

    return f"""
    <div class="accuracy-row">
      <div class="accuracy-row-head">
        <span class="accuracy-chip" style="background: var(--slot-{slot})" aria-hidden="true"></span>
        <span class="accuracy-row-label">{html.escape(label)}</span>
        <span class="accuracy-row-rate" style="--status-color: {severity["light"]}">{rate_str}</span>
      </div>
      <div class="accuracy-meter" style="--status-color: {severity["light"]}">
        <div class="accuracy-meter-fill" style="width: {fill_pct:.0f}%"></div>
      </div>
      <div class="accuracy-row-detail">{html.escape(sample_note)}</div>
      {metrics_html}
      {breakdown_html}
      <div class="accuracy-row-waiting">ждут оценки: {s["pending"]} &middot; сигналов WAIT: {s["skipped"]}</div>
    </div>
    """


def _accuracy_panel(mode: str) -> str:
    """Compact per-model accuracy summary for ONE trading mode, built from
    prediction_tracker's local log — scalping and swing are never blended
    (see the Stage 2 plan's note on why). Win-rate here means "share of
    evaluated predictions with a positive R-multiple", where R comes from
    walking the real BingX candle path (`outcome_simulator.py`), not just
    comparing price at a fixed deadline.
    """
    stats = prediction_tracker.stats_by_model_and_mode(mode)
    if not stats:
        return (
            '<section class="accuracy-panel">'
            f"<h2>Точность прогнозов &middot; {html.escape(MODE_LABELS[mode])}</h2>"
            '<p class="accuracy-empty">Пока нет ни одного прогноза для этого режима — '
            "накопится по мере обновлений.</p>"
            "</section>"
        )

    order = _label_order()
    labels = sorted(stats.keys(), key=lambda label: (order.index(label) if label in order else 99, label))
    rows_html = "".join(_accuracy_row(label, stats[label]) for label in labels)

    return f"""
    <section class="accuracy-panel">
      <h2>Точность прогнозов &middot; {html.escape(MODE_LABELS[mode])}</h2>
      <p class="accuracy-intro">
        Реальный путь цены по свечам после прогноза — куда цена пришла раньше: TP1/TP2/TP3, стоп
        или горизонт истёк без касания (TIMEOUT). Результат в R (единицах риска), с учётом
        комиссии/проскальзывания. Цвет и длина полосы = доля прогнозов с положительным R.
      </p>
      {rows_html}
      <p class="accuracy-note">
        WAIT не оценивается направленно, только считается отдельно. Свеча, где одновременно задеты
        и стоп, и цель, помечается AMBIGUOUS и консервативно засчитывается как стоп — порядок
        внутри свечи по OHLC не восстановить.
      </p>
    </section>
    """


def _paper_trading_row(label: str, s: paper_trading.BucketStats) -> str:
    win_rate = s.win_rate
    n = s.evaluated
    severity = _win_rate_severity(win_rate, s.low_sample)
    fill_pct = win_rate if win_rate is not None else 0

    if win_rate is None:
        rate_str = "н/д"
    elif s.low_sample:
        rate_str = f"{_fmt_win_rate(win_rate)} (мало данных)"
    else:
        rate_str = _fmt_win_rate(win_rate)

    still_open = s.total - n
    sample_note = f"{s.wins} побед из {n}" if n else "ещё нет закрытых сделок"
    if still_open:
        sample_note += f" · открыто/в процессе: {still_open}"

    metrics_html = ""
    breakdown_html = ""
    if n:
        metrics_html = (
            '<div class="accuracy-row-metrics">'
            f"Expectancy: <strong>{_fmt_r(s.expectancy_r)}</strong> &middot; "
            f"Profit factor: <strong>{html.escape(_fmt_profit_factor(s.profit_factor, s.profit_factor_undefined))}</strong> &middot; "
            f"Медиана: <strong>{_fmt_r(s.median_r)}</strong> &middot; "
            f"Max DD: <strong>{_fmt_r(s.max_drawdown_r)}</strong>"
            "</div>"
        )
        breakdown = _exit_reason_breakdown(s.exit_reason_counts)
        if breakdown:
            breakdown_html = f'<div class="accuracy-row-breakdown">{breakdown}</div>'

    slot = _slot_for_label(label)

    return f"""
    <div class="accuracy-row">
      <div class="accuracy-row-head">
        <span class="accuracy-chip" style="background: var(--slot-{slot})" aria-hidden="true"></span>
        <span class="accuracy-row-label">{html.escape(label)}</span>
        <span class="accuracy-row-rate" style="--status-color: {severity["light"]}">{rate_str}</span>
      </div>
      <div class="accuracy-meter" style="--status-color: {severity["light"]}">
        <div class="accuracy-meter-fill" style="width: {fill_pct:.0f}%"></div>
      </div>
      <div class="accuracy-row-detail">{html.escape(sample_note)}</div>
      {metrics_html}
      {breakdown_html}
    </div>
    """


def _paper_trading_panel(mode: str) -> str:
    """Real paper-engine performance for ONE mode, pulled live from
    journal_db (paper_orders/trade_outcomes) — the actual simulated fills,
    fees and slippage from `paper_trading.process_tick`, scoped to
    whichever AI model's plan was actually opened as a paper order.

    Deliberately separate from `_accuracy_panel` above, which grades every
    AI prediction (even ones nothing ever opened a position for) against a
    simulated candle-path outcome — different question, different number.
    Best-effort: journal.db may not exist yet (fresh deploy, runtime never
    ticked) or a read could race a concurrent writer — either way this
    degrades to the empty-state panel rather than breaking the page.
    """
    try:
        conn = journal_db.init_db()
        try:
            stats = paper_trading.compute_paper_trading_statistics(
                conn, mode=mode, window_start=0.0, window_end=time.time()
            )
        finally:
            conn.close()
    except Exception:
        stats = None

    if not stats or not stats.by_model:
        return (
            '<section class="accuracy-panel">'
            f"<h2>Paper trading &middot; {html.escape(MODE_LABELS[mode])}</h2>"
            '<p class="accuracy-empty">Пока нет ни одной открытой paper-сделки для этого режима — '
            "накопится по мере работы trading_runtime.py.</p>"
            "</section>"
        )

    order = _label_order()
    labels = sorted(stats.by_model.keys(), key=lambda label: (order.index(label) if label in order else 99, label))
    rows_html = "".join(_paper_trading_row(label, stats.by_model[label]) for label in labels)

    return f"""
    <section class="accuracy-panel">
      <h2>Paper trading &middot; {html.escape(MODE_LABELS[mode])}</h2>
      <p class="accuracy-intro">
        Реальные paper-сделки, открытые и сопровождаемые движком (реальный bid/ask, комиссии,
        проскальзывание, SL/TP/трейлинг) — не симуляция по свечам, а фактический учёт в journal.db.
      </p>
      {rows_html}
    </section>
    """


def _mode_toggle_and_grids(
    results_by_mode: dict[str, list[AIAnalysisResult]],
    default_mode: str,
    symbol: str,
    settings: RiskSettings,
    current_price: float | None = None,
) -> tuple[str, str, str, str, str]:
    """Build the segmented-control toggle, the (hidden/shown) per-mode card
    grids, the (hidden/shown) per-mode consensus + trade-plan + position/
    BingX summary, the (hidden/shown) per-mode AI-accuracy panel, and the
    (hidden/shown) per-mode real paper-trading panel — all five share the
    same `data-mode` show/hide pattern driven by `setDashboardMode()`.
    """
    modes = [m for m in ("scalping", "swing") if m in results_by_mode]
    if default_mode not in modes:
        default_mode = modes[0] if modes else "scalping"

    instrument_rules = None
    if current_price is not None:
        try:
            instrument_rules = position_service.resolve_instrument_rules(config.to_bingx_symbol(symbol), settings)
        except Exception:
            instrument_rules = None

    toggle_buttons = []
    grids = []
    summaries = []
    accuracy_panels = []
    paper_trading_panels = []
    for mode in modes:
        results = results_by_mode[mode]
        ok_count = sum(1 for r in results if r.ok)
        is_active = mode == default_mode
        toggle_buttons.append(
            f'<button type="button" class="mode-toggle-btn{" active" if is_active else ""}" '
            f'data-mode="{html.escape(mode)}" onclick="setDashboardMode(\'{mode}\')" '
            f'role="tab" aria-selected="{"true" if is_active else "false"}">'
            f'{html.escape(MODE_LABELS[mode])} '
            f'<span class="toggle-count">{ok_count}/{len(results)}</span>'
            "</button>"
        )
        cards_html = "".join(_model_card(r, i, mode) for i, r in enumerate(results))
        grids.append(
            f'<div class="cards-grid" data-mode="{html.escape(mode)}" '
            f'style="display: {"grid" if is_active else "none"}">{cards_html}</div>'
        )

        service_result = None
        if instrument_rules is not None and current_price is not None:
            service_result = position_service.calculate_active_position(
                results,
                mode,
                total_models=len(config.AI_MODELS),
                current_price=current_price,
                settings=settings,
                instrument_rules=instrument_rules,
            )
            consensus = service_result.consensus
        else:
            consensus = compute_consensus(
                results, mode, total_models=len(config.AI_MODELS), current_price=current_price
            )

        summary_html = (
            _consensus_card(consensus)
            + _trade_plan_card(consensus)
            + _position_card(service_result)
            + _bingx_card(service_result)
        )
        summaries.append(
            f'<div class="mode-summary" data-mode="{html.escape(mode)}" '
            f'style="display: {"block" if is_active else "none"}">{summary_html}</div>'
        )

        accuracy_panels.append(
            f'<div class="mode-accuracy" data-mode="{html.escape(mode)}" '
            f'style="display: {"block" if is_active else "none"}">{_accuracy_panel(mode)}</div>'
        )

        paper_trading_panels.append(
            f'<div class="mode-accuracy" data-mode="{html.escape(mode)}" '
            f'id="paper-trading-panel-{html.escape(mode)}" '
            f'style="display: {"block" if is_active else "none"}">{_paper_trading_panel(mode)}</div>'
        )

    toggle_html = f'<div class="mode-toggle" role="tablist" aria-label="Стиль анализа">{"".join(toggle_buttons)}</div>'
    return (
        toggle_html,
        "".join(grids),
        "".join(summaries),
        "".join(accuracy_panels),
        "".join(paper_trading_panels),
    )


def build_dashboard(
    snapshot: dict, results_by_mode: dict[str, list[AIAnalysisResult]], default_mode: str = "scalping"
) -> str:
    ts: datetime = snapshot["timestamp"]
    ts_str = ts.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    snapshot_age_seconds = time.time() - ts.timestamp()
    freshness_dot = _data_freshness_dot(snapshot_age_seconds)
    age_str = format_age(snapshot_age_seconds)

    stat_tiles = "".join(
        [
            _stat_tile("Текущая цена", _fmt(snapshot["current_price"])),
            _stat_tile("Изм. 15m", _fmt_pct(snapshot["change_15m"])),
            _stat_tile("Изм. 1h", _fmt_pct(snapshot["change_1h"])),
            _stat_tile("Изм. 24h", _fmt_pct(snapshot["change_24h"])),
            _stat_tile(
                "Funding rate",
                f"{snapshot['funding_rate'] * 100:.4f}%" if snapshot.get("funding_rate") is not None else "н/д",
            ),
        ]
    )

    risk_settings = risk_settings_store.load()
    toggle_html, grids_html, summary_html, accuracy_html, paper_trading_html = _mode_toggle_and_grids(
        results_by_mode, default_mode, snapshot["symbol"], risk_settings, snapshot.get("current_price")
    )
    risk_settings_modal_html = _risk_settings_modal(risk_settings)
    runtime_status = heartbeat_status(
        config.RUNTIME_HEARTBEAT_FILE,
        config.RUNTIME_HEARTBEAT_STALE_SECONDS,
        busy_max_age_seconds=config.RUNTIME_HEARTBEAT_BUSY_STALE_SECONDS,
    )
    runtime_heartbeat_html = _runtime_heartbeat_badge(runtime_status)

    return f"""<!doctype html>
<html lang="ru">
<head>
<meta charset="utf-8">
<title>{html.escape(snapshot["symbol"])} — AI Dashboard</title>
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<style>
  :root {{
    --surface-1: #fcfcfb;
    --page-plane: #f9f9f7;
    --text-primary: #0b0b0b;
    --text-secondary: #52514e;
    --text-muted: #898781;
    --border-hairline: rgba(11,11,11,0.10);
    --gridline: #e1e0d9;
    --slot-0: {_SLOT_COLORS_LIGHT[0]};
    --slot-1: {_SLOT_COLORS_LIGHT[1]};
    --slot-2: {_SLOT_COLORS_LIGHT[2]};
  }}
  @media (prefers-color-scheme: dark) {{
    :root {{
      --surface-1: #1a1a19;
      --page-plane: #0d0d0d;
      --text-primary: #ffffff;
      --text-secondary: #c3c2b7;
      --text-muted: #898781;
      --border-hairline: rgba(255,255,255,0.10);
      --gridline: #2c2c2a;
      --slot-0: {_SLOT_COLORS_DARK[0]};
      --slot-1: {_SLOT_COLORS_DARK[1]};
      --slot-2: {_SLOT_COLORS_DARK[2]};
    }}
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0;
    padding: 24px;
    background: var(--page-plane);
    color: var(--text-primary);
    font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
    line-height: 1.5;
  }}
  .wrap {{ max-width: 1200px; margin: 0 auto; }}
  header.page-header {{
    display: flex;
    flex-wrap: wrap;
    align-items: baseline;
    justify-content: space-between;
    gap: 12px;
    margin-bottom: 20px;
  }}
  header.page-header h1 {{
    margin: 0;
    font-size: 22px;
    font-weight: 700;
  }}
  .subtitle {{
    color: var(--text-secondary);
    font-size: 14px;
  }}
  .refresh-area {{
    display: flex;
    align-items: center;
    gap: 10px;
    flex: none;
  }}
  .refresh-btn {{
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 8px 14px;
    border-radius: 8px;
    border: 1px solid var(--border-hairline);
    background: var(--surface-1);
    color: var(--text-primary);
    font: inherit;
    font-size: 13px;
    font-weight: 600;
    cursor: pointer;
    flex: none;
  }}
  .refresh-btn:hover {{ background: color-mix(in srgb, var(--text-primary) 6%, var(--surface-1)); }}
  .refresh-btn:active {{ transform: translateY(1px); }}
  .refresh-btn:disabled {{ cursor: default; opacity: 0.7; }}
  .refresh-icon {{ display: inline-block; font-size: 15px; line-height: 1; }}
  .refresh-icon.loading {{ animation: refresh-spin 0.8s linear infinite; }}
  @keyframes refresh-spin {{
    from {{ transform: rotate(0deg); }}
    to {{ transform: rotate(360deg); }}
  }}
  .refresh-status {{
    font-size: 12px;
    color: var(--text-muted);
    max-width: 240px;
  }}
  .refresh-status.error {{ color: #d03b3b; }}
  .stat-row {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
    gap: 12px;
    margin-bottom: 24px;
  }}
  .stat-tile {{
    background: var(--surface-1);
    border: 1px solid var(--border-hairline);
    border-radius: 10px;
    padding: 12px 14px;
  }}
  .stat-label {{
    color: var(--text-muted);
    font-size: 12px;
    margin-bottom: 4px;
  }}
  .stat-value {{
    font-size: 20px;
    font-weight: 600;
    font-variant-numeric: tabular-nums;
  }}
  .cards-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
    gap: 16px;
  }}
  .model-card {{
    background: var(--surface-1);
    border: 1px solid var(--border-hairline);
    border-radius: 12px;
    overflow: hidden;
    display: flex;
    flex-direction: column;
  }}
  .model-card-header {{
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 14px 16px;
    border-bottom: 1px solid var(--gridline);
  }}
  .model-chip {{
    width: 10px;
    height: 10px;
    border-radius: 50%;
    flex: none;
    background: var(--slot-0);
  }}
  .model-card[style*="--slot-index: 1"] .model-chip {{ background: var(--slot-1); }}
  .model-card[style*="--slot-index: 2"] .model-chip {{ background: var(--slot-2); }}
  .model-header-text {{ flex: 1; min-width: 0; }}
  .model-header-text h3 {{
    margin: 0;
    font-size: 15px;
    font-weight: 700;
  }}
  .model-id {{
    color: var(--text-muted);
    font-size: 12px;
    margin-top: 2px;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }}
  .model-updated {{
    color: var(--text-muted);
    font-size: 11px;
    margin-top: 2px;
    font-variant-numeric: tabular-nums;
  }}
  .signal-badge {{
    display: inline-flex;
    align-items: center;
    gap: 5px;
    padding: 4px 10px;
    border-radius: 999px;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 0.02em;
    background: color-mix(in srgb, var(--status-color) 14%, transparent);
    border: 1px solid var(--status-color);
    color: var(--text-primary);
    flex: none;
  }}
  .signal-icon {{ color: var(--status-color); }}
  .signal-badge--error {{
    background: color-mix(in srgb, #d03b3b 14%, transparent);
    border: 1px solid #d03b3b;
  }}
  .signal-badge--error .signal-icon {{ color: #d03b3b; }}
  .model-refresh-btn {{
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 26px;
    height: 26px;
    border-radius: 6px;
    border: 1px solid var(--border-hairline);
    background: var(--surface-1);
    color: var(--text-secondary);
    font: inherit;
    cursor: pointer;
    flex: none;
  }}
  .model-refresh-btn:hover {{ background: color-mix(in srgb, var(--text-primary) 6%, var(--surface-1)); }}
  .model-refresh-btn:disabled {{ cursor: default; opacity: 0.6; }}
  .model-refresh-icon {{ display: inline-block; font-size: 13px; line-height: 1; }}
  .model-refresh-icon.loading {{ animation: refresh-spin 0.8s linear infinite; }}
  .model-refresh-status {{
    padding: 4px 16px;
    font-size: 11px;
    color: var(--text-muted);
    border-bottom: 1px solid var(--gridline);
  }}
  .model-refresh-status:empty {{ display: none; }}
  .model-refresh-status.error {{ color: #d03b3b; }}
  .verdict-line {{
    padding: 10px 16px;
    font-size: 13px;
    line-height: 1.6;
    color: var(--text-primary);
    border-bottom: 1px solid var(--gridline);
    background: color-mix(in srgb, var(--status-color) 10%, transparent);
  }}
  .verdict-line strong {{
    font-variant-numeric: tabular-nums;
    color: var(--text-primary);
  }}
  .verdict-pair {{
    white-space: nowrap;
  }}
  .verdict-icon {{
    color: var(--status-color);
    font-size: 13px;
  }}
  .confidence-hint {{
    display: inline-block;
    color: var(--text-muted);
    cursor: help;
    font-size: 12px;
  }}
  .model-card-body {{
    padding: 14px 16px;
    font-size: 14px;
    color: var(--text-primary);
    overflow-x: auto;
  }}
  .model-card-body h4 {{ margin: 14px 0 6px; font-size: 14px; }}
  .model-card-body h4:first-child {{ margin-top: 0; }}
  .model-card-body p {{ margin: 6px 0; }}
  .model-card-body ul {{ margin: 6px 0; padding-left: 20px; }}
  .model-card-body strong {{ color: var(--text-primary); }}
  .model-card-body hr {{ border: none; border-top: 1px solid var(--gridline); margin: 12px 0; }}
  .error-text {{ color: #d03b3b; }}
  .legacy-text {{ color: var(--text-muted); }}
  .entry-status {{
    font-size: 12px;
    color: var(--text-secondary);
    margin-bottom: 10px;
  }}
  .validator-badge {{
    display: inline-block;
    padding: 3px 8px;
    border-radius: 6px;
    font-size: 11px;
    font-weight: 700;
    margin-bottom: 8px;
  }}
  .validator-badge--valid {{
    color: #0ca30c;
    background: color-mix(in srgb, #0ca30c 14%, transparent);
  }}
  .validator-badge--warning {{
    color: #b5790a;
    background: color-mix(in srgb, #fab219 20%, transparent);
  }}
  .validator-badge--rejected {{
    color: #d03b3b;
    background: color-mix(in srgb, #d03b3b 14%, transparent);
  }}
  .validator-issues {{
    margin: 0 0 10px;
    padding-left: 20px;
    font-size: 13px;
    color: var(--text-secondary);
  }}
  .plan-block {{ margin-bottom: 10px; }}
  .plan-block h4 {{ margin: 0 0 4px; font-size: 12px; color: var(--text-muted); text-transform: uppercase; }}
  .plan-block ul {{ margin: 0; padding-left: 18px; }}
  .plan-details {{ margin-bottom: 6px; }}
  .plan-details summary {{
    cursor: pointer;
    font-size: 12px;
    font-weight: 600;
    color: var(--text-secondary);
    padding: 2px 0;
  }}
  .plan-details ul {{ margin: 4px 0 0; padding-left: 18px; font-size: 13px; }}
  .raw-response {{ margin-top: 10px; }}
  .raw-response summary {{
    cursor: pointer;
    font-size: 12px;
    color: var(--text-muted);
  }}
  .raw-response pre {{
    white-space: pre-wrap;
    word-break: break-word;
    font-size: 12px;
    background: var(--page-plane);
    border: 1px solid var(--gridline);
    border-radius: 8px;
    padding: 10px;
    margin-top: 6px;
    max-height: 320px;
    overflow-y: auto;
  }}
  .signal-badge--neutral {{
    background: color-mix(in srgb, var(--text-muted) 14%, transparent);
    border: 1px solid var(--text-muted);
  }}
  .signal-badge--neutral .signal-icon {{ color: var(--text-muted); }}
  .freshness-dot {{
    display: inline-block;
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: var(--status-color);
    margin-right: 2px;
  }}
  .runtime-heartbeat {{
    display: inline-flex;
    align-items: center;
    gap: 4px;
  }}
  .freshness-badge {{
    color: var(--text-muted);
    font-size: 11px;
    margin-top: 2px;
    font-variant-numeric: tabular-nums;
  }}
  .freshness-badge--stale {{
    color: #d03b3b;
    font-weight: 700;
  }}
  .model-card--stale {{ opacity: 0.6; }}
  .mode-summary {{ margin-bottom: 16px; }}
  .consensus-card {{
    background: var(--surface-1);
    border: 1px solid var(--border-hairline);
    border-left: 4px solid var(--status-color);
    border-radius: 12px;
    padding: 16px 18px;
    margin-bottom: 12px;
  }}
  .consensus-head {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 10px;
    margin-bottom: 8px;
  }}
  .consensus-signal {{
    font-size: 20px;
    font-weight: 800;
    color: var(--status-color);
    display: inline-flex;
    align-items: center;
    gap: 8px;
  }}
  .consensus-signal .signal-icon {{ color: var(--status-color); }}
  .consensus-state-badge {{
    display: inline-block;
    padding: 4px 10px;
    border-radius: 999px;
    font-size: 12px;
    font-weight: 700;
    color: var(--state-color);
    background: color-mix(in srgb, var(--state-color) 14%, transparent);
    border: 1px solid var(--state-color);
  }}
  .consensus-meta {{
    font-size: 13px;
    color: var(--text-secondary);
    margin-bottom: 4px;
  }}
  .consensus-meta strong {{ color: var(--text-primary); font-variant-numeric: tabular-nums; }}
  .trade-plan-card {{
    background: var(--surface-1);
    border: 1px solid var(--border-hairline);
    border-radius: 12px;
    padding: 16px 18px;
  }}
  .trade-plan-card h2 {{
    margin: 0 0 10px;
    font-size: 15px;
    font-weight: 700;
  }}
  .trade-plan-card--wait {{ border-style: dashed; }}
  .wait-summary {{
    margin: 0 0 10px;
    color: var(--text-secondary);
    font-size: 13px;
  }}
  .plan-figures {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(100px, 1fr));
    gap: 10px;
    margin-bottom: 10px;
  }}
  .plan-figure {{
    background: var(--page-plane);
    border: 1px solid var(--gridline);
    border-radius: 8px;
    padding: 8px 10px;
  }}
  .plan-figure-label {{
    display: block;
    color: var(--text-muted);
    font-size: 11px;
    margin-bottom: 2px;
  }}
  .plan-figure-value {{
    display: block;
    font-size: 15px;
    font-weight: 700;
    font-variant-numeric: tabular-nums;
    color: var(--text-primary);
  }}
  .plan-freshness {{
    font-size: 12px;
    color: var(--text-muted);
    margin-bottom: 10px;
  }}
  .trade-permission-badge {{
    display: flex;
    flex-wrap: wrap;
    align-items: baseline;
    gap: 8px;
    padding: 8px 12px;
    margin-bottom: 10px;
    border-radius: 8px;
    border: 1px solid var(--status-color);
    background: color-mix(in srgb, var(--status-color) 10%, transparent);
  }}
  .trade-permission-label {{
    font-size: 13px;
    font-weight: 800;
    color: var(--status-color);
    text-transform: uppercase;
    letter-spacing: 0.02em;
  }}
  .trade-permission-reason {{
    font-size: 12px;
    color: var(--text-secondary);
  }}
  .plan-source {{
    font-size: 12px;
    color: var(--text-muted);
    margin-bottom: 8px;
  }}
  .price-outside-zone-note {{
    margin: 0 0 10px;
    padding: 8px 10px;
    border-radius: 8px;
    background: color-mix(in srgb, #fab219 14%, transparent);
    border: 1px solid #fab219;
    color: var(--text-primary);
    font-size: 12px;
  }}
  .mode-toggle-row {{
    display: flex;
    justify-content: center;
    margin-bottom: 20px;
  }}
  .mode-toggle {{
    display: inline-flex;
    border: 1px solid var(--border-hairline);
    border-radius: 999px;
    padding: 3px;
    gap: 2px;
    background: var(--surface-1);
  }}
  .mode-toggle-btn {{
    border: none;
    background: transparent;
    color: var(--text-secondary);
    font: inherit;
    font-size: 13px;
    font-weight: 600;
    padding: 7px 16px;
    border-radius: 999px;
    cursor: pointer;
  }}
  .mode-toggle-btn.active {{
    background: var(--text-primary);
    color: var(--surface-1);
  }}
  .mode-toggle-btn .toggle-count {{
    opacity: 0.65;
    font-weight: 500;
    font-variant-numeric: tabular-nums;
  }}
  footer.page-footer {{
    margin-top: 24px;
    padding-top: 16px;
    border-top: 1px solid var(--gridline);
    color: var(--text-muted);
    font-size: 12px;
  }}
  .accuracy-panel {{
    margin-top: 24px;
    background: var(--surface-1);
    border: 1px solid var(--border-hairline);
    border-radius: 12px;
    padding: 16px 18px;
  }}
  .accuracy-panel h2 {{
    margin: 0 0 12px;
    font-size: 15px;
    font-weight: 700;
  }}
  .accuracy-intro {{
    margin: 0 0 14px;
    color: var(--text-secondary);
    font-size: 12px;
    line-height: 1.5;
  }}
  .accuracy-row {{
    padding: 10px 0;
    border-bottom: 1px solid var(--gridline);
  }}
  .accuracy-row:last-of-type {{
    border-bottom: none;
    padding-bottom: 0;
  }}
  .accuracy-row-head {{
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 6px;
  }}
  .accuracy-chip {{
    width: 10px;
    height: 10px;
    border-radius: 50%;
    flex: none;
  }}
  .accuracy-row-label {{
    flex: 1;
    min-width: 0;
    font-size: 13px;
    font-weight: 600;
    color: var(--text-primary);
  }}
  .accuracy-row-rate {{
    font-size: 13px;
    font-weight: 700;
    font-variant-numeric: tabular-nums;
    color: var(--status-color);
    flex: none;
  }}
  .accuracy-meter {{
    height: 8px;
    border-radius: 999px;
    background: color-mix(in srgb, var(--status-color) 16%, var(--page-plane));
    overflow: hidden;
  }}
  .accuracy-meter-fill {{
    height: 100%;
    min-width: 3px;
    border-radius: 999px;
    background: var(--status-color);
  }}
  .accuracy-row-detail {{
    margin-top: 6px;
    color: var(--text-muted);
    font-size: 11px;
  }}
  .accuracy-row-metrics {{
    margin-top: 4px;
    color: var(--text-secondary);
    font-size: 11px;
  }}
  .accuracy-row-metrics strong {{
    color: var(--text-primary);
    font-variant-numeric: tabular-nums;
  }}
  .accuracy-row-breakdown {{
    margin-top: 4px;
    color: var(--text-muted);
    font-size: 11px;
    font-variant-numeric: tabular-nums;
  }}
  .accuracy-row-waiting {{
    margin-top: 4px;
    color: var(--text-muted);
    font-size: 11px;
  }}
  .accuracy-empty {{
    margin: 0;
    color: var(--text-muted);
    font-size: 13px;
  }}
  .accuracy-note {{
    margin: 10px 0 0;
    color: var(--text-muted);
    font-size: 11px;
    line-height: 1.5;
  }}
  .risk-settings-overlay {{
    position: fixed;
    inset: 0;
    background: rgba(0, 0, 0, 0.5);
    display: flex;
    align-items: flex-start;
    justify-content: center;
    padding: 40px 16px;
    overflow-y: auto;
    z-index: 100;
  }}
  .risk-settings-modal {{
    background: var(--surface-1);
    border-radius: 14px;
    border: 1px solid var(--border-hairline);
    padding: 20px 22px;
    width: 100%;
    max-width: 640px;
  }}
  .risk-settings-modal h2 {{
    margin: 0 0 8px;
    font-size: 16px;
    font-weight: 700;
  }}
  .risk-settings-note {{
    margin: 0 0 14px;
    color: var(--text-muted);
    font-size: 12px;
    line-height: 1.5;
  }}
  .risk-fields-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 12px;
    margin-bottom: 14px;
  }}
  .risk-field {{
    display: flex;
    flex-direction: column;
    gap: 4px;
    font-size: 12px;
    color: var(--text-secondary);
  }}
  .risk-field input,
  .risk-field select {{
    font: inherit;
    font-size: 13px;
    padding: 7px 9px;
    border-radius: 7px;
    border: 1px solid var(--border-hairline);
    background: var(--page-plane);
    color: var(--text-primary);
  }}
  .risk-field-input {{
    display: flex;
    align-items: center;
    gap: 6px;
  }}
  .risk-field-input input {{ flex: 1; min-width: 0; }}
  .risk-field-suffix {{
    color: var(--text-muted);
    font-size: 12px;
    white-space: nowrap;
  }}
  .risk-settings-status {{
    min-height: 16px;
    margin-bottom: 10px;
    font-size: 12px;
    color: var(--text-muted);
  }}
  .risk-settings-status.error {{ color: #d03b3b; }}
  .risk-settings-actions {{
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
  }}
  .risk-btn {{
    padding: 8px 14px;
    border-radius: 8px;
    border: 1px solid var(--border-hairline);
    background: var(--page-plane);
    color: var(--text-primary);
    font: inherit;
    font-size: 13px;
    font-weight: 600;
    cursor: pointer;
  }}
  .risk-btn--primary {{
    background: var(--text-primary);
    color: var(--surface-1);
    border-color: var(--text-primary);
  }}
  .position-card, .bingx-card {{
    background: var(--surface-1);
    border: 1px solid var(--border-hairline);
    border-radius: 12px;
    padding: 16px 18px;
    margin-bottom: 12px;
  }}
  .position-card h2, .bingx-card h2 {{
    margin: 0 0 10px;
    font-size: 15px;
    font-weight: 700;
    display: inline-block;
    margin-right: 10px;
  }}
  .position-status-badge {{
    display: inline-block;
    padding: 3px 10px;
    border-radius: 999px;
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.02em;
    color: var(--status-color);
    background: color-mix(in srgb, var(--status-color) 14%, transparent);
    border: 1px solid var(--status-color);
    vertical-align: middle;
  }}
  .reference-only-note {{
    margin: 10px 0;
    padding: 8px 10px;
    border-radius: 8px;
    background: color-mix(in srgb, #fab219 14%, transparent);
    border: 1px solid #fab219;
    color: var(--text-primary);
    font-size: 12px;
  }}
  .position-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 8px 14px;
    margin: 12px 0;
  }}
  .position-row {{
    display: flex;
    justify-content: space-between;
    gap: 10px;
    padding: 6px 0;
    border-bottom: 1px solid var(--gridline);
    font-size: 12px;
  }}
  .position-row-label {{ color: var(--text-muted); }}
  .position-row-value {{
    color: var(--text-primary);
    font-weight: 600;
    font-variant-numeric: tabular-nums;
    text-align: right;
  }}
  .tp-table-wrap {{
    overflow-x: auto;
    margin-top: 10px;
  }}
  .tp-table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 12px;
    font-variant-numeric: tabular-nums;
  }}
  .tp-table th, .tp-table td {{
    padding: 7px 9px;
    text-align: right;
    border-bottom: 1px solid var(--gridline);
    white-space: nowrap;
  }}
  .tp-table th:first-child, .tp-table td:first-child {{ text-align: left; }}
  .tp-table th {{
    color: var(--text-muted);
    font-weight: 600;
  }}
  .tp-row--primary td {{ color: var(--text-primary); font-weight: 700; }}
  .tp-table tfoot td {{
    border-top: 2px solid var(--gridline);
    border-bottom: none;
    background: var(--page-plane);
    color: var(--text-primary);
    font-weight: 700;
  }}
  .bingx-fields {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 8px 14px;
    margin: 12px 0;
  }}
  .bingx-field {{
    display: flex;
    justify-content: space-between;
    gap: 10px;
    padding: 6px 0;
    border-bottom: 1px solid var(--gridline);
    font-size: 12px;
  }}
  .bingx-field-label {{ color: var(--text-muted); }}
  .bingx-field-value {{
    color: var(--text-primary);
    font-weight: 700;
    font-variant-numeric: tabular-nums;
    text-align: right;
  }}
  .bingx-extra {{
    margin: 8px 0;
    color: var(--text-secondary);
    font-size: 12px;
  }}
  .bingx-hint {{
    margin: 8px 0;
    color: var(--text-muted);
    font-size: 11px;
    line-height: 1.5;
  }}
  .bingx-action {{
    margin-top: 10px;
    padding: 9px 12px;
    border-radius: 8px;
    font-size: 13px;
    font-weight: 700;
  }}
  .bingx-action--allowed {{
    background: color-mix(in srgb, {_STATUS["LONG"]["light"]} 16%, transparent);
    border: 1px solid {_STATUS["LONG"]["light"]};
    color: var(--text-primary);
  }}
  .bingx-action--blocked {{
    background: color-mix(in srgb, {_STATUS["SHORT"]["light"]} 12%, transparent);
    border: 1px solid {_STATUS["SHORT"]["light"]};
    color: var(--text-primary);
  }}
  .stale-signal-banner {{
    margin: 10px 0;
    padding: 8px 10px;
    border-radius: 8px;
    background: color-mix(in srgb, {_STATUS["SHORT"]["light"]} 14%, transparent);
    border: 1px solid {_STATUS["SHORT"]["light"]};
    color: var(--text-primary);
    font-size: 12px;
    font-weight: 600;
  }}
  .position-card.js-stale, .bingx-card.js-stale {{ opacity: 0.6; }}
  .position-card.js-stale .position-status-badge,
  .bingx-card.js-stale .position-status-badge {{
    --status-color: {_STATUS["SHORT"]["light"]} !important;
  }}
</style>
</head>
<body>
<div class="wrap">
  <header class="page-header">
    <div>
      <h1>{html.escape(snapshot["symbol"])} &middot; {html.escape(snapshot["exchange"])}</h1>
      <div class="subtitle">{freshness_dot} {html.escape(ts_str)} &middot; данные обновлены {html.escape(age_str)}</div>
      <div class="subtitle">{runtime_heartbeat_html}</div>
    </div>
    <div class="refresh-area">
      <span id="refresh-status" class="refresh-status"></span>
      {_risk_settings_button()}
      <button id="refresh-btn" class="refresh-btn" type="button" onclick="refreshDashboard(this)"
              title="Заново запрашивает данные BingX и AI для активной вкладки (нужен запущенный python main.py --serve / python server.py).">
        <span class="refresh-icon" aria-hidden="true">&#8635;</span>
        <span class="refresh-label">Обновить</span>
      </button>
    </div>
  </header>

  {risk_settings_modal_html}

  <div class="mode-toggle-row">
    {toggle_html}
  </div>

  <div class="stat-row">
    {stat_tiles}
  </div>

  {summary_html}

  {grids_html}

  {accuracy_html}

  {paper_trading_html}

  <footer class="page-footer">
    Переключатель наверху меняет стиль анализа (скальпинг / длинная свинг-сделка). Кнопка «Обновить» заново
    запрашивает BingX и AI только для активной сейчас вкладки — вторая вкладка не трогается и останется с прежними
    данными, пока вы не переключитесь на неё и не обновите её отдельно. Обновление работает только если дашборд
    открыт через запущенный сервер (<code>python main.py --serve</code> или <code>python server.py</code>) —
    открытый напрямую файл может только показывать то, что было посчитано на момент генерации.
    Каждая модель отвечает строгим JSON-планом (сигнал, вход, стоп, цели) — числа в сводке не вытянуты
    регуляркой из текста, это структурированные поля ответа, прошедшие программную проверку
    (<code>trade_validator.py</code>). Отклонённый валидатором план никогда не показывается как готовый
    сигнал к действию. «Уверенность модели» — субъективная оценка самой модели, а не статистическая
    вероятность успеха. Это не финансовая рекомендация.
  </footer>
</div>
<script>
  var dashboardRefreshing = false;

  function setAllRefreshDisabled(disabled) {{
    var mainBtn = document.getElementById('refresh-btn');
    if (mainBtn) mainBtn.disabled = disabled;
    document.querySelectorAll('.model-refresh-btn').forEach(function (b) {{ b.disabled = disabled; }});
  }}

  function refreshDashboard(btn) {{
    if (dashboardRefreshing) return;
    var activeToggle = document.querySelector('.mode-toggle-btn.active');
    var mode = activeToggle ? activeToggle.dataset.mode : 'scalping';
    var icon = btn.querySelector('.refresh-icon');
    var label = btn.querySelector('.refresh-label');
    var status = document.getElementById('refresh-status');

    dashboardRefreshing = true;
    setAllRefreshDisabled(true);
    icon.classList.add('loading');
    status.textContent = '';
    status.classList.remove('error');

    // Some models take up to ~90s to respond — show elapsed time so a slow
    // (but working) request never looks indistinguishable from a stuck one.
    var startedAt = Date.now();
    var tick = function () {{
      var secs = Math.floor((Date.now() - startedAt) / 1000);
      label.textContent = 'Обновляю... (' + secs + 'с, может занять до ~90с)';
    }};
    tick();
    var ticker = setInterval(tick, 1000);

    fetch('/api/refresh?mode=' + encodeURIComponent(mode), {{ cache: 'no-store' }})
      .then(function (resp) {{
        return resp.json().catch(function () {{ return {{}}; }}).then(function (data) {{
          return {{ ok: resp.ok, data: data }};
        }});
      }})
      .then(function (result) {{
        if (result.ok && result.data && result.data.queued) {{
          // TRADING_MODE's own runtime cycle will produce this, not us --
          // the periodic /api/status poll below reloads the page once it
          // lands, no separate AI call was made here.
          status.textContent = 'Анализ этого режима ведёт runtime по расписанию — запросил внеочередной цикл, ' +
            'страница обновится сама, когда он завершится.';
          return;
        }}
        if (result.ok && result.data && result.data.ok) {{
          location.reload();
          return;
        }}
        var msg = (result.data && result.data.error) ? result.data.error : 'Не удалось обновить.';
        status.textContent = msg;
        status.classList.add('error');
      }})
      .catch(function () {{
        status.textContent = 'Сервер не отвечает — запущен ли python main.py --serve / python server.py?';
        status.classList.add('error');
      }})
      .finally(function () {{
        clearInterval(ticker);
        dashboardRefreshing = false;
        setAllRefreshDisabled(false);
        icon.classList.remove('loading');
        label.textContent = 'Обновить';
      }});
  }}

  function refreshModel(btn) {{
    if (dashboardRefreshing) return;
    var mode = btn.dataset.mode;
    var model = btn.dataset.model;
    var icon = btn.querySelector('.model-refresh-icon');
    var status = btn.closest('.model-card-header').nextElementSibling;

    dashboardRefreshing = true;
    setAllRefreshDisabled(true);
    icon.classList.add('loading');
    status.textContent = '';
    status.classList.remove('error');

    var startedAt = Date.now();
    var ticker = setInterval(function () {{
      var secs = Math.floor((Date.now() - startedAt) / 1000);
      status.textContent = 'Обновляю... (' + secs + 'с)';
    }}, 1000);
    status.textContent = 'Обновляю...';

    fetch('/api/refresh?mode=' + encodeURIComponent(mode) + '&model=' + encodeURIComponent(model), {{ cache: 'no-store' }})
      .then(function (resp) {{
        return resp.json().catch(function () {{ return {{}}; }}).then(function (data) {{
          return {{ ok: resp.ok, data: data }};
        }});
      }})
      .then(function (result) {{
        if (result.ok && result.data && result.data.ok) {{
          location.reload();
          return;
        }}
        var msg = (result.data && result.data.error) ? result.data.error : 'Не удалось обновить.';
        status.textContent = msg;
        status.classList.add('error');
      }})
      .catch(function () {{
        status.textContent = 'Сервер не отвечает.';
        status.classList.add('error');
      }})
      .finally(function () {{
        clearInterval(ticker);
        dashboardRefreshing = false;
        setAllRefreshDisabled(false);
        icon.classList.remove('loading');
      }});
  }}

  function setDashboardMode(mode) {{
    document.querySelectorAll('.mode-toggle-btn').forEach(function (btn) {{
      var active = btn.dataset.mode === mode;
      btn.classList.toggle('active', active);
      btn.setAttribute('aria-selected', active ? 'true' : 'false');
    }});
    document.querySelectorAll('.cards-grid').forEach(function (grid) {{
      grid.style.display = grid.dataset.mode === mode ? 'grid' : 'none';
    }});
    document.querySelectorAll('.mode-summary').forEach(function (summary) {{
      summary.style.display = summary.dataset.mode === mode ? 'block' : 'none';
    }});
    document.querySelectorAll('.mode-accuracy').forEach(function (panel) {{
      panel.style.display = panel.dataset.mode === mode ? 'block' : 'none';
    }});
  }}

  var riskSettingsBusy = false;

  function openRiskSettings() {{
    document.getElementById('risk-settings-overlay').style.display = 'flex';
  }}

  function closeRiskSettings() {{
    document.getElementById('risk-settings-overlay').style.display = 'none';
  }}

  function resetRiskSettingsForm() {{
    if (riskSettingsBusy) return;
    location.reload();
  }}

  function collectRiskSettingsForm() {{
    var overlay = document.getElementById('risk-settings-overlay');
    return {{
      account_balance_usdt: document.getElementById('risk-account-balance').value,
      available_balance_usdt: document.getElementById('risk-available-balance').value,
      risk_percent: document.getElementById('risk-percent').value,
      leverage: parseInt(document.getElementById('risk-leverage').value, 10),
      margin_mode: document.getElementById('risk-margin-mode').value,
      max_margin_percent: document.getElementById('risk-max-margin').value,
      maker_fee_percent: document.getElementById('risk-maker-fee').value,
      taker_fee_percent: document.getElementById('risk-taker-fee').value,
      slippage_percent: document.getElementById('risk-slippage').value,
      min_risk_reward: document.getElementById('risk-min-rr').value,
      entry_price_mode: document.getElementById('risk-entry-mode').value,
      bingx_order_input_mode: document.getElementById('risk-input-mode').value,
      quantity_step: overlay.dataset.quantityStep,
      price_step: overlay.dataset.priceStep,
      minimum_order_notional_usdt: overlay.dataset.minNotional,
      minimum_quantity: overlay.dataset.minQuantity
    }};
  }}

  function saveRiskSettings() {{
    if (riskSettingsBusy) return;
    riskSettingsBusy = true;
    var status = document.getElementById('risk-settings-status');
    status.textContent = 'Сохраняю...';
    status.classList.remove('error');
    var buttons = document.querySelectorAll('.risk-settings-actions .risk-btn');
    buttons.forEach(function (b) {{ b.disabled = true; }});

    fetch('/api/risk-settings', {{
      method: 'PUT',
      headers: {{ 'Content-Type': 'application/json' }},
      body: JSON.stringify(collectRiskSettingsForm()),
      cache: 'no-store'
    }})
      .then(function (resp) {{
        return resp.json().catch(function () {{ return {{}}; }}).then(function (data) {{
          return {{ ok: resp.ok, data: data }};
        }});
      }})
      .then(function (result) {{
        if (result.ok && result.data && result.data.ok) {{
          location.reload();
          return;
        }}
        var errors = (result.data && result.data.errors) ? result.data.errors.join(' ')
          : ((result.data && result.data.error) || 'Не удалось сохранить настройки.');
        status.textContent = errors;
        status.classList.add('error');
      }})
      .catch(function () {{
        status.textContent = 'Сервер не отвечает.';
        status.classList.add('error');
      }})
      .finally(function () {{
        riskSettingsBusy = false;
        buttons.forEach(function (b) {{ b.disabled = false; }});
      }});
  }}

  function recalculateActiveScenario() {{
    if (riskSettingsBusy) return;
    riskSettingsBusy = true;
    var status = document.getElementById('risk-settings-status');
    status.textContent = 'Пересчитываю...';
    status.classList.remove('error');
    var buttons = document.querySelectorAll('.risk-settings-actions .risk-btn');
    buttons.forEach(function (b) {{ b.disabled = true; }});

    var activeToggle = document.querySelector('.mode-toggle-btn.active');
    var mode = activeToggle ? activeToggle.dataset.mode : 'scalping';

    fetch('/api/recalculate-active-scenario', {{
      method: 'POST',
      headers: {{ 'Content-Type': 'application/json' }},
      body: JSON.stringify({{ mode: mode }}),
      cache: 'no-store'
    }})
      .then(function (resp) {{
        return resp.json().catch(function () {{ return {{}}; }}).then(function (data) {{
          return {{ ok: resp.ok, data: data }};
        }});
      }})
      .then(function (result) {{
        if (result.ok && result.data && result.data.ok) {{
          location.reload();
          return;
        }}
        status.textContent = (result.data && result.data.error) || 'Не удалось пересчитать.';
        status.classList.add('error');
      }})
      .catch(function () {{
        status.textContent = 'Сервер не отвечает.';
        status.classList.add('error');
      }})
      .finally(function () {{
        riskSettingsBusy = false;
        buttons.forEach(function (b) {{ b.disabled = false; }});
      }});
  }}

  function checkStaleSignals() {{
    var nowSec = Date.now() / 1000;
    document.querySelectorAll('.position-card[data-formed-at], .bingx-card[data-formed-at]').forEach(function (card) {{
      var formedAt = parseFloat(card.dataset.formedAt);
      var validMinutes = parseFloat(card.dataset.validMinutes);
      if (!isFinite(formedAt) || !isFinite(validMinutes)) return;
      var stale = nowSec >= formedAt + validMinutes * 60;
      card.classList.toggle('js-stale', stale);
      var banner = card.querySelector('.stale-signal-banner');
      if (banner) banner.style.display = stale ? 'block' : 'none';

      var action = card.querySelector('.bingx-action--allowed');
      if (action) {{
        if (stale) {{
          action.classList.remove('bingx-action--allowed');
          action.classList.add('bingx-action--blocked');
          action.textContent = 'Сигнал устарел — не открывать.';
        }}
      }} else {{
        var blocked = card.querySelector('.bingx-action--blocked[data-original-text]');
        if (blocked && !stale) {{
          blocked.classList.remove('bingx-action--blocked');
          blocked.classList.add('bingx-action--allowed');
          blocked.textContent = blocked.dataset.originalText;
        }}
      }}
    }});
  }}
  checkStaleSignals();
  setInterval(checkStaleSignals, 30000);

  var RUNTIME_HEARTBEAT_COLORS = {{ ok: '#0ca30c', stale: '#d03b3b' }};
  var TRADING_MODE = {config.TRADING_MODE!r};
  var tradingModeBaselineUpdatedAt = null;

  function formatRuntimeHeartbeat(status) {{
    if (!status || status.state === 'unknown') return 'Runtime: нет данных';
    if (status.state === 'stale') {{
      var staleAge = (status.age_seconds != null) ? Math.floor(status.age_seconds) + 'с' : '?';
      return 'Runtime: не отвечает (' + staleAge + ')';
    }}
    var label = (status.activity === 'ai_cycle') ? 'анализирует рынок' : 'активен';
    var parts = ['Runtime: ' + label + ' (' + (status.mode || '?') + ')'];
    if (status.open_paper_positions != null || status.open_real_positions != null) {{
      var paper = (status.open_paper_positions != null) ? status.open_paper_positions : '?';
      var real = (status.open_real_positions != null) ? status.open_real_positions : '?';
      parts.push(paper + ' paper / ' + real + ' real');
    }}
    parts.push('тик ' + Math.floor(status.age_seconds || 0) + 'с назад');
    return parts.join(' · ');
  }}

  function pollRuntimeHeartbeat() {{
    fetch('/api/status', {{ cache: 'no-store' }})
      .then(function (resp) {{ return resp.json(); }})
      .then(function (data) {{
        var badge = document.getElementById('runtime-heartbeat');
        var text = document.getElementById('runtime-heartbeat-text');
        if (badge && text) {{
          var status = data && data.runtime;
          text.textContent = formatRuntimeHeartbeat(status);
          var dot = badge.querySelector('.freshness-dot');
          var color = (status && RUNTIME_HEARTBEAT_COLORS[status.state]) || 'var(--text-muted)';
          if (dot) dot.style.setProperty('--status-color', color);
          badge.dataset.state = (status && status.state) || 'unknown';
        }}

        // The runtime's own scheduled AI cycle (not this page) produced
        // TRADING_MODE's results -- once its updated_at moves past what
        // was true when this page loaded, reload to show them, same as a
        // manual "Обновить" already does on success.
        var modeInfo = data && data.modes && data.modes[TRADING_MODE];
        if (modeInfo && modeInfo.updated_at != null) {{
          if (tradingModeBaselineUpdatedAt === null) {{
            tradingModeBaselineUpdatedAt = modeInfo.updated_at;
          }} else if (modeInfo.updated_at > tradingModeBaselineUpdatedAt && !dashboardRefreshing) {{
            location.reload();
          }}
        }}
      }})
      .catch(function () {{ /* dashboard server itself unreachable -- leave last-known state shown */ }});
  }}
  setInterval(pollRuntimeHeartbeat, 5000);

  function pollPaperTradingStats() {{
    fetch('/api/paper-trading-stats', {{ cache: 'no-store' }})
      .then(function (resp) {{ return resp.json(); }})
      .then(function (data) {{
        if (!data || !data.ok || !data.panels) return;
        ['scalping', 'swing'].forEach(function (mode) {{
          var el = document.getElementById('paper-trading-panel-' + mode);
          if (el && data.panels[mode] != null) el.innerHTML = data.panels[mode];
        }});
      }})
      .catch(function () {{ /* dashboard server itself unreachable -- leave last-known content shown */ }});
  }}
  pollPaperTradingStats();
  setInterval(pollPaperTradingStats, 60000);
</script>
</body>
</html>
"""
