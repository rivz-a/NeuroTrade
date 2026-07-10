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
import re
from datetime import datetime, timezone

import config
import prediction_tracker
from ai_client import AIAnalysisResult
from report_builder import MODE_LABELS
from verdict import detect_signal, extract_verdict

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


def _markdown_lite_to_html(text: str) -> str:
    """Very small, safe subset of markdown -> HTML. Input must already be escaped."""
    lines = text.split("\n")
    out: list[str] = []
    in_list = False
    for raw_line in lines:
        line = raw_line.strip()
        if line.startswith("### "):
            if in_list:
                out.append("</ul>")
                in_list = False
            out.append(f"<h4>{line[4:]}</h4>")
        elif line.startswith("## "):
            if in_list:
                out.append("</ul>")
                in_list = False
            out.append(f"<h4>{line[3:]}</h4>")
        elif line.startswith("- ") or line.startswith("* "):
            if not in_list:
                out.append("<ul>")
                in_list = True
            out.append(f"<li>{line[2:]}</li>")
        elif line == "":
            if in_list:
                out.append("</ul>")
                in_list = False
        elif line in ("---", "***", "___"):
            if in_list:
                out.append("</ul>")
                in_list = False
            out.append("<hr>")
        else:
            if in_list:
                out.append("</ul>")
                in_list = False
            out.append(f"<p>{line}</p>")
    if in_list:
        out.append("</ul>")
    joined = "\n".join(out)
    joined = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", joined)
    return joined


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


def _stat_tile(label: str, value: str) -> str:
    return (
        '<div class="stat-tile">'
        f'<div class="stat-label">{html.escape(label)}</div>'
        f'<div class="stat-value">{html.escape(value)}</div>'
        "</div>"
    )


def _verdict_line(verdict: dict[str, str | None], signal: str) -> str:
    """One-line plain-language summary: 'Открывай на LONG, вход X, стоп Y, TP1 Z, вероятность P'.

    Each "label value" pair is wrapped in its own non-breaking span so the
    line wraps at the comma boundaries (natural text flow), never splitting
    a label from its value or a number in the middle.
    """
    status = _STATUS.get(signal, _STATUS["WAIT"])
    verb = _SIGNAL_VERB.get(signal, _SIGNAL_VERB["WAIT"])
    entry = verdict.get("entry") or "н/д"
    stop_loss = verdict.get("stop_loss") or "н/д"
    take_profit = verdict.get("take_profit") or "н/д"
    probability = verdict.get("probability") or "н/д"

    def pair(label: str, value: str) -> str:
        prefix = f"{html.escape(label)} " if label else ""
        return f'<span class="verdict-pair">{prefix}<strong>{html.escape(value)}</strong></span>'

    sentence = ", ".join(
        [
            pair("", verb),
            pair("вход", entry),
            pair("стоп", stop_loss),
            pair("TP1", take_profit),
            pair("вероятность", probability),
        ]
    )
    return (
        f'<div class="verdict-line" style="--status-color: {status["light"]}; '
        f'--status-color-dark: {status["dark"]}">'
        f'<span class="verdict-icon" aria-hidden="true">{status["icon"]}</span> {sentence}'
        "</div>"
    )


def _model_card(result: AIAnalysisResult, slot: int, mode: str) -> str:
    signal = detect_signal(result.content) if result.ok else None
    status = _STATUS.get(signal) if signal else None

    if result.ok:
        badge_html = ""
        if status:
            badge_html = (
                f'<span class="signal-badge" style="--status-color: {status["light"]}; '
                f'--status-color-dark: {status["dark"]}">'
                f'<span class="signal-icon">{status["icon"]}</span>'
                f'<span class="signal-text">{status["label"]}</span>'
                "</span>"
            )
        verdict_html = _verdict_line(extract_verdict(result.content), signal)
        body_html = _markdown_lite_to_html(html.escape(result.content or ""))
    else:
        badge_html = (
            '<span class="signal-badge signal-badge--error">'
            '<span class="signal-icon">⚠</span>'
            '<span class="signal-text">ОШИБКА</span>'
            "</span>"
        )
        verdict_html = ""
        body_html = f'<p class="error-text">{html.escape(result.error or "Неизвестная ошибка")}</p>'

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

    return f"""
    <article class="model-card" style="--slot-index: {slot}">
      <header class="model-card-header">
        <span class="model-chip" aria-hidden="true"></span>
        <div class="model-header-text">
          <h3>{html.escape(result.label)}</h3>
          <div class="model-id">{html.escape(result.model)} &middot; {html.escape(latency)}</div>
          <div class="model-updated">Обновлено: {html.escape(updated_str)}</div>
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


_MIN_SCORED_FOR_RATING = 3


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


def _win_rate_severity(win_rate: float | None, scored: int) -> dict[str, str]:
    """Reuses the dashboard's existing LONG/WAIT/SHORT status colors as
    good/warning/serious (green/amber/red) instead of a new palette. Fewer
    than `_MIN_SCORED_FOR_RATING` scored predictions is treated as neutral —
    a rate computed from 1-2 samples is noise, not a signal worth coloring.
    """
    if win_rate is None or scored < _MIN_SCORED_FOR_RATING:
        return {"light": "var(--text-muted)", "dark": "var(--text-muted)"}
    if win_rate >= 60:
        return _STATUS["LONG"]
    if win_rate >= 40:
        return _STATUS["WAIT"]
    return _STATUS["SHORT"]


def _accuracy_row(label: str, s: dict) -> str:
    scored = s["scored"]
    win_rate = s["win_rate"]
    severity = _win_rate_severity(win_rate, scored)
    fill_pct = win_rate if win_rate is not None else 0

    if win_rate is None:
        rate_str = "н/д"
    elif scored < _MIN_SCORED_FOR_RATING:
        rate_str = f"{_fmt_win_rate(win_rate)} (мало данных)"
    else:
        rate_str = _fmt_win_rate(win_rate)

    detail = (
        f"побед {s['win']} &middot; поражений {s['loss']} &middot; без изменений {s['flat']} "
        f"&middot; ждут оценки {s['pending']} &middot; сигналов WAIT {s['skipped']}"
    )

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
      <div class="accuracy-row-detail">{detail}</div>
    </div>
    """


def _accuracy_panel() -> str:
    """Compact per-model accuracy summary built from prediction_tracker's
    local log — answers "which model is actually right more often", not just
    which one sounds more confident on any single card. One meter row per
    model: bar length + color = win-rate (green ≥60%, amber 40-59%, red <40%,
    grey = too few scored predictions to trust yet); exact counts are spelled
    out underneath so nothing is hidden behind color alone.
    """
    stats = prediction_tracker.stats_by_model()
    if not stats:
        return (
            '<section class="accuracy-panel">'
            "<h2>Точность прогнозов по моделям</h2>"
            '<p class="accuracy-empty">Пока нет ни одного завершённого прогноза для оценки — '
            "накопится по мере обновлений (после того как пройдёт горизонт оценки: "
            "~30 мин для скальпинга, ~8 ч для свинга).</p>"
            "</section>"
        )

    order = _label_order()
    labels = sorted(stats.keys(), key=lambda label: (order.index(label) if label in order else 99, label))
    rows_html = "".join(_accuracy_row(label, stats[label]) for label in labels)

    return f"""
    <section class="accuracy-panel">
      <h2>Точность прогнозов по моделям</h2>
      <p class="accuracy-intro">
        Как часто модель угадывает направление (LONG/SHORT) к моменту, когда проходит срок
        прогноза (~30 мин скальпинг / ~8 ч свинг). Цвет и длина полосы = win-rate; серый —
        оценённых прогнозов пока слишком мало, чтобы доверять проценту.
      </p>
      {rows_html}
      <p class="accuracy-note">
        WAIT не оценивается направленно (это не ставка на сторону вверх/вниз) — только считается
        отдельно. Оценка не учитывает, сработал бы фактически stop loss или take profit раньше —
        сравнивается только цена «сейчас» против цены в момент прогноза.
      </p>
    </section>
    """


def _mode_toggle_and_grids(
    results_by_mode: dict[str, list[AIAnalysisResult]], default_mode: str
) -> tuple[str, str]:
    """Build the segmented-control toggle and the (hidden/shown) per-mode card grids."""
    modes = [m for m in ("scalping", "swing") if m in results_by_mode]
    if default_mode not in modes:
        default_mode = modes[0] if modes else "scalping"

    toggle_buttons = []
    grids = []
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

    toggle_html = f'<div class="mode-toggle" role="tablist" aria-label="Стиль анализа">{"".join(toggle_buttons)}</div>'
    return toggle_html, "".join(grids)


def build_dashboard(
    snapshot: dict, results_by_mode: dict[str, list[AIAnalysisResult]], default_mode: str = "scalping"
) -> str:
    ts: datetime = snapshot["timestamp"]
    ts_str = ts.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

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

    toggle_html, grids_html = _mode_toggle_and_grids(results_by_mode, default_mode)
    accuracy_html = _accuracy_panel()

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
</style>
</head>
<body>
<div class="wrap">
  <header class="page-header">
    <div>
      <h1>{html.escape(snapshot["symbol"])} &middot; {html.escape(snapshot["exchange"])}</h1>
      <div class="subtitle">{html.escape(ts_str)}</div>
    </div>
    <div class="refresh-area">
      <span id="refresh-status" class="refresh-status"></span>
      <button id="refresh-btn" class="refresh-btn" type="button" onclick="refreshDashboard(this)"
              title="Заново запрашивает данные BingX и AI для активной вкладки (нужен запущенный python main.py --serve / python server.py).">
        <span class="refresh-icon" aria-hidden="true">&#8635;</span>
        <span class="refresh-label">Обновить</span>
      </button>
    </div>
  </header>

  <div class="mode-toggle-row">
    {toggle_html}
  </div>

  <div class="stat-row">
    {stat_tiles}
  </div>

  {grids_html}

  {accuracy_html}

  <footer class="page-footer">
    Переключатель наверху меняет стиль анализа (скальпинг / длинная свинг-сделка). Кнопка «Обновить» заново
    запрашивает BingX и AI только для активной сейчас вкладки — вторая вкладка не трогается и останется с прежними
    данными, пока вы не переключитесь на неё и не обновите её отдельно. Обновление работает только если дашборд
    открыт через запущенный сервер (<code>python main.py --serve</code> или <code>python server.py</code>) —
    открытый напрямую файл может только показывать то, что было посчитано на момент генерации.
    Сигнал и сводка (вероятность / вход / stop loss / TP1) определены автоматическим разбором текста ответа модели —
    «н/д» означает, что модель не указала это значение в ожидаемом формате явно, сверяйтесь с полным текстом карточки.
    Это не финансовая рекомендация.
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
  }}
</script>
</body>
</html>
"""
