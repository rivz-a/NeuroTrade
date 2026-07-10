"""Market Snapshot Copier — CLI entry point.

Fetches public ETHUSDT market data from BingX, computes indicators, and
builds a plain-text report. By default the report is then sent in parallel
to up to 3 AI APIs (configured via .env — AI_*/AI2_*/AI3_*) which each
return a scalping analysis; use --no-ai to skip that step and only produce
the report for manual copying (e.g. into ChatGPT). Results are printed to
the terminal, saved as text, and rendered into an HTML comparison dashboard.

Two analysis styles are supported: "scalping" (tight 1m/5m stops, minutes-long
holds) and "swing" (wider stops/targets anchored to 15m/1h structure,
hours-to-a-day holds). Every AI run queries every configured model for BOTH
styles at once, and the HTML dashboard has a toggle to switch between them
without re-running the script. --mode / TRADING_MODE in .env only picks which
style is used for the plain-text report (market_snapshot.txt) and which tab
is active by default in the dashboard.

This tool never places orders and never modifies any exchange account
state — it only reads public BingX market data, optionally sends the
resulting text report to the AI APIs you configure, and writes local files.
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone

if sys.platform == "win32":
    # Legacy Windows consoles default to a codepage (e.g. cp1251) that can't
    # encode box-drawing characters or Cyrillic text together, which crashes
    # rich's output. Force UTF-8 so the report renders correctly in-terminal.
    os.system("chcp 65001 >NUL 2>&1")
    for _stream_name in ("stdout", "stderr"):
        _stream = getattr(sys, _stream_name)
        if hasattr(_stream, "reconfigure"):
            try:
                _stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass

from rich.console import Console

import ai_client
import bingx_client
import config
import prediction_tracker
from dashboard_builder import build_dashboard
from market_data import fetch_snapshot
from report_builder import MODE_LABELS, build_report
from trade_validator import build_validation_context

console = Console()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Собрать публичные рыночные данные BingX и сформировать текстовый отчёт для ChatGPT."
    )
    parser.add_argument(
        "--symbol",
        default=config.SYMBOL,
        help="Торговая пара, например ETHUSDT (по умолчанию из .env/SYMBOL или ETHUSDT).",
    )
    parser.add_argument(
        "--output",
        default=str(config.REPORT_FILE),
        help="Путь к файлу отчёта (по умолчанию market_snapshot.txt).",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Не сохранять отчёт в файл, только вывести в терминал.",
    )
    parser.add_argument(
        "--no-ai",
        action="store_true",
        help="Не отправлять данные в AI API — только собрать и вывести/сохранить текстовый отчёт.",
    )
    parser.add_argument(
        "--ai-output",
        default=str(config.AI_ANALYSIS_FILE),
        help="Путь к файлу с текстовыми ответами всех AI-моделей (по умолчанию ai_analysis.txt).",
    )
    parser.add_argument(
        "--dashboard-output",
        default=str(config.AI_DASHBOARD_FILE),
        help="Путь к HTML-дашборду со сравнением моделей (по умолчанию dashboard.html).",
    )
    parser.add_argument(
        "--no-dashboard",
        action="store_true",
        help="Не генерировать HTML-дашборд, только текстовый вывод AI.",
    )
    parser.add_argument(
        "--mode",
        choices=sorted(config.VALID_TRADING_MODES),
        default=config.TRADING_MODE,
        help=(
            "Стиль анализа: scalping — узкие 1m/5m-стопы, минуты (по умолчанию); "
            "swing — стопы/цели по структуре 15m/1h, часы-сутки."
        ),
    )
    parser.add_argument(
        "--serve",
        action="store_true",
        help=(
            "После сборки отчёта запустить локальный сервер (127.0.0.1) для дашборда — "
            "кнопка «Обновить» в браузере будет реально дёргать BingX/AI, а не просто "
            "перечитывать файл. Требует включённого AI (не сочетается с --no-ai)."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        snapshot = fetch_snapshot(args.symbol, log=lambda m: console.print(f"[cyan]{m}[/cyan]"))
    except ValueError as exc:  # symbol parsing error
        console.print(f"[bold red]Неверный символ: {exc}[/bold red]")
        return 1
    except bingx_client.InvalidSymbolError as exc:
        console.print(f"[bold red]Неверный символ: {exc}[/bold red]")
        return 1
    except bingx_client.RateLimitError as exc:
        console.print(f"[bold red]Превышен лимит запросов BingX: {exc}[/bold red]")
        return 1
    except bingx_client.NetworkError as exc:
        console.print(f"[bold red]Проблема с интернет-соединением: {exc}[/bold red]")
        return 1
    except bingx_client.NoDataError as exc:
        console.print(f"[bold red]BingX не вернул данные: {exc}[/bold red]")
        return 1
    except bingx_client.APIError as exc:
        console.print(f"[bold red]BingX API недоступен: {exc}[/bold red]")
        return 1

    report_text = build_report({**snapshot, "mode_key": args.mode})

    console.rule("[bold green]Market Snapshot")
    console.print(report_text)
    console.rule()

    if not args.no_save:
        try:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(report_text)
            console.print(f"[bold green]Отчёт сохранён в {args.output}[/bold green]")
        except OSError as exc:
            console.print(f"[bold red]Не удалось сохранить отчёт в файл: {exc}[/bold red]")
            return 1

    if args.no_ai:
        if args.serve:
            console.print(
                "[yellow]--serve игнорируется вместе с --no-ai — серверу нечего обновлять "
                "без AI.[/yellow]"
            )
        return 0

    configured_models = [m for m in config.AI_MODELS if m.api_key]
    if not configured_models:
        console.print(
            "[bold red]Ни для одной модели не задан API-ключ в .env (AI_API_KEY / AI2_API_KEY / "
            "AI3_API_KEY). Укажите хотя бы один ключ или запустите с флагом --no-ai.[/bold red]"
        )
        return 1

    console.print(
        f"[cyan]Отправляю данные в {len(configured_models)} AI-модел"
        f"{'ь' if len(configured_models) == 1 else 'и'} сразу для обоих стилей "
        f"(скальпинг + свинг) — переключение будет прямо в дашборде...[/cyan]"
    )

    # Build a report per mode (indicators are identical, only the "Режим:" /
    # task text at the end differs) and query every model for both in one
    # parallel batch, so this doesn't take 2x as long as a single mode.
    report_texts = {
        mode: report_text if mode == args.mode else build_report({**snapshot, "mode_key": mode})
        for mode in ("scalping", "swing")
    }
    validation_contexts = {mode: build_validation_context(snapshot, mode) for mode in ("scalping", "swing")}

    try:
        results_by_mode = ai_client.analyze_modes(
            report_texts, config.AI_MODELS, config.AI_REQUEST_TIMEOUT, validation_contexts
        )
    except ai_client.AIConfigError as exc:
        console.print(f"[bold red]{exc}[/bold red]")
        return 1

    bingx_symbol = config.to_bingx_symbol(snapshot["symbol"])
    for mode, results in results_by_mode.items():
        prediction_tracker.record_results(mode, results, snapshot["current_price"], bingx_symbol)

    text_sections = []
    any_ok = False
    for mode in ("scalping", "swing"):
        console.rule(f"[bold yellow]{MODE_LABELS[mode].upper()}")
        for result in results_by_mode[mode]:
            console.rule(f"[bold magenta]{result.label} ({result.model})")
            if result.ok:
                any_ok = True
                console.print(result.content)
                text_sections.append(
                    f"=== [{MODE_LABELS[mode]}] {result.label} ({result.model}) ===\n{result.content}"
                )
            else:
                console.print(f"[bold red]Ошибка: {result.error}[/bold red]")
                text_sections.append(
                    f"=== [{MODE_LABELS[mode]}] {result.label} ({result.model}) ===\nОШИБКА: {result.error}"
                )
    console.rule()

    try:
        with open(args.ai_output, "w", encoding="utf-8") as f:
            f.write("\n\n".join(text_sections))
        console.print(f"[bold green]Ответы AI сохранены в {args.ai_output}[/bold green]")
    except OSError as exc:
        console.print(f"[bold red]Не удалось сохранить ответы AI в файл: {exc}[/bold red]")
        return 1

    if not args.no_dashboard:
        try:
            dashboard_html = build_dashboard(snapshot, results_by_mode, args.mode)
            with open(args.dashboard_output, "w", encoding="utf-8") as f:
                f.write(dashboard_html)
            console.print(f"[bold green]HTML-дашборд сохранён в {args.dashboard_output}[/bold green]")
        except OSError as exc:
            console.print(f"[bold red]Не удалось сохранить HTML-дашборд: {exc}[/bold red]")
            return 1

    if args.serve:
        if args.no_dashboard:
            console.print("[yellow]--serve игнорируется вместе с --no-dashboard.[/yellow]")
        else:
            import server as dashboard_server

            dashboard_server.run_server(
                symbol=args.symbol,
                initial_snapshot=snapshot,
                initial_results_by_mode=results_by_mode,
                initial_mode=args.mode,
                dashboard_output=args.dashboard_output,
                console=console,
            )

    return 0 if any_ok else 1


if __name__ == "__main__":
    sys.exit(main())
