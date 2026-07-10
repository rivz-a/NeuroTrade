"""Local live-dashboard server for Market Snapshot Copier.

Serves the generated dashboard.html and a same-origin refresh endpoint
(`/api/refresh?mode=scalping|swing`) that re-fetches BingX data and
re-queries the AI models for JUST that one mode — not both — so the
"Обновить" button on the dashboard shows real, current results instead of
just reloading a static file, and refreshing one tab never touches the
other tab's (possibly still-fresh) analysis.

Binds to 127.0.0.1 only. This can trigger real (paid) AI API calls, so it
is intentionally not meant to be reachable from other machines — do not
change DASHBOARD_HOST to 0.0.0.0 without adding auth/a firewall in front
of it yourself.

Run standalone:  python server.py
Or via main.py:  python main.py --serve
"""

from __future__ import annotations

import json
import pickle
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import ai_client
import bingx_client
import config
import prediction_tracker
from ai_client import AIAnalysisResult
from dashboard_builder import build_dashboard
from market_data import fetch_snapshot
from report_builder import MODE_LABELS, build_report
from trade_validator import build_validation_context

# Minimum gap between two refreshes of the *same* mode. This is not about
# rate-limiting a human clicking a button — it's a guard against exactly the
# failure mode a naive "is it done yet?" poller can cause: GET /api/refresh
# has a side effect (it starts a real, paid AI call) whenever no refresh is
# already in flight, so hammering it as a status check can silently burn
# through API quota. Use GET /api/status to poll instead — it never starts
# anything.
REFRESH_COOLDOWN_SECONDS = 20.0


class _State:
    def __init__(self, symbol: str, dashboard_output: str) -> None:
        self.symbol = symbol
        self.dashboard_output = dashboard_output
        self.lock = threading.Lock()
        self.snapshot: dict | None = None
        self.results_by_mode: dict[str, list[AIAnalysisResult]] = {}
        self.active_mode = config.TRADING_MODE
        self.refreshing_mode: str | None = None
        self.last_html = ""
        self.last_refresh_finished_at: dict[str, float] = {}
        self.last_updated_at: dict[str, float] = {}


def _refresh_mode(state: _State, mode: str) -> None:
    """Re-fetch BingX data and re-query all models for a single mode.

    Only `state.results_by_mode[mode]` is replaced — the other mode's last
    results are left exactly as they were, per the "only refresh the active
    tab" requirement.
    """
    snapshot = fetch_snapshot(state.symbol, log=print)
    report_text = build_report({**snapshot, "mode_key": mode})
    ctx = build_validation_context(snapshot, mode)
    results = ai_client.analyze_with_all(report_text, config.AI_MODELS, config.AI_REQUEST_TIMEOUT, ctx, mode)
    bingx_symbol = config.to_bingx_symbol(snapshot["symbol"])
    prediction_tracker.record_results(mode, results, snapshot["current_price"], bingx_symbol)

    with state.lock:
        state.snapshot = snapshot
        state.results_by_mode[mode] = results
        state.active_mode = mode
        state.last_updated_at[mode] = time.time()
        html = build_dashboard(snapshot, state.results_by_mode, mode)
        state.last_html = html
        try:
            with open(state.dashboard_output, "w", encoding="utf-8") as f:
                f.write(html)
        except OSError as exc:
            print(f"Не удалось сохранить {state.dashboard_output}: {exc}")
        _save_cache(state)


def _refresh_single(state: _State, mode: str, label: str) -> None:
    """Re-fetch BingX data and re-query exactly ONE model for one mode.

    Only the matching entry in `state.results_by_mode[mode]` is replaced —
    the other models' last results (and the other mode entirely) are left
    untouched. Backs the per-card "Обновить" button so testing/tuning one
    model never re-bills every configured model at once.
    """
    cfg = next((m for m in config.AI_MODELS if m.label == label), None)
    if cfg is None:
        raise ValueError(f"Неизвестная модель: {label}")

    snapshot = fetch_snapshot(state.symbol, log=print)
    report_text = build_report({**snapshot, "mode_key": mode})
    ctx = build_validation_context(snapshot, mode)
    result = ai_client.analyze_single(report_text, cfg, config.AI_REQUEST_TIMEOUT, ctx, mode)
    bingx_symbol = config.to_bingx_symbol(snapshot["symbol"])
    prediction_tracker.record_results(mode, [result], snapshot["current_price"], bingx_symbol)

    with state.lock:
        state.snapshot = snapshot
        results = list(state.results_by_mode.get(mode, []))
        for i, r in enumerate(results):
            if r.label == label:
                results[i] = result
                break
        else:
            results.append(result)
        state.results_by_mode[mode] = results
        state.active_mode = mode
        state.last_updated_at[mode] = time.time()
        html = build_dashboard(snapshot, state.results_by_mode, mode)
        state.last_html = html
        try:
            with open(state.dashboard_output, "w", encoding="utf-8") as f:
                f.write(html)
        except OSError as exc:
            print(f"Не удалось сохранить {state.dashboard_output}: {exc}")
        _save_cache(state)


def _save_cache(state: _State) -> None:
    """Persist state to disk so a later `python server.py` restart (e.g. after
    a code change) can reload it instead of re-fetching BingX/AI from scratch.
    Best-effort — a cache write failure should never break a live request.
    """
    try:
        payload = {
            "snapshot": state.snapshot,
            "results_by_mode": state.results_by_mode,
            "active_mode": state.active_mode,
            "last_updated_at": state.last_updated_at,
        }
        with open(config.DASHBOARD_CACHE_FILE, "wb") as f:
            pickle.dump(payload, f)
    except Exception as exc:  # pragma: no cover - best-effort cache only
        print(f"Не удалось сохранить кеш {config.DASHBOARD_CACHE_FILE}: {exc}")


def _load_cache() -> dict | None:
    """Load a previously saved state, if any. Never raises — a corrupt or
    missing cache just means "no cache", the caller falls back to fetching.
    """
    if not config.DASHBOARD_CACHE_FILE.exists():
        return None
    try:
        with open(config.DASHBOARD_CACHE_FILE, "rb") as f:
            return pickle.load(f)
    except Exception as exc:
        print(f"Не удалось прочитать кеш {config.DASHBOARD_CACHE_FILE}: {exc}")
        return None


def _status_snapshot(state: _State) -> dict:
    """Read-only status report — never triggers a refresh, safe to poll."""
    with state.lock:
        modes = {}
        for mode in ("scalping", "swing"):
            results = state.results_by_mode.get(mode, [])
            modes[mode] = {
                "label": MODE_LABELS[mode],
                "refreshing": state.refreshing_mode == mode,
                "ok_count": sum(1 for r in results if r.ok),
                "total_count": len(results),
                "updated_at": state.last_updated_at.get(mode),
            }
        return {"active_mode": state.active_mode, "modes": modes}


def _make_handler(state: _State):
    class Handler(BaseHTTPRequestHandler):
        server_version = "MarketSnapshotDashboard/1.0"

        def _send_json(self, obj: dict, status: int = 200) -> None:
            body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _send_html(self, html_str: str, status: int = 200) -> None:
            body = html_str.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802 (stdlib method name)
            parsed = urlparse(self.path)

            if parsed.path in ("/", "/dashboard.html"):
                with state.lock:
                    html_str = state.last_html
                self._send_html(html_str)
                return

            if parsed.path == "/api/status":
                # Read-only — poll this to check progress, never /api/refresh
                # (that one has a side effect: it starts a real AI call).
                self._send_json({"ok": True, **_status_snapshot(state)})
                return

            if parsed.path == "/api/refresh":
                query = parse_qs(parsed.query)
                mode = query.get("mode", [state.active_mode])[0]
                model_label = query.get("model", [None])[0]
                if mode not in config.VALID_TRADING_MODES:
                    self._send_json({"ok": False, "error": f"Неизвестный режим: {mode}"}, status=400)
                    return
                if model_label is not None and model_label not in {m.label for m in config.AI_MODELS}:
                    self._send_json({"ok": False, "error": f"Неизвестная модель: {model_label}"}, status=400)
                    return

                # Cooldown is scoped per (mode, model) so refreshing one card
                # doesn't put the whole mode — or the other cards — on cooldown.
                scope = mode if model_label is None else f"{mode}::{model_label}"
                target_desc = (
                    f"«{MODE_LABELS[mode]}» / {model_label}" if model_label else f"режим «{MODE_LABELS[mode]}»"
                )

                with state.lock:
                    if state.refreshing_mode is not None:
                        self._send_json(
                            {
                                "ok": False,
                                "error": f"Уже идёт обновление режима «{MODE_LABELS[state.refreshing_mode]}», подождите.",
                            },
                            status=409,
                        )
                        return
                    last_finished = state.last_refresh_finished_at.get(scope)
                    if last_finished is not None:
                        elapsed = time.monotonic() - last_finished
                        if elapsed < REFRESH_COOLDOWN_SECONDS:
                            wait_left = REFRESH_COOLDOWN_SECONDS - elapsed
                            self._send_json(
                                {
                                    "ok": False,
                                    "error": f"{target_desc} обновлялось только что — подождите ещё {wait_left:.0f} с.",
                                },
                                status=429,
                            )
                            return
                    # Locking is still per-mode (not per-model): only one
                    # refresh in flight per mode at a time, whole-mode or
                    # single-card, to keep results_by_mode[mode] mutations safe.
                    state.refreshing_mode = mode

                try:
                    if model_label is not None:
                        _refresh_single(state, mode, model_label)
                    else:
                        _refresh_mode(state, mode)
                    self._send_json({"ok": True, "mode": mode, "model": model_label})
                except ai_client.AIConfigError as exc:
                    self._send_json({"ok": False, "error": str(exc)}, status=500)
                except (bingx_client.BingXError, ValueError) as exc:
                    self._send_json({"ok": False, "error": f"Ошибка BingX: {exc}"}, status=502)
                except Exception as exc:  # last-resort guard — never crash the server thread
                    self._send_json({"ok": False, "error": f"Внутренняя ошибка: {exc}"}, status=500)
                finally:
                    with state.lock:
                        state.refreshing_mode = None
                        state.last_refresh_finished_at[scope] = time.monotonic()
                return

            self.send_response(404)
            self.end_headers()

        def log_message(self, format: str, *args) -> None:  # noqa: A002
            print(f"[server] {self.address_string()} - {format % args}")

    return Handler


def run_server(
    symbol: str,
    initial_snapshot: dict | None = None,
    initial_results_by_mode: dict[str, list[AIAnalysisResult]] | None = None,
    initial_mode: str = "scalping",
    dashboard_output: str | None = None,
    console=None,
    open_browser: bool = True,
    use_cache: bool = True,
) -> None:
    """Start the live dashboard server. Blocks until Ctrl+C.

    Data source priority for the initial state: explicit
    initial_snapshot/initial_results_by_mode args (e.g. from `main.py
    --serve`, which just computed them) > on-disk cache from a previous run
    (`use_cache=True`, the default) > a fresh fetch (BingX + all AI models,
    both modes) as the last resort. This means a plain restart of an already
    -seeded server never re-queries BingX or any AI model by itself.
    """

    def say(msg: str) -> None:
        if console is not None:
            console.print(msg)
        else:
            print(msg)

    dashboard_output = dashboard_output or str(config.AI_DASHBOARD_FILE)
    state = _State(symbol=symbol, dashboard_output=dashboard_output)

    cached = _load_cache() if (use_cache and initial_snapshot is None) else None

    if initial_snapshot is not None and initial_results_by_mode:
        state.snapshot = initial_snapshot
        state.results_by_mode = dict(initial_results_by_mode)
        state.active_mode = initial_mode
        state.last_html = build_dashboard(initial_snapshot, state.results_by_mode, initial_mode)
    elif cached is not None and cached.get("results_by_mode"):
        say("[cyan]Восстанавливаю последнее состояние из кеша — без новых запросов к BingX/AI...[/cyan]")
        state.snapshot = cached["snapshot"]
        state.results_by_mode = cached["results_by_mode"]
        state.active_mode = cached.get("active_mode", initial_mode)
        state.last_updated_at = cached.get("last_updated_at", {})
        state.last_html = build_dashboard(state.snapshot, state.results_by_mode, state.active_mode)
    else:
        say("[cyan]Кеша нет — первичная загрузка данных для сервера (оба режима)...[/cyan]")
        snapshot = fetch_snapshot(symbol, log=lambda m: say(f"[cyan]{m}[/cyan]"))
        report_texts = {
            mode: build_report({**snapshot, "mode_key": mode}) for mode in ("scalping", "swing")
        }
        validation_contexts = {
            mode: build_validation_context(snapshot, mode) for mode in ("scalping", "swing")
        }
        results_by_mode = ai_client.analyze_modes(
            report_texts, config.AI_MODELS, config.AI_REQUEST_TIMEOUT, validation_contexts
        )
        bingx_symbol = config.to_bingx_symbol(snapshot["symbol"])
        for mode, results in results_by_mode.items():
            prediction_tracker.record_results(mode, results, snapshot["current_price"], bingx_symbol)
        state.snapshot = snapshot
        state.results_by_mode = results_by_mode
        state.active_mode = initial_mode
        state.last_html = build_dashboard(snapshot, results_by_mode, initial_mode)

    now = time.time()
    now_monotonic = time.monotonic()
    for mode in state.results_by_mode:
        state.last_updated_at.setdefault(mode, now)
        state.last_refresh_finished_at[mode] = now_monotonic

    _save_cache(state)

    try:
        with open(dashboard_output, "w", encoding="utf-8") as f:
            f.write(state.last_html)
    except OSError as exc:
        say(f"[yellow]Не удалось сохранить {dashboard_output}: {exc}[/yellow]")

    handler = _make_handler(state)
    httpd = ThreadingHTTPServer((config.DASHBOARD_HOST, config.DASHBOARD_PORT), handler)
    url = f"http://{config.DASHBOARD_HOST}:{config.DASHBOARD_PORT}/"

    say(f"[bold green]Дашборд запущен: {url}[/bold green]")
    say("[dim]Только на этой машине (127.0.0.1) — не пробрасывайте порт наружу без авторизации.[/dim]")
    say("[dim]Ctrl+C — остановить сервер.[/dim]")

    if open_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        say("\n[cyan]Останавливаю сервер...[/cyan]")
    finally:
        httpd.server_close()


if __name__ == "__main__":
    run_server(symbol=config.SYMBOL, initial_mode=config.TRADING_MODE)
