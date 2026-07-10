"""Client for sending the market snapshot report to one or more AI APIs for analysis.

Talks to plain OpenAI-compatible Chat Completions HTTP endpoints (works with
gateways such as APINET.CLOUD that proxy to OpenAI/Claude/Gemini models, or
with those providers' own APIs). This module only sends the already-built
report text — it never touches BingX and never places any trades.

Multiple configured models are queried in parallel so a multi-model
comparison doesn't take N times as long as a single call.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

import requests

from config import AIModelConfig

SYSTEM_PROMPTS = {
    "scalping": (
        "Ты — ассистент-аналитик по скальпингу криптовалютных фьючерсов (трейдер держит "
        "позицию минуты, а не часы). Тебе присылают срез рыночных данных по одной паре (цены, "
        "индикаторы по нескольким таймфреймам, стакан заявок, funding rate, open interest, "
        "уровни поддержки/сопротивления и текущая позиция трейдера, если она есть). Отвечай на "
        "русском языке, структурированно и по существу: сигнал LONG/SHORT/WAIT, уровень входа, "
        "stop loss, take profit 1/2/3, вероятность сценария и ключевые риски. Опирайся в первую "
        "очередь на данные младших таймфреймов (1m/5m) — стоп и вход должны быть точными, в "
        "пределах локального шума. Опирайся на конкретные цифры из присланных данных, а не на "
        "общие фразы. В конце обязательно напомни, что это не финансовая рекомендация."
    ),
    "swing": (
        "Ты — ассистент-аналитик по внутридневной свинг-торговле криптовалютными фьючерсами "
        "(трейдер держит позицию от нескольких часов до суток — это НЕ скальпинг, узкие "
        "1m-стопы здесь не подходят). Тебе присылают срез рыночных данных по одной паре (цены, "
        "индикаторы по нескольким таймфреймам, стакан заявок, funding rate, open interest, "
        "уровни поддержки/сопротивления и текущая позиция трейдера, если она есть). Отвечай на "
        "русском языке, структурированно и по существу: сигнал LONG/SHORT/WAIT, уровень входа, "
        "stop loss, take profit 1/2/3, вероятность сценария и ключевые риски. Stop loss "
        "размещай за структурными уровнями 15m/1h (EMA20/EMA50, локальные экстремумы, ATR "
        "старшего таймфрейма) — не в паре долларов от входа. Take profit выбирай на более "
        "удалённых уровнях поддержки/сопротивления так, чтобы итоговое соотношение риск/прибыль "
        "было не хуже 1:1.5; если ближайшие уровни дают хуже — прямо скажи об этом и предложи "
        "более дальнюю цель или сигнал WAIT. Опирайся на конкретные цифры из присланных данных, "
        "а не на общие фразы. В конце обязательно напомни, что это не финансовая рекомендация."
    ),
}


class AIError(Exception):
    """Base class for AI client errors."""


class AIConfigError(AIError):
    """Raised when no AI model is configured with an API key at all."""


@dataclass(frozen=True)
class AIAnalysisResult:
    label: str
    model: str
    content: str | None
    error: str | None
    latency_seconds: float
    # Wall-clock time the call finished (epoch seconds). Optional/defaulted so
    # AIAnalysisResult instances pickled by an older version of this app (in
    # dashboard_cache.pkl) still unpickle fine after this field was added —
    # they just won't have a real value, callers should treat None as "н/д".
    created_at: float | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


def _call_model(cfg: AIModelConfig, report_text: str, timeout: float, system_prompt: str) -> AIAnalysisResult:
    start = time.monotonic()

    url = f"{cfg.base_url.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {cfg.api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": cfg.model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": report_text},
        ],
    }

    def _fail(message: str) -> AIAnalysisResult:
        return AIAnalysisResult(cfg.label, cfg.model, None, message, time.monotonic() - start, time.time())

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=timeout)
    except requests.exceptions.Timeout:
        return _fail(f"Тайм-аут запроса к AI API ({url})")
    except requests.exceptions.ConnectionError:
        return _fail(f"Не удалось подключиться к AI API. Проверьте интернет-соединение. ({url})")
    except requests.exceptions.RequestException as exc:
        return _fail(f"Ошибка сети при обращении к AI API: {exc}")

    if response.status_code in (401, 403):
        return _fail(
            f"AI API отклонил запрос (HTTP {response.status_code}). Проверьте API-ключ/баланс. "
            f"Ответ шлюза: {response.text[:300]}"
        )
    if response.status_code == 429:
        return _fail("AI API вернул 429: превышен лимит запросов.")
    if response.status_code >= 400:
        return _fail(f"AI API вернул ошибку (HTTP {response.status_code}): {response.text[:300]}")

    try:
        data = response.json()
    except ValueError:
        return _fail("AI API вернул некорректный (не-JSON) ответ.")

    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        return _fail(f"Неожиданный формат ответа AI API: {data}")

    if not content or not content.strip():
        return _fail("AI API вернул пустой ответ.")

    return AIAnalysisResult(cfg.label, cfg.model, content.strip(), None, time.monotonic() - start, time.time())


def analyze_modes(
    report_texts: dict[str, str], models: list[AIModelConfig], timeout: float
) -> dict[str, list[AIAnalysisResult]]:
    """Query every configured model for every given mode in a single parallel batch.

    `report_texts` maps mode key ("scalping"/"swing") to the report text built
    for that mode. All model×mode combinations fire concurrently (not mode
    after mode), so total wall-clock time stays close to the single slowest
    call rather than the sum of each mode's round.
    """
    configured = [m for m in models if m.api_key]
    if not configured:
        raise AIConfigError(
            "Ни для одной модели не задан API-ключ в .env (AI_API_KEY / AI2_API_KEY / "
            "AI3_API_KEY). Укажите хотя бы один ключ или запустите с флагом --no-ai."
        )

    tasks = [
        (mode, cfg, text, SYSTEM_PROMPTS.get(mode, SYSTEM_PROMPTS["scalping"]))
        for mode, text in report_texts.items()
        for cfg in configured
    ]

    results: dict[tuple[str, str], AIAnalysisResult] = {}
    with ThreadPoolExecutor(max_workers=len(tasks)) as executor:
        future_to_key = {
            executor.submit(_call_model, cfg, text, timeout, system_prompt): (mode, cfg.label)
            for mode, cfg, text, system_prompt in tasks
        }
        for future in as_completed(future_to_key):
            mode, label = future_to_key[future]
            results[(mode, label)] = future.result()

    return {
        mode: [results[(mode, cfg.label)] for cfg in configured] for mode in report_texts
    }


def analyze_with_all(
    report_text: str, models: list[AIModelConfig], timeout: float, mode: str = "scalping"
) -> list[AIAnalysisResult]:
    """Query every configured model in parallel for a single mode."""
    return analyze_modes({mode: report_text}, models, timeout)[mode]


def analyze_single(
    report_text: str, cfg: AIModelConfig, timeout: float, mode: str = "scalping"
) -> AIAnalysisResult:
    """Query exactly one model for one mode.

    Used by the dashboard's per-model refresh button so refreshing/retrying a
    single card never re-queries (and re-bills) the other configured models.
    """
    system_prompt = SYSTEM_PROMPTS.get(mode, SYSTEM_PROMPTS["scalping"])
    return _call_model(cfg, report_text, timeout, system_prompt)
