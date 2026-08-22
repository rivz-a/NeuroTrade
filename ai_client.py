"""Client for sending the market snapshot report to one or more AI APIs for analysis.

Talks to plain OpenAI-compatible Chat Completions HTTP endpoints (works with
gateways such as APINET.CLOUD that proxy to OpenAI/Claude/Gemini models, or
with those providers' own APIs). This module only sends the already-built
report text — it never touches BingX and never places any trades.

Every model is required to answer with a single strict JSON object (see
`ai_schema.TradePlan`) instead of free-form prose — no more regex-scraping of
entry/stop/target numbers out of text. If the first response doesn't parse or
validate against the schema, exactly one "please fix your JSON" repair round
trip is attempted before giving up (`error_code=INVALID_JSON`). A
successfully parsed plan is then run through `trade_validator.py`; a plan
that fails validation is never surfaced as a usable signal.

Multiple configured models are queried in parallel so a multi-model
comparison doesn't take N times as long as a single call.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

import requests

import app_log
import error_codes
from ai_schema import SchemaError, TradePlan, parse_ai_json
from config import AIModelConfig
from trade_validator import ValidationContext, ValidationResult, validate_trade_plan

JSON_INSTRUCTIONS = (
    "\n\nОТВЕЧАЙ СТРОГО В ФОРМАТЕ JSON — один JSON-объект и ничего больше "
    "(без markdown-ограждений ```, без пояснений до или после). Схема:\n"
    "{\n"
    '  "signal": "LONG" | "SHORT" | "WAIT",\n'
    '  "entry_status": "ENTER_NOW" | "WAIT_PULLBACK" | "WAIT_BREAKOUT" | "WAIT_CONFIRMATION" | '
    '"LATE_ENTRY" | "REJECTED" | "NO_TRADE",\n'
    '  "confidence": число 0-100 (твоя субъективная уверенность, НЕ статистическая вероятность),\n'
    '  "market_regime": "TREND_UP" | "TREND_DOWN" | "RANGE" | "VOLATILITY_EXPANSION" | '
    '"VOLATILITY_COMPRESSION" | "REVERSAL_RISK" | "UNSTABLE" | "UNKNOWN" '
    "(твоя собственная оценка ПОСЛЕ того, как тебе уже показали программно определённый режим — "
    "не обязана совпадать; если не совпадает, отметь это в contradictions, не подгоняй ответ под чужой),\n"
    '  "entry": {"type": строка, "from": число, "to": число, "trigger": строка},\n'
    '  "stop_loss": число,\n'
    '  "take_profits": [{"label": "TP1", "price": число, "close_percent": число}, ...],\n'
    '  "time_horizon_minutes": число,\n'
    '  "valid_for_minutes": число > 0,\n'
    '  "reasons": [строка, ...],\n'
    '  "risks": [строка, ...],\n'
    '  "invalidation_conditions": [строка, ...],\n'
    '  "wait_conditions": [строка, ...] (обязательно непустой список или заполненный '
    "invalidation_conditions, если signal=\"WAIT\"),\n"
    '  "contradictions": [строка, ...] (противоречия между признаками/режимом/оценками движка — '
    'пустой список, если противоречий нет, НЕ выдумывай их),\n'
    '  "missing_context": [строка, ...] (важные данные, которых, на твой взгляд, не хватает в '
    'присланном срезе — пустой список, если всё нужное есть),\n'
    '  "summary": строка\n'
    "}\n"
    "Для LONG: stop_loss ниже входа, TP1 < TP2 < TP3 и все выше входа. Для SHORT — зеркально "
    "(stop_loss выше входа, TP1 > TP2 > TP3 и все ниже входа). Все числа — обычные JSON-числа "
    "(точка как разделитель дробной части, без пробелов и разделителей тысяч). НЕ включай в ответ "
    "поле risk_reward или любой расчёт размера позиции/маржи/комиссий — этого поля больше нет в "
    "схеме, такие расчёты выполняет код."
)

_ROLE_TEXT = (
    "Твоя роль изменилась: ты БОЛЬШЕ НЕ единственный источник торгового решения. Направление, "
    "оценка сценариев (LONG_SCORE/SHORT_SCORE/NO_TRADE_SCORE) и режим рынка уже посчитаны кодом "
    "и присланы тебе в разделе «Программный анализ» — это не черновик, который нужно улучшить, "
    "а готовый программный вывод. Твоя задача:\n"
    "1) интерпретировать присланные признаки и объяснить, что они означают вместе;\n"
    "2) находить противоречия — между таймфреймами, между твоим прочтением и программным режимом/"
    "оценками, между импульсом и структурой и т.д. — и явно перечислять их в contradictions;\n"
    "3) предлагать конкретный сценарий (вход/стоп/цели/условия), а не просто пересказывать цифры;\n"
    "4) объяснять риски сценария понятным языком, не только цифрами;\n"
    "5) проверять, не пропущен ли важный контекст (устаревшие данные, отсутствующий funding/OI, "
    "тонкий стакан и т.п.) и перечислять такие пробелы в missing_context;\n"
    "6) формировать понятное summary — как для человека, который бегло читает дашборд.\n\n"
    "Тебе ЗАПРЕЩЕНО: рассчитывать размер позиции; определять комиссии или маржу; самостоятельно "
    "вычислять итоговый R:R (поля risk_reward в схеме больше нет); менять заданные пользователем "
    "риск-ограничения (риск на сделку, макс. маржа, минимальный R:R, плечо — они присланы в разделе "
    "«Риск-ограничения» и фиксированы). Всё это делает код детерминированно, не ты. Твой план — "
    "предложение, а не команда на исполнение: сделка никогда не открывается автоматически и не "
    "минует программные фильтры (валидацию плана, расчёт позиции, проверки риска) независимо от "
    "того, что ты укажешь."
)

SYSTEM_PROMPTS = {
    "scalping": (
        "Ты — ассистент-аналитик по скальпингу криптовалютных фьючерсов (трейдер держит "
        "позицию минуты, а не часы). Тебе присылают срез рыночных данных по одной паре (цены, "
        "индикаторы по нескольким таймфреймам, стакан заявок, funding rate, open interest, "
        "уровни поддержки/сопротивления, программный анализ режима/оценок и текущая позиция "
        "трейдера, если она есть). Опирайся в первую очередь на данные младших таймфреймов "
        "(1m/5m) — стоп и вход должны быть точными, в пределах локального шума. Опирайся на "
        "конкретные цифры из присланных данных, а не на общие фразы.\n\n" + _ROLE_TEXT + JSON_INSTRUCTIONS
    ),
    "swing": (
        "Ты — ассистент-аналитик по внутридневной свинг-торговле криптовалютными фьючерсами "
        "(трейдер держит позицию от нескольких часов до суток — это НЕ скальпинг, узкие "
        "1m-стопы здесь не подходят). Тебе присылают срез рыночных данных по одной паре (цены, "
        "индикаторы по нескольким таймфреймам, стакан заявок, funding rate, open interest, "
        "уровни поддержки/сопротивления, программный анализ режима/оценок и текущая позиция "
        "трейдера, если она есть). Stop loss размещай за структурными уровнями 15m/1h "
        "(EMA20/EMA50, локальные экстремумы, ATR старшего таймфрейма) — не в паре долларов от "
        "входа. Take profit выбирай на более удалённых уровнях поддержки/сопротивления. Опирайся "
        "на конкретные цифры из присланных данных, а не на общие фразы.\n\n" + _ROLE_TEXT + JSON_INSTRUCTIONS
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
    # dashboard_cache.pkl) still unpickle fine after a field was added —
    # they just won't have a real value, callers should treat None as "н/д".
    created_at: float | None = None
    # Structured plan + validator verdict (None for legacy cached results
    # from before the JSON-schema migration, or when parsing/repair failed).
    trade_plan: TradePlan | None = None
    validation: ValidationResult | None = None
    error_code: str | None = None
    repaired: bool = False

    @property
    def ok(self) -> bool:
        """A result counts as "ok" only when there's an actually usable,
        validator-approved trade plan — not just "the HTTP call succeeded".
        A parsed-but-rejected plan (validation.status == "rejected") always
        carries a non-None `error`, so it's excluded here too.
        """
        return self.error is None and self.trade_plan is not None


def _call_model(
    cfg: AIModelConfig,
    report_text: str,
    timeout: float,
    system_prompt: str,
    mode: str,
    validation_ctx: ValidationContext,
) -> AIAnalysisResult:
    start = time.monotonic()

    url = f"{cfg.base_url.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {cfg.api_key}",
        "Content-Type": "application/json",
    }
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": report_text},
    ]

    def _fail(code: str, log_detail: str | None = None, content: str | None = None) -> AIAnalysisResult:
        if log_detail:
            app_log.log_error(code, cfg.model, mode, log_detail)
        return AIAnalysisResult(
            label=cfg.label,
            model=cfg.model,
            content=content,
            error=error_codes.message_for(code),
            latency_seconds=time.monotonic() - start,
            created_at=time.time(),
            error_code=code,
        )

    def _post(msgs: list[dict]) -> requests.Response:
        return requests.post(url, headers=headers, json={"model": cfg.model, "messages": msgs}, timeout=timeout)

    try:
        response = _post(messages)
    except requests.exceptions.Timeout:
        return _fail(error_codes.PROVIDER_TIMEOUT, log_detail=f"timeout url={url}")
    except requests.exceptions.ConnectionError as exc:
        return _fail(error_codes.NETWORK_ERROR, log_detail=f"connection error: {exc}")
    except requests.exceptions.RequestException as exc:
        return _fail(error_codes.NETWORK_ERROR, log_detail=f"request exception: {exc}")

    if response.status_code in (401, 403):
        return _fail(
            error_codes.AI_AUTH_ERROR,
            log_detail=f"HTTP {response.status_code} url={url} body={response.text[:500]}",
        )
    if response.status_code == 429:
        return _fail(error_codes.RATE_LIMIT, log_detail=f"HTTP 429 url={url}")
    if response.status_code >= 400:
        return _fail(
            error_codes.UNKNOWN_ERROR,
            log_detail=f"HTTP {response.status_code} url={url} body={response.text[:500]}",
        )

    try:
        data = response.json()
    except ValueError:
        return _fail(error_codes.UNKNOWN_ERROR, log_detail="HTTP envelope wasn't JSON")

    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        return _fail(error_codes.UNKNOWN_ERROR, log_detail=f"unexpected envelope shape: {data}")

    if not content or not content.strip():
        return _fail(error_codes.UNKNOWN_ERROR, log_detail="empty content")

    content = content.strip()
    repaired = False

    try:
        plan = parse_ai_json(content)
    except SchemaError as exc:
        app_log.log_ai_failure(cfg.model, mode, "first_attempt", exc.detail, exc.raw_text)
        app_log.log_ai_repair(cfg.model, mode)

        repair_messages = messages + [
            {"role": "assistant", "content": content},
            {
                "role": "user",
                "content": (
                    f"Твой предыдущий ответ не прошёл проверку схемы: {exc.detail}\n"
                    "Верни ИСПРАВЛЕННЫЙ ответ — ТОЛЬКО валидный JSON по той же схеме, без "
                    "markdown-ограждений и пояснений."
                ),
            },
        ]
        try:
            repair_response = _post(repair_messages)
        except requests.exceptions.RequestException as repair_exc:
            return _fail(error_codes.INVALID_JSON, log_detail=f"repair request failed: {repair_exc}", content=content)

        if repair_response.status_code >= 400:
            return _fail(
                error_codes.INVALID_JSON,
                log_detail=f"repair HTTP {repair_response.status_code}: {repair_response.text[:500]}",
                content=content,
            )

        try:
            repair_data = repair_response.json()
            repair_content = repair_data["choices"][0]["message"]["content"].strip()
        except (ValueError, KeyError, IndexError, TypeError):
            return _fail(error_codes.INVALID_JSON, log_detail="repair envelope malformed", content=content)

        content = repair_content
        try:
            plan = parse_ai_json(content)
            repaired = True
        except SchemaError as exc2:
            app_log.log_ai_failure(cfg.model, mode, "repair_attempt", exc2.detail, exc2.raw_text)
            return _fail(error_codes.INVALID_JSON, content=content)

    validation = validate_trade_plan(plan, mode, validation_ctx)
    error_message = None
    error_code = None
    if validation.status == "rejected":
        error_code = error_codes.VALIDATION_FAILED
        error_message = error_codes.message_for(error_code)

    return AIAnalysisResult(
        label=cfg.label,
        model=cfg.model,
        content=content,
        error=error_message,
        latency_seconds=time.monotonic() - start,
        created_at=time.time(),
        trade_plan=plan,
        validation=validation,
        error_code=error_code,
        repaired=repaired,
    )


def analyze_modes(
    report_texts: dict[str, str],
    models: list[AIModelConfig],
    timeout: float,
    validation_contexts: dict[str, ValidationContext],
) -> dict[str, list[AIAnalysisResult]]:
    """Query every configured model for every given mode in a single parallel batch.

    `report_texts` maps mode key ("scalping"/"swing") to the report text built
    for that mode; `validation_contexts` maps the same keys to the
    `ValidationContext` (current price/ATR/spread) used to check that mode's
    plans. All model×mode combinations fire concurrently (not mode after
    mode), so total wall-clock time stays close to the single slowest call
    rather than the sum of each mode's round.
    """
    configured = [m for m in models if m.api_key]
    if not configured:
        raise AIConfigError(
            "Ни для одной модели не задан API-ключ в .env (AI_API_KEY / "
            "AI2_API_KEY). Укажите хотя бы один ключ или запустите с флагом --no-ai."
        )

    tasks = [
        (mode, cfg, text, SYSTEM_PROMPTS.get(mode, SYSTEM_PROMPTS["scalping"]), validation_contexts[mode])
        for mode, text in report_texts.items()
        for cfg in configured
    ]

    results: dict[tuple[str, str], AIAnalysisResult] = {}
    with ThreadPoolExecutor(max_workers=len(tasks)) as executor:
        future_to_key = {
            executor.submit(_call_model, cfg, text, timeout, system_prompt, mode, ctx): (mode, cfg.label)
            for mode, cfg, text, system_prompt, ctx in tasks
        }
        for future in as_completed(future_to_key):
            mode, label = future_to_key[future]
            results[(mode, label)] = future.result()

    return {
        mode: [results[(mode, cfg.label)] for cfg in configured] for mode in report_texts
    }


def analyze_with_all(
    report_text: str,
    models: list[AIModelConfig],
    timeout: float,
    validation_ctx: ValidationContext,
    mode: str = "scalping",
) -> list[AIAnalysisResult]:
    """Query every configured model in parallel for a single mode."""
    return analyze_modes({mode: report_text}, models, timeout, {mode: validation_ctx})[mode]


def analyze_single(
    report_text: str,
    cfg: AIModelConfig,
    timeout: float,
    validation_ctx: ValidationContext,
    mode: str = "scalping",
) -> AIAnalysisResult:
    """Query exactly one model for one mode.

    Used by the dashboard's per-model refresh button so refreshing/retrying a
    single card never re-queries (and re-bills) the other configured models.
    """
    system_prompt = SYSTEM_PROMPTS.get(mode, SYSTEM_PROMPTS["scalping"])
    return _call_model(cfg, report_text, timeout, system_prompt, mode, validation_ctx)
