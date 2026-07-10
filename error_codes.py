"""Normalized error codes shown to the user.

Raw diagnostics (gateway URLs, response bodies, API keys, account IDs) must
never reach the UI — they go to the local log via `app_log.py` instead. This
module maps each internal failure to a short code + a safe, generic Russian
message that's fine to render on a dashboard card.
"""

from __future__ import annotations

AI_AUTH_ERROR = "AI_AUTH_ERROR"
RATE_LIMIT = "RATE_LIMIT"
PROVIDER_TIMEOUT = "PROVIDER_TIMEOUT"
NETWORK_ERROR = "NETWORK_ERROR"
INVALID_JSON = "INVALID_JSON"
VALIDATION_FAILED = "VALIDATION_FAILED"
BINGX_UNAVAILABLE = "BINGX_UNAVAILABLE"
STALE_MARKET_DATA = "STALE_MARKET_DATA"
UNKNOWN_ERROR = "UNKNOWN_ERROR"

MESSAGES: dict[str, str] = {
    AI_AUTH_ERROR: "Модель отклонила запрос (ключ или баланс/квота). Проверьте настройки провайдера.",
    RATE_LIMIT: "Превышен лимит запросов к модели. Попробуйте позже.",
    PROVIDER_TIMEOUT: "Модель не ответила за отведённое время.",
    NETWORK_ERROR: "Не удалось связаться с AI API — проверьте интернет-соединение.",
    INVALID_JSON: "Модель не смогла вернуть структурированный план в нужном формате.",
    VALIDATION_FAILED: "Торговый план не прошёл проверку валидатора.",
    BINGX_UNAVAILABLE: "BingX недоступен — рыночные данные не получены.",
    STALE_MARKET_DATA: "Рыночные данные устарели.",
    UNKNOWN_ERROR: "Внутренняя ошибка при обработке ответа модели.",
}


def message_for(code: str) -> str:
    return MESSAGES.get(code, MESSAGES[UNKNOWN_ERROR])
