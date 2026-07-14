"""Local JSON persistence for `risk_manager.RiskSettings` — the user's risk
parameters, edited from the dashboard's "Настройки риска" panel.

Plain JSON file (`config.RISK_SETTINGS_FILE`), Decimal fields stored as
strings so a read never round-trips through `float`. No API keys or any
BingX/AI credentials live here — only numeric/enum risk parameters.
"""

from __future__ import annotations

import json
from decimal import InvalidOperation

import config
from risk_manager import DEFAULT_RISK_SETTINGS, RiskSettings, validate_settings

# Exactly `RiskSettings.from_raw`'s keyword-argument names — the one place
# that shape is spelled out, so callers can filter an HTTP body down to only
# the fields this module understands before spreading it as kwargs. Public
# (not `FIELDS`) because `position_service`'s what-if calculator needs it
# too, to merge a partial override dict on top of the saved settings.
FIELDS = [
    "account_balance_usdt",
    "available_balance_usdt",
    "risk_percent",
    "leverage",
    "margin_mode",
    "max_margin_percent",
    "maker_fee_percent",
    "taker_fee_percent",
    "slippage_percent",
    "min_risk_reward",
    "entry_price_mode",
    "quantity_step",
    "price_step",
    "minimum_order_notional_usdt",
    "minimum_quantity",
    "bingx_order_input_mode",
]


def to_raw(settings: RiskSettings) -> dict:
    """Public — reused by `position_service`'s what-if calculator to build a
    starting point for a partial-override merge.
    """
    return {
        "account_balance_usdt": str(settings.account_balance_usdt),
        "available_balance_usdt": (
            str(settings.available_balance_usdt) if settings.available_balance_usdt is not None else None
        ),
        "risk_percent": str(settings.risk_percent),
        "leverage": settings.leverage,
        "margin_mode": settings.margin_mode.value,
        "max_margin_percent": str(settings.max_margin_percent),
        "maker_fee_percent": str(settings.maker_fee_percent),
        "taker_fee_percent": str(settings.taker_fee_percent),
        "slippage_percent": str(settings.slippage_percent),
        "min_risk_reward": str(settings.min_risk_reward),
        "entry_price_mode": settings.entry_price_mode.value,
        "quantity_step": str(settings.quantity_step),
        "price_step": str(settings.price_step),
        "minimum_order_notional_usdt": str(settings.minimum_order_notional_usdt),
        "minimum_quantity": str(settings.minimum_quantity),
        "bingx_order_input_mode": settings.bingx_order_input_mode.value,
    }


def _write(raw: dict) -> None:
    with open(config.RISK_SETTINGS_FILE, "w", encoding="utf-8") as f:
        json.dump(raw, f, ensure_ascii=False, indent=2)


def build(raw: dict) -> RiskSettings | list[str]:
    """Pure — validates a raw dict into a `RiskSettings` without touching
    disk. `save()` is this plus a write; `position_service`'s what-if
    calculator uses this directly so a "calculate with different numbers"
    request never persists anything.
    """
    filtered = {k: v for k, v in raw.items() if k in FIELDS}
    try:
        settings = RiskSettings.from_raw(**filtered)
    except (TypeError, ValueError, InvalidOperation) as exc:
        return [f"Некорректные настройки: {exc}"]

    issues = validate_settings(settings)
    if issues:
        return issues
    return settings


def load() -> RiskSettings:
    """Reads saved settings, creating the file with defaults on first run.
    Never raises — a missing/corrupt file falls back to (and is overwritten
    with) `DEFAULT_RISK_SETTINGS`, the same "never break the dashboard"
    spirit as `server.py`'s pickle cache.
    """
    if not config.RISK_SETTINGS_FILE.exists():
        _write(to_raw(DEFAULT_RISK_SETTINGS))
        return DEFAULT_RISK_SETTINGS
    try:
        with open(config.RISK_SETTINGS_FILE, encoding="utf-8") as f:
            raw = json.load(f)
        filtered = {k: v for k, v in raw.items() if k in FIELDS}
        return RiskSettings.from_raw(**filtered)
    except (OSError, json.JSONDecodeError, TypeError, ValueError, InvalidOperation, KeyError):
        return DEFAULT_RISK_SETTINGS


def save(raw: dict) -> RiskSettings | list[str]:
    """Validates and persists new settings. Returns the saved `RiskSettings`
    on success, or a list of human-readable problems on failure — nothing is
    written to disk when validation fails. Expects the FULL settings shape
    (a PUT replaces the whole object; the settings panel always submits
    every field), not a partial patch.
    """
    result = build(raw)
    if isinstance(result, list):
        return result
    _write(to_raw(result))
    return result


def merge_overrides(base: RiskSettings, overrides: dict) -> RiskSettings | list[str]:
    """Applies a partial dict of overrides on top of `base` — used by the
    what-if `POST /api/calculate-position` endpoint, which lets a caller
    override just a few fields (balance, risk%, leverage, ...) without
    resubmitting the entire settings object. Never touches disk.
    """
    merged = to_raw(base)
    merged.update({k: v for k, v in (overrides or {}).items() if k in FIELDS})
    return build(merged)
