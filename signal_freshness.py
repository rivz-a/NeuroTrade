"""How old is a piece of data, and (for AI results) has its signal expired?

Shared by `consensus_engine.py` (a stale vote doesn't count) and
`dashboard_builder.py` (both the per-card freshness badge and the market
header's "Обновлено N назад" line use the same age-formatting logic, so the
two surfaces never phrase the same thing two different ways).
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from ai_client import AIAnalysisResult


@dataclass(frozen=True)
class Freshness:
    age_seconds: float
    expires_at: float | None
    seconds_remaining: float | None
    is_stale: bool


def compute_freshness(result: AIAnalysisResult, now: float | None = None) -> Freshness | None:
    """None when there's nothing to compute from — no `created_at` timestamp,
    or no `trade_plan` (so no `valid_for_minutes` to expire against).
    """
    if result.created_at is None or result.trade_plan is None:
        return None
    now = now if now is not None else time.time()
    age = now - result.created_at
    valid_seconds = result.trade_plan.valid_for_minutes * 60
    expires_at = result.created_at + valid_seconds
    remaining = expires_at - now
    return Freshness(age_seconds=age, expires_at=expires_at, seconds_remaining=remaining, is_stale=remaining <= 0)


def format_age(seconds: float | None) -> str:
    """Human-readable age: '12 секунд назад' / '4 минуты назад' / '27 минут назад'."""
    if seconds is None:
        return "н/д"
    if seconds < 0:
        seconds = 0
    if seconds < 60:
        return f"{seconds:.0f} с назад"
    minutes = seconds / 60
    if minutes < 90:
        return f"{minutes:.0f} мин назад"
    hours = minutes / 60
    return f"{hours:.1f} ч назад"


def format_remaining(seconds: float | None) -> str:
    """Human-readable time-to-expiry: 'осталось 14 минут' / 'истёк 3 мин назад'."""
    if seconds is None:
        return "н/д"
    if seconds <= 0:
        return f"истёк {format_age(-seconds)}"
    minutes = seconds / 60
    if minutes < 90:
        return f"осталось {minutes:.0f} мин"
    hours = minutes / 60
    return f"осталось {hours:.1f} ч"
