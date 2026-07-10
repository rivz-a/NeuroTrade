"""Shared parsing of a trading verdict (signal + key numbers) out of free-form
AI response text.

Used by both `dashboard_builder.py` (to render the on-card summary) and
`prediction_tracker.py` (to record what a model actually called, so its
accuracy can be scored later) — kept in one place so the two never drift out
of sync with each other.
"""

from __future__ import annotations

import re

SIGNAL_RE = re.compile(r"\b(LONG|SHORT|WAIT)\b", re.IGNORECASE)

_NUM = r"[\d]+(?:[.,]\d+)?"
_RANGE = rf"{_NUM}(?:\s*[-–—]\s*{_NUM})?"

VERDICT_PATTERNS = {
    "probability": re.compile(rf"вероятност\w*[\s\S]{{0,150}}?({_RANGE}\s*%)", re.IGNORECASE),
    "entry": re.compile(
        rf"(?:уровень входа|вход)(?:\s*\([^)]*\))?[^\d]{{0,20}}({_RANGE})", re.IGNORECASE
    ),
    "stop_loss": re.compile(
        rf"(?:stop[\s\-]*loss|стоп[\s\-]*лосс)[^\d]{{0,15}}({_NUM})", re.IGNORECASE
    ),
    "take_profit": re.compile(
        rf"(?:take profit\s*1|tp\s*1|тейк[\s\-]*профит\s*1)[^\d]{{0,15}}({_NUM})", re.IGNORECASE
    ),
}


def detect_signal(content: str | None) -> str:
    if not content:
        return "WAIT"
    match = SIGNAL_RE.search(content)
    return match.group(1).upper() if match else "WAIT"


def extract_verdict(content: str | None) -> dict[str, str | None]:
    """Best-effort extraction of key numbers from free-form AI text.

    Regex-based against Russian/English labels the system prompt asks for.
    Returns None per field when a label isn't found — surfaced as "н/д" in
    the UI rather than silently omitted, so a parsing miss is visible.
    """
    if not content:
        return {key: None for key in VERDICT_PATTERNS}
    result: dict[str, str | None] = {}
    for key, pattern in VERDICT_PATTERNS.items():
        match = pattern.search(content)
        result[key] = match.group(1).replace(" ", "") if match else None
    return result
