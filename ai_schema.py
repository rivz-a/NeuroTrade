"""Strict schema for AI trade-plan responses (Pydantic).

Every model is now required to answer with exactly one JSON object matching
`TradePlan` — no more scraping numbers out of free-form prose with regex.
`parse_ai_json()` is the single entry point `ai_client.py` uses to turn a raw
model response into a validated `TradePlan`, raising `SchemaError` (which
carries enough detail to drive a one-shot "fix your JSON" repair request) on
any failure.
"""

from __future__ import annotations

import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

Signal = Literal["LONG", "SHORT", "WAIT"]

EntryStatus = Literal[
    "ENTER_NOW",
    "WAIT_PULLBACK",
    "WAIT_BREAKOUT",
    "WAIT_CONFIRMATION",
    "LATE_ENTRY",
    "REJECTED",
    "NO_TRADE",
]

MarketRegime = Literal[
    "TREND_UP",
    "TREND_DOWN",
    "RANGE",
    "VOLATILITY_EXPANSION",
    "VOLATILITY_COMPRESSION",
    "REVERSAL_RISK",
    "UNSTABLE",
    "UNKNOWN",
]


class EntryZone(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    type: str
    from_: float = Field(alias="from")
    to: float
    trigger: str = ""


class TakeProfit(BaseModel):
    label: str
    price: float
    close_percent: float = Field(ge=0, le=100)


class RiskReward(BaseModel):
    tp1: float | None = None
    tp2: float | None = None
    tp3: float | None = None


class TradePlan(BaseModel):
    signal: Signal
    entry_status: EntryStatus
    confidence: int = Field(ge=0, le=100)
    market_regime: MarketRegime
    entry: EntryZone
    stop_loss: float
    take_profits: list[TakeProfit] = Field(default_factory=list)
    risk_reward: RiskReward = Field(default_factory=RiskReward)
    time_horizon_minutes: int
    valid_for_minutes: int = Field(gt=0)
    reasons: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    invalidation_conditions: list[str] = Field(default_factory=list)
    wait_conditions: list[str] = Field(default_factory=list)
    summary: str = ""


class SchemaError(Exception):
    """Raised when raw model text isn't a valid TradePlan.

    `detail` is a short, human-readable description of what went wrong —
    used both in the local diagnostic log and in the one-shot repair prompt
    sent back to the model ("your previous answer failed because: ...").
    """

    def __init__(self, detail: str, raw_text: str):
        super().__init__(detail)
        self.detail = detail
        self.raw_text = raw_text


def _strip_markdown_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines:
            lines = lines[1:]  # drop opening ```json / ```
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


def _extract_first_json_object(text: str) -> str:
    """Find the first balanced {...} object in text (JSON boundary detection,
    not a regex extraction of trading numbers — a different concern from the
    banned free-text level-scraping).
    """
    start = text.find("{")
    if start == -1:
        raise SchemaError("В ответе не найдено ни одной '{' — это не JSON.", text)
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    raise SchemaError("Не найдена парная закрывающая '}' — JSON-объект оборван.", text)


def parse_ai_json(raw_text: str) -> TradePlan:
    if not raw_text or not raw_text.strip():
        raise SchemaError("Пустой ответ модели.", raw_text or "")

    candidate = _strip_markdown_fence(raw_text)
    json_slice = _extract_first_json_object(candidate)

    try:
        data = json.loads(json_slice)
    except json.JSONDecodeError as exc:
        raise SchemaError(f"Невалидный JSON: {exc}", raw_text) from exc

    try:
        return TradePlan.model_validate(data)
    except ValidationError as exc:
        problems = "; ".join(
            f"{'.'.join(str(p) for p in err['loc'])}: {err['msg']}" for err in exc.errors()
        )
        raise SchemaError(f"JSON не соответствует схеме: {problems}", raw_text) from exc
