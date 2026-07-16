"""Offline tests for ai_schema.parse_ai_json — no network, no AI calls.

Covers the malformed-response edge cases from the product spec (section 31):
markdown fences, missing fields, out-of-range confidence, invalid enum
values, plain prose instead of JSON, prose wrapped around a JSON object, and
JSON-syntax breakage (a comma inside a number literal).
"""

import json

import pytest

from ai_schema import SchemaError, parse_ai_json

VALID_LONG_JSON = """
{
  "signal": "LONG",
  "entry_status": "WAIT_CONFIRMATION",
  "confidence": 68,
  "market_regime": "TREND_UP",
  "entry": {"type": "LIMIT_ZONE", "from": 1776.80, "to": 1777.20, "trigger": "Close above 1777.20"},
  "stop_loss": 1774.80,
  "take_profits": [
    {"label": "TP1", "price": 1779.20, "close_percent": 40},
    {"label": "TP2", "price": 1781.00, "close_percent": 35},
    {"label": "TP3", "price": 1785.00, "close_percent": 25}
  ],
  "time_horizon_minutes": 30,
  "valid_for_minutes": 15,
  "reasons": ["reason1"],
  "risks": ["risk1"],
  "invalidation_conditions": ["cond1"],
  "wait_conditions": [],
  "contradictions": [],
  "missing_context": [],
  "summary": "test"
}
"""


def test_valid_json_parses():
    plan = parse_ai_json(VALID_LONG_JSON)
    assert plan.signal == "LONG"
    assert plan.entry.from_ == 1776.80
    assert plan.stop_loss == 1774.80
    assert plan.take_profits[0].label == "TP1"


def test_markdown_fence_is_stripped():
    wrapped = "```json\n" + VALID_LONG_JSON + "\n```"
    plan = parse_ai_json(wrapped)
    assert plan.signal == "LONG"


def test_prose_before_and_after_json_object_extracted():
    text = f"Вот план:\n{VALID_LONG_JSON}\nНадеюсь, поможет."
    plan = parse_ai_json(text)
    assert plan.signal == "LONG"


def test_missing_stop_loss_raises():
    data = json.loads(VALID_LONG_JSON)
    del data["stop_loss"]
    with pytest.raises(SchemaError):
        parse_ai_json(json.dumps(data))


def test_missing_confidence_raises():
    data = json.loads(VALID_LONG_JSON)
    del data["confidence"]
    with pytest.raises(SchemaError):
        parse_ai_json(json.dumps(data))


def test_confidence_over_100_raises():
    data = json.loads(VALID_LONG_JSON)
    data["confidence"] = 150
    with pytest.raises(SchemaError):
        parse_ai_json(json.dumps(data))


def test_invalid_signal_raises():
    data = json.loads(VALID_LONG_JSON)
    data["signal"] = "BUY"
    with pytest.raises(SchemaError):
        parse_ai_json(json.dumps(data))


def test_plain_text_raises():
    with pytest.raises(SchemaError):
        parse_ai_json("Я думаю, что стоит открыть LONG около 1777.")


def test_malformed_json_with_comma_in_number_raises():
    bad = VALID_LONG_JSON.replace('"stop_loss": 1774.80', '"stop_loss": 1,774.80')
    with pytest.raises(SchemaError):
        parse_ai_json(bad)


def test_empty_response_raises():
    with pytest.raises(SchemaError):
        parse_ai_json("")


def test_wait_signal_parses():
    data = json.loads(VALID_LONG_JSON)
    data["signal"] = "WAIT"
    data["wait_conditions"] = ["Закрепление выше 1780"]
    plan = parse_ai_json(json.dumps(data))
    assert plan.signal == "WAIT"


def test_contradictions_and_missing_context_populated():
    data = json.loads(VALID_LONG_JSON)
    data["contradictions"] = ["Собственная оценка режима TREND_UP расходится с переданным RANGE"]
    data["missing_context"] = ["Не передана история ликвидаций"]
    plan = parse_ai_json(json.dumps(data))
    assert plan.contradictions == ["Собственная оценка режима TREND_UP расходится с переданным RANGE"]
    assert plan.missing_context == ["Не передана история ликвидаций"]


def test_contradictions_and_missing_context_default_to_empty_list():
    data = json.loads(VALID_LONG_JSON)
    del data["contradictions"]
    del data["missing_context"]
    plan = parse_ai_json(json.dumps(data))
    assert plan.contradictions == []
    assert plan.missing_context == []
