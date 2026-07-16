"""Offline test of ai_client._call_model's repair round-trip — no network.

`requests.post` is monkeypatched to return canned responses, so this proves
the retry-on-invalid-JSON logic actually fires a second HTTP call with the
conversation history + a "fix your JSON" message, and that a second failure
correctly gives up with `error_code=INVALID_JSON` rather than looping.
"""

from types import SimpleNamespace

import ai_client
from config import AIModelConfig
from trade_validator import ValidationContext

VALID_JSON_TEXT = """
{
  "signal": "LONG",
  "entry_status": "ENTER_NOW",
  "confidence": 60,
  "market_regime": "TREND_UP",
  "entry": {"type": "LIMIT_ZONE", "from": 1776.80, "to": 1777.20, "trigger": "x"},
  "stop_loss": 1774.80,
  "take_profits": [{"label": "TP1", "price": 1779.20, "close_percent": 100}],
  "time_horizon_minutes": 30,
  "valid_for_minutes": 15,
  "reasons": ["r"],
  "risks": ["r"],
  "invalidation_conditions": ["c"],
  "wait_conditions": [],
  "summary": "s"
}
"""


def _resp(status_code, payload):
    return SimpleNamespace(status_code=status_code, text=str(payload), json=lambda: payload)


def _cfg() -> AIModelConfig:
    return AIModelConfig(label="Test", api_key="key", base_url="https://example.test/v1", model="test-model")


def _ctx() -> ValidationContext:
    return ValidationContext(current_price=1777.0, atr=2.0, spread=0.1)


def test_repair_flow_recovers_from_invalid_json(monkeypatch):
    calls = []

    def fake_post(url, headers=None, json=None, timeout=None):
        calls.append(json)
        if len(calls) == 1:
            return _resp(200, {"choices": [{"message": {"content": "не JSON вообще"}}]})
        return _resp(200, {"choices": [{"message": {"content": VALID_JSON_TEXT}}]})

    monkeypatch.setattr(ai_client.requests, "post", fake_post)

    result = ai_client._call_model(_cfg(), "report text", 10, ai_client.SYSTEM_PROMPTS["scalping"], "scalping", _ctx())

    assert len(calls) == 2
    # Second call must include the first (bad) assistant reply + a repair instruction.
    assert calls[1]["messages"][-2]["role"] == "assistant"
    assert calls[1]["messages"][-1]["role"] == "user"
    assert result.repaired is True
    assert result.trade_plan is not None
    assert result.ok is True


def test_double_invalid_json_gives_up_cleanly(monkeypatch):
    def fake_post(url, headers=None, json=None, timeout=None):
        return _resp(200, {"choices": [{"message": {"content": "всё ещё не JSON"}}]})

    monkeypatch.setattr(ai_client.requests, "post", fake_post)

    result = ai_client._call_model(_cfg(), "report text", 10, ai_client.SYSTEM_PROMPTS["scalping"], "scalping", _ctx())

    assert result.trade_plan is None
    assert result.error_code == "INVALID_JSON"
    assert result.ok is False


def test_valid_first_response_skips_repair(monkeypatch):
    calls = []

    def fake_post(url, headers=None, json=None, timeout=None):
        calls.append(json)
        return _resp(200, {"choices": [{"message": {"content": VALID_JSON_TEXT}}]})

    monkeypatch.setattr(ai_client.requests, "post", fake_post)

    result = ai_client._call_model(_cfg(), "report text", 10, ai_client.SYSTEM_PROMPTS["scalping"], "scalping", _ctx())

    assert len(calls) == 1
    assert result.repaired is False
    assert result.ok is True


def test_auth_error_maps_to_normalized_code_without_leaking_body(monkeypatch):
    def fake_post(url, headers=None, json=None, timeout=None):
        return _resp(403, {"error": {"message": "insufficient_user_quota secret-account-id-123"}})

    monkeypatch.setattr(ai_client.requests, "post", fake_post)

    result = ai_client._call_model(_cfg(), "report text", 10, ai_client.SYSTEM_PROMPTS["scalping"], "scalping", _ctx())

    assert result.error_code == "AI_AUTH_ERROR"
    assert result.ok is False
    # The raw provider body (with the account id) must never reach the user-facing error text.
    assert "secret-account-id-123" not in (result.error or "")
