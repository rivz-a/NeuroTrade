"""Offline tests for prediction_tracker.py — no real network, no AI calls.
`bingx_client.get_klines_range` is monkeypatched so `evaluate_due` never
hits the real exchange; the JSONL log is redirected to a temp file per test
so nothing here touches the real predictions.jsonl.
"""

import time

import pandas as pd
import pytest

import bingx_client
import config
import prediction_tracker as pt
from ai_client import AIAnalysisResult
from ai_schema import EntryZone, RiskReward, TakeProfit, TradePlan


@pytest.fixture(autouse=True)
def _isolated_history_file(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "PREDICTION_HISTORY_FILE", tmp_path / "predictions.jsonl")


def _plan(signal="LONG") -> TradePlan:
    return TradePlan(
        signal=signal,
        entry_status="ENTER_NOW",
        confidence=65,
        market_regime="TREND_UP",
        entry=EntryZone(type="LIMIT_ZONE", from_=100.0, to=101.0, trigger="x"),
        stop_loss=98.0,
        take_profits=[TakeProfit(label="TP1", price=105.0, close_percent=100)],
        risk_reward=RiskReward(tp1=2.0),
        time_horizon_minutes=30,
        valid_for_minutes=30,
        reasons=["r"], risks=["k"], invalidation_conditions=["c"], wait_conditions=[], summary="s",
    )


def _ok_result(label="GPT", plan=None) -> AIAnalysisResult:
    return AIAnalysisResult(
        label=label, model=label.lower(), content="{}", error=None,
        latency_seconds=1.0, created_at=None, trade_plan=plan or _plan(),
    )


def _candles_hitting_tp1() -> pd.DataFrame:
    base = pd.Timestamp("2026-01-01T00:00:00Z")
    return pd.DataFrame(
        [{"time": base, "open": 100.5, "high": 106.0, "low": 99.0, "close": 105.0, "volume": 1.0}]
    )


def test_record_results_stores_plan_fields():
    pt.record_results("scalping", [_ok_result()], price_at_prediction=100.5, symbol="ETH-USDT")
    entries = pt._load_all()
    assert len(entries) == 1
    entry = entries[0]
    assert entry["schema"] == "v2"
    assert entry["symbol"] == "ETH-USDT"
    assert entry["entry_from"] == 100.0
    assert entry["entry_to"] == 101.0
    assert entry["stop_loss"] == 98.0
    assert entry["take_profits"] == [{"label": "TP1", "price": 105.0}]
    assert entry["status"] == "pending"


def test_record_results_skips_result_without_trade_plan():
    no_plan_result = AIAnalysisResult(
        label="Broken", model="broken", content=None, error="AI API отклонил запрос.",
        latency_seconds=1.0, created_at=None, trade_plan=None,
    )
    pt.record_results("scalping", [no_plan_result], price_at_prediction=100.5, symbol="ETH-USDT")
    assert pt._load_all() == []


def test_evaluate_due_scores_pending_prediction_with_mocked_candles(monkeypatch):
    now = 1_000_000.0
    pt.record_results("scalping", [_ok_result()], price_at_prediction=100.5, symbol="ETH-USDT")
    entries = pt._load_all()
    entries[0]["predicted_at"] = now - 3600  # long past the 30-minute horizon
    pt._save_all(entries)

    def fake_get_klines_range(symbol, interval, start_ms, end_ms, limit=1500):
        assert symbol == "ETH-USDT"
        return _candles_hitting_tp1()

    monkeypatch.setattr(bingx_client, "get_klines_range", fake_get_klines_range)

    scored = pt.evaluate_due("ETH-USDT", now=now)

    assert scored == 1
    entry = pt._load_all()[0]
    assert entry["status"] == "evaluated"
    assert entry["exit_reason"] == "TP1"
    assert entry["r_multiple"] > 0


def test_evaluate_due_leaves_entry_pending_on_bingx_error(monkeypatch):
    now = 1_000_000.0
    pt.record_results("scalping", [_ok_result()], price_at_prediction=100.5, symbol="ETH-USDT")
    entries = pt._load_all()
    entries[0]["predicted_at"] = now - 3600
    pt._save_all(entries)

    def raising_get_klines_range(symbol, interval, start_ms, end_ms, limit=1500):
        raise bingx_client.NetworkError("simulated outage")

    monkeypatch.setattr(bingx_client, "get_klines_range", raising_get_klines_range)

    scored = pt.evaluate_due("ETH-USDT", now=now)

    assert scored == 0
    entry = pt._load_all()[0]
    assert entry["status"] == "pending"


def test_evaluate_due_ignores_entries_not_yet_due(monkeypatch):
    pt.record_results("scalping", [_ok_result()], price_at_prediction=100.5, symbol="ETH-USDT")
    # predicted_at is real time.time() from record_results — horizon (30 min) hasn't elapsed yet.

    def unexpected_call(*args, **kwargs):
        raise AssertionError("get_klines_range should not be called for a not-yet-due entry")

    monkeypatch.setattr(bingx_client, "get_klines_range", unexpected_call)

    scored = pt.evaluate_due("ETH-USDT", now=time.time())
    assert scored == 0


def test_wait_signal_is_skipped_not_pending():
    pt.record_results("scalping", [_ok_result(plan=_plan(signal="WAIT"))], price_at_prediction=100.5, symbol="ETH-USDT")
    entry = pt._load_all()[0]
    assert entry["status"] == "skipped"
