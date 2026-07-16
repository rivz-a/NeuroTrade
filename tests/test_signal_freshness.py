"""Offline tests for signal_freshness — no network, no AI calls."""

from ai_client import AIAnalysisResult
from ai_schema import EntryZone, TakeProfit, TradePlan
from signal_freshness import compute_freshness, format_age, format_remaining

NOW = 1_000_000.0


def _plan(valid_for_minutes: int = 30) -> TradePlan:
    return TradePlan(
        signal="LONG",
        entry_status="ENTER_NOW",
        confidence=60,
        market_regime="TREND_UP",
        entry=EntryZone(type="LIMIT_ZONE", from_=100.0, to=101.0, trigger="x"),
        stop_loss=98.0,
        take_profits=[TakeProfit(label="TP1", price=105.0, close_percent=100)],
        time_horizon_minutes=30,
        valid_for_minutes=valid_for_minutes,
        reasons=["r"],
        risks=["r"],
        invalidation_conditions=["c"],
        wait_conditions=[],
        summary="s",
    )


def _result(created_at: float | None, plan: TradePlan | None) -> AIAnalysisResult:
    return AIAnalysisResult(
        label="Test",
        model="test-model",
        content="{}",
        error=None,
        latency_seconds=1.0,
        created_at=created_at,
        trade_plan=plan,
    )


def test_freshly_created_is_not_stale():
    result = _result(NOW, _plan(30))
    fresh = compute_freshness(result, now=NOW)
    assert fresh.is_stale is False
    assert fresh.age_seconds == 0
    assert fresh.seconds_remaining == 30 * 60


def test_half_expired_is_not_stale():
    result = _result(NOW - 900, _plan(30))
    fresh = compute_freshness(result, now=NOW)
    assert fresh.is_stale is False
    assert fresh.seconds_remaining == 900


def test_exactly_at_expiry_is_stale():
    result = _result(NOW - 1800, _plan(30))
    fresh = compute_freshness(result, now=NOW)
    assert fresh.is_stale is True
    assert fresh.seconds_remaining == 0


def test_long_expired_is_stale():
    result = _result(NOW - 3600, _plan(30))
    fresh = compute_freshness(result, now=NOW)
    assert fresh.is_stale is True
    assert fresh.seconds_remaining < 0


def test_no_created_at_returns_none():
    result = _result(None, _plan(30))
    assert compute_freshness(result, now=NOW) is None


def test_no_trade_plan_returns_none():
    result = _result(NOW, None)
    assert compute_freshness(result, now=NOW) is None


def test_format_age_seconds():
    assert format_age(30) == "30 с назад"


def test_format_age_minutes():
    assert format_age(125) == "2 мин назад"


def test_format_age_hours():
    assert format_age(5400) == "1.5 ч назад"


def test_format_age_none():
    assert format_age(None) == "н/д"


def test_format_remaining_minutes():
    assert format_remaining(840) == "осталось 14 мин"


def test_format_remaining_expired():
    assert format_remaining(-120) == "истёк 2 мин назад"
