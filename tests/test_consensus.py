"""Offline tests for consensus_engine.compute_consensus — pure arithmetic,
no network, no AI calls. Covers every voting branch from the Stage 2 plan
(mirrors product spec section 4) plus the median-based plan aggregation.
"""

from ai_client import AIAnalysisResult
from ai_schema import EntryZone, RiskReward, TakeProfit, TradePlan
from consensus_engine import compute_consensus

NOW = 1_000_000.0


def _plan(
    signal="LONG",
    confidence=60,
    entry_from=100.0,
    entry_to=101.0,
    stop_loss=98.0,
    tp1=105.0,
    valid_for_minutes=30,
) -> TradePlan:
    return TradePlan(
        signal=signal,
        entry_status="ENTER_NOW",
        confidence=confidence,
        market_regime="TREND_UP",
        entry=EntryZone(type="LIMIT_ZONE", from_=entry_from, to=entry_to, trigger="x"),
        stop_loss=stop_loss,
        take_profits=[TakeProfit(label="TP1", price=tp1, close_percent=100)],
        risk_reward=RiskReward(tp1=2.0),
        time_horizon_minutes=30,
        valid_for_minutes=valid_for_minutes,
        reasons=["r"],
        risks=["k"],
        invalidation_conditions=["c"],
        wait_conditions=[],
        summary="s",
    )


def _ok_result(label: str, plan: TradePlan, created_at: float = NOW) -> AIAnalysisResult:
    return AIAnalysisResult(
        label=label, model=label.lower(), content="{}", error=None,
        latency_seconds=1.0, created_at=created_at, trade_plan=plan,
    )


def _rejected_result(label: str, plan: TradePlan) -> AIAnalysisResult:
    return AIAnalysisResult(
        label=label, model=label.lower(), content="{}", error="rejected",
        latency_seconds=1.0, created_at=NOW, trade_plan=plan,
    )


def test_three_agreeing_long_is_strong():
    votes = [_ok_result("A", _plan("LONG")), _ok_result("B", _plan("LONG")), _ok_result("C", _plan("LONG"))]
    result = compute_consensus(votes, "scalping", total_models=3, now=NOW)
    assert result.overall_signal == "LONG"
    assert result.state == "strong"
    assert result.vote_count == 3
    assert result.plan is not None


def test_two_long_one_wait_is_moderate():
    votes = [_ok_result("A", _plan("LONG")), _ok_result("B", _plan("LONG")), _ok_result("C", _plan("WAIT"))]
    result = compute_consensus(votes, "scalping", total_models=3, now=NOW)
    assert result.overall_signal == "LONG"
    assert result.state == "moderate"


def test_two_long_one_short_is_conflict():
    votes = [_ok_result("A", _plan("LONG")), _ok_result("B", _plan("LONG")), _ok_result("C", _plan("SHORT"))]
    result = compute_consensus(votes, "scalping", total_models=3, now=NOW)
    assert result.overall_signal == "WAIT"
    assert result.state == "conflict"


def test_long_and_short_together_is_conflict():
    votes = [_ok_result("A", _plan("LONG")), _ok_result("B", _plan("SHORT")), _ok_result("C", _plan("WAIT"))]
    result = compute_consensus(votes, "scalping", total_models=3, now=NOW)
    assert result.overall_signal == "WAIT"
    assert result.state == "conflict"


def test_all_wait_is_strong_wait_with_no_plan():
    votes = [_ok_result("A", _plan("WAIT")), _ok_result("B", _plan("WAIT")), _ok_result("C", _plan("WAIT"))]
    result = compute_consensus(votes, "scalping", total_models=3, now=NOW)
    assert result.overall_signal == "WAIT"
    assert result.state == "strong"
    assert result.plan is None


def test_no_votes_is_insufficient_data():
    result = compute_consensus([], "scalping", total_models=3, now=NOW)
    assert result.overall_signal == "WAIT"
    assert result.state == "insufficient_data"
    assert result.vote_count == 0


def test_single_vote_is_insufficient_data():
    votes = [_ok_result("A", _plan("LONG"))]
    result = compute_consensus(votes, "scalping", total_models=3, now=NOW)
    assert result.overall_signal == "LONG"
    assert result.state == "insufficient_data"


def test_rejected_vote_does_not_count():
    votes = [
        _ok_result("A", _plan("LONG")),
        _ok_result("B", _plan("LONG")),
        _rejected_result("C", _plan("SHORT")),
    ]
    result = compute_consensus(votes, "scalping", total_models=3, now=NOW)
    assert result.vote_count == 2
    assert result.overall_signal == "LONG"
    assert result.state == "moderate"


def test_stale_vote_does_not_count():
    stale_plan = _plan("SHORT", valid_for_minutes=10)
    votes = [
        _ok_result("A", _plan("LONG")),
        _ok_result("B", _plan("LONG")),
        _ok_result("C", stale_plan, created_at=NOW - 3600),  # 1h old, only valid 10 min
    ]
    result = compute_consensus(votes, "scalping", total_models=3, now=NOW)
    assert result.vote_count == 2
    assert result.overall_signal == "LONG"


def test_aggregated_plan_uses_median_levels():
    votes = [
        _ok_result("A", _plan("LONG", entry_from=100.0, entry_to=101.0, stop_loss=98.0, tp1=105.0), created_at=NOW - 60),
        _ok_result("B", _plan("LONG", entry_from=100.5, entry_to=101.5, stop_loss=98.5, tp1=106.0), created_at=NOW - 30),
        _ok_result("C", _plan("LONG", entry_from=101.0, entry_to=102.0, stop_loss=99.0, tp1=107.0), created_at=NOW),
    ]
    result = compute_consensus(votes, "scalping", total_models=3, now=NOW)
    assert result.plan is not None
    assert result.plan.entry_from == 100.5
    assert result.plan.stop_loss == 98.5
    assert result.plan.take_profits[0] == ("TP1", 106.0)
    # formed_at is the EARLIEST contributing vote — the aggregated plan is
    # only as fresh as its oldest agreeing model.
    assert result.plan.formed_at == NOW - 60


def test_avg_confidence_only_over_agreeing_votes():
    votes = [
        _ok_result("A", _plan("LONG", confidence=60)),
        _ok_result("B", _plan("LONG", confidence=80)),
        _ok_result("C", _plan("WAIT", confidence=50)),
    ]
    result = compute_consensus(votes, "scalping", total_models=3, now=NOW)
    assert result.overall_signal == "LONG"
    assert result.avg_confidence == 70
