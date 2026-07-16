"""Offline tests for consensus_engine.compute_consensus — pure arithmetic,
no network, no AI calls. Covers every voting branch (mirrors product spec
section 4), BEST_VALID_MODEL plan selection (no cross-model blending), and
trade_permission (separate from direction consensus and from plan validity).
"""

from ai_client import AIAnalysisResult
from ai_schema import EntryZone, TakeProfit, TradePlan
from consensus_engine import _compute_trade_permission, compute_consensus
from trade_validator import ValidationResult

NOW = 1_000_000.0


def _plan(
    signal="LONG",
    confidence=60,
    entry_from=100.0,
    entry_to=101.0,
    stop_loss=98.0,
    tp1=105.0,
    valid_for_minutes=30,
    entry_status="ENTER_NOW",
    contradictions=None,
    missing_context=None,
) -> TradePlan:
    return TradePlan(
        signal=signal,
        entry_status=entry_status,
        confidence=confidence,
        market_regime="TREND_UP",
        entry=EntryZone(type="LIMIT_ZONE", from_=entry_from, to=entry_to, trigger="x"),
        stop_loss=stop_loss,
        take_profits=[TakeProfit(label="TP1", price=tp1, close_percent=100)],
        time_horizon_minutes=30,
        valid_for_minutes=valid_for_minutes,
        reasons=["r"],
        risks=["k"],
        invalidation_conditions=["c"],
        wait_conditions=[],
        contradictions=contradictions or [],
        missing_context=missing_context or [],
        summary="s",
    )


def _ok_result(label: str, plan: TradePlan, created_at: float = NOW, validation: ValidationResult | None = None) -> AIAnalysisResult:
    return AIAnalysisResult(
        label=label, model=label.lower(), content="{}", error=None,
        latency_seconds=1.0, created_at=created_at, trade_plan=plan, validation=validation,
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


def test_best_valid_model_is_selected_intact():
    """The chosen plan's levels must match ONE contributing model exactly —
    never a statistics.median() blend of entry/stop/TP across models.
    """
    plan_a = _plan("LONG", entry_from=100.0, entry_to=101.0, stop_loss=98.0, tp1=105.0)
    plan_b = _plan("LONG", entry_from=100.5, entry_to=101.5, stop_loss=98.5, tp1=106.0)
    plan_c = _plan("LONG", entry_from=101.0, entry_to=102.0, stop_loss=99.0, tp1=107.0)
    votes = [_ok_result("A", plan_a), _ok_result("B", plan_b), _ok_result("C", plan_c)]
    result = compute_consensus(votes, "scalping", total_models=3, now=NOW)
    assert result.plan is not None
    chosen = (result.plan.entry_from, result.plan.entry_to, result.plan.stop_loss, result.plan.take_profits[0][1])
    candidates = {
        (plan_a.entry.from_, plan_a.entry.to, plan_a.stop_loss, plan_a.take_profits[0].price),
        (plan_b.entry.from_, plan_b.entry.to, plan_b.stop_loss, plan_b.take_profits[0].price),
        (plan_c.entry.from_, plan_c.entry.to, plan_c.stop_loss, plan_c.take_profits[0].price),
    }
    assert chosen in candidates, "chosen plan must be one real model's plan intact, not a blend"
    assert result.plan.source_label in ("A", "B", "C")


def test_best_valid_model_prefers_higher_gross_rr_on_tie_confidence():
    # Same confidence for both — B's TP is farther away (better R:R), should win.
    plan_a = _plan("LONG", confidence=70, entry_from=100.0, entry_to=100.0, stop_loss=99.0, tp1=101.0)  # R:R = 1
    plan_b = _plan("LONG", confidence=70, entry_from=100.0, entry_to=100.0, stop_loss=99.0, tp1=104.0)  # R:R = 4
    votes = [_ok_result("A", plan_a), _ok_result("B", plan_b)]
    result = compute_consensus(votes, "scalping", total_models=3, now=NOW)
    assert result.plan.source_label == "B"


def test_agreeing_count_distinguishes_direction_from_total_valid_votes():
    # 2 LONG + 1 WAIT: 3 valid votes total, but only 2 agree with the winning direction.
    votes = [_ok_result("A", _plan("LONG")), _ok_result("B", _plan("LONG")), _ok_result("C", _plan("WAIT"))]
    result = compute_consensus(votes, "scalping", total_models=3, now=NOW)
    assert result.vote_count == 3
    assert result.agreeing_count == 2
    assert result.agreeing_count != result.vote_count


def test_avg_confidence_only_over_agreeing_votes():
    votes = [
        _ok_result("A", _plan("LONG", confidence=60)),
        _ok_result("B", _plan("LONG", confidence=80)),
        _ok_result("C", _plan("WAIT", confidence=50)),
    ]
    result = compute_consensus(votes, "scalping", total_models=3, now=NOW)
    assert result.overall_signal == "LONG"
    assert result.avg_confidence == 70


def test_trade_permission_allowed_when_enter_now_and_price_in_zone():
    votes = [
        _ok_result("A", _plan("LONG", entry_from=100.0, entry_to=101.0)),
        _ok_result("B", _plan("LONG", entry_from=100.0, entry_to=101.0)),
    ]
    result = compute_consensus(votes, "scalping", total_models=3, now=NOW, current_price=100.5)
    assert result.trade_permission == "ALLOWED"


def test_trade_permission_waiting_trigger_for_wait_pullback():
    votes = [
        _ok_result("A", _plan("LONG", entry_status="WAIT_PULLBACK")),
        _ok_result("B", _plan("LONG", entry_status="WAIT_PULLBACK")),
    ]
    result = compute_consensus(votes, "scalping", total_models=3, now=NOW, current_price=100.5)
    assert result.trade_permission == "WAITING_TRIGGER"


def test_trade_permission_price_outside_entry_zone():
    votes = [
        _ok_result("A", _plan("LONG", entry_from=100.0, entry_to=101.0)),
        _ok_result("B", _plan("LONG", entry_from=100.0, entry_to=101.0)),
    ]
    result = compute_consensus(votes, "scalping", total_models=3, now=NOW, current_price=110.0)
    assert result.trade_permission == "PRICE_OUTSIDE_ENTRY_ZONE"
    assert "выше зоны" in result.trade_permission_reason


def test_trade_permission_wait_when_consensus_is_wait():
    votes = [_ok_result("A", _plan("WAIT")), _ok_result("B", _plan("WAIT"))]
    result = compute_consensus(votes, "scalping", total_models=3, now=NOW, current_price=100.5)
    assert result.trade_permission == "WAIT"


def test_trade_permission_no_price_given_skips_zone_check():
    votes = [
        _ok_result("A", _plan("LONG", entry_from=100.0, entry_to=101.0)),
        _ok_result("B", _plan("LONG", entry_from=100.0, entry_to=101.0)),
    ]
    result = compute_consensus(votes, "scalping", total_models=3, now=NOW, current_price=None)
    assert result.trade_permission == "ALLOWED"


def test_compute_trade_permission_invalid_plan_when_no_candidate():
    # Direct unit test of the pure helper — compute_consensus itself never
    # reaches this combination (overall_signal LONG/SHORT implies a
    # non-empty `agreeing` list), but the function must still handle it
    # defensively rather than crash.
    permission, reason = _compute_trade_permission("LONG", None, 100.0)
    assert permission == "INVALID_PLAN"
    assert reason


def test_plan_with_contradictions_and_missing_context_still_wins_consensus():
    # contradictions/missing_context are advisory-only text fields (not yet
    # surfaced on SelectedPlan/the dashboard — out of scope for this stage);
    # this only guards that carrying them on a TradePlan doesn't change
    # voting/selection behaviour versus an otherwise-identical plan.
    plan_a = _plan(
        "LONG",
        contradictions=["Модель видит RANGE, хотя передан TREND_UP"],
        missing_context=["Нет данных по ликвидациям"],
    )
    plan_b = _plan("LONG")
    votes = [_ok_result("A", plan_a), _ok_result("B", plan_b)]
    result = compute_consensus(votes, "scalping", total_models=3, now=NOW)
    assert result.overall_signal == "LONG"
    assert result.state == "moderate"
    assert result.plan is not None
    assert result.plan.source_label in ("A", "B")
