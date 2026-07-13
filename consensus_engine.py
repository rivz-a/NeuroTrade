"""Deterministic, code-only consensus across a mode's configured models.

No AI call decides the consensus — it's arithmetic over the already-parsed,
already-validated `TradePlan`s (see `ai_schema.py`, `trade_validator.py`),
same spirit as `trade_validator.py`: numbers a human could check by hand.

Voting rules (only `result.ok` AND non-stale results get a vote — see
`signal_freshness.py`):
- 0 votes                                  -> WAIT, "insufficient_data"
- all votes agree (incl. all-WAIT)         -> "strong" (>=3 votes),
                                               "moderate" (2), "insufficient_data" (1)
- LONG and SHORT both present among votes  -> WAIT, "conflict"
  (this merges the spec's separate "2 vs 1 opposite" and "LONG+SHORT at
  once" cases into one "conflict" state — see the Stage 2 plan's notes)
- one direction is a strict majority       -> that direction, "moderate"
- one direction ties with the rest         -> that direction, "weak"
- otherwise (no majority, no clean tie)    -> WAIT, "weak"

Direction agreement ("how many voters picked the winning direction") is a
DIFFERENT number from the total vote count ("how many models produced any
valid, fresh plan at all") — a 2-LONG/1-WAIT split is "moderate" with 2
agreeing out of 3 valid votes, not "3 of 3". Both numbers are exposed on
`ConsensusResult` (`agreeing_count` vs `vote_count`) so the dashboard never
has to imply unanimous agreement when it wasn't.

The trade plan shown for a LONG/SHORT consensus is now the intact plan of
ONE real model (`_select_best_plan`, "BEST_VALID_MODEL") — never a
statistics.median() blend of entry/stop/TP across different models' plans.
Blending would synthesize a plan no model actually proposed and whose R:R
was never independently re-validated; picking one real (already-validated)
model's plan sidesteps that whole class of bug. Which model was picked is
exposed as `SelectedPlan.source_label` so the UI can disclose it.

`trade_permission` answers "is showing an 'open LONG/SHORT' instruction to
the user actually warranted right now" — separate from `overall_signal`
(direction) and from the plan's own validity (already gated before a result
can vote at all). See `_compute_trade_permission`.

Historical-accuracy/regime/pair-aware weighting is intentionally NOT done
here yet — `prediction_tracker.stats_by_model()` currently mixes scalping and
swing stats, and weighting a consensus off an incorrectly-blended number
would be worse than an equal-weight vote. Revisit once that stats source is
segmented by mode.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Literal

from ai_client import AIAnalysisResult
from ai_schema import TradePlan
from signal_freshness import compute_freshness

ConsensusState = Literal["strong", "moderate", "weak", "conflict", "insufficient_data"]

# Whether it's warranted to tell the user "open LONG/SHORT now". EXPIRED is
# declared for forward use by a future live (client-side) recheck — a stale
# result can never reach `_select_best_plan` in the first place (filtered by
# `signal_freshness` before voting), so `compute_consensus` never produces it.
TradePermission = Literal[
    "ALLOWED",
    "NOT_ALLOWED",
    "EXPIRED",
    "PRICE_OUTSIDE_ENTRY_ZONE",
    "WAITING_TRIGGER",
    "INVALID_PLAN",
    "WAIT",
]

# How far price may drift from the entry zone before it still counts as
# "in the zone" — absorbs quote noise/rounding, not a real tolerance for
# chasing price.
_PRICE_ZONE_TOLERANCE_FRACTION = 0.0005  # 0.05% of the zone's own price level

_ENTRY_STATUS_WAIT_REASONS = {
    "WAIT_PULLBACK": "ожидается откат в зону входа",
    "WAIT_BREAKOUT": "ожидается пробой уровня",
    "WAIT_CONFIRMATION": "ожидается подтверждение объёмом",
    "LATE_ENTRY": "вход считается поздним",
    "REJECTED": "вход отклонён валидатором",
    "NO_TRADE": "сделки нет",
}


@dataclass(frozen=True)
class SelectedPlan:
    source_label: str
    entry_status: str
    entry_from: float
    entry_to: float
    stop_loss: float
    take_profits: list[tuple[str, float]]
    risk_reward_tp1: float | None
    time_horizon_minutes: int
    valid_for_minutes: int
    # The chosen model's own `created_at` — the anchor the dashboard uses to
    # show "formed at / age / expires at" for this plan.
    formed_at: float | None


@dataclass(frozen=True)
class ConsensusResult:
    mode: str
    overall_signal: Literal["LONG", "SHORT", "WAIT"]
    state: ConsensusState
    agreement_fraction: float
    agreeing_count: int
    vote_count: int
    total_models: int
    avg_confidence: float | None
    plan: SelectedPlan | None
    trade_permission: TradePermission
    trade_permission_reason: str
    reasons: list[str]
    risks: list[str]
    wait_or_invalidation: list[str]


def _collect_unique(items_lists, limit: int = 5) -> list[str]:
    seen: list[str] = []
    for items in items_lists:
        for item in items:
            if item not in seen:
                seen.append(item)
    return seen[:limit]


def _decide(votes: list[str], total_valid: int) -> tuple[str, ConsensusState]:
    if total_valid == 0:
        return "WAIT", "insufficient_data"

    counts = Counter(votes)
    has_long = counts.get("LONG", 0) > 0
    has_short = counts.get("SHORT", 0) > 0

    if has_long and has_short:
        return "WAIT", "conflict"

    top_signal, top_count = counts.most_common(1)[0]

    if len(counts) == 1:
        if total_valid >= 3:
            return top_signal, "strong"
        if total_valid == 2:
            return top_signal, "moderate"
        return top_signal, "insufficient_data"

    if top_signal != "WAIT" and top_count * 2 > total_valid:
        return top_signal, "moderate"
    if top_signal != "WAIT" and top_count * 2 == total_valid:
        return top_signal, "weak"
    return "WAIT", "weak"


def _gross_rr_to_tp1(plan: TradePlan) -> float | None:
    if not plan.take_profits:
        return None
    entry_mid = (plan.entry.from_ + plan.entry.to) / 2
    risk = abs(entry_mid - plan.stop_loss)
    if risk <= 0:
        return None
    reward = abs(plan.take_profits[0].price - entry_mid)
    return reward / risk


def _plan_score(result: AIAnalysisResult, original_index: int) -> tuple:
    """Ranking key for BEST_VALID_MODEL — all candidates already passed
    `trade_validator` (that's what makes them eligible to vote at all), so
    this only breaks ties: a clean "valid" beats a "warning", higher R:R
    wins, then higher confidence, then earliest position in `config.AI_MODELS`
    (via `-original_index`, since the input list is already in that order) —
    fully deterministic, no randomness, no dict-ordering surprises.
    """
    status_weight = 1 if (result.validation and result.validation.status == "valid") else 0
    rr = _gross_rr_to_tp1(result.trade_plan)
    rr_key = rr if rr is not None else -1.0
    return (status_weight, rr_key, result.trade_plan.confidence, -original_index)


def _select_best_plan(agreeing: list[AIAnalysisResult]) -> SelectedPlan | None:
    if not agreeing:
        return None
    _, best_result = max(enumerate(agreeing), key=lambda pair: _plan_score(pair[1], pair[0]))
    plan = best_result.trade_plan
    return SelectedPlan(
        source_label=best_result.label,
        entry_status=plan.entry_status,
        entry_from=plan.entry.from_,
        entry_to=plan.entry.to,
        stop_loss=plan.stop_loss,
        take_profits=[(tp.label, tp.price) for tp in plan.take_profits],
        risk_reward_tp1=_gross_rr_to_tp1(plan),
        time_horizon_minutes=plan.time_horizon_minutes,
        valid_for_minutes=plan.valid_for_minutes,
        formed_at=best_result.created_at,
    )


def _compute_trade_permission(
    overall_signal: str, plan: SelectedPlan | None, current_price: float | None
) -> tuple[TradePermission, str]:
    if overall_signal == "WAIT":
        return "WAIT", "Консенсус моделей — WAIT, нет направленной идеи для входа."

    if plan is None:
        return "INVALID_PLAN", "Нет ни одного валидного согласованного торгового плана."

    if plan.entry_status != "ENTER_NOW":
        reason = _ENTRY_STATUS_WAIT_REASONS.get(plan.entry_status, "условие входа ещё не выполнено")
        return "WAITING_TRIGGER", f"Направление определено, но {reason}."

    if current_price is not None:
        lo, hi = min(plan.entry_from, plan.entry_to), max(plan.entry_from, plan.entry_to)
        tolerance = hi * _PRICE_ZONE_TOLERANCE_FRACTION
        if current_price < lo - tolerance:
            return "PRICE_OUTSIDE_ENTRY_ZONE", f"Цена ниже зоны входа на {lo - current_price:.2f}."
        if current_price > hi + tolerance:
            return "PRICE_OUTSIDE_ENTRY_ZONE", f"Цена выше зоны входа на {current_price - hi:.2f}."

    return "ALLOWED", "План прошёл валидацию, статус входа ENTER_NOW, цена в зоне входа."


def compute_consensus(
    results: list[AIAnalysisResult],
    mode: str,
    total_models: int,
    now: float | None = None,
    current_price: float | None = None,
) -> ConsensusResult:
    live_votes: list[AIAnalysisResult] = []
    for result in results:
        if not result.ok:
            continue
        freshness = compute_freshness(result, now)
        if freshness is not None and freshness.is_stale:
            continue
        live_votes.append(result)

    vote_count = len(live_votes)
    overall_signal, state = _decide([r.trade_plan.signal for r in live_votes], vote_count)

    agreeing = [r for r in live_votes if r.trade_plan.signal == overall_signal]
    agreeing_count = len(agreeing)
    agreement_fraction = (agreeing_count / vote_count) if vote_count else 0.0

    avg_confidence = (
        sum(r.trade_plan.confidence for r in agreeing) / len(agreeing) if agreeing else None
    )

    reasons_source = agreeing or live_votes
    risks_source = agreeing or live_votes
    reasons = _collect_unique(r.trade_plan.reasons for r in reasons_source)
    risks = _collect_unique(r.trade_plan.risks for r in risks_source)

    if overall_signal == "WAIT":
        wait_source = agreeing or live_votes
        wait_or_invalidation = _collect_unique(
            list(r.trade_plan.wait_conditions) + list(r.trade_plan.invalidation_conditions)
            for r in wait_source
        )
    else:
        wait_or_invalidation = []

    plan = _select_best_plan(agreeing) if overall_signal in ("LONG", "SHORT") else None
    trade_permission, trade_permission_reason = _compute_trade_permission(overall_signal, plan, current_price)

    return ConsensusResult(
        mode=mode,
        overall_signal=overall_signal,
        state=state,
        agreement_fraction=agreement_fraction,
        agreeing_count=agreeing_count,
        vote_count=vote_count,
        total_models=total_models,
        avg_confidence=avg_confidence,
        plan=plan,
        trade_permission=trade_permission,
        trade_permission_reason=trade_permission_reason,
        reasons=reasons,
        risks=risks,
        wait_or_invalidation=wait_or_invalidation,
    )
