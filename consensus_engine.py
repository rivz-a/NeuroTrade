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

Historical-accuracy/regime/pair-aware weighting is intentionally NOT done
here yet — `prediction_tracker.stats_by_model()` currently mixes scalping and
swing stats, and weighting a consensus off an incorrectly-blended number
would be worse than an equal-weight vote. Revisit once that stats source is
segmented by mode.
"""

from __future__ import annotations

import statistics
from collections import Counter
from dataclasses import dataclass
from typing import Literal

from ai_client import AIAnalysisResult
from ai_schema import TradePlan
from signal_freshness import compute_freshness

ConsensusState = Literal["strong", "moderate", "weak", "conflict", "insufficient_data"]


@dataclass(frozen=True)
class AggregatedPlan:
    entry_status: str
    entry_from: float
    entry_to: float
    stop_loss: float
    take_profits: list[tuple[str, float]]
    risk_reward_tp1: float | None
    time_horizon_minutes: int
    valid_for_minutes: int
    # Earliest `created_at` among the contributing votes — the anchor the
    # dashboard uses to show "formed at / age / expires at" for the
    # synthesized plan. Conservative (matches `valid_for_minutes` = MIN):
    # the aggregated plan is only as fresh as its oldest contributor.
    formed_at: float | None


@dataclass(frozen=True)
class ConsensusResult:
    mode: str
    overall_signal: Literal["LONG", "SHORT", "WAIT"]
    state: ConsensusState
    agreement_fraction: float
    vote_count: int
    total_models: int
    avg_confidence: float | None
    plan: AggregatedPlan | None
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


def _aggregate_plan(agreeing: list[AIAnalysisResult]) -> AggregatedPlan | None:
    if not agreeing:
        return None
    agreeing_plans = [r.trade_plan for r in agreeing]

    entry_from = statistics.median(p.entry.from_ for p in agreeing_plans)
    entry_to = statistics.median(p.entry.to for p in agreeing_plans)
    stop_loss = statistics.median(p.stop_loss for p in agreeing_plans)

    tp_by_label: dict[str, list[float]] = {}
    for plan in agreeing_plans:
        for tp in plan.take_profits:
            tp_by_label.setdefault(tp.label, []).append(tp.price)
    min_agreement = min(2, len(agreeing_plans))
    take_profits = [
        (label, statistics.median(prices))
        for label, prices in sorted(tp_by_label.items())
        if len(prices) >= min_agreement
    ]

    entry_mid = (entry_from + entry_to) / 2
    risk = abs(entry_mid - stop_loss)
    tp1_price = take_profits[0][1] if take_profits else None
    risk_reward_tp1 = (abs(tp1_price - entry_mid) / risk) if (tp1_price is not None and risk > 0) else None

    entry_status = Counter(p.entry_status for p in agreeing_plans).most_common(1)[0][0]
    time_horizon_minutes = round(sum(p.time_horizon_minutes for p in agreeing_plans) / len(agreeing_plans))
    valid_for_minutes = min(p.valid_for_minutes for p in agreeing_plans)
    created_ats = [r.created_at for r in agreeing if r.created_at is not None]
    formed_at = min(created_ats) if created_ats else None

    return AggregatedPlan(
        entry_status=entry_status,
        entry_from=entry_from,
        entry_to=entry_to,
        stop_loss=stop_loss,
        take_profits=take_profits,
        risk_reward_tp1=risk_reward_tp1,
        time_horizon_minutes=time_horizon_minutes,
        valid_for_minutes=valid_for_minutes,
        formed_at=formed_at,
    )


def compute_consensus(
    results: list[AIAnalysisResult], mode: str, total_models: int, now: float | None = None
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
    agreement_fraction = (len(agreeing) / vote_count) if vote_count else 0.0

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

    plan = _aggregate_plan(agreeing) if overall_signal in ("LONG", "SHORT") else None

    return ConsensusResult(
        mode=mode,
        overall_signal=overall_signal,
        state=state,
        agreement_fraction=agreement_fraction,
        vote_count=vote_count,
        total_models=total_models,
        avg_confidence=avg_confidence,
        plan=plan,
        reasons=reasons,
        risks=risks,
        wait_or_invalidation=wait_or_invalidation,
    )
