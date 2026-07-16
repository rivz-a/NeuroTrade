"""Dynamic per-model vote weighting, derived from prediction_tracker.py's
predictions.jsonl — the only existing source with a genuine,
independently-scored outcome per AI model, not just whichever model's plan
happened to win consensus. (journal_db.py's trade_outcomes is 1:1 with the
single consensus-selected trade_plan per signal, not per model — a
losing model's own prediction has no outcome row of its own there.)

Standalone, tested module — NOT wired into consensus_engine.py's live
voting this stage. Two exact plug-in points for a future integration
stage, left untouched here:
  - consensus_engine._decide()'s `Counter(votes)` (a plain unweighted
    frequency count) -> would become a weighted sum keyed by effective_weight.
  - consensus_engine._plan_score()'s 4-tuple
    `(status_weight, rr_key, confidence, -original_index)` -> would gain a
    weight term (most naturally between rr_key and confidence).

Formula:
    effective_weight = smoothed_base_weight * confidence_factor
    smoothed_base_weight = NEUTRAL + (base_weight - NEUTRAL) * sample_size_factor

- base_weight: derived from the matched bucket's historical expectancy_r
  (mean R-multiple) via a clamped linear map — a break-even model
  (expectancy_r=0) sits at config.MODEL_WEIGHT_NEUTRAL; a profitable model
  is pulled toward config.MODEL_WEIGHT_MAX, an unprofitable one toward
  config.MODEL_WEIGHT_MIN.
- confidence_factor: the CURRENT vote's own self-reported confidence/100
  (not a historical calibration curve) — trusts the model's stated
  certainty for THIS specific call, layered on top of its historical base
  rate.
- sample_size_factor: a 0..1 ramp between config.MODEL_WEIGHT_MIN_SAMPLE
  and config.MODEL_WEIGHT_FULL_SAMPLE that shrinks base_weight toward
  NEUTRAL when the matched bucket's history is thin.

"Не использовать вес, если мало данных" (don't use the weight when there's
too little data) is implemented as a `None` return, not a weight of 0.0 —
effective_weight() returns None when even the coarsest bucket (model +
mode alone) hasn't reached config.MODEL_WEIGHT_MIN_SAMPLE evaluated
samples. Callers MUST treat None as "don't apply weighting here" (fall
back to an unweighted/neutral multiplier), never as "zero out this
model's vote" — a literal 0.0 would silently delete the vote, which is a
different thing from "we don't know yet."

Bucket resolution is a hierarchical fallback (see resolve_bucket) — the
user's own example weights are scoped to only 1-2 dimensions at a time
("Gemini: TREND_DOWN+SWING=0.86", "GPT: VOLATILITY_EXPANSION=0.54"), which
only makes sense if the system tries the most specific combination first
and progressively widens the bucket until enough samples exist, rather
than requiring all 7 requested dimensions to match simultaneously (which
would almost never accumulate enough samples in practice).

Known limitation, not introduced by this module: there is no true "model
version" field anywhere in this app — a provider silently upgrading a
model behind the same id string (e.g. "gpt-4o-mini") is undetectable; the
`model` string is the closest available proxy and is what "model version"
means here.
"""

from __future__ import annotations

import typing
from dataclasses import dataclass

import ai_schema
import config
import prediction_tracker

_ALL_REGIMES: tuple[str, ...] = typing.get_args(ai_schema.MarketRegime)

# Dropped from the filter set in this order (front first) as the bucket
# widens. "mode" is never dropped (mixing scalping/swing is already a
# named problem elsewhere in this codebase); "model" is always the fixed
# subject being weighted, not a droppable filter.
_DIMENSION_DROP_ORDER = ("prompt_version", "confidence_bucket", "symbol", "signal", "market_regime")


@dataclass(frozen=True)
class BucketSample:
    n: int
    expectancy_r: float | None
    win_rate: float | None
    dimensions_used: tuple[str, ...]


def _confidence_bucket(confidence: int | None) -> str | None:
    if confidence is None:
        return None
    for upper, label in config.CONFIDENCE_BUCKETS:
        if confidence <= upper:
            return label
    return config.CONFIDENCE_BUCKETS[-1][1]


def _load_evaluated_entries(model: str) -> list[dict]:
    """Reuses prediction_tracker's own JSONL loader directly (same
    precedent as backtest_engine.py importing market_data_engine's private
    _interval_seconds in Stage 9) rather than re-implementing file
    parsing.
    """
    entries = prediction_tracker._load_all()
    return [
        e
        for e in entries
        if e.get("model") == model
        and e.get("schema") == "v2"
        and e.get("status") == "evaluated"
        and e.get("r_multiple") is not None
    ]


def _entry_matches(entry: dict, filters: dict) -> bool:
    for key, value in filters.items():
        if key == "confidence_bucket":
            if _confidence_bucket(entry.get("confidence")) != value:
                return False
        else:
            if entry.get(key) != value:
                return False
    return True


def _aggregate(entries: list[dict]) -> tuple[int, float | None, float | None]:
    r_values = [e["r_multiple"] for e in entries]
    n = len(r_values)
    if n == 0:
        return 0, None, None
    expectancy_r = sum(r_values) / n
    win_rate = sum(1 for r in r_values if r > 0) / n * 100
    return n, expectancy_r, win_rate


def resolve_bucket(
    model: str,
    *,
    mode: str,
    regime: str | None = None,
    symbol: str | None = None,
    direction: str | None = None,
    confidence: int | None = None,
    prompt_version: str | None = None,
) -> BucketSample | None:
    """Tries filter sets from most specific to least, dropping one
    dimension at a time per _DIMENSION_DROP_ORDER, until the sample size
    reaches config.MODEL_WEIGHT_MIN_SAMPLE. Returns None if even
    "model + mode" alone doesn't reach the minimum.
    """
    entries = _load_evaluated_entries(model)

    optional_filters = [
        ("prompt_version", prompt_version),
        ("confidence_bucket", _confidence_bucket(confidence)),
        ("symbol", symbol),
        ("signal", direction),
        ("market_regime", regime),
    ]
    active = [(k, v) for k, v in optional_filters if v is not None]

    for drop_count in range(len(active) + 1):
        remaining = active[drop_count:]
        filters = {"mode": mode, **dict(remaining)}
        matched = [e for e in entries if _entry_matches(e, filters)]
        n, expectancy_r, win_rate = _aggregate(matched)
        if n >= config.MODEL_WEIGHT_MIN_SAMPLE:
            return BucketSample(
                n=n, expectancy_r=expectancy_r, win_rate=win_rate,
                dimensions_used=("mode",) + tuple(k for k, _ in remaining),
            )
    return None


def _base_weight(expectancy_r: float) -> float:
    raw = config.MODEL_WEIGHT_NEUTRAL + expectancy_r * config.MODEL_WEIGHT_EXPECTANCY_SCALE
    return max(config.MODEL_WEIGHT_MIN, min(config.MODEL_WEIGHT_MAX, raw))


def _sample_size_factor(n: int) -> float:
    min_n, full_n = config.MODEL_WEIGHT_MIN_SAMPLE, config.MODEL_WEIGHT_FULL_SAMPLE
    if full_n <= min_n or n >= full_n:
        return 1.0
    return max(0.0, (n - min_n) / (full_n - min_n))


def _smoothed_base_weight(bucket: BucketSample) -> float | None:
    if bucket.expectancy_r is None:
        return None
    base = _base_weight(bucket.expectancy_r)
    factor = _sample_size_factor(bucket.n)
    return config.MODEL_WEIGHT_NEUTRAL + (base - config.MODEL_WEIGHT_NEUTRAL) * factor


def effective_weight(
    model: str,
    *,
    mode: str,
    confidence: int,
    regime: str | None = None,
    symbol: str | None = None,
    direction: str | None = None,
    prompt_version: str | None = None,
) -> float | None:
    """None means "not enough history to weight this vote" — callers must
    fall back to a neutral/unweighted multiplier, never treat None as 0.0.
    """
    bucket = resolve_bucket(
        model, mode=mode, regime=regime, symbol=symbol, direction=direction,
        confidence=confidence, prompt_version=prompt_version,
    )
    if bucket is None:
        return None
    smoothed = _smoothed_base_weight(bucket)
    if smoothed is None:
        return None
    return smoothed * (confidence / 100)


def bucket_report(
    models: list[str], *, modes: tuple[str, ...] = ("scalping", "swing"), regimes: tuple[str | None, ...] | None = None
) -> list[dict]:
    """Inspection/debugging table matching the user's own example shape
    (e.g. "Gemini: TREND_DOWN+SWING=0.86") — reports the BASE weight
    (before any single vote's confidence_factor, which only applies at
    effective_weight() call time for a specific live vote) per
    (model, mode, regime) combination that has enough history.
    """
    regime_values = regimes if regimes is not None else (None,) + _ALL_REGIMES
    rows: list[dict] = []
    for model in models:
        for mode in modes:
            for regime in regime_values:
                bucket = resolve_bucket(model, mode=mode, regime=regime)
                if bucket is None:
                    continue
                rows.append(
                    {
                        "model": model,
                        "mode": mode,
                        "regime": regime,
                        "n": bucket.n,
                        "expectancy_r": bucket.expectancy_r,
                        "win_rate": bucket.win_rate,
                        "dimensions_used": bucket.dimensions_used,
                        "base_weight": _smoothed_base_weight(bucket),
                    }
                )
    return rows
