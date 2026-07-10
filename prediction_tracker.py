"""Tracks each AI model's directional call over time and scores it once its
horizon has elapsed, so the dashboard can show which model is actually
right more often — not just which one sounds more confident.

Storage is a local JSON Lines log (`config.PREDICTION_HISTORY_FILE`), one
entry per (mode, model) prediction. Scoring is directional-only: a LONG call
is a "win" if price is higher than it was at prediction time once the
mode's horizon (`config.PREDICTION_HORIZON_SECONDS`) has passed, a "loss" if
lower — it does not simulate whether stop loss/take profit would have been
hit first, since that needs continuous price monitoring this app doesn't do.
WAIT calls are not scored (there's no directional bet to grade) but are
still counted so the UI can show how often each model says WAIT.

Best-effort throughout: a read/write failure here should never break a
snapshot fetch or an AI call.
"""

from __future__ import annotations

import json
import time
import uuid

import config
from ai_client import AIAnalysisResult
from verdict import detect_signal


def _load_all() -> list[dict]:
    try:
        with open(config.PREDICTION_HISTORY_FILE, encoding="utf-8") as f:
            lines = f.readlines()
    except OSError:
        return []
    entries = []
    for line in lines:
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return entries


def _save_all(entries: list[dict]) -> None:
    trimmed = entries[-config.PREDICTION_HISTORY_MAX_ENTRIES :]
    try:
        with open(config.PREDICTION_HISTORY_FILE, "w", encoding="utf-8") as f:
            for entry in trimmed:
                f.write(json.dumps(entry) + "\n")
    except OSError:
        return


def record_results(mode: str, results: list[AIAnalysisResult], price_at_prediction: float | None) -> None:
    """Log one entry per successful (ok) result. Errors aren't scoreable calls."""
    if price_at_prediction is None:
        return
    horizon = config.PREDICTION_HORIZON_SECONDS.get(mode, config.PREDICTION_HORIZON_SECONDS["scalping"])
    now = time.time()
    entries = _load_all()
    for result in results:
        if not result.ok:
            continue
        signal = detect_signal(result.content)
        entries.append(
            {
                "id": uuid.uuid4().hex,
                "mode": mode,
                "label": result.label,
                "model": result.model,
                "signal": signal,
                "predicted_at": now,
                "price_at_prediction": price_at_prediction,
                "horizon_seconds": horizon,
                "status": "pending" if signal in ("LONG", "SHORT") else "skipped",
                "outcome": None,
                "evaluated_at": None,
                "price_at_evaluation": None,
            }
        )
    _save_all(entries)


def evaluate_due(current_price: float | None, now: float | None = None) -> int:
    """Score every pending prediction whose horizon has elapsed. Returns how
    many were scored. Safe to call on every snapshot fetch — a no-op when
    nothing is due yet.
    """
    if current_price is None:
        return 0
    now = now if now is not None else time.time()
    entries = _load_all()
    scored = 0
    for entry in entries:
        if entry.get("status") != "pending":
            continue
        deadline = entry["predicted_at"] + entry["horizon_seconds"]
        if now < deadline:
            continue
        direction = 1 if entry["signal"] == "LONG" else -1
        diff = (current_price - entry["price_at_prediction"]) * direction
        outcome = "win" if diff > 0 else ("loss" if diff < 0 else "flat")
        entry["status"] = "evaluated"
        entry["outcome"] = outcome
        entry["evaluated_at"] = now
        entry["price_at_evaluation"] = current_price
        scored += 1
    if scored:
        _save_all(entries)
    return scored


def stats_by_model() -> dict[str, dict]:
    """Aggregate win/loss/pending/skipped counts per model label, across
    both modes — used by the dashboard to show a compact accuracy summary.
    """
    entries = _load_all()
    stats: dict[str, dict] = {}
    for entry in entries:
        label = entry["label"]
        s = stats.setdefault(
            label, {"win": 0, "loss": 0, "flat": 0, "pending": 0, "skipped": 0, "total": 0}
        )
        s["total"] += 1
        status = entry.get("status")
        if status == "pending":
            s["pending"] += 1
        elif status == "skipped":
            s["skipped"] += 1
        elif status == "evaluated":
            outcome = entry.get("outcome")
            if outcome in ("win", "loss", "flat"):
                s[outcome] += 1

    for s in stats.values():
        scored = s["win"] + s["loss"] + s["flat"]
        s["scored"] = scored
        s["win_rate"] = (s["win"] / scored * 100) if scored else None

    return stats
