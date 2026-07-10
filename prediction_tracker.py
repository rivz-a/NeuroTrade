"""Tracks each AI model's trade plan over time and scores it once its horizon
has elapsed, so the dashboard can show which model is actually right more
often — not just which one sounds more confident.

Storage is a local JSON Lines log (`config.PREDICTION_HISTORY_FILE`), one
entry per (mode, model) prediction. Scoring walks the real BingX candle path
over the prediction's [predicted_at, predicted_at + horizon] window
(`outcome_simulator.py`) to see whether stop loss or a take profit was hit
first, rather than just comparing price at a fixed deadline — see the
Stage 3 plan for why the old directional-only approach wasn't good enough.
WAIT calls are not scored (there's no directional bet to grade) but are
still counted so the UI can show how often each model says WAIT.

Best-effort throughout: a read/write/BingX failure here should never break a
snapshot fetch or an AI call — a failed evaluation just stays "pending" and
is retried on the next call.
"""

from __future__ import annotations

import json
import statistics
import time
import uuid

import app_log
import bingx_client
import config
import outcome_simulator
from ai_client import AIAnalysisResult

_EXIT_REASONS = ("SL", "TP1", "TP2", "TP3", "TIMEOUT", "AMBIGUOUS")


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


def record_results(
    mode: str, results: list[AIAnalysisResult], price_at_prediction: float | None, symbol: str
) -> None:
    """Log one entry per result with a usable, validator-approved trade plan.

    Results with no `trade_plan` (network/parse/validation failures, or
    legacy pre-migration cache entries) aren't scoreable — skipped entirely.
    """
    if price_at_prediction is None:
        return
    horizon = config.PREDICTION_HORIZON_SECONDS.get(mode, config.PREDICTION_HORIZON_SECONDS["scalping"])
    now = time.time()
    entries = _load_all()
    for result in results:
        plan = result.trade_plan
        if plan is None:
            continue
        entries.append(
            {
                "schema": "v2",
                "id": uuid.uuid4().hex,
                "mode": mode,
                "symbol": symbol,
                "label": result.label,
                "model": result.model,
                "signal": plan.signal,
                "predicted_at": now,
                "price_at_prediction": price_at_prediction,
                "horizon_seconds": horizon,
                "status": "pending" if plan.signal in ("LONG", "SHORT") else "skipped",
                "entry_from": plan.entry.from_,
                "entry_to": plan.entry.to,
                "stop_loss": plan.stop_loss,
                "take_profits": [{"label": tp.label, "price": tp.price} for tp in plan.take_profits],
                # Filled in by evaluate_due() once the horizon elapses:
                "exit_reason": None,
                "r_multiple": None,
                "mfe_r": None,
                "mae_r": None,
                "exit_price": None,
                "exit_time": None,
                "duration_seconds": None,
                "evaluated_at": None,
            }
        )
    _save_all(entries)


def evaluate_due(symbol: str, now: float | None = None) -> int:
    """Simulate every pending prediction whose horizon has elapsed by walking
    the real BingX candle path for its window. Returns how many were scored.
    Safe to call on every snapshot fetch — a no-op when nothing is due, and
    a BingX failure on one entry just leaves it pending for the next try.
    """
    now = now if now is not None else time.time()
    entries = _load_all()
    scored = 0
    for entry in entries:
        if entry.get("status") != "pending" or entry.get("schema") != "v2":
            continue
        deadline = entry["predicted_at"] + entry["horizon_seconds"]
        if now < deadline:
            continue

        entry_symbol = entry.get("symbol") or symbol
        take_profits = [(tp["label"], tp["price"]) for tp in entry.get("take_profits", [])]
        start_ms = int(entry["predicted_at"] * 1000)
        end_ms = int(deadline * 1000)

        try:
            candles = bingx_client.get_klines_range(entry_symbol, config.OUTCOME_KLINE_INTERVAL, start_ms, end_ms)
        except bingx_client.BingXError as exc:
            app_log.log_error("OUTCOME_KLINES_FAILED", entry.get("model", ""), entry.get("mode", ""), str(exc))
            continue

        entry_price = (entry["entry_from"] + entry["entry_to"]) / 2
        outcome = outcome_simulator.simulate_outcome(
            entry["signal"],
            entry_price,
            entry["stop_loss"],
            take_profits,
            candles,
            config.TRADING_COMMISSION_PCT,
            config.TRADING_SLIPPAGE_PCT,
        )
        if outcome is None:
            app_log.log_error(
                "OUTCOME_SIMULATION_EMPTY", entry.get("model", ""), entry.get("mode", ""),
                "no candles in window or zero-risk plan",
            )
            continue

        entry["status"] = "evaluated"
        entry["exit_reason"] = outcome.exit_reason
        entry["r_multiple"] = outcome.r_multiple
        entry["mfe_r"] = outcome.mfe_r
        entry["mae_r"] = outcome.mae_r
        entry["exit_price"] = outcome.exit_price
        entry["exit_time"] = outcome.exit_time
        entry["duration_seconds"] = outcome.duration_seconds
        entry["evaluated_at"] = now
        scored += 1

    if scored:
        _save_all(entries)
    return scored


def _max_drawdown(r_values: list[float]) -> float | None:
    """Largest peak-to-trough drop in the running cumulative R sequence —
    r_values must already be in chronological order (append-only log).
    """
    if not r_values:
        return None
    cumulative = 0.0
    peak = 0.0
    max_dd = 0.0
    for r in r_values:
        cumulative += r
        peak = max(peak, cumulative)
        max_dd = max(max_dd, peak - cumulative)
    return max_dd


def stats_by_model_and_mode(mode: str | None = None) -> dict[str, dict]:
    """Per-model accuracy stats, optionally restricted to one trading mode —
    scalping and swing must never be blended into one number (see the
    Stage 2 plan's note on why consensus weighting was deferred).

    Only `schema="v2"` evaluated entries (real candle-path simulations)
    contribute to the R-based metrics; older pre-Stage-3 "evaluated" entries
    have no `r_multiple` and are excluded from these numbers — there's no
    way to retroactively simulate their candle path without a fresh BingX
    call for a window that's already partially expired from the exchange's
    perspective, and the sample size was tiny synthetic test data anyway.
    """
    entries = _load_all()
    if mode is not None:
        entries = [e for e in entries if e.get("mode") == mode]

    raw: dict[str, dict] = {}
    for entry in entries:
        label = entry["label"]
        bucket = raw.setdefault(
            label,
            {
                "total": 0,
                "pending": 0,
                "skipped": 0,
                "wins": 0,
                "r_values": [],
                "mfe_values": [],
                "mae_values": [],
                "durations": [],
                "exit_reason_counts": {reason: 0 for reason in _EXIT_REASONS},
            },
        )
        bucket["total"] += 1
        status = entry.get("status")
        if status == "pending":
            bucket["pending"] += 1
            continue
        if status == "skipped":
            bucket["skipped"] += 1
            continue
        if status == "evaluated" and entry.get("schema") == "v2" and entry.get("r_multiple") is not None:
            r = entry["r_multiple"]
            bucket["r_values"].append(r)
            if r > 0:
                bucket["wins"] += 1
            bucket["mfe_values"].append(entry.get("mfe_r") or 0.0)
            bucket["mae_values"].append(entry.get("mae_r") or 0.0)
            bucket["durations"].append(entry.get("duration_seconds") or 0.0)
            reason = entry.get("exit_reason")
            if reason in bucket["exit_reason_counts"]:
                bucket["exit_reason_counts"][reason] += 1

    stats: dict[str, dict] = {}
    for label, bucket in raw.items():
        r_values = bucket["r_values"]
        n = len(r_values)
        gains = sum(r for r in r_values if r > 0)
        losses = sum(-r for r in r_values if r < 0)
        if losses > 0:
            profit_factor = gains / losses
        elif gains > 0:
            profit_factor = None  # undefined ("infinite") — no losing trades yet
        else:
            profit_factor = None

        stats[label] = {
            "total": bucket["total"],
            "pending": bucket["pending"],
            "skipped": bucket["skipped"],
            "evaluated": n,
            "wins": bucket["wins"],
            "win_rate": (bucket["wins"] / n * 100) if n else None,
            "expectancy_r": (sum(r_values) / n) if n else None,
            "median_r": statistics.median(r_values) if r_values else None,
            "profit_factor": profit_factor,
            "profit_factor_undefined": losses == 0 and gains > 0,
            "max_drawdown_r": _max_drawdown(r_values),
            "avg_mfe_r": (sum(bucket["mfe_values"]) / n) if n else None,
            "avg_mae_r": (sum(bucket["mae_values"]) / n) if n else None,
            "avg_duration_seconds": (sum(bucket["durations"]) / n) if n else None,
            "exit_reason_counts": bucket["exit_reason_counts"],
            "low_sample": n < config.MIN_SAMPLE_FOR_STATS,
        }

    return stats
