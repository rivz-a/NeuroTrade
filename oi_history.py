"""Local rolling history of open-interest samples.

BingX's public API exposes only the *current* open interest — unlike funding
rate, there is no history endpoint for it. So instead of a single snapshot
value, every `fetch_snapshot()` call records one sample here (JSON Lines,
capped at `config.OI_HISTORY_MAX_ENTRIES`) and diffs the new reading against
the previous one for that symbol, giving a real (not synthetic) OI trend
that gets more useful the more the app is used over time.

Best-effort throughout: a read/write failure here should never block a
snapshot fetch.
"""

from __future__ import annotations

import json
import time

import config


def _load_lines() -> list[str]:
    try:
        with open(config.OI_HISTORY_FILE, encoding="utf-8") as f:
            return f.readlines()
    except OSError:
        return []


def load_samples(symbol: str) -> list[dict]:
    samples: list[dict] = []
    for line in _load_lines():
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if entry.get("symbol") == symbol:
            samples.append(entry)
    samples.sort(key=lambda s: s.get("ts", 0))
    return samples


def _append_and_trim(entry: dict) -> None:
    try:
        with open(config.OI_HISTORY_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except OSError:
        return

    lines = _load_lines()
    if len(lines) <= config.OI_HISTORY_MAX_ENTRIES:
        return
    try:
        with open(config.OI_HISTORY_FILE, "w", encoding="utf-8") as f:
            f.writelines(lines[-config.OI_HISTORY_MAX_ENTRIES :])
    except OSError:
        return


def record_and_get_trend(symbol: str, current_oi: float | None, price: float | None) -> dict | None:
    """Append the current OI sample and return the change since the previous
    recorded sample for this symbol. Returns None if there is no current OI
    to record, or if this is the first sample ever seen for the symbol.
    """
    if current_oi is None:
        return None

    prior_samples = load_samples(symbol)
    prev = prior_samples[-1] if prior_samples else None

    _append_and_trim({"ts": time.time(), "symbol": symbol, "open_interest": current_oi, "price": price})

    if prev is None or prev.get("open_interest") is None:
        return None

    prev_oi = prev["open_interest"]
    delta = current_oi - prev_oi
    pct = (delta / prev_oi * 100) if prev_oi else None
    return {
        "prev_oi": prev_oi,
        "delta": delta,
        "pct": pct,
        "age_seconds": time.time() - prev["ts"],
    }
