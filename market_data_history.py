"""Local rolling history of compact market-data samples — top-of-book
(best bid/ask/spread/imbalance) plus the instrument specification in
effect at the time, one entry per `market_data_engine.collect_snapshot()`
call.

Mirrors `oi_history.py`'s append/trim JSONL pattern exactly (same reasons
apply: this is a growing time series that needs to survive restarts and
be queried historically, not a "last known state" cache). Not the full
order book (20 levels) — that would bloat the file fast for no benefit;
top-of-book is enough to reconstruct a spread/liquidity history, and
serves double duty as the source `market_data_engine` compares against to
detect an instrument-specification change between two snapshots.

Best-effort throughout: a read/write failure here should never block a
snapshot collection.
"""

from __future__ import annotations

import json

import config


def _load_lines() -> list[str]:
    try:
        with open(config.MARKET_DATA_HISTORY_FILE, encoding="utf-8") as f:
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


def load_recent(symbol: str, min_seconds_ago: float, max_seconds_ago: float, now: float) -> list[dict]:
    """Samples for `symbol` whose `ts` falls in [now-max_seconds_ago,
    now-min_seconds_ago] — e.g. "everything from 30 to 120 seconds ago",
    for a rolling window average. Returns `[]` (not a fabricated value)
    when nothing falls in that window — this app has no fixed-cadence
    background poller, so whether enough samples exist at all depends
    entirely on how often the caller happens to run.
    """
    lo = now - max_seconds_ago
    hi = now - min_seconds_ago
    return [s for s in load_samples(symbol) if lo <= s.get("ts", -1) <= hi]


def append_and_trim(entry: dict) -> None:
    try:
        with open(config.MARKET_DATA_HISTORY_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except OSError:
        return

    lines = _load_lines()
    if len(lines) <= config.MARKET_DATA_HISTORY_MAX_ENTRIES:
        return
    try:
        with open(config.MARKET_DATA_HISTORY_FILE, "w", encoding="utf-8") as f:
            f.writelines(lines[-config.MARKET_DATA_HISTORY_MAX_ENTRIES :])
    except OSError:
        return
