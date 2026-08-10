"""Liveness reporting for trading_runtime.py — a small JSON file (same
local-file convention as dashboard_cache.pkl/kill_switch.flag) a human, a
future dashboard panel, or a monitoring script can read to answer "is the
runtime actually alive and ticking" without needing to inspect the lock
file or process list directly.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path


def write_heartbeat(path: Path, **fields) -> None:
    """Atomic write (temp file + os.replace) so a reader never sees a
    half-written file, even if it reads concurrently with a write.
    """
    payload = {"last_heartbeat_at": time.time(), **fields}
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload), encoding="utf-8")
    os.replace(tmp_path, path)


def read_heartbeat(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def is_heartbeat_stale(heartbeat: dict, max_age_seconds: float, now: float | None = None) -> bool:
    now = now if now is not None else time.time()
    last = heartbeat.get("last_heartbeat_at")
    if last is None:
        return True
    return (now - last) > max_age_seconds


def heartbeat_status(path: Path, max_age_seconds: float, now: float | None = None) -> dict:
    """Normalized summary for UI/monitoring consumers (dashboard badge,
    /api/status) — one place to turn the raw heartbeat file into a
    tri-state "state" so callers never re-derive the ok/stale/unknown logic
    themselves.

    "unknown" (no file yet, e.g. right after a fresh deploy before the
    runtime's first tick, or the file failed to parse) is kept distinct
    from "stale" (was alive, has since gone quiet) since they call for
    different UI treatment.
    """
    now = now if now is not None else time.time()
    heartbeat = read_heartbeat(path)
    if heartbeat is None:
        return {"state": "unknown", "age_seconds": None}

    last = heartbeat.get("last_heartbeat_at")
    age_seconds = (now - last) if last is not None else None
    state = "stale" if is_heartbeat_stale(heartbeat, max_age_seconds, now=now) else "ok"
    return {
        "state": state,
        "age_seconds": age_seconds,
        "mode": heartbeat.get("mode"),
        "symbol": heartbeat.get("symbol"),
        "open_paper_positions": heartbeat.get("open_paper_positions"),
        "open_real_positions": heartbeat.get("open_real_positions"),
        "last_error": heartbeat.get("last_error"),
    }
