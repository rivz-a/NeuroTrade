"""Shared on-disk dashboard cache -- the bridge between trading_runtime.py's
scheduled AI cycle (runtime/ai_cycle.py, a SEPARATE process from server.py)
and what's actually served. Whoever computes a fresh AIAnalysisResult batch
for one mode calls save_mode_results() here; server.py notices the file
changed and reloads/re-renders from it (see server.py's
_maybe_reload_from_disk) instead of only ever picking up new results from
its own dashboard-side "Обновить" click.

Same shape as server.py's own long-standing _save_cache/_load_cache
(snapshot/results_by_mode/active_mode/last_updated_at) -- promoted to a
shared module now that TWO processes write to the same file.
"""

from __future__ import annotations

import os
import pickle
import time
from pathlib import Path

import config
from ai_client import AIAnalysisResult


def load() -> dict | None:
    """Never raises -- a missing/corrupt cache just means "nothing to load"."""
    if not config.DASHBOARD_CACHE_FILE.exists():
        return None
    try:
        with open(config.DASHBOARD_CACHE_FILE, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None


def _atomic_write(path: Path, data: bytes) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_bytes(data)
    os.replace(tmp_path, path)


def save_mode_results(mode: str, results: list[AIAnalysisResult], snapshot: dict) -> None:
    """Merges a fresh results batch for ONE mode into the shared cache --
    the other mode's last-known results are always left untouched, the
    same "only the active tab's data changes" rule server.py's own
    _refresh_mode already follows. Atomic write (temp file + os.replace)
    so a concurrent reader (server.py's poll) never sees a half-written
    file.
    """
    cached = load() or {}
    results_by_mode = dict(cached.get("results_by_mode") or {})
    results_by_mode[mode] = results
    last_updated_at = dict(cached.get("last_updated_at") or {})
    last_updated_at[mode] = time.time()

    payload = {
        "snapshot": snapshot,
        "results_by_mode": results_by_mode,
        "active_mode": cached.get("active_mode") or mode,
        "last_updated_at": last_updated_at,
    }
    _atomic_write(config.DASHBOARD_CACHE_FILE, pickle.dumps(payload))
