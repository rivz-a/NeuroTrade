"""Offline tests for dashboard_state.py -- the shared on-disk cache bridging
trading_runtime.py's ai_cycle (a separate process) and server.py's dashboard.
No network; everything is local file I/O against a pytest tmp_path.
"""

from __future__ import annotations

import pickle

import pytest

import config
import dashboard_state
from ai_client import AIAnalysisResult

NOW = 1_700_000_000.0


@pytest.fixture(autouse=True)
def _use_tmp_cache(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "DASHBOARD_CACHE_FILE", tmp_path / "dashboard_cache.pkl")


def _result(label="GPT-4o mini") -> AIAnalysisResult:
    return AIAnalysisResult(
        label=label, model=label.lower(), content="{}", error=None,
        latency_seconds=1.0, created_at=NOW, trade_plan=None, validation=None,
    )


def test_load_missing_file_returns_none():
    assert dashboard_state.load() is None


def test_save_and_load_round_trip():
    dashboard_state.save_mode_results("scalping", [_result()], {"symbol": "ETHUSDT"})
    cached = dashboard_state.load()
    assert cached["results_by_mode"]["scalping"][0].label == "GPT-4o mini"
    assert cached["snapshot"] == {"symbol": "ETHUSDT"}
    assert "scalping" in cached["last_updated_at"]


def test_save_preserves_the_other_modes_results():
    dashboard_state.save_mode_results("scalping", [_result("A")], {"symbol": "ETHUSDT"})
    dashboard_state.save_mode_results("swing", [_result("B")], {"symbol": "ETHUSDT"})

    cached = dashboard_state.load()
    assert cached["results_by_mode"]["scalping"][0].label == "A"
    assert cached["results_by_mode"]["swing"][0].label == "B"


def test_save_overwrites_only_the_same_mode():
    dashboard_state.save_mode_results("scalping", [_result("A")], {"symbol": "ETHUSDT"})
    dashboard_state.save_mode_results("scalping", [_result("C")], {"symbol": "ETHUSDT"})

    cached = dashboard_state.load()
    assert len(cached["results_by_mode"]) == 1
    assert cached["results_by_mode"]["scalping"][0].label == "C"


def test_save_bumps_last_updated_at_for_the_saved_mode_only():
    dashboard_state.save_mode_results("scalping", [_result()], {"symbol": "ETHUSDT"})
    first_ts = dashboard_state.load()["last_updated_at"]["scalping"]

    dashboard_state.save_mode_results("swing", [_result()], {"symbol": "ETHUSDT"})
    cached = dashboard_state.load()
    assert cached["last_updated_at"]["scalping"] == first_ts
    assert "swing" in cached["last_updated_at"]


def test_load_corrupt_file_returns_none(tmp_path):
    config.DASHBOARD_CACHE_FILE.write_bytes(b"not a pickle")
    assert dashboard_state.load() is None


def test_save_leaves_no_leftover_tmp_file():
    dashboard_state.save_mode_results("scalping", [_result()], {"symbol": "ETHUSDT"})
    tmp_path = config.DASHBOARD_CACHE_FILE.with_suffix(config.DASHBOARD_CACHE_FILE.suffix + ".tmp")
    assert not tmp_path.exists()
    assert config.DASHBOARD_CACHE_FILE.exists()


def test_active_mode_keeps_first_saved_mode_if_never_set(tmp_path):
    dashboard_state.save_mode_results("swing", [_result()], {"symbol": "ETHUSDT"})
    cached = dashboard_state.load()
    assert cached["active_mode"] == "swing"

    dashboard_state.save_mode_results("scalping", [_result()], {"symbol": "ETHUSDT"})
    cached = dashboard_state.load()
    assert cached["active_mode"] == "swing"  # not overwritten by a later save
