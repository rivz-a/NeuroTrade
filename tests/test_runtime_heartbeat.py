"""Offline tests for runtime/heartbeat.py — no network."""

from __future__ import annotations

from runtime.heartbeat import is_heartbeat_stale, read_heartbeat, write_heartbeat


def test_write_read_round_trip(tmp_path):
    path = tmp_path / "heartbeat.json"
    write_heartbeat(path, mode="PAPER", symbol="ETHUSDT", open_paper_positions=2)
    result = read_heartbeat(path)
    assert result["mode"] == "PAPER"
    assert result["symbol"] == "ETHUSDT"
    assert result["open_paper_positions"] == 2
    assert "last_heartbeat_at" in result


def test_read_missing_file_returns_none(tmp_path):
    assert read_heartbeat(tmp_path / "does_not_exist.json") is None


def test_write_overwrites_previous_content(tmp_path):
    path = tmp_path / "heartbeat.json"
    write_heartbeat(path, open_paper_positions=1)
    write_heartbeat(path, open_paper_positions=5)
    assert read_heartbeat(path)["open_paper_positions"] == 5


def test_is_heartbeat_stale_just_within_threshold():
    hb = {"last_heartbeat_at": 1000.0}
    assert is_heartbeat_stale(hb, max_age_seconds=15, now=1010.0) is False


def test_is_heartbeat_stale_past_threshold():
    hb = {"last_heartbeat_at": 1000.0}
    assert is_heartbeat_stale(hb, max_age_seconds=15, now=1020.0) is True


def test_is_heartbeat_stale_missing_timestamp_is_stale():
    assert is_heartbeat_stale({}, max_age_seconds=15) is True
