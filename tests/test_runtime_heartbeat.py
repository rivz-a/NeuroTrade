"""Offline tests for runtime/heartbeat.py — no network."""

from __future__ import annotations

from runtime.heartbeat import heartbeat_status, is_heartbeat_stale, read_heartbeat, write_heartbeat


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


def test_heartbeat_status_missing_file_is_unknown(tmp_path):
    status = heartbeat_status(tmp_path / "does_not_exist.json", max_age_seconds=15)
    assert status == {"state": "unknown", "age_seconds": None}


def test_heartbeat_status_fresh_is_ok(tmp_path):
    path = tmp_path / "heartbeat.json"
    write_heartbeat(
        path, mode="PAPER", symbol="ETHUSDT", open_paper_positions=1, open_real_positions=0, last_error=None
    )
    status = heartbeat_status(path, max_age_seconds=15)
    assert status["state"] == "ok"
    assert status["mode"] == "PAPER"
    assert status["symbol"] == "ETHUSDT"
    assert status["open_paper_positions"] == 1
    assert status["open_real_positions"] == 0
    assert status["age_seconds"] >= 0


def test_heartbeat_status_past_threshold_is_stale(tmp_path):
    path = tmp_path / "heartbeat.json"
    write_heartbeat(path, mode="PAPER", symbol="ETHUSDT")
    hb = read_heartbeat(path)
    status = heartbeat_status(path, max_age_seconds=15, now=hb["last_heartbeat_at"] + 20)
    assert status["state"] == "stale"
    assert status["age_seconds"] == 20


def test_heartbeat_status_surfaces_last_error(tmp_path):
    path = tmp_path / "heartbeat.json"
    write_heartbeat(path, mode="PAPER", symbol="ETHUSDT", last_error="process_tick failed: boom")
    status = heartbeat_status(path, max_age_seconds=15)
    assert status["last_error"] == "process_tick failed: boom"


def test_heartbeat_status_ai_cycle_uses_busy_threshold_instead_of_stale(tmp_path):
    path = tmp_path / "heartbeat.json"
    write_heartbeat(path, mode="PAPER", symbol="ETHUSDT", activity="ai_cycle")
    hb = read_heartbeat(path)
    # 40s would be "stale" under the normal 15s budget, but this heartbeat
    # is tagged as a busy AI cycle, so the wider 150s budget applies.
    status = heartbeat_status(path, max_age_seconds=15, busy_max_age_seconds=150, now=hb["last_heartbeat_at"] + 40)
    assert status["state"] == "ok"
    assert status["activity"] == "ai_cycle"


def test_heartbeat_status_ai_cycle_still_goes_stale_past_the_busy_budget(tmp_path):
    path = tmp_path / "heartbeat.json"
    write_heartbeat(path, mode="PAPER", symbol="ETHUSDT", activity="ai_cycle")
    hb = read_heartbeat(path)
    status = heartbeat_status(path, max_age_seconds=15, busy_max_age_seconds=150, now=hb["last_heartbeat_at"] + 200)
    assert status["state"] == "stale"


def test_heartbeat_status_non_ai_cycle_ignores_busy_threshold(tmp_path):
    path = tmp_path / "heartbeat.json"
    write_heartbeat(path, mode="PAPER", symbol="ETHUSDT")  # no "activity" field
    hb = read_heartbeat(path)
    status = heartbeat_status(path, max_age_seconds=15, busy_max_age_seconds=150, now=hb["last_heartbeat_at"] + 40)
    assert status["state"] == "stale"
