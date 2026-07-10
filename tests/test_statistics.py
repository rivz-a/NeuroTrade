"""Offline tests for prediction_tracker.stats_by_model_and_mode — pure
aggregation over synthetic JSONL entries, no network, no AI calls.
"""

import pytest

import config
import prediction_tracker as pt


@pytest.fixture(autouse=True)
def _isolated_history_file(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "PREDICTION_HISTORY_FILE", tmp_path / "predictions.jsonl")


def _v2_entry(label, mode, status="evaluated", r_multiple=None, exit_reason="TP1", mfe_r=None, mae_r=0.2, duration=600):
    return {
        "schema": "v2", "id": "x", "mode": mode, "symbol": "ETH-USDT", "label": label, "model": label.lower(),
        "signal": "LONG", "predicted_at": 0, "price_at_prediction": 100, "horizon_seconds": 1800,
        "status": status,
        "entry_from": 100, "entry_to": 101, "stop_loss": 98,
        "take_profits": [{"label": "TP1", "price": 105}],
        "exit_reason": exit_reason if status == "evaluated" else None,
        "r_multiple": r_multiple if status == "evaluated" else None,
        "mfe_r": (mfe_r if mfe_r is not None else max(r_multiple or 0, 0)) if status == "evaluated" else None,
        "mae_r": mae_r if status == "evaluated" else None,
        "exit_price": 105, "exit_time": 100,
        "duration_seconds": duration if status == "evaluated" else None,
        "evaluated_at": 200 if status == "evaluated" else None,
    }


def _v1_legacy_entry(label, mode):
    """Pre-Stage-3 format: no `schema` key, directional win/loss/flat outcome."""
    return {
        "id": "legacy", "mode": mode, "label": label, "model": label.lower(),
        "signal": "LONG", "predicted_at": 0, "price_at_prediction": 100, "horizon_seconds": 1800,
        "status": "evaluated", "outcome": "win", "evaluated_at": 200, "price_at_evaluation": 110,
    }


def test_basic_metrics():
    entries = [
        _v2_entry("GPT", "scalping", r_multiple=2.0),
        _v2_entry("GPT", "scalping", r_multiple=-1.0),
        _v2_entry("GPT", "scalping", r_multiple=1.5),
    ]
    pt._save_all(entries)
    stats = pt.stats_by_model_and_mode("scalping")["GPT"]
    assert stats["evaluated"] == 3
    assert stats["wins"] == 2
    assert stats["win_rate"] == pytest.approx(200 / 3)
    assert stats["expectancy_r"] == pytest.approx(2.5 / 3)
    assert stats["median_r"] == 1.5
    assert stats["profit_factor"] == pytest.approx(3.5)
    assert stats["low_sample"] is False


def test_mode_segmentation_does_not_blend():
    entries = [
        _v2_entry("GPT", "scalping", r_multiple=1.0),
        _v2_entry("GPT", "scalping", r_multiple=1.0),
        _v2_entry("GPT", "swing", r_multiple=-3.0),
    ]
    pt._save_all(entries)
    scalping_stats = pt.stats_by_model_and_mode("scalping")["GPT"]
    swing_stats = pt.stats_by_model_and_mode("swing")["GPT"]
    assert scalping_stats["evaluated"] == 2
    assert scalping_stats["expectancy_r"] == pytest.approx(1.0)
    assert swing_stats["evaluated"] == 1
    assert swing_stats["expectancy_r"] == pytest.approx(-3.0)


def test_max_drawdown_on_running_r_sequence():
    entries = [
        _v2_entry("GPT", "scalping", r_multiple=1.0),
        _v2_entry("GPT", "scalping", r_multiple=-2.0),
        _v2_entry("GPT", "scalping", r_multiple=1.0),
        _v2_entry("GPT", "scalping", r_multiple=-0.5),
    ]
    pt._save_all(entries)
    stats = pt.stats_by_model_and_mode("scalping")["GPT"]
    # cumulative: 1.0, -1.0, 0.0, -0.5 -> peak stays 1.0 -> worst drawdown = 1.0 - (-1.0) = 2.0
    assert stats["max_drawdown_r"] == pytest.approx(2.0)


def test_profit_factor_undefined_when_no_losses():
    entries = [
        _v2_entry("GPT", "scalping", r_multiple=1.0),
        _v2_entry("GPT", "scalping", r_multiple=2.0),
    ]
    pt._save_all(entries)
    stats = pt.stats_by_model_and_mode("scalping")["GPT"]
    assert stats["profit_factor"] is None
    assert stats["profit_factor_undefined"] is True


def test_low_sample_flag_below_threshold():
    pt._save_all([_v2_entry("GPT", "scalping", r_multiple=1.0)])
    stats = pt.stats_by_model_and_mode("scalping")["GPT"]
    assert stats["evaluated"] == 1
    assert stats["low_sample"] is True


def test_pending_and_skipped_counted_separately():
    entries = [
        _v2_entry("GPT", "scalping", status="pending"),
        _v2_entry("GPT", "scalping", status="skipped"),
        _v2_entry("GPT", "scalping", r_multiple=1.0),
    ]
    pt._save_all(entries)
    stats = pt.stats_by_model_and_mode("scalping")["GPT"]
    assert stats["pending"] == 1
    assert stats["skipped"] == 1
    assert stats["evaluated"] == 1
    assert stats["total"] == 3


def test_legacy_v1_entries_excluded_from_r_metrics_but_counted_in_total():
    entries = [_v1_legacy_entry("GPT", "scalping"), _v2_entry("GPT", "scalping", r_multiple=1.0)]
    pt._save_all(entries)
    stats = pt.stats_by_model_and_mode("scalping")["GPT"]
    assert stats["total"] == 2
    assert stats["evaluated"] == 1  # only the v2 entry contributes an R value
