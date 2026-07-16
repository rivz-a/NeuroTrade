"""Offline tests for model_weights.py — no network, no AI. Synthetic
predictions.jsonl entries built by hand; JSONL log redirected to a pytest
tmp_path, same isolation pattern as tests/test_prediction_tracker.py.
"""

from __future__ import annotations

import pytest

import config
import model_weights
import prediction_tracker as pt


@pytest.fixture(autouse=True)
def _isolated_history_file(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "PREDICTION_HISTORY_FILE", tmp_path / "predictions.jsonl")


def _entry(
    model="gpt", mode="scalping", symbol="ETHUSDT", signal="LONG", confidence=70,
    market_regime="TREND_UP", prompt_version="v1", r_multiple=1.0, status="evaluated",
) -> dict:
    return {
        "schema": "v2", "id": "x", "mode": mode, "symbol": symbol, "label": model, "model": model,
        "signal": signal, "confidence": confidence, "market_regime": market_regime,
        "prompt_version": prompt_version, "predicted_at": 0.0, "price_at_prediction": 100.0,
        "horizon_seconds": 1800, "status": status, "entry_from": 100.0, "entry_to": 101.0,
        "stop_loss": 98.0, "take_profits": [],
        "exit_reason": "TP1" if r_multiple > 0 else "SL", "r_multiple": r_multiple,
        "mfe_r": abs(r_multiple), "mae_r": 0.1, "exit_price": 105.0, "exit_time": 100.0,
        "duration_seconds": 60.0, "evaluated_at": 100.0,
    }


# ---------------------------------------------------------------------------
# _confidence_bucket
# ---------------------------------------------------------------------------


def test_confidence_bucket_boundaries():
    assert model_weights._confidence_bucket(59) == "LOW"
    assert model_weights._confidence_bucket(60) == "MEDIUM"
    assert model_weights._confidence_bucket(74) == "MEDIUM"
    assert model_weights._confidence_bucket(75) == "HIGH"
    assert model_weights._confidence_bucket(89) == "HIGH"
    assert model_weights._confidence_bucket(90) == "VERY_HIGH"
    assert model_weights._confidence_bucket(100) == "VERY_HIGH"
    assert model_weights._confidence_bucket(None) is None


# ---------------------------------------------------------------------------
# resolve_bucket
# ---------------------------------------------------------------------------


def test_resolve_bucket_full_specificity_when_enough_samples():
    entries = [
        _entry(model="gpt", mode="scalping", market_regime="TREND_UP", signal="LONG", confidence=80, prompt_version="v1", r_multiple=1.0)
        for _ in range(12)
    ]
    pt._save_all(entries)
    bucket = model_weights.resolve_bucket("gpt", mode="scalping", regime="TREND_UP", direction="LONG", confidence=80, prompt_version="v1")
    assert bucket is not None
    assert bucket.n == 12
    assert bucket.dimensions_used == ("mode", "prompt_version", "confidence_bucket", "signal", "market_regime")


def test_resolve_bucket_falls_back_to_coarser_level():
    entries = [
        _entry(model="gpt", mode="scalping", market_regime="TREND_UP", signal="LONG", confidence=80, prompt_version="v1", r_multiple=1.0)
        for _ in range(3)
    ] + [
        _entry(model="gpt", mode="scalping", market_regime="TREND_UP", signal="SHORT", confidence=40, prompt_version="v2", r_multiple=0.5)
        for _ in range(12)
    ]
    pt._save_all(entries)
    bucket = model_weights.resolve_bucket("gpt", mode="scalping", regime="TREND_UP", direction="LONG", confidence=80, prompt_version="v1")
    assert bucket is not None
    assert bucket.n == 15
    assert bucket.dimensions_used == ("mode", "market_regime")


def test_resolve_bucket_returns_none_when_insufficient_at_coarsest_level():
    entries = [_entry(model="gpt", mode="scalping", r_multiple=1.0) for _ in range(config.MODEL_WEIGHT_MIN_SAMPLE - 1)]
    pt._save_all(entries)
    assert model_weights.resolve_bucket("gpt", mode="scalping") is None


def test_resolve_bucket_mode_never_dropped():
    entries = [_entry(model="gpt", mode="scalping", r_multiple=1.0) for _ in range(15)] + [
        _entry(model="gpt", mode="swing", r_multiple=-1.0) for _ in range(15)
    ]
    pt._save_all(entries)
    bucket = model_weights.resolve_bucket("gpt", mode="scalping")
    assert bucket is not None
    assert bucket.n == 15
    assert bucket.expectancy_r == pytest.approx(1.0)


def test_resolve_bucket_different_models_not_blended():
    entries = [_entry(model="gpt", mode="scalping", r_multiple=1.0) for _ in range(15)] + [
        _entry(model="gemini", mode="scalping", r_multiple=-1.0) for _ in range(15)
    ]
    pt._save_all(entries)
    bucket = model_weights.resolve_bucket("gpt", mode="scalping")
    assert bucket.n == 15
    assert bucket.expectancy_r == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# _base_weight / _sample_size_factor
# ---------------------------------------------------------------------------


def test_base_weight_neutral_at_zero_expectancy():
    assert model_weights._base_weight(0.0) == config.MODEL_WEIGHT_NEUTRAL


def test_base_weight_clamped_to_max():
    assert model_weights._base_weight(100.0) == config.MODEL_WEIGHT_MAX


def test_base_weight_clamped_to_min():
    assert model_weights._base_weight(-100.0) == config.MODEL_WEIGHT_MIN


def test_sample_size_factor_at_min_is_zero():
    assert model_weights._sample_size_factor(config.MODEL_WEIGHT_MIN_SAMPLE) == pytest.approx(0.0)


def test_sample_size_factor_at_full_is_one():
    assert model_weights._sample_size_factor(config.MODEL_WEIGHT_FULL_SAMPLE) == pytest.approx(1.0)


def test_sample_size_factor_ramps_between():
    mid = (config.MODEL_WEIGHT_MIN_SAMPLE + config.MODEL_WEIGHT_FULL_SAMPLE) // 2
    factor = model_weights._sample_size_factor(mid)
    assert 0.0 < factor < 1.0


def test_sample_size_factor_above_full_is_one():
    assert model_weights._sample_size_factor(config.MODEL_WEIGHT_FULL_SAMPLE + 100) == 1.0


# ---------------------------------------------------------------------------
# effective_weight
# ---------------------------------------------------------------------------


def test_effective_weight_hand_computed_at_full_sample():
    entries = [_entry(model="gpt", mode="scalping", r_multiple=1.0) for _ in range(config.MODEL_WEIGHT_FULL_SAMPLE)]
    pt._save_all(entries)
    weight = model_weights.effective_weight("gpt", mode="scalping", confidence=80)
    expected_base = model_weights._base_weight(1.0)
    assert weight == pytest.approx(expected_base * 0.8)


def test_effective_weight_none_when_insufficient_data():
    entries = [_entry(model="gpt", mode="scalping", r_multiple=1.0) for _ in range(3)]
    pt._save_all(entries)
    assert model_weights.effective_weight("gpt", mode="scalping", confidence=80) is None


def test_effective_weight_scales_with_confidence():
    entries = [_entry(model="gpt", mode="scalping", r_multiple=1.0) for _ in range(config.MODEL_WEIGHT_FULL_SAMPLE)]
    pt._save_all(entries)
    high_conf = model_weights.effective_weight("gpt", mode="scalping", confidence=90)
    low_conf = model_weights.effective_weight("gpt", mode="scalping", confidence=30)
    assert high_conf > low_conf


def test_effective_weight_unprofitable_model_below_neutral():
    entries = [_entry(model="gpt", mode="scalping", r_multiple=-1.0) for _ in range(config.MODEL_WEIGHT_FULL_SAMPLE)]
    pt._save_all(entries)
    weight = model_weights.effective_weight("gpt", mode="scalping", confidence=100)
    assert weight < config.MODEL_WEIGHT_NEUTRAL


def test_effective_weight_thin_sample_shrinks_toward_neutral():
    # Exactly at MIN_SAMPLE: sample_size_factor == 0 -> smoothed == NEUTRAL regardless of expectancy_r.
    entries = [_entry(model="gpt", mode="scalping", r_multiple=5.0) for _ in range(config.MODEL_WEIGHT_MIN_SAMPLE)]
    pt._save_all(entries)
    weight = model_weights.effective_weight("gpt", mode="scalping", confidence=100)
    assert weight == pytest.approx(config.MODEL_WEIGHT_NEUTRAL)


# ---------------------------------------------------------------------------
# bucket_report
# ---------------------------------------------------------------------------


def test_bucket_report_basic_shape():
    entries = [_entry(model="gpt", mode="scalping", market_regime="TREND_UP", r_multiple=1.0) for _ in range(15)]
    pt._save_all(entries)
    rows = model_weights.bucket_report(["gpt"], modes=("scalping",), regimes=(None, "TREND_UP"))
    assert len(rows) >= 1
    for row in rows:
        assert row["model"] == "gpt"
        assert row["n"] >= config.MODEL_WEIGHT_MIN_SAMPLE
        assert row["base_weight"] is not None


def test_bucket_report_skips_models_with_no_data():
    rows = model_weights.bucket_report(["nonexistent-model"], modes=("scalping",), regimes=(None,))
    assert rows == []
