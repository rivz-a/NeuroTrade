"""Offline tests for outcome_simulator.simulate_outcome — pure function over
synthetic OHLC data, no network, no AI calls.
"""

import pandas as pd
import pytest

from outcome_simulator import simulate_outcome


def _candles(rows: list[dict]) -> pd.DataFrame:
    base = pd.Timestamp("2026-01-01T00:00:00Z")
    data = []
    for i, row in enumerate(rows):
        data.append(
            {
                "time": base + pd.Timedelta(minutes=i),
                "open": row.get("open", row["close"]),
                "high": row["high"],
                "low": row["low"],
                "close": row["close"],
                "volume": 1.0,
            }
        )
    return pd.DataFrame(data)


def test_long_hits_tp1_cleanly():
    candles = _candles([{"high": 106, "low": 99, "close": 105}])
    outcome = simulate_outcome(
        "LONG", entry_price=100, stop_loss=98,
        take_profits=[("TP1", 105), ("TP2", 110), ("TP3", 115)],
        candles=candles, commission_pct=0, slippage_pct=0,
    )
    assert outcome.exit_reason == "TP1"
    assert outcome.exit_price == 105
    assert outcome.r_multiple == pytest.approx(2.5)


def test_long_hits_sl_cleanly():
    candles = _candles([{"high": 101, "low": 97, "close": 98}])
    outcome = simulate_outcome(
        "LONG", entry_price=100, stop_loss=98,
        take_profits=[("TP1", 105)], candles=candles, commission_pct=0, slippage_pct=0,
    )
    assert outcome.exit_reason == "SL"
    assert outcome.r_multiple == pytest.approx(-1.0)


def test_short_mirrors_long():
    candles = _candles([{"high": 101, "low": 94, "close": 95}])
    outcome = simulate_outcome(
        "SHORT", entry_price=100, stop_loss=102,
        take_profits=[("TP1", 95)], candles=candles, commission_pct=0, slippage_pct=0,
    )
    assert outcome.exit_reason == "TP1"
    assert outcome.r_multiple == pytest.approx(2.5)


def test_timeout_when_nothing_hit():
    candles = _candles(
        [
            {"high": 102, "low": 99, "close": 101},
            {"high": 103, "low": 100, "close": 102},
            {"high": 104, "low": 101, "close": 103},
        ]
    )
    outcome = simulate_outcome(
        "LONG", entry_price=100, stop_loss=98,
        take_profits=[("TP1", 105)], candles=candles, commission_pct=0, slippage_pct=0,
    )
    assert outcome.exit_reason == "TIMEOUT"
    assert outcome.exit_price == 103  # last candle's close


def test_ambiguous_same_candle_scores_conservatively_as_sl():
    candles = _candles([{"high": 106, "low": 97, "close": 100}])
    outcome = simulate_outcome(
        "LONG", entry_price=100, stop_loss=98,
        take_profits=[("TP1", 105)], candles=candles, commission_pct=0, slippage_pct=0,
    )
    assert outcome.exit_reason == "AMBIGUOUS"
    assert outcome.exit_price == 98
    assert outcome.r_multiple == pytest.approx(-1.0)


def test_mfe_mae_tracked_across_full_path():
    candles = _candles(
        [
            {"high": 103, "low": 99, "close": 102},   # favorable move, no hit
            {"high": 101, "low": 96, "close": 97},     # adverse move, no hit
            {"high": 112, "low": 97, "close": 111},    # hits TP1 at 110
        ]
    )
    outcome = simulate_outcome(
        "LONG", entry_price=100, stop_loss=95,
        take_profits=[("TP1", 110)], candles=candles, commission_pct=0, slippage_pct=0,
    )
    assert outcome.exit_reason == "TP1"
    assert outcome.mfe_r == pytest.approx(2.4)  # (112-100)/5
    assert outcome.mae_r == pytest.approx(0.8)   # (100-96)/5


def test_commission_reduces_r_multiple_by_expected_amount():
    candles = _candles([{"high": 106, "low": 99, "close": 105}])
    no_commission = simulate_outcome(
        "LONG", entry_price=100, stop_loss=98,
        take_profits=[("TP1", 105)], candles=candles, commission_pct=0, slippage_pct=0,
    )
    with_commission = simulate_outcome(
        "LONG", entry_price=100, stop_loss=98,
        take_profits=[("TP1", 105)], candles=candles, commission_pct=0.001, slippage_pct=0,
    )
    expected_commission_r = (2 * 0.001 * 100) / 2  # risk = 2
    assert no_commission.r_multiple - with_commission.r_multiple == pytest.approx(expected_commission_r)


def test_empty_candles_returns_none():
    outcome = simulate_outcome(
        "LONG", entry_price=100, stop_loss=98, take_profits=[("TP1", 105)],
        candles=pd.DataFrame(), commission_pct=0, slippage_pct=0,
    )
    assert outcome is None


def test_zero_risk_returns_none():
    candles = _candles([{"high": 101, "low": 99, "close": 100}])
    outcome = simulate_outcome(
        "LONG", entry_price=100, stop_loss=100, take_profits=[("TP1", 105)],
        candles=candles, commission_pct=0, slippage_pct=0,
    )
    assert outcome is None
