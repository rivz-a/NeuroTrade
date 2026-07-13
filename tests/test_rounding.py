"""Offline tests for risk_manager's step-rounding helpers. No network, no AI."""

from decimal import Decimal

from risk_manager import round_down_to_step, round_to_step


def test_round_down_normal_case():
    assert round_down_to_step(Decimal("0.0847"), Decimal("0.001")) == Decimal("0.084")


def test_round_down_already_exact_multiple():
    assert round_down_to_step(Decimal("0.084"), Decimal("0.001")) == Decimal("0.084")


def test_round_down_below_one_step_goes_to_zero():
    assert round_down_to_step(Decimal("0.0009"), Decimal("0.001")) == Decimal("0.000")


def test_round_down_larger_step():
    assert round_down_to_step(Decimal("1.9999"), Decimal("0.5")) == Decimal("1.5")


def test_round_down_never_rounds_up():
    # Regardless of how close to the next step, ROUND_DOWN must never cross it —
    # rounding a position's quantity up would silently increase its risk.
    value = round_down_to_step(Decimal("0.0999999"), Decimal("0.001"))
    assert value <= Decimal("0.0999999")
    assert value == Decimal("0.099")


def test_round_down_step_zero_is_a_noop():
    assert round_down_to_step(Decimal("1.23456"), Decimal("0")) == Decimal("1.23456")


def test_round_to_step_rounds_half_up():
    assert round_to_step(Decimal("1782.146"), Decimal("0.01")) == Decimal("1782.15")
    assert round_to_step(Decimal("1782.144"), Decimal("0.01")) == Decimal("1782.14")
    assert round_to_step(Decimal("1782.145"), Decimal("0.01")) == Decimal("1782.15")


def test_round_to_step_zero_is_a_noop():
    assert round_to_step(Decimal("1782.146"), Decimal("0")) == Decimal("1782.146")
