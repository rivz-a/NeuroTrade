"""Offline tests for bingx_client.get_instrument_rules — the BingX
`_get` call is mocked; no real network request happens here (the one real
verification of this function against the live endpoint was done manually
during development, not in the test suite).
"""

from decimal import Decimal
from unittest.mock import patch

from bingx_client import BingXError, get_instrument_rules

CONTRACTS_RESPONSE = [
    {
        "symbol": "ETH-USDT",
        "quantityPrecision": 2,
        "pricePrecision": 2,
        "tradeMinQuantity": "0.01",
        "tradeMinUSDT": "2",
        "makerFeeRate": "0.0002",
        "takerFeeRate": "0.0005",
    },
    {
        "symbol": "BTC-USDT",
        "quantityPrecision": 3,
        "pricePrecision": 1,
        "tradeMinQuantity": "0.001",
        "tradeMinUSDT": "2",
        "makerFeeRate": "0.0002",
        "takerFeeRate": "0.0005",
    },
]


def test_finds_requested_symbol_and_converts_precision_to_step():
    with patch("bingx_client._get", return_value=CONTRACTS_RESPONSE):
        rules = get_instrument_rules("ETH-USDT")
    assert rules is not None
    assert rules["quantity_step"] == Decimal("0.01")
    assert rules["price_step"] == Decimal("0.01")
    assert rules["minimum_quantity"] == Decimal("0.01")
    assert rules["minimum_notional_usdt"] == Decimal("2")
    assert rules["maker_fee_percent"] == Decimal("0.02")
    assert rules["taker_fee_percent"] == Decimal("0.05")


def test_different_precision_produces_different_step():
    with patch("bingx_client._get", return_value=CONTRACTS_RESPONSE):
        rules = get_instrument_rules("BTC-USDT")
    assert rules is not None
    assert rules["quantity_step"] == Decimal("0.001")
    assert rules["price_step"] == Decimal("0.1")


def test_symbol_not_in_response_returns_none():
    with patch("bingx_client._get", return_value=CONTRACTS_RESPONSE):
        assert get_instrument_rules("SOL-USDT") is None


def test_network_failure_returns_none():
    with patch("bingx_client._get", side_effect=BingXError("timeout")):
        assert get_instrument_rules("ETH-USDT") is None


def test_non_list_response_returns_none():
    with patch("bingx_client._get", return_value={"unexpected": "shape"}):
        assert get_instrument_rules("ETH-USDT") is None


def test_malformed_item_missing_precision_returns_none():
    malformed = [{"symbol": "ETH-USDT", "tradeMinQuantity": "0.01"}]
    with patch("bingx_client._get", return_value=malformed):
        assert get_instrument_rules("ETH-USDT") is None


def test_missing_fee_fields_default_to_zero():
    minimal = [
        {
            "symbol": "ETH-USDT",
            "quantityPrecision": 2,
            "pricePrecision": 2,
        }
    ]
    with patch("bingx_client._get", return_value=minimal):
        rules = get_instrument_rules("ETH-USDT")
    assert rules is not None
    assert rules["minimum_quantity"] == Decimal("0")
    assert rules["minimum_notional_usdt"] == Decimal("0")
    assert rules["maker_fee_percent"] == Decimal("0")
    assert rules["taker_fee_percent"] == Decimal("0")
