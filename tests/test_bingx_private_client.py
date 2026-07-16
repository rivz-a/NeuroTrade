"""Offline tests for bingx_private_client.py — no network. DRY_RUN mode
(the hard default) must build and sign every request without ever calling
`requests.Session.request`; a monkeypatched `request` that raises on any
call proves that.
"""

from __future__ import annotations

import hashlib
import hmac

import pytest

import bingx_private_client as bpc
import config


@pytest.fixture(autouse=True)
def _credentials(monkeypatch):
    monkeypatch.setattr(config, "BINGX_API_KEY", "test-api-key")
    monkeypatch.setattr(config, "BINGX_API_SECRET", "test-api-secret")
    monkeypatch.setattr(config, "EXECUTION_DRY_RUN", True)


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    def _boom(*args, **kwargs):
        raise AssertionError("requests.Session.request should never be called in DRY_RUN mode")

    monkeypatch.setattr(bpc._SESSION, "request", _boom)


# ---------------------------------------------------------------------------
# _sign
# ---------------------------------------------------------------------------


def test_sign_hand_computed():
    params = {"symbol": "ETH-USDT", "timestamp": 1000}
    expected_qs = "symbol=ETH-USDT&timestamp=1000"
    expected = hmac.new(b"test-api-secret", expected_qs.encode(), hashlib.sha256).hexdigest()
    assert bpc._sign(params) == expected


def test_sign_sorts_alphabetically_regardless_of_input_order():
    a = bpc._sign({"b": 2, "a": 1})
    b = bpc._sign({"a": 1, "b": 2})
    assert a == b


# ---------------------------------------------------------------------------
# _request — DRY_RUN / credentials
# ---------------------------------------------------------------------------


def test_request_missing_credentials_raises_signature_error(monkeypatch):
    monkeypatch.setattr(config, "BINGX_API_KEY", "")
    with pytest.raises(bpc.SignatureError):
        bpc._request("GET", "/openApi/swap/v2/user/balance", {})


def test_request_dry_run_raises_before_network_call():
    with pytest.raises(bpc.DryRunNotSent) as exc_info:
        bpc._request("GET", "/openApi/swap/v2/user/balance", {"symbol": "ETH-USDT"})
    err = exc_info.value
    assert err.method == "GET"
    assert err.url.endswith("/openApi/swap/v2/user/balance")
    assert err.params["symbol"] == "ETH-USDT"
    assert "signature" in err.params
    assert err.headers["X-BX-APIKEY"] == "test-api-key"


def test_request_live_mode_calls_network(monkeypatch):
    monkeypatch.setattr(config, "EXECUTION_DRY_RUN", False)

    class _FakeResponse:
        status_code = 200

        def json(self):
            return {"code": 0, "data": {"ok": True}}

    monkeypatch.setattr(bpc._SESSION, "request", lambda *a, **k: _FakeResponse())
    data = bpc._request("GET", "/openApi/swap/v2/user/balance", {})
    assert data == {"ok": True}


# ---------------------------------------------------------------------------
# Endpoint wrappers — DRY_RUN request shape
# ---------------------------------------------------------------------------


def _dry_run_params(fn, *args, **kwargs) -> dict:
    with pytest.raises(bpc.DryRunNotSent) as exc_info:
        fn(*args, **kwargs)
    return exc_info.value


def test_place_order_builds_correct_params():
    err = _dry_run_params(
        bpc.place_order, "ETH-USDT", "BUY", "LONG", "MARKET", "0.01", client_order_id="neurotrade-1"
    )
    assert err.method == "POST"
    assert err.url.endswith("/openApi/swap/v2/trade/order")
    assert err.params["symbol"] == "ETH-USDT"
    assert err.params["side"] == "BUY"
    assert err.params["positionSide"] == "LONG"
    assert err.params["type"] == "MARKET"
    assert err.params["quantity"] == "0.01"
    assert err.params["clientOrderID"] == "neurotrade-1"
    assert "price" not in err.params
    assert "stopPrice" not in err.params


def test_place_order_includes_price_and_stop_price_when_given():
    err = _dry_run_params(
        bpc.place_order, "ETH-USDT", "SELL", "SHORT", "STOP_MARKET", "0.01", price="100.5", stop_price="99.0"
    )
    assert err.params["price"] == "100.5"
    assert err.params["stopPrice"] == "99.0"


def test_cancel_order_uses_delete():
    err = _dry_run_params(bpc.cancel_order, "ETH-USDT", "12345")
    assert err.method == "DELETE"
    assert err.url.endswith("/openApi/swap/v2/trade/order")
    assert err.params["orderId"] == "12345"


def test_set_leverage_params():
    err = _dry_run_params(bpc.set_leverage, "ETH-USDT", "LONG", 5)
    assert err.method == "POST"
    assert err.url.endswith("/openApi/swap/v2/trade/leverage")
    assert err.params["leverage"] == 5
    assert err.params["side"] == "LONG"


def test_set_margin_type_refuses_nothing_itself_but_passes_value_through():
    err = _dry_run_params(bpc.set_margin_type, "ETH-USDT", "ISOLATED")
    assert err.url.endswith("/openApi/swap/v2/trade/marginType")
    assert err.params["marginType"] == "ISOLATED"


def test_get_positions_params_with_symbol():
    err = _dry_run_params(bpc.get_positions, "ETH-USDT")
    assert err.method == "GET"
    assert err.url.endswith("/openApi/swap/v2/user/positions")
    assert err.params["symbol"] == "ETH-USDT"


def test_get_positions_params_without_symbol():
    err = _dry_run_params(bpc.get_positions)
    assert "symbol" not in err.params


def test_get_balance_params():
    err = _dry_run_params(bpc.get_balance)
    assert err.url.endswith("/openApi/swap/v2/user/balance")


def test_get_open_orders_params():
    err = _dry_run_params(bpc.get_open_orders, "ETH-USDT")
    assert err.url.endswith("/openApi/swap/v2/trade/openOrders")
    assert err.params["symbol"] == "ETH-USDT"
