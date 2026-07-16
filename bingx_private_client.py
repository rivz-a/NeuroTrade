"""Thin client for BingX Perpetual Futures PRIVATE (authenticated,
account-mutating) REST endpoints — order placement, leverage/margin-mode
changes, balance/position reads. Unlike `bingx_client.py` (public,
unauthenticated market data only), every call here can move real money once
`config.EXECUTION_DRY_RUN` is False.

Signing (HMAC-SHA256 hex over an alphabetically-sorted query string,
`X-BX-APIKEY` header, `timestamp`+`recvWindow`) is confirmed with reasonable
confidence from multiple independent public sources. The exact
`stopLoss`/`takeProfit` sub-parameter shape and the client-order-id
parameter name for exchange-side idempotency are NOT confirmed — BingX's own
docs (bingx-api.github.io/docs) are a JS-rendered SPA this tooling can't
extract, and its GitHub doc repos redirect back to that same page. See the
Stage 11 plan for the full account of what was and wasn't verifiable.

Because of that gap, `config.EXECUTION_DRY_RUN` defaults to True: every call
below still builds the full method/URL/params and signs it exactly as a live
call would, but `_request` raises `DryRunNotSent` (carrying the complete,
inspectable request) instead of ever calling `requests.*`. Flip
EXECUTION_DRY_RUN=false only after independently verifying the request shape
against BingX's real documentation or support — and only ever at the user's
own explicit, in-the-moment decision, never automatically.

order_manager.py's own idempotency (a real_orders row written to journal_db
BEFORE any call here) does not depend on this module's client_order_id
parameter being correct — so even if that parameter name turns out wrong,
a crash mid-flow still can't silently double-place an order from this app's
side.
"""

from __future__ import annotations

import hashlib
import hmac
import time
from typing import Any, Literal
from urllib.parse import urlencode

import requests

import bingx_client
import config

_SESSION = requests.Session()


class BingXPrivateError(bingx_client.BingXError):
    """Base class for all bingx_private_client errors."""


class SignatureError(BingXPrivateError):
    """Raised when BINGX_API_KEY/BINGX_API_SECRET are missing or empty —
    refused before any request is built, live or dry-run."""


class DryRunNotSent(BingXPrivateError):
    """Raised in DRY_RUN mode (the default) instead of ever calling
    `requests.*` — carries the fully built, signed request so a caller can
    inspect/log it. Never raised when config.EXECUTION_DRY_RUN is False.
    """

    def __init__(self, method: str, url: str, params: dict[str, Any], headers: dict[str, str]) -> None:
        self.method = method
        self.url = url
        self.params = params
        self.headers = headers
        masked_key = headers.get("X-BX-APIKEY", "")
        masked = (masked_key[:4] + "…") if masked_key else ""
        super().__init__(
            f"DRY_RUN: would send {method} {url} params={params} X-BX-APIKEY={masked}"
        )


def _sign(params: dict[str, Any]) -> str:
    """Alphabetically-sorted `key=value` query string, HMAC-SHA256 keyed by
    BINGX_API_SECRET, hex digest. Pure function, no I/O — independently
    confirmed by multiple sources, the one piece of the signing scheme with
    genuinely high confidence.
    """
    query_string = urlencode(sorted(params.items()))
    return hmac.new(config.BINGX_API_SECRET.encode(), query_string.encode(), hashlib.sha256).hexdigest()


def _timestamp_params() -> dict[str, Any]:
    return {"timestamp": int(time.time() * 1000), "recvWindow": config.BINGX_PRIVATE_RECV_WINDOW_MS}


def _request(method: Literal["GET", "POST", "DELETE"], path: str, params: dict[str, Any]) -> Any:
    if not config.BINGX_API_KEY or not config.BINGX_API_SECRET:
        raise SignatureError("BINGX_API_KEY/BINGX_API_SECRET не заданы — приватный запрос к BingX невозможен.")

    full_params = {**params, **_timestamp_params()}
    full_params["signature"] = _sign(full_params)
    url = f"{config.BASE_URL}{path}"
    headers = {"X-BX-APIKEY": config.BINGX_API_KEY}

    if config.EXECUTION_DRY_RUN:
        raise DryRunNotSent(method, url, full_params, headers)

    try:
        response = _SESSION.request(method, url, params=full_params, headers=headers, timeout=config.REQUEST_TIMEOUT)
    except requests.exceptions.Timeout as exc:
        raise bingx_client.NetworkError(f"Тайм-аут запроса к BingX: {url}") from exc
    except requests.exceptions.ConnectionError as exc:
        raise bingx_client.NetworkError(
            f"Не удалось подключиться к BingX. Проверьте интернет-соединение. ({url})"
        ) from exc
    except requests.exceptions.RequestException as exc:
        raise bingx_client.NetworkError(f"Ошибка сети при обращении к BingX: {exc}") from exc

    if response.status_code == 429:
        raise bingx_client.RateLimitError(
            "BingX вернул ошибку 429: превышен лимит запросов. Повторите попытку позже."
        )
    if response.status_code >= 500:
        raise bingx_client.APIError(f"BingX недоступен (HTTP {response.status_code}).")
    if response.status_code >= 400:
        raise bingx_client.APIError(f"Ошибка запроса к BingX (HTTP {response.status_code}): {response.text[:200]}")

    try:
        payload = response.json()
    except ValueError as exc:
        raise bingx_client.APIError("BingX вернул некорректный (не-JSON) ответ.") from exc

    code = payload.get("code")
    if code not in (0, None):
        raise bingx_client.APIError(f"BingX вернул ошибку (code={code}): {payload.get('msg', '')}")

    return payload.get("data")


# ---------------------------------------------------------------------------
# Endpoint wrappers
#
# Path/param confidence: HIGH for place_order/set_leverage/set_margin_type/
# get_positions/get_balance (cross-confirmed by independent SDKs/docs for
# adjacent BingX products). LOWER for cancel_order (path confirmed, method
# assumed DELETE by REST convention) and get_open_orders (path is a
# best guess, not independently cross-confirmed) — flagged in their own
# docstrings. DRY_RUN mode exists specifically so every one of these can be
# inspected before it is ever trusted with real funds.
# ---------------------------------------------------------------------------


def place_order(
    symbol: str,
    side: Literal["BUY", "SELL"],
    position_side: Literal["LONG", "SHORT"],
    order_type: Literal["MARKET", "LIMIT", "STOP_MARKET", "TAKE_PROFIT_MARKET"],
    quantity: str,
    *,
    price: str | None = None,
    stop_price: str | None = None,
    client_order_id: str | None = None,
) -> Any:
    """POST /openApi/swap/v2/trade/order — places one entry, stop-loss, or
    take-profit order. Callers place entry/SL/TP as three separate calls
    (per the Stage 11 spec), not a single bracket-order request.
    """
    params: dict[str, Any] = {
        "symbol": symbol,
        "side": side,
        "positionSide": position_side,
        "type": order_type,
        "quantity": quantity,
    }
    if price is not None:
        params["price"] = price
    if stop_price is not None:
        params["stopPrice"] = stop_price
    if client_order_id is not None:
        params["clientOrderID"] = client_order_id
    return _request("POST", "/openApi/swap/v2/trade/order", params)


def cancel_order(symbol: str, order_id: str) -> Any:
    """DELETE /openApi/swap/v2/trade/order."""
    return _request("DELETE", "/openApi/swap/v2/trade/order", {"symbol": symbol, "orderId": order_id})


def set_leverage(symbol: str, side: Literal["LONG", "SHORT"], leverage: int) -> Any:
    """POST /openApi/swap/v2/trade/leverage."""
    return _request(
        "POST", "/openApi/swap/v2/trade/leverage", {"symbol": symbol, "side": side, "leverage": leverage}
    )


def set_margin_type(symbol: str, margin_type: Literal["ISOLATED", "CROSSED"]) -> Any:
    """POST /openApi/swap/v2/trade/marginType. Callers (order_manager.py)
    hard-refuse anything but ISOLATED before this is ever called — see the
    Stage 11 plan's "только Isolated" safeguard.
    """
    return _request("POST", "/openApi/swap/v2/trade/marginType", {"symbol": symbol, "marginType": margin_type})


def get_positions(symbol: str | None = None) -> Any:
    """GET /openApi/swap/v2/user/positions."""
    return _request("GET", "/openApi/swap/v2/user/positions", {"symbol": symbol} if symbol else {})


def get_balance() -> Any:
    """GET /openApi/swap/v2/user/balance."""
    return _request("GET", "/openApi/swap/v2/user/balance", {})


def get_open_orders(symbol: str | None = None) -> Any:
    """GET /openApi/swap/v2/trade/openOrders — lowest-confidence endpoint
    of this module; the path was not independently cross-confirmed the way
    the others were. Verify against real BingX documentation before relying
    on it for anything beyond DRY_RUN inspection.
    """
    return _request("GET", "/openApi/swap/v2/trade/openOrders", {"symbol": symbol} if symbol else {})
