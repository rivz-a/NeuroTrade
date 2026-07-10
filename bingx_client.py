"""Thin read-only client for BingX public (unauthenticated) market-data endpoints.

Only GET requests against public "quote" endpoints are made here. Nothing in
this module places orders, signs requests, or talks to any account-level API.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import requests

from config import BASE_URL, REQUEST_TIMEOUT


class BingXError(Exception):
    """Base class for all BingX client errors."""


class NetworkError(BingXError):
    """Raised when BingX cannot be reached at all (DNS, connection, timeout)."""


class RateLimitError(BingXError):
    """Raised when BingX responds with HTTP 429 / rate-limit error code."""


class InvalidSymbolError(BingXError):
    """Raised when the requested symbol is not recognized by BingX."""


class NoDataError(BingXError):
    """Raised when BingX responds successfully but with empty/missing data."""


class APIError(BingXError):
    """Raised for any other non-successful BingX API response."""


_SESSION = requests.Session()


def _get(path: str, params: dict[str, Any]) -> Any:
    url = f"{BASE_URL}{path}"
    try:
        response = _SESSION.get(url, params=params, timeout=REQUEST_TIMEOUT)
    except requests.exceptions.Timeout as exc:
        raise NetworkError(f"Тайм-аут запроса к BingX: {url}") from exc
    except requests.exceptions.ConnectionError as exc:
        raise NetworkError(
            f"Не удалось подключиться к BingX. Проверьте интернет-соединение. ({url})"
        ) from exc
    except requests.exceptions.RequestException as exc:
        raise NetworkError(f"Ошибка сети при обращении к BingX: {exc}") from exc

    if response.status_code == 429:
        raise RateLimitError(
            "BingX вернул ошибку 429: превышен лимит запросов. Повторите попытку позже."
        )
    if response.status_code >= 500:
        raise APIError(f"BingX недоступен (HTTP {response.status_code}).")
    if response.status_code >= 400:
        raise APIError(f"Ошибка запроса к BingX (HTTP {response.status_code}): {response.text[:200]}")

    try:
        payload = response.json()
    except ValueError as exc:
        raise APIError("BingX вернул некорректный (не-JSON) ответ.") from exc

    code = payload.get("code")
    if code not in (0, None):
        msg = str(payload.get("msg", "")).lower()
        if code in (100202, 100400, 100001) or "symbol" in msg or "not exist" in msg:
            raise InvalidSymbolError(
                f"BingX не распознал символ. Ответ биржи: {payload.get('msg', code)}"
            )
        if code in (100410, 100403) or "frequent" in msg or "limit" in msg:
            raise RateLimitError(f"BingX сообщил о превышении лимита запросов: {payload.get('msg', code)}")
        raise APIError(f"BingX вернул ошибку (code={code}): {payload.get('msg', '')}")

    data = payload.get("data")
    if data is None:
        raise NoDataError(f"BingX не вернул данные для запроса {path}.")
    return data


def get_price(symbol: str) -> float:
    data = _get("/openApi/swap/v2/quote/price", {"symbol": symbol})
    price = data.get("price") if isinstance(data, dict) else None
    if price is None:
        raise NoDataError(f"BingX не вернул текущую цену для {symbol}.")
    return float(price)


def get_klines(symbol: str, interval: str, limit: int) -> pd.DataFrame:
    data = _get(
        "/openApi/swap/v3/quote/klines",
        {"symbol": symbol, "interval": interval, "limit": limit},
    )
    if not data:
        raise NoDataError(f"BingX не вернул свечи {interval} для {symbol}.")

    df = pd.DataFrame(data)
    required_cols = {"open", "high", "low", "close", "volume", "time"}
    if not required_cols.issubset(df.columns):
        raise NoDataError(
            f"Неожиданный формат свечей от BingX для {symbol} ({interval}): "
            f"колонки {list(df.columns)}"
        )

    for col in ("open", "high", "low", "close", "volume"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["time"] = pd.to_datetime(pd.to_numeric(df["time"], errors="coerce"), unit="ms", utc=True)
    df = df.dropna(subset=["open", "high", "low", "close", "volume"])
    df = df.sort_values("time").reset_index(drop=True)
    if df.empty:
        raise NoDataError(f"После обработки свечей {interval} для {symbol} данных не осталось.")
    return df


def get_orderbook(symbol: str, depth: int) -> dict[str, list[list[float]]]:
    data = _get("/openApi/swap/v2/quote/depth", {"symbol": symbol, "limit": depth})
    bids = data.get("bids", []) if isinstance(data, dict) else []
    asks = data.get("asks", []) if isinstance(data, dict) else []
    parsed_bids = [[float(p), float(q)] for p, q in bids]
    parsed_asks = [[float(p), float(q)] for p, q in asks]
    return {"bids": parsed_bids, "asks": parsed_asks}


def get_funding_rate(symbol: str) -> float | None:
    try:
        data = _get("/openApi/swap/v2/quote/premiumIndex", {"symbol": symbol})
    except (NoDataError, APIError):
        return None
    if not isinstance(data, dict):
        return None
    rate = data.get("lastFundingRate")
    return float(rate) if rate is not None else None


def get_funding_rate_history(symbol: str, limit: int = 5) -> list[dict]:
    """Last `limit` actual funding-rate settlements (BingX pays these ~every
    8h), oldest to newest is NOT guaranteed by the API — caller should not
    assume order. Distinct from `get_funding_rate`, which is the current
    predicted rate for the next settlement, not a history.
    """
    try:
        data = _get("/openApi/swap/v2/quote/fundingRate", {"symbol": symbol, "limit": limit})
    except (NoDataError, APIError):
        return []
    if not isinstance(data, list):
        return []
    history: list[dict] = []
    for item in data:
        try:
            history.append({"rate": float(item["fundingRate"]), "time": int(item["fundingTime"])})
        except (KeyError, TypeError, ValueError):
            continue
    history.sort(key=lambda h: h["time"])
    return history


def get_open_interest(symbol: str) -> float | None:
    try:
        data = _get("/openApi/swap/v2/quote/openInterest", {"symbol": symbol})
    except (NoDataError, APIError):
        return None
    if not isinstance(data, dict):
        return None
    oi = data.get("openInterest")
    return float(oi) if oi is not None else None


def get_ticker_24h(symbol: str) -> dict[str, float] | None:
    try:
        data = _get("/openApi/swap/v2/quote/ticker", {"symbol": symbol})
    except (NoDataError, APIError):
        return None
    if not isinstance(data, dict):
        return None
    try:
        return {
            "price_change_percent": float(data.get("priceChangePercent", 0.0)),
            "high_price": float(data.get("highPrice", 0.0)),
            "low_price": float(data.get("lowPrice", 0.0)),
            "volume": float(data.get("volume", 0.0)),
            "quote_volume": float(data.get("quoteVolume", 0.0)),
        }
    except (TypeError, ValueError):
        return None
