"""Offline tests for market_data_engine.collect_snapshot and the two new
bingx_client wrappers (get_book_ticker, get_recent_trades). Every BingX
call is mocked (unittest.mock.patch), same style as test_instrument_rules.py
— no network, no AI. History files (oi_history.jsonl / market_data_history.jsonl)
are redirected to pytest tmp_path so nothing here touches real local state.
"""

from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pandas as pd
import pytest

import config
import market_data_engine
from bingx_client import APIError, BingXError

NOW = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc).timestamp()

INSTRUMENT_RULES = {
    "quantity_step": "0.01",
    "price_step": "0.01",
    "minimum_quantity": "0.01",
    "minimum_notional_usdt": "2",
    "maker_fee_percent": "0.02",
    "taker_fee_percent": "0.05",
}

BOOK_TICKER = {"bid_price": 3404.5, "bid_qty": 1.0, "ask_price": 3405.0, "ask_qty": 2.0, "time": int(NOW * 1000)}


def _klines_df(interval_seconds: int, count: int = 20, end_time: float = NOW, gap_before_last: bool = False, zero_volume_last: bool = False) -> pd.DataFrame:
    times = [datetime.fromtimestamp(end_time - i * interval_seconds, tz=timezone.utc) for i in range(count)][::-1]
    if gap_before_last:
        times[-1] = times[-1] + timedelta(seconds=interval_seconds * 5)
    rows = []
    for i, t in enumerate(times):
        volume = 0.0 if (zero_volume_last and i == len(times) - 1) else 10.0
        rows.append({"open": 3400.0, "high": 3410.0, "low": 3390.0, "close": 3405.0, "volume": volume, "time": t})
    return pd.DataFrame(rows)


def _orderbook(age_seconds: float = 1.0):
    return {
        "bids": [[3404.5, 1.0]],
        "asks": [[3405.0, 2.0]],
        "timestamp_ms": int((NOW - age_seconds) * 1000),
    }


def _use_tmp_history(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "OI_HISTORY_FILE", tmp_path / "oi_history.jsonl")
    monkeypatch.setattr(config, "MARKET_DATA_HISTORY_FILE", tmp_path / "market_data_history.jsonl")


def _patch_good_bingx(zero_volume_last=False, gap_before_last=False, orderbook_age=1.0, orderbook=Ellipsis):
    ob = orderbook if orderbook is not Ellipsis else _orderbook(orderbook_age)
    return [
        patch("bingx_client.get_price", return_value=3405.0),
        patch(
            "bingx_client.get_klines",
            side_effect=lambda symbol, interval, limit: _klines_df(
                market_data_engine._interval_seconds(interval),
                zero_volume_last=zero_volume_last,
                gap_before_last=gap_before_last,
            ),
        ),
        patch("bingx_client.get_book_ticker", return_value=BOOK_TICKER),
        patch("bingx_client.get_orderbook", return_value=ob),
        patch("bingx_client.get_funding_rate", return_value=0.0001),
        patch("bingx_client.get_funding_rate_history", return_value=[{"rate": 0.0001, "time": int(NOW * 1000)}]),
        patch("bingx_client.get_open_interest", return_value=123456.0),
        patch("bingx_client.get_ticker_24h", return_value={"price_change_percent": 1.2, "high_price": 3500.0, "low_price": 3300.0, "volume": 999.0, "quote_volume": 1000000.0}),
        patch("bingx_client.get_recent_trades", return_value=[{"price": 3405.0, "qty": 1.0, "time": int(NOW * 1000), "is_buyer_maker": True}]),
        patch("bingx_client.get_instrument_rules", return_value=INSTRUMENT_RULES),
    ]


class _Patchers:
    """Applies a list of unittest.mock.patch context managers together."""

    def __init__(self, patchers):
        self.patchers = patchers

    def __enter__(self):
        for p in self.patchers:
            p.start()
        return self

    def __exit__(self, *exc):
        for p in reversed(self.patchers):
            p.stop()


def test_good_snapshot(monkeypatch, tmp_path):
    _use_tmp_history(monkeypatch, tmp_path)
    with _Patchers(_patch_good_bingx()):
        snap = market_data_engine.collect_snapshot("ETHUSDT", now=NOW)
    assert snap.data_quality == "GOOD"
    assert snap.quality_issues == []
    assert snap.price == 3405.0
    assert snap.bid == 3404.5 and snap.ask == 3405.0
    assert snap.spread == pytest.approx(0.5)
    assert set(snap.timeframes.keys()) == set(config.TIMEFRAMES)
    assert snap.orderbook is not None
    assert snap.instrument_rules["quantity_step"] == "0.01"


def test_price_unavailable_is_no_trade(monkeypatch, tmp_path):
    _use_tmp_history(monkeypatch, tmp_path)
    patchers = _patch_good_bingx()
    patchers[0] = patch("bingx_client.get_price", side_effect=BingXError("down"))
    with _Patchers(patchers):
        snap = market_data_engine.collect_snapshot("ETHUSDT", now=NOW)
    assert snap.data_quality == "NO_TRADE"
    assert any("цену" in i for i in snap.quality_issues)
    assert snap.price is None


def test_missing_base_timeframe_candles_is_no_trade(monkeypatch, tmp_path):
    _use_tmp_history(monkeypatch, tmp_path)
    patchers = _patch_good_bingx()

    def raise_for_base(symbol, interval, limit):
        if interval == config.TIMEFRAMES[0]:
            raise BingXError("no data")
        return _klines_df(market_data_engine._interval_seconds(interval))

    patchers[1] = patch("bingx_client.get_klines", side_effect=raise_for_base)
    with _Patchers(patchers):
        snap = market_data_engine.collect_snapshot("ETHUSDT", now=NOW)
    assert snap.data_quality == "NO_TRADE"
    assert config.TIMEFRAMES[0] not in snap.timeframes


def test_candle_gap_is_no_trade(monkeypatch, tmp_path):
    _use_tmp_history(monkeypatch, tmp_path)
    with _Patchers(_patch_good_bingx(gap_before_last=True)):
        snap = market_data_engine.collect_snapshot("ETHUSDT", now=NOW)
    assert snap.data_quality == "NO_TRADE"
    assert any("Пропуск свечи" in i for i in snap.quality_issues)


def test_stale_last_candle_is_no_trade(monkeypatch, tmp_path):
    _use_tmp_history(monkeypatch, tmp_path)
    # now far in the future relative to the candles' own timestamps (built around NOW)
    future = NOW + 10_000
    with _Patchers(_patch_good_bingx()):
        snap = market_data_engine.collect_snapshot("ETHUSDT", now=future)
    assert snap.data_quality == "NO_TRADE"
    assert any("устарела" in i for i in snap.quality_issues)


def test_zero_volume_alone_is_degraded_not_no_trade(monkeypatch, tmp_path):
    _use_tmp_history(monkeypatch, tmp_path)
    with _Patchers(_patch_good_bingx(zero_volume_last=True)):
        snap = market_data_engine.collect_snapshot("ETHUSDT", now=NOW)
    assert snap.data_quality == "DEGRADED"
    assert any("Нулевой объём" in i for i in snap.quality_issues)


def test_missing_orderbook_is_degraded(monkeypatch, tmp_path):
    _use_tmp_history(monkeypatch, tmp_path)
    patchers = _patch_good_bingx()
    patchers[3] = patch("bingx_client.get_orderbook", side_effect=BingXError("down"))
    with _Patchers(patchers):
        snap = market_data_engine.collect_snapshot("ETHUSDT", now=NOW)
    assert snap.data_quality == "DEGRADED"
    assert snap.orderbook is None
    assert any("Стакан недоступен" in i for i in snap.quality_issues)


def test_stale_orderbook_is_degraded(monkeypatch, tmp_path):
    _use_tmp_history(monkeypatch, tmp_path)
    stale_ob = _orderbook(age_seconds=config.MARKET_DATA_MAX_ORDERBOOK_AGE_SECONDS + 5)
    with _Patchers(_patch_good_bingx(orderbook=stale_ob)):
        snap = market_data_engine.collect_snapshot("ETHUSDT", now=NOW)
    assert snap.data_quality == "DEGRADED"
    assert any("Стакан устарел" in i for i in snap.quality_issues)


def test_instrument_spec_change_is_degraded_on_second_call(monkeypatch, tmp_path):
    _use_tmp_history(monkeypatch, tmp_path)
    with _Patchers(_patch_good_bingx()):
        first = market_data_engine.collect_snapshot("ETHUSDT", now=NOW)
    assert first.data_quality == "GOOD"

    changed_rules = dict(INSTRUMENT_RULES, quantity_step="0.1")
    patchers = _patch_good_bingx()
    patchers[-1] = patch("bingx_client.get_instrument_rules", return_value=changed_rules)
    with _Patchers(patchers):
        second = market_data_engine.collect_snapshot("ETHUSDT", now=NOW + 60)
    assert second.data_quality == "DEGRADED"
    assert any("Спецификация инструмента изменилась" in i for i in second.quality_issues)


def test_history_grows_across_calls(monkeypatch, tmp_path):
    _use_tmp_history(monkeypatch, tmp_path)
    with _Patchers(_patch_good_bingx()):
        first = market_data_engine.collect_snapshot("ETHUSDT", now=NOW)
        assert first.orderbook_history == []
        assert first.open_interest_history == []

        second = market_data_engine.collect_snapshot("ETHUSDT", now=NOW + 60)
        assert len(second.orderbook_history) == 1
        assert len(second.open_interest_history) == 1


def test_invalid_symbol_returns_no_trade_without_raising():
    snap = market_data_engine.collect_snapshot("!!!not-a-symbol!!!", now=NOW)
    assert snap.data_quality == "NO_TRADE"
    assert snap.quality_issues


def test_get_book_ticker_parses_real_shape():
    payload = {"book_ticker": {"bid_price": "3404.5", "bid_qty": "1", "ask_price": "3405", "ask_qty": "2", "time": 123}}
    with patch("bingx_client._get", return_value=payload):
        result = market_data_engine.bingx_client.get_book_ticker("ETH-USDT")
    assert result == {"bid_price": 3404.5, "bid_qty": 1.0, "ask_price": 3405.0, "ask_qty": 2.0, "time": 123}


def test_get_book_ticker_returns_none_on_failure():
    with patch("bingx_client._get", side_effect=APIError("down")):
        assert market_data_engine.bingx_client.get_book_ticker("ETH-USDT") is None


def test_get_recent_trades_parses_and_sorts():
    payload = [
        {"price": "3405", "qty": "1", "time": 200, "isBuyerMaker": True},
        {"price": "3404", "qty": "2", "time": 100, "isBuyerMaker": False},
    ]
    with patch("bingx_client._get", return_value=payload):
        result = market_data_engine.bingx_client.get_recent_trades("ETH-USDT", limit=5)
    assert [t["time"] for t in result] == [100, 200]
    assert result[0]["is_buyer_maker"] is False


def test_get_recent_trades_returns_empty_on_failure():
    with patch("bingx_client._get", side_effect=APIError("down")):
        assert market_data_engine.bingx_client.get_recent_trades("ETH-USDT") == []
