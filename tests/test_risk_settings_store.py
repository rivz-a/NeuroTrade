"""Offline tests for risk_settings_store — local JSON persistence for
RiskSettings. Every test redirects config.RISK_SETTINGS_FILE to a pytest
tmp_path so nothing here ever touches the real risk_settings.json.
"""

from decimal import Decimal

import config
import risk_settings_store
from risk_manager import DEFAULT_RISK_SETTINGS, RiskSettings

VALID_RAW = {
    "account_balance_usdt": "250.00",
    "available_balance_usdt": "200.00",
    "risk_percent": "2.0",
    "leverage": 10,
    "margin_mode": "CROSS",
    "max_margin_percent": "40.0",
    "maker_fee_percent": "0.02",
    "taker_fee_percent": "0.05",
    "slippage_percent": "0.03",
    "min_risk_reward": "1.5",
    "entry_price_mode": "CONSERVATIVE",
    "quantity_step": "0.01",
    "price_step": "0.01",
    "minimum_order_notional_usdt": "2",
    "minimum_quantity": "0.01",
    "bingx_order_input_mode": "NOTIONAL_USDT",
}


def _use_tmp_file(monkeypatch, tmp_path):
    path = tmp_path / "risk_settings.json"
    monkeypatch.setattr(config, "RISK_SETTINGS_FILE", path)
    return path


def test_load_creates_file_with_defaults_when_missing(monkeypatch, tmp_path):
    path = _use_tmp_file(monkeypatch, tmp_path)
    assert not path.exists()
    settings = risk_settings_store.load()
    assert settings == DEFAULT_RISK_SETTINGS
    assert path.exists()


def test_save_then_load_round_trips(monkeypatch, tmp_path):
    _use_tmp_file(monkeypatch, tmp_path)
    saved = risk_settings_store.save(VALID_RAW)
    assert isinstance(saved, RiskSettings)
    assert saved.account_balance_usdt == Decimal("250.00")
    assert saved.leverage == 10
    assert saved.margin_mode.value == "CROSS"

    reloaded = risk_settings_store.load()
    assert reloaded == saved


def test_save_with_invalid_settings_returns_errors_and_does_not_write(monkeypatch, tmp_path):
    path = _use_tmp_file(monkeypatch, tmp_path)
    invalid = dict(VALID_RAW, account_balance_usdt="-5")
    result = risk_settings_store.save(invalid)
    assert isinstance(result, list)
    assert result
    assert not path.exists()


def test_save_with_invalid_settings_does_not_overwrite_existing_file(monkeypatch, tmp_path):
    path = _use_tmp_file(monkeypatch, tmp_path)
    risk_settings_store.save(VALID_RAW)
    before = path.read_text(encoding="utf-8")

    invalid = dict(VALID_RAW, leverage=0)
    result = risk_settings_store.save(invalid)
    assert isinstance(result, list)
    assert path.read_text(encoding="utf-8") == before


def test_build_is_pure_and_does_not_touch_disk(monkeypatch, tmp_path):
    path = _use_tmp_file(monkeypatch, tmp_path)
    result = risk_settings_store.build(VALID_RAW)
    assert isinstance(result, RiskSettings)
    assert not path.exists()


def test_merge_overrides_applies_partial_changes_without_persisting(monkeypatch, tmp_path):
    path = _use_tmp_file(monkeypatch, tmp_path)
    base = risk_settings_store.save(VALID_RAW)
    assert path.exists()
    before = path.read_text(encoding="utf-8")

    merged = risk_settings_store.merge_overrides(base, {"risk_percent": "5.0", "leverage": 20})
    assert isinstance(merged, RiskSettings)
    assert merged.risk_percent == Decimal("5.0")
    assert merged.leverage == 20
    # Everything else carried over from base unchanged.
    assert merged.account_balance_usdt == base.account_balance_usdt
    # Never touches disk.
    assert path.read_text(encoding="utf-8") == before


def test_merge_overrides_with_invalid_value_returns_errors():
    base = DEFAULT_RISK_SETTINGS
    result = risk_settings_store.merge_overrides(base, {"account_balance_usdt": "-1"})
    assert isinstance(result, list)
    assert result


def test_merge_overrides_ignores_unknown_fields():
    base = DEFAULT_RISK_SETTINGS
    merged = risk_settings_store.merge_overrides(base, {"api_key": "should-be-ignored", "risk_percent": "3.0"})
    assert isinstance(merged, RiskSettings)
    assert merged.risk_percent == Decimal("3.0")


def test_load_falls_back_to_defaults_on_corrupt_file(monkeypatch, tmp_path):
    path = _use_tmp_file(monkeypatch, tmp_path)
    path.write_text("{not valid json", encoding="utf-8")
    settings = risk_settings_store.load()
    assert settings == DEFAULT_RISK_SETTINGS
