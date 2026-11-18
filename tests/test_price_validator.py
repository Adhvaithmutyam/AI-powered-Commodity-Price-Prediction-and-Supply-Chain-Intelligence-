import pytest


from copy import deepcopy

from src.data_engineering.validator import validate_price_record


@pytest.fixture
def valid_record():
    return {
        "commodity": "copper",
        "timestamp": "2026-11-17T00:00:00Z",
        "open": 6.0,
        "high": 6.2,
        "low": 5.8,
        "close": 6.1,
        "volume": 1000,
        "ticker": "HG=F",
        "benchmark": "COMEX Copper Futures",
        "unit": "USD/lb",
        "source": "Yahoo Finance"
    }


def test_valid_price_record(valid_record):
    is_valid, errors = validate_price_record(valid_record)

    assert is_valid is True
    assert errors == []


def test_invalid_ohlc_relationship(valid_record):
    record = deepcopy(valid_record)
    record["high"] = 5.5

    is_valid, errors = validate_price_record(record)

    assert is_valid is False
    assert any("High" in error for error in errors)


def test_negative_volume(valid_record):
    record = deepcopy(valid_record)
    record["volume"] = -100

    is_valid, errors = validate_price_record(record)

    assert is_valid is False
    assert any("Volume" in error for error in errors)


def test_incorrect_ticker(valid_record):
    record = deepcopy(valid_record)
    record["ticker"] = "GC=F"

    is_valid, errors = validate_price_record(record)

    assert is_valid is False
    assert "Incorrect ticker for commodity" in errors


def test_missing_required_field(valid_record):
    record = deepcopy(valid_record)
    del record["close"]

    is_valid, errors = validate_price_record(record)

    assert is_valid is False
    assert "Missing field: close" in errors
