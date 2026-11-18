
import pytest

from src.data_engineering.normalizer import (
    normalize_timestamp,
    normalize_price_record
)


def test_normalize_timestamp():
    result = normalize_timestamp("2026-11-18T00:00:00Z")
    assert result == "2026-11-18T00:00:00Z"


def test_normalize_timestamp_with_timezone():
    result = normalize_timestamp("2026-11-18T05:30:00+05:30")
    assert result == "2026-11-18T00:00:00Z"


def test_normalize_price_record():
    record = {
        "commodity": "Copper",
        "timestamp": "2026-11-18T00:00:00Z",
        "open": 6.0,
        "high": 6.1,
        "low": 5.9,
        "close": 6.05,
        "volume": 1000
    }

    result = normalize_price_record(record)

    assert result["commodity"] == "copper"
    assert result["open"] == 6.0
    assert result["close"] == 6.05
    assert result["volume"] == 1000.0


def test_missing_required_field():
    record = {
        "commodity": "copper",
        "timestamp": "2026-11-18T00:00:00Z"
    }

    with pytest.raises(ValueError):
        normalize_price_record(record)
