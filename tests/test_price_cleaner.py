
from src.data_engineering.cleaner import clean_price_records


def sample_record():
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


def test_valid_record():
    records = [sample_record()]

    valid, invalid = clean_price_records(records)

    assert len(valid) == 1
    assert len(invalid) == 0


def test_duplicate_records():
    record = sample_record()
    records = [record, record.copy()]

    valid, invalid = clean_price_records(records)

    assert len(valid) == 1
    assert len(invalid) == 0


def test_invalid_record():
    record = sample_record()
    record["high"] = 5.0

    valid, invalid = clean_price_records([record])

    assert len(valid) == 0
    assert len(invalid) == 1
