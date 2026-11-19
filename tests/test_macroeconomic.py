import json
from pathlib import Path

import pytest

from src.data_engineering.macroeconomic.normalizer import normalize_macro_record, normalize_macro_records
from src.data_engineering.macroeconomic.validator import validate_macro_record, validate_macro_records
from src.data_engineering.macroeconomic.storage import save_macro_records, load_macro_records


def sample_record():
    return {
        "country_code": "USA",
        "indicator": "gdp_growth",
        "indicator_code": "NY.GDP.MKTP.KD.ZG",
        "year": 2020,
        "value": 2.5,
        "unit": "GDP growth (annual %)",
        "source": "World Bank"
    }


def test_normalize_macro_record():
    result = normalize_macro_record(sample_record())
    assert result["country_code"] == "USA"
    assert result["indicator"] == "gdp_growth"
    assert result["year"] == 2020
    assert result["frequency"] == "annual"


def test_normalize_macro_records():
    result = normalize_macro_records([sample_record()])
    assert len(result) == 1


def test_normalizer_rejects_missing_field():
    record = sample_record()
    del record["year"]
    with pytest.raises(ValueError):
        normalize_macro_record(record)


def test_normalizer_rejects_infinite_value():
    record = sample_record()
    record["value"] = float("inf")
    with pytest.raises(ValueError):
        normalize_macro_record(record)


def test_validator_accepts_valid_record():
    record = normalize_macro_record(sample_record())
    is_valid, reason = validate_macro_record(record)
    assert is_valid is True
    assert reason == "Valid"


def test_validator_rejects_unsupported_country():
    record = normalize_macro_record(sample_record())
    record["country_code"] = "XXX"
    is_valid, reason = validate_macro_record(record)
    assert is_valid is False
    assert reason == "Unsupported country"


def test_validator_rejects_wrong_indicator_code():
    record = normalize_macro_record(sample_record())
    record["indicator_code"] = "WRONG"
    is_valid, reason = validate_macro_record(record)
    assert is_valid is False
    assert reason == "Indicator code mismatch"


def test_validator_rejects_boolean_value():
    record = normalize_macro_record(sample_record())
    record["value"] = True
    is_valid, reason = validate_macro_record(record)
    assert is_valid is False
    assert reason == "Value cannot be boolean"


def test_validate_macro_records():
    valid = normalize_macro_record(sample_record())
    invalid = dict(valid)
    invalid["country_code"] = "XXX"

    valid_records, invalid_records = validate_macro_records([valid, invalid])
    assert len(valid_records) == 1
    assert len(invalid_records) == 1


def test_save_and_load_macro_records(tmp_path, monkeypatch):
    import src.data_engineering.macroeconomic.storage as storage

    monkeypatch.setattr(storage, "DATA_DIR", tmp_path)
    records = [normalize_macro_record(sample_record())]

    file_path = save_macro_records(records, "USA", "gdp_growth")
    loaded_records = load_macro_records(file_path)

    assert len(loaded_records) == 1
    assert loaded_records[0] == records[0]


def test_load_missing_file_returns_empty_list(tmp_path):
    missing_file = tmp_path / "missing.jsonl"
    assert load_macro_records(missing_file) == []
