from src.data_engineering.unified_features_pipeline import (
    as_date,
    normalize_commodity,
    read_jsonl,
    write_jsonl,
)


def test_normalize_commodity():
    assert normalize_commodity("crude_oil") == "crude oil"
    assert normalize_commodity("Copper") == "copper"


def test_as_date():
    assert str(as_date("2025-01-15T00:00:00Z")) == "2025-01-15"


def test_jsonl_storage(tmp_path):
    path = tmp_path / "test.jsonl"
    records = [{"commodity": "gold", "close": 2000.0}]
    write_jsonl(path, records)
    assert read_jsonl(path) == records


def test_read_missing_file(tmp_path):
    assert read_jsonl(tmp_path / "missing.jsonl") == []
