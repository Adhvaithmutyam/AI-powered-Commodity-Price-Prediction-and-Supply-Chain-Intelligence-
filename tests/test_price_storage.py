from src.data_engineering.storage import save_jsonl, load_jsonl


def test_save_and_load_jsonl(tmp_path):
    records = [
        {"commodity": "copper", "close": 6.1},
        {"commodity": "gold", "close": 3850.0}
    ]

    filepath = tmp_path / "prices.jsonl"

    save_jsonl(filepath, records)
    loaded = load_jsonl(filepath)

    assert loaded == records


def test_load_missing_file(tmp_path):
    filepath = tmp_path / "missing.jsonl"

    loaded = load_jsonl(filepath)

    assert loaded == []


def test_save_empty_records(tmp_path):
    filepath = tmp_path / "empty.jsonl"

    save_jsonl(filepath, [])

    assert load_jsonl(filepath) == []