
from src.data_engineering.pipeline import run_pipeline


def create_sample_record():
    return {
        "commodity": "copper",
        "timestamp": "2026-11-17T00:00:00Z",
        "open": 6.0,
        "high": 6.2,
        "low": 5.8,
        "close": 6.1,
        "volume": 1000,
        "currency": "USD",
        "unit": "USD/lb",
        "source": "Yahoo Finance",
        "ticker": "HG=F",
        "benchmark": "COMEX Copper Futures"
    }


def test_pipeline_success(tmp_path, monkeypatch):
    from src.data_engineering import pipeline

    monkeypatch.setattr(
        pipeline, "RAW_DATA_DIR", tmp_path / "raw"
    )
    monkeypatch.setattr(
        pipeline, "PROCESSED_DATA_DIR", tmp_path / "processed"
    )
    monkeypatch.setattr(
        pipeline, "VALIDATED_DATA_DIR", tmp_path / "validated"
    )

    records = [create_sample_record()]

    result = run_pipeline(records)

    assert result["input_records"] == 1
    assert result["normalized_records"] == 1
    assert result["valid_records"] == 1
    assert result["invalid_records"] == 0
    assert result["feature_records"] == 1

    assert (tmp_path / "raw" / "price_data.jsonl").exists()
    assert (tmp_path / "validated" / "price_data.jsonl").exists()
    assert (tmp_path / "processed" / "price_features.jsonl").exists()


def test_pipeline_invalid_record(tmp_path, monkeypatch):
    from src.data_engineering import pipeline

    monkeypatch.setattr(
        pipeline, "RAW_DATA_DIR", tmp_path / "raw"
    )
    monkeypatch.setattr(
        pipeline, "PROCESSED_DATA_DIR", tmp_path / "processed"
    )
    monkeypatch.setattr(
        pipeline, "VALIDATED_DATA_DIR", tmp_path / "validated"
    )

    record = create_sample_record()
    record["high"] = 5.0

    result = run_pipeline([record])

    assert result["input_records"] == 1
    assert result["valid_records"] == 0
    assert result["invalid_records"] == 1
    assert result["feature_records"] == 0
