import json

from src.data_engineering.eia.quality_report import EIAQualityReport


def test_eia_quality_report(tmp_path, monkeypatch):
    from src.data_engineering.eia import quality_report

    data_dir = tmp_path
    validated_dir = data_dir / "validated" / "eia"
    validated_dir.mkdir(parents=True)

    records = [
        {"observation_date": "2026-01-02T00:00:00+00:00", "value": 100},
        {"observation_date": "2026-01-09T00:00:00+00:00", "value": 110},
        {"observation_date": "2026-01-23T00:00:00+00:00", "value": 120},
    ]

    file_path = validated_dir / "test_inventory.jsonl"

    with file_path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record) + "\n")

    monkeypatch.setattr(quality_report, "DATA_DIR", data_dir)

    report = EIAQualityReport.generate("test_inventory")

    assert report["total_records"] == 3
    assert report["missing_values"] == 0
    assert report["duplicate_records"] == 0
    assert report["missing_weeks"] == 1
    assert report["missing_week_dates"] == ["2026-01-16"]


def test_eia_production_quality_report(tmp_path, monkeypatch):
    from src.data_engineering.eia import quality_report

    data_dir = tmp_path
    validated_dir = data_dir / "validated" / "eia"
    validated_dir.mkdir(parents=True)

    records = [
        {"observation_date": "2026-01-02T00:00:00+00:00", "value": 14000},
        {"observation_date": "2026-01-09T00:00:00+00:00", "value": 14100},
        {"observation_date": "2026-01-23T00:00:00+00:00", "value": 14200},
    ]

    file_path = validated_dir / "crude_oil_production.jsonl"

    with file_path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record) + "\n")

    monkeypatch.setattr(quality_report, "DATA_DIR", data_dir)

    report = EIAQualityReport.generate_production()

    assert report["production_type"] == "crude_oil_production"
    assert report["total_records"] == 3
    assert report["missing_values"] == 0
    assert report["duplicate_records"] == 0
    assert report["missing_weeks"] == 1
    assert report["missing_week_dates"] == ["2026-01-16"]

def test_eia_production_quality_report(tmp_path, monkeypatch):
    from src.data_engineering.eia import quality_report

    data_dir = tmp_path
    validated_dir = data_dir / "validated" / "eia"
    validated_dir.mkdir(parents=True)

    records = [
        {"observation_date": "2026-01-02T00:00:00+00:00", "value": 14000},
        {"observation_date": "2026-01-09T00:00:00+00:00", "value": 14100},
        {"observation_date": "2026-01-23T00:00:00+00:00", "value": 14200},
    ]

    file_path = validated_dir / "crude_oil_production.jsonl"

    with file_path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record) + "\n")

    monkeypatch.setattr(quality_report, "DATA_DIR", data_dir)

    report = EIAQualityReport.generate_production()

    assert report["production_type"] == "crude_oil_production"
    assert report["total_records"] == 3
    assert report["missing_values"] == 0
    assert report["duplicate_records"] == 0
    assert report["missing_weeks"] == 1
    assert report["missing_week_dates"] == ["2026-01-16"]
