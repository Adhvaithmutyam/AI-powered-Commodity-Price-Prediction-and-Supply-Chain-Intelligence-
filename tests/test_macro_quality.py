import json
from src.data_engineering.macroeconomic.quality_report import generate_quality_report


def test_quality_report_valid_records(tmp_path):
    file_path = tmp_path / "sample.jsonl"

    records = [
        {"country_code": "USA", "indicator": "gdp_growth", "year": 2022, "value": 2.5},
        {"country_code": "USA", "indicator": "gdp_growth", "year": 2023, "value": 2.8}
    ]

    with file_path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record) + "\n")

    report = generate_quality_report(file_path)

    assert report["total_records"] == 2
    assert report["valid_records"] == 2
    assert report["missing_values"] == 0


def test_quality_report_missing_values(tmp_path):
    file_path = tmp_path / "sample.jsonl"

    records = [
        {"country_code": "USA", "indicator": "gdp_growth", "year": 2022, "value": 2.5},
        {"country_code": "USA", "indicator": "gdp_growth", "year": 2023, "value": None}
    ]

    with file_path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record) + "\n")

    report = generate_quality_report(file_path)

    assert report["missing_values"] == 1
    assert report["valid_records"] == 1


def test_quality_report_duplicates(tmp_path):
    file_path = tmp_path / "sample.jsonl"

    record = {
        "country_code": "USA",
        "indicator": "gdp_growth",
        "year": 2022,
        "value": 2.5
    }

    with file_path.open("w", encoding="utf-8") as file:
        file.write(json.dumps(record) + "\n")
        file.write(json.dumps(record) + "\n")

    report = generate_quality_report(file_path)

    assert report["duplicate_records"] == 1
    assert report["valid_records"] == 1


def test_quality_report_missing_file(tmp_path):
    file_path = tmp_path / "missing.jsonl"

    try:
        generate_quality_report(file_path)
        assert False
    except FileNotFoundError:
        assert True
