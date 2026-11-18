import json
from pathlib import Path


def generate_quality_report(file_path):
    path = Path(file_path)

    report = {
        "file": path.name,
        "total_records": 0,
        "missing_values": 0,
        "duplicate_records": 0,
        "invalid_year_records": 0,
        "missing_years": [],
        "valid_records": 0
    }

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    records = []

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                records.append(json.loads(line))

    report["total_records"] = len(records)

    seen = set()
    years = []

    for record in records:
        key = (
            record.get("country_code"),
            record.get("indicator"),
            record.get("year")
        )

        if key in seen:
            report["duplicate_records"] += 1
        else:
            seen.add(key)

        if record.get("value") is None:
            report["missing_values"] += 1

        year = record.get("year")

        if not isinstance(year, int) or isinstance(year, bool) or year < 1900:
            report["invalid_year_records"] += 1
        else:
            years.append(year)

    if years:
        expected_years = set(range(min(years), max(years) + 1))
        actual_years = set(years)
        report["missing_years"] = sorted(expected_years - actual_years)

    report["valid_records"] = (
        report["total_records"]
        - report["duplicate_records"]
        - report["invalid_year_records"]
        - report["missing_values"]
    )

    return report


def main():
    data_dir = Path("data/raw/macroeconomic")

    for file_path in sorted(data_dir.glob("*.jsonl")):
        report = generate_quality_report(file_path)
        print("\nMACROECONOMIC DATA QUALITY REPORT")
        for key, value in report.items():
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
