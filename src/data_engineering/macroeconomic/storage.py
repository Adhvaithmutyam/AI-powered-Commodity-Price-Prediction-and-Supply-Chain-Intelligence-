import json
from pathlib import Path

from src.data_engineering.config import DATA_DIR

MACRO_DATA_DIR = DATA_DIR / "raw" / "macroeconomic"


def save_macro_records(records, country_code, indicator_name):

    MACRO_DATA_DIR.mkdir(parents=True, exist_ok=True)

    file_path = MACRO_DATA_DIR / f"{country_code}_{indicator_name}.jsonl"

    with file_path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

    return file_path


def load_macro_records(file_path):

    path = Path(file_path)
    records = []

    if not path.exists():
        return records

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                records.append(json.loads(line))

    return records
