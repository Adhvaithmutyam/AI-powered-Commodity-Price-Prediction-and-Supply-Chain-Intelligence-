import json
from pathlib import Path

from src.data_engineering.config import DATA_DIR


def save_macro_records(records, country_code, indicator_name, validated=False):
    folder = "validated" if validated else "raw"
    data_dir = DATA_DIR / folder / "macroeconomic"
    data_dir.mkdir(parents=True, exist_ok=True)

    file_path = data_dir / f"{country_code}_{indicator_name}.jsonl"

    with file_path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + chr(10))

    return file_path


def load_macro_records(file_path):
    path = Path(file_path)
    if not path.exists():
        return []

    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]
