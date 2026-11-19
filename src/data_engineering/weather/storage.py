import json
from pathlib import Path
from . import config

class WeatherStorage:
    @staticmethod
    def save(records, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as file:
            for record in records:
                file.write(json.dumps(record, allow_nan=False) + "\n")

    @staticmethod
    def load(path):
        path = Path(path)
        if not path.exists():
            return []
        with path.open("r", encoding="utf-8") as file:
            return [json.loads(line) for line in file if line.strip()]

    @classmethod
    def save_raw(cls, records, location):
        cls.save(records, config.RAW_DIR / f"{location}.jsonl")

    @classmethod
    def save_validated(cls, records, location):
        cls.save(records, config.VALIDATED_DIR / f"{location}.jsonl")

    @classmethod
    def save_invalid(cls, records, location):
        cls.save(records, config.VALIDATED_DIR / f"{location}_invalid.jsonl")
