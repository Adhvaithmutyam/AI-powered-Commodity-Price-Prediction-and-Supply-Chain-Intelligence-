import json
from pathlib import Path

from ..config import DATA_DIR


COMTRADE_RAW_DIR = DATA_DIR / "raw" / "comtrade"
COMTRADE_VALIDATED_DIR = DATA_DIR / "validated" / "comtrade"


class ComtradeStorage:

    @staticmethod
    def save_jsonl(records, file_path):
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        with file_path.open("w", encoding="utf-8") as file:
            for record in records:
                file.write(
                    json.dumps(record, ensure_ascii=False) + "\n"
                )

        return file_path

    @staticmethod
    def load_jsonl(file_path):
        file_path = Path(file_path)

        if not file_path.exists():
            return []

        records = []

        with file_path.open("r", encoding="utf-8") as file:
            for line in file:
                if line.strip():
                    records.append(json.loads(line))

        return records

    @classmethod
    def save_raw(cls, records, commodity, flow, period):
        return cls.save_jsonl(
            records,
            COMTRADE_RAW_DIR
            / f"{commodity}_{flow}_{period}.jsonl",
        )

    @classmethod
    def save_validated(cls, records, commodity, flow, period):
        return cls.save_jsonl(
            records,
            COMTRADE_VALIDATED_DIR
            / f"{commodity}_{flow}_{period}.jsonl",
        )

    @classmethod
    def save_invalid(cls, records, commodity, flow, period):
        return cls.save_jsonl(
            records,
            COMTRADE_VALIDATED_DIR
            / f"{commodity}_{flow}_{period}_invalid.jsonl",
        )

    @classmethod
    def save_normalization_errors(
        cls, records, commodity, flow, period
    ):
        return cls.save_jsonl(
            records,
            COMTRADE_VALIDATED_DIR
            / f"{commodity}_{flow}_{period}_normalization_errors.jsonl",
        )
