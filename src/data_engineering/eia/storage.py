import json
from pathlib import Path
from ..config import DATA_DIR

EIA_RAW_DIR = DATA_DIR / "raw" / "eia"
EIA_VALIDATED_DIR = DATA_DIR / "validated" / "eia"


class EIAStorage:

    @staticmethod
    def save_jsonl(records, file_path):
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with file_path.open("w", encoding="utf-8") as file:
            for record in records:
                file.write(json.dumps(record, ensure_ascii=False) + "\n")
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
    def save_raw(cls, records, inventory_type):
        return cls.save_jsonl(records, EIA_RAW_DIR / f"{inventory_type}.jsonl")

    @classmethod
    def save_validated(cls, records, inventory_type):
        return cls.save_jsonl(records, EIA_VALIDATED_DIR / f"{inventory_type}.jsonl")

    @classmethod
    def save_invalid(cls, records, inventory_type):
        return cls.save_jsonl(records, EIA_VALIDATED_DIR / f"{inventory_type}_invalid.jsonl")

    @classmethod
    def save_normalization_errors(cls, records, inventory_type):
        return cls.save_jsonl(records, EIA_VALIDATED_DIR / f"{inventory_type}_normalization_errors.jsonl")

    @classmethod
    def save_production_raw(cls, records, production_type):
        return cls.save_jsonl(records, EIA_RAW_DIR / f"{production_type}.jsonl")

    @classmethod
    def save_production_validated(cls, records, production_type):
        return cls.save_jsonl(records, EIA_VALIDATED_DIR / f"{production_type}.jsonl")

    @classmethod
    def save_production_invalid(cls, records, production_type):
        return cls.save_jsonl(records, EIA_VALIDATED_DIR / f"{production_type}_invalid.jsonl")

    @classmethod
    def save_production_normalization_errors(cls, records, production_type):
        return cls.save_jsonl(
            records,
            EIA_VALIDATED_DIR
            / f"{production_type}_normalization_errors.jsonl",
        )

    @classmethod
    def save_trade_raw(cls, records, trade_type):
        return cls.save_jsonl(
            records, EIA_RAW_DIR / f"{trade_type}.jsonl"
        )

    @classmethod
    def save_trade_validated(cls, records, trade_type):
        return cls.save_jsonl(
            records, EIA_VALIDATED_DIR / f"{trade_type}.jsonl"
        )

    @classmethod
    def save_trade_invalid(cls, records, trade_type):
        return cls.save_jsonl(
            records, EIA_VALIDATED_DIR / f"{trade_type}_invalid.jsonl"
        )

    @classmethod
    def save_trade_normalization_errors(cls, records, trade_type):
        return cls.save_jsonl(
            records,
            EIA_VALIDATED_DIR / f"{trade_type}_normalization_errors.jsonl",
        )
