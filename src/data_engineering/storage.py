
import json
import math
from pathlib import Path


def json_serializer(value):

    if hasattr(value, "isoformat"):
        return value.isoformat()

    if hasattr(value, "item"):
        return value.item()

    raise TypeError(
        f"Object of type {type(value).__name__} is not JSON serializable"
    )


def save_jsonl(filepath, records):

    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with filepath.open("w", encoding="utf-8") as file:

        for record in records:

            file.write(
                json.dumps(
                    record,
                    default=json_serializer,
                    allow_nan=False
                ) + "\n"
            )


def load_jsonl(filepath):

    filepath = Path(filepath)

    if not filepath.exists():
        return []

    records = []

    with filepath.open("r", encoding="utf-8") as file:

        for line in file:

            if not line.strip():
                continue

            try:
                records.append(json.loads(line))

            except json.JSONDecodeError:
                continue

    return records
