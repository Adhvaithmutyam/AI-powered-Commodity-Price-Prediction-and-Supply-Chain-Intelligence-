from datetime import datetime, timezone
import math


def normalize_macro_record(record):

    required_fields = [
        "country_code",
        "indicator",
        "indicator_code",
        "year",
        "value",
        "unit",
        "source"
    ]

    for field in required_fields:
        if field not in record:
            raise ValueError(f"Missing field: {field}")

    year = int(record["year"])

    if year < 1900 or year > datetime.now(timezone.utc).year:
        raise ValueError("Invalid observation year")

    value = record["value"]

    if value is not None:
        value = float(value)

        if not math.isfinite(value):
            raise ValueError("Invalid indicator value")

    return {
        "country_code": record["country_code"].upper(),
        "indicator": record["indicator"].lower(),
        "indicator_code": record["indicator_code"],
        "year": year,
        "value": value,
        "unit": record["unit"],
        "source": record["source"],
        "frequency": "annual"
    }


def normalize_macro_records(records):

    normalized = []

    for record in records:
        normalized.append(normalize_macro_record(record))

    return normalized