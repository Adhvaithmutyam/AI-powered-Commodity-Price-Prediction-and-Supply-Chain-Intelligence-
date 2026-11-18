import math

from src.data_engineering.macroeconomic.config import (
    COUNTRY_CODES,
    INDICATORS
)


def validate_macro_record(record):

    required_fields = [
        "country_code",
        "indicator",
        "indicator_code",
        "year",
        "value",
        "unit",
        "source",
        "frequency"
    ]

    for field in required_fields:
        if field not in record:
            return False, f"Missing field: {field}"

    if record["country_code"] not in COUNTRY_CODES:
        return False, "Unsupported country"

    if record["indicator"] not in INDICATORS:
        return False, "Unsupported indicator"

    expected_code = INDICATORS[record["indicator"]]["code"]

    if record["indicator_code"] != expected_code:
        return False, "Indicator code mismatch"

    if record["frequency"] != "annual":
        return False, "Invalid frequency"

    if not isinstance(record["year"], int) or isinstance(record["year"], bool):
        return False, "Year must be an integer"

    if record["year"] < 1900:
        return False, "Invalid year"

    value = record["value"]

    if value is not None:
        if isinstance(value, bool):
            return False, "Value cannot be boolean"

        if not isinstance(value, (int, float)):
            return False, "Value must be numeric or None"

        if not math.isfinite(value):
            return False, "Value must be finite"

    if not record["source"] or not record["unit"]:
        return False, "Source and unit are required"

    return True, "Valid"


def validate_macro_records(records):

    valid_records = []
    invalid_records = []

    for record in records:
        is_valid, reason = validate_macro_record(record)

        if is_valid:
            valid_records.append(record)
        else:
            invalid_records.append({
                "record": record,
                "reason": reason
            })

    return valid_records, invalid_records
