
from src.data_engineering.validator import validate_price_record


def clean_price_records(records):

    valid_records = []
    invalid_records = []
    seen = set()

    for record in records:

        is_valid, errors = validate_price_record(record)

        if not is_valid:
            invalid_records.append({
                "record": record,
                "errors": errors
            })
            continue

        # Identify duplicate market observations
        key = (
            record["commodity"],
            record["benchmark"],
            record["ticker"],
            record["timestamp"],
            record["source"]
        )

        if key in seen:
            continue

        seen.add(key)
        valid_records.append(record)

    return valid_records, invalid_records
