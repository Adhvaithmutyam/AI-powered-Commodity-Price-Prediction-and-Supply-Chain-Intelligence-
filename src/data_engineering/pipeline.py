
from src.data_engineering.normalizer import normalize_price_record
from src.data_engineering.cleaner import clean_price_records
from src.data_engineering.feature_engineering import generate_price_features
from src.data_engineering.storage import save_jsonl
from src.data_engineering.config import (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    VALIDATED_DATA_DIR
)


def run_pipeline(raw_records):

    # Preserve the original input
    save_jsonl(
        RAW_DATA_DIR / "price_data.jsonl",
        raw_records
    )

    normalized_records = []
    normalization_errors = []

    # Normalize incoming records
    for record in raw_records:
        try:
            normalized_records.append(
                normalize_price_record(record)
            )
        except (ValueError, TypeError, AttributeError) as error:
            normalization_errors.append({
                "record": record,
                "error": str(error)
            })

    # Validate and remove duplicates
    valid_records, invalid_records = clean_price_records(
        normalized_records
    )

    # Save validated data
    save_jsonl(
        VALIDATED_DATA_DIR / "price_data.jsonl",
        valid_records
    )

    # Save rejected records
    save_jsonl(
        RAW_DATA_DIR / "invalid_price_data.jsonl",
        invalid_records + normalization_errors
    )

    # Generate features
    feature_records = generate_price_features(valid_records)

    # Save processed data
    save_jsonl(
        PROCESSED_DATA_DIR / "price_features.jsonl",
        feature_records
    )

    return {
        "input_records": len(raw_records),
        "normalized_records": len(normalized_records),
        "valid_records": len(valid_records),
        "invalid_records": len(invalid_records) + len(normalization_errors),
        "feature_records": len(feature_records)
    }
