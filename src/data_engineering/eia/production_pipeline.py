from .client import EIAClient
from .normalizer import EIANormalizer
from .validator import EIAValidator
from .storage import EIAStorage


class EIAProductionPipeline:

    def __init__(self):
        self.client = EIAClient()
        self.normalizer = EIANormalizer()
        self.validator = EIAValidator()

    def run(self, production_type="crude_oil_production"):
        raw_records = self.client.fetch_production(production_type)

        normalized_records = []
        normalization_errors = []

        for record in raw_records:
            try:
                normalized_records.append(
                    self.normalizer.normalize_production(record, production_type)
                )
            except (ValueError, TypeError) as error:
                normalization_errors.append({
                    "record": record,
                    "error": str(error),
                })

        valid_records = []
        invalid_records = []

        for record in normalized_records:
            is_valid, errors = self.validator.validate_production(record)

            if is_valid:
                valid_records.append(record)
            else:
                invalid_records.append({
                    "record": record,
                    "errors": errors,
                })

        EIAStorage.save_production_raw(raw_records, production_type)
        EIAStorage.save_production_validated(valid_records, production_type)
        EIAStorage.save_production_invalid(invalid_records, production_type)

        return {
            "production_type": production_type,
            "raw_records": len(raw_records),
            "normalized_records": len(normalized_records),
            "valid_records": len(valid_records),
            "invalid_records": len(invalid_records),
            "normalization_errors": len(normalization_errors),
            "raw_file": f"data/raw/eia/{production_type}.jsonl",
            "validated_file": f"data/validated/eia/{production_type}.jsonl",
        }


if __name__ == "__main__":
    result = EIAProductionPipeline().run()
    print("\nEIA PRODUCTION PIPELINE RESULTS")
    print("-" * 40)
    for key, value in result.items():
        print(f"{key}: {value}")
