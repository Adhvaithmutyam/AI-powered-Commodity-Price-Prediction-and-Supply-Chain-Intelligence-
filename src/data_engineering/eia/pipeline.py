from .client import EIAClient
from .normalizer import EIANormalizer
from .validator import EIAValidator
from .storage import EIAStorage


class EIAPipeline:

    def __init__(self):
        self.client = EIAClient()
        self.normalizer = EIANormalizer()
        self.validator = EIAValidator()

    def run(self, inventory_type="commercial_crude_oil"):
        raw_records = self.client.fetch_inventory(inventory_type)

        normalized_records = []
        normalization_errors = []

        for record in raw_records:
            try:
                normalized_records.append(
                    self.normalizer.normalize(record, inventory_type)
                )
            except (ValueError, TypeError) as error:
                normalization_errors.append({
                    "record": record,
                    "error": str(error),
                })

        valid_records = []
        invalid_records = []

        for record in normalized_records:
            is_valid, errors = self.validator.validate(record)

            if is_valid:
                valid_records.append(record)
            else:
                invalid_records.append({
                    "record": record,
                    "errors": errors,
                })

        self.storage_paths(
            raw_records,
            valid_records,
            invalid_records,
            normalization_errors,
            inventory_type,
        )

        return {
            "inventory_type": inventory_type,
            "raw_records": len(raw_records),
            "normalized_records": len(normalized_records),
            "valid_records": len(valid_records),
            "invalid_records": len(invalid_records),
            "normalization_errors": len(normalization_errors),
            "raw_file": str(self.storage_path("raw", inventory_type)),
            "validated_file": str(self.storage_path("validated", inventory_type)),
            "normalization_error_file": str(
                self.storage_path("normalization_errors", inventory_type)
            ),
        }

    @staticmethod
    def storage_path(category, inventory_type):
        from ..config import DATA_DIR

        if category == "raw":
            return DATA_DIR / "raw" / "eia" / f"{inventory_type}.jsonl"

        if category == "normalization_errors":
            return (
                DATA_DIR / "validated" / "eia"
                / f"{inventory_type}_normalization_errors.jsonl"
            )

        return DATA_DIR / "validated" / "eia" / f"{inventory_type}.jsonl"

    @staticmethod
    def storage_paths(raw, valid, invalid, normalization_errors, inventory_type):
        EIAStorage.save_raw(raw, inventory_type)
        EIAStorage.save_validated(valid, inventory_type)
        EIAStorage.save_invalid(invalid, inventory_type)
        EIAStorage.save_normalization_errors(normalization_errors, inventory_type)

    def run_production(self, production_type="crude_oil_production"):
        raw_records = self.client.fetch_production(production_type)

        normalized_records = []
        normalization_errors = []

        for record in raw_records:
            try:
                normalized_records.append(
                    self.normalizer.normalize_production(
                        record, production_type
                    )
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
        EIAStorage.save_production_normalization_errors(
            normalization_errors, production_type
        )

        return {
            "production_type": production_type,
            "raw_records": len(raw_records),
            "normalized_records": len(normalized_records),
            "valid_records": len(valid_records),
            "invalid_records": len(invalid_records),
            "normalization_errors": len(normalization_errors),
        }

    def run_trade(self, trade_type="crude_oil_imports"):
        raw_records = self.client.fetch_trade(trade_type)

        normalized_records = []
        normalization_errors = []

        for record in raw_records:
            try:
                normalized_records.append(
                    self.normalizer.normalize_trade(record, trade_type)
                )
            except (ValueError, TypeError) as error:
                normalization_errors.append({
                    "record": record,
                    "error": str(error),
                })

        valid_records = []
        invalid_records = []

        for record in normalized_records:
            is_valid, errors = self.validator.validate_trade(
                record, trade_type
            )

            if is_valid:
                valid_records.append(record)
            else:
                invalid_records.append({
                    "record": record,
                    "errors": errors,
                })

        EIAStorage.save_trade_raw(raw_records, trade_type)
        EIAStorage.save_trade_validated(valid_records, trade_type)
        EIAStorage.save_trade_invalid(invalid_records, trade_type)
        EIAStorage.save_trade_normalization_errors(
            normalization_errors, trade_type
        )

        return {
            "trade_type": trade_type,
            "raw_records": len(raw_records),
            "normalized_records": len(normalized_records),
            "valid_records": len(valid_records),
            "invalid_records": len(invalid_records),
            "normalization_errors": len(normalization_errors),
        }
