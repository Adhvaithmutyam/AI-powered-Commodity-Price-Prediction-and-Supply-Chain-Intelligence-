from .client import ComtradeClient
from .normalizer import ComtradeNormalizer
from .validator import ComtradeValidator
from .storage import ComtradeStorage


class ComtradePipeline:

    def __init__(self):
        self.client = ComtradeClient()
        self.validator = ComtradeValidator()

    def run_month(self, commodity, flow, period):
        raw_records = self.client.fetch_month(
            commodity,
            flow,
            period,
        )

        normalization_errors = []
        valid_records = []
        invalid_records = []

        for record in raw_records:
            try:
                normalized = ComtradeNormalizer.normalize(
                    record,
                    commodity,
                    flow,
                )

                is_valid, errors = self.validator.validate(normalized)

                if is_valid:
                    valid_records.append(normalized)
                else:
                    invalid_records.append({
                        "record": normalized,
                        "errors": errors,
                    })

            except (ValueError, TypeError) as error:
                normalization_errors.append({
                    "record": record,
                    "error": str(error),
                })

        self._save_results(
            raw_records,
            valid_records,
            invalid_records,
            normalization_errors,
            commodity,
            flow,
            period,
        )

        return {
            "commodity": commodity,
            "flow": flow,
            "period": period,
            "raw_count": len(raw_records),
            "valid_count": len(valid_records),
            "invalid_count": len(invalid_records),
            "normalization_error_count": len(normalization_errors),
        }

    @staticmethod
    def _save_results(
        raw_records,
        valid_records,
        invalid_records,
        normalization_errors,
        commodity,
        flow,
        period,
    ):
        ComtradeStorage.save_raw(
            raw_records, commodity, flow, period
        )
        ComtradeStorage.save_validated(
            valid_records, commodity, flow, period
        )
        ComtradeStorage.save_invalid(
            invalid_records, commodity, flow, period
        )
        ComtradeStorage.save_normalization_errors(
            normalization_errors, commodity, flow, period
        )
