import json
import logging

from src.data_engineering.macroeconomic.world_bank_client import WorldBankClient
from src.data_engineering.macroeconomic.normalizer import normalize_macro_records
from src.data_engineering.macroeconomic.validator import validate_macro_records
from src.data_engineering.macroeconomic.storage import save_macro_records
from src.data_engineering.macroeconomic.config import COUNTRY_CODES, INDICATORS

logger = logging.getLogger(__name__)


class MacroeconomicPipeline:

    def __init__(self):
        self.client = WorldBankClient()

    def run(self, country_code, indicator_name):

        raw_records = self.client.fetch_indicator(
            country_code,
            indicator_name
        )

        normalized_records = normalize_macro_records(raw_records)

        valid_records, invalid_records = validate_macro_records(
            normalized_records
        )

        valid_path = save_macro_records(
            valid_records,
            country_code,
            indicator_name,
            validated=True
        )

        invalid_path = None

        if invalid_records:
            invalid_path = save_macro_records(
                invalid_records,
                country_code,
                f"{indicator_name}_invalid"
            )

        summary = {
            "country": country_code,
            "indicator": indicator_name,
            "raw_records": len(raw_records),
            "normalized_records": len(normalized_records),
            "valid_records": len(valid_records),
            "invalid_records": len(invalid_records),
            "valid_file": str(valid_path),
            "invalid_file": str(invalid_path) if invalid_path else None
        }

        logger.info("Macroeconomic pipeline summary: %s", summary)

        return summary


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)

    pipeline = MacroeconomicPipeline()

    for country in COUNTRY_CODES:
        for indicator in INDICATORS:
            try:
                result = pipeline.run(country, indicator)
                print(result)

            except Exception as error:
                logger.error(
                    "Failed for %s - %s: %s",
                    country,
                    indicator,
                    error
                )
