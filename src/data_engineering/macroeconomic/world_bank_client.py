import logging
import requests

from src.data_engineering.macroeconomic.config import (
    BASE_URL,
    COUNTRY_CODES,
    INDICATORS,
    START_YEAR,
    END_YEAR,
    REQUEST_TIMEOUT
)

logger = logging.getLogger(__name__)


class WorldBankClient:

    def __init__(self):
        self.session = requests.Session()

    def fetch_indicator(self, country_code, indicator_name):

        if country_code not in COUNTRY_CODES:
            raise ValueError(f"Unsupported country: {country_code}")

        if indicator_name not in INDICATORS:
            raise ValueError(f"Unsupported indicator: {indicator_name}")

        indicator_code = INDICATORS[indicator_name]["code"]

        url = (
            f"{BASE_URL}/country/{country_code}"
            f"/indicator/{indicator_code}"
        )

        params = {
            "format": "json",
            "date": f"{START_YEAR}:{END_YEAR}",
            "per_page": 100
        }

        records = []
        page = 1

        while True:
            params["page"] = page
            response = None

            for attempt in range(3):
                try:
                    response = self.session.get(
                        url,
                        params=params,
                        timeout=REQUEST_TIMEOUT
                    )
                    response.raise_for_status()
                    break

                except requests.exceptions.Timeout:
                    logger.warning(
                        "Request timed out. Attempt %d of 3",
                        attempt + 1
                    )

                    if attempt == 2:
                        raise

            if response is None:
                raise RuntimeError("World Bank API request failed")

            result = response.json()

            if not isinstance(result, list) or len(result) < 2:
                raise ValueError("Unexpected World Bank API response")

            metadata, data = result

            if data:
                for item in data:
                    records.append({
                        "country_code": country_code,
                        "indicator": indicator_name,
                        "indicator_code": indicator_code,
                        "year": int(item["date"]),
                        "value": item["value"],
                        "unit": INDICATORS[indicator_name]["description"],
                        "source": "World Bank"
                    })

            total_pages = metadata.get("pages", 1)

            if page >= total_pages:
                break

            page += 1

        logger.info(
            "Downloaded %d records for %s - %s",
            len(records),
            country_code,
            indicator_name
        )

        return records
