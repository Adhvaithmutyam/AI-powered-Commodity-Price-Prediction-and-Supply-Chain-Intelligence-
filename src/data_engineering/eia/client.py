import time
import requests

from .config import (
    BASE_URL,
    API_KEY,
    REQUEST_TIMEOUT,
    MAX_RETRIES,
    PAGE_SIZE,
    INVENTORY_SERIES,
    PRODUCTION_SERIES,
    TRADE_SERIES,
)


class EIAClient:
    def __init__(self):
        if not API_KEY:
            raise ValueError("EIA_API_KEY environment variable is not configured.")

        self.session = requests.Session()

    def _fetch_series(self, endpoint, series_id):
        records = []
        offset = 0

        while True:
            params = {
                "api_key": API_KEY,
                "frequency": "weekly",
                "data[0]": "value",
                "facets[series][]": series_id,
                "sort[0][column]": "period",
                "sort[0][direction]": "asc",
                "length": PAGE_SIZE,
                "offset": offset,
            }

            response = None

            for attempt in range(MAX_RETRIES):
                try:
                    response = self.session.get(
                        endpoint,
                        params=params,
                        timeout=REQUEST_TIMEOUT,
                    )
                    response.raise_for_status()
                    break
                except requests.RequestException:
                    if attempt == MAX_RETRIES - 1:
                        raise
                    time.sleep(2 ** attempt)

            payload = response.json()
            page_records = payload.get("response", {}).get("data", [])

            if not page_records:
                break

            records.extend(page_records)

            if len(page_records) < PAGE_SIZE:
                break

            offset += len(page_records)

        return records

    def fetch_inventory(self, inventory_type="commercial_crude_oil"):
        if inventory_type not in INVENTORY_SERIES:
            raise ValueError(f"Unsupported inventory type: {inventory_type}")

        series = INVENTORY_SERIES[inventory_type]
        endpoint = f"{BASE_URL}/petroleum/stoc/wstk/data/"

        return self._fetch_series(endpoint, series["series_id"])

    def fetch_production(self, production_type="crude_oil_production"):
        if production_type not in PRODUCTION_SERIES:
            raise ValueError(f"Unsupported production type: {production_type}")

        series = PRODUCTION_SERIES[production_type]
        endpoint = f"{BASE_URL}/{series['route']}/data/"

        return self._fetch_series(endpoint, series["series_id"])

    def fetch_trade(self, trade_type="crude_oil_imports"):
        if trade_type not in TRADE_SERIES:
            raise ValueError(f"Unsupported trade type: {trade_type}")

        series = TRADE_SERIES[trade_type]
        endpoint = f"{BASE_URL}/{series['route']}/data/"

        return self._fetch_series(endpoint, series["series_id"])
