import os
import time
import requests

from src.data_engineering.comtrade import config


class ComtradeClient:
    def __init__(self):
        self.api_key = os.getenv("COMTRADE_API_KEY")

        if not self.api_key:
            raise ValueError(
                "COMTRADE_API_KEY environment variable is not set."
            )

        self.url = (
            f"{config.BASE_URL}/C/"
            f"{config.FREQUENCY}/{config.CLASSIFICATION}"
        )

        self.session = requests.Session()
        self.session.headers.update({
            "Ocp-Apim-Subscription-Key": self.api_key
        })

        self.last_request_time = 0

    def fetch_month(self, commodity, flow, period):
        if commodity not in config.COMMODITIES:
            raise ValueError(f"Unknown commodity: {commodity}")

        if flow not in config.TRADE_FLOWS:
            raise ValueError(f"Unknown trade flow: {flow}")

        if (
            not isinstance(period, str)
            or len(period) != 6
            or not period.isdigit()
            or not 1 <= int(period[4:]) <= 12
        ):
            raise ValueError("Period must be in YYYYMM format.")

        params = {
            "reporterCode": config.REPORTER_CODE,
            "period": period,
            "cmdCode": config.COMMODITIES[commodity]["hs_code"],
            "flowCode": config.TRADE_FLOWS[flow],
            "partnerCode": config.PARTNER_CODE,
            "maxRecords": config.MAX_RECORDS,
        }

        for attempt in range(config.MAX_RETRIES):
            elapsed = time.monotonic() - self.last_request_time
            wait_time = config.REQUEST_DELAY - elapsed

            if wait_time > 0:
                time.sleep(wait_time)

            try:
                response = self.session.get(
                    self.url,
                    params=params,
                    timeout=config.REQUEST_TIMEOUT,
                )

                self.last_request_time = time.monotonic()

                if response.status_code == 429:
                    if attempt == config.MAX_RETRIES - 1:
                        response.raise_for_status()

                    time.sleep(config.RETRY_BACKOFF * (attempt + 1))
                    continue

                response.raise_for_status()
                result = response.json()

                if result.get("data") is None:
                    raise ValueError(
                        f"Unexpected API response: {result}"
                    )

                return result["data"]

            except requests.RequestException:
                if attempt == config.MAX_RETRIES - 1:
                    raise

                time.sleep(config.RETRY_BACKOFF * (attempt + 1))

        return []
