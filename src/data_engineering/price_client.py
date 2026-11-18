
import logging
from datetime import datetime, timezone

import yfinance as yf

from src.data_engineering.source_registry import COMMODITY_REGISTRY

logger = logging.getLogger(__name__)


class PriceClient:

    def fetch_historical_prices(self, commodity, period="5y"):

        commodity = commodity.strip().lower()

        if commodity not in COMMODITY_REGISTRY:
            raise ValueError(f"Unsupported commodity: {commodity}")

        config = COMMODITY_REGISTRY[commodity]
        ticker = config["ticker"]

        logger.info("Downloading %s data for %s", commodity, ticker)

        data = yf.download(
            ticker,
            period=period,
            interval="1d",
            progress=False,
            auto_adjust=False
        )

        if data.empty:
            raise ValueError(f"No price data returned for {ticker}")

        records = []

        for timestamp, row in data.iterrows():

            def get_value(field):
                value = row[field]

                # Handle yfinance MultiIndex columns
                if hasattr(value, "iloc"):
                    value = value.iloc[0]

                return float(value)

            record = {
                "commodity": commodity,
                "timestamp": timestamp.to_pydatetime().replace(
                    tzinfo=timezone.utc
                ).isoformat().replace("+00:00", "Z"),
                "open": get_value("Open"),
                "high": get_value("High"),
                "low": get_value("Low"),
                "close": get_value("Close"),
                "volume": get_value("Volume"),
                "currency": config["currency"],
                "unit": config["unit"],
                "source": config["source"],
                "ticker": ticker,
                "benchmark": config["benchmark"]
            }

            records.append(record)

        logger.info("Downloaded %d records for %s", len(records), commodity)

        return records
