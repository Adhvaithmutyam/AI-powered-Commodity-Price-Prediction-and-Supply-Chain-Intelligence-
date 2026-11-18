
import logging

from src.data_engineering.price_client import PriceClient
from src.data_engineering.pipeline import run_pipeline
from src.data_engineering.config import COMMODITIES


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


def main():

    client = PriceClient()
    all_records = []

    for commodity in COMMODITIES:

        print(f"\nDownloading {commodity} data...")

        records = client.fetch_historical_prices(
            commodity,
            period="5y"
        )

        print(f"Downloaded {len(records)} records.")

        all_records.extend(records)

    if not all_records:
        raise ValueError("No historical price records downloaded.")

    result = run_pipeline(all_records)

    print("\nPRICE DATA PIPELINE SUMMARY")
    print("---------------------------")

    for key, value in result.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
