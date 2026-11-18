
import math
from datetime import datetime

from src.data_engineering.source_registry import COMMODITY_REGISTRY


def validate_price_record(record):

    errors = []

    required_fields = [
        "commodity",
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "ticker",
        "benchmark",
        "unit",
        "source"
    ]

    for field in required_fields:
        if field not in record or record[field] is None:
            errors.append(f"Missing field: {field}")

    if errors:
        return False, errors

    commodity = record["commodity"]

    if commodity not in COMMODITY_REGISTRY:
        errors.append("Unsupported commodity")
        return False, errors

    config = COMMODITY_REGISTRY[commodity]

    if record["ticker"] != config["ticker"]:
        errors.append("Incorrect ticker for commodity")

    if record["benchmark"] != config["benchmark"]:
        errors.append("Incorrect benchmark for commodity")

    if record["unit"] != config["unit"]:
        errors.append("Incorrect price unit for commodity")

    if record["source"] != config["source"]:
        errors.append("Incorrect data source")

    try:
        datetime.fromisoformat(
            record["timestamp"].replace("Z", "+00:00")
        )
    except (ValueError, TypeError, AttributeError):
        errors.append("Invalid timestamp")

    price_fields = ["open", "high", "low", "close"]
    prices = {}

    for field in price_fields:
        value = record[field]

        if isinstance(value, bool):
            errors.append(f"{field} cannot be boolean")
            continue

        try:
            number = float(value)

            if not math.isfinite(number):
                errors.append(f"{field} must be finite")
            else:
                prices[field] = number

        except (ValueError, TypeError):
            errors.append(f"{field} must be numeric")

    if len(prices) == 4:
        if prices["high"] < prices["low"]:
            errors.append("High cannot be lower than Low")

        if prices["high"] < max(prices["open"], prices["close"]):
            errors.append("High must be greater than or equal to Open and Close")

        if prices["low"] > min(prices["open"], prices["close"]):
            errors.append("Low must be less than or equal to Open and Close")

    volume = record.get("volume")

    if volume is not None:
        if isinstance(volume, bool):
            errors.append("Volume cannot be boolean")
        else:
            try:
                volume = float(volume)

                if not math.isfinite(volume) or volume < 0:
                    errors.append("Volume must be finite and non-negative")

            except (ValueError, TypeError):
                errors.append("Volume must be numeric")

    return len(errors) == 0, errors
