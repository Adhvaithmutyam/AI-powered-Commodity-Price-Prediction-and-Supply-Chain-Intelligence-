from datetime import datetime
import math

from .config import INVENTORY_SERIES, PRODUCTION_SERIES, TRADE_SERIES


class EIAValidator:

    REQUIRED_FIELDS = [
        "observation_date",
        "commodity",
        "value",
        "unit",
        "frequency",
        "series_id",
        "source",
    ]

    @classmethod
    def _validate_common(cls, record, expected_unit, expected_series):
        errors = []

        if not isinstance(record, dict):
            return False, ["Record must be a dictionary"]

        for field in cls.REQUIRED_FIELDS:
            if field not in record:
                errors.append(f"Missing required field: {field}")

        if errors:
            return False, errors

        if record["commodity"] != "crude oil":
            errors.append("Invalid commodity")

        if record["unit"] != expected_unit:
            errors.append("Invalid unit")

        if record["frequency"] != "weekly":
            errors.append("Invalid frequency")

        if record["series_id"] != expected_series:
            errors.append("Invalid series ID")

        if record["source"] != "EIA":
            errors.append("Invalid source")

        try:
            datetime.fromisoformat(record["observation_date"])
        except (TypeError, ValueError):
            errors.append("Invalid observation date")

        value = record["value"]

        if value is not None:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                errors.append("Value must be numeric or None")
            elif not math.isfinite(value):
                errors.append("Value must be finite")
            elif value < 0:
                errors.append("Value cannot be negative")

        return len(errors) == 0, errors

    @classmethod
    def validate(cls, record, inventory_type="commercial_crude_oil"):
        if inventory_type not in INVENTORY_SERIES:
            return False, ["Unsupported inventory type"]

        if not isinstance(record, dict):
            return False, ["Record must be a dictionary"]

        if record.get("inventory_type") != inventory_type:
            return False, ["Invalid inventory type"]

        series = INVENTORY_SERIES[inventory_type]

        return cls._validate_common(
            record,
            series["unit"],
            series["series_id"],
        )

    @classmethod
    def validate_production(
        cls, record, production_type="crude_oil_production"
    ):
        if production_type not in PRODUCTION_SERIES:
            return False, ["Unsupported production type"]

        if not isinstance(record, dict):
            return False, ["Record must be a dictionary"]

        if record.get("production_type") != production_type:
            return False, ["Invalid production type"]

        series = PRODUCTION_SERIES[production_type]

        return cls._validate_common(
            record,
            series["unit"],
            series["series_id"],
        )

    @classmethod
    def validate_trade(cls, record, trade_type="crude_oil_imports"):
        if trade_type not in TRADE_SERIES:
            return False, ["Unsupported trade type"]

        if not isinstance(record, dict):
            return False, ["Record must be a dictionary"]

        if record.get("trade_type") != trade_type:
            return False, ["Invalid trade type"]

        series = TRADE_SERIES[trade_type]

        return cls._validate_common(
            record,
            series["unit"],
            series["series_id"],
        )
