import math

from .config import COMMODITIES, TRADE_FLOWS, REPORTER_CODE


class ComtradeValidator:

    REQUIRED_FIELDS = [
        "observation_date",
        "period",
        "commodity",
        "trade_flow",
        "trade_value_usd",
        "reporter_code",
        "partner_code",
        "hs_code",
        "classification",
        "source",
    ]

    @staticmethod
    def _valid_number(value):
        if value is None:
            return True

        try:
            return math.isfinite(float(value))
        except (TypeError, ValueError):
            return False

    def validate(self, record):
        errors = []

        if not isinstance(record, dict):
            return False, ["Record must be a dictionary"]

        for field in self.REQUIRED_FIELDS:
            if record.get(field) is None:
                errors.append(f"Missing required field: {field}")

        if errors:
            return False, errors

        commodity = record["commodity"]
        flow = record["trade_flow"]

        if commodity not in COMMODITIES:
            errors.append("Unsupported commodity")
        elif record["hs_code"] != COMMODITIES[commodity]["hs_code"]:
            errors.append("HS code does not match commodity")

        if flow not in TRADE_FLOWS:
            errors.append("Unsupported trade flow")

        if record["reporter_code"] != REPORTER_CODE:
            errors.append("Unexpected reporter code")

        if record["source"] != "UN Comtrade":
            errors.append("Invalid data source")

        if record["period"] != record["observation_date"][:7].replace("-", ""):
            errors.append("Period and observation date do not match")

        if not self._valid_number(record["trade_value_usd"]):
            errors.append("Invalid trade value")
        elif record["trade_value_usd"] < 0:
            errors.append("Trade value cannot be negative")

        numeric_fields = [
            "cif_value_usd",
            "fob_value_usd",
            "quantity",
            "net_weight_kg",
            "gross_weight_kg",
        ]

        for field in numeric_fields:
            value = record.get(field)

            if not self._valid_number(value):
                errors.append(f"Invalid numeric value: {field}")
            elif value is not None and value < 0:
                errors.append(f"Negative value not allowed: {field}")

        if record["frequency"] != "monthly":
            errors.append("Frequency must be monthly")

        if record["unit"] != "USD":
            errors.append("Trade value unit must be USD")

        return len(errors) == 0, errors
