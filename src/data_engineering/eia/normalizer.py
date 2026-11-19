from datetime import datetime, timezone
import math


class EIANormalizer:

    @staticmethod
    def _normalize_value(value):
        if value is None:
            return None

        try:
            number = float(value)
        except (TypeError, ValueError):
            raise ValueError("Invalid numeric value")

        if not math.isfinite(number):
            raise ValueError("Value must be finite")

        return number

    @staticmethod
    def _normalize_date(value):
        try:
            parsed = datetime.strptime(value, "%Y-%m-%d")
        except (TypeError, ValueError):
            raise ValueError("Invalid observation date")

        return parsed.replace(tzinfo=timezone.utc).isoformat()

    @classmethod
    def normalize(cls, record, inventory_type="commercial_crude_oil"):
        from .config import INVENTORY_SERIES

        if inventory_type not in INVENTORY_SERIES:
            raise ValueError("Unsupported inventory type")

        series = INVENTORY_SERIES[inventory_type]

        return {
            "observation_date": cls._normalize_date(record["period"]),
            "commodity": "crude oil",
            "inventory_type": inventory_type,
            "value": cls._normalize_value(record.get("value")),
            "unit": series["unit"],
            "frequency": series["frequency"],
            "series_id": series["series_id"],
            "area": record.get("area-name"),
            "product": record.get("product-name"),
            "process": record.get("process-name"),
            "source": "EIA",
        }

    @classmethod
    def normalize_production(cls, record, production_type="crude_oil_production"):
        from .config import PRODUCTION_SERIES

        if production_type not in PRODUCTION_SERIES:
            raise ValueError("Unsupported production type")

        series = PRODUCTION_SERIES[production_type]

        return {
            "observation_date": cls._normalize_date(record["period"]),
            "commodity": "crude oil",
            "production_type": production_type,
            "value": cls._normalize_value(record.get("value")),
            "unit": series["unit"],
            "frequency": series["frequency"],
            "series_id": series["series_id"],
            "area": record.get("area-name"),
            "product": record.get("product-name"),
            "process": record.get("process-name"),
            "source": "EIA",
        }

    @classmethod
    def normalize_trade(cls, record, trade_type="crude_oil_imports"):
        from .config import TRADE_SERIES

        if trade_type not in TRADE_SERIES:
            raise ValueError("Unsupported trade type")

        series = TRADE_SERIES[trade_type]

        return {
            "observation_date": cls._normalize_date(record["period"]),
            "commodity": "crude oil",
            "trade_type": trade_type,
            "value": cls._normalize_value(record.get("value")),
            "unit": series["unit"],
            "frequency": series["frequency"],
            "series_id": series["series_id"],
            "area": record.get("area-name"),
            "product": record.get("product-name"),
            "process": record.get("process-name"),
            "source": "EIA",
        }
