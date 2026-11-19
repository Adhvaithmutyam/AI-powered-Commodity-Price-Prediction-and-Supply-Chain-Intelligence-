from datetime import datetime, timezone
import math


class ComtradeNormalizer:

    @staticmethod
    def _normalize_number(value):
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
    def _normalize_date(period):
        if (
            not isinstance(period, str)
            or len(period) != 6
            or not period.isdigit()
        ):
            raise ValueError("Invalid monthly period")

        try:
            parsed = datetime.strptime(period, "%Y%m")
        except ValueError:
            raise ValueError("Invalid monthly period")

        return parsed.replace(
            tzinfo=timezone.utc
        ).isoformat()

    @classmethod
    def normalize(cls, record, commodity, flow):
        from .config import COMMODITIES, TRADE_FLOWS

        if commodity not in COMMODITIES:
            raise ValueError("Unsupported commodity")

        if flow not in TRADE_FLOWS:
            raise ValueError("Unsupported trade flow")

        if not isinstance(record, dict):
            raise ValueError("Record must be a dictionary")

        period = record.get("period")
        if not period:
            raise ValueError("Missing trade period")

        expected_flow = TRADE_FLOWS[flow]
        if record.get("flowCode") != expected_flow:
            raise ValueError("Trade flow does not match")

        commodity_config = COMMODITIES[commodity]

        if record.get("cmdCode") != commodity_config["hs_code"]:
            raise ValueError("Commodity HS code does not match")

        return {
            "observation_date": cls._normalize_date(period),
            "period": period,
            "commodity": commodity,
            "commodity_description": commodity_config["description"],
            "trade_flow": flow,
            "trade_value_usd": cls._normalize_number(
                record.get("primaryValue")
            ),
            "cif_value_usd": cls._normalize_number(
                record.get("cifvalue")
            ),
            "fob_value_usd": cls._normalize_number(
                record.get("fobvalue")
            ),
            "quantity": cls._normalize_number(record.get("qty")),
            "quantity_unit": record.get("qtyUnitAbbr"),
            "quantity_estimated": record.get("isQtyEstimated"),
            "net_weight_kg": cls._normalize_number(record.get("netWgt")),
            "net_weight_estimated": record.get("isNetWgtEstimated"),
            "gross_weight_kg": cls._normalize_number(
                record.get("grossWgt")
            ),
            "gross_weight_estimated": record.get("isGrossWgtEstimated"),
            "reporter_code": record.get("reporterCode"),
            "reporter": record.get("reporterISO")
            or record.get("reporterDesc"),
            "partner_code": record.get("partnerCode"),
            "partner": record.get("partnerISO")
            or record.get("partnerDesc"),
            "hs_code": record.get("cmdCode"),
            "classification": record.get("classificationCode"),
            "is_original_classification": record.get(
                "isOriginalClassification"
            ),
            "is_reported": record.get("isReported"),
            "is_aggregate": record.get("isAggregate"),
            "legacy_estimation_flag": record.get("legacyEstimationFlag"),
            "source": "UN Comtrade",
            "frequency": "monthly",
            "unit": "USD",
        }
