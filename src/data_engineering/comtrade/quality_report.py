from .storage import ComtradeStorage, COMTRADE_VALIDATED_DIR


class ComtradeQualityReport:

    @staticmethod
    def generate(commodity, flow):
        files = sorted(
            COMTRADE_VALIDATED_DIR.glob(
                f"{commodity}_{flow}_*.jsonl"
            )
        )

        records = []
        for file_path in files:
            if file_path.stem.endswith(
                ("_invalid", "_normalization_errors")
            ):
                continue
            records.extend(ComtradeStorage.load_jsonl(file_path))

        periods = sorted({
            record["period"]
            for record in records
            if record.get("period")
        })

        expected_periods = set()
        if periods:
            year, month = int(periods[0][:4]), int(periods[0][4:])
            end_year, end_month = (
                int(periods[-1][:4]), int(periods[-1][4:])
            )

            while (year, month) <= (end_year, end_month):
                expected_periods.add(f"{year:04d}{month:02d}")
                month += 1
                if month == 13:
                    year += 1
                    month = 1

        missing_periods = sorted(expected_periods - set(periods))

        dates = [
            record.get("observation_date")
            for record in records
            if record.get("observation_date")
        ]

        missing_values = sum(
            record.get("trade_value_usd") is None
            for record in records
        )

        return {
            "commodity": commodity,
            "flow": flow,
            "total_records": len(records),
            "months_available": len(periods),
            "first_period": periods[0] if periods else None,
            "last_period": periods[-1] if periods else None,
            "missing_periods": missing_periods,
            "duplicate_observations": len(dates) - len(set(dates)),
            "missing_trade_values": missing_values,
            "missing_net_weights": sum(
                record.get("net_weight_kg") is None
                for record in records
            ),
            "estimated_quantities": sum(
                record.get("quantity_estimated") is True
                for record in records
            ),
        }
