import json
from collections import Counter
from datetime import date, timedelta

from ..config import DATA_DIR


class EIAQualityReport:

    @staticmethod
    def _generate(file_path, report_type, type_name):
        if not file_path.exists():
            raise FileNotFoundError(file_path)

        with file_path.open("r", encoding="utf-8") as file:
            records = [json.loads(line) for line in file if line.strip()]

        dates = [record["observation_date"][:10] for record in records]
        date_counts = Counter(dates)
        valid_dates = sorted(date_counts)

        missing_values = sum(
            record.get("value") is None for record in records
        )

        duplicates = sum(
            count - 1 for count in date_counts.values() if count > 1
        )

        missing_week_dates = []

        if valid_dates:
            current = date.fromisoformat(valid_dates[0])
            latest = date.fromisoformat(valid_dates[-1])

            while current <= latest:
                if current.isoformat() not in date_counts:
                    missing_week_dates.append(current.isoformat())
                current += timedelta(days=7)

        return {
            type_name: report_type,
            "total_records": len(records),
            "missing_values": missing_values,
            "duplicate_records": duplicates,
            "missing_weeks": len(missing_week_dates),
            "missing_week_dates": missing_week_dates,
            "earliest_date": valid_dates[0] if valid_dates else None,
            "latest_date": valid_dates[-1] if valid_dates else None,
        }

    @classmethod
    def generate(cls, inventory_type):
        file_path = (
            DATA_DIR / "validated" / "eia" / f"{inventory_type}.jsonl"
        )
        return cls._generate(file_path, inventory_type, "inventory_type")

    @classmethod
    def generate_production(cls, production_type="crude_oil_production"):
        file_path = (
            DATA_DIR / "validated" / "eia" / f"{production_type}.jsonl"
        )
        return cls._generate(file_path, production_type, "production_type")


    @classmethod
    def generate_trade(cls, trade_type="crude_oil_imports"):
        file_path = (
            DATA_DIR / "validated" / "eia" / f"{trade_type}.jsonl"
        )
        return cls._generate(file_path, trade_type, "trade_type")

if __name__ == "__main__":
    for inventory in ["commercial_crude_oil", "total_crude_oil"]:
        print(EIAQualityReport.generate(inventory))

    print(EIAQualityReport.generate_production())
