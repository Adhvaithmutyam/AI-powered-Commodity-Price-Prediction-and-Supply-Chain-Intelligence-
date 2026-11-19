from datetime import date, timedelta
from . import config
from .storage import WeatherStorage

class WeatherQualityReport:
    def generate(self, location_id):
        records = WeatherStorage.load(config.VALIDATED_DIR / f"{location_id}.jsonl")
        dates = [r["observation_date"] for r in records]
        unique = sorted(set(dates))
        missing = 0
        if unique:
            current, end = date.fromisoformat(unique[0]), date.fromisoformat(unique[-1])
            available = set(unique)
            while current <= end:
                missing += current.isoformat() not in available
                current += timedelta(days=1)
        return {
            "location": location_id, "total_records": len(records),
            "first_date": unique[0] if unique else None,
            "last_date": unique[-1] if unique else None,
            "missing_dates": missing, "duplicate_dates": len(dates)-len(unique),
            "missing_precipitation": sum(r.get("precipitation_mm") is None for r in records),
            "missing_temperature": sum(r.get("temperature_mean_c") is None for r in records),
        }

    def generate_all(self):
        return {k: self.generate(k) for k in config.LOCATIONS}
