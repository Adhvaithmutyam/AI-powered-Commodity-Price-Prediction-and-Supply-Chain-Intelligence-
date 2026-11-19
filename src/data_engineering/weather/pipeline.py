from datetime import datetime, timedelta

from . import config
from .client import WeatherClient
from .normalizer import WeatherNormalizer
from .validator import WeatherValidator
from .storage import WeatherStorage


class WeatherPipeline:
    def __init__(self):
        self.client = WeatherClient()
        self.normalizer = WeatherNormalizer()
        self.validator = WeatherValidator()

    def _is_complete(self, location_id):
        path = config.VALIDATED_DIR / f"{location_id}.jsonl"
        records = WeatherStorage.load(path)

        if not records:
            return False

        start = datetime.strptime(config.START_DATE, "%Y-%m-%d").date()
        end = datetime.strptime(config.END_DATE, "%Y-%m-%d").date()

        expected = set()
        current = start
        while current <= end:
            expected.add(current.isoformat())
            current += timedelta(days=1)

        actual = {
            str(record.get("observation_date", ""))[:10]
            for record in records
        }

        return actual == expected

    def run_location(self, location_id):
        if self._is_complete(location_id):
            print(f"Skipping {location_id}: complete data already exists.")
            records = WeatherStorage.load(
                config.VALIDATED_DIR / f"{location_id}.jsonl"
            )
            return {
                "location": location_id,
                "status": "skipped",
                "valid_count": len(records),
            }

        payload = self.client.fetch(location_id)
        daily = payload["daily"]

        raw_records = [
            {
                "date": day,
                **{
                    key: values[i]
                    for key, values in daily.items()
                    if key != "time" and i < len(values)
                },
            }
            for i, day in enumerate(daily.get("time", []))
        ]

        records = self.normalizer.normalize(payload, location_id)
        valid, invalid = [], []

        for record in records:
            ok, errors = self.validator.validate(record)
            if ok:
                valid.append(record)
            else:
                invalid.append({"record": record, "errors": errors})

        WeatherStorage.save_raw(raw_records, location_id)
        WeatherStorage.save_validated(valid, location_id)
        WeatherStorage.save_invalid(invalid, location_id)

        return {
            "location": location_id,
            "status": "completed",
            "raw_count": len(raw_records),
            "valid_count": len(valid),
            "invalid_count": len(invalid),
        }

    def run_all(self):
        results = []

        for location_id in config.LOCATIONS:
            print(f"\nProcessing: {location_id}")

            try:
                results.append(self.run_location(location_id))
            except Exception as error:
                result = {
                    "location": location_id,
                    "status": "failed",
                    "error": str(error),
                }
                results.append(result)
                print(f"Failed: {location_id} - {error}")

        return results
