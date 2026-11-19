from datetime import date
import math

class WeatherValidator:
    NUMERIC_FIELDS = [
        "latitude", "longitude", "temperature_mean_c", "temperature_max_c",
        "temperature_min_c", "precipitation_mm", "relative_humidity_mean_pct",
        "wind_speed_max_kmh", "soil_moisture_0_100cm",
    ]

    def validate(self, record):
        errors = []
        for field in ("observation_date", "location_id", "country", "source", "frequency"):
            if record.get(field) is None:
                errors.append(f"Missing {field}")
        try:
            date.fromisoformat(record["observation_date"])
        except (ValueError, TypeError, KeyError):
            errors.append("Invalid observation_date")
        if record.get("frequency") != "daily":
            errors.append("Frequency must be daily")
        for field in self.NUMERIC_FIELDS:
            value = record.get(field)
            if value is not None:
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                    errors.append(f"Invalid numeric field: {field}")
        for field in ("precipitation_mm", "wind_speed_max_kmh"):
            if record.get(field) is not None and record[field] < 0:
                errors.append(f"Negative value: {field}")
        humidity = record.get("relative_humidity_mean_pct")
        if humidity is not None and not 0 <= humidity <= 100:
            errors.append("Humidity outside 0-100")
        if record.get("source") != "Open-Meteo Historical Weather API":
            errors.append("Invalid source")
        return not errors, errors
