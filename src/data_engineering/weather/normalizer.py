from . import config

class WeatherNormalizer:
    @staticmethod
    def normalize(payload, location_id):
        if not isinstance(payload, dict) or not isinstance(payload.get("daily"), dict):
            raise ValueError("Invalid weather response")

        country, _, _, commodities = config.LOCATIONS[location_id]
        daily = payload["daily"]
        mapping = {
            "temperature_2m_mean": "temperature_mean_c",
            "temperature_2m_max": "temperature_max_c",
            "temperature_2m_min": "temperature_min_c",
            "precipitation_sum": "precipitation_mm",
            "relative_humidity_2m_mean": "relative_humidity_mean_pct",
            "wind_speed_10m_max": "wind_speed_max_kmh",
            "soil_moisture_0_to_100cm_mean": "soil_moisture_0_100cm",
        }
        records = []
        for i, day in enumerate(daily.get("time", [])):
            record = {
                "observation_date": day, "location_id": location_id,
                "country": country, "commodities": commodities,
                "latitude": payload.get("latitude"),
                "longitude": payload.get("longitude"),
                "source": "Open-Meteo Historical Weather API",
                "frequency": "daily",
            }
            for source, target in mapping.items():
                values = daily.get(source, [])
                value = values[i] if i < len(values) else None
                record[target] = float(value) if value is not None else None
            records.append(record)
        return records
