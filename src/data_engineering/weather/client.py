import time
import requests
from . import config


class WeatherClient:
    def __init__(self):
        self.session = requests.Session()
        self.last_request = 0
        self.max_attempts = 4

    def fetch(self, location_id):
        if location_id not in config.LOCATIONS:
            raise ValueError(f"Unknown location: {location_id}")

        country, latitude, longitude, commodities = config.LOCATIONS[location_id]

        params = {
            "latitude": latitude,
            "longitude": longitude,
            "start_date": config.START_DATE,
            "end_date": config.END_DATE,
            "daily": ",".join(config.DAILY_VARIABLES),
            "timezone": "UTC",
            "temperature_unit": "celsius",
            "wind_speed_unit": "kmh",
            "precipitation_unit": "mm",
        }

        for attempt in range(self.max_attempts):
            wait = config.REQUEST_DELAY - (time.time() - self.last_request)
            if wait > 0:
                time.sleep(wait)

            try:
                response = self.session.get(
                    config.BASE_URL, params=params, timeout=90
                )
                self.last_request = time.time()

                if response.status_code == 429:
                    if attempt == self.max_attempts - 1:
                        response.raise_for_status()

                    retry_after = response.headers.get("Retry-After")
                    delay = int(retry_after) if retry_after and retry_after.isdigit() else 60 * (2 ** attempt)
                    print(f"Rate limited for {location_id}. Retrying in {delay} seconds...")
                    time.sleep(delay)
                    continue

                response.raise_for_status()
                return response.json()

            except (requests.Timeout, requests.ConnectionError):
                if attempt == self.max_attempts - 1:
                    raise
                delay = 15 * (2 ** attempt)
                print(f"Connection issue for {location_id}. Retrying in {delay} seconds...")
                time.sleep(delay)

        raise RuntimeError(f"Weather download failed for {location_id}")
