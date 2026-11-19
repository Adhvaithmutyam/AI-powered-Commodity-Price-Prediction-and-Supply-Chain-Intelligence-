from src.data_engineering.weather.normalizer import WeatherNormalizer
from src.data_engineering.weather.validator import WeatherValidator
from src.data_engineering.weather.storage import WeatherStorage

def payload():
    return {
        "latitude": -24.27, "longitude": -69.07,
        "daily": {
            "time": ["2025-01-01", "2025-01-02"],
            "temperature_2m_mean": [22.5, 23.0],
            "temperature_2m_max": [30, 31],
            "temperature_2m_min": [15, 16],
            "precipitation_sum": [0, 1.2],
            "relative_humidity_2m_mean": [40, 45],
            "wind_speed_10m_max": [18, 20],
            "soil_moisture_0_to_100cm_mean": [0.2, 0.21],
        }
    }

def test_normalizer():
    records = WeatherNormalizer.normalize(payload(), "chile_escondida")
    assert len(records) == 2
    assert records[0]["temperature_mean_c"] == 22.5

def test_validator():
    record = WeatherNormalizer.normalize(payload(), "chile_escondida")[0]
    assert WeatherValidator().validate(record)[0]
    record["relative_humidity_mean_pct"] = 120
    assert not WeatherValidator().validate(record)[0]

def test_storage(tmp_path):
    records = WeatherNormalizer.normalize(payload(), "chile_escondida")
    path = tmp_path / "weather.jsonl"
    WeatherStorage.save(records, path)
    assert len(WeatherStorage.load(path)) == 2
