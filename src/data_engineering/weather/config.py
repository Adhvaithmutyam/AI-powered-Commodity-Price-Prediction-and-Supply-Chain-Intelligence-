from src.data_engineering.config import DATA_DIR

BASE_URL = "https://archive-api.open-meteo.com/v1/archive"
START_DATE = "2020-01-01"
END_DATE = "2025-12-31"
REQUEST_DELAY = 1.2

DAILY_VARIABLES = [
    "temperature_2m_mean", "temperature_2m_max", "temperature_2m_min",
    "precipitation_sum", "relative_humidity_2m_mean",
    "wind_speed_10m_max", "soil_moisture_0_to_100cm_mean",
]

LOCATIONS = {
    "chile_escondida": ("Chile", -24.27, -69.07, ["copper"]),
    "peru_cerro_verde": ("Peru", -16.53, -71.59, ["copper"]),
    "china_dexing": ("China", 29.02, 117.71, ["copper"]),
    "drc_tenke": ("DR Congo", -10.60, 26.20, ["copper"]),
    "australia_kalgoorlie": ("Australia", -30.75, 121.47, ["gold"]),
    "russia_krasnoyarsk": ("Russia", 56.01, 92.87, ["gold"]),
    "canada_timmins": ("Canada", 48.48, -81.33, ["gold"]),
    "south_africa_johannesburg": ("South Africa", -26.20, 28.04, ["gold"]),
    "china_shandong": ("China", 37.40, 121.40, ["gold"]),
    "usa_midland": ("USA", 31.997, -102.08, ["crude_oil"]),
    "saudi_ghawar": ("Saudi Arabia", 25.40, 49.60, ["crude_oil"]),
    "russia_surgut": ("Russia", 61.25, 73.40, ["crude_oil"]),
    "canada_fort_mcmurray": ("Canada", 56.70, -111.40, ["crude_oil"]),
}

RAW_DIR = DATA_DIR / "raw" / "weather"
VALIDATED_DIR = DATA_DIR / "validated" / "weather"
