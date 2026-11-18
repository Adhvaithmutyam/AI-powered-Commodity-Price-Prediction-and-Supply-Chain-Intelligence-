from pathlib import Path

# Project directory
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Data directories
DATA_DIR = PROJECT_ROOT / "data"

RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
VALIDATED_DATA_DIR = DATA_DIR / "validated"

# Supported commodities
COMMODITIES = [
    "copper",
    "gold",
    "crude oil"
]

# Prediction horizons
PREDICTION_HORIZONS = [1, 5]

# Data frequency
DATA_FREQUENCY = "daily"