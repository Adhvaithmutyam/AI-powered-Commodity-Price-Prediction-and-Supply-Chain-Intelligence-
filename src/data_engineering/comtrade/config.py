BASE_URL = "https://comtradeapi.un.org/data/v1/get"

REPORTER_CODE = 842
REPORTER_NAME = "USA"
PARTNER_CODE = 0

COMMODITIES = {
    "copper_ore": {
        "hs_code": "2603",
        "description": "Copper ores and concentrates",
    },
    "refined_copper": {
        "hs_code": "7403",
        "description": "Refined copper and copper alloys",
    },
    "gold": {
        "hs_code": "7108",
        "description": "Gold",
    },
}

TRADE_FLOWS = {
    "imports": "M",
    "exports": "X",
}

FREQUENCY = "M"
CLASSIFICATION = "HS"

START_YEAR = 2020
END_YEAR = 2025

REQUEST_TIMEOUT = 30
MAX_RETRIES = 3
RETRY_BACKOFF = 5
REQUEST_DELAY = 1.2

MAX_RECORDS = 500
