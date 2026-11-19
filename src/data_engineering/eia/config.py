import os

BASE_URL = "https://api.eia.gov/v2"

API_KEY = os.environ.get("EIA_API_KEY")

REQUEST_TIMEOUT = 30
MAX_RETRIES = 3
PAGE_SIZE = 5000

INVENTORY_SERIES = {
    "commercial_crude_oil": {
        "series_id": "WCESTUS1",
        "description": "U.S. crude oil stocks excluding SPR",
        "frequency": "weekly",
        "unit": "thousand_barrels",
    },
    "total_crude_oil": {
        "series_id": "WCRSTUS1",
        "description": "U.S. total crude oil stocks",
        "frequency": "weekly",
        "unit": "thousand_barrels",
    },
}

PRODUCTION_SERIES = {
    "crude_oil_production": {
        "series_id": "WCRFPUS2",
        "description": "U.S. weekly field production of crude oil",
        "frequency": "weekly",
        "unit": "thousand_barrels_per_day",
        "route": "petroleum/sum/sndw",
    },
}

TRADE_SERIES = {
    "crude_oil_imports": {
        "series_id": "WCRIMUS2",
        "description": "U.S. imports of crude oil",
        "frequency": "weekly",
        "unit": "thousand_barrels_per_day",
        "route": "petroleum/move/wkly",
    },
    "crude_oil_exports": {
        "series_id": "WCREXUS2",
        "description": "U.S. exports of crude oil",
        "frequency": "weekly",
        "unit": "thousand_barrels_per_day",
        "route": "petroleum/move/wkly",
    },
}
