
BASE_URL = "https://api.worldbank.org/v2"

COUNTRY_CODES = ["USA", "CHN"]

INDICATORS = {
    "gdp_growth": {
        "code": "NY.GDP.MKTP.KD.ZG",
        "description": "GDP growth (annual %)"
    },
    "inflation": {
        "code": "FP.CPI.TOTL.ZG",
        "description": "Consumer price inflation (annual %)"
    },
    "manufacturing": {
        "code": "NV.IND.MANF.ZS",
        "description": "Manufacturing value added (% of GDP)"
    },
    "imports": {
        "code": "NE.IMP.GNFS.CD",
        "description": "Imports of goods and services (current USD)"
    },
    "exports": {
        "code": "NE.EXP.GNFS.CD",
        "description": "Exports of goods and services (current USD)"
    }
}

START_YEAR = 2015
END_YEAR = 2026

REQUEST_TIMEOUT = 30
MAX_RETRIES = 3
