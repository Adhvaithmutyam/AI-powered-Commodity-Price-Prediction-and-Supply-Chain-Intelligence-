
COMMODITY_REGISTRY = {
    "copper": {
        "name": "Copper",
        "category": "metal",
        "ticker": "HG=F",
        "benchmark": "COMEX Copper Futures",
        "currency": "USD",
        "unit": "USD/lb",
        "frequency": "daily",
        "source": "Yahoo Finance"
    },

    "gold": {
        "name": "Gold",
        "category": "metal",
        "ticker": "GC=F",
        "benchmark": "COMEX Gold Futures",
        "currency": "USD",
        "unit": "USD/troy_ounce",
        "frequency": "daily",
        "source": "Yahoo Finance"
    },

    "crude oil": {
        "name": "Crude Oil",
        "category": "energy",
        "ticker": "CL=F",
        "benchmark": "WTI Crude Oil Futures",
        "currency": "USD",
        "unit": "USD/barrel",
        "frequency": "daily",
        "source": "Yahoo Finance"
    }
}


DATA_SOURCES = {
    "historical_prices": ["Yahoo Finance"],
    "fundamentals": [],
    "macroeconomic_indicators": [],
    "weather": [],
    "trade_data": []
}
