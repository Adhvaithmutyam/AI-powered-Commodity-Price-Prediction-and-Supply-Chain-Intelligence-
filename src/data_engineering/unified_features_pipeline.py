import json
import math
from pathlib import Path
from datetime import date, timedelta
from calendar import monthrange
from collections import defaultdict
from bisect import bisect_right
from statistics import pstdev

ROOT = Path("data")
VALID = ROOT / "validated"
OUTPUT = ROOT / "features"


def read_jsonl(path):
    if not path.exists():
        return []
    records = []
    with path.open(encoding="utf-8-sig") as file:
        for line in file:
            if line.strip():
                records.append(json.loads(line))
    return records


def write_jsonl(path, records):
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")


def as_date(value):
    return date.fromisoformat(str(value)[:10])


def normalize_commodity(value):
    value = str(value).strip().lower()
    if value == "crude_oil":
        return "crude oil"
    return value


def main():
    # 1. Load daily prices
    prices = read_jsonl(VALID / "price_data.jsonl")
    prices.sort(key=lambda r: (r["commodity"], r["timestamp"]))

    if not prices:
        raise ValueError("No validated price records found.")

    # 2. Calculate price features
    price_features = {}
    price_history = defaultdict(list)
    return_history = defaultdict(list)
    ema_values = {}

    for price in prices:
        commodity = normalize_commodity(price["commodity"])
        day = as_date(price["timestamp"])
        close = float(price["close"])

        closes = price_history[commodity]
        returns = return_history[commodity]

        previous_close = closes[-1] if closes else None

        daily_return = None
        log_return = None

        if previous_close is not None and previous_close > 0 and close > 0:
            daily_return = close / previous_close - 1
            log_return = math.log(close / previous_close)
            returns.append(daily_return)

        closes.append(close)

        ema = ema_values.get(commodity)
        if ema is None:
            ema = close
        else:
            alpha = 2 / 11
            ema = alpha * close + (1 - alpha) * ema

        ema_values[commodity] = ema

        price_features[(commodity, day)] = {
            "daily_return": daily_return,
            "log_return": log_return,
            "sma_5": sum(closes[-5:]) / 5 if len(closes) >= 5 else None,
            "sma_20": sum(closes[-20:]) / 20 if len(closes) >= 20 else None,
            "ema_10": ema,
            "volatility_5": pstdev(returns[-5:]) if len(returns) >= 5 else None,
        }


    # 3. Daily weather: aggregate regional locations per commodity/date
    weather = defaultdict(list)

    for file in (VALID / "weather").glob("*.jsonl"):
        for record in read_jsonl(file):
            day = as_date(record["observation_date"])

            for commodity in record.get("commodities", []):
                key = (normalize_commodity(commodity), day)
                weather[key].append(record)

    weather_features = {}

    weather_fields = [
        "temperature_mean_c",
        "temperature_max_c",
        "temperature_min_c",
        "precipitation_mm",
        "relative_humidity_mean_pct",
        "wind_speed_max_kmh",
        "soil_moisture_0_100cm",
    ]

    for key, records in weather.items():
        result = {}

        for field in weather_fields:
            values = [
                float(record[field])
                for record in records
                if record.get(field) is not None
            ]
            result["weather_" + field] = (
                sum(values) / len(values) if values else None
            )

        result["weather_location_count"] = len(records)
        weather_features[key] = result


    # 4. EIA weekly crude-oil data
    eia_files = {
        "commercial_crude_oil": "eia_commercial_inventory",
        "total_crude_oil": "eia_total_inventory",
        "crude_oil_production": "eia_production",
        "crude_oil_imports": "eia_imports",
        "crude_oil_exports": "eia_exports",
    }

    eia_series = defaultdict(list)
    eia_dir = VALID / "eia"

    for filename, feature_name in eia_files.items():
        for record in read_jsonl(eia_dir / f"{filename}.jsonl"):
            if record.get("value") is not None:
                observed = as_date(record["observation_date"])
                available = observed + timedelta(days=7)

                eia_series[feature_name].append(
                    (available, float(record["value"]), observed)
                )

    for values in eia_series.values():
        values.sort(key=lambda item: item[0])


    # 5. Monthly UN Comtrade data
    comtrade = defaultdict(list)
    comtrade_dir = VALID / "comtrade"

    for file in comtrade_dir.glob("*.jsonl"):
        if "_invalid" in file.name or "_normalization_errors" in file.name:
            continue

        for record in read_jsonl(file):
            period = record.get("period")
            value = record.get("trade_value_usd")

            if not period or value is None:
                continue

            year = int(period[:4])
            month = int(period[4:6])

            period_end = date(
                year, month, monthrange(year, month)[1]
            )
            available = period_end + timedelta(days=90)

            commodity = record["commodity"]
            flow = record["trade_flow"]
            name = f"comtrade_{commodity}_{flow}"

            comtrade[name].append(
                (available, float(value), period_end)
            )

    for values in comtrade.values():
        values.sort(key=lambda item: item[0])


    # 6. Annual World Bank macroeconomic indicators
    macro = defaultdict(list)
    macro_dir = VALID / "macroeconomic"

    for file in macro_dir.glob("*.jsonl"):
        for record in read_jsonl(file):
            if record.get("value") is None:
                continue

            year = int(record["year"])
            available = date(year + 1, 12, 31)

            name = (
                f"macro_{record['country_code'].lower()}_"
                f"{record['indicator']}"
            )

            macro[name].append(
                (available, float(record["value"]), date(year, 12, 31))
            )

    for values in macro.values():
        values.sort(key=lambda item: item[0])


    # 7. Prepare sorted indexes for as-of joins
    indexes = {}

    for collection in (eia_series, comtrade, macro):
        for name, values in collection.items():
            indexes[name] = ([item[0] for item in values], values)


    def latest_available(name, day):
        if name not in indexes:
            return None, None

        dates, values = indexes[name]
        position = bisect_right(dates, day) - 1

        if position < 0:
            return None, None

        item = values[position]
        return item[1], item[2].isoformat()


    # 8. Build unified daily records
    output = []

    trade_names = [
        f"comtrade_{commodity}_{flow}"
        for commodity in ("copper_ore", "refined_copper", "gold")
        for flow in ("imports", "exports")
    ]

    for price in prices:
        commodity = normalize_commodity(price["commodity"])
        day = as_date(price["timestamp"])

        row = {
            "date": day.isoformat(),
            "commodity": commodity,
            "open": price["open"],
            "high": price["high"],
            "low": price["low"],
            "close": price["close"],
            "volume": price["volume"],
            "currency": price["currency"],
            "unit": price["unit"],
            "price_source": price["source"],
        }

        # Calculated daily price indicators
        row.update(price_features.get((commodity, day), {}))

        # Regional weather averages
        row.update(weather_features.get((commodity, day), {
            "weather_temperature_mean_c": None,
            "weather_temperature_max_c": None,
            "weather_temperature_min_c": None,
            "weather_precipitation_mm": None,
            "weather_relative_humidity_mean_pct": None,
            "weather_wind_speed_max_kmh": None,
            "weather_soil_moisture_0_100cm": None,
            "weather_location_count": 0,
        }))

        # EIA applies only to crude oil
        for name in eia_series:
            value, observed = (
                latest_available(name, day)
                if commodity == "crude oil"
                else (None, None)
            )
            row[name] = value
            row[name + "_observation_date"] = observed

        # Comtrade applies to copper and gold
        for name in trade_names:
            applies = (
                commodity == "copper"
                and name.startswith((
                    "comtrade_copper_ore_",
                    "comtrade_refined_copper_",
                ))
            ) or (
                commodity == "gold"
                and name.startswith("comtrade_gold_")
            )

            value, observed = (
                latest_available(name, day)
                if applies
                else (None, None)
            )

            row[name] = value
            row[name + "_period_end"] = observed

        # Country-level macroeconomic indicators
        for name in macro:
            value, observed = latest_available(name, day)
            row[name] = value
            row[name + "_reference_period_end"] = observed

        output.append(row)


    # 9. Save unified dataset
    output.sort(key=lambda row: (row["commodity"], row["date"]))

    target = OUTPUT / "unified_daily_features.jsonl"
    write_jsonl(target, output)


    # 10. Save summary
    commodities = sorted({row["commodity"] for row in output})

    summary = {
        "output_file": str(target),
        "total_records": len(output),
        "commodities": commodities,
        "date_start": min(row["date"] for row in output),
        "date_end": max(row["date"] for row in output),
        "records_per_commodity": {
            commodity: sum(
                row["commodity"] == commodity for row in output
            )
            for commodity in commodities
        },
        "feature_columns": len(output[0]),
        "price_features": [
            "daily_return", "log_return", "sma_5",
            "sma_20", "ema_10", "volatility_5",
        ],
        "alignment_assumptions": {
            "eia": "Observation date plus 7 days",
            "comtrade": "Month end plus 90 days",
            "world_bank": "Reference year end plus 1 year",
            "weather": "Same-day regional average",
        },
        "note": (
            "Availability dates are conservative assumptions, not verified "
            "source publication timestamps. Weather data currently ends "
            "on 2025-12-31."
        ),
    }

    with (OUTPUT / "unified_features_summary.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(summary, file, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
