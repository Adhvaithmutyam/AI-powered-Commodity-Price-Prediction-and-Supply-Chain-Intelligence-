import numpy as np
import pandas as pd


def generate_price_features(records):

    if not records:
        return []

    df = pd.DataFrame(records)

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        utc=True
    )

    df = df.sort_values(
        ["commodity", "benchmark", "timestamp"]
    ).copy()

    df["close"] = pd.to_numeric(
        df["close"],
        errors="coerce"
    )

    group_columns = ["commodity", "benchmark"]

    grouped = df.groupby(
        group_columns,
        sort=False
    )["close"]

    # Daily percentage return
    df["daily_return"] = grouped.pct_change(fill_method=None)

    # Log return
    df["log_return"] = grouped.transform(
        lambda values: np.log(
            values.where(values > 0) /
            values.shift(1).where(values.shift(1) > 0)
        )
    )

    # Simple moving averages
    df["sma_5"] = grouped.transform(
        lambda values: values.rolling(5).mean()
    )

    df["sma_20"] = grouped.transform(
        lambda values: values.rolling(20).mean()
    )

    # Exponential moving average
    df["ema_10"] = grouped.transform(
        lambda values: values.ewm(
            span=10,
            adjust=False
        ).mean()
    )

    # Rolling volatility
    df["volatility_5"] = grouped.transform(
        lambda values: values.pct_change(
            fill_method=None
        ).rolling(5).std()
    )

    # Replace invalid numeric values with null
    df = df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    # Convert timestamps to JSON-friendly strings
    df["timestamp"] = df["timestamp"].apply(
        lambda value: value.isoformat().replace("+00:00", "Z")
    )

    # Convert pandas NaN values to Python None
    df = df.astype(object).where(pd.notna(df), None)

    return df.to_dict(orient="records")
