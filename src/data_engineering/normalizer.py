
from datetime import datetime, timezone


def normalize_timestamp(timestamp):

    if not timestamp:
        raise ValueError("Timestamp cannot be empty")

    dt = datetime.fromisoformat(
        str(timestamp).replace("Z", "+00:00")
    )

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)

    return (
        dt.astimezone(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z")
    )


def normalize_price_record(record):

    required_fields = [
        "commodity",
        "timestamp",
        "open",
        "high",
        "low",
        "close"
    ]

    for field in required_fields:
        if field not in record or record[field] is None:
            raise ValueError(f"Missing required field: {field}")

    normalized = {
        "commodity": record["commodity"].strip().lower(),
        "timestamp": normalize_timestamp(record["timestamp"]),
        "open": float(record["open"]),
        "high": float(record["high"]),
        "low": float(record["low"]),
        "close": float(record["close"]),
        "volume": record.get("volume"),
        "currency": record.get("currency", "USD"),
        "unit": record.get("unit"),
        "source": record.get("source"),
        "ticker": record.get("ticker"),
        "benchmark": record.get("benchmark")
    }

    if normalized["volume"] is not None:
        normalized["volume"] = float(normalized["volume"])

    return normalized
