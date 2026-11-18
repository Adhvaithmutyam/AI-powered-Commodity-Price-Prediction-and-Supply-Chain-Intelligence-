
from src.data_engineering.feature_engineering import generate_price_features


def create_sample_records():
    records = []

    for day in range(1, 26):
        records.append({
            "commodity": "copper",
            "benchmark": "COMEX Copper Futures",
            "timestamp": f"2026-10-{day:02d}T00:00:00Z",
            "close": 6.0 + (day * 0.01)
        })

    return records


def test_feature_record_count():
    records = create_sample_records()

    features = generate_price_features(records)

    assert len(features) == 25


def test_daily_return():
    records = create_sample_records()

    features = generate_price_features(records)

    assert features[1]["daily_return"] is not None


def test_moving_average():
    records = create_sample_records()

    features = generate_price_features(records)

    assert features[19]["sma_20"] is not None


def test_ema():
    records = create_sample_records()

    features = generate_price_features(records)

    assert features[9]["ema_10"] is not None


def test_volatility():
    records = create_sample_records()

    features = generate_price_features(records)

    assert features[5]["volatility_5"] is not None


def test_empty_records():
    features = generate_price_features([])

    assert features == []
