from src.data_engineering.comtrade.quality_report import ComtradeQualityReport
from src.data_engineering.comtrade import quality_report


def test_quality_report(tmp_path, monkeypatch):
    monkeypatch.setattr(
        quality_report,
        "COMTRADE_VALIDATED_DIR",
        tmp_path,
    )

    records = [
        {
            "period": "202501",
            "observation_date": "2025-01-01T00:00:00+00:00",
            "trade_value_usd": 100,
            "net_weight_kg": None,
            "quantity_estimated": False,
        },
        {
            "period": "202503",
            "observation_date": "2025-03-01T00:00:00+00:00",
            "trade_value_usd": 200,
            "net_weight_kg": 50,
            "quantity_estimated": True,
        },
    ]

    from src.data_engineering.comtrade.storage import ComtradeStorage

    ComtradeStorage.save_jsonl(
        records,
        tmp_path / "gold_imports_202501.jsonl",
    )

    # Store March separately, as the real pipeline does.
    ComtradeStorage.save_jsonl(
        records[1:],
        tmp_path / "gold_imports_202503.jsonl",
    )

    report = ComtradeQualityReport.generate("gold", "imports")

    assert report["total_records"] == 3
    assert report["months_available"] == 2
    assert report["missing_periods"] == ["202502"]
    assert report["missing_net_weights"] == 1
    assert report["estimated_quantities"] == 2
