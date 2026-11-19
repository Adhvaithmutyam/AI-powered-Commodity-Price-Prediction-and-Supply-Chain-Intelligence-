import pytest

from src.data_engineering.comtrade.normalizer import ComtradeNormalizer
from src.data_engineering.comtrade.validator import ComtradeValidator
from src.data_engineering.comtrade.pipeline import ComtradePipeline
from src.data_engineering.comtrade.storage import ComtradeStorage


@pytest.fixture
def raw_record():
    return {
        "period": "202501",
        "flowCode": "M",
        "cmdCode": "7403",
        "primaryValue": 705413208,
        "cifvalue": 705413208,
        "fobvalue": 699593084,
        "qty": 0,
        "qtyUnitAbbr": None,
        "isQtyEstimated": False,
        "netWgt": None,
        "isNetWgtEstimated": False,
        "grossWgt": None,
        "isGrossWgtEstimated": False,
        "reporterCode": 842,
        "reporterISO": "USA",
        "partnerCode": 0,
        "partnerISO": None,
        "classificationCode": "H6",
        "isOriginalClassification": True,
        "isReported": False,
        "isAggregate": True,
        "legacyEstimationFlag": None,
    }


def test_normalizer(raw_record):
    record = ComtradeNormalizer.normalize(
        raw_record, "refined_copper", "imports"
    )
    assert record["trade_value_usd"] == 705413208
    assert record["period"] == "202501"
    assert record["commodity"] == "refined_copper"


def test_normalizer_rejects_wrong_flow(raw_record):
    with pytest.raises(ValueError):
        ComtradeNormalizer.normalize(
            raw_record, "refined_copper", "exports"
        )


def test_normalizer_rejects_wrong_commodity(raw_record):
    with pytest.raises(ValueError):
        ComtradeNormalizer.normalize(
            raw_record, "gold", "imports"
        )


def test_validator_accepts_valid_record(raw_record):
    record = ComtradeNormalizer.normalize(
        raw_record, "refined_copper", "imports"
    )
    is_valid, errors = ComtradeValidator().validate(record)
    assert is_valid
    assert errors == []


def test_validator_rejects_negative_trade_value(raw_record):
    record = ComtradeNormalizer.normalize(
        raw_record, "refined_copper", "imports"
    )
    record["trade_value_usd"] = -100

    is_valid, errors = ComtradeValidator().validate(record)

    assert not is_valid
    assert "Trade value cannot be negative" in errors


def test_storage_save_and_load(tmp_path):
    file_path = tmp_path / "comtrade.jsonl"
    records = [{"commodity": "gold", "trade_value_usd": 100}]

    ComtradeStorage.save_jsonl(records, file_path)

    assert ComtradeStorage.load_jsonl(file_path) == records


def test_pipeline_with_mocked_api(raw_record, tmp_path, monkeypatch):
    from src.data_engineering.comtrade import storage

    monkeypatch.setattr(
        storage, "COMTRADE_RAW_DIR", tmp_path / "raw"
    )
    monkeypatch.setattr(
        storage, "COMTRADE_VALIDATED_DIR", tmp_path / "validated"
    )

    pipeline = ComtradePipeline()

    class MockClient:
        def fetch_month(self, commodity, flow, period):
            return [raw_record]

    pipeline.client = MockClient()

    result = pipeline.run_month(
        "refined_copper", "imports", "202501"
    )

    assert result["raw_count"] == 1
    assert result["valid_count"] == 1
    assert result["invalid_count"] == 0
    assert result["normalization_error_count"] == 0

    saved = ComtradeStorage.load_jsonl(
        tmp_path / "validated"
        / "refined_copper_imports_202501.jsonl"
    )
    assert len(saved) == 1
