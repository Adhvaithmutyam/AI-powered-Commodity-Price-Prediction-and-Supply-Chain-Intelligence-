import pytest

from src.data_engineering.eia.normalizer import EIANormalizer
from src.data_engineering.eia.validator import EIAValidator
from src.data_engineering.eia.storage import EIAStorage


@pytest.fixture
def raw_record():
    return {
        "period": "2026-11-13",
        "value": "417797",
        "series": "WCESTUS1",
        "duoarea": "NUS",
        "product": "EPC0",
        "process": "SAX",
    }


def test_normalize_inventory_record(raw_record):
    record = EIANormalizer.normalize(raw_record)

    assert record["commodity"] == "crude oil"
    assert record["value"] == 417797.0
    assert record["unit"] == "thousand_barrels"
    assert record["frequency"] == "weekly"


def test_normalizer_rejects_invalid_date(raw_record):
    raw_record["period"] = "invalid-date"

    with pytest.raises(ValueError):
        EIANormalizer.normalize(raw_record)


def test_normalizer_rejects_infinite_value(raw_record):
    raw_record["value"] = "inf"

    with pytest.raises(ValueError):
        EIANormalizer.normalize(raw_record)


def test_validator_accepts_valid_record(raw_record):
    record = EIANormalizer.normalize(raw_record)

    is_valid, errors = EIAValidator().validate(record)

    assert is_valid is True
    assert errors == []


def test_validator_rejects_negative_value(raw_record):
    record = EIANormalizer.normalize(raw_record)
    record["value"] = -100

    is_valid, errors = EIAValidator().validate(record)

    assert is_valid is False
    assert errors


def test_save_and_load_inventory_records(tmp_path):
    records = [
        {
            "observation_date": "2026-11-13T00:00:00+00:00",
            "commodity": "crude oil",
            "inventory_type": "commercial_crude_oil",
            "value": 417797.0,
            "unit": "thousand_barrels",
            "frequency": "weekly",
            "series_id": "WCESTUS1",
            "area": "NUS",
            "product": "EPC0",
            "process": "SAX",
            "source": "U.S. Energy Information Administration",
        }
    ]

    file_path = tmp_path / "inventory.jsonl"

    EIAStorage.save_jsonl(records, file_path)
    loaded = EIAStorage.load_jsonl(file_path)

    assert loaded == records


def test_save_normalization_errors(tmp_path, monkeypatch):
    from src.data_engineering.eia import storage

    monkeypatch.setattr(storage, "EIA_VALIDATED_DIR", tmp_path)

    records = [
        {
            "record": {"period": "invalid-date"},
            "error": "Invalid observation period.",
        }
    ]

    path = EIAStorage.save_normalization_errors(
        records, "test_inventory"
    )

    assert path.exists()
    assert EIAStorage.load_jsonl(path) == records

@pytest.fixture
def raw_trade_record():
    return {
        "period": "2026-11-13",
        "value": "5576",
        "series": "WCRIMUS2",
        "area-name": "U.S.",
        "product-name": "Crude Oil",
        "process-name": "Imports",
    }


@pytest.mark.parametrize(
    "trade_type,series_id,process",
    [
        ("crude_oil_imports", "WCRIMUS2", "Imports"),
        ("crude_oil_exports", "WCREXUS2", "Exports"),
    ],
)
def test_normalize_trade_record(
    raw_trade_record, trade_type, series_id, process
):
    raw_trade_record["process-name"] = process

    record = EIANormalizer.normalize_trade(
        raw_trade_record, trade_type
    )

    assert record["commodity"] == "crude oil"
    assert record["trade_type"] == trade_type
    assert record["series_id"] == series_id
    assert record["frequency"] == "weekly"
    assert record["unit"] == "thousand_barrels_per_day"
    assert record["source"] == "EIA"


@pytest.mark.parametrize(
    "trade_type,series_id,process",
    [
        ("crude_oil_imports", "WCRIMUS2", "Imports"),
        ("crude_oil_exports", "WCREXUS2", "Exports"),
    ],
)
def test_validator_accepts_valid_trade_record(
    raw_trade_record, trade_type, series_id, process
):
    raw_trade_record["process-name"] = process

    record = EIANormalizer.normalize_trade(
        raw_trade_record, trade_type
    )

    is_valid, errors = EIAValidator.validate_trade(
        record, trade_type
    )

    assert is_valid is True
    assert errors == []


def test_trade_validator_rejects_negative_value(raw_trade_record):
    record = EIANormalizer.normalize_trade(
        raw_trade_record, "crude_oil_imports"
    )
    record["value"] = -100

    is_valid, errors = EIAValidator.validate_trade(
        record, "crude_oil_imports"
    )

    assert is_valid is False
    assert errors


def test_save_and_load_trade_records(tmp_path, monkeypatch):
    from src.data_engineering.eia import storage

    monkeypatch.setattr(storage, "EIA_VALIDATED_DIR", tmp_path)

    records = [
        {
            "observation_date": "2026-11-13T00:00:00+00:00",
            "commodity": "crude oil",
            "trade_type": "crude_oil_imports",
            "value": 5576.0,
            "unit": "thousand_barrels_per_day",
            "frequency": "weekly",
            "series_id": "WCRIMUS2",
            "source": "EIA",
        }
    ]

    path = EIAStorage.save_trade_validated(
        records, "crude_oil_imports"
    )

    assert path.exists()
    assert EIAStorage.load_jsonl(path) == records
