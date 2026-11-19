"""
Unit and integration tests for Real Historical Commodity Data Loader (MLOps 2).
All tests execute 100% offline with zero external network calls using mocked providers.
Compatible with both standard library unittest and pytest.
"""

from datetime import datetime, timedelta, timezone
from pathlib import Path
import unittest
from unittest.mock import MagicMock, patch

from src.price_prediction.config import PredictionConfig
from src.price_prediction.data_loader import (
    BasePriceProvider,
    DataUnavailableError,
    DataValidationError,
    PriceDataLoader,
    ProviderError,
    YahooFinanceProvider,
)


class TestPriceDataLoader(unittest.TestCase):
    """Test suite for PriceDataLoader, caching, validation, and normalization."""

    def setUp(self):
        self.test_dir = Path("data/raw/prices/test_tmp")
        self.test_dir.mkdir(parents=True, exist_ok=True)
        self.config = PredictionConfig(
            price_data_dir=self.test_dir,
            cache_expiry_hours=24,
            lookback_years=3,
        )

        # 10 days of valid synthetic OHLCV records
        self.mock_records = []
        for i in range(10):
            day = 5 + i
            dt_str = f"2026-01-{day:02d}T18:00:00Z"
            close = 75.0 + i * 0.5
            self.mock_records.append({
                "timestamp": dt_str,
                "open": close - 0.2,
                "high": close + 0.8,
                "low": close - 0.6,
                "close": close,
                "volume": 50000 + i * 1000,
            })

    def tearDown(self):
        # Clean up test files
        if self.test_dir.exists():
            for f in self.test_dir.glob("*.csv"):
                try:
                    f.unlink()
                except Exception:
                    pass
            try:
                self.test_dir.rmdir()
            except Exception:
                pass

    def test_validate_ohlcv_standardizes_schema(self):
        loader = PriceDataLoader(config=self.config)
        validated = loader.validate_ohlcv(self.mock_records, commodity="crude oil")

        self.assertEqual(len(validated), 10)
        first = validated[0]
        for field in ["timestamp", "commodity", "open", "high", "low", "close", "volume"]:
            self.assertIn(field, first)
        self.assertEqual(first["commodity"], "crude oil")
        self.assertTrue(first["timestamp"].endswith("T18:00:00Z"))

    def test_validate_ohlcv_filters_invalid_prices(self):
        loader = PriceDataLoader(config=self.config)
        corrupted = [
            {"timestamp": "2026-01-05T18:00:00Z", "open": 75.0, "high": 76.0, "low": 74.0, "close": 75.5, "volume": 1000},
            {"timestamp": "2026-01-06T18:00:00Z", "open": -10.0, "high": 76.0, "low": 74.0, "close": 75.0, "volume": 1000},  # Negative open
            {"timestamp": "2026-01-07T18:00:00Z", "open": 75.0, "high": 70.0, "low": 74.0, "close": 72.0, "volume": 1000},   # High < Low
            {"timestamp": "2026-01-08T18:00:00Z", "open": 75.0, "high": 76.0, "low": 74.0, "close": 0.0, "volume": 1000},    # Zero close
        ]
        validated = loader.validate_ohlcv(corrupted, commodity="crude oil")
        self.assertEqual(len(validated), 1)
        self.assertEqual(validated[0]["timestamp"], "2026-01-05T18:00:00Z")

    def test_validate_ohlcv_all_invalid_raises_error(self):
        loader = PriceDataLoader(config=self.config)
        all_invalid = [
            {"timestamp": "2026-01-05T18:00:00Z", "open": -50.0, "high": 20.0, "low": 30.0, "close": -10.0, "volume": 1000},
        ]
        with self.assertRaises(DataValidationError):
            loader.validate_ohlcv(all_invalid, commodity="copper")

    def test_duplicate_timestamp_removal(self):
        loader = PriceDataLoader(config=self.config)
        duplicates = [
            {"timestamp": "2026-01-05T18:00:00Z", "open": 75.0, "high": 76.0, "low": 74.0, "close": 75.0, "volume": 1000},
            {"timestamp": "2026-01-05T18:00:00Z", "open": 75.0, "high": 77.0, "low": 74.0, "close": 76.5, "volume": 2000},  # Duplicate
            {"timestamp": "2026-01-06T18:00:00Z", "open": 76.5, "high": 78.0, "low": 76.0, "close": 77.0, "volume": 1500},
        ]
        validated = loader.validate_ohlcv(duplicates, commodity="gold")
        self.assertEqual(len(validated), 2)
        # Unique timestamp preserved
        ts_list = [r["timestamp"] for r in validated]
        self.assertEqual(len(ts_list), len(set(ts_list)))

    def test_weekend_and_holiday_gaps_preserved_without_ffill(self):
        """Verify that weekend gaps are strictly preserved without synthetic zero-return observations."""
        loader = PriceDataLoader(config=self.config)
        records = [
            {"timestamp": "2026-01-09T18:00:00Z", "open": 75.0, "high": 76.0, "low": 74.0, "close": 75.5, "volume": 1000},  # Friday
            {"timestamp": "2026-01-12T18:00:00Z", "open": 76.0, "high": 77.0, "low": 75.0, "close": 76.8, "volume": 1200},  # Monday
        ]
        validated = loader.validate_ohlcv(records, commodity="crude oil")
        self.assertEqual(len(validated), 2)
        timestamps = [r["timestamp"] for r in validated]
        self.assertNotIn("2026-01-10T18:00:00Z", timestamps)
        self.assertNotIn("2026-01-11T18:00:00Z", timestamps)

    def test_fetch_creates_csv_cache(self):
        mock_provider = MagicMock(spec=BasePriceProvider)
        mock_provider.fetch_daily_ohlcv.return_value = self.mock_records

        loader = PriceDataLoader(config=self.config, provider=mock_provider)
        cache_file = loader.get_cache_path("copper")

        self.assertFalse(cache_file.exists())
        data = loader.load_commodity_data("copper")

        self.assertTrue(cache_file.exists())
        self.assertEqual(len(data), 10)
        mock_provider.fetch_daily_ohlcv.assert_called_once()

    def test_cache_freshness_reuse_without_provider_call(self):
        mock_provider = MagicMock(spec=BasePriceProvider)
        mock_provider.fetch_daily_ohlcv.return_value = self.mock_records

        loader = PriceDataLoader(config=self.config, provider=mock_provider)
        loader.load_commodity_data("gold")
        self.assertEqual(mock_provider.fetch_daily_ohlcv.call_count, 1)

        # Fresh load reads from cache
        cached = loader.load_commodity_data("gold")
        self.assertEqual(len(cached), 10)
        self.assertEqual(mock_provider.fetch_daily_ohlcv.call_count, 1)  # Provider not called again

    def test_offline_mode_reads_cache_without_provider(self):
        mock_provider = MagicMock(spec=BasePriceProvider)
        mock_provider.fetch_daily_ohlcv.return_value = self.mock_records

        loader = PriceDataLoader(config=self.config, provider=mock_provider)
        loader.load_commodity_data("crude oil")
        mock_provider.reset_mock()

        offline_data = loader.load_commodity_data("crude oil", offline_mode=True)
        self.assertEqual(len(offline_data), 10)
        mock_provider.fetch_daily_ohlcv.assert_not_called()

    def test_offline_mode_missing_cache_raises_error(self):
        mock_provider = MagicMock(spec=BasePriceProvider)
        loader = PriceDataLoader(config=self.config, provider=mock_provider)

        with self.assertRaises(DataUnavailableError):
            loader.load_commodity_data("copper", offline_mode=True)
        mock_provider.fetch_daily_ohlcv.assert_not_called()

    def test_network_failure_falls_back_to_cache(self):
        mock_provider = MagicMock(spec=BasePriceProvider)
        mock_provider.fetch_daily_ohlcv.return_value = self.mock_records

        loader = PriceDataLoader(config=self.config, provider=mock_provider)
        loader.load_commodity_data("gold")

        # Provider fails on refresh
        mock_provider.fetch_daily_ohlcv.side_effect = ProviderError("Yahoo Connection Timed Out")
        fallback_data = loader.load_commodity_data("gold", refresh=True)
        self.assertEqual(len(fallback_data), 10)

    def test_empty_provider_response_raises_error(self):
        mock_provider = MagicMock(spec=BasePriceProvider)
        mock_provider.fetch_daily_ohlcv.return_value = []

        loader = PriceDataLoader(config=self.config, provider=mock_provider)
        with self.assertRaises(DataUnavailableError):
            loader.load_commodity_data("crude oil")

    def test_invalid_commodity_raises_value_error(self):
        loader = PriceDataLoader(config=self.config)
        with self.assertRaises(ValueError):
            loader.load_commodity_data("lithium")  # Unsupported commodity

    def test_multiple_commodity_symbols(self):
        loader = PriceDataLoader(config=self.config)
        expected = {
            "crude oil": "BZ=F",
            "copper": "HG=F",
            "gold": "GC=F",
        }
        for cmd, sym in expected.items():
            self.assertEqual(self.config.commodity_symbol_map[cmd]["symbol"], sym)
            self.assertEqual(loader.get_cache_path(cmd).name, f"{cmd.replace(' ', '_')}.csv")


if __name__ == "__main__":
    unittest.main()
