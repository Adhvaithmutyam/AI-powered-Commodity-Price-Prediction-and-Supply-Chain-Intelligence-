"""
Real Historical Commodity Price Data Loader (MLOps 2).
Provides provider abstractions (Yahoo Finance via yfinance and direct v8 REST endpoint),
local CSV caching, OHLCV validation, and session-date normalization for
Brent Crude (BZ=F), COMEX Copper (HG=F), and COMEX Gold (GC=F) futures.
"""

from abc import ABC, abstractmethod
from datetime import datetime, timedelta, timezone
import json
import logging
import os
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import requests

from .config import PredictionConfig

logger = logging.getLogger(__name__)


class DataValidationError(Exception):
    """Raised when incoming OHLCV price series violates market integrity constraints."""
    pass


class ProviderError(Exception):
    """Raised when external data provider encounters network, rate-limiting, or symbol errors."""
    pass


class DataUnavailableError(Exception):
    """Raised when neither fresh/cached data nor online data can be loaded."""
    pass


class BasePriceProvider(ABC):
    """Abstract base class defining the commodity price provider interface."""

    @abstractmethod
    def fetch_daily_ohlcv(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Any:
        """
        Fetches historical daily OHLCV bars for a financial symbol.
        Returns a pandas DataFrame or standard list of dict records.
        """
        pass


class YahooFinanceProvider(BasePriceProvider):
    """
    Yahoo Finance provider supporting both yfinance library and direct v8 chart REST endpoint.
    Implements retry with bounded exponential backoff and MultiIndex column flattening.
    """

    def __init__(self, max_retries: int = 3, timeout_seconds: int = 15):
        self.max_retries = max_retries
        self.timeout_seconds = timeout_seconds
        self.headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

    def _fetch_via_rest(self, symbol: str, range_str: str = "3y") -> List[Dict[str, Any]]:
        """Direct REST fallback querying Yahoo Finance v8 chart API."""
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?range={range_str}&interval=1d"
        resp = requests.get(url, headers=self.headers, timeout=self.timeout_seconds)
        resp.raise_for_status()
        data = resp.json()

        chart = data.get("chart", {})
        results = chart.get("result")
        if not results:
            error = chart.get("error", {})
            raise ProviderError(f"Yahoo Finance returned error for {symbol}: {error}")

        res = results[0]
        timestamps = res.get("timestamp", [])
        quote = res.get("indicators", {}).get("quote", [{}])[0]

        opens = quote.get("open", [])
        highs = quote.get("high", [])
        lows = quote.get("low", [])
        closes = quote.get("close", [])
        volumes = quote.get("volume", [])

        records = []
        for i, ts in enumerate(timestamps):
            c = closes[i] if i < len(closes) else None
            # Skip holiday/empty bars where settlement is null
            if c is None:
                continue
            o = opens[i] if (i < len(opens) and opens[i] is not None) else c
            h = highs[i] if (i < len(highs) and highs[i] is not None) else max(o, c)
            l = lows[i] if (i < len(lows) and lows[i] is not None) else min(o, c)
            v = volumes[i] if (i < len(volumes) and volumes[i] is not None) else 0.0

            dt = datetime.fromtimestamp(ts, tz=timezone.utc)
            date_str = dt.strftime("%Y-%m-%dT18:00:00Z")

            records.append({
                "timestamp": date_str,
                "open": round(float(o), 4),
                "high": round(float(h), 4),
                "low": round(float(l), 4),
                "close": round(float(c), 4),
                "volume": float(v),
            })

        return records

    def fetch_daily_ohlcv(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Any:
        """
        Downloads historical daily OHLCV data using yfinance (if available) or direct REST fallback.
        """
        attempt = 0
        last_error = None

        while attempt < self.max_retries:
            try:
                # 1. Try yfinance first if installed and functional
                try:
                    import yfinance as yf
                    import pandas as pd
                    ticker = yf.Ticker(symbol)
                    if start_date and end_date:
                        df = ticker.history(start=start_date, end=end_date, auto_adjust=False, timeout=self.timeout_seconds)
                    else:
                        df = ticker.history(period="3y", auto_adjust=False, timeout=self.timeout_seconds)

                    if df is not None and not df.empty:
                        if isinstance(df.columns, pd.MultiIndex):
                            df.columns = [col[0] for col in df.columns]
                        return df.reset_index()
                except Exception as yf_err:
                    logger.debug("yfinance failed (%s), falling back to direct REST endpoint...", yf_err)

                # 2. Direct REST endpoint fallback
                records = self._fetch_via_rest(symbol, range_str="3y")
                if records:
                    try:
                        import pandas as pd
                        return pd.DataFrame(records)
                    except Exception:
                        return records
                raise ProviderError(f"No price records returned for {symbol}")

            except Exception as e:
                attempt += 1
                last_error = e
                backoff_wait = 2 ** (attempt - 1)
                logger.warning(
                    "Error fetching %s (attempt %d/%d): %s. Retrying in %ds...",
                    symbol, attempt, self.max_retries, str(e), backoff_wait
                )
                time.sleep(backoff_wait)

        raise ProviderError(f"Failed to fetch data for symbol '{symbol}' after {self.max_retries} attempts: {last_error}")


class PriceDataLoader:
    """
    Manages loading, validation, and local CSV caching of real commodity price series.
    Provides offline-first resilience and guarantees no forward-filling across non-trading days.
    """

    def __init__(
        self,
        config: Optional[PredictionConfig] = None,
        provider: Optional[BasePriceProvider] = None,
    ):
        self.config = config or PredictionConfig()
        self.provider = provider or YahooFinanceProvider()
        self.cache_dir = Path(self.config.price_data_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_cache_path(self, commodity: str) -> Path:
        """Returns standard local CSV cache path for a commodity (e.g. data/raw/prices/crude_oil.csv)."""
        safe_name = commodity.lower().replace(" ", "_")
        return self.cache_dir / f"{safe_name}.csv"

    def is_cache_fresh(self, commodity: str) -> bool:
        """Checks if local CSV cache exists and was modified within cache_expiry_hours."""
        cache_path = self.get_cache_path(commodity)
        if not cache_path.exists():
            return False
        file_age_seconds = time.time() - cache_path.stat().st_mtime
        return file_age_seconds < (self.config.cache_expiry_hours * 3600)

    def save_to_cache(self, commodity: str, df: Any) -> Path:
        """Saves validated OHLCV DataFrame or records to local CSV cache."""
        cache_path = self.get_cache_path(commodity)
        if hasattr(df, "to_csv"):
            df.to_csv(cache_path, index=False)
        else:
            # Native CSV writing without pandas dependency
            import csv
            records = df if isinstance(df, list) else []
            if records:
                fieldnames = ["timestamp", "commodity", "open", "high", "low", "close", "volume"]
                with open(cache_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    for r in records:
                        writer.writerow({k: r.get(k, "") for k in fieldnames})

        logger.info("Saved cache for %s to %s", commodity, cache_path)
        return cache_path

    def load_from_cache(self, commodity: str) -> Optional[Any]:
        """Loads and normalizes commodity OHLCV records from local CSV cache if present."""
        cache_path = self.get_cache_path(commodity)
        if not cache_path.exists():
            return None
        try:
            try:
                import pandas as pd
                df = pd.read_csv(cache_path)
                return self.validate_ohlcv(df, commodity=commodity)
            except Exception:
                # Lightweight CSV reader fallback
                import csv
                records = []
                with open(cache_path, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        records.append({
                            "timestamp": row["timestamp"],
                            "commodity": row["commodity"],
                            "open": float(row["open"]),
                            "high": float(row["high"]),
                            "low": float(row["low"]),
                            "close": float(row["close"]),
                            "volume": float(row["volume"]),
                        })
                return self.validate_ohlcv(records, commodity=commodity)
        except Exception as e:
            logger.warning("Failed to load cache from %s: %s", cache_path, str(e))
            return None

    def normalize_session_timestamps(self, series: Any) -> Any:
        """
        Normalizes session dates to UTC ISO 8601 strings (%Y-%m-%dT18:00:00Z).
        Exchange daily settlement prices represent the end of the trading session (18:00:00 UTC),
        strictly preventing lookahead leakage from news published after market close.
        """
        if hasattr(series, "dt"):
            return pd.to_datetime(series, errors="coerce").dt.strftime("%Y-%m-%dT18:00:00Z")
        elif isinstance(series, list):
            out = []
            for item in series:
                if isinstance(item, str) and item.endswith("T18:00:00Z"):
                    out.append(item)
                else:
                    try:
                        d = datetime.fromisoformat(str(item).replace("Z", "+00:00"))
                        out.append(d.strftime("%Y-%m-%dT18:00:00Z"))
                    except Exception:
                        out.append(str(item)[:10] + "T18:00:00Z")
            return out
        return series

    def validate_ohlcv(self, df: Any, commodity: str) -> Any:
        """
        Strict data validation and normalization:
        1. Ensures standard columns: timestamp, commodity, open, high, low, close, volume.
        2. Validates positive prices and non-negative volume.
        3. Enforces High >= Low, High >= Open/Close, Low <= Open/Close.
        4. Drops duplicates and sorts chronologically.
        5. NEVER forward-fills across missing trading sessions.
        """
        # If input is a list of dicts, convert or validate directly
        if isinstance(df, list):
            try:
                import pandas as pd
                df = pd.DataFrame(df)
            except Exception:
                # Dict list validation
                valid = []
                seen_ts = set()
                for r in df:
                    ts = r.get("timestamp") or r.get("Date")
                    if not ts:
                        continue
                    ts_str = str(ts)[:10] + "T18:00:00Z" if not str(ts).endswith("T18:00:00Z") else str(ts)
                    if ts_str in seen_ts:
                        continue
                    seen_ts.add(ts_str)

                    c = float(r.get("close") or r.get("Close") or 0.0)
                    o = float(r.get("open") or r.get("Open") or c)
                    h = float(r.get("high") or r.get("High") or max(o, c))
                    l = float(r.get("low") or r.get("Low") or min(o, c))
                    v = float(r.get("volume") or r.get("Volume") or 0.0)

                    if c > 0 and o > 0 and h > 0 and l > 0 and h >= l - 1e-4 and v >= 0:
                        valid.append({
                            "timestamp": ts_str,
                            "commodity": commodity.lower(),
                            "open": round(o, 4),
                            "high": round(h, 4),
                            "low": round(l, 4),
                            "close": round(c, 4),
                            "volume": round(v, 2),
                        })
                valid.sort(key=lambda x: x["timestamp"])
                if not valid:
                    raise DataValidationError(f"All price records for '{commodity}' failed validation checks.")
                return valid

        # Pandas DataFrame validation
        df = df.copy()
        rename_map = {
            "Date": "timestamp",
            "Datetime": "timestamp",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
            "Adj Close": "adj_close",
        }
        df = df.rename(columns=rename_map)

        if "timestamp" not in df.columns or "close" not in df.columns:
            raise DataValidationError("DataFrame missing required 'timestamp' or 'close' column.")

        for col in ["open", "high", "low", "close", "volume"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            else:
                df[col] = df["close"] if col in ["open", "high", "low"] else 0.0

        df["timestamp"] = self.normalize_session_timestamps(df["timestamp"])
        df["commodity"] = commodity.lower()

        initial_count = len(df)
        df = df.dropna(subset=["timestamp", "close"])

        # Constraints
        valid_mask = (
            (df["close"] > 0) & (df["open"] > 0) & (df["high"] > 0) & (df["low"] > 0) &
            (df["high"] >= df["low"] - 1e-4) &
            (df["high"] >= df["open"] - 1e-4) &
            (df["high"] >= df["close"] - 1e-4) &
            (df["low"] <= df["open"] + 1e-4) &
            (df["low"] <= df["close"] + 1e-4)
        )
        df = df[valid_mask]
        df["volume"] = df["volume"].clip(lower=0.0).fillna(0.0)

        # Deduplication and chronological sorting
        df = df.drop_duplicates(subset=["timestamp"], keep="last")
        df = df.sort_values("timestamp", ascending=True).reset_index(drop=True)

        if len(df) < initial_count:
            logger.info("Validation filtered %d invalid/duplicate records for %s", initial_count - len(df), commodity)

        if df.empty:
            raise DataValidationError(f"All price records for '{commodity}' failed validation checks.")

        standard_cols = ["timestamp", "commodity", "open", "high", "low", "close", "volume"]
        return df[standard_cols]

    def load_commodity_data(
        self,
        commodity: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        offline_mode: Optional[bool] = None,
        refresh: bool = False,
    ) -> Any:
        """
        Primary interface: Loads 3 years of validated daily OHLCV price series.
        Prioritizes fresh cache -> Online fetch -> Stale cache fallback.
        Raises DataUnavailableError if real data cannot be retrieved.
        """
        commodity_key = commodity.lower().strip()
        if commodity_key not in self.config.commodity_symbol_map:
            raise ValueError(
                f"Unknown commodity '{commodity}'. Supported commodities: {list(self.config.commodity_symbol_map.keys())}"
            )

        is_offline = offline_mode if offline_mode is not None else self.config.offline_mode

        # 1. Offline Mode: Strictly use local cache
        if is_offline:
            cached_df = self.load_from_cache(commodity_key)
            if cached_df is not None and (len(cached_df) > 0 if isinstance(cached_df, list) else not cached_df.empty):
                logger.info("Loaded %d cached records for %s (offline mode)", len(cached_df), commodity_key)
                return cached_df
            raise DataUnavailableError(
                f"Offline mode is active, but no local cache file exists at {self.get_cache_path(commodity_key)}. "
                f"Run without --offline to download historical data."
            )

        # 2. Cache Hit: Check if local cache is fresh and refresh not requested
        if not refresh and self.is_cache_fresh(commodity_key):
            cached_df = self.load_from_cache(commodity_key)
            if cached_df is not None and (len(cached_df) > 0 if isinstance(cached_df, list) else not cached_df.empty):
                logger.info("Loaded %d fresh cached records for %s", len(cached_df), commodity_key)
                return cached_df

        # 3. Online Fetch: Request from provider
        symbol_info = self.config.commodity_symbol_map[commodity_key]
        symbol = symbol_info["symbol"]

        try:
            if not start_date:
                calc_start = datetime.now(timezone.utc) - timedelta(days=self.config.lookback_years * 365 + 30)
                start_date = calc_start.strftime("%Y-%m-%d")

            logger.info("Fetching real historical OHLCV for %s (%s) from provider...", commodity_key, symbol)
            raw_data = self.provider.fetch_daily_ohlcv(symbol, start_date=start_date, end_date=end_date)
            validated_data = self.validate_ohlcv(raw_data, commodity=commodity_key)
            self.save_to_cache(commodity_key, validated_data)
            return validated_data

        except Exception as e:
            logger.warning("Online fetch failed for %s (%s): %s", commodity_key, symbol, str(e))
            # Fall back to stale cache if present
            cached_df = self.load_from_cache(commodity_key)
            if cached_df is not None and (len(cached_df) > 0 if isinstance(cached_df, list) else not cached_df.empty):
                logger.info("Falling back to existing local cache for %s (%d records)", commodity_key, len(cached_df))
                return cached_df

            raise DataUnavailableError(
                f"Could not load real market data for '{commodity_key}'. "
                f"Online provider failed ({e}) and no local cache was available."
            )
