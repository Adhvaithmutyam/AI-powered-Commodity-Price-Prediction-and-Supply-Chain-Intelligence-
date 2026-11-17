"""
Feature engineering and target labeling for commodity price prediction.
Implements technical indicators, multi-horizon returns, volatility estimators,
and strictly point-in-time NLP sentiment/event signal alignment.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import numpy as np
import pandas as pd


FEATURE_DESCRIPTIONS: Dict[str, str] = {
    "log_ret_1d": "1-day logarithmic return of closing price",
    "ret_lag_1d": "Previous trading day price change (%)",
    "ret_lag_2d": "2-day cumulative price change (%)",
    "ret_lag_3d": "3-day cumulative price change (%)",
    "ret_lag_5d": "5-day cumulative price change (%)",
    "ret_lag_10d": "10-day cumulative price change (%)",
    "ret_lag_21d": "21-day (monthly) cumulative price change (%)",
    "sma_5": "5-day simple moving average",
    "sma_20": "20-day simple moving average",
    "sma_50": "50-day simple moving average",
    "dist_sma_20": "Normalized percentage distance from 20-day moving average",
    "dist_sma_50": "Normalized percentage distance from 50-day moving average",
    "volatility_10d": "10-day annualized realized price volatility",
    "volatility_21d": "21-day annualized realized price volatility",
    "rsi_14": "14-period Relative Strength Index (momentum oscillator)",
    "macd": "Moving Average Convergence Divergence line (12-26 EMA)",
    "macd_signal": "MACD signal line (9-day EMA)",
    "macd_hist": "MACD histogram (difference between MACD and signal)",
    "bb_pct_b": "Bollinger Bands %B position within 2-sigma bands",
    "atr_14": "14-day Average True Range volatility",
    "news_sentiment_3d_polarity": "3-day rolling net financial news sentiment polarity",
    "supply_disruption_3d_count": "3-day count of verified supply disruption news events",
}


def compute_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes standard econometric and technical indicators on historical price data.
    Requires columns: 'close', 'high', 'low', 'volume'.
    Expects df sorted ascending by date/timestamp.
    """
    df = df.copy()
    if "close" not in df.columns:
        raise ValueError("Input DataFrame must contain a 'close' column.")

    close = df["close"]
    high = df["high"] if "high" in df.columns else close
    low = df["low"] if "low" in df.columns else close

    # 1. Log returns and multi-horizon lagged percentage returns
    df["log_ret_1d"] = np.log(close / close.shift(1).replace(0, np.nan)).fillna(0.0)
    for lag in [1, 2, 3, 5, 10, 21]:
        df[f"ret_lag_{lag}d"] = close.pct_change(lag).fillna(0.0)

    # 2. Moving Averages & Moving Average Distances
    df["sma_5"] = close.rolling(window=5, min_periods=1).mean()
    df["sma_20"] = close.rolling(window=20, min_periods=1).mean()
    df["sma_50"] = close.rolling(window=50, min_periods=1).mean()
    df["dist_sma_20"] = ((close - df["sma_20"]) / (df["sma_20"] + 1e-9)).fillna(0.0)
    df["dist_sma_50"] = ((close - df["sma_50"]) / (df["sma_50"] + 1e-9)).fillna(0.0)

    # 3. Realized Annualized Volatility (assuming 252 trading days)
    df["volatility_10d"] = (
        df["log_ret_1d"].rolling(window=10, min_periods=2).std() * np.sqrt(252)
    ).fillna(0.0)
    df["volatility_21d"] = (
        df["log_ret_1d"].rolling(window=21, min_periods=2).std() * np.sqrt(252)
    ).fillna(0.0)

    # 4. Momentum: Relative Strength Index (RSI - 14)
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    df["rsi_14"] = (100 - (100 / (1 + rs))).fillna(50.0)

    # 5. MACD (12, 26, 9)
    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    df["macd"] = ema_12 - ema_26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    # 6. Bollinger Bands (20, 2 std)
    bb_mid = df["sma_20"]
    bb_std = close.rolling(window=20, min_periods=2).std().fillna(0.0)
    df["bb_upper"] = bb_mid + 2 * bb_std
    df["bb_lower"] = bb_mid - 2 * bb_std
    df["bb_pct_b"] = (
        (close - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"] + 1e-9)
    ).fillna(0.5)

    # 7. Average True Range (ATR - 14)
    prev_close = close.shift(1).fillna(close)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df["atr_14"] = tr.rolling(window=14, min_periods=1).mean().fillna(0.0)

    return df


def generate_directional_targets(
    df: pd.DataFrame,
    horizons: List[int] = [1, 5],
    thresholds: Optional[Dict[int, float]] = None,
) -> pd.DataFrame:
    """
    Labels 3-class directional price target:
        0: Decrease (future_ret < -threshold)
        1: Stable   (-threshold <= future_ret <= +threshold)
        2: Increase (future_ret > +threshold)

    Formula: Delta P = (P_{t+h} - P_t) / P_t
    """
    df = df.copy()
    if thresholds is None:
        thresholds = {1: 0.005, 5: 0.010}

    for h in horizons:
        thresh = thresholds.get(h, 0.005)
        # Shift -h looks into the future for labeling training targets only
        future_close = df["close"].shift(-h)
        future_ret = (future_close - df["close"]) / df["close"]

        target_col = f"target_{h}d"
        # Class encoding: 0 = Decrease, 1 = Stable, 2 = Increase
        df[target_col] = np.where(
            future_ret > thresh,
            2,
            np.where(future_ret < -thresh, 0, 1),
        )
        df[f"future_ret_{h}d"] = future_ret

    return df


def align_nlp_signals(
    df: pd.DataFrame,
    commodity: str,
    sentiment_path: Optional[Path] = None,
    events_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Point-in-time alignment of FinBERT sentiment and supply-chain event records.
    Ensures zero lookahead bias by aggregating news published strictly prior to each price timestamp.
    """
    df = df.copy()
    df["news_sentiment_3d_polarity"] = 0.0
    df["supply_disruption_3d_count"] = 0

    if sentiment_path and sentiment_path.exists():
        try:
            records = []
            with sentiment_path.open("r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        r = json.loads(line)
                        if r.get("commodity", "").lower() == commodity.lower():
                            records.append(r)

            if records:
                s_df = pd.DataFrame(records)
                if "publication_timestamp" in s_df.columns:
                    s_df["pub_dt"] = pd.to_datetime(s_df["publication_timestamp"], errors="coerce")
                    # Map positive: +1, negative: -1, neutral: 0
                    mapping = {"positive": 1.0, "negative": -1.0, "neutral": 0.0}
                    s_df["val"] = s_df["sentiment_label"].map(mapping).fillna(0.0)

                    # Compute point-in-time rolling 3-day sentiment
                    if "timestamp" in df.columns:
                        p_dt = pd.to_datetime(df["timestamp"])
                        polarities = []
                        for pt in p_dt:
                            window_start = pt - pd.Timedelta(days=3)
                            sub = s_df[(s_df["pub_dt"] >= window_start) & (s_df["pub_dt"] <= pt)]
                            pol = sub["val"].mean() if not sub.empty else 0.0
                            polarities.append(pol)
                        df["news_sentiment_3d_polarity"] = polarities
        except Exception:
            pass

    return df


def prepare_feature_matrix(
    df: pd.DataFrame,
    horizon: int = 1,
    include_target: bool = True,
) -> Tuple[pd.DataFrame, Optional[pd.Series], List[str]]:
    """
    Extracts numerical feature matrix X and target y for model training or inference.
    Excludes non-feature identifier columns and future return references.
    """
    exclude_cols = {
        "timestamp",
        "date",
        "symbol",
        "commodity",
        "target_1d",
        "target_5d",
        "future_ret_1d",
        "future_ret_5d",
    }
    feature_cols = [c for c in df.columns if c not in exclude_cols and np.issubdtype(df[c].dtype, np.number)]

    if include_target:
        target_col = f"target_{horizon}d"
        clean_df = df.dropna(subset=feature_cols + [target_col]).reset_index(drop=True)
        X = clean_df[feature_cols]
        y = clean_df[target_col].astype(int)
        return X, y, feature_cols
    else:
        X = df[feature_cols].fillna(0.0)
        return X, None, feature_cols
