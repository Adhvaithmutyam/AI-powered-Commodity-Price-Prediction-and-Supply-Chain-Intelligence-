"""
Configuration module for the Commodity Price Prediction & Forecasting pipeline.
Defines prediction horizons, target thresholds, model parameters, and path defaults.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class PredictionConfig:
    """Configuration settings for quantitative features, models, data loader, and backtesting."""

    # Target commodities
    commodities: List[str] = field(
        default_factory=lambda: ["crude oil", "copper", "gold"]
    )

    # Commodity instrument & provider symbol mapping (Yahoo Finance Futures)
    # Preserves benchmark instrument identity:
    # - Crude Oil: Brent Crude Futures (BZ=F)
    # - Copper: COMEX Copper Futures (HG=F)
    # - Gold: COMEX Gold Futures (GC=F)
    commodity_symbol_map: Dict[str, Dict[str, str]] = field(
        default_factory=lambda: {
            "crude oil": {
                "symbol": "BZ=F",
                "name": "Brent Crude Oil Futures",
                "benchmark": "Brent Crude Spot",
                "exchange": "ICE Europe",
            },
            "copper": {
                "symbol": "HG=F",
                "name": "Copper Futures (COMEX)",
                "benchmark": "COMEX Copper Grade A",
                "exchange": "COMEX",
            },
            "gold": {
                "symbol": "GC=F",
                "name": "Gold Futures (COMEX)",
                "benchmark": "London Gold Fixing",
                "exchange": "COMEX",
            },
        }
    )

    # Data loading and caching configuration
    use_synthetic_data: bool = False  # Explicit fallback only; False by default
    offline_mode: bool = False        # When True, reads strictly from local CSV cache
    cache_expiry_hours: int = 24      # Cache validity window before re-fetching
    lookback_years: int = 3           # Historical daily lookback window (approx 3 years)

    # Forecast horizons in trading days: T+1 (next day) and T+5 (next week)
    horizons: List[int] = field(default_factory=lambda: [1, 5])

    # 3-Class directional thresholds: Increase (> +T), Stable ([-T, +T]), Decrease (< -T)
    # T+1: ±0.5% (0.005)
    # T+5: ±1.0% (0.010)
    horizon_thresholds: Dict[int, float] = field(
        default_factory=lambda: {
            1: 0.005,
            5: 0.010,
        }
    )

    # Model training hyper-parameters (LightGBM multi-class GBDT)
    lgbm_params: Dict[str, Any] = field(
        default_factory=lambda: {
            "objective": "multiclass",
            "num_class": 3,
            "metric": "multi_logloss",
            "boosting_type": "gbdt",
            "n_estimators": 150,
            "learning_rate": 0.03,
            "num_leaves": 15,
            "max_depth": 4,
            "feature_fraction": 0.8,
            "class_weight": "balanced",
            "random_state": 42,
            "verbose": -1,
        }
    )

    # Walk-forward cross-validation settings (trading days)
    train_window_size: int = 252  # ~1 trading year
    test_window_size: int = 42    # ~2 trading months

    # File and directory paths
    price_data_dir: Path = field(default_factory=lambda: Path("data/raw/prices"))
    sentiment_data_path: Path = field(
        default_factory=lambda: Path("data/processed/commodity_news_sentiment.jsonl")
    )
    events_data_path: Path = field(
        default_factory=lambda: Path("data/processed/commodity_events.jsonl")
    )
    models_dir: Path = field(default_factory=lambda: Path("models"))
    output_predictions_path: Path = field(
        default_factory=lambda: Path("data/processed/commodity_predictions.jsonl")
    )
    output_evaluation_path: Path = field(
        default_factory=lambda: Path("data/processed/evaluation_metrics.json")
    )
