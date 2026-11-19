"""
End-to-end Pipeline orchestrator for Commodity Price Prediction (MLOps 2).
Integrates real historical price data loading, feature engineering, model training,
SHAP explainability, purged walk-forward cross-validation, and REST API payload formatting.
"""

from datetime import datetime, timezone
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .config import PredictionConfig
from .data_loader import PriceDataLoader, DataUnavailableError
from .features import (
    compute_technical_indicators,
    generate_directional_targets,
    align_nlp_signals,
    prepare_feature_matrix,
)
from .models import CommodityDirectionClassifier, CLASS_LABELS
from .explain import ShapExplainer
from .backtest import run_backtest

logger = logging.getLogger(__name__)


class PricePredictionPipeline:
    """
    Executes training, evaluation, and inference pipelines for multi-commodity forecasting.
    Prioritizes real historical commodity prices, retaining synthetic data only as an explicit demo fallback.
    """

    def __init__(
        self,
        config: Optional[PredictionConfig] = None,
        data_loader: Optional[PriceDataLoader] = None,
    ):
        self.config = config or PredictionConfig()
        self.data_loader = data_loader or PriceDataLoader(config=self.config)
        self.models: Dict[str, Dict[int, CommodityDirectionClassifier]] = {}
        self.explainers: Dict[str, Dict[int, ShapExplainer]] = {}

    def generate_synthetic_historical_data(
        self, commodity: str, days: int = 400
    ) -> pd.DataFrame:
        """
        Generates realistic commodity time-series for local demo/offline testing
        when explicit synthetic mode is enabled (--synthetic).
        """
        np.random.seed(42 if commodity == "copper" else 100 if commodity == "gold" else 200)
        base_prices = {"crude oil": 75.0, "copper": 4.10, "gold": 2150.0}
        base_price = base_prices.get(commodity.lower(), 100.0)

        dates = pd.date_range(end=datetime.now(timezone.utc), periods=days, freq="B")
        returns = np.random.normal(loc=0.0003, scale=0.015, size=days)
        price_series = base_price * np.exp(np.cumsum(returns))

        high = price_series * (1.0 + np.abs(np.random.normal(0, 0.005, days)))
        low = price_series * (1.0 - np.abs(np.random.normal(0, 0.005, days)))
        open_p = (high + low) / 2.0
        volume = np.random.randint(10000, 150000, size=days)

        df = pd.DataFrame({
            "timestamp": dates.strftime("%Y-%m-%dT18:00:00Z"),
            "commodity": commodity.lower(),
            "open": np.round(open_p, 2),
            "high": np.round(high, 2),
            "low": np.round(low, 2),
            "close": np.round(price_series, 2),
            "volume": volume.astype(float),
        })
        return df

    def train_and_evaluate(
        self,
        commodity: str,
        price_df: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        """
        Trains models for T+1 and T+5 horizons, performs backtesting, and fits explainers.
        Loads real historical data by default; synthetic generation only occurs if explicitly configured.
        """
        commodity_key = commodity.lower().strip()

        # Ingestion logic: Real price data by default; synthetic only when explicitly enabled
        if price_df is None or price_df.empty:
            if self.config.use_synthetic_data:
                logger.info("Using synthetic historical data (explicitly requested) for %s", commodity_key)
                price_df = self.generate_synthetic_historical_data(commodity_key)
                data_source_name = "Synthetic Geometric Brownian Motion Generator"
            else:
                price_df = self.data_loader.load_commodity_data(
                    commodity_key, offline_mode=self.config.offline_mode
                )
                symbol_meta = self.config.commodity_symbol_map.get(commodity_key, {})
                data_source_name = f"Yahoo Finance ({symbol_meta.get('symbol', 'Unknown')})"
        else:
            data_source_name = "User-Supplied DataFrame"

        # 1. Feature Engineering
        feat_df = compute_technical_indicators(price_df)
        feat_df = align_nlp_signals(
            feat_df,
            commodity=commodity_key,
            sentiment_path=self.config.sentiment_data_path,
            events_path=self.config.events_data_path,
        )
        feat_df = generate_directional_targets(
            feat_df,
            horizons=self.config.horizons,
            thresholds=self.config.horizon_thresholds,
        )

        # Dataset metadata
        start_date = str(price_df["timestamp"].iloc[0]) if "timestamp" in price_df else "Unknown"
        end_date = str(price_df["timestamp"].iloc[-1]) if "timestamp" in price_df else "Unknown"
        symbol_info = self.config.commodity_symbol_map.get(commodity_key, {})

        results: Dict[str, Any] = {
            "commodity": commodity_key,
            "instrument_symbol": symbol_info.get("symbol", "N/A"),
            "data_source": data_source_name,
            "date_range": {"start": start_date, "end": end_date},
            "observation_count": len(price_df),
            "horizons": {},
        }

        self.models[commodity_key] = {}
        self.explainers[commodity_key] = {}

        for h in self.config.horizons:
            # Backtesting
            bt_results = run_backtest(feat_df, horizon=h, config=self.config)

            # Final Full-Sample Model Fitting
            X, y, feature_names = prepare_feature_matrix(feat_df, horizon=h, include_target=True)
            clf = CommodityDirectionClassifier(horizon_days=h, params=self.config.lgbm_params)
            clf.fit(X, y)

            # Explainer Fitting (uses background of latest 100 observations)
            sample_size = min(100, len(X))
            explainer = ShapExplainer(clf, background_sample=X.tail(sample_size))

            self.models[commodity_key][h] = clf
            self.explainers[commodity_key][h] = explainer

            eval_metrics = clf.evaluate(X.tail(min(50, len(X))), y.tail(min(50, len(y))))
            results["horizons"][f"{h}d"] = {
                "backtest": bt_results,
                "recent_eval": eval_metrics,
            }

        return results

    def predict_latest(
        self,
        commodity: str,
        horizon: int = 1,
        latest_price_df: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        """
        Formats single latest prediction output conforming exactly to /api/v1/predictions/latest schema.
        """
        commodity_key = commodity.lower().strip()

        if latest_price_df is None or latest_price_df.empty:
            if self.config.use_synthetic_data:
                latest_price_df = self.generate_synthetic_historical_data(commodity_key, days=60)
            else:
                latest_price_df = self.data_loader.load_commodity_data(
                    commodity_key, offline_mode=self.config.offline_mode
                )

        if commodity_key not in self.models or horizon not in self.models[commodity_key]:
            self.train_and_evaluate(commodity_key, latest_price_df)

        clf = self.models[commodity_key][horizon]
        explainer = self.explainers[commodity_key][horizon]

        feat_df = compute_technical_indicators(latest_price_df)
        feat_df = align_nlp_signals(
            feat_df,
            commodity=commodity_key,
            sentiment_path=self.config.sentiment_data_path,
            events_path=self.config.events_data_path,
        )
        X, _, _ = prepare_feature_matrix(feat_df, horizon=horizon, include_target=False)

        latest_row = X.iloc[[-1]]
        probs = clf.predict_proba(latest_row)[0]
        pred_class_idx = int(np.argmax(probs))
        pred_direction = CLASS_LABELS.get(pred_class_idx, "Stable")

        top_features = explainer.get_top_features(
            single_row=latest_row,
            predicted_class_idx=pred_class_idx,
            top_k=5,
        )

        latest_close = float(latest_price_df["close"].iloc[-1])
        latest_ts = str(latest_price_df["timestamp"].iloc[-1]) if "timestamp" in latest_price_df else datetime.now(timezone.utc).isoformat()

        symbol_info = self.config.commodity_symbol_map.get(commodity_key, {})
        benchmark_name = symbol_info.get("benchmark", f"{commodity.title()} Benchmark")

        return {
            "commodity": commodity_key,
            "horizon": f"{horizon}d",
            "as_of": datetime.now(timezone.utc).isoformat(),
            "predicted_direction": pred_direction,
            "probabilities": {
                "decrease": round(float(probs[0]), 3),
                "stable": round(float(probs[1]), 3),
                "increase": round(float(probs[2]), 3),
            },
            "current_price": {
                "benchmark": benchmark_name,
                "value": latest_close,
                "currency": "USD",
                "timestamp": latest_ts,
            },
            "top_features": top_features,
            "model_version": "v1.0.0",
            "model_name": "lightgbm_direction_classifier",
        }

    def run_all(self, refresh_data: bool = False) -> Dict[str, Any]:
        """
        Executes full training and inference across all configured commodities and horizons,
        writing JSONL predictions and evaluation summaries to disk.
        """
        self.config.output_predictions_path.parent.mkdir(parents=True, exist_ok=True)
        predictions = []
        evaluations = {}

        for commodity in self.config.commodities:
            logger.info("--- Processing commodity: %s ---", commodity.upper())
            price_df = None
            if not self.config.use_synthetic_data:
                price_df = self.data_loader.load_commodity_data(commodity, refresh=refresh_data)

            eval_res = self.train_and_evaluate(commodity, price_df=price_df)
            evaluations[commodity] = eval_res

            for h in self.config.horizons:
                pred = self.predict_latest(commodity=commodity, horizon=h, latest_price_df=price_df)
                predictions.append(pred)

        # Write predictions JSONL
        with self.config.output_predictions_path.open("w", encoding="utf-8") as f:
            for p in predictions:
                f.write(json.dumps(p) + "\n")

        # Write evaluation summary JSON
        with self.config.output_evaluation_path.open("w", encoding="utf-8") as f:
            json.dump(evaluations, f, indent=2)

        return {
            "status": "success",
            "commodities_processed": self.config.commodities,
            "horizons_evaluated": self.config.horizons,
            "total_predictions_generated": len(predictions),
            "output_predictions_path": str(self.config.output_predictions_path),
            "output_evaluation_path": str(self.config.output_evaluation_path),
            "evaluation_metrics": evaluations,
        }
