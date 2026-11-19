#!/usr/bin/env python
"""
Entrypoint script for running Commodity Price Prediction and Walk-Forward Backtesting (MLOps 2).
Supports real historical market data ingestion (Yahoo Finance) and explicit synthetic demo mode.
Usage:
    python run_prediction.py
    python run_prediction.py --commodity copper --horizon 1
    python run_prediction.py --all
    python run_prediction.py --offline
    python run_prediction.py --synthetic
    python run_prediction.py --refresh-data
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.price_prediction import PredictionConfig, PricePredictionPipeline

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(asctime)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run multi-commodity price prediction, backtesting, and explainability."
    )
    parser.add_argument(
        "--commodity",
        type=str,
        default="crude oil",
        choices=["crude oil", "copper", "gold"],
        help="Target commodity for individual prediction (default: 'crude oil').",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=1,
        choices=[1, 5],
        help="Prediction horizon in trading days: 1 (T+1) or 5 (T+5).",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Execute end-to-end training and inference for all commodities and horizons.",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Explicitly enable synthetic demo data (disables real market data ingestion).",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Run in offline mode using only existing local CSV cache files.",
    )
    parser.add_argument(
        "--refresh-data",
        action="store_true",
        help="Force a fresh download of market prices from Yahoo Finance, bypassing cache freshness.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = PredictionConfig(
        use_synthetic_data=args.synthetic,
        offline_mode=args.offline,
    )
    pipeline = PricePredictionPipeline(config=config)

    print("=" * 75)
    print(" AI-Powered Commodity Price Prediction & Quantitative Forecasting")
    print("=" * 75)
    mode_str = "SYNTHETIC DEMO" if args.synthetic else ("OFFLINE CACHE" if args.offline else "REAL MARKET DATA (Yahoo Finance)")
    print(f"Data Mode             : {mode_str}")
    print(f"Lookback Window       : ~{config.lookback_years} Years Daily History")

    if args.all:
        print("\n[INFO] Running full pipeline across all commodities and horizons...")
        results = pipeline.run_all(refresh_data=args.refresh_data)
        print("\n" + "=" * 75)
        print(" [SUCCESS] Completed training and prediction generation.")
        print(f"Processed Commodities : {results['commodities_processed']}")
        print(f"Predictions Written   : {results['output_predictions_path']}")
        print(f"Evaluation Metrics    : {results['output_evaluation_path']}")
        print("=" * 75)
    else:
        print(f"\n[INFO] Training model and generating forecast for {args.commodity.upper()} (T+{args.horizon}d)...")
        price_df = None
        if not args.synthetic:
            price_df = pipeline.data_loader.load_commodity_data(
                args.commodity, offline_mode=args.offline, refresh=args.refresh_data
            )

        eval_metrics = pipeline.train_and_evaluate(args.commodity, price_df=price_df)
        pred = pipeline.predict_latest(commodity=args.commodity, horizon=args.horizon, latest_price_df=price_df)

        print("\n" + "-" * 75)
        print(" LATEST PREDICTION SUMMARY (API Schema Conforming)")
        print("-" * 75)
        print(json.dumps(pred, indent=2))

        print("\n" + "-" * 75)
        print(" WALK-FORWARD CROSS-VALIDATION SUMMARY")
        print("-" * 75)
        h_key = f"{args.horizon}d"
        bt = eval_metrics["horizons"][h_key]["backtest"]
        print(f"Commodity             : {eval_metrics['commodity'].upper()} ({eval_metrics['instrument_symbol']})")
        print(f"Data Source           : {eval_metrics['data_source']}")
        print(f"Date Range            : {eval_metrics['date_range']['start']} to {eval_metrics['date_range']['end']}")
        print(f"Observations          : {eval_metrics['observation_count']} daily sessions")
        print(f"Horizon               : T+{bt['horizon_days']} trading day(s)")
        print(f"Completed Folds       : {bt['n_folds']}")
        print(f"Mean Macro-F1         : {bt['mean_macro_f1']} (Std: {bt['std_macro_f1']})")
        print(f"Baseline Macro-F1     : {bt['baseline_mean_macro_f1']}")
        print(f"Outperforms Baseline  : {'YES' if bt['beats_baseline'] else 'NO'}")
        print("=" * 75)


if __name__ == "__main__":
    main()
