#!/usr/bin/env python
"""
Entrypoint script for running Commodity Price Prediction and Walk-Forward Backtesting (MLOps 2).
Usage:
    python run_prediction.py
    python run_prediction.py --commodity copper --horizon 1
    python run_prediction.py --all
"""

import argparse
import json
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.price_prediction import PredictionConfig, PricePredictionPipeline


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
    return parser.parse_args()


def main():
    args = parse_args()
    config = PredictionConfig()
    pipeline = PricePredictionPipeline(config=config)

    print("=" * 70)
    print(" AI-Powered Commodity Price Prediction & Forecasting Pipeline")
    print("=" * 70)

    if args.all:
        print("\n[INFO] Running full pipeline across all commodities and horizons...")
        results = pipeline.run_all()
        print(f"\n[SUCCESS] Completed training and prediction generation.")
        print(f"Processed Commodities : {results['commodities_processed']}")
        print(f"Predictions Written   : {results['output_predictions_path']}")
        print(f"Evaluation Metrics    : {results['output_evaluation_path']}")
    else:
        print(f"\n[INFO] Training model and generating forecast for {args.commodity.upper()} (T+{args.horizon}d)...")
        eval_metrics = pipeline.train_and_evaluate(args.commodity)
        pred = pipeline.predict_latest(commodity=args.commodity, horizon=args.horizon)

        print("\n" + "-" * 70)
        print(" LATEST PREDICTION SUMMARY (API Schema Conforming)")
        print("-" * 70)
        print(json.dumps(pred, indent=2))

        print("\n" + "-" * 70)
        print(" WALK-FORWARD CROSS-VALIDATION SUMMARY")
        print("-" * 70)
        h_key = f"{args.horizon}d"
        bt = eval_metrics["horizons"][h_key]["backtest"]
        print(f"Horizon               : T+{bt['horizon_days']} trading day(s)")
        print(f"Completed Folds       : {bt['n_folds']}")
        print(f"Mean Macro-F1         : {bt['mean_macro_f1']} (Std: {bt['std_macro_f1']})")
        print(f"Baseline Macro-F1     : {bt['baseline_mean_macro_f1']}")
        print(f"Outperforms Baseline  : {'YES' if bt['beats_baseline'] else 'NO'}")
        print("=" * 70)


if __name__ == "__main__":
    main()
