"""
Unit and integration tests for Commodity Price Prediction & Forecasting (MLOps 2).
Tests run deterministically offline and verify zero lookahead leakage and API contract compliance.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.price_prediction import (
    PredictionConfig,
    PricePredictionPipeline,
    CommodityDirectionClassifier,
    BaselineMajorityClassifier,
    PurgedWalkForwardCV,
    compute_technical_indicators,
    generate_directional_targets,
    prepare_feature_matrix,
    run_backtest,
    ShapExplainer,
)


@pytest.fixture
def sample_price_series() -> pd.DataFrame:
    """Generates deterministic mock historical price series for 120 trading days."""
    np.random.seed(42)
    dates = pd.date_range(start="2026-01-01", periods=120, freq="B")
    close = 100.0 + np.cumsum(np.random.normal(0.05, 1.2, size=120))
    high = close + np.random.uniform(0.5, 1.5, size=120)
    low = close - np.random.uniform(0.5, 1.5, size=120)
    open_p = (high + low) / 2.0
    volume = np.random.randint(5000, 25000, size=120)

    return pd.DataFrame({
        "timestamp": dates.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "commodity": "copper",
        "open": open_p,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    })


# -------------------------------------------------------------------------
# 1. Feature Engineering & Indicator Tests
# -------------------------------------------------------------------------

def test_compute_technical_indicators(sample_price_series):
    feat_df = compute_technical_indicators(sample_price_series)

    expected_cols = [
        "log_ret_1d",
        "ret_lag_1d",
        "ret_lag_5d",
        "sma_20",
        "dist_sma_20",
        "volatility_21d",
        "rsi_14",
        "macd",
        "macd_signal",
        "bb_pct_b",
        "atr_14",
    ]
    for col in expected_cols:
        assert col in feat_df.columns
        assert not feat_df[col].isnull().all(), f"Column {col} has all NaN values"


def test_generate_directional_targets(sample_price_series):
    df_targets = generate_directional_targets(
        sample_price_series,
        horizons=[1, 5],
        thresholds={1: 0.005, 5: 0.010},
    )

    assert "target_1d" in df_targets.columns
    assert "target_5d" in df_targets.columns

    # Verify class labels are strictly within {0, 1, 2}
    t1_valid = df_targets["target_1d"].dropna().unique()
    assert set(t1_valid).issubset({0, 1, 2})


# -------------------------------------------------------------------------
# 2. Anti-Leakage Cross-Validation Tests
# -------------------------------------------------------------------------

def test_purged_walk_forward_cv_purge_window():
    """Verify that train indices and test indices are separated by at least purge_window."""
    dummy_df = pd.DataFrame({"val": range(350)})
    purge_w = 5
    cv = PurgedWalkForwardCV(train_window_size=100, test_window_size=30, purge_window=purge_w)

    splits = list(cv.split(dummy_df))
    assert len(splits) > 0

    for train_idx, test_idx in splits:
        gap = test_idx[0] - train_idx[-1]
        assert gap > purge_w, f"Purge window violated: gap {gap} <= purge {purge_w}"


# -------------------------------------------------------------------------
# 3. Model Training & Baseline Benchmarking Tests
# -------------------------------------------------------------------------

def test_baseline_majority_classifier():
    X = pd.DataFrame({"f1": [1.0, 2.0, 3.0, 4.0, 5.0]})
    y = pd.Series([0, 1, 1, 1, 2])

    clf = BaselineMajorityClassifier().fit(X, y)
    preds = clf.predict(X)
    probs = clf.predict_proba(X)

    assert (preds == 1).all()
    assert probs.shape == (5, 3)
    np.testing.assert_almost_equal(probs[0, 1], 0.6)


def test_commodity_direction_classifier_fit_and_evaluate(sample_price_series):
    feat_df = compute_technical_indicators(sample_price_series)
    df_targets = generate_directional_targets(feat_df, horizons=[1])
    X, y, _ = prepare_feature_matrix(df_targets, horizon=1, include_target=True)

    clf = CommodityDirectionClassifier(horizon_days=1)
    clf.fit(X.iloc[:80], y.iloc[:80])

    probs = clf.predict_proba(X.iloc[80:])
    assert probs.shape == (len(X) - 80, 3)
    np.testing.assert_almost_equal(probs.sum(axis=1), np.ones(len(probs)), decimal=4)

    eval_results = clf.evaluate(X.iloc[80:], y.iloc[80:])
    assert "macro_f1" in eval_results
    assert "baseline_macro_f1" in eval_results
    assert "per_class_metrics" in eval_results


# -------------------------------------------------------------------------
# 4. Explainability & Contract Schema Tests
# -------------------------------------------------------------------------

def test_shap_explainer_schema_conformance(sample_price_series):
    feat_df = compute_technical_indicators(sample_price_series)
    df_targets = generate_directional_targets(feat_df, horizons=[1])
    X, y, _ = prepare_feature_matrix(df_targets, horizon=1, include_target=True)

    clf = CommodityDirectionClassifier(horizon_days=1).fit(X, y)
    explainer = ShapExplainer(clf, background_sample=X)

    top_features = explainer.get_top_features(single_row=X.iloc[[-1]], predicted_class_idx=2, top_k=5)
    assert len(top_features) == 5
    for item in top_features:
        assert "feature" in item
        assert "impact" in item
        assert "value" in item
        assert "description" in item


def test_end_to_end_pipeline_predict_latest(sample_price_series):
    pipeline = PricePredictionPipeline()
    pred = pipeline.predict_latest(
        commodity="copper",
        horizon=1,
        latest_price_df=sample_price_series,
    )

    # Validate against integration API contract
    assert pred["commodity"] == "copper"
    assert pred["horizon"] == "1d"
    assert pred["predicted_direction"] in ["Decrease", "Stable", "Increase"]
    assert "probabilities" in pred
    assert set(pred["probabilities"].keys()) == {"decrease", "stable", "increase"}
    prob_sum = sum(pred["probabilities"].values())
    assert 0.98 <= prob_sum <= 1.02

    assert "current_price" in pred
    assert pred["current_price"]["value"] > 0
    assert pred["current_price"]["currency"] == "USD"

    assert len(pred["top_features"]) == 5
    assert pred["model_name"] == "lightgbm_direction_classifier"
    assert pred["model_version"] == "v1.0.0"
