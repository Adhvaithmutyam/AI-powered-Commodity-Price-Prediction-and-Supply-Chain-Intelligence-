"""
Price Prediction and Forecasting Module (MLOps 2).
Provides technical indicator engineering, directional target labeling,
LightGBM/XGBoost multi-class classifiers, SHAP explainability, and purged walk-forward cross-validation.
"""

from .config import PredictionConfig
from .features import (
    compute_technical_indicators,
    generate_directional_targets,
    prepare_feature_matrix,
)
from .models import CommodityDirectionClassifier, BaselineMajorityClassifier
from .explain import ShapExplainer
from .backtest import PurgedWalkForwardCV, run_backtest
from .pipeline import PricePredictionPipeline

__all__ = [
    "PredictionConfig",
    "compute_technical_indicators",
    "generate_directional_targets",
    "prepare_feature_matrix",
    "CommodityDirectionClassifier",
    "BaselineMajorityClassifier",
    "ShapExplainer",
    "PurgedWalkForwardCV",
    "run_backtest",
    "PricePredictionPipeline",
]
