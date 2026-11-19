"""
Price Prediction and Forecasting Module (MLOps 2).
Provides technical indicator engineering, directional target labeling,
real historical price ingestion, LightGBM/XGBoost multi-class classifiers,
SHAP explainability, and purged walk-forward cross-validation.
"""

from .config import PredictionConfig
from .data_loader import (
    BasePriceProvider,
    YahooFinanceProvider,
    PriceDataLoader,
    DataValidationError,
    ProviderError,
    DataUnavailableError,
)

try:
    from .features import (
        compute_technical_indicators,
        generate_directional_targets,
        prepare_feature_matrix,
    )
    from .models import CommodityDirectionClassifier, BaselineMajorityClassifier
    from .explain import ShapExplainer
    from .backtest import PurgedWalkForwardCV, run_backtest
    from .pipeline import PricePredictionPipeline
except ImportError:
    # Allows data_loader and config to be used even if pandas/numpy C-extensions are restricted
    compute_technical_indicators = None
    generate_directional_targets = None
    prepare_feature_matrix = None
    CommodityDirectionClassifier = None
    BaselineMajorityClassifier = None
    ShapExplainer = None
    PurgedWalkForwardCV = None
    run_backtest = None
    PricePredictionPipeline = None

__all__ = [
    "PredictionConfig",
    "BasePriceProvider",
    "YahooFinanceProvider",
    "PriceDataLoader",
    "DataValidationError",
    "ProviderError",
    "DataUnavailableError",
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
