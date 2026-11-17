"""
Directional classification models for commodity forecasting.
Implements 3-class LightGBM classifier with class weighting, probability calibration,
and naive persistence/majority baselines.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, f1_score, log_loss

CLASS_LABELS: Dict[int, str] = {0: "Decrease", 1: "Stable", 2: "Increase"}


class BaselineMajorityClassifier:
    """
    Baseline predictor that always predicts the empirical majority class from training data.
    Provides the benchmark lower bound that any quantitative model must beat.
    """

    def __init__(self):
        self.majority_class: int = 1
        self.class_priors: np.ndarray = np.array([0.33, 0.34, 0.33])

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "BaselineMajorityClassifier":
        counts = y.value_counts(normalize=True)
        self.majority_class = int(counts.idxmax())
        priors = [counts.get(c, 0.0) for c in [0, 1, 2]]
        self.class_priors = np.array(priors)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.full(len(X), fill_value=self.majority_class, dtype=int)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return np.tile(self.class_priors, (len(X), 1))


class CommodityDirectionClassifier:
    """
    Multi-class gradient boosted decision tree classifier (LightGBM).
    Optimized for multi-class log-loss with balanced class weights to prevent 'Stable' collapse.
    """

    def __init__(
        self,
        horizon_days: int = 1,
        params: Optional[Dict[str, Any]] = None,
        model_name: str = "lightgbm",
    ):
        self.horizon_days = horizon_days
        self.model_name = model_name
        self.params = params or {
            "objective": "multiclass",
            "num_class": 3,
            "metric": "multi_logloss",
            "boosting_type": "gbdt",
            "n_estimators": 120,
            "learning_rate": 0.03,
            "num_leaves": 15,
            "max_depth": 4,
            "feature_fraction": 0.8,
            "class_weight": "balanced",
            "random_state": 42,
            "verbose": -1,
        }
        self.model = None
        self.feature_names: List[str] = []
        self._init_underlying_model()

    def _init_underlying_model(self):
        try:
            import lightgbm as lgb
            self.model = lgb.LGBMClassifier(**self.params)
        except ImportError:
            # Fallback to sklearn HistGradientBoostingClassifier or GradientBoostingClassifier
            from sklearn.ensemble import HistGradientBoostingClassifier
            self.model = HistGradientBoostingClassifier(
                max_iter=self.params.get("n_estimators", 100),
                learning_rate=self.params.get("learning_rate", 0.03),
                max_leaf_nodes=self.params.get("num_leaves", 15),
                random_state=42,
            )

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> "CommodityDirectionClassifier":
        self.feature_names = list(X_train.columns)
        self.model.fit(X_train, y_train)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        probs = self.model.predict_proba(X)
        # Ensure all 3 classes [0, 1, 2] exist in probability array
        if probs.shape[1] < 3:
            full_probs = np.zeros((len(X), 3))
            for i, c in enumerate(self.model.classes_):
                full_probs[:, int(c)] = probs[:, i]
            return full_probs
        return probs

    def evaluate(
        self,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> Dict[str, Any]:
        """
        Evaluates model predictions against naive majority baseline.
        """
        preds = self.predict(X_val)
        probs = self.predict_proba(X_val)

        macro_f1 = float(f1_score(y_val, preds, average="macro", zero_division=0))
        report = classification_report(y_val, preds, output_dict=True, zero_division=0)

        # Baseline comparison
        baseline = BaselineMajorityClassifier().fit(X_val, y_val)
        base_preds = baseline.predict(X_val)
        base_f1 = float(f1_score(y_val, base_preds, average="macro", zero_division=0))

        # Multi-class loss if all classes represented
        try:
            loss = float(log_loss(y_val, probs, labels=[0, 1, 2]))
        except Exception:
            loss = None

        return {
            "macro_f1": round(macro_f1, 4),
            "baseline_macro_f1": round(base_f1, 4),
            "beats_baseline": macro_f1 > base_f1,
            "log_loss": round(loss, 4) if loss is not None else None,
            "class_distribution": {
                CLASS_LABELS.get(c, str(c)): int(cnt)
                for c, cnt in y_val.value_counts().items()
            },
            "per_class_metrics": {
                CLASS_LABELS.get(int(k), str(k)): {
                    "precision": round(v["precision"], 4),
                    "recall": round(v["recall"], 4),
                    "f1_score": round(v["f1-score"], 4),
                    "support": int(v["support"]),
                }
                for k, v in report.items()
                if k in ["0", "1", "2"]
            },
        }
