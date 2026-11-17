"""
Backtesting and walk-forward cross-validation module.
Enforces purged and embargoed rolling splits to eliminate lookahead bias.
"""

from typing import Dict, Generator, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score

from .config import PredictionConfig
from .features import prepare_feature_matrix
from .models import CommodityDirectionClassifier, BaselineMajorityClassifier


class PurgedWalkForwardCV:
    """
    Time-series cross validation with an explicit purge/embargo window.
    Ensures that multi-period target labels (e.g. T+5) from the training window
    do not overlap into or leak future data into the test evaluation window.
    """

    def __init__(
        self,
        train_window_size: int = 252,
        test_window_size: int = 42,
        purge_window: int = 1,
    ):
        self.train_window_size = train_window_size
        self.test_window_size = test_window_size
        self.purge_window = purge_window

    def split(self, df: pd.DataFrame) -> Generator[Tuple[pd.Index, pd.Index], None, None]:
        n = len(df)
        total_fold_size = self.train_window_size + self.purge_window + self.test_window_size
        if n < total_fold_size:
            # If dataset is smaller than full fold, use single 70/30 split with purge
            train_len = int(n * 0.7)
            test_start = train_len + self.purge_window
            if test_start < n:
                yield df.index[:train_len], df.index[test_start:]
            return

        start = 0
        while start + total_fold_size <= n:
            train_idx = df.index[start : start + self.train_window_size]
            test_start = start + self.train_window_size + self.purge_window
            test_idx = df.index[test_start : test_start + self.test_window_size]
            yield train_idx, test_idx
            start += self.test_window_size


def run_backtest(
    df: pd.DataFrame,
    horizon: int = 1,
    config: Optional[PredictionConfig] = None,
) -> Dict[str, Any]:
    """
    Executes walk-forward backtesting for a given horizon across all available historical periods.
    """
    cfg = config or PredictionConfig()
    X, y, feature_cols = prepare_feature_matrix(df, horizon=horizon, include_target=True)

    cv = PurgedWalkForwardCV(
        train_window_size=cfg.train_window_size,
        test_window_size=cfg.test_window_size,
        purge_window=horizon,
    )

    fold_scores: List[float] = []
    baseline_scores: List[float] = []
    accuracies: List[float] = []

    for train_idx, test_idx in cv.split(X):
        X_train, y_train = X.loc[train_idx], y.loc[train_idx]
        X_test, y_test = X.loc[test_idx], y.loc[test_idx]

        # Model evaluation
        clf = CommodityDirectionClassifier(horizon_days=horizon)
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)

        macro_f1 = float(f1_score(y_test, preds, average="macro", zero_division=0))
        acc = float(accuracy_score(y_test, preds))
        fold_scores.append(round(macro_f1, 4))
        accuracies.append(round(acc, 4))

        # Baseline evaluation
        base_clf = BaselineMajorityClassifier().fit(X_train, y_train)
        base_preds = base_clf.predict(X_test)
        base_f1 = float(f1_score(y_test, base_preds, average="macro", zero_division=0))
        baseline_scores.append(round(base_f1, 4))

    mean_f1 = float(np.mean(fold_scores)) if fold_scores else 0.0
    mean_base = float(np.mean(baseline_scores)) if baseline_scores else 0.0

    return {
        "horizon_days": horizon,
        "n_folds": len(fold_scores),
        "mean_macro_f1": round(mean_f1, 4),
        "std_macro_f1": round(float(np.std(fold_scores)), 4) if fold_scores else 0.0,
        "mean_accuracy": round(float(np.mean(accuracies)), 4) if accuracies else 0.0,
        "baseline_mean_macro_f1": round(mean_base, 4),
        "beats_baseline": mean_f1 > mean_base,
        "fold_macro_f1s": fold_scores,
    }
