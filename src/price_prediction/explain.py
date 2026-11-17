"""
Explainability module providing SHAP feature attributions for commodity forecasts.
Conforms to the backend API schema contract for transparent decision support.
"""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from .features import FEATURE_DESCRIPTIONS


class ShapExplainer:
    """
    Computes local SHAP attributions using TreeExplainer for LightGBM models.
    Falls back gracefully to model feature importances if SHAP is uninstalled or input is unsupported.
    """

    def __init__(self, model: Any, background_sample: pd.DataFrame):
        self.raw_model = model.model if hasattr(model, "model") else model
        self.feature_names = list(background_sample.columns)
        self.explainer = None
        self._init_explainer(background_sample)

    def _init_explainer(self, background_sample: pd.DataFrame):
        try:
            import shap
            # Fast tree explainer for tree-based models
            self.explainer = shap.TreeExplainer(self.raw_model)
        except Exception:
            self.explainer = None

    def get_top_features(
        self,
        single_row: pd.DataFrame,
        predicted_class_idx: int,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Extracts top-k features influencing the predicted directional probability.
        """
        if self.explainer is not None:
            try:
                shap_values = self.explainer.shap_values(single_row)
                if isinstance(shap_values, list):
                    class_shap = shap_values[predicted_class_idx][0]
                elif hasattr(shap_values, "shape") and len(shap_values.shape) == 3:
                    class_shap = shap_values[0, :, predicted_class_idx]
                else:
                    class_shap = shap_values[0]

                impacts = []
                for idx, feat in enumerate(self.feature_names):
                    val = float(single_row[feat].values[0]) if feat in single_row else 0.0
                    impact = float(class_shap[idx])
                    impacts.append({
                        "feature": feat,
                        "impact": round(impact, 4),
                        "value": round(val, 4),
                        "description": FEATURE_DESCRIPTIONS.get(
                            feat, f"Quantitative impact of {feat}"
                        ),
                    })

                impacts.sort(key=lambda x: abs(x["impact"]), reverse=True)
                return impacts[:top_k]
            except Exception:
                pass

        # Fallback to normalized feature importances if TreeExplainer fails
        return self._fallback_feature_importances(single_row, top_k)

    def _fallback_feature_importances(
        self, single_row: pd.DataFrame, top_k: int
    ) -> List[Dict[str, Any]]:
        importances = getattr(self.raw_model, "feature_importances_", None)
        if importances is None:
            importances = np.ones(len(self.feature_names)) / len(self.feature_names)

        total = np.sum(importances) + 1e-9
        norm_imp = importances / total

        impacts = []
        for idx, feat in enumerate(self.feature_names):
            val = float(single_row[feat].values[0]) if feat in single_row else 0.0
            impacts.append({
                "feature": feat,
                "impact": round(float(norm_imp[idx]), 4),
                "value": round(val, 4),
                "description": FEATURE_DESCRIPTIONS.get(
                    feat, f"Quantitative impact of {feat}"
                ),
            })

        impacts.sort(key=lambda x: abs(x["impact"]), reverse=True)
        return impacts[:top_k]
