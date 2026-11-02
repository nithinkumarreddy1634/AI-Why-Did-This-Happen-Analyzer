"""SHAP feature dependence analysis and non-linear response curves."""

from dataclasses import dataclass
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd

from modules.explainability.shap_explainer import SHAPCalculationResult


@dataclass
class DependenceDataResult:
    """Coordinates and metadata for a SHAP dependence plot."""
    primary_feature: str
    feature_values: List[Any]
    shap_values: List[float]
    interaction_feature: Optional[str]
    interaction_values: Optional[List[Any]]
    correlation: float
    description: str
    disclaimer: str = (
        "SHAP dependence curves visualize how the model alters its prediction across observed feature values. "
        "Non-linearities or threshold steps reflect model logic, not verified real-world causal mechanisms."
    )

    def to_df(self) -> pd.DataFrame:
        data = {
            self.primary_feature: self.feature_values,
            f"SHAP_{self.primary_feature}": self.shap_values,
        }
        if self.interaction_feature and self.interaction_values is not None:
            data[self.interaction_feature] = self.interaction_values
        return pd.DataFrame(data)


def compute_feature_dependence_data(
    calc_result: SHAPCalculationResult,
    primary_feature: str,
    interaction_feature: Optional[str] = None,
) -> DependenceDataResult:
    """Extract coordinates and correlation dynamics for feature dependence analysis."""
    feature_names = calc_result.feature_names
    if primary_feature not in feature_names:
        raise ValueError(f"Feature '{primary_feature}' not found in explained features.")

    feat_idx = feature_names.index(primary_feature)
    shaps = calc_result.shap_values[:, feat_idx]
    x_vals = calc_result.X_explained[primary_feature].values

    # Determine optimal interaction feature if none provided
    if interaction_feature is None:
        best_inter = None
        best_score = -1.0
        for idx, other_name in enumerate(feature_names):
            if other_name == primary_feature:
                continue
            other_x = calc_result.X_explained[other_name].values
            # Compute correlation with current feature's SHAP values
            try:
                valid_mask = ~np.isnan(other_x) & ~np.isnan(shaps)
                if np.sum(valid_mask) > 5 and np.std(other_x[valid_mask]) > 1e-6:
                    corr = abs(float(np.corrcoef(other_x[valid_mask], shaps[valid_mask])[0, 1]))
                    if not np.isnan(corr) and corr > best_score:
                        best_score = corr
                        best_inter = other_name
            except Exception:
                continue
        interaction_feature = best_inter

    inter_vals = None
    if interaction_feature and interaction_feature in feature_names:
        inter_vals = calc_result.X_explained[interaction_feature].tolist()

    # Overall linear correlation between feature value and SHAP value
    try:
        valid_m = ~np.isnan(x_vals) & ~np.isnan(shaps)
        corr_val = float(np.corrcoef(x_vals[valid_m], shaps[valid_m])[0, 1]) if np.sum(valid_m) > 5 else 0.0
        corr_val = 0.0 if np.isnan(corr_val) else corr_val
    except Exception:
        corr_val = 0.0

    trend_str = "positive association" if corr_val > 0.3 else ("negative association" if corr_val < -0.3 else "non-linear or weak linear trend")
    inter_desc = f"colored by interacting feature '{interaction_feature}'" if interaction_feature else "without interaction coloring"
    desc = f"Dependence profile for '{primary_feature}' reveals a {trend_str} (r = {corr_val:.2f}) with SHAP attribution, {inter_desc}."

    return DependenceDataResult(
        primary_feature=primary_feature,
        feature_values=x_vals.tolist(),
        shap_values=shaps.tolist(),
        interaction_feature=interaction_feature,
        interaction_values=inter_vals,
        correlation=round(corr_val, 3),
        description=desc,
    )

