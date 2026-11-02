"""Global SHAP feature importance and aggregate attribution metrics."""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd

from modules.explainability.shap_explainer import SHAPCalculationResult


@dataclass
class GlobalSHAPResult:
    """Summary of population-level mean absolute SHAP values and contribution rankings."""
    importance_df: pd.DataFrame
    top_features: List[str]
    mean_abs_shaps: Dict[str, float]
    target_class_name: Optional[str]
    total_explained_samples: int
    summary_text: str
    disclaimer: str = (
        "Global SHAP feature importance quantifies how strongly each feature shifts model predictions on average. "
        "It measures predictive magnitude across the dataset, not real-world causality or actionable levers."
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "top_features": self.top_features,
            "target_class_name": self.target_class_name,
            "total_explained_samples": self.total_explained_samples,
            "summary_text": self.summary_text,
            "records": self.importance_df.to_dict(orient="records"),
        }


def compute_global_shap_importance(
    calc_result: SHAPCalculationResult,
    top_n: Optional[int] = None,
) -> GlobalSHAPResult:
    """Calculate mean absolute SHAP importance and contribution directions across explained instances."""
    shap_vals = calc_result.shap_values  # (n_samples, n_features)
    feature_names = calc_result.feature_names
    n_samples, n_features = shap_vals.shape

    mean_abs = np.mean(np.abs(shap_vals), axis=0)
    total_impact = np.sum(mean_abs)
    norm_impact = mean_abs / total_impact if total_impact > 0 else np.zeros_like(mean_abs)

    rows: List[Dict[str, Any]] = []

    for idx, fname in enumerate(feature_names):
        col_shaps = shap_vals[:, idx]
        pos_mask = col_shaps > 0
        neg_mask = col_shaps < 0

        pos_pct = float(np.mean(pos_mask) * 100.0) if n_samples > 0 else 0.0
        neg_pct = float(np.mean(neg_mask) * 100.0) if n_samples > 0 else 0.0
        
        mean_pos = float(np.mean(col_shaps[pos_mask])) if np.any(pos_mask) else 0.0
        mean_neg = float(np.mean(col_shaps[neg_mask])) if np.any(neg_mask) else 0.0

        if pos_pct > 65.0:
            direction_label = "Predominantly Increases Output (+)"
        elif neg_pct > 65.0:
            direction_label = "Predominantly Decreases Output (-)"
        else:
            direction_label = "Bidirectional / Context-Dependent"

        rows.append({
            "Feature": fname,
            "Mean_Abs_SHAP": float(mean_abs[idx]),
            "Relative_Importance": float(norm_impact[idx]),
            "Contribution_Pct": float(norm_impact[idx] * 100.0),
            "Positive_Impact_Frequency": round(pos_pct, 1),
            "Negative_Impact_Frequency": round(neg_pct, 1),
            "Mean_Positive_Shift": round(mean_pos, 4),
            "Mean_Negative_Shift": round(mean_neg, 4),
            "Direction_Tendency": direction_label,
        })

    df_importance = pd.DataFrame(rows).sort_values(by="Mean_Abs_SHAP", ascending=False).reset_index(drop=True)
    df_importance["Rank"] = range(1, len(df_importance) + 1)

    top_feature_list = df_importance["Feature"].tolist()
    if top_n is not None and top_n > 0:
        top_feature_list = top_feature_list[:top_n]

    top_3 = top_feature_list[:3]
    top_3_str = ", ".join([f"'{f}'" for f in top_3])
    target_str = f" for '{calc_result.target_class_name}'" if calc_result.target_class_name else ""

    summary = (
        f"Across {n_samples:,} evaluated observations{target_str}, the 3 most influential predictive features "
        f"are {top_3_str}, accounting for {df_importance['Contribution_Pct'].head(3).sum():.1f}% of total model attribution."
    )

    return GlobalSHAPResult(
        importance_df=df_importance,
        top_features=top_feature_list,
        mean_abs_shaps={row["Feature"]: row["Mean_Abs_SHAP"] for row in rows},
        target_class_name=calc_result.target_class_name,
        total_explained_samples=n_samples,
        summary_text=summary,
    )

