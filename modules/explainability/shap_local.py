"""Local SHAP individual prediction explanation and waterfall attribution builder."""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd

from modules.explainability.shap_explainer import SHAPCalculationResult
from utils.shap_helpers import check_shap_additivity


@dataclass
class LocalSHAPResult:
    """Detailed attribution profile for a single observation."""
    sample_index: int
    base_value: float
    model_output: float
    actual_outcome: Optional[Any]
    predicted_label: Optional[str]
    target_class_name: Optional[str]
    contributions_df: pd.DataFrame
    positive_drivers: List[Dict[str, Any]]
    negative_drivers: List[Dict[str, Any]]
    additivity_audit: Dict[str, Any]
    narrative_summary: str
    disclaimer: str = (
        "Local SHAP explanations isolate how feature values shift the model's prediction relative to baseline. "
        "A positive shift increases the model score, while a negative shift decreases it."
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sample_index": self.sample_index,
            "base_value": self.base_value,
            "model_output": self.model_output,
            "actual_outcome": self.actual_outcome,
            "predicted_label": self.predicted_label,
            "target_class_name": self.target_class_name,
            "additivity_audit": self.additivity_audit,
            "narrative_summary": self.narrative_summary,
            "contributions": self.contributions_df.to_dict(orient="records"),
        }


def explain_individual_prediction(
    calc_result: SHAPCalculationResult,
    sample_index: int = 0,
    actual_outcomes: Optional[np.ndarray] = None,
    predicted_labels: Optional[List[str]] = None,
) -> LocalSHAPResult:
    """Generate row-level attribution breakdown, waterfall steps, and narrative interpretation."""
    n_samples = calc_result.n_samples
    if sample_index < 0 or sample_index >= n_samples:
        raise IndexError(f"sample_index {sample_index} out of range for {n_samples} explained samples.")

    row_shap = calc_result.shap_values[sample_index]
    row_x = calc_result.X_explained.iloc[sample_index]
    base_val = calc_result.base_value
    model_out = float(calc_result.model_predictions[sample_index])

    actual_val = None
    if actual_outcomes is not None and len(actual_outcomes) > sample_index:
        actual_val = actual_outcomes[sample_index]

    pred_label = None
    if predicted_labels is not None and len(predicted_labels) > sample_index:
        pred_label = predicted_labels[sample_index]

    feature_names = calc_result.feature_names
    rows: List[Dict[str, Any]] = []

    for fname, shap_v in zip(feature_names, row_shap):
        feat_val = row_x[fname]
        shift_dir = "Pushes Prediction Higher (+)" if shap_v > 0 else ("Pushes Prediction Lower (-)" if shap_v < 0 else "Neutral (0)")
        rows.append({
            "Feature": fname,
            "Feature_Value": feat_val,
            "SHAP_Value": float(shap_v),
            "Abs_SHAP": float(abs(shap_v)),
            "Direction": shift_dir,
            "Sign": 1 if shap_v > 0 else (-1 if shap_v < 0 else 0),
        })

    df_contrib = pd.DataFrame(rows).sort_values(by="Abs_SHAP", ascending=False).reset_index(drop=True)
    df_contrib["Rank"] = range(1, len(df_contrib) + 1)

    # Positive and negative driver subsets
    pos_drivers = df_contrib[df_contrib["SHAP_Value"] > 0].to_dict(orient="records")
    neg_drivers = df_contrib[df_contrib["SHAP_Value"] < 0].to_dict(orient="records")

    # Additivity check
    audit = check_shap_additivity(
        base_value=base_val,
        shap_values=row_shap,
        model_prediction=model_out,
    )

    # Narrative generation
    target_label = f"'{calc_result.target_class_name}'" if calc_result.target_class_name else "Target Outcome"
    
    pos_top = [f"**{d['Feature']}** (`{d['Feature_Value']}`, +{d['SHAP_Value']:.3f})" for d in pos_drivers[:2]]
    neg_top = [f"**{d['Feature']}** (`{d['Feature_Value']}`, {d['SHAP_Value']:.3f})" for d in neg_drivers[:2]]

    narrative_parts = [
        f"For observation **#{sample_index}**, the model output is **{model_out:.3f}** "
        f"(compared to the population base expected value of **{base_val:.3f}**)."
    ]

    if pos_top:
        narrative_parts.append(f"The largest upward drivers increasing the score were {', and '.join(pos_top)}.")
    if neg_top:
        narrative_parts.append(f"The largest downward drivers counteracting this score were {', and '.join(neg_top)}.")

    narrative = " ".join(narrative_parts)

    return LocalSHAPResult(
        sample_index=sample_index,
        base_value=base_val,
        model_output=model_out,
        actual_outcome=actual_val,
        predicted_label=pred_label,
        target_class_name=calc_result.target_class_name,
        contributions_df=df_contrib,
        positive_drivers=pos_drivers,
        negative_drivers=neg_drivers,
        additivity_audit=audit,
        narrative_summary=narrative,
    )

