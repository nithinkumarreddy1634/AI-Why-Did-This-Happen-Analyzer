"""LIME local observation explanation container and fidelity decomposition."""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd

from utils.lime_helpers import parse_lime_feature_condition, format_lime_weight


@dataclass
class LIMEExplanationResult:
    """Standardized output container for a single instance LIME explanation."""
    sample_index: int
    is_classification: bool
    target_class_idx: int
    target_class_name: str
    actual_outcome: Optional[Any]
    predicted_outcome: Any
    predicted_probabilities: Optional[List[float]]
    contributions_df: pd.DataFrame
    positive_drivers: List[Dict[str, Any]]
    negative_drivers: List[Dict[str, Any]]
    local_surrogate_prediction: float
    local_fidelity_r2: float
    local_intercept: float
    fidelity_gap: float
    raw_feature_values: Dict[str, Any]
    narrative_summary: str
    disclaimer: str = (
        "LIME fits a local, interpretable surrogate model (e.g. Ridge regression) in a perturbed neighborhood around the "
        "selected observation. Feature weights explain the local surrogate's approximation, not global rules or real-world causality."
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sample_index": self.sample_index,
            "is_classification": self.is_classification,
            "target_class_idx": self.target_class_idx,
            "target_class_name": self.target_class_name,
            "actual_outcome": self.actual_outcome,
            "predicted_outcome": self.predicted_outcome,
            "predicted_probabilities": self.predicted_probabilities,
            "local_surrogate_prediction": round(self.local_surrogate_prediction, 4),
            "local_fidelity_r2": round(self.local_fidelity_r2, 4),
            "local_intercept": round(self.local_intercept, 4),
            "fidelity_gap": round(self.fidelity_gap, 4),
            "narrative_summary": self.narrative_summary,
            "contributions": self.contributions_df.to_dict(orient="records"),
        }


def parse_lime_raw_explanation(
    exp: Any,
    sample_index: int,
    raw_instance_row: pd.Series,
    feature_names: List[str],
    is_classification: bool = True,
    target_class_idx: int = 1,
    class_names: Optional[List[str]] = None,
    actual_outcome: Optional[Any] = None,
    predicted_outcome: Optional[Any] = None,
    predicted_probabilities: Optional[np.ndarray] = None,
) -> LIMEExplanationResult:
    """Parse raw lime.explanation.Explanation object into structured LIMEExplanationResult."""
    label_to_explain = target_class_idx if is_classification else 1

    # Extract raw list of (condition, weight)
    try:
        raw_list = exp.as_list(label=label_to_explain)
    except Exception:
        # Fallback if label indexing is different
        avail_labels = exp.available_labels() if hasattr(exp, "available_labels") else [0]
        label_to_explain = avail_labels[0] if avail_labels else 0
        raw_list = exp.as_list(label=label_to_explain)

    # Local surrogate score (R-squared fidelity) and intercept
    fidelity_r2 = float(getattr(exp, "score", 0.0))
    # Intercept can be dict or array
    intercept_val = 0.0
    if hasattr(exp, "intercept"):
        if isinstance(exp.intercept, dict) and label_to_explain in exp.intercept:
            intercept_val = float(exp.intercept[label_to_explain])
        elif isinstance(exp.intercept, (list, np.ndarray)) and len(exp.intercept) > label_to_explain:
            intercept_val = float(exp.intercept[label_to_explain])

    # Local surrogate prediction
    local_pred_val = 0.0
    if hasattr(exp, "local_pred"):
        if isinstance(exp.local_pred, (list, np.ndarray)):
            if len(exp.local_pred) > label_to_explain:
                local_pred_val = float(exp.local_pred[label_to_explain])
            else:
                local_pred_val = float(exp.local_pred[0])
        else:
            local_pred_val = float(exp.local_pred)
    else:
        # Calculate local pred from sum of weights + intercept
        local_pred_val = float(intercept_val + sum(w for _, w in raw_list))

    # Model predicted outcome / probability
    if is_classification:
        if predicted_probabilities is not None and len(predicted_probabilities) > label_to_explain:
            orig_model_pred = float(predicted_probabilities[label_to_explain])
        else:
            orig_model_pred = float(local_pred_val)
    else:
        orig_model_pred = float(predicted_outcome) if predicted_outcome is not None else float(local_pred_val)

    fidelity_gap = abs(orig_model_pred - local_pred_val)

    # Build contributions DataFrame
    rows: List[Dict[str, Any]] = []
    for cond_str, weight in raw_list:
        parsed = parse_lime_feature_condition(cond_str)
        fname = parsed["feature_name"]

        # Retrieve raw observed value from row
        obs_val = raw_instance_row.get(fname, None)
        if obs_val is None:
            # Check if fname is substring or matches a clean column
            for col in raw_instance_row.index:
                if col in fname or fname in col:
                    obs_val = raw_instance_row[col]
                    break

        shift_dir = "Supports Outcome (+)" if weight > 0 else "Opposes Outcome (-)"

        rows.append({
            "Condition": cond_str,
            "Feature": fname,
            "Observed_Value": obs_val,
            "LIME_Weight": float(weight),
            "Abs_Weight": float(abs(weight)),
            "Formatted_Weight": format_lime_weight(weight),
            "Direction": shift_dir,
            "Sign": 1 if weight > 0 else -1,
        })

    df_contrib = pd.DataFrame(rows).sort_values(by="Abs_Weight", ascending=False).reset_index(drop=True)
    df_contrib["Rank"] = range(1, len(df_contrib) + 1)

    pos_drivers = df_contrib[df_contrib["LIME_Weight"] > 0].to_dict(orient="records")
    neg_drivers = df_contrib[df_contrib["LIME_Weight"] < 0].to_dict(orient="records")

    class_name = None
    if class_names and len(class_names) > target_class_idx:
        class_name = class_names[target_class_idx]
    elif is_classification:
        class_name = f"Class {target_class_idx}"
    else:
        class_name = "Target Value"

    # Narrative generation
    target_desc = f"'{class_name}'"
    pos_top = [f"**{d['Condition']}** ({d['Formatted_Weight']})" for d in pos_drivers[:2]]
    neg_top = [f"**{d['Condition']}** ({d['Formatted_Weight']})" for d in neg_drivers[:2]]

    narrative_parts = [
        f"LIME explained prediction for record **#{sample_index}** target {target_desc} "
        f"with a local surrogate fidelity of **R² = {fidelity_r2:.3f}**."
    ]

    if pos_top:
        narrative_parts.append(f"Features most strongly supporting this outcome: {', and '.join(pos_top)}.")
    if neg_top:
        narrative_parts.append(f"Features opposing or counteracting this outcome: {', and '.join(neg_top)}.")

    narrative = " ".join(narrative_parts)

    raw_vals = {str(k): v for k, v in raw_instance_row.to_dict().items()}

    return LIMEExplanationResult(
        sample_index=sample_index,
        is_classification=is_classification,
        target_class_idx=target_class_idx,
        target_class_name=class_name,
        actual_outcome=actual_outcome,
        predicted_outcome=predicted_outcome,
        predicted_probabilities=predicted_probabilities.tolist() if isinstance(predicted_probabilities, np.ndarray) else predicted_probabilities,
        contributions_df=df_contrib,
        positive_drivers=pos_drivers,
        negative_drivers=neg_drivers,
        local_surrogate_prediction=local_pred_val,
        local_fidelity_r2=fidelity_r2,
        local_intercept=intercept_val,
        fidelity_gap=fidelity_gap,
        raw_feature_values=raw_vals,
        narrative_summary=narrative,
    )

