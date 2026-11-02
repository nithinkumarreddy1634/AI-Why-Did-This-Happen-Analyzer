"""Batch LIME explanation processing across multiple selected observations."""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np

from modules.explainability.lime_explainer import LIMEExplainerWrapper
from modules.explainability.lime_classification import explain_classification_instance
from modules.explainability.lime_regression import explain_regression_instance
from modules.explainability.lime_local_analysis import LIMEExplanationResult


@dataclass
class BatchLIMEResult:
    """Summary and tabular attributions for a cohort of explained observations."""
    n_explained: int
    explanations: List[LIMEExplanationResult]
    summary_table: pd.DataFrame
    recurring_top_features: List[Dict[str, Any]]
    mean_fidelity_r2: float

    def to_csv(self) -> str:
        """Export batch explanation table to CSV string."""
        return self.summary_table.to_csv(index=False)


def run_batch_lime_explanations(
    wrapper: LIMEExplainerWrapper,
    X_eval_df: pd.DataFrame,
    sample_indices: List[int],
    target_class_idx: int = 1,
    num_features: int = 6,
    num_samples: int = 500,
    actual_outcomes: Optional[np.ndarray] = None,
) -> BatchLIMEResult:
    """Execute batch LIME explanations across a curated list of record indices."""
    explanations: List[LIMEExplanationResult] = []
    rows: List[Dict[str, Any]] = []
    feature_counts: Dict[str, int] = {}
    fidelities: List[float] = []

    for idx in sample_indices:
        if idx < 0 or idx >= len(X_eval_df):
            continue

        row_series = X_eval_df.iloc[idx]
        actual_val = actual_outcomes[idx] if actual_outcomes is not None and len(actual_outcomes) > idx else None

        if wrapper.is_classification:
            res = explain_classification_instance(
                wrapper=wrapper,
                instance_row=row_series,
                sample_index=idx,
                target_class_idx=target_class_idx,
                num_features=num_features,
                num_samples=num_samples,
                actual_outcome=actual_val,
            )
        else:
            res = explain_regression_instance(
                wrapper=wrapper,
                instance_row=row_series,
                sample_index=idx,
                num_features=num_features,
                num_samples=num_samples,
                actual_outcome=actual_val,
            )

        explanations.append(res)
        fidelities.append(res.local_fidelity_r2)

        # Track top 3 features for recurrence analysis
        top_3 = res.contributions_df.head(3)["Feature"].tolist()
        for f in top_3:
            feature_counts[f] = feature_counts.get(f, 0) + 1

        # Summary row
        rows.append({
            "Sample_Index": idx,
            "Target": res.target_class_name,
            "Actual": res.actual_outcome,
            "Predicted": res.predicted_outcome,
            "Surrogate_Pred": round(res.local_surrogate_prediction, 3),
            "Fidelity_R2": round(res.local_fidelity_r2, 3),
            "Top_Positive_Driver": res.positive_drivers[0]["Condition"] if res.positive_drivers else "None",
            "Top_Negative_Driver": res.negative_drivers[0]["Condition"] if res.negative_drivers else "None",
        })

    df_summary = pd.DataFrame(rows)
    mean_fid = float(np.mean(fidelities)) if fidelities else 0.0

    # Recurring features breakdown
    recurring = [
        {"Feature": k, "Appearance_Count": v, "Frequency_Pct": round(v / len(explanations) * 100, 1)}
        for k, v in sorted(feature_counts.items(), key=lambda item: item[1], reverse=True)
    ]

    return BatchLIMEResult(
        n_explained=len(explanations),
        explanations=explanations,
        summary_table=df_summary,
        recurring_top_features=recurring,
        mean_fidelity_r2=round(mean_fid, 4),
    )

