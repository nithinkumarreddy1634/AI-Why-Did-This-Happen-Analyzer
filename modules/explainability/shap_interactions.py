"""SHAP feature interaction analysis and pairwise attribution matrices."""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd

from modules.explainability.shap_explainer import SHAPExplainerWrapper, SHAPCalculationResult
from utils.shap_helpers import sample_dataset_safely


@dataclass
class InteractionAnalysisResult:
    """Container holding pairwise feature interaction importances and diagnostics."""
    is_supported: bool
    top_interacting_pairs: List[Dict[str, Any]]
    interaction_matrix: Optional[pd.DataFrame] = None
    samples_used: int = 0
    message: str = ""
    disclaimer: str = (
        "SHAP interaction values decompose how the joint presence of two features shifts model predictions "
        "beyond the sum of their individual main effects."
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_supported": self.is_supported,
            "samples_used": self.samples_used,
            "message": self.message,
            "top_pairs": self.top_interacting_pairs,
        }


def compute_shap_interactions(
    wrapper: SHAPExplainerWrapper,
    calc_result: SHAPCalculationResult,
    max_samples: int = 40,
    top_k_features: int = 6,
) -> InteractionAnalysisResult:
    """Compute pairwise SHAP interaction values with defensive computational guardrails."""
    if wrapper.metadata.explainer_type != "TreeExplainer":
        return InteractionAnalysisResult(
            is_supported=False,
            top_interacting_pairs=[],
            message=(
                f"SHAP interaction analysis is only efficiently supported for tree-based models via TreeExplainer. "
                f"Active explainer is '{wrapper.metadata.explainer_type}'."
            ),
        )

    # Filter to top K most important features to keep O(K^2) manageable
    feature_names = calc_result.feature_names
    mean_abs_shaps = np.mean(np.abs(calc_result.shap_values), axis=0)
    top_indices = np.argsort(mean_abs_shaps)[::-1][:top_k_features]
    active_features = [feature_names[i] for i in top_indices]

    # Subsample instances
    X_arr = calc_result.X_explained.to_numpy()
    X_sub = sample_dataset_safely(X_arr, max_samples=max_samples, random_state=wrapper.random_state)
    n_sub = len(X_sub)

    try:
        # Check if explainer supports shap_interaction_values
        if hasattr(wrapper.explainer, "shap_interaction_values"):
            raw_interactions = wrapper.explainer.shap_interaction_values(X_sub)
        else:
            return InteractionAnalysisResult(
                is_supported=False,
                top_interacting_pairs=[],
                message="Underlying tree explainer does not expose shap_interaction_values.",
            )

        # Handle multiclass or 4D shape (n_samples, n_features, n_features, n_classes)
        if isinstance(raw_interactions, list):
            eff_idx = min(calc_result.target_class_idx, len(raw_interactions) - 1)
            inter_matrix = raw_interactions[eff_idx]
        elif isinstance(raw_interactions, np.ndarray) and raw_interactions.ndim == 4:
            eff_idx = min(calc_result.target_class_idx, raw_interactions.shape[3] - 1)
            inter_matrix = raw_interactions[:, :, :, eff_idx]
        else:
            inter_matrix = np.asarray(raw_interactions, dtype=float)

        # Slice to top_indices
        sliced_matrix = inter_matrix[:, top_indices, :][:, :, top_indices]

        # Mean absolute interaction strength across samples
        mean_abs_inter = np.mean(np.abs(sliced_matrix), axis=0)

        # Extract non-diagonal pairwise interactions
        pairs: List[Dict[str, Any]] = []
        k = len(active_features)
        for i in range(k):
            for j in range(i + 1, k):
                strength = float(mean_abs_inter[i, j] + mean_abs_inter[j, i])
                pairs.append({
                    "Feature_1": active_features[i],
                    "Feature_2": active_features[j],
                    "Pair": f"{active_features[i]} ✖ {active_features[j]}",
                    "Interaction_Strength": round(strength, 4),
                })

        df_pairs = pd.DataFrame(pairs).sort_values(by="Interaction_Strength", ascending=False).reset_index(drop=True)
        top_pairs_list = df_pairs.to_dict(orient="records")

        # Symmetric interaction DataFrame for heatmap
        df_inter = pd.DataFrame(mean_abs_inter, index=active_features, columns=active_features)

        msg = (
            f"Successfully computed SHAP interaction strengths across {n_sub} subsampled records "
            f"for the top {k} predictive features."
        )

        return InteractionAnalysisResult(
            is_supported=True,
            top_interacting_pairs=top_pairs_list,
            interaction_matrix=df_inter,
            samples_used=n_sub,
            message=msg,
        )

    except Exception as ex:
        return InteractionAnalysisResult(
            is_supported=False,
            top_interacting_pairs=[],
            message=f"Interaction calculation failed or timed out: {str(ex)}",
        )

