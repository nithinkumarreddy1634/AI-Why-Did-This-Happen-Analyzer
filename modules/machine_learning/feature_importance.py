"""Feature importance analysis supporting tree-based, permutation, and linear coefficient methods."""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np

from sklearn.inspection import permutation_importance
from sklearn.base import BaseEstimator


@dataclass
class FeatureImportanceResult:
    """Container holding feature importance rankings, method metadata, and disclaimers."""
    method: str
    method_display: str
    importance_df: pd.DataFrame
    top_features: List[str]
    model_name: str
    is_causal: bool = False
    explanation: str = ""
    disclaimer: str = (
        "Note: Feature importance scores quantify statistical predictive associations within this specific model. "
        "High feature importance does NOT prove causal influence or direct real-world mechanisms."
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "method_display": self.method_display,
            "top_features": self.top_features,
            "model_name": self.model_name,
            "is_causal": self.is_causal,
            "explanation": self.explanation,
            "records": self.importance_df.to_dict(orient="records"),
        }


def extract_feature_importance(
    estimator: BaseEstimator,
    feature_names: List[str],
    X_test: Optional[np.ndarray] = None,
    y_test: Optional[np.ndarray] = None,
    preferred_method: str = "auto",  # 'auto', 'tree', 'permutation', 'coefficients'
    model_name: str = "Model",
    n_repeats: int = 5,
    random_state: int = 42,
) -> FeatureImportanceResult:
    """Calculate feature importance using tree metrics, permutation importance, or linear coefficients.
    
    Parameters
    ----------
    estimator : Fitted scikit-learn model.
    feature_names : Transformed feature column names.
    X_test, y_test : Optional test set data (required for permutation importance).
    preferred_method : Strategy ('auto', 'tree', 'permutation', 'coefficients').
    model_name : Name of the algorithm for reporting.
    """
    n_features = len(feature_names)
    
    has_tree_importance = hasattr(estimator, "feature_importances_")
    has_coef = hasattr(estimator, "coef_")
    can_permute = X_test is not None and y_test is not None and len(X_test) > 0

    chosen_method = preferred_method
    if chosen_method == "auto":
        if has_tree_importance:
            chosen_method = "tree"
        elif has_coef:
            chosen_method = "coefficients"
        elif can_permute:
            chosen_method = "permutation"
        else:
            chosen_method = "none"

    rows: List[Dict[str, Any]] = []

    # 1. Tree-based Feature Importance (MDI / Gini / Impurity decrease)
    if chosen_method == "tree" and has_tree_importance:
        raw_imp = estimator.feature_importances_
        if len(raw_imp) == n_features:
            sum_imp = np.sum(raw_imp)
            norm_imp = raw_imp / sum_imp if sum_imp > 0 else raw_imp
            for fname, imp, n_imp in zip(feature_names, raw_imp, norm_imp):
                rows.append({
                    "Feature": fname,
                    "Importance": float(imp),
                    "Normalized_Importance": float(n_imp),
                    "Percentage": float(n_imp * 100.0),
                    "Direction": "Positive (Impurity Reduction)",
                })
            method_display = "Mean Decrease in Impurity (Tree-Based)"
            explanation = "Calculates total reduction in split impurity (Gini/Entropy/MSE) brought by each feature across all trees."

    # 2. Linear Model Coefficients
    elif chosen_method == "coefficients" and has_coef:
        coef = estimator.coef_
        # For multiclass, take mean absolute coefficient across classes
        if coef.ndim > 1 and coef.shape[0] > 1:
            raw_imp = np.mean(np.abs(coef), axis=0)
            directions = ["Multiclass Average" for _ in range(len(raw_imp))]
        elif coef.ndim > 1:
            raw_imp = coef[0]
            directions = ["Positive (+)" if c >= 0 else "Negative (-)" for c in raw_imp]
        else:
            raw_imp = coef
            directions = ["Positive (+)" if c >= 0 else "Negative (-)" for c in raw_imp]

        if len(raw_imp) == n_features:
            abs_imp = np.abs(raw_imp)
            sum_abs = np.sum(abs_imp)
            norm_imp = abs_imp / sum_abs if sum_abs > 0 else abs_imp
            for fname, raw_c, n_imp, d in zip(feature_names, raw_imp, norm_imp, directions):
                rows.append({
                    "Feature": fname,
                    "Importance": float(abs(raw_c)),
                    "Normalized_Importance": float(n_imp),
                    "Percentage": float(n_imp * 100.0),
                    "Direction": d,
                    "Raw_Coefficient": float(raw_c),
                })
            method_display = "Linear Model Coefficients (Absolute Magnitude)"
            explanation = "Measures the unit change in log-odds or predicted continuous target per unit change in standardized feature."

    # 3. Permutation Feature Importance
    elif (chosen_method == "permutation" or not rows) and can_permute:
        perm_res = permutation_importance(
            estimator,
            X_test,
            y_test,
            n_repeats=n_repeats,
            random_state=random_state,
            n_jobs=-1,
        )
        raw_imp = np.maximum(0.0, perm_res.importances_mean)
        sum_imp = np.sum(raw_imp)
        norm_imp = raw_imp / sum_imp if sum_imp > 0 else np.zeros_like(raw_imp)
        
        for fname, imp, n_imp, std_val in zip(feature_names, raw_imp, norm_imp, perm_res.importances_std):
            rows.append({
                "Feature": fname,
                "Importance": float(imp),
                "Normalized_Importance": float(n_imp),
                "Percentage": float(n_imp * 100.0),
                "Std_Error": float(std_val),
                "Direction": "Model Performance Drop",
            })
        method_display = "Permutation Importance (Model-Agnostic)"
        explanation = "Evaluates the decrease in holdout test evaluation score when values of a single feature are randomly shuffled."

    # Fallback if no method succeeded
    if not rows:
        for fname in feature_names:
            rows.append({
                "Feature": fname,
                "Importance": 0.0,
                "Normalized_Importance": 0.0,
                "Percentage": 0.0,
                "Direction": "N/A",
            })
        method_display = "Unavailable"
        explanation = "Feature importance calculation is not supported for this estimator configuration."

    df_imp = pd.DataFrame(rows)
    df_imp = df_imp.sort_values(by="Importance", ascending=False).reset_index(drop=True)
    top_features = df_imp["Feature"].head(5).tolist()

    return FeatureImportanceResult(
        method=chosen_method,
        method_display=method_display,
        importance_df=df_imp,
        top_features=top_features,
        model_name=model_name,
        is_causal=False,
        explanation=explanation,
    )

