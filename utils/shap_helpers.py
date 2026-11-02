"""Utility helper functions for SHAP calculations, formatting, and data validation."""

from typing import List, Optional, Dict, Any, Union
import numpy as np
import pandas as pd


def clean_feature_names(feature_names: List[str]) -> List[str]:
    """Clean feature names by removing sklearn pipeline prefixes like 'num__' or 'cat__'."""
    cleaned = []
    for name in feature_names:
        clean = str(name).replace("num__", "").replace("cat__", "").replace("remainder__", "")
        cleaned.append(clean)
    return cleaned


def extract_scalar_base_value(base_value: Any, class_idx: int = 1) -> float:
    """Safely extract a float scalar base value (expected value) from various SHAP structures."""
    if base_value is None:
        return 0.0

    # If it's already a float or int
    if isinstance(base_value, (int, float, np.floating, np.integer)):
        return float(base_value)

    # If it's a list or tuple
    if isinstance(base_value, (list, tuple)):
        if len(base_value) == 0:
            return 0.0
        if len(base_value) > class_idx:
            return extract_scalar_base_value(base_value[class_idx])
        return extract_scalar_base_value(base_value[0])

    # If it's a numpy array
    if isinstance(base_value, np.ndarray):
        if base_value.ndim == 0:
            return float(base_value.item())
        if base_value.ndim == 1:
            if len(base_value) > class_idx:
                return float(base_value[class_idx])
            return float(base_value[0])
        # Higher dimensional
        flat = base_value.flatten()
        if len(flat) > class_idx:
            return float(flat[class_idx])
        return float(flat[0])

    try:
        return float(base_value)
    except Exception:
        return 0.0


def check_shap_additivity(
    base_value: float,
    shap_values: np.ndarray,
    model_prediction: float,
    tolerance: float = 1e-2,
) -> Dict[str, Any]:
    """Verify SHAP additive property: Base Value + Sum(SHAP) ~= Model Output."""
    shap_sum = float(np.sum(shap_values))
    expected_output = base_value + shap_sum
    difference = abs(expected_output - model_prediction)
    is_valid = bool(difference <= tolerance or (abs(model_prediction) > 0 and difference / abs(model_prediction) <= 0.05))

    return {
        "base_value": round(base_value, 4),
        "shap_sum": round(shap_sum, 4),
        "expected_output": round(expected_output, 4),
        "actual_model_output": round(model_prediction, 4),
        "discrepancy": round(difference, 5),
        "is_additive": is_valid,
    }


def sample_dataset_safely(
    X: Union[np.ndarray, pd.DataFrame],
    max_samples: int = 100,
    random_state: int = 42,
) -> Union[np.ndarray, pd.DataFrame]:
    """Sample dataset safely to prevent combinatorial slowdown in KernelExplainer or interactions."""
    n_rows = len(X)
    if n_rows <= max_samples:
        return X

    rng = np.random.RandomState(random_state)
    sample_indices = rng.choice(n_rows, size=max_samples, replace=False)

    if isinstance(X, pd.DataFrame):
        return X.iloc[sample_indices].reset_index(drop=True)
    return X[sample_indices]
