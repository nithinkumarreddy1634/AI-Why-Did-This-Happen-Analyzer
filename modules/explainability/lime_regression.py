"""Local LIME explanations for regression models."""

from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd

from modules.explainability.lime_explainer import LIMEExplainerWrapper
from modules.explainability.lime_local_analysis import (
    LIMEExplanationResult,
    parse_lime_raw_explanation,
)


def explain_regression_instance(
    wrapper: LIMEExplainerWrapper,
    instance_row: Union[np.ndarray, pd.Series],
    sample_index: int = 0,
    num_features: int = 10,
    num_samples: int = 1000,
    actual_outcome: Optional[Any] = None,
) -> LIMEExplanationResult:
    """Explain a single regression instance using LimeTabularExplainer."""
    if wrapper.explainer is None:
        raise RuntimeError("LIME explainer is not initialized. Initialize with background data first.")

    # Convert instance to 1D numpy array
    if isinstance(instance_row, pd.Series):
        x_inst = instance_row.to_numpy(dtype=float)
        series_inst = instance_row
    else:
        x_inst = np.asarray(instance_row, dtype=float).flatten()
        series_inst = pd.Series(x_inst, index=wrapper.feature_names)

    predict_fn = wrapper.get_predict_fn()

    # Model prediction value
    pred_val = float(predict_fn(x_inst.reshape(1, -1))[0])

    # Generate LIME explanation
    exp = wrapper.explainer.explain_instance(
        data_row=x_inst,
        predict_fn=predict_fn,
        num_features=num_features,
        num_samples=num_samples,
    )

    return parse_lime_raw_explanation(
        exp=exp,
        sample_index=sample_index,
        raw_instance_row=series_inst,
        feature_names=wrapper.feature_names,
        is_classification=False,
        target_class_idx=0,
        actual_outcome=actual_outcome,
        predicted_outcome=pred_val,
    )

