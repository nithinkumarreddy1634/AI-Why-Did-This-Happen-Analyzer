"""Local LIME explanations for classification models."""

from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd

from modules.explainability.lime_explainer import LIMEExplainerWrapper
from modules.explainability.lime_local_analysis import (
    LIMEExplanationResult,
    parse_lime_raw_explanation,
)


def explain_classification_instance(
    wrapper: LIMEExplainerWrapper,
    instance_row: Union[np.ndarray, pd.Series],
    sample_index: int = 0,
    target_class_idx: int = 1,
    num_features: int = 10,
    num_samples: int = 1000,
    actual_outcome: Optional[Any] = None,
) -> LIMEExplanationResult:
    """Explain a single classification instance using LimeTabularExplainer."""
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

    # Model prediction probabilities
    proba = predict_fn(x_inst.reshape(1, -1))[0]
    pred_class_idx = int(np.argmax(proba))
    pred_label = (
        wrapper.class_names[pred_class_idx]
        if wrapper.class_names and len(wrapper.class_names) > pred_class_idx
        else str(pred_class_idx)
    )

    # Generate LIME explanation
    exp = wrapper.explainer.explain_instance(
        data_row=x_inst,
        predict_fn=predict_fn,
        labels=(target_class_idx,),
        num_features=num_features,
        num_samples=num_samples,
    )

    return parse_lime_raw_explanation(
        exp=exp,
        sample_index=sample_index,
        raw_instance_row=series_inst,
        feature_names=wrapper.feature_names,
        is_classification=True,
        target_class_idx=target_class_idx,
        class_names=wrapper.class_names,
        actual_outcome=actual_outcome,
        predicted_outcome=pred_label,
        predicted_probabilities=proba,
    )
