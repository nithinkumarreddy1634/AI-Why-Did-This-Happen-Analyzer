"""Core SHAP Explainer wrapper managing TreeExplainer, LinearExplainer, and KernelExplainer."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import shap

from modules.explainability.explainer_selector import (
    ExplainerMetadata,
    identify_compatible_explainer,
)
from utils.shap_helpers import (
    clean_feature_names,
    extract_scalar_base_value,
    sample_dataset_safely,
)


@dataclass
class SHAPCalculationResult:
    """Standardized container for computed SHAP values, base values, and metadata."""
    shap_values: np.ndarray  # Shape: (n_samples, n_features) for the active target/class
    base_value: float  # Expected value in model output space
    feature_names: List[str]
    X_explained: pd.DataFrame
    model_predictions: np.ndarray
    explainer_metadata: ExplainerMetadata
    target_class_idx: int = 1
    target_class_name: Optional[str] = None
    all_class_shap_values: Optional[List[np.ndarray]] = None
    raw_shap_explanation: Optional[Any] = None

    @property
    def n_samples(self) -> int:
        return self.shap_values.shape[0]

    @property
    def n_features(self) -> int:
        return self.shap_values.shape[1]


class SHAPExplainerWrapper:
    """Robust wrapper orchestrating explainer fitting, sampling, and output normalization."""

    def __init__(
        self,
        estimator: Any,
        feature_names: List[str],
        is_classification: bool = True,
        class_labels: Optional[List[str]] = None,
        max_background_samples: int = 100,
        random_state: int = 42,
    ) -> None:
        self.estimator = estimator
        self.raw_feature_names = feature_names
        self.clean_feature_names = clean_feature_names(feature_names)
        self.is_classification = is_classification
        self.class_labels = class_labels or (["Class 0", "Class 1"] if is_classification else None)
        self.max_background_samples = max_background_samples
        self.random_state = random_state

        self.metadata = identify_compatible_explainer(estimator)
        self.explainer: Optional[Any] = None
        self._background_data: Optional[np.ndarray] = None

    def initialize_explainer(self, X_train: Union[np.ndarray, pd.DataFrame]) -> None:
        """Initialize and fit the appropriate SHAP explainer on background training data."""
        if not self.metadata.is_supported:
            raise ValueError(f"Cannot initialize explainer: {self.metadata.selection_reason}")

        X_bg = X_train.to_numpy() if isinstance(X_train, pd.DataFrame) else np.asarray(X_train, dtype=float)

        # Subsample background data to prevent slow downs
        sample_size = min(self.max_background_samples, len(X_bg))
        self._background_data = sample_dataset_safely(
            X_bg, max_samples=sample_size, random_state=self.random_state
        )

        explainer_type = self.metadata.explainer_type

        if explainer_type == "TreeExplainer":
            # TreeExplainer fits directly on the tree estimator
            try:
                self.explainer = shap.TreeExplainer(
                    self.estimator,
                    data=self._background_data,
                    feature_perturbation="interventional",
                )
            except Exception:
                # Fallback to path-dependent tree perturbation if interventional fails
                self.explainer = shap.TreeExplainer(self.estimator)

        elif explainer_type == "LinearExplainer":
            # LinearExplainer uses background feature distribution for feature covariances
            self.explainer = shap.LinearExplainer(
                self.estimator,
                self._background_data,
            )

        elif explainer_type == "KernelExplainer":
            # Model-agnostic KernelExplainer with prediction wrapper
            if self.is_classification and hasattr(self.estimator, "predict_proba"):
                pred_fn = self.estimator.predict_proba
            else:
                pred_fn = self.estimator.predict

            # Use kmeans or small sample for kernel background
            if len(self._background_data) > 30:
                bg_sample = shap.kmeans(self._background_data, min(25, len(self._background_data)))
            else:
                bg_sample = self._background_data

            self.explainer = shap.KernelExplainer(pred_fn, bg_sample)

        else:
            raise NotImplementedError(f"Unsupported explainer type: {explainer_type}")

    def explain(
        self,
        X_eval: Union[np.ndarray, pd.DataFrame],
        target_class_idx: int = 1,
        max_eval_samples: Optional[int] = None,
    ) -> SHAPCalculationResult:
        """Calculate SHAP values for evaluation dataset and standardize into unified structure."""
        if self.explainer is None:
            raise RuntimeError("Explainer is not initialized. Call initialize_explainer(X_train) first.")

        X_arr = X_eval.to_numpy() if isinstance(X_eval, pd.DataFrame) else np.asarray(X_eval, dtype=float)

        # Optional evaluation subsampling
        if max_eval_samples is not None and len(X_arr) > max_eval_samples:
            X_arr = sample_dataset_safely(X_arr, max_samples=max_eval_samples, random_state=self.random_state)

        df_explained = pd.DataFrame(X_arr, columns=self.clean_feature_names)

        # Compute raw predictions
        if self.is_classification and hasattr(self.estimator, "predict_proba"):
            preds = self.estimator.predict_proba(X_arr)
            # If binary and 2D, take positive class probability
            if preds.ndim == 2 and preds.shape[1] > target_class_idx:
                model_preds = preds[:, target_class_idx]
            else:
                model_preds = preds[:, -1]
        else:
            model_preds = self.estimator.predict(X_arr)

        # Compute raw SHAP explanation
        try:
            raw_explanation = self.explainer(X_arr, check_additivity=False)
        except (TypeError, Exception):
            raw_explanation = self.explainer(X_arr)

        # Standardize SHAP values array and base values across different SHAP versions
        shap_values_raw = getattr(raw_explanation, "values", raw_explanation)
        base_vals_raw = getattr(raw_explanation, "base_values", getattr(self.explainer, "expected_value", 0.0))

        # Shape handling:
        # Case A: 3D array (n_samples, n_features, n_classes)
        all_class_shaps = None
        if isinstance(shap_values_raw, np.ndarray) and shap_values_raw.ndim == 3:
            n_classes = shap_values_raw.shape[2]
            all_class_shaps = [shap_values_raw[:, :, c] for c in range(n_classes)]
            eff_idx = min(target_class_idx, n_classes - 1)
            active_shap = shap_values_raw[:, :, eff_idx]
            active_base = extract_scalar_base_value(base_vals_raw, class_idx=eff_idx)

        # Case B: List of 2D arrays (one per class)
        elif isinstance(shap_values_raw, list):
            all_class_shaps = shap_values_raw
            eff_idx = min(target_class_idx, len(shap_values_raw) - 1)
            active_shap = shap_values_raw[eff_idx]
            active_base = extract_scalar_base_value(base_vals_raw, class_idx=eff_idx)

        # Case C: 2D array (n_samples, n_features) - standard regression or binary single output
        elif isinstance(shap_values_raw, np.ndarray) and shap_values_raw.ndim == 2:
            active_shap = shap_values_raw
            active_base = extract_scalar_base_value(base_vals_raw, class_idx=target_class_idx)

        else:
            active_shap = np.asarray(shap_values_raw, dtype=float)
            active_base = extract_scalar_base_value(base_vals_raw, class_idx=target_class_idx)

        class_name = None
        if self.class_labels and len(self.class_labels) > target_class_idx:
            class_name = self.class_labels[target_class_idx]
        elif self.is_classification:
            class_name = f"Class {target_class_idx}"

        return SHAPCalculationResult(
            shap_values=np.asarray(active_shap, dtype=float),
            base_value=active_base,
            feature_names=self.clean_feature_names,
            X_explained=df_explained,
            model_predictions=np.asarray(model_preds, dtype=float),
            explainer_metadata=self.metadata,
            target_class_idx=target_class_idx,
            target_class_name=class_name,
            all_class_shap_values=all_class_shaps,
            raw_shap_explanation=raw_explanation,
        )
