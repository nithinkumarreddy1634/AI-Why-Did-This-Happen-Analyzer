"""Core LIME Tabular Explainer wrapper managing LimeTabularExplainer initialization and configuration."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd
from lime.lime_tabular import LimeTabularExplainer

from utils.shap_helpers import clean_feature_names


@dataclass
class LIMEConfig:
    """Configuration parameters for LIME perturbations and local surrogate fitting."""
    num_samples: int = 1000
    num_features: int = 10
    kernel_width: Optional[float] = None
    discretize_continuous: bool = True
    discretizer: str = "quartile"  # 'quartile', 'decile', 'entropy'
    random_state: int = 42


@dataclass
class LIMEExplainerMetadata:
    """Metadata describing the active LIME explainer setup."""
    mode: str  # 'classification' or 'regression'
    model_class_name: str
    feature_names: List[str]
    n_features: int
    n_background_samples: int
    class_names: Optional[List[str]] = None
    discretizer: str = "quartile"
    is_supported: bool = True
    message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "model_class_name": self.model_class_name,
            "n_features": self.n_features,
            "n_background_samples": self.n_background_samples,
            "class_names": self.class_names,
            "discretizer": self.discretizer,
            "is_supported": self.is_supported,
            "message": self.message,
        }


class LIMEExplainerWrapper:
    """Wrapper encapsulating LimeTabularExplainer configuration and prediction routing."""

    def __init__(
        self,
        estimator: Any,
        feature_names: List[str],
        is_classification: bool = True,
        class_names: Optional[List[str]] = None,
        config: Optional[LIMEConfig] = None,
    ) -> None:
        self.estimator = estimator
        self.raw_feature_names = feature_names
        self.feature_names = clean_feature_names(feature_names)
        self.is_classification = is_classification
        self.class_names = class_names or (["Class 0", "Class 1"] if is_classification else None)
        self.config = config or LIMEConfig()

        self.explainer: Optional[LimeTabularExplainer] = None
        self._background_data: Optional[np.ndarray] = None
        self.metadata = LIMEExplainerMetadata(
            mode="classification" if is_classification else "regression",
            model_class_name=estimator.__class__.__name__ if estimator else "None",
            feature_names=self.feature_names,
            n_features=len(self.feature_names),
            n_background_samples=0,
            class_names=self.class_names,
            discretizer=self.config.discretizer,
        )

    def initialize_explainer(self, X_train: Union[np.ndarray, pd.DataFrame]) -> None:
        """Initialize LimeTabularExplainer on preprocessed background training data."""
        X_bg = X_train.to_numpy() if isinstance(X_train, pd.DataFrame) else np.asarray(X_train, dtype=float)
        self._background_data = X_bg
        self.metadata.n_background_samples = len(X_bg)

        mode_str = "classification" if self.is_classification else "regression"

        self.explainer = LimeTabularExplainer(
            training_data=X_bg,
            feature_names=self.feature_names,
            class_names=self.class_names,
            mode=mode_str,
            kernel_width=self.config.kernel_width,
            discretize_continuous=self.config.discretize_continuous,
            discretizer=self.config.discretizer,
            random_state=self.config.random_state,
        )

    def get_predict_fn(self) -> Any:
        """Return compatible prediction function for LIME perturbation sampling."""
        if self.is_classification:
            if hasattr(self.estimator, "predict_proba"):
                return self.estimator.predict_proba
            elif hasattr(self.estimator, "decision_function"):
                # Wrap decision function into pseudo-probabilities for SVM without probability=True
                def _decision_proba(x: np.ndarray) -> np.ndarray:
                    scores = self.estimator.decision_function(x)
                    if scores.ndim == 1:
                        # Sigmoid
                        prob_1 = 1.0 / (1.0 + np.exp(-scores))
                        return np.column_stack([1.0 - prob_1, prob_1])
                    else:
                        exp_s = np.exp(scores - np.max(scores, axis=1, keepdims=True))
                        return exp_s / np.sum(exp_s, axis=1, keepdims=True)
                return _decision_proba
            else:
                # Fallback to one-hot prediction
                def _pred_one_hot(x: np.ndarray) -> np.ndarray:
                    preds = self.estimator.predict(x)
                    n_classes = len(self.class_names) if self.class_names else 2
                    out = np.zeros((len(preds), n_classes))
                    for i, p in enumerate(preds):
                        idx = int(p) if int(p) < n_classes else 0
                        out[i, idx] = 1.0
                    return out
                return _pred_one_hot
        else:
            return self.estimator.predict

