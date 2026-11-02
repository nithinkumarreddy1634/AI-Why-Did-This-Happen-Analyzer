"""Master orchestrator and caching manager for LIME Explainability Engine."""

from typing import Dict, Any, Optional, List, Tuple, Union
import numpy as np
import pandas as pd

from modules.explainability.lime_explainer import (
    LIMEConfig,
    LIMEExplainerWrapper,
    LIMEExplainerMetadata,
)
from modules.explainability.lime_local_analysis import LIMEExplanationResult
from modules.explainability.lime_classification import explain_classification_instance
from modules.explainability.lime_regression import explain_regression_instance
from modules.explainability.lime_shap_comparison import (
    SHAPLIMEComparisonResult,
    compare_shap_and_lime,
)
from modules.explainability.lime_batch_analysis import (
    BatchLIMEResult,
    run_batch_lime_explanations,
)
from modules.explainability.lime_interpreter import (
    LIMEInsight,
    generate_lime_narrative_insights,
)
from modules.explainability.shap_local import LocalSHAPResult


class LIMEExplanationManager:
    """Master facade providing explainer caching, instance explanation, SHAP comparison, and batch execution."""

    def __init__(self) -> None:
        self._explainer_cache: Dict[str, LIMEExplainerWrapper] = {}
        self._explanation_cache: Dict[str, LIMEExplanationResult] = {}

    def clear_cache(self) -> None:
        """Clear all cached LIME explainers and instance explanations."""
        self._explainer_cache.clear()
        self._explanation_cache.clear()

    def build_explainer_wrapper(
        self,
        estimator: Any,
        feature_names: List[str],
        X_train_trans: Union[np.ndarray, pd.DataFrame],
        is_classification: bool = True,
        class_names: Optional[List[str]] = None,
        model_id: str = "default_model",
        config: Optional[LIMEConfig] = None,
    ) -> LIMEExplainerWrapper:
        """Initialize or retrieve cached LIME explainer fitted on background training data."""
        cfg = config or LIMEConfig()
        cache_key = f"{model_id}_{len(feature_names)}_{cfg.discretizer}_{cfg.random_state}"

        if cache_key in self._explainer_cache:
            return self._explainer_cache[cache_key]

        wrapper = LIMEExplainerWrapper(
            estimator=estimator,
            feature_names=feature_names,
            is_classification=is_classification,
            class_names=class_names,
            config=cfg,
        )
        wrapper.initialize_explainer(X_train_trans)
        self._explainer_cache[cache_key] = wrapper
        return wrapper

    def explain_instance(
        self,
        wrapper: LIMEExplainerWrapper,
        instance_row: Union[np.ndarray, pd.Series],
        sample_index: int = 0,
        target_class_idx: int = 1,
        num_features: int = 10,
        num_samples: int = 1000,
        actual_outcome: Optional[Any] = None,
        cache_key: Optional[str] = None,
    ) -> LIMEExplanationResult:
        """Generate and cache a local LIME explanation for a single observation."""
        if cache_key and cache_key in self._explanation_cache:
            return self._explanation_cache[cache_key]

        if wrapper.is_classification:
            res = explain_classification_instance(
                wrapper=wrapper,
                instance_row=instance_row,
                sample_index=sample_index,
                target_class_idx=target_class_idx,
                num_features=num_features,
                num_samples=num_samples,
                actual_outcome=actual_outcome,
            )
        else:
            res = explain_regression_instance(
                wrapper=wrapper,
                instance_row=instance_row,
                sample_index=sample_index,
                num_features=num_features,
                num_samples=num_samples,
                actual_outcome=actual_outcome,
            )

        if cache_key:
            self._explanation_cache[cache_key] = res

        return res

    def compare_with_shap(
        self,
        shap_result: LocalSHAPResult,
        lime_result: LIMEExplanationResult,
        top_k: int = 8,
    ) -> SHAPLIMEComparisonResult:
        """Benchmark local explanations between SHAP and LIME for the same instance."""
        return compare_shap_and_lime(
            shap_result=shap_result,
            lime_result=lime_result,
            top_k=top_k,
        )

    def explain_batch(
        self,
        wrapper: LIMEExplainerWrapper,
        X_eval_df: pd.DataFrame,
        sample_indices: List[int],
        target_class_idx: int = 1,
        num_features: int = 6,
        num_samples: int = 500,
        actual_outcomes: Optional[np.ndarray] = None,
    ) -> BatchLIMEResult:
        """Execute batch LIME explanations across multiple selected observations."""
        return run_batch_lime_explanations(
            wrapper=wrapper,
            X_eval_df=X_eval_df,
            sample_indices=sample_indices,
            target_class_idx=target_class_idx,
            num_features=num_features,
            num_samples=num_samples,
            actual_outcomes=actual_outcomes,
        )

    def generate_narrative(
        self,
        lime_result: LIMEExplanationResult,
        comparison_result: Optional[SHAPLIMEComparisonResult] = None,
        batch_result: Optional[BatchLIMEResult] = None,
    ) -> List[LIMEInsight]:
        """Synthesize automated narrative findings."""
        return generate_lime_narrative_insights(
            lime_result=lime_result,
            comparison_result=comparison_result,
            batch_result=batch_result,
        )

