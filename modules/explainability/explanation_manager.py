"""Master orchestrator and caching manager for SHAP Explainability Engine."""

from typing import Dict, Any, Optional, List, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from modules.explainability.explainer_selector import (
    ExplainerMetadata,
    identify_compatible_explainer,
)
from modules.explainability.shap_explainer import (
    SHAPExplainerWrapper,
    SHAPCalculationResult,
)
from modules.explainability.shap_global import (
    GlobalSHAPResult,
    compute_global_shap_importance,
)
from modules.explainability.shap_local import (
    LocalSHAPResult,
    explain_individual_prediction,
)
from modules.explainability.shap_dependence import (
    DependenceDataResult,
    compute_feature_dependence_data,
)
from modules.explainability.shap_interactions import (
    InteractionAnalysisResult,
    compute_shap_interactions,
)
from modules.explainability.shap_insights import (
    SHAPInsight,
    generate_shap_narrative_insights,
)


class SHAPExplanationManager:
    """Master facade providing caching, validation, and unified XAI workflows."""

    def __init__(self) -> None:
        # Cache keyed by model signature
        self._explainer_cache: Dict[str, SHAPExplainerWrapper] = {}
        self._results_cache: Dict[str, SHAPCalculationResult] = {}

    def clear_cache(self) -> None:
        """Clear all cached explainers and calculations."""
        self._explainer_cache.clear()
        self._results_cache.clear()

    def build_explainer_wrapper(
        self,
        estimator: Any,
        feature_names: List[str],
        X_train_trans: Union[np.ndarray, pd.DataFrame],
        is_classification: bool = True,
        class_labels: Optional[List[str]] = None,
        model_id: str = "default_model",
        max_background_samples: int = 100,
        random_state: int = 42,
    ) -> SHAPExplainerWrapper:
        """Instantiate and initialize a fitted SHAP explainer with caching."""
        cache_key = f"{model_id}_{len(feature_names)}"

        if cache_key in self._explainer_cache:
            return self._explainer_cache[cache_key]

        wrapper = SHAPExplainerWrapper(
            estimator=estimator,
            feature_names=feature_names,
            is_classification=is_classification,
            class_labels=class_labels,
            max_background_samples=max_background_samples,
            random_state=random_state,
        )
        wrapper.initialize_explainer(X_train_trans)
        self._explainer_cache[cache_key] = wrapper
        return wrapper

    def compute_explanations(
        self,
        wrapper: SHAPExplainerWrapper,
        X_eval_trans: Union[np.ndarray, pd.DataFrame],
        target_class_idx: int = 1,
        max_eval_samples: int = 250,
        cache_key: Optional[str] = None,
    ) -> SHAPCalculationResult:
        """Compute and cache SHAP values on transformed evaluation data."""
        if cache_key and cache_key in self._results_cache:
            return self._results_cache[cache_key]

        res = wrapper.explain(
            X_eval=X_eval_trans,
            target_class_idx=target_class_idx,
            max_eval_samples=max_eval_samples,
        )

        if cache_key:
            self._results_cache[cache_key] = res

        return res

    def extract_global_importance(
        self,
        calc_result: SHAPCalculationResult,
        top_n: Optional[int] = None,
    ) -> GlobalSHAPResult:
        """Compute population-level SHAP rankings."""
        return compute_global_shap_importance(calc_result, top_n=top_n)

    def extract_local_explanation(
        self,
        calc_result: SHAPCalculationResult,
        sample_index: int = 0,
        actual_outcomes: Optional[np.ndarray] = None,
        predicted_labels: Optional[List[str]] = None,
    ) -> LocalSHAPResult:
        """Compute individual observation attribution."""
        return explain_individual_prediction(
            calc_result=calc_result,
            sample_index=sample_index,
            actual_outcomes=actual_outcomes,
            predicted_labels=predicted_labels,
        )

    def extract_dependence_profile(
        self,
        calc_result: SHAPCalculationResult,
        primary_feature: str,
        interaction_feature: Optional[str] = None,
    ) -> DependenceDataResult:
        """Compute feature dependence scatter coordinates."""
        return compute_feature_dependence_data(
            calc_result=calc_result,
            primary_feature=primary_feature,
            interaction_feature=interaction_feature,
        )

    def extract_interactions(
        self,
        wrapper: SHAPExplainerWrapper,
        calc_result: SHAPCalculationResult,
        max_samples: int = 40,
        top_k: int = 6,
    ) -> InteractionAnalysisResult:
        """Compute pairwise feature interactions."""
        return compute_shap_interactions(
            wrapper=wrapper,
            calc_result=calc_result,
            max_samples=max_samples,
            top_k_features=top_k,
        )

    def generate_narrative(
        self,
        global_result: GlobalSHAPResult,
        local_result: Optional[LocalSHAPResult] = None,
    ) -> List[SHAPInsight]:
        """Synthesize automated narrative findings."""
        return generate_shap_narrative_insights(
            global_result=global_result,
            local_result=local_result,
        )

