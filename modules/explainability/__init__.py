"""Explainable AI (XAI) package for Phase 8 SHAP and Phase 9 LIME model interpretability."""

# SHAP Engine Exports
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
from modules.explainability.explanation_manager import SHAPExplanationManager

# LIME Engine Exports
from modules.explainability.lime_explainer import (
    LIMEConfig,
    LIMEExplainerWrapper,
    LIMEExplainerMetadata,
)
from modules.explainability.lime_local_analysis import (
    LIMEExplanationResult,
    parse_lime_raw_explanation,
)
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
from modules.explainability.lime_manager import LIMEExplanationManager

__all__ = [
    # SHAP
    "ExplainerMetadata",
    "identify_compatible_explainer",
    "SHAPExplainerWrapper",
    "SHAPCalculationResult",
    "GlobalSHAPResult",
    "compute_global_shap_importance",
    "LocalSHAPResult",
    "explain_individual_prediction",
    "DependenceDataResult",
    "compute_feature_dependence_data",
    "InteractionAnalysisResult",
    "compute_shap_interactions",
    "SHAPInsight",
    "generate_shap_narrative_insights",
    "SHAPExplanationManager",
    # LIME
    "LIMEConfig",
    "LIMEExplainerWrapper",
    "LIMEExplainerMetadata",
    "LIMEExplanationResult",
    "parse_lime_raw_explanation",
    "explain_classification_instance",
    "explain_regression_instance",
    "SHAPLIMEComparisonResult",
    "compare_shap_and_lime",
    "BatchLIMEResult",
    "run_batch_lime_explanations",
    "LIMEInsight",
    "generate_lime_narrative_insights",
    "LIMEExplanationManager",
]
