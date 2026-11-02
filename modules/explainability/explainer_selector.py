"""Automatic explainer selection and model compatibility analyzer."""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Type
import sklearn.ensemble
import sklearn.tree
import sklearn.linear_model
import sklearn.neighbors
import sklearn.svm
import sklearn.naive_bayes


@dataclass
class ExplainerMetadata:
    """Metadata describing the chosen SHAP explainer and compatibility profile."""
    explainer_type: str  # 'TreeExplainer', 'LinearExplainer', 'KernelExplainer', 'Unsupported'
    model_class_name: str
    is_supported: bool
    selection_reason: str
    is_approximate: bool
    recommended_sample_size: int
    supports_interactions: bool
    warning_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "explainer_type": self.explainer_type,
            "model_class_name": self.model_class_name,
            "is_supported": self.is_supported,
            "selection_reason": self.selection_reason,
            "is_approximate": self.is_approximate,
            "recommended_sample_size": self.recommended_sample_size,
            "supports_interactions": self.supports_interactions,
            "warning_message": self.warning_message,
        }


def identify_compatible_explainer(estimator: Any) -> ExplainerMetadata:
    """Analyze a fitted estimator and select the optimal, mathematically compatible SHAP explainer."""
    if estimator is None:
        return ExplainerMetadata(
            explainer_type="Unsupported",
            model_class_name="None",
            is_supported=False,
            selection_reason="Estimator is None.",
            is_approximate=False,
            recommended_sample_size=0,
            supports_interactions=False,
            warning_message="No trained estimator provided.",
        )

    class_name = estimator.__class__.__name__

    # 1. Tree-based Models -> TreeExplainer (Exact, fast, polynomial-time TreeSHAP)
    tree_classes = (
        sklearn.ensemble.RandomForestClassifier,
        sklearn.ensemble.RandomForestRegressor,
        sklearn.tree.DecisionTreeClassifier,
        sklearn.tree.DecisionTreeRegressor,
        sklearn.ensemble.GradientBoostingClassifier,
        sklearn.ensemble.GradientBoostingRegressor,
        sklearn.ensemble.ExtraTreesClassifier,
        sklearn.ensemble.ExtraTreesRegressor,
    )

    is_tree = isinstance(estimator, tree_classes)
    # Check for XGBoost by class name or module to avoid hard import error
    if not is_tree and "xgb" in estimator.__class__.__module__.lower():
        is_tree = True

    if is_tree:
        return ExplainerMetadata(
            explainer_type="TreeExplainer",
            model_class_name=class_name,
            is_supported=True,
            selection_reason=(
                f"Selected TreeExplainer for '{class_name}'. TreeSHAP provides fast, exact O(TLD^2) "
                "polynomial-time calculation of Shapley values for decision trees and tree ensembles."
            ),
            is_approximate=False,
            recommended_sample_size=500,
            supports_interactions=True,
        )

    # 2. Linear Models -> LinearExplainer (Exact analytical solution for linear combinations)
    linear_classes = (
        sklearn.linear_model.LogisticRegression,
        sklearn.linear_model.LinearRegression,
        sklearn.linear_model.Ridge,
        sklearn.linear_model.Lasso,
        sklearn.linear_model.ElasticNet,
        sklearn.linear_model.SGDClassifier,
        sklearn.linear_model.SGDRegressor,
    )

    if isinstance(estimator, linear_classes):
        return ExplainerMetadata(
            explainer_type="LinearExplainer",
            model_class_name=class_name,
            is_supported=True,
            selection_reason=(
                f"Selected LinearExplainer for '{class_name}'. LinearSHAP computes exact Shapley values "
                "directly from model coefficients and feature covariances in linear space."
            ),
            is_approximate=False,
            recommended_sample_size=1000,
            supports_interactions=False,
        )

    # 3. Non-linear, non-tree models -> KernelExplainer (Model-agnostic weighted linear surrogate)
    kernel_classes = (
        sklearn.neighbors.KNeighborsClassifier,
        sklearn.neighbors.KNeighborsRegressor,
        sklearn.svm.SVC,
        sklearn.svm.SVR,
        sklearn.naive_bayes.GaussianNB,
    )

    if isinstance(estimator, kernel_classes):
        return ExplainerMetadata(
            explainer_type="KernelExplainer",
            model_class_name=class_name,
            is_supported=True,
            selection_reason=(
                f"Selected KernelExplainer for '{class_name}'. As a non-tree, non-linear model, "
                "KernelSHAP uses weighted local surrogate regression to estimate Shapley values."
            ),
            is_approximate=True,
            recommended_sample_size=50,
            supports_interactions=False,
            warning_message=(
                "KernelExplainer relies on combinatorial sampling. Background data and evaluation instances "
                "are automatically subsampled to maintain responsive computation."
            ),
        )

    # 4. Fallback: Generic estimator check
    if hasattr(estimator, "predict"):
        return ExplainerMetadata(
            explainer_type="KernelExplainer",
            model_class_name=class_name,
            is_supported=True,
            selection_reason=(
                f"Fallback to model-agnostic KernelExplainer for custom estimator '{class_name}'."
            ),
            is_approximate=True,
            recommended_sample_size=30,
            supports_interactions=False,
            warning_message="Model-agnostic sampling will be applied for KernelSHAP estimation.",
        )

    return ExplainerMetadata(
        explainer_type="Unsupported",
        model_class_name=class_name,
        is_supported=False,
        selection_reason=f"Model class '{class_name}' does not implement standard scikit-learn prediction interfaces.",
        is_approximate=False,
        recommended_sample_size=0,
        supports_interactions=False,
        warning_message="Unsupported estimator architecture.",
    )

