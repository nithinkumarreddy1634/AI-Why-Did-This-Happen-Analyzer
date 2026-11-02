"""Classification algorithms, configurations, and estimators."""

from typing import Dict, Any, Optional, List
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

# Optional XGBoost import
try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except (ImportError, Exception):
    HAS_XGBOOST = False
    XGBClassifier = None


CLASSIFICATION_MODELS_REGISTRY: Dict[str, Dict[str, Any]] = {
    "logistic_regression": {
        "display_name": "Logistic Regression",
        "description": "Linear classifier using the logistic sigmoid function with L2 regularization.",
        "supports_probabilities": True,
        "supports_feature_importance": True,
        "is_tree_based": False,
        "default_params": {"C": 1.0, "max_iter": 1000},
    },
    "decision_tree": {
        "display_name": "Decision Tree Classifier",
        "description": "Single interpretable decision tree splitting on feature thresholds.",
        "supports_probabilities": True,
        "supports_feature_importance": True,
        "is_tree_based": True,
        "default_params": {"max_depth": 5, "min_samples_split": 5},
    },
    "random_forest": {
        "display_name": "Random Forest Classifier",
        "description": "Ensemble of bagging decision trees reducing variance and overfitting.",
        "supports_probabilities": True,
        "supports_feature_importance": True,
        "is_tree_based": True,
        "default_params": {"n_estimators": 100, "max_depth": 8, "min_samples_split": 4},
    },
    "knn": {
        "display_name": "K-Nearest Neighbors",
        "description": "Instance-based non-parametric classifier voting across closest neighbors.",
        "supports_probabilities": True,
        "supports_feature_importance": False,
        "is_tree_based": False,
        "default_params": {"n_neighbors": 5, "weights": "uniform"},
    },
    "svm": {
        "display_name": "Support Vector Machine (SVC)",
        "description": "Maximum-margin hyperplane classifier using RBF kernel.",
        "supports_probabilities": True,
        "supports_feature_importance": False,
        "is_tree_based": False,
        "default_params": {"C": 1.0, "kernel": "rbf", "probability": True},
    },
    "naive_bayes": {
        "display_name": "Gaussian Naive Bayes",
        "description": "Probabilistic classifier applying Bayes' theorem under feature independence.",
        "supports_probabilities": True,
        "supports_feature_importance": False,
        "is_tree_based": False,
        "default_params": {},
    },
    "gradient_boosting": {
        "display_name": "Gradient Boosting Classifier",
        "description": "Sequential boosting ensemble minimizing pseudo-residuals iteratively.",
        "supports_probabilities": True,
        "supports_feature_importance": True,
        "is_tree_based": True,
        "default_params": {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 3},
    },
}

if HAS_XGBOOST:
    CLASSIFICATION_MODELS_REGISTRY["xgboost"] = {
        "display_name": "XGBoost Classifier",
        "description": "Scalable, distributed gradient-boosted decision tree framework.",
        "supports_probabilities": True,
        "supports_feature_importance": True,
        "is_tree_based": True,
        "default_params": {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 4, "eval_metric": "logloss"},
    }


def get_classification_model(
    model_key: str,
    hyperparameters: Optional[Dict[str, Any]] = None,
    random_state: int = 42,
    handle_imbalance: bool = False,
) -> BaseEstimator:
    """Instantiate a classification model with specified hyperparameters and settings."""
    if model_key not in CLASSIFICATION_MODELS_REGISTRY:
        raise ValueError(
            f"Unsupported classification model '{model_key}'. "
            f"Available: {list(CLASSIFICATION_MODELS_REGISTRY.keys())}"
        )

    meta = CLASSIFICATION_MODELS_REGISTRY[model_key]
    params = dict(meta["default_params"])
    if hyperparameters:
        params.update(hyperparameters)

    # Class weight adjustment if requested
    class_weight = "balanced" if handle_imbalance else None

    if model_key == "logistic_regression":
        return LogisticRegression(
            C=float(params.get("C", 1.0)),
            max_iter=int(params.get("max_iter", 1000)),
            class_weight=class_weight,
            random_state=random_state,
        )
    elif model_key == "decision_tree":
        return DecisionTreeClassifier(
            max_depth=params.get("max_depth", 5),
            min_samples_split=int(params.get("min_samples_split", 5)),
            class_weight=class_weight,
            random_state=random_state,
        )
    elif model_key == "random_forest":
        return RandomForestClassifier(
            n_estimators=int(params.get("n_estimators", 100)),
            max_depth=params.get("max_depth", 8),
            min_samples_split=int(params.get("min_samples_split", 4)),
            class_weight=class_weight,
            random_state=random_state,
            n_jobs=-1,
        )
    elif model_key == "knn":
        return KNeighborsClassifier(
            n_neighbors=int(params.get("n_neighbors", 5)),
            weights=str(params.get("weights", "uniform")),
            n_jobs=-1,
        )
    elif model_key == "svm":
        return SVC(
            C=float(params.get("C", 1.0)),
            kernel=str(params.get("kernel", "rbf")),
            probability=True,
            class_weight=class_weight,
            random_state=random_state,
        )
    elif model_key == "naive_bayes":
        return GaussianNB()
    elif model_key == "gradient_boosting":
        return GradientBoostingClassifier(
            n_estimators=int(params.get("n_estimators", 100)),
            learning_rate=float(params.get("learning_rate", 0.1)),
            max_depth=int(params.get("max_depth", 3)),
            random_state=random_state,
        )
    elif model_key == "xgboost" and HAS_XGBOOST:
        return XGBClassifier(
            n_estimators=int(params.get("n_estimators", 100)),
            learning_rate=float(params.get("learning_rate", 0.1)),
            max_depth=int(params.get("max_depth", 4)),
            eval_metric=params.get("eval_metric", "logloss"),
            random_state=random_state,
            n_jobs=-1,
        )
    else:
        raise ValueError(f"Model key '{model_key}' could not be instantiated.")

