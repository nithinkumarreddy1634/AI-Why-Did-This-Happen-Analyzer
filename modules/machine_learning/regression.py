"""Regression algorithms, configurations, and estimators."""

from typing import Dict, Any, Optional
from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR

# Optional XGBoost import
try:
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except (ImportError, Exception):
    HAS_XGBOOST = False
    XGBRegressor = None


REGRESSION_MODELS_REGISTRY: Dict[str, Dict[str, Any]] = {
    "linear_regression": {
        "display_name": "Linear Regression (OLS)",
        "description": "Standard ordinary least squares linear regression without regularization.",
        "supports_feature_importance": True,
        "is_tree_based": False,
        "default_params": {},
    },
    "ridge": {
        "display_name": "Ridge Regression (L2)",
        "description": "Linear regression with L2 Tikhonov penalty shrinkage to handle collinearity.",
        "supports_feature_importance": True,
        "is_tree_based": False,
        "default_params": {"alpha": 1.0},
    },
    "lasso": {
        "display_name": "Lasso Regression (L1)",
        "description": "Linear regression with L1 norm penalty enforcing sparse feature selection.",
        "supports_feature_importance": True,
        "is_tree_based": False,
        "default_params": {"alpha": 1.0},
    },
    "decision_tree": {
        "display_name": "Decision Tree Regressor",
        "description": "Non-linear regression tree splitting partitions to minimize variance/MSE.",
        "supports_feature_importance": True,
        "is_tree_based": True,
        "default_params": {"max_depth": 5, "min_samples_split": 5},
    },
    "random_forest": {
        "display_name": "Random Forest Regressor",
        "description": "Ensemble of regression trees averaging predictions to prevent overfitting.",
        "supports_feature_importance": True,
        "is_tree_based": True,
        "default_params": {"n_estimators": 100, "max_depth": 8, "min_samples_split": 4},
    },
    "knn": {
        "display_name": "K-Nearest Neighbors Regressor",
        "description": "Non-parametric regression averaging continuous targets of k closest points.",
        "supports_feature_importance": False,
        "is_tree_based": False,
        "default_params": {"n_neighbors": 5, "weights": "uniform"},
    },
    "svm": {
        "display_name": "Support Vector Regressor (SVR)",
        "description": "Epsilon-insensitive margin regression fitting non-linear boundary with RBF kernel.",
        "supports_feature_importance": False,
        "is_tree_based": False,
        "default_params": {"C": 1.0, "epsilon": 0.1, "kernel": "rbf"},
    },
    "gradient_boosting": {
        "display_name": "Gradient Boosting Regressor",
        "description": "Sequential boosting ensemble optimizing least squares loss iteratively.",
        "supports_feature_importance": True,
        "is_tree_based": True,
        "default_params": {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 3},
    },
}

if HAS_XGBOOST:
    REGRESSION_MODELS_REGISTRY["xgboost"] = {
        "display_name": "XGBoost Regressor",
        "description": "High-performance gradient boosting decision tree framework for regression.",
        "supports_feature_importance": True,
        "is_tree_based": True,
        "default_params": {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 4},
    }


def get_regression_model(
    model_key: str,
    hyperparameters: Optional[Dict[str, Any]] = None,
    random_state: int = 42,
) -> BaseEstimator:
    """Instantiate a regression model with specified hyperparameters."""
    if model_key not in REGRESSION_MODELS_REGISTRY:
        raise ValueError(
            f"Unsupported regression model '{model_key}'. "
            f"Available: {list(REGRESSION_MODELS_REGISTRY.keys())}"
        )

    meta = REGRESSION_MODELS_REGISTRY[model_key]
    params = dict(meta["default_params"])
    if hyperparameters:
        params.update(hyperparameters)

    if model_key == "linear_regression":
        return LinearRegression()
    elif model_key == "ridge":
        return Ridge(alpha=float(params.get("alpha", 1.0)), random_state=random_state)
    elif model_key == "lasso":
        return Lasso(alpha=float(params.get("alpha", 1.0)), random_state=random_state)
    elif model_key == "decision_tree":
        return DecisionTreeRegressor(
            max_depth=params.get("max_depth", 5),
            min_samples_split=int(params.get("min_samples_split", 5)),
            random_state=random_state,
        )
    elif model_key == "random_forest":
        return RandomForestRegressor(
            n_estimators=int(params.get("n_estimators", 100)),
            max_depth=params.get("max_depth", 8),
            min_samples_split=int(params.get("min_samples_split", 4)),
            random_state=random_state,
            n_jobs=-1,
        )
    elif model_key == "knn":
        return KNeighborsRegressor(
            n_neighbors=int(params.get("n_neighbors", 5)),
            weights=str(params.get("weights", "uniform")),
            n_jobs=-1,
        )
    elif model_key == "svm":
        return SVR(
            C=float(params.get("C", 1.0)),
            epsilon=float(params.get("epsilon", 0.1)),
            kernel=str(params.get("kernel", "rbf")),
        )
    elif model_key == "gradient_boosting":
        return GradientBoostingRegressor(
            n_estimators=int(params.get("n_estimators", 100)),
            learning_rate=float(params.get("learning_rate", 0.1)),
            max_depth=int(params.get("max_depth", 3)),
            random_state=random_state,
        )
    elif model_key == "xgboost" and HAS_XGBOOST:
        return XGBRegressor(
            n_estimators=int(params.get("n_estimators", 100)),
            learning_rate=float(params.get("learning_rate", 0.1)),
            max_depth=int(params.get("max_depth", 4)),
            random_state=random_state,
            n_jobs=-1,
        )
    else:
        raise ValueError(f"Model key '{model_key}' could not be instantiated.")

