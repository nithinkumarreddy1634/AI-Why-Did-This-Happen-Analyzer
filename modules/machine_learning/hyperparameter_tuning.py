"""Hyperparameter optimization using GridSearchCV and RandomizedSearchCV."""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.base import BaseEstimator
from modules.machine_learning.problem_detector import ProblemType
from modules.machine_learning.classification import get_classification_model
from modules.machine_learning.regression import get_regression_model


# Safe, curated hyperparameter search grids for rapid, robust tuning
DEFAULT_CLASSIFICATION_PARAM_GRIDS: Dict[str, Dict[str, List[Any]]] = {
    "logistic_regression": {
        "C": [0.01, 0.1, 1.0, 10.0],
        "max_iter": [500, 1000],
    },
    "decision_tree": {
        "max_depth": [3, 5, 8, 12, None],
        "min_samples_split": [2, 5, 10],
    },
    "random_forest": {
        "n_estimators": [50, 100, 150],
        "max_depth": [5, 8, 12, None],
        "min_samples_split": [2, 5],
    },
    "knn": {
        "n_neighbors": [3, 5, 7, 9],
        "weights": ["uniform", "distance"],
    },
    "svm": {
        "C": [0.1, 1.0, 10.0],
        "kernel": ["linear", "rbf"],
    },
    "gradient_boosting": {
        "n_estimators": [50, 100],
        "learning_rate": [0.05, 0.1, 0.2],
        "max_depth": [3, 4],
    },
}

DEFAULT_REGRESSION_PARAM_GRIDS: Dict[str, Dict[str, List[Any]]] = {
    "ridge": {
        "alpha": [0.1, 1.0, 10.0, 100.0],
    },
    "lasso": {
        "alpha": [0.01, 0.1, 1.0, 10.0],
    },
    "decision_tree": {
        "max_depth": [3, 5, 8, 12, None],
        "min_samples_split": [2, 5, 10],
    },
    "random_forest": {
        "n_estimators": [50, 100, 150],
        "max_depth": [5, 8, 12, None],
        "min_samples_split": [2, 5],
    },
    "knn": {
        "n_neighbors": [3, 5, 7, 9],
        "weights": ["uniform", "distance"],
    },
    "svm": {
        "C": [0.1, 1.0, 10.0],
        "epsilon": [0.05, 0.1, 0.2],
    },
    "gradient_boosting": {
        "n_estimators": [50, 100],
        "learning_rate": [0.05, 0.1, 0.2],
        "max_depth": [3, 4],
    },
}


@dataclass
class TuningResult:
    """Output container for hyperparameter optimization search."""
    best_estimator: BaseEstimator
    best_params: Dict[str, Any]
    best_score: float
    scoring_metric: str
    search_type: str
    cv_results_summary: pd.DataFrame
    n_splits: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "best_params": self.best_params,
            "best_score": round(self.best_score, 4),
            "scoring_metric": self.scoring_metric,
            "search_type": self.search_type,
            "n_splits": self.n_splits,
            "top_candidates": self.cv_results_summary.head(5).to_dict(orient="records"),
        }


def tune_model_hyperparameters(
    model_key: str,
    X_train_trans: np.ndarray,
    y_train: np.ndarray,
    problem_type: ProblemType,
    search_type: str = "grid",  # 'grid' or 'random'
    custom_param_grid: Optional[Dict[str, List[Any]]] = None,
    cv_folds: int = 3,
    n_iter: int = 8,
    random_state: int = 42,
) -> TuningResult:
    """Perform hyperparameter tuning strictly on training data."""
    is_classification = problem_type in (
        ProblemType.BINARY_CLASSIFICATION,
        ProblemType.MULTICLASS_CLASSIFICATION,
    )

    if is_classification:
        base_estimator = get_classification_model(model_key=model_key, random_state=random_state)
        default_grid = DEFAULT_CLASSIFICATION_PARAM_GRIDS.get(model_key, {"C": [1.0]})
        scoring = "f1_macro" if is_classification else "neg_root_mean_squared_error"
    else:
        base_estimator = get_regression_model(model_key=model_key, random_state=random_state)
        default_grid = DEFAULT_REGRESSION_PARAM_GRIDS.get(model_key, {"alpha": [1.0]})
        scoring = "neg_root_mean_squared_error"

    param_grid = custom_param_grid or default_grid

    # Cap cv_folds to sample size
    effective_cv = min(cv_folds, len(y_train))
    effective_cv = max(2, effective_cv)

    if search_type.lower() == "random":
        search = RandomizedSearchCV(
            estimator=base_estimator,
            param_distributions=param_grid,
            n_iter=min(n_iter, 20),
            cv=effective_cv,
            scoring=scoring,
            random_state=random_state,
            n_jobs=-1,
        )
    else:
        search = GridSearchCV(
            estimator=base_estimator,
            param_grid=param_grid,
            cv=effective_cv,
            scoring=scoring,
            n_jobs=-1,
        )

    search.fit(X_train_trans, y_train)

    # Format cv_results into a clean DataFrame
    df_cv = pd.DataFrame(search.cv_results_)
    display_cols = ["params", "mean_test_score", "std_test_score", "rank_test_score"]
    avail_cols = [c for c in display_cols if c in df_cv.columns]
    summary_df = df_cv[avail_cols].sort_values(by="rank_test_score").reset_index(drop=True)

    # Convert negative RMSE back to positive for clear display if regression
    best_score = float(search.best_score_)
    if not is_classification and scoring == "neg_root_mean_squared_error":
        best_score = abs(best_score)
        summary_df["mean_test_score"] = summary_df["mean_test_score"].abs()

    return TuningResult(
        best_estimator=search.best_estimator_,
        best_params=search.best_params_,
        best_score=best_score,
        scoring_metric=scoring,
        search_type="Grid Search" if search_type.lower() == "grid" else "Randomized Search",
        cv_results_summary=summary_df,
        n_splits=effective_cv,
    )

