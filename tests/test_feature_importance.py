"""Tests for Feature Importance, Hyperparameter Tuning, Insights, and Registry."""

import pytest
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from modules.machine_learning.feature_importance import extract_feature_importance
from modules.machine_learning.hyperparameter_tuning import tune_model_hyperparameters
from modules.machine_learning.problem_detector import ProblemType, TargetSummary
from modules.machine_learning.ml_insights import generate_ml_insights
from modules.machine_learning.model_comparison import ModelComparisonResult
from modules.machine_learning.model_manager import ModelRegistry


def test_tree_and_permutation_and_linear_importance():
    X = np.array([
        [1.0, 10.0],
        [2.0, 20.0],
        [3.0, 30.0],
        [4.0, 40.0],
        [5.0, 50.0],
        [6.0, 60.0],
    ])
    y = np.array([0, 0, 0, 1, 1, 1])
    fnames = ["feat_low", "feat_high"]

    # 1. Tree Importance
    rf = RandomForestClassifier(n_estimators=10, random_state=42)
    rf.fit(X, y)
    res_tree = extract_feature_importance(rf, fnames, preferred_method="tree")
    assert res_tree.method == "tree"
    assert len(res_tree.top_features) == 2
    assert res_tree.is_causal is False
    assert "not prove causal" in res_tree.disclaimer.lower()

    # 2. Linear Coefficients
    lr = LogisticRegression(random_state=42)
    lr.fit(X, y)
    res_lin = extract_feature_importance(lr, fnames, preferred_method="coefficients")
    assert res_lin.method == "coefficients"
    assert "Normalized_Importance" in res_lin.importance_df.columns

    # 3. Permutation Importance
    res_perm = extract_feature_importance(rf, fnames, X_test=X, y_test=y, preferred_method="permutation")
    assert res_perm.method == "permutation"
    assert len(res_perm.importance_df) == 2


def test_hyperparameter_tuning_grid_and_random():
    X = np.random.randn(30, 4)
    y = np.array([0, 1] * 15)

    # Grid Search
    tune_res = tune_model_hyperparameters(
        model_key="decision_tree",
        X_train_trans=X,
        y_train=y,
        problem_type=ProblemType.BINARY_CLASSIFICATION,
        search_type="grid",
        custom_param_grid={"max_depth": [2, 3], "min_samples_split": [2, 4]},
        cv_folds=2,
    )
    assert tune_res.best_estimator is not None
    assert "max_depth" in tune_res.best_params
    assert tune_res.search_type == "Grid Search"

    # Randomized Search
    tune_rnd = tune_model_hyperparameters(
        model_key="decision_tree",
        X_train_trans=X,
        y_train=y,
        problem_type=ProblemType.BINARY_CLASSIFICATION,
        search_type="random",
        custom_param_grid={"max_depth": [2, 3, 4], "min_samples_split": [2, 4, 6]},
        cv_folds=2,
        n_iter=3,
    )
    assert tune_rnd.search_type == "Randomized Search"


def test_ml_insights_generation():
    df_comp = pd.DataFrame([
        {
            "Model Key": "random_forest",
            "Model": "Random Forest",
            "Accuracy": 0.90,
            "F1-Score (Macro)": 0.88,
            "ROC-AUC": 0.92,
            "CV Mean": 0.87,
        },
        {
            "Model Key": "logistic_regression",
            "Model": "Logistic Regression",
            "Accuracy": 0.80,
            "F1-Score (Macro)": 0.76,
            "ROC-AUC": 0.81,
            "CV Mean": 0.77,
        },
    ])

    comp_res = ModelComparisonResult(
        comparison_table=df_comp,
        results_by_model={},
        best_model_key="random_forest",
        primary_metric="F1-Score (Macro)",
        problem_type=ProblemType.BINARY_CLASSIFICATION,
        models_evaluated=["random_forest", "logistic_regression"],
    )

    target_sum = TargetSummary(
        target_name="churn",
        problem_type=ProblemType.BINARY_CLASSIFICATION,
        data_type="int64",
        n_samples=100,
        n_missing=0,
        n_unique=2,
        is_imbalanced=True,
        imbalance_ratio=4.5,
    )

    insights = generate_ml_insights(comp_res, target_sum)
    assert len(insights) >= 3
    # Verify non-causality alert exists
    causal_alerts = [i for i in insights if i.category == "Causality Alert"]
    assert len(causal_alerts) == 1


def test_model_registry_save_and_retrieve():
    registry = ModelRegistry()
    pipeline = Pipeline([("clf", LogisticRegression())])

    art = registry.register_model(
        model_name="Logistic Regression",
        pipeline=pipeline,
        problem_type=ProblemType.BINARY_CLASSIFICATION,
        target_name="churn",
        feature_names_in=["tenure", "charges"],
        feature_names_out=["tenure", "charges"],
        metrics={"accuracy": 0.85},
    )

    assert art.model_id in registry.models
    retrieved = registry.get_model(art.model_id)
    assert retrieved is not None
    assert retrieved.model_name == "Logistic Regression"
    assert len(registry.list_models()) == 1

    # Safe bytes serialization
    model_bytes = registry.export_model_bytes(art.model_id)
    assert model_bytes is not None

    imported = registry.import_model_bytes(model_bytes)
    assert imported.model_name == "Logistic Regression"

