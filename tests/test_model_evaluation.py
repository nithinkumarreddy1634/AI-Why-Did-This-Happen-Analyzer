"""Tests for Model Evaluation, Comparison, and Prediction Diagnostics."""

import pytest
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, Ridge

from modules.machine_learning.model_evaluator import (
    evaluate_classification_model,
    evaluate_regression_model,
)
from modules.machine_learning.problem_detector import ProblemType
from modules.machine_learning.model_comparison import run_model_comparison
from modules.machine_learning.prediction_engine import generate_prediction_exploration
from modules.machine_learning.ml_preprocessing import fit_ml_preprocessor, MLPreprocessorConfig


def test_classification_metrics_computation():
    y_true = np.array([0, 1, 0, 0, 1, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 1, 1, 0, 0, 1])
    y_proba = np.array([
        [0.8, 0.2],
        [0.1, 0.9],
        [0.7, 0.3],
        [0.4, 0.6],
        [0.2, 0.8],
        [0.6, 0.4],
        [0.9, 0.1],
        [0.3, 0.7],
    ])

    metrics = evaluate_classification_model(y_true, y_pred, y_proba=y_proba, labels=["No", "Yes"])
    assert metrics.accuracy == 0.75
    assert metrics.f1_macro > 0.0
    assert metrics.roc_auc is not None
    assert metrics.pr_auc is not None
    assert len(metrics.confusion_matrix) == 2
    assert "Yes" in metrics.labels


def test_regression_metrics_computation():
    y_true = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    y_pred = np.array([12.0, 18.0, 31.0, 38.0, 52.0])

    metrics = evaluate_regression_model(y_true, y_pred, n_features=2)
    assert round(metrics.mae, 2) == 1.8
    assert metrics.rmse > 0.0
    assert metrics.r2 > 0.95
    assert metrics.adjusted_r2 is not None
    assert metrics.mape is not None
    assert len(metrics.residuals) == 5


def test_multi_model_comparison_engine():
    X_train = pd.DataFrame({"a": [1, 2, 3, 4, 5, 6, 7, 8], "b": [2, 1, 4, 3, 6, 5, 8, 7]})
    y_train = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    X_test = pd.DataFrame({"a": [2, 7], "b": [3, 6]})
    y_test = np.array([0, 1])

    prep_res = fit_ml_preprocessor(
        X_train=X_train,
        numeric_features=["a", "b"],
        categorical_features=[],
        excluded_columns=[],
    )

    comp_res = run_model_comparison(
        selected_model_keys=["logistic_regression", "decision_tree"],
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        preprocessor=prep_res.preprocessor,
        problem_type=ProblemType.BINARY_CLASSIFICATION,
        primary_metric="Accuracy",
    )

    assert comp_res.comparison_table.shape[0] == 2
    assert comp_res.best_model_key in ["logistic_regression", "decision_tree"]
    assert "Accuracy" in comp_res.comparison_table.columns


def test_prediction_exploration_and_csv_export():
    X_train = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0]})
    y_train = np.array([0, 0, 1, 1])
    X_test = pd.DataFrame({"x": [1.5, 3.5]})
    y_test = np.array([0, 1])

    prep_res = fit_ml_preprocessor(
        X_train=X_train,
        numeric_features=["x"],
        categorical_features=[],
        excluded_columns=[],
    )

    clf = LogisticRegression()
    clf.fit(prep_res.preprocessor.transform(X_train), y_train)

    pipeline = Pipeline([("preprocessor", prep_res.preprocessor), ("clf", clf)])

    pred_res = generate_prediction_exploration(
        pipeline=pipeline,
        X_test=X_test,
        y_test=y_test,
        target_name="target",
        problem_type=ProblemType.BINARY_CLASSIFICATION,
        labels=["Negative", "Positive"],
    )

    assert pred_res.n_samples == 2
    assert pred_res.n_correct is not None
    assert "Actual_target" in pred_res.predictions_df.columns
    assert "Predicted_target" in pred_res.predictions_df.columns

    csv_str = pred_res.to_csv()
    assert "Actual_target" in csv_str
    assert "Predicted_target" in csv_str

