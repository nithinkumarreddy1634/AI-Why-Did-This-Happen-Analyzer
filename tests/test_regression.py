"""Tests for Regression Models and Estimators."""

import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import make_regression

from modules.machine_learning.regression import (
    REGRESSION_MODELS_REGISTRY,
    get_regression_model,
    HAS_XGBOOST,
)
from modules.machine_learning.model_evaluator import evaluate_regression_model


@pytest.fixture
def regression_data():
    X_raw, y = make_regression(
        n_samples=100,
        n_features=5,
        n_informative=3,
        noise=0.1,
        random_state=42,
    )
    df = pd.DataFrame(X_raw, columns=[f"feat_{i}" for i in range(5)])
    return df, y


def test_all_regression_models_instantiation():
    for model_key in REGRESSION_MODELS_REGISTRY.keys():
        if model_key == "xgboost" and not HAS_XGBOOST:
            continue
        model = get_regression_model(model_key, random_state=42)
        assert model is not None


def test_regression_fit_and_predict(regression_data):
    df, y = regression_data
    X_train, X_test = df.iloc[:80], df.iloc[80:]
    y_train, y_test = y[:80], y[80:]

    core_models = [
        "linear_regression",
        "ridge",
        "lasso",
        "decision_tree",
        "random_forest",
        "knn",
        "svm",
        "gradient_boosting",
    ]

    for model_key in core_models:
        model = get_regression_model(model_key, random_state=42)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        
        assert len(preds) == len(X_test)
        metrics = evaluate_regression_model(y_test, preds, n_features=X_train.shape[1])
        assert metrics.rmse >= 0.0
        assert metrics.mae >= 0.0
        assert metrics.residuals is not None
        assert len(metrics.residuals) == len(X_test)


def test_regression_reproducibility(regression_data):
    df, y = regression_data
    X_train, X_test = df.iloc[:70], df.iloc[70:]
    y_train, y_test = y[:70], y[70:]

    rf1 = get_regression_model("random_forest", random_state=99)
    rf1.fit(X_train, y_train)
    p1 = rf1.predict(X_test)

    rf2 = get_regression_model("random_forest", random_state=99)
    rf2.fit(X_train, y_train)
    p2 = rf2.predict(X_test)

    np.testing.assert_allclose(p1, p2, rtol=1e-5)


def test_small_sample_size_regression():
    X = pd.DataFrame(np.random.randn(15, 3), columns=["a", "b", "c"])
    y = np.random.randn(15) * 10

    model = get_regression_model("ridge", random_state=42)
    model.fit(X, y)
    preds = model.predict(X)
    assert len(preds) == 15

