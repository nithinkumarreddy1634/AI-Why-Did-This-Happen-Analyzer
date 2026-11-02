"""Tests for Classification Models and Estimators."""

import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

from modules.machine_learning.classification import (
    CLASSIFICATION_MODELS_REGISTRY,
    get_classification_model,
    HAS_XGBOOST,
)
from modules.machine_learning.problem_detector import ProblemType
from modules.machine_learning.ml_preprocessing import fit_ml_preprocessor, MLPreprocessorConfig
from modules.machine_learning.model_evaluator import evaluate_classification_model


@pytest.fixture
def binary_data():
    X_raw, y = make_classification(
        n_samples=100,
        n_features=6,
        n_informative=4,
        n_classes=2,
        weights=[0.85, 0.15],
        random_state=42,
    )
    df = pd.DataFrame(X_raw, columns=[f"feat_{i}" for i in range(6)])
    return df, y


@pytest.fixture
def multiclass_data():
    X_raw, y = make_classification(
        n_samples=90,
        n_features=6,
        n_informative=4,
        n_redundant=1,
        n_classes=3,
        random_state=42,
    )
    df = pd.DataFrame(X_raw, columns=[f"feat_{i}" for i in range(6)])
    return df, y


def test_all_classification_models_instantiation():
    for model_key in CLASSIFICATION_MODELS_REGISTRY.keys():
        if model_key == "xgboost" and not HAS_XGBOOST:
            continue
        model = get_classification_model(model_key, random_state=42)
        assert model is not None


def test_binary_classification_fit_and_predict(binary_data):
    df, y = binary_data
    X_train, X_test = df.iloc[:80], df.iloc[80:]
    y_train, y_test = y[:80], y[80:]

    core_models = ["logistic_regression", "decision_tree", "random_forest", "knn", "svm", "naive_bayes", "gradient_boosting"]

    for model_key in core_models:
        model = get_classification_model(model_key, random_state=42, handle_imbalance=True)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        
        assert len(preds) == len(X_test)
        assert set(preds).issubset({0, 1})

        # Evaluate
        y_proba = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None
        metrics = evaluate_classification_model(y_test, preds, y_proba=y_proba)
        assert 0.0 <= metrics.accuracy <= 1.0
        assert 0.0 <= metrics.f1_macro <= 1.0


def test_multiclass_classification_fit_and_predict(multiclass_data):
    df, y = multiclass_data
    X_train, X_test = df.iloc[:70], df.iloc[70:]
    y_train, y_test = y[:70], y[70:]

    for model_key in ["logistic_regression", "random_forest", "gradient_boosting"]:
        model = get_classification_model(model_key, random_state=42)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        
        assert len(preds) == len(X_test)
        assert set(preds).issubset({0, 1, 2})


def test_reproducibility_with_seed(binary_data):
    df, y = binary_data
    X_train, X_test = df.iloc[:70], df.iloc[70:]
    y_train, y_test = y[:70], y[70:]

    rf1 = get_classification_model("random_forest", random_state=123)
    rf1.fit(X_train, y_train)
    p1 = rf1.predict(X_test)

    rf2 = get_classification_model("random_forest", random_state=123)
    rf2.fit(X_train, y_train)
    p2 = rf2.predict(X_test)

    np.testing.assert_array_equal(p1, p2)


def test_small_sample_size_classification():
    # Only 15 rows
    X = pd.DataFrame(np.random.randn(15, 3), columns=["a", "b", "c"])
    y = np.array([0, 1] * 7 + [0])

    model = get_classification_model("decision_tree", random_state=42)
    model.fit(X, y)
    preds = model.predict(X)
    assert len(preds) == 15
