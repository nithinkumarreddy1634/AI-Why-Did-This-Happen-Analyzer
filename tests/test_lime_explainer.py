"""Unit tests for LIME explainer initialization, configuration, and prediction routing."""

import pytest
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression

from modules.explainability.lime_explainer import (
    LIMEConfig,
    LIMEExplainerWrapper,
    LIMEExplainerMetadata,
)


def test_lime_explainer_initialization_classification():
    X_train = np.array([
        [1.0, 10.0],
        [2.0, 20.0],
        [3.0, 30.0],
        [4.0, 40.0],
        [5.0, 50.0],
    ])
    y_train = np.array([0, 0, 1, 1, 1])
    fnames = ["tenure", "charges"]

    clf = LogisticRegression(random_state=42)
    clf.fit(X_train, y_train)

    wrapper = LIMEExplainerWrapper(
        estimator=clf,
        feature_names=fnames,
        is_classification=True,
        class_names=["Retained", "Churned"],
        config=LIMEConfig(num_samples=250, random_state=42),
    )
    wrapper.initialize_explainer(X_train)

    assert wrapper.explainer is not None
    assert wrapper.metadata.mode == "classification"
    assert wrapper.metadata.n_features == 2
    assert wrapper.metadata.n_background_samples == 5
    assert wrapper.metadata.class_names == ["Retained", "Churned"]

    # Verify predict_fn returns 2D probabilities summing to 1.0
    pred_fn = wrapper.get_predict_fn()
    probs = pred_fn(X_train[:2])
    assert probs.shape == (2, 2)
    np.testing.assert_allclose(np.sum(probs, axis=1), [1.0, 1.0], atol=1e-5)


def test_lime_explainer_initialization_regression():
    X_train = np.array([
        [10.0, 5.0],
        [20.0, 10.0],
        [30.0, 15.0],
    ])
    y_train = np.array([100.0, 200.0, 300.0])
    fnames = ["x1", "x2"]

    reg = LinearRegression()
    reg.fit(X_train, y_train)

    wrapper = LIMEExplainerWrapper(
        estimator=reg,
        feature_names=fnames,
        is_classification=False,
    )
    wrapper.initialize_explainer(X_train)

    assert wrapper.explainer is not None
    assert wrapper.metadata.mode == "regression"
    assert wrapper.metadata.class_names is None

    pred_fn = wrapper.get_predict_fn()
    preds = pred_fn(X_train[:2])
    assert len(preds) == 2


def test_lime_reproducibility_with_seed():
    X_train = np.random.randn(20, 3)
    y_train = np.array([0, 1] * 10)
    fnames = ["a", "b", "c"]

    rf = RandomForestClassifier(n_estimators=10, random_state=42)
    rf.fit(X_train, y_train)

    w1 = LIMEExplainerWrapper(
        estimator=rf,
        feature_names=fnames,
        is_classification=True,
        config=LIMEConfig(num_samples=300, random_state=99),
    )
    w1.initialize_explainer(X_train)
    exp1 = w1.explainer.explain_instance(X_train[0], w1.get_predict_fn(), num_features=3, num_samples=300)

    w2 = LIMEExplainerWrapper(
        estimator=rf,
        feature_names=fnames,
        is_classification=True,
        config=LIMEConfig(num_samples=300, random_state=99),
    )
    w2.initialize_explainer(X_train)
    exp2 = w2.explainer.explain_instance(X_train[0], w2.get_predict_fn(), num_features=3, num_samples=300)

    # Weights should be identical with matching random_state
    weights1 = [w for _, w in exp1.as_list()]
    weights2 = [w for _, w in exp2.as_list()]
    np.testing.assert_allclose(weights1, weights2, atol=1e-5)

