"""Unit tests for LIME local regression explanations."""

import pytest
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor

from modules.explainability.lime_explainer import LIMEConfig, LIMEExplainerWrapper
from modules.explainability.lime_regression import explain_regression_instance


@pytest.fixture
def regression_setup():
    np.random.seed(42)
    X = np.random.randn(50, 4)
    # y = 3*x0 - 2*x1 + 0.5*x2 + noise
    y = 3.0 * X[:, 0] - 2.0 * X[:, 1] + 0.5 * X[:, 2] + np.random.normal(0, 0.1, 50)
    fnames = ["sqft", "bedrooms", "age", "dist_city"]

    reg = Ridge(alpha=1.0, random_state=42)
    reg.fit(X, y)

    wrapper = LIMEExplainerWrapper(
        estimator=reg,
        feature_names=fnames,
        is_classification=False,
        config=LIMEConfig(num_samples=300, random_state=42),
    )
    wrapper.initialize_explainer(X)
    return wrapper, X, y


def test_regression_explanation_numpy_input(regression_setup):
    wrapper, X, y = regression_setup
    sample_idx = 0
    row = X[sample_idx]

    res = explain_regression_instance(
        wrapper=wrapper,
        instance_row=row,
        sample_index=sample_idx,
        num_features=4,
        num_samples=300,
        actual_outcome=float(y[sample_idx]),
    )

    assert res.sample_index == sample_idx
    assert res.is_classification is False
    assert res.target_class_idx == 0
    assert len(res.contributions_df) > 0
    assert "Feature" in res.contributions_df.columns
    assert "LIME_Weight" in res.contributions_df.columns
    assert "Abs_Weight" in res.contributions_df.columns
    assert "Condition" in res.contributions_df.columns

    # Verify surrogate fidelity and predictions
    assert isinstance(res.local_fidelity_r2, float)
    assert res.local_surrogate_prediction is not None
    assert res.predicted_outcome is not None
    assert res.actual_outcome is not None

    # Check approximation gap is non-negative float
    approx_gap = abs(res.predicted_outcome - res.local_surrogate_prediction)
    assert approx_gap >= 0.0

    # Serialization test
    d = res.to_dict()
    assert d["is_classification"] is False
    assert "contributions" in d


def test_regression_explanation_series_input(regression_setup):
    wrapper, X, y = regression_setup
    sample_idx = 3
    series_row = pd.Series(X[sample_idx], index=["sqft", "bedrooms", "age", "dist_city"])

    res = explain_regression_instance(
        wrapper=wrapper,
        instance_row=series_row,
        sample_index=sample_idx,
        num_features=3,
        num_samples=300,
    )

    assert res.sample_index == sample_idx
    assert len(res.contributions_df) <= 3
    assert len(res.narrative_summary) > 0


def test_uninitialized_regression_explainer_raises():
    reg = LinearRegression()
    wrapper = LIMEExplainerWrapper(
        estimator=reg,
        feature_names=["a", "b"],
        is_classification=False,
    )
    with pytest.raises(RuntimeError, match="LIME explainer is not initialized"):
        explain_regression_instance(
            wrapper=wrapper,
            instance_row=np.array([1.0, 2.0]),
        )
