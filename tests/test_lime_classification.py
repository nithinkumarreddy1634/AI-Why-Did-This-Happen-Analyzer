"""Unit tests for LIME local classification explanations."""

import pytest
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from modules.explainability.lime_explainer import LIMEConfig, LIMEExplainerWrapper
from modules.explainability.lime_classification import explain_classification_instance


@pytest.fixture
def binary_clf_setup():
    np.random.seed(42)
    X = np.random.randn(60, 4)
    # Target linearly dependent on feature 0 and 1
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    fnames = ["feature_0", "feature_1", "feature_2", "feature_3"]

    clf = LogisticRegression(random_state=42)
    clf.fit(X, y)

    wrapper = LIMEExplainerWrapper(
        estimator=clf,
        feature_names=fnames,
        is_classification=True,
        class_names=["Class 0", "Class 1"],
        config=LIMEConfig(num_samples=300, random_state=42),
    )
    wrapper.initialize_explainer(X)
    return wrapper, X, y


@pytest.fixture
def multiclass_clf_setup():
    np.random.seed(42)
    X = np.random.randn(60, 3)
    y = np.array([0, 1, 2] * 20)
    fnames = ["f0", "f1", "f2"]

    clf = RandomForestClassifier(n_estimators=10, random_state=42)
    clf.fit(X, y)

    wrapper = LIMEExplainerWrapper(
        estimator=clf,
        feature_names=fnames,
        is_classification=True,
        class_names=["Low", "Medium", "High"],
        config=LIMEConfig(num_samples=300, random_state=42),
    )
    wrapper.initialize_explainer(X)
    return wrapper, X, y


def test_binary_classification_explanation(binary_clf_setup):
    wrapper, X, y = binary_clf_setup
    sample_idx = 2
    row = X[sample_idx]

    res = explain_classification_instance(
        wrapper=wrapper,
        instance_row=row,
        sample_index=sample_idx,
        target_class_idx=1,
        num_features=4,
        num_samples=300,
        actual_outcome=y[sample_idx],
    )

    assert res.sample_index == sample_idx
    assert res.is_classification is True
    assert res.target_class_idx == 1
    assert res.target_class_name == "Class 1"
    assert len(res.contributions_df) > 0
    assert "Feature" in res.contributions_df.columns
    assert "LIME_Weight" in res.contributions_df.columns
    assert "Direction" in res.contributions_df.columns
    assert "Condition" in res.contributions_df.columns

    # Check fidelity metrics
    assert isinstance(res.local_fidelity_r2, float)
    assert res.local_surrogate_prediction is not None
    assert len(res.narrative_summary) > 0

    # Serialization test
    d = res.to_dict()
    assert d["sample_index"] == sample_idx
    assert "contributions" in d


def test_multiclass_classification_explanation(multiclass_clf_setup):
    wrapper, X, y = multiclass_clf_setup
    sample_idx = 5
    row = X[sample_idx]

    # Explain target class 2 ("High")
    res = explain_classification_instance(
        wrapper=wrapper,
        instance_row=row,
        sample_index=sample_idx,
        target_class_idx=2,
        num_features=3,
        num_samples=300,
        actual_outcome=y[sample_idx],
    )

    assert res.sample_index == sample_idx
    assert res.target_class_idx == 2
    assert res.target_class_name == "High"
    assert res.is_classification is True
    assert len(res.contributions_df) <= 3
    assert len(res.predicted_probabilities) == 3


def test_uninitialized_explainer_raises():
    clf = LogisticRegression()
    wrapper = LIMEExplainerWrapper(
        estimator=clf,
        feature_names=["a", "b"],
        is_classification=True,
    )
    # Do not call initialize_explainer
    with pytest.raises(RuntimeError, match="LIME explainer is not initialized"):
        explain_classification_instance(
            wrapper=wrapper,
            instance_row=np.array([1.0, 2.0]),
        )
