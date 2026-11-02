"""Unit tests for SHAP explainer selection, initialization, and mathematical additivity."""

import pytest
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighborsClassifier

from modules.explainability.explainer_selector import identify_compatible_explainer
from modules.explainability.shap_explainer import SHAPExplainerWrapper
from utils.shap_helpers import check_shap_additivity, extract_scalar_base_value


def test_explainer_selection_mapping():
    # 1. Tree models -> TreeExplainer
    rf = RandomForestClassifier(n_estimators=10, random_state=42)
    meta_rf = identify_compatible_explainer(rf)
    assert meta_rf.explainer_type == "TreeExplainer"
    assert meta_rf.is_supported is True
    assert meta_rf.supports_interactions is True

    # 2. Linear models -> LinearExplainer
    lr = LogisticRegression(random_state=42)
    meta_lr = identify_compatible_explainer(lr)
    assert meta_lr.explainer_type == "LinearExplainer"
    assert meta_lr.is_supported is True

    # 3. Non-linear, non-tree -> KernelExplainer
    knn = KNeighborsClassifier()
    meta_knn = identify_compatible_explainer(knn)
    assert meta_knn.explainer_type == "KernelExplainer"
    assert meta_knn.is_approximate is True

    # 4. None / Unsupported
    meta_none = identify_compatible_explainer(None)
    assert meta_none.is_supported is False


def test_tree_explainer_binary_classification_and_additivity():
    X = np.array([
        [1.0, 5.0, 10.0],
        [2.0, 3.0, 12.0],
        [3.0, 6.0, 9.0],
        [4.0, 1.0, 15.0],
        [5.0, 4.0, 11.0],
        [6.0, 2.0, 14.0],
    ])
    y = np.array([0, 0, 0, 1, 1, 1])
    fnames = ["feat_a", "feat_b", "feat_c"]

    rf = RandomForestClassifier(n_estimators=20, random_state=42)
    rf.fit(X, y)

    wrapper = SHAPExplainerWrapper(
        estimator=rf,
        feature_names=fnames,
        is_classification=True,
        class_labels=["No", "Yes"],
    )
    wrapper.initialize_explainer(X)

    calc_res = wrapper.explain(X, target_class_idx=1)
    assert calc_res.shap_values.shape == (6, 3)
    assert len(calc_res.feature_names) == 3
    assert calc_res.target_class_name == "Yes"

    # Verify additivity for first observation
    audit = check_shap_additivity(
        base_value=calc_res.base_value,
        shap_values=calc_res.shap_values[0],
        model_prediction=calc_res.model_predictions[0],
    )
    assert audit["is_additive"] is True


def test_linear_explainer_regression_and_additivity():
    X = np.array([
        [10.0, 2.0],
        [20.0, 4.0],
        [30.0, 6.0],
        [40.0, 8.0],
        [50.0, 10.0],
    ])
    y = np.array([24.0, 48.0, 72.0, 96.0, 120.0])
    fnames = ["feature_x", "feature_z"]

    lin = LinearRegression()
    lin.fit(X, y)

    wrapper = SHAPExplainerWrapper(
        estimator=lin,
        feature_names=fnames,
        is_classification=False,
    )
    wrapper.initialize_explainer(X)

    calc_res = wrapper.explain(X)
    assert calc_res.shap_values.shape == (5, 2)

    # Check that sum of SHAP + base value equals model prediction
    audit = check_shap_additivity(
        base_value=calc_res.base_value,
        shap_values=calc_res.shap_values[0],
        model_prediction=calc_res.model_predictions[0],
        tolerance=1e-3,
    )
    assert audit["is_additive"] is True


def test_extract_scalar_base_value():
    assert extract_scalar_base_value(0.45) == 0.45
    assert extract_scalar_base_value([0.2, 0.8], class_idx=1) == 0.8
    assert extract_scalar_base_value(np.array([0.3, 0.7]), class_idx=0) == 0.3
    assert extract_scalar_base_value(np.array([[0.1, 0.9]]), class_idx=1) == 0.9

