"""Unit tests for Local SHAP Individual Prediction Explanations and Waterfall Breakdowns."""

import pytest
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from modules.explainability.shap_explainer import SHAPExplainerWrapper
from modules.explainability.shap_local import explain_individual_prediction


@pytest.fixture
def calc_res_fixture():
    X = np.array([
        [1.0, 50.0],
        [2.0, 30.0],
        [8.0, 90.0],
        [9.0, 85.0],
    ])
    y = np.array([0, 0, 1, 1])
    fnames = ["feature_x", "feature_y"]

    rf = RandomForestClassifier(n_estimators=10, random_state=42)
    rf.fit(X, y)

    wrapper = SHAPExplainerWrapper(
        estimator=rf,
        feature_names=fnames,
        is_classification=True,
        class_labels=["Inactive", "Active"],
    )
    wrapper.initialize_explainer(X)
    return wrapper.explain(X, target_class_idx=1)


def test_individual_prediction_explanation(calc_res_fixture):
    local_res = explain_individual_prediction(
        calc_result=calc_res_fixture,
        sample_index=0,
        actual_outcomes=np.array([0, 0, 1, 1]),
        predicted_labels=["Inactive", "Inactive", "Active", "Active"],
    )

    assert local_res.sample_index == 0
    assert local_res.target_class_name == "Active"
    assert local_res.actual_outcome == 0
    assert local_res.predicted_label == "Inactive"
    assert len(local_res.contributions_df) == 2

    # Verify additivity audit
    audit = local_res.additivity_audit
    assert "base_value" in audit
    assert "shap_sum" in audit
    assert audit["is_additive"] is True

    # Check narrative contains record index
    assert "#0" in local_res.narrative_summary


def test_local_out_of_bounds_index(calc_res_fixture):
    with pytest.raises(IndexError):
        explain_individual_prediction(calc_res_fixture, sample_index=999)

