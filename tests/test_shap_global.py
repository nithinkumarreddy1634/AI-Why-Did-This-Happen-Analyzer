"""Unit tests for Global SHAP Feature Importance and population-level attributions."""

import pytest
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from modules.explainability.shap_explainer import SHAPExplainerWrapper
from modules.explainability.shap_global import compute_global_shap_importance


@pytest.fixture
def fitted_shap_calculation():
    X = np.array([
        [1.0, 10.0, 100.0],
        [2.0, 20.0, 200.0],
        [3.0, 30.0, 300.0],
        [4.0, 40.0, 400.0],
        [5.0, 50.0, 500.0],
        [6.0, 60.0, 600.0],
    ])
    y = np.array([0, 0, 0, 1, 1, 1])
    fnames = ["small_val", "mid_val", "large_val"]

    rf = RandomForestClassifier(n_estimators=15, random_state=42)
    rf.fit(X, y)

    wrapper = SHAPExplainerWrapper(
        estimator=rf,
        feature_names=fnames,
        is_classification=True,
    )
    wrapper.initialize_explainer(X)
    return wrapper.explain(X, target_class_idx=1)


def test_global_shap_importance_calculation(fitted_shap_calculation):
    global_res = compute_global_shap_importance(fitted_shap_calculation)

    assert global_res.total_explained_samples == 6
    assert len(global_res.top_features) == 3
    assert not global_res.importance_df.empty

    df_imp = global_res.importance_df
    assert "Mean_Abs_SHAP" in df_imp.columns
    assert "Contribution_Pct" in df_imp.columns
    assert "Direction_Tendency" in df_imp.columns

    # Verify descending sort
    mean_shaps = df_imp["Mean_Abs_SHAP"].tolist()
    assert mean_shaps == sorted(mean_shaps, reverse=True)

    # Check percentage sum equals approximately 100%
    pct_sum = df_imp["Contribution_Pct"].sum()
    assert abs(pct_sum - 100.0) < 1.0


def test_global_shap_top_n_filtering(fitted_shap_calculation):
    global_res = compute_global_shap_importance(fitted_shap_calculation, top_n=2)
    assert len(global_res.top_features) == 2

