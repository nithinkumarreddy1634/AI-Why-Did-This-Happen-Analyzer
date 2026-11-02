"""End-to-end integration tests for SHAP Explainability Engine."""

import pytest
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor

from modules.explainability.explanation_manager import SHAPExplanationManager
from modules.explainability.shap_dependence import compute_feature_dependence_data
from modules.explainability.shap_interactions import compute_shap_interactions
from modules.explainability.shap_insights import generate_shap_narrative_insights
from modules.machine_learning.ml_preprocessing import fit_ml_preprocessor, MLPreprocessorConfig


@pytest.fixture
def sample_dataset():
    df = pd.DataFrame({
        "tenure": [1.0, 5.0, 12.0, 24.0, 36.0, 48.0, 60.0, 72.0],
        "monthly_charges": [20.0, 45.0, 60.0, 75.0, 80.0, 85.0, 90.0, 105.0],
        "contract": ["month", "month", "month", "one_year", "one_year", "two_year", "two_year", "two_year"],
        "churn": [1, 1, 1, 0, 0, 0, 0, 0],
    })
    return df


def test_end_to_end_shap_workflow(sample_dataset):
    df = sample_dataset
    target_col = "churn"
    num_cols = ["tenure", "monthly_charges"]
    cat_cols = ["contract"]

    # Preprocessing
    prep_res = fit_ml_preprocessor(
        X_train=df,
        numeric_features=num_cols,
        categorical_features=cat_cols,
        excluded_columns=[],
    )

    X_trans = prep_res.preprocessor.transform(df)
    y = df[target_col].to_numpy()

    # Train model
    clf = RandomForestClassifier(n_estimators=15, random_state=42)
    clf.fit(X_trans, y)

    # SHAP Explanation Manager
    manager = SHAPExplanationManager()
    wrapper = manager.build_explainer_wrapper(
        estimator=clf,
        feature_names=prep_res.feature_names_out,
        X_train_trans=X_trans,
        is_classification=True,
        class_labels=["No Churn", "Churn"],
        model_id="test_rf",
    )

    calc_res = manager.compute_explanations(
        wrapper=wrapper,
        X_eval_trans=X_trans,
        target_class_idx=1,
    )

    assert calc_res.n_samples == 8
    assert calc_res.n_features == len(prep_res.feature_names_out)

    # 1. Global Importance
    global_res = manager.extract_global_importance(calc_res)
    assert len(global_res.top_features) > 0
    assert global_res.total_explained_samples == 8

    # 2. Local Explanation
    local_res = manager.extract_local_explanation(calc_res, sample_index=0)
    assert local_res.sample_index == 0
    assert local_res.additivity_audit["is_additive"] is True

    # 3. Dependence Profile
    dep_res = manager.extract_dependence_profile(calc_res, primary_feature="tenure")
    assert dep_res.primary_feature == "tenure"
    assert len(dep_res.feature_values) == 8
    assert len(dep_res.shap_values) == 8

    # 4. Pairwise Interactions
    inter_res = manager.extract_interactions(wrapper, calc_res, max_samples=8, top_k=4)
    assert inter_res.is_supported is True
    assert len(inter_res.top_interacting_pairs) > 0

    # 5. Narrative Insights
    insights = manager.generate_narrative(global_res, local_res)
    assert len(insights) >= 3
    # Check for Causality Alert
    alert_categories = [i.category for i in insights]
    assert "Causality Alert" in alert_categories

    # 6. Cache clearing
    manager.clear_cache()
    assert len(manager._explainer_cache) == 0
    assert len(manager._results_cache) == 0


def test_regression_shap_workflow():
    X = np.array([
        [1.0, 10.0],
        [2.0, 20.0],
        [3.0, 30.0],
        [4.0, 40.0],
        [5.0, 50.0],
    ])
    y = np.array([12.0, 22.0, 32.0, 42.0, 52.0])
    fnames = ["x1", "x2"]

    reg = GradientBoostingRegressor(n_estimators=10, random_state=42)
    reg.fit(X, y)

    manager = SHAPExplanationManager()
    wrapper = manager.build_explainer_wrapper(
        estimator=reg,
        feature_names=fnames,
        X_train_trans=X,
        is_classification=False,
        model_id="test_gbr",
    )

    calc_res = manager.compute_explanations(wrapper=wrapper, X_eval_trans=X)
    assert calc_res.shap_values.shape == (5, 2)

    global_res = manager.extract_global_importance(calc_res)
    assert len(global_res.top_features) == 2

