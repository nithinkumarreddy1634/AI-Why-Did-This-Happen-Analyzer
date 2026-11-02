"""End-to-end integration tests for LIME Explainability Engine."""

import pytest
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor

from modules.machine_learning.ml_preprocessing import fit_ml_preprocessor
from modules.explainability.lime_manager import LIMEExplanationManager
from modules.explainability.lime_explainer import LIMEConfig
from modules.explainability.lime_interpreter import generate_lime_narrative_insights


@pytest.fixture
def sample_dataset():
    df = pd.DataFrame({
        "tenure": [1.0, 5.0, 12.0, 24.0, 36.0, 48.0, 60.0, 72.0],
        "monthly_charges": [20.0, 45.0, 60.0, 75.0, 80.0, 85.0, 90.0, 105.0],
        "contract": ["month", "month", "month", "one_year", "one_year", "two_year", "two_year", "two_year"],
        "churn": [1, 1, 1, 0, 0, 0, 0, 0],
    })
    return df


def test_end_to_end_lime_classification_workflow(sample_dataset):
    df = sample_dataset
    target_col = "churn"
    num_cols = ["tenure", "monthly_charges"]
    cat_cols = ["contract"]

    # 1. Preprocessing
    prep_res = fit_ml_preprocessor(
        X_train=df,
        numeric_features=num_cols,
        categorical_features=cat_cols,
        excluded_columns=[],
    )
    X_trans = prep_res.preprocessor.transform(df)
    y = df[target_col].to_numpy()

    # 2. Train model
    clf = RandomForestClassifier(n_estimators=10, random_state=42)
    clf.fit(X_trans, y)

    # 3. LIME Explanation Manager
    manager = LIMEExplanationManager()
    wrapper = manager.build_explainer_wrapper(
        estimator=clf,
        feature_names=prep_res.feature_names_out,
        X_train_trans=X_trans,
        is_classification=True,
        class_names=["Retained", "Churned"],
        model_id="rf_churn_test",
        config=LIMEConfig(num_samples=250, random_state=42),
    )

    assert wrapper.explainer is not None
    assert len(manager._explainer_cache) > 0

    # 4. Local explanation for instance 0
    row_0 = X_trans[0]
    local_exp = manager.explain_instance(
        wrapper=wrapper,
        instance_row=row_0,
        sample_index=0,
        target_class_idx=1,
        num_features=4,
        num_samples=250,
        actual_outcome=y[0],
    )

    assert local_exp.sample_index == 0
    assert len(local_exp.contributions_df) > 0
    assert local_exp.local_fidelity_r2 is not None
    assert local_exp.local_surrogate_prediction is not None

    # 5. Batch explanation on subset
    eval_df = pd.DataFrame(X_trans, columns=prep_res.feature_names_out)
    batch_exp = manager.explain_batch(
        wrapper=wrapper,
        X_eval_df=eval_df,
        sample_indices=[0, 1, 2, 3],
        target_class_idx=1,
        num_features=3,
        num_samples=200,
        actual_outcomes=y,
    )

    assert batch_exp.n_explained == 4
    assert len(batch_exp.explanations) == 4
    assert len(batch_exp.recurring_top_features) > 0
    assert isinstance(batch_exp.mean_fidelity_r2, float)

    # 6. Narrative Insights
    insights = manager.generate_narrative(lime_result=local_exp, batch_result=batch_exp)
    assert len(insights) >= 3
    categories = [ins.category for ins in insights]
    assert "Causality Alert" in categories

    # 7. Cache operations
    manager.clear_cache()
    assert len(manager._explainer_cache) == 0
    assert len(manager._explanation_cache) == 0


def test_end_to_end_lime_regression_workflow(sample_dataset):
    df = sample_dataset
    # Predict monthly charges
    num_cols = ["tenure"]
    cat_cols = ["contract"]

    prep_res = fit_ml_preprocessor(
        X_train=df,
        numeric_features=num_cols,
        categorical_features=cat_cols,
        excluded_columns=["churn"],
    )
    X_trans = prep_res.preprocessor.transform(df)
    y = df["monthly_charges"].to_numpy()

    reg = GradientBoostingRegressor(n_estimators=10, random_state=42)
    reg.fit(X_trans, y)

    manager = LIMEExplanationManager()
    wrapper = manager.build_explainer_wrapper(
        estimator=reg,
        feature_names=prep_res.feature_names_out,
        X_train_trans=X_trans,
        is_classification=False,
        model_id="gbr_charges_test",
        config=LIMEConfig(num_samples=250, random_state=42),
    )

    local_exp = manager.explain_instance(
        wrapper=wrapper,
        instance_row=X_trans[2],
        sample_index=2,
        num_features=3,
        num_samples=250,
        actual_outcome=float(y[2]),
    )

    assert local_exp.is_classification is False
    assert local_exp.sample_index == 2
    assert len(local_exp.contributions_df) > 0
    assert isinstance(local_exp.local_fidelity_r2, float)
