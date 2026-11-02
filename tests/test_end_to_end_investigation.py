"""Comprehensive end-to-end multi-phase integration tests across synthetic datasets (Phase 14)."""

import pytest
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor

from modules.machine_learning.ml_preprocessing import fit_ml_preprocessor
from modules.statistical_analysis.statistical_engine import StatisticalEngine
from modules.unexpected_detection.detection_engine import UnexpectedDetectionEngine
from modules.explainability.explanation_manager import SHAPExplanationManager
from modules.explainability.lime_manager import LIMEExplanationManager
from modules.root_cause.investigation_manager import RootCauseInvestigationManager
from modules.reporting.report_manager import AutomatedReportManager


@pytest.fixture
def dataset_a_customer_churn():
    """Synthetic customer churn dataset with controlled shifts."""
    np.random.seed(101)
    n = 100
    contract = ["Month-to-month"] * 50 + ["Two year"] * 50
    monthly_charges = np.concatenate([np.random.normal(90, 8, 50), np.random.normal(55, 8, 50)])
    tenure = np.concatenate([np.random.normal(12, 4, 50), np.random.normal(55, 8, 50)])
    churn = [1] * 40 + [0] * 10 + [0] * 45 + [1] * 5
    return pd.DataFrame({
        "contract": contract,
        "monthly_charges": monthly_charges,
        "tenure": tenure,
        "churn": churn,
    })


@pytest.fixture
def dataset_b_sales_performance():
    """Synthetic sales dataset with known regional and product category declines."""
    np.random.seed(102)
    n = 100
    region = ["East"] * 50 + ["West"] * 50
    discount = np.concatenate([np.random.uniform(0.0, 0.1, 50), np.random.uniform(0.2, 0.4, 50)])
    quantity = np.random.poisson(5, 100)
    sales = 200 * quantity * (1.0 - discount) + np.random.normal(0, 20, 100)
    return pd.DataFrame({
        "region": region,
        "discount": discount,
        "quantity": quantity,
        "sales": sales,
    })


@pytest.fixture
def dataset_c_edge_cases():
    """Edge cases: missing values, constant columns, duplicate rows, small sample."""
    return pd.DataFrame({
        "group": ["A", "A", "A", "B", "B", "B", "B", "B"],
        "constant_col": [1.0] * 8,
        "feature_with_nulls": [10.0, np.nan, 30.0, 40.0, 50.0, np.nan, 70.0, 80.0],
        "target": [0, 0, 1, 1, 1, 1, 0, 1],
    })


def test_workflow_customer_churn_investigation(dataset_a_customer_churn):
    df = dataset_a_customer_churn
    target = "churn"

    # 1. Statistical investigation
    stat_engine = StatisticalEngine()
    t_res = stat_engine.run_independent_t_test(
        group1=df.loc[df["contract"] == "Two year", "monthly_charges"],
        group2=df.loc[df["contract"] == "Month-to-month", "monthly_charges"],
        group1_name="Two year",
        group2_name="Month-to-month",
        variable_name="monthly_charges",
    )
    assert t_res.is_significant is True

    # 2. ML & SHAP
    prep_res = fit_ml_preprocessor(
        X_train=df,
        numeric_features=["monthly_charges", "tenure"],
        categorical_features=["contract"],
        excluded_columns=["churn"],
    )
    X_trans = prep_res.preprocessor.transform(df)
    y = df[target].to_numpy()

    clf = RandomForestClassifier(n_estimators=10, random_state=42)
    clf.fit(X_trans, y)

    shap_mgr = SHAPExplanationManager()
    wrapper = shap_mgr.build_explainer_wrapper(
        estimator=clf,
        feature_names=prep_res.feature_names_out,
        X_train_trans=X_trans,
        is_classification=True,
    )
    calc_res = shap_mgr.compute_explanations(wrapper, X_trans[:10], target_class_idx=1)
    global_shap = shap_mgr.extract_global_importance(calc_res)
    assert len(global_shap.top_features) > 0

    # 3. LIME
    lime_mgr = LIMEExplanationManager()
    lime_wrapper = lime_mgr.build_explainer_wrapper(
        estimator=clf,
        feature_names=prep_res.feature_names_out,
        X_train_trans=X_trans,
        is_classification=True,
    )
    lime_res = lime_mgr.explain_instance(lime_wrapper, X_trans[0], sample_index=0, num_samples=150)
    assert len(lime_res.contributions_df) > 0

    # 4. Root Cause Engine
    rc_mgr = RootCauseInvestigationManager()
    rc_pkg = rc_mgr.execute_investigation(
        df=df,
        target_col=target,
        baseline_mask=(df["contract"] == "Two year"),
        comparison_mask=(df["contract"] == "Month-to-month"),
        title="Telecom Churn Full Pipeline",
        candidate_features=["monthly_charges", "tenure"],
        shap_global_res=global_shap,
        lime_local_res=lime_res,
    )
    assert rc_pkg.ranking_result.top_candidate_name in ["monthly_charges", "tenure"]
    assert len(rc_pkg.ranking_result.ranked_candidates) == 2

    # 5. Reporting Engine
    rep_mgr = AutomatedReportManager()
    collected, narratives = rep_mgr.assemble_report(
        df=df,
        active_investigation_pkg=rc_pkg,
        shap_global_res=global_shap,
        lime_active_res=lime_res,
        title="Quarterly Churn Investigation Dossier",
    )

    pdf_bytes = rep_mgr.generate_pdf(collected, narratives)
    docx_bytes = rep_mgr.generate_docx(collected, narratives)
    html_str = rep_mgr.generate_html(collected, narratives)
    json_str = rep_mgr.generate_json(collected, narratives)

    assert len(pdf_bytes) > 2000
    assert len(docx_bytes) > 2000
    assert len(html_str) > 1000
    assert len(json_str) > 1000


def test_workflow_sales_regression_investigation(dataset_b_sales_performance):
    df = dataset_b_sales_performance
    target = "sales"

    rc_mgr = RootCauseInvestigationManager()
    rc_pkg = rc_mgr.execute_investigation(
        df=df,
        target_col=target,
        baseline_mask=(df["region"] == "East"),
        comparison_mask=(df["region"] == "West"),
        title="Sales Decline Across Regions",
        candidate_features=["discount", "quantity"],
    )

    assert rc_pkg.outcome_def.outcome_type == "numerical"
    assert len(rc_pkg.ranking_result.ranked_candidates) == 2
    assert rc_pkg.ranking_result.top_candidate_name in ["discount", "quantity"]


def test_edge_cases_safety(dataset_c_edge_cases):
    df = dataset_c_edge_cases

    rc_mgr = RootCauseInvestigationManager()
    # Should not raise exception despite nulls and constant columns
    rc_pkg = rc_mgr.execute_investigation(
        df=df,
        target_col="target",
        baseline_mask=(df["group"] == "A"),
        comparison_mask=(df["group"] == "B"),
        title="Edge Cases Investigation",
    )
    assert rc_pkg is not None
    assert len(rc_pkg.ranking_result.ranked_candidates) >= 1
