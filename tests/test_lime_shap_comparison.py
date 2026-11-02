"""Unit tests for SHAP vs LIME explainability comparison benchmarking."""

import pytest
import numpy as np
import pandas as pd

from modules.explainability.shap_local import LocalSHAPResult
from modules.explainability.lime_local_analysis import LIMEExplanationResult
from modules.explainability.lime_shap_comparison import compare_shap_and_lime, SHAPLIMEComparisonResult
from utils.lime_helpers import calculate_jaccard_similarity, compute_rank_correlation


def test_jaccard_similarity_helper():
    s1 = ["f1", "f2", "f3", "f4"]
    s2 = ["f2", "f3", "f5", "f6"]
    # Intersection: {f2, f3} (size 2), Union: {f1, f2, f3, f4, f5, f6} (size 6) -> 2/6 = 0.3333
    sim = calculate_jaccard_similarity(s1, s2)
    assert pytest.approx(sim, 0.01) == 0.333

    assert calculate_jaccard_similarity([], []) == 1.0
    assert calculate_jaccard_similarity(["a"], ["b"]) == 0.0
    assert calculate_jaccard_similarity(["a", "b"], ["a", "b"]) == 1.0


def test_rank_correlation_helper():
    r1 = {"a": 1, "b": 2, "c": 3}
    r2 = {"a": 1, "b": 2, "c": 3}
    corr = compute_rank_correlation(r1, r2)
    assert pytest.approx(corr, 0.001) == 1.0

    r3 = {"a": 3, "b": 2, "c": 1}
    corr_inv = compute_rank_correlation(r1, r3)
    assert pytest.approx(corr_inv, 0.001) == -1.0


@pytest.fixture
def dummy_shap_and_lime_results():
    # Construct mock LocalSHAPResult
    shap_df = pd.DataFrame({
        "Feature": ["tenure", "monthly_charges", "total_charges", "contract"],
        "SHAP_Value": [-0.45, 0.30, -0.15, 0.05],
        "Abs_SHAP": [0.45, 0.30, 0.15, 0.05],
        "Direction": ["Decreases Prediction", "Increases Prediction", "Decreases Prediction", "Increases Prediction"],
        "Feature_Value": [12.0, 75.0, 900.0, 1.0],
    })
    shap_res = LocalSHAPResult(
        sample_index=0,
        base_value=0.5,
        model_output=0.25,
        actual_outcome=0,
        predicted_label="Retained",
        target_class_name="Churn",
        contributions_df=shap_df,
        positive_drivers=shap_df[shap_df["SHAP_Value"] > 0].to_dict(orient="records"),
        negative_drivers=shap_df[shap_df["SHAP_Value"] < 0].to_dict(orient="records"),
        additivity_audit={"is_additive": True, "discrepancy": 0.0},
        narrative_summary="Mock SHAP summary",
    )

    # Construct mock LIMEExplanationResult
    lime_df = pd.DataFrame({
        "Condition": ["tenure <= 15", "monthly_charges > 70", "payment_method=1", "contract <= 1"],
        "Feature": ["tenure", "monthly_charges", "payment_method", "contract"],
        "Observed_Value": [12.0, 75.0, 1.0, 1.0],
        "LIME_Weight": [-0.40, 0.28, 0.12, -0.04],
        "Abs_Weight": [0.40, 0.28, 0.12, 0.04],
        "Formatted_Weight": ["-0.4000", "+0.2800", "+0.1200", "-0.0400"],
        "Direction": ["Opposes Outcome (-)", "Supports Outcome (+)", "Supports Outcome (+)", "Opposes Outcome (-)"],
        "Sign": [-1, 1, 1, -1],
        "Rank": [1, 2, 3, 4],
    })
    lime_res = LIMEExplanationResult(
        sample_index=0,
        is_classification=True,
        target_class_idx=1,
        target_class_name="Churned",
        actual_outcome=0,
        predicted_outcome="0.25",
        predicted_probabilities=[0.75, 0.25],
        contributions_df=lime_df,
        positive_drivers=lime_df[lime_df["LIME_Weight"] > 0].to_dict(orient="records"),
        negative_drivers=lime_df[lime_df["LIME_Weight"] < 0].to_dict(orient="records"),
        local_surrogate_prediction=0.26,
        local_fidelity_r2=0.88,
        local_intercept=0.48,
        fidelity_gap=0.01,
        raw_feature_values={"tenure": 12.0, "monthly_charges": 75.0, "payment_method": 1.0, "contract": 1.0},
        narrative_summary="Mock LIME summary",
    )

    return shap_res, lime_res


def test_compare_shap_and_lime(dummy_shap_and_lime_results):
    shap_res, lime_res = dummy_shap_and_lime_results

    cmp = compare_shap_and_lime(shap_res, lime_res, top_k=4)

    assert isinstance(cmp, SHAPLIMEComparisonResult)
    assert cmp.sample_index == 0
    assert len(cmp.comparison_df) > 0
    assert "Feature" in cmp.comparison_df.columns
    assert "SHAP_Value" in cmp.comparison_df.columns
    assert "LIME_Weight" in cmp.comparison_df.columns
    assert "Agreement_Status" in cmp.comparison_df.columns

    # Common features should be tenure, monthly_charges, contract
    assert "tenure" in cmp.common_top_features
    assert "monthly_charges" in cmp.common_top_features
    assert "total_charges" in cmp.shap_exclusive_features
    assert "payment_method" in cmp.lime_exclusive_features

    # Jaccard should be 3 / 5 = 0.6
    assert pytest.approx(cmp.jaccard_similarity, 0.01) == 0.60

    # Direction agreement percentage
    assert cmp.direction_agreement_pct >= 0.0

    # Non-causal disclaimer present
    assert "NOT real-world causality" in cmp.disclaimer

    # Dict serialization
    d = cmp.to_dict()
    assert d["sample_index"] == 0
    assert "records" in d
