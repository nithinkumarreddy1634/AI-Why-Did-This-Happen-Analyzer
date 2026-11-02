"""Automated statistical test recommendation engine."""

from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np


def recommend_statistical_test(
    df: pd.DataFrame,
    target_col: str,
    predictor_col: Optional[str] = None,
    study_design: str = "independent",
) -> Dict[str, Any]:
    """Analyze data properties and recommend the optimal statistical test.
    
    Args:
        df: Input DataFrame.
        target_col: Primary outcome or response variable.
        predictor_col: Comparison or grouping variable.
        study_design: 'independent' or 'paired'.
    """
    if df is None or df.empty:
        return {
            "status": "error",
            "message": "Dataset is empty.",
            "recommended_test": "None",
            "alternative_tests": [],
            "justification": "No data available.",
            "assumptions_to_check": [],
        }

    if target_col not in df.columns:
        return {
            "status": "error",
            "message": f"Target column '{target_col}' not found in dataset.",
            "recommended_test": "None",
            "alternative_tests": [],
            "justification": "Invalid target variable.",
            "assumptions_to_check": [],
        }

    target_s = df[target_col].dropna()
    is_target_num = pd.api.types.is_numeric_dtype(target_s) and not pd.api.types.is_bool_dtype(target_s)
    target_unique = target_s.nunique()

    # Paired design between two numerical columns
    if study_design == "paired" and predictor_col and predictor_col in df.columns:
        pred_s = df[predictor_col].dropna()
        is_pred_num = pd.api.types.is_numeric_dtype(pred_s) and not pd.api.types.is_bool_dtype(pred_s)
        if is_target_num and is_pred_num:
            return {
                "status": "success",
                "recommended_test": "Paired Samples T-Test",
                "alternative_tests": ["Wilcoxon Signed-Rank Test (Non-parametric)"],
                "justification": (
                    f"Comparing two matched numerical variables ('{target_col}' and '{predictor_col}') "
                    f"under a paired or repeated-measures design."
                ),
                "assumptions_to_check": [
                    "Observations are matched/paired across the two conditions.",
                    "Paired differences are approximately normally distributed.",
                    "No extreme outlier differences distorting the sample mean.",
                ],
            }

    # Case 1: Target is Numerical
    if is_target_num and target_unique > 5:
        if predictor_col is None or predictor_col not in df.columns:
            return {
                "status": "success",
                "recommended_test": "One-Sample T-Test / Normality Check",
                "alternative_tests": ["Shapiro-Wilk Test", "Single Group Mean CI"],
                "justification": f"Evaluating distribution of continuous metric '{target_col}' without a grouping variable.",
                "assumptions_to_check": ["Normality of values", "Absence of extreme outliers"],
            }

        pred_s = df[predictor_col].dropna()
        is_pred_num = pd.api.types.is_numeric_dtype(pred_s) and not pd.api.types.is_bool_dtype(pred_s)
        pred_unique = pred_s.nunique()

        # Subcase 1A: Numerical vs Numerical -> Correlation
        if is_pred_num and pred_unique > 5:
            return {
                "status": "success",
                "recommended_test": "Pearson Linear Correlation",
                "alternative_tests": [
                    "Spearman Rank Correlation (Non-linear or Outlier-robust)",
                    "Kendall's Tau Correlation (Small sample or heavy tied ranks)",
                ],
                "justification": (
                    f"Both '{target_col}' and '{predictor_col}' are continuous numerical features. "
                    f"Pearson tests for linear association, while Spearman evaluates monotonic relationships."
                ),
                "assumptions_to_check": [
                    "Bivariate approximate normality.",
                    "Linear relationship between variables.",
                    "Homoscedasticity across the range.",
                ],
            }

        # Subcase 1B: Numerical vs Categorical (2 groups) -> Independent t-test
        elif pred_unique == 2:
            return {
                "status": "success",
                "recommended_test": "Independent Samples T-Test (Welch's / Student's)",
                "alternative_tests": ["Mann-Whitney U Test (Non-parametric)"],
                "justification": (
                    f"Comparing numerical outcome '{target_col}' across exactly 2 groups of '{predictor_col}' "
                    f"({list(pred_s.unique())}). Welch's t-test protects against unequal group variances."
                ),
                "assumptions_to_check": [
                    "Independent observations between the two groups.",
                    "Approximate normality in both groups.",
                    "Homogeneity of variance (tested via Levene's test).",
                ],
            }

        # Subcase 1C: Numerical vs Categorical (>= 3 groups) -> One-Way ANOVA
        elif pred_unique >= 3:
            return {
                "status": "success",
                "recommended_test": "One-Way ANOVA",
                "alternative_tests": ["Kruskal-Wallis Test (Non-parametric)"],
                "justification": (
                    f"Comparing numerical outcome '{target_col}' across {pred_unique} distinct groups of '{predictor_col}'. "
                    f"ANOVA evaluates whether any group mean deviates significantly from the others."
                ),
                "assumptions_to_check": [
                    "Independence of observations across groups.",
                    "Approximate normality within each group.",
                    "Homogeneity of variance (Levene's test).",
                ],
            }

    # Case 2: Target is Categorical / Binary
    else:
        if predictor_col is None or predictor_col not in df.columns:
            return {
                "status": "success",
                "recommended_test": "Goodness-of-Fit / Frequency Audit",
                "alternative_tests": ["Chi-Square Goodness-of-Fit"],
                "justification": f"Auditing categorical distribution of '{target_col}'.",
                "assumptions_to_check": ["Minimum expected counts ≥ 5"],
            }

        pred_s = df[predictor_col].dropna()
        pred_unique = pred_s.nunique()

        # Subcase 2A: 2x2 Contingency Table
        if target_unique == 2 and pred_unique == 2:
            return {
                "status": "success",
                "recommended_test": "Chi-Square Test of Independence",
                "alternative_tests": [
                    "Fisher's Exact Test (Recommended if expected cell counts < 5)",
                    "Two-Proportion Z-Test (Direct proportion delta comparison)",
                ],
                "justification": (
                    f"Both outcome '{target_col}' and predictor '{predictor_col}' are binary (2×2 cross-tabulation). "
                    f"Chi-square evaluates general independence; Fisher's exact gives exact p-values for smaller samples."
                ),
                "assumptions_to_check": [
                    "Independence of observations.",
                    "Expected cell frequencies ≥ 5 (if violated, use Fisher's Exact).",
                ],
            }

        # Subcase 2B: General R x C Categorical Association
        else:
            return {
                "status": "success",
                "recommended_test": "Chi-Square Test of Independence",
                "alternative_tests": ["Cramér's V Association Strength"],
                "justification": (
                    f"Evaluating contingency association between multi-class categorical features '{target_col}' "
                    f"({target_unique} classes) and '{predictor_col}' ({pred_unique} classes)."
                ),
                "assumptions_to_check": [
                    "Adequate sample size: no more than 20% of expected cells under 5.",
                    "Independent observations.",
                ],
            }

    return {
        "status": "warning",
        "recommended_test": "Custom Investigation",
        "alternative_tests": ["Chi-Square", "ANOVA", "Correlation"],
        "justification": "Variable types require custom selection.",
        "assumptions_to_check": [],
    }

