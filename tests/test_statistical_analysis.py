"""Comprehensive test suite for Phase 6: Statistical Investigation & Hypothesis Testing Engine.

Covers all 16 mandated scenarios:
1. Two independent numerical groups with different means.
2. Two independent groups with similar means.
3. Valid paired measurements.
4. Three or more numerical groups.
5. Categorical variables with association.
6. Categorical variables without association.
7. Small 2x2 contingency tables.
8. Strong positive correlation.
9. Strong negative correlation.
10. No meaningful correlation.
11. Constant numerical variables.
12. Missing values.
13. Empty groups.
14. Small sample sizes.
15. Multiple hypothesis corrections.
16. Invalid variable selections.
"""

import pytest
import numpy as np
import pandas as pd
from scipy import stats

from modules.statistical_analysis.hypothesis_testing import HypothesisTestResult
from modules.statistical_analysis.numerical_tests import (
    independent_t_test,
    paired_t_test,
    one_way_anova,
    mann_whitney_u_test,
    wilcoxon_signed_rank_test,
    kruskal_wallis_test,
)
from modules.statistical_analysis.categorical_tests import (
    chi_square_independence_test,
    fisher_exact_test,
    two_proportion_test,
)
from modules.statistical_analysis.correlation_tests import (
    compute_pearson_correlation,
    compute_spearman_correlation,
    compute_kendall_correlation,
)
from modules.statistical_analysis.assumption_checks import (
    check_normality,
    check_equal_variance,
    generate_qq_plot_data,
)
from modules.statistical_analysis.effect_sizes import (
    compute_cohens_d,
    compute_hedges_g,
    compute_paired_cohens_d,
    compute_eta_squared,
    compute_cramers_v,
)
from modules.statistical_analysis.confidence_intervals import (
    ci_difference_in_means,
    ci_paired_difference,
    ci_difference_in_proportions,
    ci_pearson_correlation,
)
from modules.statistical_analysis.multiple_testing import (
    bonferroni_correction,
    benjamini_hochberg_correction,
    adjust_test_results,
)
from modules.statistical_analysis.test_recommender import recommend_statistical_test
from modules.statistical_analysis.result_interpreter import StatisticalResultInterpreter
from modules.statistical_analysis.statistical_engine import StatisticalEngine
from utils.statistical_helpers import clean_numeric_series, clean_paired_series, safe_divide


# ==============================================================================
# Helper & Confidence Interval Tests
# ==============================================================================

def test_clean_series_utilities():
    """Verify numeric cleaning and paired series alignment."""
    s1 = pd.Series([1.0, 2.0, np.nan, 4.0, "invalid"])
    clean_s, n_drop = clean_numeric_series(s1)
    assert len(clean_s) == 3
    assert n_drop == 2

    s_a = pd.Series([10.0, 20.0, np.nan, 40.0])
    s_b = pd.Series([15.0, np.nan, 25.0, 45.0])
    c_a, c_b, n_dropped = clean_paired_series(s_a, s_b)
    assert len(c_a) == len(c_b) == 2
    assert n_dropped == 2


def test_confidence_intervals_calculations():
    """Verify confidence interval mathematics against standard statistical bounds."""
    # Two independent means
    g1 = pd.Series([10, 11, 12, 10, 11])
    g2 = pd.Series([20, 21, 22, 20, 21])
    diff, low, high = ci_difference_in_means(g1, g2, confidence=0.95, equal_var=True)
    assert diff == pytest.approx(-10.0)
    assert low < diff < high

    # Paired differences
    diffs = pd.Series([2.0, 3.0, 2.5, 3.5, 2.0])
    m_d, low_d, high_d = ci_paired_difference(diffs, confidence=0.95)
    assert m_d == pytest.approx(2.6)
    assert low_d < m_d < high_d

    # Pearson correlation CI via Fisher z
    r, r_low, r_high = ci_pearson_correlation(0.8, 50, confidence=0.95)
    assert r == 0.8
    assert 0.6 < r_low < 0.8 < r_high < 1.0


# ==============================================================================
# 16 Mandated Scenarios (Prompt Section 16)
# ==============================================================================

class TestMandatedStatisticalScenarios:
    """Rigorous verification of the 16 testing scenarios mandated in Master Prompt."""

    def test_scenario_1_independent_numerical_different_means(self):
        """Scenario 1: Two independent numerical groups with different means."""
        np.random.seed(42)
        g1 = pd.Series(np.random.normal(50.0, 5.0, 100))
        g2 = pd.Series(np.random.normal(70.0, 5.0, 100))  # Clear mean difference

        res = independent_t_test(g1, g2, "Cohort A", "Cohort B", "Usage", alpha=0.05)
        scipy_res = stats.ttest_ind(g1, g2, equal_var=True)

        assert res.is_significant is True
        assert res.p_value < 0.001
        assert res.test_statistic == pytest.approx(float(scipy_res.statistic), rel=1e-3)
        assert res.effect_size is not None
        assert abs(res.effect_size) > 0.8  # Large effect size
        assert "Large effect" in (res.effect_size_interpretation or "")

    def test_scenario_2_independent_numerical_similar_means(self):
        """Scenario 2: Two independent groups with similar means (fail to reject H0)."""
        np.random.seed(42)
        g1 = pd.Series(np.random.normal(50.0, 5.0, 100))
        g2 = pd.Series(np.random.normal(50.2, 5.0, 100))  # Similar means

        res = independent_t_test(g1, g2, "Control", "Treatment", "Score", alpha=0.05)
        assert res.is_significant is False
        assert res.p_value > 0.05
        assert "Fail to Reject H0" in res.decision

    def test_scenario_3_valid_paired_measurements(self):
        """Scenario 3: Valid paired measurements (before vs after)."""
        np.random.seed(42)
        pre = pd.Series(np.random.normal(100.0, 10.0, 60))
        # Post treatment increases by 8 units with small noise
        post = pre + np.random.normal(8.0, 2.0, 60)

        res = paired_t_test(pre, post, "Pre-Training", "Post-Training", "Productivity", alpha=0.05)
        scipy_paired = stats.ttest_rel(post, pre)

        assert res.is_significant is True
        assert res.p_value < 0.001
        assert res.test_statistic == pytest.approx(float(scipy_paired.statistic), rel=1e-3)
        assert res.ci_lower is not None and res.ci_upper is not None
        assert res.ci_lower > 0  # Entire CI is strictly positive

        # Non-parametric Wilcoxon on same paired data
        res_wilc = wilcoxon_signed_rank_test(pre, post, "Productivity", alpha=0.05)
        assert res_wilc.is_significant is True

    def test_scenario_4_three_or_more_numerical_groups(self):
        """Scenario 4: Three or more numerical groups (ANOVA & Kruskal-Wallis)."""
        np.random.seed(42)
        groups = {
            "Tier 1": pd.Series(np.random.normal(40, 5, 50)),
            "Tier 2": pd.Series(np.random.normal(55, 5, 50)),
            "Tier 3": pd.Series(np.random.normal(70, 5, 50)),
        }
        res_anova = one_way_anova(groups, "Charges", "Tier", alpha=0.05)
        scipy_f = stats.f_oneway(groups["Tier 1"], groups["Tier 2"], groups["Tier 3"])

        assert res_anova.is_significant is True
        assert res_anova.test_statistic == pytest.approx(float(scipy_f.statistic), rel=1e-3)
        assert res_anova.effect_size is not None
        assert res_anova.effect_size > 0.14  # Large effect size for eta-squared

        # Kruskal-Wallis
        res_kw = kruskal_wallis_test(groups, "Charges", "Tier", alpha=0.05)
        assert res_kw.is_significant is True

    def test_scenario_5_categorical_with_association(self):
        """Scenario 5: Categorical variables with strong association (Chi-Square & Cramér's V)."""
        # Contract type strongly associated with Churn
        # Month-to-month has high churn; Two-year has low churn
        contract = pd.Series(["Month-to-month"] * 100 + ["Two-year"] * 100)
        churn = pd.Series(["Yes"] * 70 + ["No"] * 30 + ["Yes"] * 10 + ["No"] * 90)

        res = chi_square_independence_test(contract, churn, "Contract", "Churn", alpha=0.05)
        assert res.is_significant is True
        assert res.p_value < 0.001
        assert res.effect_size is not None
        assert res.effect_size > 0.3  # Moderate to strong Cramér's V

    def test_scenario_6_categorical_without_association(self):
        """Scenario 6: Categorical variables without association."""
        np.random.seed(42)
        n = 200
        cat_a = pd.Series(np.random.choice(["Red", "Blue"], n))
        cat_b = pd.Series(np.random.choice(["Pass", "Fail"], n))

        res = chi_square_independence_test(cat_a, cat_b, "Color", "Outcome", alpha=0.05)
        assert res.is_significant is False
        assert res.p_value > 0.05
        assert "Fail to Reject H0" in res.decision

    def test_scenario_7_small_2x2_contingency_table(self):
        """Scenario 7: Small 2x2 contingency table (Fisher's exact test)."""
        # Rare disease treatment clinical trial
        ct = pd.DataFrame(
            [[3, 1], [1, 5]],
            index=["Treated", "Placebo"],
            columns=["Recovered", "Not Recovered"],
        )
        res = fisher_exact_test(ct, "Treatment", "Outcome", alpha=0.05)
        scipy_odds, scipy_p = stats.fisher_exact(ct.values)

        assert res.test_statistic == pytest.approx(float(scipy_odds), rel=1e-3)
        assert res.p_value == pytest.approx(float(scipy_p), rel=1e-3)

    def test_scenario_8_strong_positive_correlation(self):
        """Scenario 8: Strong positive correlation."""
        np.random.seed(42)
        x = pd.Series(np.linspace(10, 100, 50))
        y = 2.5 * x + np.random.normal(0, 5, 50)

        res_p = compute_pearson_correlation(x, y, "Hours Studied", "Exam Score", alpha=0.05)
        scipy_r, scipy_p = stats.pearsonr(x, y)

        assert res_p.is_significant is True
        assert res_p.test_statistic == pytest.approx(float(scipy_r), rel=1e-3)
        assert res_p.test_statistic > 0.9
        assert res_p.ci_lower is not None and res_p.ci_upper is not None
        assert res_p.ci_lower > 0.85

    def test_scenario_9_strong_negative_correlation(self):
        """Scenario 9: Strong negative correlation."""
        np.random.seed(42)
        x = pd.Series(np.linspace(10, 100, 50))
        y = -3.0 * x + np.random.normal(0, 8, 50)

        res = compute_pearson_correlation(x, y, "Price", "Demand", alpha=0.05)
        assert res.is_significant is True
        assert res.test_statistic < -0.9
        assert "negative" in (res.effect_size_interpretation or "")

    def test_scenario_10_no_meaningful_correlation(self):
        """Scenario 10: No meaningful correlation (near zero, non-significant)."""
        np.random.seed(42)
        x = pd.Series(np.random.normal(50, 10, 100))
        y = pd.Series(np.random.normal(50, 10, 100))

        res = compute_pearson_correlation(x, y, "X", "Y", alpha=0.05)
        assert res.is_significant is False
        assert res.p_value > 0.05
        assert abs(res.test_statistic) < 0.2

    def test_scenario_11_constant_numerical_variables(self):
        """Scenario 11: Constant numerical variables (zero variance safe handling)."""
        const_x = pd.Series([42.0] * 50)
        vary_y = pd.Series(range(50))

        res_corr = compute_pearson_correlation(const_x, vary_y, "Const", "Vary")
        assert res_corr.is_significant is False
        assert res_corr.test_statistic == 0.0

        res_t = independent_t_test(const_x, pd.Series([42.0] * 50), "G1", "G2")
        assert res_t.is_significant is False

    def test_scenario_12_missing_values(self):
        """Scenario 12: Missing values handled gracefully."""
        g1 = pd.Series([10.0, np.nan, 30.0, 40.0, np.nan, 60.0])
        g2 = pd.Series([20.0, 25.0, np.nan, 35.0, 45.0, 55.0])

        res = independent_t_test(g1, g2, "A", "B")
        assert res.sample_sizes["A"] == 4
        assert res.sample_sizes["B"] == 5
        assert not np.isnan(res.test_statistic)

    def test_scenario_13_empty_groups(self):
        """Scenario 13: Empty groups handled safely."""
        empty_s = pd.Series([], dtype=float)
        valid_s = pd.Series([10.0, 20.0, 30.0])

        res = independent_t_test(empty_s, valid_s, "Empty", "Valid")
        assert res.is_significant is False
        assert "Insufficient sample size" in res.interpretation

    def test_scenario_14_small_sample_sizes(self):
        """Scenario 14: Small sample sizes (e.g. N = 2)."""
        small_1 = pd.Series([10.0, 12.0])
        small_2 = pd.Series([20.0, 22.0])

        res = independent_t_test(small_1, small_2, "S1", "S2")
        assert res.sample_sizes["S1"] == 2
        assert res.sample_sizes["S2"] == 2
        assert not np.isnan(res.test_statistic)

    def test_scenario_15_multiple_hypothesis_corrections(self):
        """Scenario 15: Multiple hypothesis testing corrections (Bonferroni & Benjamini-Hochberg)."""
        # 10 p-values, some significant, some borderline, some non-significant
        p_vals = [0.001, 0.008, 0.03, 0.045, 0.06, 0.12, 0.25, 0.40, 0.65, 0.90]

        # Bonferroni: m = 10, so 0.03 * 10 = 0.30 (loses significance at alpha=0.05)
        bonf_adj, bonf_sig = bonferroni_correction(p_vals, alpha=0.05)
        assert len(bonf_adj) == 10
        assert bonf_adj[0] == pytest.approx(0.010)
        assert bonf_adj[1] == pytest.approx(0.080)
        assert bonf_sig[0] is True
        assert bonf_sig[1] is False  # 0.008 adjusted to 0.080 > 0.05

        # Benjamini-Hochberg (FDR) is less conservative than Bonferroni
        bh_adj, bh_sig = benjamini_hochberg_correction(p_vals, alpha=0.05)
        assert len(bh_adj) == 10
        assert bh_adj[0] <= bonf_adj[0]
        # First 2 tests should retain significance under FDR
        assert bh_sig[0] is True
        assert bh_sig[1] is True

    def test_scenario_16_invalid_variable_selections(self):
        """Scenario 16: Invalid variable selections handled gracefully."""
        df = pd.DataFrame({"colA": [1, 2, 3]})
        rec = recommend_statistical_test(df, target_col="non_existent_column")
        assert rec["status"] == "error"
        assert "not found" in rec["message"].lower()


# ==============================================================================
# Engine Facade & Integration Tests
# ==============================================================================

def test_statistical_engine_facade():
    """Verify master StatisticalEngine logging, multiple adjustments, and interpretations."""
    engine = StatisticalEngine(default_alpha=0.05)
    df = pd.DataFrame({
        "churn": ["Yes", "No", "Yes", "No", "Yes", "No"] * 20,
        "charges": list(np.random.normal(70, 10, 60)) + list(np.random.normal(40, 10, 60)),
        "contract": ["Month-to-month", "Two-year", "Month-to-month", "Two-year", "One-year", "Two-year"] * 20,
    })

    # Run t-test
    g_yes = df.loc[df["churn"] == "Yes", "charges"]
    g_no = df.loc[df["churn"] == "No", "charges"]
    res_t = engine.run_independent_t_test(g_yes, g_no, "Churned", "Retained", "Monthly Charges")
    assert len(engine.history) == 1

    # Run Chi-Square
    res_chi = engine.run_chi_square(df["contract"], df["churn"], "Contract", "Churn")
    assert len(engine.history) == 2

    # Narrative interpretation
    narrative = engine.interpret(res_t)
    assert "Interpretation" in narrative["title"]
    assert "Causality Caveat" in narrative["non_causality_notice"]

    # Multiplicity table
    adj_df = engine.apply_multiple_testing_correction(method="benjamini_hochberg")
    assert len(adj_df) == 2
    assert "Raw P-Value" in adj_df.columns
    assert "Adjusted P-Value" in adj_df.columns
