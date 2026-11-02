"""Categorical hypothesis testing: Chi-Square, Fisher's Exact, and Two-Proportion tests."""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import stats

from modules.statistical_analysis.hypothesis_testing import HypothesisTestResult
from modules.statistical_analysis.effect_sizes import compute_cramers_v
from modules.statistical_analysis.confidence_intervals import ci_difference_in_proportions


def chi_square_independence_test(
    series1: pd.Series,
    series2: pd.Series,
    col1_name: str = "Variable 1",
    col2_name: str = "Variable 2",
    alpha: float = 0.05,
) -> HypothesisTestResult:
    """Run Chi-Square Test of Independence between two categorical features."""
    valid_mask = series1.notna() & series2.notna()
    s1 = series1[valid_mask].astype(str)
    s2 = series2[valid_mask].astype(str)
    n = len(s1)

    research_q = f"Is there a statistically significant association between '{col1_name}' and '{col2_name}'?"
    h0 = f"'{col1_name}' and '{col2_name}' are independent (no statistical association)."
    h1 = f"'{col1_name}' and '{col2_name}' are dependent / associated."

    if n < 4 or s1.nunique() < 2 or s2.nunique() < 2:
        return HypothesisTestResult(
            test_name="Chi-Square Test of Independence",
            research_question=research_q,
            h0=h0,
            h1=h1,
            test_statistic=0.0,
            statistic_name="χ²",
            p_value=1.0,
            alpha=alpha,
            interpretation="Insufficient data or categories (minimum 2 categories per variable with N ≥ 4 required).",
            limitations=["Contingency table could not be formed with at least 2 rows and 2 columns."],
            sample_sizes={"Total Records": n},
        )

    observed_ct = pd.crosstab(s1, s2)
    chi2_val, p_val, dof, expected_arr = stats.chi2_contingency(observed_ct)

    expected_df = pd.DataFrame(
        expected_arr.round(2), index=observed_ct.index, columns=observed_ct.columns
    )

    # Check Cochran's rule (no expected cell < 1, and no more than 20% < 5)
    cells_under_5 = int((expected_arr < 5).sum())
    total_cells = expected_arr.size
    pct_under_5 = (cells_under_5 / total_cells) * 100.0 if total_cells > 0 else 0.0

    v_val, v_interp = compute_cramers_v(observed_ct)

    sig_text = "statistically significant" if p_val < alpha else "not statistically significant"
    interpretation = (
        f"The association between '{col1_name}' and '{col2_name}' is {sig_text} "
        f"(χ²({dof}) = {chi2_val:.3f}, p = {p_val:.4f}, α = {alpha}). "
        f"The association strength is {v_interp.lower()} (Cramér's V = {v_val:.3f})."
    )

    limitations = [
        "Chi-square tests detect statistical dependence, but do NOT prove causal influence.",
    ]
    if pct_under_5 > 20.0:
        limitations.append(
            f"Caution: {pct_under_5:.1f}% of expected cells have counts < 5 (violating Cochran's guideline). "
            f"If table is 2x2, Fisher's Exact test is recommended."
        )

    return HypothesisTestResult(
        test_name="Chi-Square Test of Independence",
        research_question=research_q,
        h0=h0,
        h1=h1,
        test_statistic=float(chi2_val),
        statistic_name="χ²",
        p_value=float(p_val),
        alpha=alpha,
        effect_size=v_val,
        effect_size_name="Cramér's V",
        effect_size_interpretation=v_interp,
        interpretation=interpretation,
        sample_sizes={"Total Sample Size (N)": n},
        assumptions_met={
            "expected_counts_adequate": pct_under_5 <= 20.0,
            "min_expected_count": float(expected_arr.min()),
        },
        limitations=limitations,
        additional_metrics={
            "dof": int(dof),
            "observed_table": observed_ct,
            "expected_table": expected_df,
            "pct_cells_under_5": pct_under_5,
        },
    )


def fisher_exact_test(
    contingency_table_2x2: pd.DataFrame,
    col1_name: str = "Row Variable",
    col2_name: str = "Column Variable",
    alpha: float = 0.05,
) -> HypothesisTestResult:
    """Run Fisher's Exact Test on a 2x2 contingency table (ideal for small samples)."""
    research_q = f"Is there an exact association between '{col1_name}' and '{col2_name}'?"
    h0 = f"The odds of '{col2_name}' are identical across categories of '{col1_name}' (Odds Ratio = 1.0)."
    h1 = f"The odds ratio deviates from 1.0 (true association exists)."

    if contingency_table_2x2.shape != (2, 2):
        return HypothesisTestResult(
            test_name="Fisher's Exact Test",
            research_question=research_q,
            h0=h0,
            h1=h1,
            test_statistic=1.0,
            statistic_name="Odds Ratio",
            p_value=1.0,
            alpha=alpha,
            interpretation="Fisher's exact test requires exactly a 2x2 contingency table.",
            limitations=["Table shape is not 2x2."],
        )

    odds_ratio, p_val = stats.fisher_exact(contingency_table_2x2.values, alternative="two-sided")
    odds_ratio = float(odds_ratio) if not np.isnan(odds_ratio) else 1.0
    p_val = float(p_val) if not np.isnan(p_val) else 1.0

    sig_text = "statistically significant" if p_val < alpha else "not statistically significant"
    interpretation = (
        f"Fisher's exact test indicates the association between '{col1_name}' and '{col2_name}' is {sig_text} "
        f"(Odds Ratio = {odds_ratio:.3f}, p = {p_val:.4f}, α = {alpha})."
    )

    return HypothesisTestResult(
        test_name="Fisher's Exact Test (2x2)",
        research_question=research_q,
        h0=h0,
        h1=h1,
        test_statistic=odds_ratio,
        statistic_name="Odds Ratio",
        p_value=p_val,
        alpha=alpha,
        effect_size=odds_ratio,
        effect_size_name="Odds Ratio",
        interpretation=interpretation,
        sample_sizes={"Total Count": int(contingency_table_2x2.values.sum())},
        limitations=[
            "Exact test calculation condition: row and column marginal totals are assumed fixed.",
            "Especially suitable when expected cell frequencies fall below 5.",
        ],
        additional_metrics={"contingency_table": contingency_table_2x2},
    )


def two_proportion_test(
    count1: int,
    nobs1: int,
    count2: int,
    nobs2: int,
    group1_name: str = "Group 1",
    group2_name: str = "Group 2",
    outcome_label: str = "Event Rate",
    alpha: float = 0.05,
) -> HypothesisTestResult:
    """Run Two-Sample Z-Test for Difference in Proportions (p1 - p2)."""
    research_q = f"Is the proportion of '{outcome_label}' significantly different between '{group1_name}' and '{group2_name}'?"
    h0 = f"The true population proportions are equal (p1 = p2)."
    h1 = f"The true population proportions are different (p1 ≠ p2)."

    if nobs1 <= 0 or nobs2 <= 0:
        return HypothesisTestResult(
            test_name="Two-Proportion Z-Test",
            research_question=research_q,
            h0=h0,
            h1=h1,
            test_statistic=0.0,
            statistic_name="z",
            p_value=1.0,
            alpha=alpha,
            interpretation="Sample size in one or both cohorts is zero.",
        )

    p1 = count1 / nobs1
    p2 = count2 / nobs2
    diff = p1 - p2

    # Pooled proportion
    p_pool = (count1 + count2) / (nobs1 + nobs2)
    se_pooled = np.sqrt(p_pool * (1.0 - p_pool) * ((1.0 / nobs1) + (1.0 / nobs2)))

    if se_pooled == 0.0:
        z_stat = 0.0
        p_val = 1.0
    else:
        z_stat = float(diff / se_pooled)
        p_val = float(2.0 * (1.0 - stats.norm.cdf(abs(z_stat))))

    diff_val, ci_low, ci_high = ci_difference_in_proportions(count1, nobs1, count2, nobs2, confidence=0.95)

    sig_text = "statistically significant" if p_val < alpha else "not statistically significant"
    interpretation = (
        f"The difference in proportion of '{outcome_label}' between {group1_name} ({p1*100:.1f}%) and "
        f"{group2_name} ({p2*100:.1f}%) is {sig_text} (z = {z_stat:.3f}, p = {p_val:.4f}, α = {alpha}). "
        f"Absolute risk difference is {diff*100:+.2f} percentage points (95% CI: [{ci_low*100:+.2f}%, {ci_high*100:+.2f}%])."
    )

    return HypothesisTestResult(
        test_name="Two-Proportion Z-Test",
        research_question=research_q,
        h0=h0,
        h1=h1,
        test_statistic=z_stat,
        statistic_name="z",
        p_value=p_val,
        alpha=alpha,
        effect_size=round(diff, 4),
        effect_size_name="Difference in Proportions (Risk Difference)",
        ci_lower=ci_low,
        ci_upper=ci_high,
        confidence_level=0.95,
        interpretation=interpretation,
        sample_sizes={group1_name: nobs1, group2_name: nobs2},
        limitations=[
            "Assumes independent binomial samples and large enough sample sizes (n*p >= 5 and n*(1-p) >= 5).",
        ],
        additional_metrics={
            "prop1": p1,
            "prop2": p2,
            "risk_difference": diff,
        },
    )

