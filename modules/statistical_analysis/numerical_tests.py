"""Numerical hypothesis testing: Independent t-test, Paired t-test, ANOVA, and non-parametric equivalents."""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from scipy import stats

from modules.statistical_analysis.hypothesis_testing import HypothesisTestResult
from modules.statistical_analysis.effect_sizes import (
    compute_cohens_d,
    compute_hedges_g,
    compute_paired_cohens_d,
    compute_eta_squared,
)
from modules.statistical_analysis.confidence_intervals import (
    ci_difference_in_means,
    ci_paired_difference,
)
from modules.statistical_analysis.assumption_checks import (
    check_normality,
    check_equal_variance,
)
from utils.statistical_helpers import clean_numeric_series, clean_paired_series


def independent_t_test(
    group1: pd.Series,
    group2: pd.Series,
    group1_name: str = "Group 1",
    group2_name: str = "Group 2",
    variable_name: str = "Metric",
    alpha: float = 0.05,
    equal_var: str = "auto",
) -> HypothesisTestResult:
    """Run Two-Sample Independent T-Test (Student's or Welch's t-test)."""
    g1, n_drop1 = clean_numeric_series(group1)
    g2, n_drop2 = clean_numeric_series(group2)
    n1, n2 = len(g1), len(g2)

    research_q = f"Is there a statistically significant difference in mean '{variable_name}' between '{group1_name}' and '{group2_name}'?"
    h0 = f"The true population mean '{variable_name}' is equal between '{group1_name}' and '{group2_name}' (μ1 - μ2 = 0)."
    h1 = f"The true population mean '{variable_name}' differs between '{group1_name}' and '{group2_name}' (μ1 - μ2 ≠ 0)."

    if n1 < 2 or n2 < 2:
        return HypothesisTestResult(
            test_name="Independent Samples T-Test",
            research_question=research_q,
            h0=h0,
            h1=h1,
            test_statistic=0.0,
            statistic_name="t",
            p_value=1.0,
            alpha=alpha,
            interpretation=f"Insufficient sample size (N1={n1}, N2={n2}). Minimum 2 observations required in each group.",
            limitations=["Sample size is too small to perform independent t-test."],
            sample_sizes={group1_name: n1, group2_name: n2},
        )

    # Check normality assumptions
    norm1 = check_normality(g1, alpha=alpha)
    norm2 = check_normality(g2, alpha=alpha)

    # Check zero variance in both groups
    v1 = float(g1.var(ddof=1)) if n1 > 1 else 0.0
    v2 = float(g2.var(ddof=1)) if n2 > 1 else 0.0
    if v1 == 0.0 and v2 == 0.0:
        m1, m2 = float(g1.mean()), float(g2.mean())
        t_stat = 0.0 if m1 == m2 else (float("inf") if m1 > m2 else float("-inf"))
        p_val = 1.0 if m1 == m2 else 0.0
        use_equal_var = True
    else:
        # Check variance homogeneity
        var_res = check_equal_variance([g1, g2], alpha=alpha)

        # Determine equal variance flag
        if equal_var == "auto":
            use_equal_var = bool(var_res.get("equal_variance", True))
        elif isinstance(equal_var, bool):
            use_equal_var = equal_var
        else:
            use_equal_var = equal_var.lower() == "true"

        # Execute t-test
        res = stats.ttest_ind(g1, g2, equal_var=use_equal_var, nan_policy="omit")
        t_stat = float(res.statistic) if not np.isnan(res.statistic) else 0.0
        p_val = float(res.pvalue) if not np.isnan(res.pvalue) else 1.0

    test_variant = "Student's Two-Sample T-Test" if use_equal_var else "Welch's T-Test (Unequal Variances)"

    # Effect size and CI
    d_val, d_interp = compute_cohens_d(g1, g2, pooled=use_equal_var)
    diff, ci_low, ci_high = ci_difference_in_means(g1, g2, confidence=0.95, equal_var=use_equal_var)

    m1, m2 = float(g1.mean()), float(g2.mean())
    sig_text = "statistically significant" if p_val < alpha else "not statistically significant"

    interpretation = (
        f"The difference in mean {variable_name} between {group1_name} (M={m1:.2f}) and {group2_name} "
        f"(M={m2:.2f}) is {sig_text} (t = {t_stat:.3f}, p = {p_val:.4f}, α = {alpha}). "
        f"Mean difference is {diff:+.2f} with a 95% CI of [{ci_low:.2f}, {ci_high:.2f}], "
        f"representing a {d_interp.lower()} (Cohen's d = {d_val:.2f})."
    )

    limitations = [
        "Statistical significance does not imply practical importance.",
        "Observational differences do not establish a causal relationship.",
    ]
    if not use_equal_var:
        limitations.append("Heterogeneity of variance was detected; Welch's t-test with adjusted degrees of freedom was used.")
    if not (norm1["is_normal"] and norm2["is_normal"]):
        limitations.append("Departure from normality detected in one or both cohorts; consider reviewing Mann-Whitney U test.")
    if n_drop1 + n_drop2 > 0:
        limitations.append(f"Excluded {n_drop1 + n_drop2} missing observations before testing.")

    return HypothesisTestResult(
        test_name=test_variant,
        research_question=research_q,
        h0=h0,
        h1=h1,
        test_statistic=t_stat,
        statistic_name="t",
        p_value=p_val,
        alpha=alpha,
        effect_size=d_val,
        effect_size_name="Cohen's d",
        effect_size_interpretation=d_interp,
        ci_lower=ci_low,
        ci_upper=ci_high,
        confidence_level=0.95,
        interpretation=interpretation,
        assumptions_met={
            "equal_variance": use_equal_var,
            "normality_group1": norm1["is_normal"],
            "normality_group2": norm2["is_normal"],
        },
        limitations=limitations,
        sample_sizes={group1_name: n1, group2_name: n2},
        additional_metrics={
            "mean1": m1,
            "mean2": m2,
            "difference": diff,
            "welch_applied": not use_equal_var,
        },
    )


def paired_t_test(
    before_series: pd.Series,
    after_series: pd.Series,
    before_name: str = "Condition 1 / Pre",
    after_name: str = "Condition 2 / Post",
    variable_name: str = "Metric",
    alpha: float = 0.05,
) -> HypothesisTestResult:
    """Run Paired Samples T-Test for matched/repeated measurements."""
    s1, s2, n_drop = clean_paired_series(before_series, after_series)
    n = len(s1)

    research_q = f"Is there a statistically significant difference in paired '{variable_name}' between '{before_name}' and '{after_name}'?"
    h0 = f"The true mean of paired differences is zero (μ_diff = 0)."
    h1 = f"The true mean of paired differences is not zero (μ_diff ≠ 0)."

    if n < 2:
        return HypothesisTestResult(
            test_name="Paired Samples T-Test",
            research_question=research_q,
            h0=h0,
            h1=h1,
            test_statistic=0.0,
            statistic_name="t",
            p_value=1.0,
            alpha=alpha,
            interpretation=f"Insufficient matched pairs (N={n}). Minimum 2 matched pairs required.",
            limitations=["Observations could not be matched with sufficient sample size."],
            sample_sizes={"Matched Pairs": n},
        )

    diffs = s2 - s1
    res = stats.ttest_rel(s2, s1, nan_policy="omit")
    t_stat = float(res.statistic) if not np.isnan(res.statistic) else 0.0
    p_val = float(res.pvalue) if not np.isnan(res.pvalue) else 1.0

    mean_d, ci_low, ci_high = ci_paired_difference(diffs, confidence=0.95)
    d_val, d_interp = compute_paired_cohens_d(diffs)

    sig_text = "statistically significant" if p_val < alpha else "not statistically significant"
    interpretation = (
        f"The paired mean difference ({after_name} minus {before_name}) of {mean_d:+.2f} is {sig_text} "
        f"(t = {t_stat:.3f}, p = {p_val:.4f}, α = {alpha}). "
        f"The 95% confidence interval of the mean difference is [{ci_low:.2f}, {ci_high:.2f}] "
        f"with a {d_interp.lower()} (Cohen's d_z = {d_val:.2f})."
    )

    limitations = [
        "Paired t-test assumes observations are matched and independent across pairs.",
        "External confounders during the intervening interval cannot be ruled out without experimental control.",
    ]
    if n_drop > 0:
        limitations.append(f"Excluded {n_drop} unmatched or incomplete record pairs.")

    return HypothesisTestResult(
        test_name="Paired Samples T-Test",
        research_question=research_q,
        h0=h0,
        h1=h1,
        test_statistic=t_stat,
        statistic_name="t",
        p_value=p_val,
        alpha=alpha,
        effect_size=d_val,
        effect_size_name="Cohen's d_z",
        effect_size_interpretation=d_interp,
        ci_lower=ci_low,
        ci_upper=ci_high,
        confidence_level=0.95,
        interpretation=interpretation,
        sample_sizes={"Matched Pairs": n},
        limitations=limitations,
        additional_metrics={"mean_difference": mean_d},
    )


def one_way_anova(
    groups_dict: Dict[str, pd.Series],
    outcome_name: str = "Metric",
    group_col_name: str = "Group",
    alpha: float = 0.05,
) -> HypothesisTestResult:
    """Run One-Way Analysis of Variance (ANOVA) across three or more groups."""
    valid_groups = {}
    for name, s in groups_dict.items():
        clean_s, _ = clean_numeric_series(s)
        if len(clean_s) > 1:
            valid_groups[name] = clean_s

    k = len(valid_groups)
    research_q = f"Does the mean of '{outcome_name}' differ across groups of '{group_col_name}' ({', '.join(valid_groups.keys())})?"
    h0 = f"The true means of '{outcome_name}' are equal across all groups (μ1 = μ2 = ... = μk)."
    h1 = f"At least one group mean of '{outcome_name}' differs significantly from the others."

    if k < 2:
        return HypothesisTestResult(
            test_name="One-Way ANOVA",
            research_question=research_q,
            h0=h0,
            h1=h1,
            test_statistic=0.0,
            statistic_name="F",
            p_value=1.0,
            alpha=alpha,
            interpretation="Fewer than 2 valid groups available for ANOVA comparison.",
            limitations=["At least two distinct groups with N > 1 are required."],
        )

    arrays = list(valid_groups.values())
    res = stats.f_oneway(*arrays)
    f_stat = float(res.statistic) if not np.isnan(res.statistic) else 0.0
    p_val = float(res.pvalue) if not np.isnan(res.pvalue) else 1.0

    total_n = sum(len(a) for a in arrays)
    df_between = k - 1
    df_within = total_n - k

    eta_sq, eta_interp = compute_eta_squared(f_stat, df_between, df_within)

    sig_text = "statistically significant" if p_val < alpha else "not statistically significant"
    interpretation = (
        f"The omnibus difference in '{outcome_name}' across {k} groups is {sig_text} "
        f"(F({df_between}, {df_within}) = {f_stat:.3f}, p = {p_val:.4f}, α = {alpha}). "
        f"Grouping explains approximately {eta_sq * 100:.1f}% of variance ({eta_interp}, η² = {eta_sq:.3f})."
    )

    limitations = [
        "ANOVA tests the omnibus hypothesis only; post-hoc pairwise testing is required to identify specific differing pairs.",
        "Assumes independence of observations, approximate normality, and homogeneity of variances.",
    ]

    return HypothesisTestResult(
        test_name="One-Way ANOVA",
        research_question=research_q,
        h0=h0,
        h1=h1,
        test_statistic=f_stat,
        statistic_name="F",
        p_value=p_val,
        alpha=alpha,
        effect_size=eta_sq,
        effect_size_name="Eta-squared (η²)",
        effect_size_interpretation=eta_interp,
        interpretation=interpretation,
        sample_sizes={name: len(s) for name, s in valid_groups.items()},
        limitations=limitations,
        additional_metrics={"df_between": df_between, "df_within": df_within},
    )


def mann_whitney_u_test(
    group1: pd.Series,
    group2: pd.Series,
    group1_name: str = "Group 1",
    group2_name: str = "Group 2",
    variable_name: str = "Metric",
    alpha: float = 0.05,
) -> HypothesisTestResult:
    """Non-parametric Mann-Whitney U test for two independent groups."""
    g1, _ = clean_numeric_series(group1)
    g2, _ = clean_numeric_series(group2)
    n1, n2 = len(g1), len(g2)

    research_q = f"Does the distribution of '{variable_name}' differ between '{group1_name}' and '{group2_name}'?"
    h0 = f"The distributions of '{variable_name}' in '{group1_name}' and '{group2_name}' are identical."
    h1 = f"One group tends to have systematically higher/lower values than the other."

    if n1 < 1 or n2 < 1:
        return HypothesisTestResult(
            test_name="Mann-Whitney U Test",
            research_question=research_q,
            h0=h0,
            h1=h1,
            test_statistic=0.0,
            statistic_name="U",
            p_value=1.0,
            alpha=alpha,
            interpretation="Insufficient data points for Mann-Whitney U test.",
        )

    res = stats.mannwhitneyu(g1, g2, alternative="two-sided")
    u_stat = float(res.statistic)
    p_val = float(res.pvalue)

    # Rank-biserial correlation effect size r = 1 - (2*U)/(n1*n2)
    r_biserial = 1.0 - (2.0 * u_stat) / (n1 * n2) if (n1 * n2) > 0 else 0.0
    abs_r = abs(r_biserial)
    r_interp = "Small" if abs_r < 0.3 else ("Medium" if abs_r < 0.5 else "Large")

    sig_text = "statistically significant" if p_val < alpha else "not statistically significant"
    interpretation = (
        f"The non-parametric rank comparison for '{variable_name}' between {group1_name} (median={g1.median():.2f}) "
        f"and {group2_name} (median={g2.median():.2f}) is {sig_text} (U = {u_stat:.1f}, p = {p_val:.4f}, α = {alpha}). "
        f"Rank-biserial correlation is {r_biserial:+.2f} ({r_interp} effect)."
    )

    return HypothesisTestResult(
        test_name="Mann-Whitney U Test",
        research_question=research_q,
        h0=h0,
        h1=h1,
        test_statistic=u_stat,
        statistic_name="U",
        p_value=p_val,
        alpha=alpha,
        effect_size=round(r_biserial, 4),
        effect_size_name="Rank-Biserial Correlation",
        effect_size_interpretation=f"{r_interp} effect",
        interpretation=interpretation,
        sample_sizes={group1_name: n1, group2_name: n2},
        limitations=[
            "Mann-Whitney U tests for stochastic dominance (rank distribution shift), not strictly a difference in medians unless shapes are identical.",
            "Does not assume normality, making it robust against extreme outliers and severe skewness.",
        ],
    )


def wilcoxon_signed_rank_test(
    before_series: pd.Series,
    after_series: pd.Series,
    variable_name: str = "Metric",
    alpha: float = 0.05,
) -> HypothesisTestResult:
    """Non-parametric Wilcoxon Signed-Rank test for paired observations."""
    s1, s2, n_drop = clean_paired_series(before_series, after_series)
    diffs = s2 - s1
    # Wilcoxon requires non-zero differences
    non_zero_diffs = diffs[diffs != 0]
    n = len(non_zero_diffs)

    research_q = f"Is the median of paired differences for '{variable_name}' significantly different from zero?"
    h0 = f"The distribution of paired differences is symmetric about zero."
    h1 = f"The distribution of paired differences is not symmetric about zero."

    if n < 2:
        return HypothesisTestResult(
            test_name="Wilcoxon Signed-Rank Test",
            research_question=research_q,
            h0=h0,
            h1=h1,
            test_statistic=0.0,
            statistic_name="W",
            p_value=1.0,
            alpha=alpha,
            interpretation="Insufficient non-zero paired differences for Wilcoxon test.",
        )

    res = stats.wilcoxon(s1, s2)
    w_stat = float(res.statistic)
    p_val = float(res.pvalue)

    sig_text = "statistically significant" if p_val < alpha else "not statistically significant"
    interpretation = (
        f"The non-parametric paired comparison for '{variable_name}' across {n} non-zero paired differences is {sig_text} "
        f"(W = {w_stat:.1f}, p = {p_val:.4f}, α = {alpha})."
    )

    return HypothesisTestResult(
        test_name="Wilcoxon Signed-Rank Test",
        research_question=research_q,
        h0=h0,
        h1=h1,
        test_statistic=w_stat,
        statistic_name="W",
        p_value=p_val,
        alpha=alpha,
        interpretation=interpretation,
        sample_sizes={"Valid Pairs": n},
        limitations=[
            "Non-parametric alternative to paired t-test; does not assume normal differences.",
            "Zero difference pairs are excluded from rank sum calculations.",
        ],
    )


def kruskal_wallis_test(
    groups_dict: Dict[str, pd.Series],
    outcome_name: str = "Metric",
    group_col_name: str = "Group",
    alpha: float = 0.05,
) -> HypothesisTestResult:
    """Non-parametric Kruskal-Wallis H-test across three or more groups."""
    valid_groups = {}
    for name, s in groups_dict.items():
        clean_s, _ = clean_numeric_series(s)
        if len(clean_s) > 0:
            valid_groups[name] = clean_s

    k = len(valid_groups)
    research_q = f"Do the median ranks of '{outcome_name}' differ across groups of '{group_col_name}'?"
    h0 = f"The population distributions of '{outcome_name}' are identical across all groups."
    h1 = f"At least one group has a systematically different distribution."

    if k < 2:
        return HypothesisTestResult(
            test_name="Kruskal-Wallis Test",
            research_question=research_q,
            h0=h0,
            h1=h1,
            test_statistic=0.0,
            statistic_name="H",
            p_value=1.0,
            alpha=alpha,
            interpretation="At least two groups are required for Kruskal-Wallis test.",
        )

    arrays = list(valid_groups.values())
    res = stats.kruskal(*arrays)
    h_stat = float(res.statistic)
    p_val = float(res.pvalue)

    total_n = sum(len(a) for a in arrays)
    # Epsilon-squared effect size: eps2 = (H - k + 1) / (N - k)
    eps_sq = (h_stat - k + 1) / (total_n - k) if (total_n - k) > 0 else 0.0
    eps_sq = float(np.clip(eps_sq, 0.0, 1.0))

    sig_text = "statistically significant" if p_val < alpha else "not statistically significant"
    interpretation = (
        f"The non-parametric rank variance across {k} groups is {sig_text} "
        f"(H = {h_stat:.3f}, df = {k - 1}, p = {p_val:.4f}, α = {alpha}). "
        f"Epsilon-squared effect size is {eps_sq:.3f}."
    )

    return HypothesisTestResult(
        test_name="Kruskal-Wallis Test",
        research_question=research_q,
        h0=h0,
        h1=h1,
        test_statistic=h_stat,
        statistic_name="H",
        p_value=p_val,
        alpha=alpha,
        effect_size=round(eps_sq, 4),
        effect_size_name="Epsilon-squared (ε²)",
        interpretation=interpretation,
        sample_sizes={name: len(s) for name, s in valid_groups.items()},
        limitations=[
            "Non-parametric alternative to one-way ANOVA; robust against non-normal distributions.",
            "Does not pinpoint which specific groups differ; post-hoc Dunn's testing is recommended.",
        ],
    )
