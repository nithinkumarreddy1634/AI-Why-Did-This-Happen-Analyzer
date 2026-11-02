"""Correlation significance testing: Pearson, Spearman rank, and Kendall rank."""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from scipy import stats

from modules.statistical_analysis.hypothesis_testing import HypothesisTestResult
from modules.statistical_analysis.confidence_intervals import ci_pearson_correlation
from utils.statistical_helpers import clean_paired_series


def compute_pearson_correlation(
    x: pd.Series,
    y: pd.Series,
    x_name: str = "Variable X",
    y_name: str = "Variable Y",
    alpha: float = 0.05,
) -> HypothesisTestResult:
    """Run Pearson Linear Correlation significance test."""
    clean_x, clean_y, n_drop = clean_paired_series(x, y)
    n = len(clean_x)

    research_q = f"Is there a statistically significant linear correlation between '{x_name}' and '{y_name}'?"
    h0 = f"There is no linear association between '{x_name}' and '{y_name}' (ρ = 0)."
    h1 = f"There is a non-zero linear association between '{x_name}' and '{y_name}' (ρ ≠ 0)."

    if n < 3 or clean_x.nunique() <= 1 or clean_y.nunique() <= 1:
        return HypothesisTestResult(
            test_name="Pearson Linear Correlation",
            research_question=research_q,
            h0=h0,
            h1=h1,
            test_statistic=0.0,
            statistic_name="r",
            p_value=1.0,
            alpha=alpha,
            interpretation="Insufficient data or zero variance in one/both variables (N ≥ 3 required).",
            limitations=["Requires variation in both variables and at least 3 pairs."],
            sample_sizes={"Valid Pairs": n},
        )

    res = stats.pearsonr(clean_x, clean_y)
    r_val = float(res.statistic)
    p_val = float(res.pvalue)

    # 95% Confidence Interval via Fisher z-transform
    r_val, ci_low, ci_high = ci_pearson_correlation(r_val, n, confidence=0.95)

    # Interpretation
    abs_r = abs(r_val)
    if abs_r < 0.3:
        strength = "Weak"
    elif abs_r < 0.7:
        strength = "Moderate"
    else:
        strength = "Strong"
    direction = "positive" if r_val > 0 else "negative"

    sig_text = "statistically significant" if p_val < alpha else "not statistically significant"
    interpretation = (
        f"The linear relationship between '{x_name}' and '{y_name}' is {sig_text} "
        f"(r = {r_val:.3f}, p = {p_val:.4f}, α = {alpha}). "
        f"The association exhibits a {strength.lower()} {direction} trend with a 95% CI of [{ci_low:.3f}, {ci_high:.3f}]."
    )

    limitations = [
        "Important: Correlation indicates statistical association, NOT causation.",
        "Pearson r measures strictly linear relationships; non-linear dependencies may go undetected.",
    ]
    if n_drop > 0:
        limitations.append(f"Excluded {n_drop} incomplete pairs.")

    return HypothesisTestResult(
        test_name="Pearson Linear Correlation",
        research_question=research_q,
        h0=h0,
        h1=h1,
        test_statistic=r_val,
        statistic_name="r",
        p_value=p_val,
        alpha=alpha,
        effect_size=r_val,
        effect_size_name="Pearson r",
        effect_size_interpretation=f"{strength} {direction}",
        ci_lower=ci_low,
        ci_upper=ci_high,
        confidence_level=0.95,
        interpretation=interpretation,
        sample_sizes={"Valid Pairs (N)": n},
        limitations=limitations,
        additional_metrics={
            "r_squared": round(r_val**2, 4),
            "strength": strength,
            "direction": direction,
        },
    )


def compute_spearman_correlation(
    x: pd.Series,
    y: pd.Series,
    x_name: str = "Variable X",
    y_name: str = "Variable Y",
    alpha: float = 0.05,
) -> HypothesisTestResult:
    """Run Spearman Rank-Order Monotonic Correlation test."""
    clean_x, clean_y, n_drop = clean_paired_series(x, y)
    n = len(clean_x)

    research_q = f"Is there a statistically significant monotonic relationship between ranks of '{x_name}' and '{y_name}'?"
    h0 = f"There is no monotonic association between '{x_name}' and '{y_name}' (ρ_s = 0)."
    h1 = f"There is a monotonic association between '{x_name}' and '{y_name}' (ρ_s ≠ 0)."

    if n < 3 or clean_x.nunique() <= 1 or clean_y.nunique() <= 1:
        return HypothesisTestResult(
            test_name="Spearman Rank Correlation",
            research_question=research_q,
            h0=h0,
            h1=h1,
            test_statistic=0.0,
            statistic_name="ρ",
            p_value=1.0,
            alpha=alpha,
            interpretation="Insufficient data or constant values.",
        )

    res = stats.spearmanr(clean_x, clean_y)
    rho_val = float(res.statistic)
    p_val = float(res.pvalue)

    abs_rho = abs(rho_val)
    strength = "Weak" if abs_rho < 0.3 else ("Moderate" if abs_rho < 0.7 else "Strong")
    direction = "positive" if rho_val > 0 else "negative"

    sig_text = "statistically significant" if p_val < alpha else "not statistically significant"
    interpretation = (
        f"The non-parametric rank association between '{x_name}' and '{y_name}' is {sig_text} "
        f"(ρ = {rho_val:.3f}, p = {p_val:.4f}, α = {alpha}). "
        f"Reflects a {strength.lower()} {direction} monotonic trend."
    )

    return HypothesisTestResult(
        test_name="Spearman Rank Correlation",
        research_question=research_q,
        h0=h0,
        h1=h1,
        test_statistic=rho_val,
        statistic_name="ρ",
        p_value=p_val,
        alpha=alpha,
        effect_size=rho_val,
        effect_size_name="Spearman ρ",
        effect_size_interpretation=f"{strength} {direction}",
        interpretation=interpretation,
        sample_sizes={"Valid Pairs (N)": n},
        limitations=[
            "Spearman rank test is robust against outliers and non-linear monotonic trends.",
            "Statistical association does NOT imply causal mechanism.",
        ],
    )


def compute_kendall_correlation(
    x: pd.Series,
    y: pd.Series,
    x_name: str = "Variable X",
    y_name: str = "Variable Y",
    alpha: float = 0.05,
) -> HypothesisTestResult:
    """Run Kendall's Tau Rank Correlation test (ideal for small samples and tied ranks)."""
    clean_x, clean_y, n_drop = clean_paired_series(x, y)
    n = len(clean_x)

    research_q = f"Is there a statistically significant concordant/discordant relationship between '{x_name}' and '{y_name}'?"
    h0 = f"Ranks of '{x_name}' and '{y_name}' are mutually independent (τ = 0)."
    h1 = f"There is a concordant or discordant rank association (τ ≠ 0)."

    if n < 3 or clean_x.nunique() <= 1 or clean_y.nunique() <= 1:
        return HypothesisTestResult(
            test_name="Kendall's Tau Correlation",
            research_question=research_q,
            h0=h0,
            h1=h1,
            test_statistic=0.0,
            statistic_name="τ",
            p_value=1.0,
            alpha=alpha,
            interpretation="Insufficient data or constant values.",
        )

    res = stats.kendalltau(clean_x, clean_y)
    tau_val = float(res.statistic)
    p_val = float(res.pvalue)

    abs_tau = abs(tau_val)
    strength = "Weak" if abs_tau < 0.25 else ("Moderate" if abs_tau < 0.55 else "Strong")
    direction = "positive" if tau_val > 0 else "negative"

    sig_text = "statistically significant" if p_val < alpha else "not statistically significant"
    interpretation = (
        f"Kendall rank correlation between '{x_name}' and '{y_name}' is {sig_text} "
        f"(τ = {tau_val:.3f}, p = {p_val:.4f}, α = {alpha}). "
        f"Shows a {strength.lower()} {direction} concordance."
    )

    return HypothesisTestResult(
        test_name="Kendall's Tau Correlation",
        research_question=research_q,
        h0=h0,
        h1=h1,
        test_statistic=tau_val,
        statistic_name="τ",
        p_value=p_val,
        alpha=alpha,
        effect_size=tau_val,
        effect_size_name="Kendall τ",
        effect_size_interpretation=f"{strength} {direction}",
        interpretation=interpretation,
        sample_sizes={"Valid Pairs (N)": n},
        limitations=[
            "Kendall's tau has smaller gross error sensitivity than Spearman rho in small datasets.",
            "Does NOT imply causal influence.",
        ],
    )
