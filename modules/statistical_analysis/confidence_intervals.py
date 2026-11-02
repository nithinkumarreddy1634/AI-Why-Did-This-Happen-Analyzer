"""Confidence interval calculation routines for means, proportions, and correlations."""

from typing import Tuple, Optional
import numpy as np
import pandas as pd
from scipy import stats


def ci_difference_in_means(
    group1: pd.Series,
    group2: pd.Series,
    confidence: float = 0.95,
    equal_var: bool = True,
) -> Tuple[float, float, float]:
    """Compute confidence interval for difference between two independent means (mean1 - mean2).
    
    Supports both Student's pooled variance and Welch-Satterthwaite unequal variances.
    Returns:
        (diff, ci_lower, ci_upper)
    """
    g1 = group1.dropna().astype(float)
    g2 = group2.dropna().astype(float)
    n1, n2 = len(g1), len(g2)

    if n1 < 2 or n2 < 2:
        return 0.0, 0.0, 0.0

    m1, m2 = float(g1.mean()), float(g2.mean())
    diff = m1 - m2
    v1, v2 = float(g1.var(ddof=1)), float(g2.var(ddof=1))
    alpha = 1.0 - confidence

    if equal_var:
        df = n1 + n2 - 2
        s_pooled = np.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / max(df, 1))
        se = s_pooled * np.sqrt((1.0 / n1) + (1.0 / n2))
    else:
        # Welch-Satterthwaite degrees of freedom
        se = np.sqrt((v1 / n1) + (v2 / n2))
        if se == 0:
            return diff, diff, diff
        df_num = ((v1 / n1) + (v2 / n2)) ** 2
        df_den = ((v1 / n1) ** 2 / (n1 - 1)) + ((v2 / n2) ** 2 / (n2 - 1))
        df = df_num / df_den if df_den > 0 else (n1 + n2 - 2)

    t_crit = stats.t.ppf(1.0 - alpha / 2.0, df=df)
    ci_lower = diff - (t_crit * se)
    ci_upper = diff + (t_crit * se)
    return round(diff, 4), round(float(ci_lower), 4), round(float(ci_upper), 4)


def ci_paired_difference(
    differences: pd.Series,
    confidence: float = 0.95,
) -> Tuple[float, float, float]:
    """Compute confidence interval for the mean of paired differences.
    
    Returns:
        (mean_diff, ci_lower, ci_upper)
    """
    diffs = differences.dropna().astype(float)
    n = len(diffs)
    if n < 2:
        return 0.0, 0.0, 0.0

    mean_d = float(diffs.mean())
    std_d = float(diffs.std(ddof=1))
    se = std_d / np.sqrt(n)
    alpha = 1.0 - confidence
    t_crit = stats.t.ppf(1.0 - alpha / 2.0, df=n - 1)

    ci_lower = mean_d - (t_crit * se)
    ci_upper = mean_d + (t_crit * se)
    return round(mean_d, 4), round(float(ci_lower), 4), round(float(ci_upper), 4)


def ci_difference_in_proportions(
    count1: int,
    n1: int,
    count2: int,
    n2: int,
    confidence: float = 0.95,
) -> Tuple[float, float, float]:
    """Compute confidence interval for difference between two independent proportions (p1 - p2).
    
    Returns:
        (diff, ci_lower, ci_upper)
    """
    if n1 <= 0 or n2 <= 0:
        return 0.0, 0.0, 0.0

    p1 = count1 / n1
    p2 = count2 / n2
    diff = p1 - p2
    se = np.sqrt((p1 * (1.0 - p1) / n1) + (p2 * (1.0 - p2) / n2))

    alpha = 1.0 - confidence
    z_crit = stats.norm.ppf(1.0 - alpha / 2.0)

    ci_lower = max(-1.0, diff - (z_crit * se))
    ci_upper = min(1.0, diff + (z_crit * se))
    return round(diff, 4), round(float(ci_lower), 4), round(float(ci_upper), 4)


def ci_pearson_correlation(
    r: float,
    n: int,
    confidence: float = 0.95,
) -> Tuple[float, float, float]:
    """Compute Fisher z-transform confidence interval for Pearson correlation coefficient r.
    
    Formula:
        z = arctanh(r) = 0.5 * ln((1 + r) / (1 - r))
        SE_z = 1 / sqrt(n - 3)
    """
    if n < 4 or np.isnan(r) or abs(r) >= 1.0:
        return r, r, r

    z = np.arctanh(r)
    se_z = 1.0 / np.sqrt(n - 3)
    alpha = 1.0 - confidence
    z_crit = stats.norm.ppf(1.0 - alpha / 2.0)

    z_low = z - (z_crit * se_z)
    z_high = z + (z_crit * se_z)

    r_low = float(np.tanh(z_low))
    r_high = float(np.tanh(z_high))
    return round(r, 4), round(r_low, 4), round(r_high, 4)

