"""Effect size estimation functions and standardized magnitude interpreters."""

from typing import Optional, Tuple
import numpy as np
import pandas as pd


def compute_cohens_d(
    group1: pd.Series, group2: pd.Series, pooled: bool = True
) -> Tuple[float, str]:
    """Compute Cohen's d standardized mean difference between two independent groups.
    
    Formula:
        d = (mean1 - mean2) / s_pooled
    """
    g1 = group1.dropna().astype(float)
    g2 = group2.dropna().astype(float)
    n1, n2 = len(g1), len(g2)

    if n1 < 2 or n2 < 2:
        return 0.0, "Insufficient sample size"

    m1, m2 = float(g1.mean()), float(g2.mean())
    v1, v2 = float(g1.var(ddof=1)), float(g2.var(ddof=1))

    if pooled:
        pooled_var = ((n1 - 1) * v1 + (n2 - 1) * v2) / max(n1 + n2 - 2, 1)
        s_pooled = np.sqrt(pooled_var) if pooled_var > 0 else 0.0
    else:
        # Glass's delta or average std
        s_pooled = np.sqrt((v1 + v2) / 2.0)

    if s_pooled == 0.0:
        return 0.0, "Zero pooled variance"

    d = (m1 - m2) / s_pooled
    interpretation = interpret_cohens_d(d)
    return round(float(d), 4), interpretation


def compute_hedges_g(group1: pd.Series, group2: pd.Series) -> Tuple[float, str]:
    """Compute Hedges' g effect size with small-sample bias correction.
    
    Formula:
        g = d * (1 - 3 / (4 * (n1 + n2) - 9))
    """
    d, _ = compute_cohens_d(group1, group2)
    n1, n2 = len(group1.dropna()), len(group2.dropna())
    df = n1 + n2 - 2
    if df < 2:
        return d, "Insufficient sample size"

    # Correction factor J(df)
    j = 1.0 - (3.0 / (4.0 * df - 1.0))
    g = d * j
    interpretation = interpret_cohens_d(g)
    return round(float(g), 4), interpretation


def compute_paired_cohens_d(
    differences: pd.Series,
) -> Tuple[float, str]:
    """Compute Cohen's d for paired differences (d_z).
    
    Formula:
        d_z = mean(diff) / std(diff)
    """
    clean_diff = differences.dropna().astype(float)
    if len(clean_diff) < 2:
        return 0.0, "Insufficient observations"

    mean_diff = float(clean_diff.mean())
    std_diff = float(clean_diff.std(ddof=1))

    if std_diff == 0.0:
        return 0.0, "Zero variance in paired differences"

    d_z = mean_diff / std_diff
    interpretation = interpret_cohens_d(d_z)
    return round(float(d_z), 4), interpretation


def compute_eta_squared(
    f_stat: float, df_between: int, df_within: int
) -> Tuple[float, str]:
    """Compute Eta-squared effect size from ANOVA F-statistic and degrees of freedom.
    
    Formula:
        eta^2 = (F * df_between) / (F * df_between + df_within)
    """
    if f_stat <= 0 or np.isnan(f_stat) or df_between <= 0 or df_within <= 0:
        return 0.0, "Negligible effect"

    ss_ratio = (f_stat * df_between) / (f_stat * df_between + df_within)
    eta_sq = float(np.clip(ss_ratio, 0.0, 1.0))
    interpretation = interpret_eta_squared(eta_sq)
    return round(eta_sq, 4), interpretation


def compute_cramers_v(contingency_table: pd.DataFrame) -> Tuple[float, str]:
    """Compute Cramér's V effect size for categorical contingency tables.
    
    Formula:
        V = sqrt(chi2 / (N * (min(r, c) - 1)))
    """
    from scipy.stats import chi2_contingency

    if contingency_table.empty or contingency_table.shape[0] < 2 or contingency_table.shape[1] < 2:
        return 0.0, "Insufficient table dimensions"

    chi2, _, _, _ = chi2_contingency(contingency_table)
    n = contingency_table.values.sum()
    r, c = contingency_table.shape
    min_dim = min(r, c) - 1

    if n == 0 or min_dim < 1:
        return 0.0, "Negligible association"

    v = np.sqrt(chi2 / (n * min_dim))
    interpretation = interpret_cramers_v(v)
    return round(float(v), 4), interpretation


def interpret_cohens_d(d: float) -> str:
    """Standard Cohen (1988) heuristic benchmarks for standardized mean differences."""
    abs_d = abs(d)
    if abs_d < 0.2:
        return "Negligible effect (|d| < 0.2)"
    elif abs_d < 0.5:
        return "Small effect (0.2 ≤ |d| < 0.5)"
    elif abs_d < 0.8:
        return "Medium effect (0.5 ≤ |d| < 0.8)"
    else:
        return "Large effect (|d| ≥ 0.8)"


def interpret_eta_squared(eta_sq: float) -> str:
    """Standard Cohen benchmarks for variance explained in ANOVA."""
    if eta_sq < 0.01:
        return "Negligible effect (η² < 0.01)"
    elif eta_sq < 0.06:
        return "Small effect (0.01 ≤ η² < 0.06)"
    elif eta_sq < 0.14:
        return "Medium effect (0.06 ≤ η² < 0.14)"
    else:
        return "Large effect (η² ≥ 0.14)"


def interpret_cramers_v(v: float) -> str:
    """Standard Cramér's V benchmark for categorical association strength."""
    if v < 0.1:
        return "Negligible association (V < 0.1)"
    elif v < 0.3:
        return "Weak association (0.1 ≤ V < 0.3)"
    elif v < 0.5:
        return "Moderate association (0.3 ≤ V < 0.5)"
    else:
        return "Strong association (V ≥ 0.5)"

