"""Statistical assumption validation: normality and homogeneity of variance."""

from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
from scipy import stats


def check_normality(
    series: pd.Series, alpha: float = 0.05
) -> Dict[str, Any]:
    """Evaluate normality using Shapiro-Wilk or D'Agostino-Pearson tests and skew/kurtosis.
    
    Includes caveats regarding small and large sample sizes.
    """
    clean = series.dropna().astype(float)
    n = len(clean)

    if n < 3:
        return {
            "test_name": "Normality Test",
            "sample_size": n,
            "status": "warning",
            "is_normal": True,
            "message": "Sample size too small (< 3) to test normality reliably.",
            "skewness": 0.0,
            "kurtosis": 0.0,
            "p_value": 1.0,
            "statistic": 0.0,
        }

    # Constant column check
    if clean.nunique() <= 1:
        return {
            "test_name": "Constant Check",
            "sample_size": n,
            "status": "warning",
            "is_normal": False,
            "message": "Column has zero variance (constant values).",
            "skewness": 0.0,
            "kurtosis": 0.0,
            "p_value": 0.0,
            "statistic": 0.0,
        }

    # Skewness and excess kurtosis
    skew = float(stats.skew(clean))
    kurt = float(stats.kurtosis(clean))

    # Shapiro-Wilk for n <= 5000, D'Agostino-Pearson for larger n
    if n <= 5000:
        test_name = "Shapiro-Wilk Test"
        stat_name = "W"
        stat, p_val = stats.shapiro(clean)
    else:
        test_name = "D'Agostino's K-squared Test"
        stat_name = "K²"
        stat, p_val = stats.normaltest(clean)

    is_normal = bool(p_val >= alpha)

    caveat = ""
    if n > 1000:
        caveat = "With large sample sizes (N > 1,000), normality tests have extreme power and often reject H0 for trivial, benign departures that do not violate the Central Limit Theorem."
    elif n < 20:
        caveat = "With small sample sizes (N < 20), normality tests have low statistical power to detect meaningful non-normality."

    return {
        "test_name": test_name,
        "statistic_name": stat_name,
        "statistic": round(float(stat), 4),
        "p_value": float(p_val),
        "sample_size": n,
        "skewness": round(skew, 3),
        "kurtosis": round(kurt, 3),
        "is_normal": is_normal,
        "alpha": alpha,
        "status": "success",
        "caveat": caveat,
        "interpretation": (
            f"Data appears normally distributed (p = {p_val:.4f} ≥ {alpha})."
            if is_normal
            else f"Data departs significantly from a normal distribution (p = {p_val:.4f} < {alpha})."
        ),
    }


def generate_qq_plot_data(series: pd.Series) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """Compute theoretical normal quantiles and sample quantiles for Q-Q plotting.
    
    Returns:
        (theoretical_quantiles, sample_quantiles, slope, intercept)
    """
    clean = series.dropna().astype(float)
    if len(clean) < 3:
        return np.array([]), np.array([]), 1.0, 0.0

    (osm, osr), (slope, intercept, r) = stats.probplot(clean, dist="norm")
    return np.array(osm), np.array(osr), float(slope), float(intercept)


def check_equal_variance(
    groups: List[pd.Series], alpha: float = 0.05, center: str = "median"
) -> Dict[str, Any]:
    """Evaluate homogeneity of variance across multiple groups using Levene's test.
    
    Args:
        groups: List of pandas Series.
        alpha: Significance threshold.
        center: 'median' (Brown-Forsythe robust test) or 'mean' (standard Levene).
    """
    clean_groups = [g.dropna().astype(float) for g in groups if len(g.dropna()) > 1]

    if len(clean_groups) < 2:
        return {
            "test_name": "Levene's Test",
            "status": "warning",
            "equal_variance": True,
            "message": "At least two valid groups with N > 1 are required to test variance.",
            "statistic": 0.0,
            "p_value": 1.0,
        }

    total_n = sum(len(g) for g in clean_groups)
    if total_n <= len(clean_groups) + 1:
        return {
            "test_name": "Levene's Test",
            "status": "warning",
            "equal_variance": True,
            "message": "Insufficient degrees of freedom to evaluate variance across groups.",
            "statistic": 0.0,
            "p_value": 1.0,
        }

    # Zero variance check
    variances = [float(g.var(ddof=1)) for g in clean_groups]
    if all(v == 0.0 for v in variances):
        return {
            "test_name": "Levene's Test",
            "status": "warning",
            "equal_variance": True,
            "message": "All groups have zero variance.",
            "statistic": 0.0,
            "p_value": 1.0,
        }

    try:
        stat, p_val = stats.levene(*clean_groups, center=center)
        if np.isnan(stat) or np.isnan(p_val):
            stat, p_val = 0.0, 1.0
    except Exception:
        stat, p_val = 0.0, 1.0

    equal_var = bool(p_val >= alpha)

    method_label = "Brown-Forsythe (Levene with Median)" if center == "median" else "Levene's Test (Mean-centered)"

    return {
        "test_name": method_label,
        "statistic_name": "W",
        "statistic": round(float(stat), 4),
        "p_value": float(p_val),
        "equal_variance": equal_var,
        "alpha": alpha,
        "group_sizes": [len(g) for g in clean_groups],
        "group_variances": [round(v, 4) for v in variances],
        "status": "success",
        "recommendation": (
            "Equal variances can be assumed. Standard Student's t-test or ANOVA is appropriate."
            if equal_var
            else "Equal variances cannot be assumed. Welch's t-test or Welch's ANOVA is strongly recommended."
        ),
        "interpretation": (
            f"Variances across groups are statistically homogeneous (p = {p_val:.4f} ≥ {alpha})."
            if equal_var
            else f"Variances across groups are significantly heterogeneous (p = {p_val:.4f} < {alpha})."
        ),
    }
