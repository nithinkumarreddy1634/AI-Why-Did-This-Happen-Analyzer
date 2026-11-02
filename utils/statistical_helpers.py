"""Statistical analysis helper functions and validation routines.

Provides data cleaning, paired validation, safe arithmetic, and formatting
utilities for hypothesis testing engines.
"""

from typing import Tuple, Optional, Any, Dict, List
import numpy as np
import pandas as pd


def clean_numeric_series(series: pd.Series) -> Tuple[pd.Series, int]:
    """Clean a pandas series into numeric floats, removing nulls and infs.
    
    Returns:
        (clean_series, missing_count)
    """
    if series is None or series.empty:
        return pd.Series(dtype=float), 0

    s_num = pd.to_numeric(series, errors="coerce")
    valid_mask = s_num.notna() & np.isfinite(s_num)
    clean = s_num[valid_mask].astype(float)
    missing_count = len(series) - len(clean)
    return clean, missing_count


def clean_paired_series(
    series1: pd.Series, series2: pd.Series
) -> Tuple[pd.Series, pd.Series, int]:
    """Pairwise-clean two numerical series for paired tests.
    
    Returns:
        (clean_s1, clean_s2, dropped_count)
    """
    if series1 is None or series2 is None or len(series1) == 0 or len(series2) == 0:
        return pd.Series(dtype=float), pd.Series(dtype=float), 0

    s1_num = pd.to_numeric(series1, errors="coerce")
    s2_num = pd.to_numeric(series2, errors="coerce")
    
    valid_mask = (
        s1_num.notna() & np.isfinite(s1_num) &
        s2_num.notna() & np.isfinite(s2_num)
    )
    clean_s1 = s1_num[valid_mask].astype(float)
    clean_s2 = s2_num[valid_mask].astype(float)
    dropped_count = max(len(series1), len(series2)) - len(clean_s1)
    return clean_s1, clean_s2, dropped_count


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Divide two numbers safely, preventing ZeroDivisionError and NaN."""
    if denominator == 0.0 or np.isnan(denominator) or np.isnan(numerator):
        return default
    return float(numerator / denominator)


def format_p_value(p_val: Optional[float]) -> str:
    """Format p-value for reporting with standard scientific precision."""
    if p_val is None or np.isnan(p_val):
        return "N/A"
    if p_val < 0.0001:
        return "< 0.0001"
    if p_val < 0.001:
        return f"{p_val:.4f}"
    return f"{p_val:.4f}"


def format_stat(val: Optional[float], decimals: int = 3) -> str:
    """Format floating point numbers safely with specified precision."""
    if val is None or np.isnan(val):
        return "N/A"
    return f"{val:.{decimals}f}"

