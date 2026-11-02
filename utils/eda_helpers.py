"""Helper utilities for Exploratory Data Analysis (EDA) engine.

Provides type filtering, statistical calculations, and formatting routines.
"""

from typing import List, Dict, Any, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy import stats


def get_eda_numerical_columns(df: pd.DataFrame) -> List[str]:
    """Return column names that are strictly numeric (excluding booleans and identifiers)."""
    if df is None or df.empty:
        return []
    num_cols = []
    for col in df.columns:
        s = df[col]
        if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
            num_cols.append(col)
    return num_cols


def get_eda_categorical_columns(df: pd.DataFrame) -> List[str]:
    """Return column names that are categorical, object, string, or boolean."""
    if df is None or df.empty:
        return []
    cat_cols = []
    for col in df.columns:
        s = df[col]
        if (
            pd.api.types.is_object_dtype(s)
            or pd.api.types.is_string_dtype(s)
            or isinstance(s.dtype, pd.CategoricalDtype)
            or pd.api.types.is_bool_dtype(s)
        ):
            cat_cols.append(col)
    return cat_cols


def get_eda_datetime_columns(df: pd.DataFrame) -> List[str]:
    """Detect datetime columns or string columns that parse reliably as dates."""
    if df is None or df.empty:
        return []
    dt_cols = []
    for col in df.columns:
        s = df[col]
        if pd.api.types.is_datetime64_any_dtype(s) or pd.api.types.is_timedelta64_dtype(s):
            dt_cols.append(col)
            continue
        # Heuristic check for string columns with date-like names
        col_lower = str(col).lower()
        if any(kw in col_lower for kw in ("date", "time", "timestamp", "year", "month", "day", "_at", "_on")):
            non_null = s.dropna()
            if not non_null.empty:
                sample = non_null.head(20)
                try:
                    parsed = pd.to_datetime(sample, errors="coerce")
                    if parsed.notna().sum() >= len(sample) * 0.75:
                        dt_cols.append(col)
                except Exception:
                    pass
    return dt_cols


def safe_skewness(series: pd.Series) -> Optional[float]:
    """Compute Fisher-Pearson skewness safely without division by zero or NaN crashes."""
    s = series.dropna().astype(float)
    if len(s) < 3 or s.nunique() <= 1:
        return 0.0
    std_val = float(s.std(ddof=1))
    if std_val == 0.0 or np.isnan(std_val):
        return 0.0
    val = float(stats.skew(s, bias=False))
    return round(val, 4) if not np.isnan(val) else 0.0


def safe_kurtosis(series: pd.Series) -> Optional[float]:
    """Compute excess kurtosis safely (Fisher's definition: normal distribution = 0.0)."""
    s = series.dropna().astype(float)
    if len(s) < 4 or s.nunique() <= 1:
        return 0.0
    std_val = float(s.std(ddof=1))
    if std_val == 0.0 or np.isnan(std_val):
        return 0.0
    val = float(stats.kurtosis(s, bias=False, fisher=True))
    return round(val, 4) if not np.isnan(val) else 0.0


def format_stat_number(val: Any, decimals: int = 2) -> str:
    """Format floating or numeric values cleanly for enterprise dashboard display."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "N/A"
    if isinstance(val, (int, np.integer)):
        return f"{val:,}"
    if isinstance(val, (float, np.floating)):
        if abs(val) >= 1_000_000:
            return f"{val:,.2f}"
        if abs(val) < 0.001 and val != 0:
            return f"{val:.4e}"
        return f"{val:.{decimals}f}"
    return str(val)

