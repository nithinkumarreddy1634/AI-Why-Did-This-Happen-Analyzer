"""Numerical outlier detection and handling strategies."""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd


def get_numerical_columns(df: pd.DataFrame) -> List[str]:
    """Identify strictly numerical columns suitable for outlier analysis."""
    num_cols = []
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]) and not pd.api.types.is_bool_dtype(df[col]):
            # Check that it has more than 1 distinct value (not a constant column)
            if df[col].dropna().nunique() > 1:
                num_cols.append(col)
    return num_cols


def detect_column_outliers(
    df: pd.DataFrame,
    column: str,
    method: str = "iqr",
    threshold: float = 1.5,
) -> Dict[str, Any]:
    """Detect outliers for a single numerical column.
    
    Args:
        df: Input DataFrame.
        column: Numerical column name.
        method: 'iqr' (Interquartile Range) or 'zscore' (Standard Deviations).
        threshold: Multiplier for IQR (default 1.5) or standard deviation z-cutoff (default 3.0).

    Returns:
        Dict with lower_bound, upper_bound, outlier_count, outlier_pct,
        indices, min, max, median, q1, q3, mean, std.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' does not exist.")

    series = df[column].dropna()
    total_non_null = len(series)
    if total_non_null < 4:
        return {
            "column": column,
            "method": method,
            "threshold": threshold,
            "lower_bound": None,
            "upper_bound": None,
            "outlier_count": 0,
            "outlier_pct": 0.0,
            "lower_outliers_count": 0,
            "upper_outliers_count": 0,
            "min": float(series.min()) if total_non_null > 0 else 0.0,
            "max": float(series.max()) if total_non_null > 0 else 0.0,
            "q1": None,
            "q3": None,
            "mean": float(series.mean()) if total_non_null > 0 else 0.0,
            "std": float(series.std()) if total_non_null > 1 else 0.0,
            "outlier_indices": [],
        }

    q1 = float(series.quantile(0.25))
    q3 = float(series.quantile(0.75))
    iqr_val = q3 - q1
    mean_val = float(series.mean())
    std_val = float(series.std(ddof=1)) if len(series) > 1 else 0.0

    if method == "iqr":
        lower_bound = q1 - (threshold * iqr_val)
        upper_bound = q3 + (threshold * iqr_val)
    elif method == "zscore":
        if std_val == 0.0:
            lower_bound = mean_val
            upper_bound = mean_val
        else:
            lower_bound = mean_val - (threshold * std_val)
            upper_bound = mean_val + (threshold * std_val)
    else:
        raise ValueError(f"Unsupported outlier method: '{method}' (choose 'iqr' or 'zscore').")

    lower_mask = series < lower_bound
    upper_mask = series > upper_bound
    outlier_mask = lower_mask | upper_mask
    
    outlier_count = int(outlier_mask.sum())
    lower_count = int(lower_mask.sum())
    upper_count = int(upper_mask.sum())
    outlier_pct = (outlier_count / total_non_null * 100.0) if total_non_null > 0 else 0.0
    outlier_indices = series[outlier_mask].index.tolist()

    return {
        "column": column,
        "method": method,
        "threshold": threshold,
        "lower_bound": round(lower_bound, 4),
        "upper_bound": round(upper_bound, 4),
        "outlier_count": outlier_count,
        "outlier_pct": round(outlier_pct, 2),
        "lower_outliers_count": lower_count,
        "upper_outliers_count": upper_count,
        "min": round(float(series.min()), 4),
        "max": round(float(series.max()), 4),
        "q1": round(q1, 4),
        "q3": round(q3, 4),
        "mean": round(mean_val, 4),
        "std": round(std_val, 4),
        "outlier_indices": outlier_indices,
    }


def detect_all_outliers(
    df: pd.DataFrame,
    method: str = "iqr",
    threshold: float = 1.5,
) -> pd.DataFrame:
    """Detect outliers across all numerical columns and return summary DataFrame."""
    num_cols = get_numerical_columns(df)
    results = []

    for col in num_cols:
        info = detect_column_outliers(df, col, method=method, threshold=threshold)
        results.append(
            {
                "Column": col,
                "Outlier Count": info["outlier_count"],
                "Outlier %": info["outlier_pct"],
                "Lower Bound": info["lower_bound"],
                "Upper Bound": info["upper_bound"],
                "Min Value": info["min"],
                "Max Value": info["max"],
                "Method": method.upper(),
            }
        )

    summary_df = pd.DataFrame(results)
    if not summary_df.empty:
        summary_df = summary_df.sort_values(by="Outlier Count", ascending=False).reset_index(drop=True)
    return summary_df


def handle_column_outliers(
    df: pd.DataFrame,
    column: str,
    strategy: str,
    method: str = "iqr",
    threshold: float = 1.5,
    percentile_limits: Tuple[float, float] = (1.0, 99.0),
) -> Tuple[pd.DataFrame, int, Dict[str, Any], str]:
    """Apply an outlier treatment strategy to a specific column.
    
    Strategies:
    - 'keep': Retain original values untouched (safe exploration).
    - 'remove': Drop records where the column value is an outlier.
    - 'iqr_cap': Cap values at lower and upper IQR boundaries.
    - 'percentile_cap': Cap values between specified percentiles (e.g. 1st and 99th).
    - 'winsorize': Winsorize extreme tails to 5th and 95th percentiles (or custom limits).

    Returns:
    (new_df, affected_count, detection_info, message)
    """
    detection_info = detect_column_outliers(df, column, method=method, threshold=threshold)
    outlier_count = detection_info["outlier_count"]

    if strategy == "keep" or outlier_count == 0:
        return (
            df.copy(),
            0,
            detection_info,
            f"Retained all values in '{column}' without modification ({outlier_count} detected outliers kept).",
        )

    new_df = df.copy()
    lower_bound = detection_info["lower_bound"]
    upper_bound = detection_info["upper_bound"]

    if strategy == "remove":
        # Drop rows where column values exceed bounds
        mask_outliers = (new_df[column] < lower_bound) | (new_df[column] > upper_bound)
        removed_count = int(mask_outliers.sum())
        new_df = new_df[~mask_outliers].reset_index(drop=True)
        msg = (
            f"Removed {removed_count:,} outlier records from '{column}' beyond "
            f"[{lower_bound:.2f}, {upper_bound:.2f}]. New total rows: {len(new_df):,}."
        )
        return new_df, removed_count, detection_info, msg

    elif strategy == "iqr_cap":
        # Cap to IQR bounds
        capped_count = int(((new_df[column] < lower_bound) | (new_df[column] > upper_bound)).sum())
        new_df[column] = new_df[column].clip(lower=lower_bound, upper=upper_bound)
        msg = (
            f"Capped {capped_count:,} values in '{column}' to IQR bounds "
            f"[{lower_bound:.2f}, {upper_bound:.2f}]."
        )
        return new_df, capped_count, detection_info, msg

    elif strategy in ("percentile_cap", "winsorize"):
        low_pct, high_pct = percentile_limits
        low_val = float(df[column].quantile(low_pct / 100.0))
        high_val = float(df[column].quantile(high_pct / 100.0))
        capped_count = int(((new_df[column] < low_val) | (new_df[column] > high_val)).sum())
        new_df[column] = new_df[column].clip(lower=low_val, upper=high_val)
        label = "Winsorized" if strategy == "winsorize" else "Percentile-capped"
        msg = (
            f"{label} {capped_count:,} values in '{column}' between "
            f"{low_pct}th ({low_val:.2f}) and {high_pct}th ({high_val:.2f}) percentiles."
        )
        return new_df, capped_count, detection_info, msg

    else:
        raise ValueError(f"Unsupported outlier handling strategy: '{strategy}'")
