"""Univariate analysis engine for numerical variables."""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from utils.eda_helpers import get_eda_numerical_columns, safe_skewness, safe_kurtosis


def compute_numerical_univariate_stats(
    df: pd.DataFrame,
    column: str,
) -> Dict[str, Any]:
    """Compute comprehensive descriptive statistics for a single numerical column.
    
    Calculates: count, missing values, mean, median, mode, standard deviation,
    variance, min, max, Q1, Q2, Q3, IQR, skewness, and kurtosis.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' does not exist in dataset.")

    total_records = len(df)
    series = df[column]
    non_null_series = series.dropna()
    non_null_count = int(len(non_null_series))
    missing_count = int(series.isna().sum())
    missing_pct = round((missing_count / total_records * 100.0), 2) if total_records > 0 else 0.0

    if non_null_count == 0:
        return {
            "column": column,
            "total_records": total_records,
            "count": 0,
            "missing_count": missing_count,
            "missing_pct": missing_pct,
            "mean": None,
            "median": None,
            "mode": None,
            "std": None,
            "variance": None,
            "min": None,
            "max": None,
            "q1": None,
            "q2": None,
            "q3": None,
            "iqr": None,
            "skewness": None,
            "kurtosis": None,
            "is_constant": True,
            "zeros_count": 0,
            "zeros_pct": 0.0,
        }

    s_num = non_null_series.astype(float)
    min_val = float(s_num.min())
    max_val = float(s_num.max())
    mean_val = float(s_num.mean())
    median_val = float(s_num.median())
    std_val = float(s_num.std(ddof=1)) if non_null_count > 1 else 0.0
    var_val = float(s_num.var(ddof=1)) if non_null_count > 1 else 0.0

    # Quartiles and IQR
    q1_val = float(s_num.quantile(0.25))
    q2_val = median_val
    q3_val = float(s_num.quantile(0.75))
    iqr_val = float(q3_val - q1_val)

    # Mode
    mode_series = non_null_series.mode()
    mode_val = float(mode_series.iloc[0]) if not mode_series.empty else median_val

    # Skewness and Kurtosis
    skew_val = safe_skewness(s_num)
    kurt_val = safe_kurtosis(s_num)

    # Zero count
    zeros_count = int((s_num == 0.0).sum())
    zeros_pct = round((zeros_count / non_null_count * 100.0), 2) if non_null_count > 0 else 0.0

    is_constant = bool(non_null_series.nunique() <= 1)

    return {
        "column": column,
        "total_records": total_records,
        "count": non_null_count,
        "missing_count": missing_count,
        "missing_pct": missing_pct,
        "mean": round(mean_val, 4),
        "median": round(median_val, 4),
        "mode": round(mode_val, 4),
        "std": round(std_val, 4),
        "variance": round(var_val, 4),
        "min": round(min_val, 4),
        "max": round(max_val, 4),
        "q1": round(q1_val, 4),
        "q2": round(q2_val, 4),
        "q3": round(q3_val, 4),
        "iqr": round(iqr_val, 4),
        "skewness": skew_val,
        "kurtosis": kurt_val,
        "is_constant": is_constant,
        "zeros_count": zeros_count,
        "zeros_pct": zeros_pct,
    }


def compute_all_numerical_summaries(df: pd.DataFrame) -> pd.DataFrame:
    """Generate a high-level summary DataFrame for all numerical columns in the dataset."""
    num_cols = get_eda_numerical_columns(df)
    records = []
    for col in num_cols:
        st = compute_numerical_univariate_stats(df, col)
        records.append({
            "Column": col,
            "Count": st["count"],
            "Missing %": f"{st['missing_pct']}%",
            "Mean": st["mean"],
            "Std Dev": st["std"],
            "Min": st["min"],
            "Q1 (25%)": st["q1"],
            "Median": st["median"],
            "Q3 (75%)": st["q3"],
            "Max": st["max"],
            "IQR": st["iqr"],
            "Skewness": st["skewness"],
            "Kurtosis": st["kurtosis"],
        })
    return pd.DataFrame(records)

