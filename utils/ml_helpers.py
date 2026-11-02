"""Machine learning utility and helper functions."""

from typing import List, Any, Optional, Dict
import pandas as pd
import numpy as np


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safe division guarding against ZeroDivisionError and NaN."""
    if denominator == 0 or np.isnan(denominator) or np.isinf(denominator):
        return default
    val = numerator / denominator
    return default if (np.isnan(val) or np.isinf(val)) else float(val)


def safe_mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-7) -> Optional[float]:
    """Calculate Mean Absolute Percentage Error (MAPE) safely.
    
    Returns None if a substantial proportion of actual values are zero or near-zero.
    """
    y_t = np.asarray(y_true, dtype=float)
    y_p = np.asarray(y_pred, dtype=float)
    
    mask = ~np.isnan(y_t) & ~np.isnan(y_p)
    y_t = y_t[mask]
    y_p = y_p[mask]
    
    if len(y_t) == 0:
        return None
        
    near_zeros = np.abs(y_t) < eps
    if np.mean(near_zeros) > 0.1:  # More than 10% near-zero target values
        return None
        
    # Calculate only on non-zero points or with epsilon
    denom = np.where(near_zeros, eps, np.abs(y_t))
    mape = np.mean(np.abs((y_t - y_p) / denom)) * 100.0
    return float(mape)


def detect_identifier_columns(df: pd.DataFrame, max_id_ratio: float = 0.95) -> List[str]:
    """Identify columns that appear to be IDs, keys, or sequence tokens.
    
    Heuristics:
    1. Column name contains 'id', 'uuid', 'guid', 'key', 'index', 'token', 'pk', 'fk'.
    2. High cardinality: Unique count / total non-null records > max_id_ratio for string or integer types.
    """
    id_cols = []
    n_rows = len(df)
    if n_rows == 0:
        return []

    id_name_keywords = {"id", "uuid", "guid", "key", "index", "token", "pk", "fk", "hash", "code"}

    for col in df.columns:
        col_str = str(col).lower()
        col_parts = col_str.replace("_", " ").replace("-", " ").split()
        
        has_id_name = any(kw in col_parts or col_str == kw or col_str.endswith("_id") or col_str.startswith("id_") for kw in id_name_keywords)
        
        non_null_count = df[col].count()
        if non_null_count == 0:
            continue
            
        unique_count = df[col].nunique()
        ratio = unique_count / non_null_count
        
        # If name matches ID and cardinality is moderate-to-high, or ratio is near 1.0 for non-floats
        if has_id_name and (ratio > 0.5 or non_null_count < 20):
            id_cols.append(col)
        elif ratio >= max_id_ratio and not pd.api.types.is_float_dtype(df[col]) and non_null_count > 10:
            id_cols.append(col)

    return id_cols


def detect_constant_columns(df: pd.DataFrame) -> List[str]:
    """Detect columns with zero variance (single unique value)."""
    return [col for col in df.columns if df[col].nunique(dropna=False) <= 1]


def format_metric(value: Optional[float], metric_name: str = "") -> str:
    """Format evaluation metrics consistently."""
    if value is None or np.isnan(value) or np.isinf(value):
        return "N/A"
    if "percentage" in metric_name.lower() or "mape" in metric_name.lower():
        return f"{value:.2f}%"
    if abs(value) < 0.001 and value != 0:
        return f"{value:.4e}"
    return f"{value:.4f}"

