"""Helper functions for unexpected outcome detection and intelligent change identification."""

from typing import Tuple, Optional, Any, Dict, List
import pandas as pd
import numpy as np


def safe_pct_change(baseline_val: float, current_val: float) -> Optional[float]:
    """Calculate percentage change safely without zero division errors.
    
    Returns:
        Percentage change as a float (e.g. 50.0 for a 50% increase),
        or None if baseline is zero / invalid.
    """
    if pd.isna(baseline_val) or pd.isna(current_val):
        return None
    try:
        b = float(baseline_val)
        c = float(current_val)
    except (ValueError, TypeError):
        return None

    if b == 0:
        return None  # Cannot compute percentage change when baseline is 0
    return ((c - b) / abs(b)) * 100.0


def safe_difference(baseline_val: float, current_val: float) -> Optional[float]:
    """Calculate absolute numerical difference safely."""
    if pd.isna(baseline_val) or pd.isna(current_val):
        return None
    try:
        return float(current_val) - float(baseline_val)
    except (ValueError, TypeError):
        return None


def detect_column_types(df: pd.DataFrame) -> Dict[str, str]:
    """Identify column data types suitable for anomaly and change detection.
    
    Returns a dictionary mapping column name to one of:
    'numerical', 'categorical', 'datetime', 'boolean', 'constant', 'identifier'
    """
    types_map: Dict[str, str] = {}
    n_rows = len(df)
    
    for col in df.columns:
        series = df[col].dropna()
        if series.empty:
            types_map[col] = "constant"
            continue
            
        n_unique = series.nunique()
        if n_unique <= 1:
            types_map[col] = "constant"
            continue
            
        # Check datetime
        if pd.api.types.is_datetime64_any_dtype(series):
            types_map[col] = "datetime"
            continue
            
        # Try parsing strings as datetime if column looks like a date/time
        if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
            sample = series.head(20).astype(str)
            # Heuristic check: check if it parses consistently
            try:
                converted = pd.to_datetime(sample, errors="coerce", format="mixed")
                if converted.notna().sum() / len(sample) >= 0.8:
                    types_map[col] = "datetime"
                    continue
            except Exception:
                pass
                
        # Check boolean
        if pd.api.types.is_bool_dtype(series) or (n_unique == 2 and set(series.unique()).issubset({0, 1, True, False, "0", "1", "true", "false", "True", "False"})):
            types_map[col] = "boolean"
            continue
            
        # Check numerical
        if pd.api.types.is_numeric_dtype(series):
            # Check if identifier / ID
            if n_unique == n_rows and n_rows > 50 and col.lower() in ("id", "index", "uuid", "customer_id", "user_id"):
                types_map[col] = "identifier"
            else:
                types_map[col] = "numerical"
            continue
            
        # Check categorical / identifier / text
        if n_unique == n_rows and n_rows > 50:
            types_map[col] = "identifier"
        else:
            types_map[col] = "categorical"
            
    return types_map


def split_baseline_comparison(
    df: pd.DataFrame,
    split_type: str,
    split_column: str,
    baseline_value: Any,
    comparison_value: Any = None,
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], str]:
    """Split a dataset into baseline and comparison cohorts.
    
    Args:
        df: Input DataFrame.
        split_type: 'category' (two values in a category) or 'datetime_cutoff' or 'numerical_cutoff'.
        split_column: Column used for partitioning.
        baseline_value: Value or threshold for baseline.
        comparison_value: Value or threshold for comparison group.
        
    Returns:
        (baseline_df, comparison_df, description_message)
    """
    if df is None or df.empty:
        return None, None, "Dataset is empty."
        
    if split_column not in df.columns:
        return None, None, f"Partition column '{split_column}' not found in dataset."

    if split_type == "category":
        b_df = df[df[split_column] == baseline_value]
        c_df = df[df[split_column] == comparison_value]
        msg = f"Categorical split on '{split_column}': Baseline='{baseline_value}' ({len(b_df):,} rows), Comparison='{comparison_value}' ({len(c_df):,} rows)"
        return b_df, c_df, msg
        
    elif split_type == "datetime_cutoff":
        try:
            date_series = pd.to_datetime(df[split_column], errors="coerce")
            cutoff = pd.to_datetime(baseline_value)
            b_df = df[date_series < cutoff]
            c_df = df[date_series >= cutoff]
            msg = f"Time cutoff on '{split_column}': Baseline (< {cutoff.strftime('%Y-%m-%d')}: {len(b_df):,} rows), Comparison (>= {cutoff.strftime('%Y-%m-%d')}: {len(c_df):,} rows)"
            return b_df, c_df, msg
        except Exception as e:
            return None, None, f"Error parsing datetime cutoff: {str(e)}"
            
    elif split_type == "numerical_cutoff":
        try:
            threshold = float(baseline_value)
            b_df = df[df[split_column] <= threshold]
            c_df = df[df[split_column] > threshold]
            msg = f"Numerical split on '{split_column}': Baseline (<= {threshold}: {len(b_df):,} rows), Comparison (> {threshold}: {len(c_df):,} rows)"
            return b_df, c_df, msg
        except Exception as e:
            return None, None, f"Error parsing numerical threshold: {str(e)}"
            
    return None, None, f"Unsupported split type '{split_type}'."
