"""Missing value detection and imputation strategies."""

from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


def detect_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Detect missing values across all columns and compute statistics.
    
    Returns a DataFrame with columns:
    ['column', 'missing_count', 'missing_percentage', 'non_null_count', 'dtype', 'is_all_missing']
    """
    total_rows = len(df)
    records = []

    for col in df.columns:
        null_count = int(df[col].isna().sum())
        null_pct = (null_count / total_rows * 100.0) if total_rows > 0 else 0.0
        records.append(
            {
                "column": col,
                "missing_count": null_count,
                "missing_percentage": round(null_pct, 2),
                "non_null_count": total_rows - null_count,
                "dtype": str(df[col].dtype),
                "is_all_missing": null_count == total_rows,
            }
        )

    summary_df = pd.DataFrame(records)
    if not summary_df.empty:
        summary_df = summary_df.sort_values(by="missing_count", ascending=False).reset_index(drop=True)
    return summary_df


def get_missing_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """Get high-level missing value summary metrics."""
    total_cells = df.size
    total_missing_cells = int(df.isna().sum().sum())
    total_rows = len(df)
    rows_with_missing = int(df.isna().any(axis=1).sum()) if total_rows > 0 else 0
    cols_with_missing = int((df.isna().sum() > 0).sum())
    
    missing_pct = (total_missing_cells / total_cells * 100.0) if total_cells > 0 else 0.0
    rows_missing_pct = (rows_with_missing / total_rows * 100.0) if total_rows > 0 else 0.0

    return {
        "total_cells": total_cells,
        "total_missing_cells": total_missing_cells,
        "missing_percentage": round(missing_pct, 2),
        "total_rows": total_rows,
        "rows_with_missing": rows_with_missing,
        "rows_with_missing_pct": round(rows_missing_pct, 2),
        "cols_with_missing": cols_with_missing,
        "total_cols": df.shape[1],
    }


def impute_numerical_column(
    df: pd.DataFrame,
    column: str,
    strategy: str = "median",
    constant_value: float = 0.0,
) -> Tuple[pd.DataFrame, int, str]:
    """Impute missing values in a numerical column.
    
    Supported strategies:
    - 'mean': Arithmetic mean of non-null values
    - 'median': 50th percentile of non-null values
    - 'constant': User-defined constant number
    - 'ffill': Forward fill along the column
    - 'bfill': Backward fill along the column

    Returns:
    (new_df, imputed_count, message)
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' does not exist in dataset.")

    new_df = df.copy()
    series = new_df[column]
    missing_mask = series.isna()
    missing_count = int(missing_mask.sum())

    if missing_count == 0:
        return new_df, 0, f"No missing values found in column '{column}'."

    # Handle 100% missing value columns
    if missing_count == len(df):
        if strategy in ("mean", "median"):
            # Cannot compute mean/median for 100% NaN column; fallback to 0.0 with warning
            new_df[column] = constant_value
            return (
                new_df,
                missing_count,
                f"Column '{column}' was 100% missing. Imputed with constant {constant_value} instead of {strategy}.",
            )

    if strategy == "mean":
        fill_val = float(series.mean())
        new_df[column] = series.fillna(fill_val)
        msg = f"Imputed {missing_count:,} missing values in '{column}' using mean ({fill_val:.3f})."
    elif strategy == "median":
        fill_val = float(series.median())
        new_df[column] = series.fillna(fill_val)
        msg = f"Imputed {missing_count:,} missing values in '{column}' using median ({fill_val:.3f})."
    elif strategy == "constant":
        new_df[column] = series.fillna(constant_value)
        msg = f"Imputed {missing_count:,} missing values in '{column}' using constant ({constant_value})."
    elif strategy == "ffill":
        new_df[column] = series.ffill().bfill()  # bfill handles initial NaNs
        msg = f"Imputed {missing_count:,} missing values in '{column}' using forward fill."
    elif strategy == "bfill":
        new_df[column] = series.bfill().ffill()  # ffill handles trailing NaNs
        msg = f"Imputed {missing_count:,} missing values in '{column}' using backward fill."
    else:
        raise ValueError(f"Unsupported numerical imputation strategy: '{strategy}'")

    return new_df, missing_count, msg


def impute_categorical_column(
    df: pd.DataFrame,
    column: str,
    strategy: str = "mode",
    custom_value: str = "Unknown",
) -> Tuple[pd.DataFrame, int, str]:
    """Impute missing values in a categorical column.
    
    Supported strategies:
    - 'mode': Most frequent category
    - 'unknown': Fixed string 'Unknown'
    - 'custom': User-specified custom string
    - 'ffill': Forward fill
    - 'bfill': Backward fill

    Returns:
    (new_df, imputed_count, message)
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' does not exist in dataset.")

    new_df = df.copy()
    series = new_df[column]
    missing_mask = series.isna()
    missing_count = int(missing_mask.sum())

    if missing_count == 0:
        return new_df, 0, f"No missing values found in column '{column}'."

    # Handle 100% missing values
    if missing_count == len(df):
        fill_val = custom_value if strategy in ("custom", "unknown") else "Unknown"
        new_df[column] = series.fillna(fill_val).astype(str)
        return (
            new_df,
            missing_count,
            f"Column '{column}' was 100% missing. Imputed all with '{fill_val}'.",
        )

    if strategy == "mode":
        mode_series = series.mode(dropna=True)
        fill_val = mode_series.iloc[0] if not mode_series.empty else "Unknown"
        new_df[column] = series.fillna(fill_val)
        msg = f"Imputed {missing_count:,} missing values in '{column}' using mode ('{fill_val}')."
    elif strategy == "unknown":
        new_df[column] = series.fillna("Unknown")
        msg = f"Imputed {missing_count:,} missing values in '{column}' with 'Unknown'."
    elif strategy == "custom":
        new_df[column] = series.fillna(custom_value)
        msg = f"Imputed {missing_count:,} missing values in '{column}' with '{custom_value}'."
    elif strategy == "ffill":
        new_df[column] = series.ffill().bfill()
        msg = f"Imputed {missing_count:,} missing values in '{column}' using forward fill."
    elif strategy == "bfill":
        new_df[column] = series.bfill().ffill()
        msg = f"Imputed {missing_count:,} missing values in '{column}' using backward fill."
    else:
        raise ValueError(f"Unsupported categorical imputation strategy: '{strategy}'")

    return new_df, missing_count, msg


def drop_missing_rows(
    df: pd.DataFrame,
    subset: Optional[List[str]] = None,
    how: str = "any",
) -> Tuple[pd.DataFrame, int, str]:
    """Drop rows containing missing values in specified columns or all columns.
    
    Returns:
    (new_df, dropped_rows_count, message)
    """
    new_df = df.copy()
    orig_rows = len(new_df)
    
    target_cols = subset if subset else list(new_df.columns)
    new_df = new_df.dropna(how=how, subset=target_cols).reset_index(drop=True)
    dropped_count = orig_rows - len(new_df)
    
    cols_desc = f"in columns {subset}" if subset else "across all columns"
    msg = f"Dropped {dropped_count:,} rows with missing values {cols_desc} (retained {len(new_df):,} rows)."
    return new_df, dropped_count, msg


def drop_missing_columns(
    df: pd.DataFrame,
    threshold_pct: float = 50.0,
) -> Tuple[pd.DataFrame, List[str], str]:
    """Drop columns where missing value percentage exceeds the specified threshold.
    
    Returns:
    (new_df, dropped_columns_list, message)
    """
    new_df = df.copy()
    total_rows = len(new_df)
    if total_rows == 0:
        return new_df, [], "Dataset is empty; no columns dropped."

    dropped_cols: List[str] = []
    for col in df.columns:
        null_pct = (df[col].isna().sum() / total_rows) * 100.0
        if null_pct >= threshold_pct:
            dropped_cols.append(col)

    if dropped_cols:
        new_df = new_df.drop(columns=dropped_cols)
        msg = (
            f"Dropped {len(dropped_cols)} column(s) exceeding {threshold_pct}% missing threshold: "
            f"{', '.join(dropped_cols)}."
        )
    else:
        msg = f"No columns exceeded the {threshold_pct}% missing threshold."

    return new_df, dropped_cols, msg
