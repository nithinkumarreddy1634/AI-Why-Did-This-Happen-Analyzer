"""Validation utilities for user inputs and data files."""

from typing import Tuple, List, Optional
from pathlib import Path
import pandas as pd


def validate_file_extension(filename: str, allowed_extensions: List[str]) -> Tuple[bool, str]:
    """Validate that the uploaded file has a supported extension."""
    if not filename:
        return False, "Filename cannot be empty."
        
    suffix = Path(filename).suffix.lower()
    if not suffix:
        return False, "File has no extension. Please upload a .csv, .xlsx, or .xls file."
        
    normalized_allowed = [ext.lower() for ext in allowed_extensions]
    if suffix not in normalized_allowed:
        return (
            False,
            f"Unsupported file format '{suffix}'. Allowed formats: {', '.join(allowed_extensions)}",
        )
    return True, "Valid file format."


def validate_file_size(size_bytes: int, max_mb: int) -> Tuple[bool, str]:
    """Validate that the file size is within acceptable threshold."""
    if size_bytes <= 0:
        return False, "File is empty (0 bytes)."
        
    max_bytes = max_mb * 1024 * 1024
    if size_bytes > max_bytes:
        return (
            False,
            f"File exceeds maximum allowed size of {max_mb} MB (current: {size_bytes / (1024 * 1024):.1f} MB).",
        )
    return True, "Valid file size."


def validate_dataset(df: Optional[pd.DataFrame]) -> Tuple[bool, str]:
    """Validate that the loaded pandas DataFrame is usable for analysis."""
    if df is None:
        return False, "Dataset is empty or could not be parsed."
    if not isinstance(df, pd.DataFrame):
        return False, "Provided object is not a valid pandas DataFrame."
    if df.empty or df.shape[0] == 0:
        return False, "Dataset contains 0 rows (empty dataset)."
    if df.shape[1] == 0:
        return False, "Dataset contains 0 columns."
    if df.shape[1] < 2:
        return (
            False,
            f"Dataset must have at least 2 columns to perform comparative investigation (found {df.shape[1]} column).",
        )
    
    # Check for duplicate column names
    if len(df.columns) != len(set(df.columns)):
        dup_cols = [c for c in df.columns if list(df.columns).count(c) > 1]
        return False, f"Dataset contains duplicate column names: {set(dup_cols)}."

    return True, f"Dataset valid ({df.shape[0]:,} records, {df.shape[1]} columns)."


def validate_target_variable(df: pd.DataFrame, target_col: str) -> Tuple[bool, str]:
    """Validate that a selected target variable is suitable for investigation."""
    if df is None or df.empty:
        return False, "Dataset is empty."
    if target_col not in df.columns:
        return False, f"Target column '{target_col}' not found in dataset."
    
    series = df[target_col].dropna()
    if series.empty:
        return False, f"Target column '{target_col}' contains only missing values."
    
    unique_count = series.nunique()
    if unique_count < 2:
        return (
            False,
            f"Target column '{target_col}' has only {unique_count} distinct value ('{series.iloc[0]}'). "
            "Investigation requires at least 2 distinct values to analyze differences.",
        )
        
    return True, f"Valid target variable with {unique_count:,} unique values."
