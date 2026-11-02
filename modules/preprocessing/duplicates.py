"""Duplicate record detection and resolution."""

from typing import Dict, Any, List, Optional, Tuple
import pandas as pd


def detect_duplicates(
    df: pd.DataFrame,
    subset: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Identify duplicate rows in the dataset.
    
    Args:
        df: Input DataFrame.
        subset: Optional list of columns to consider for identifying duplicates.
                If None, all columns are considered.

    Returns:
        Dict containing total_duplicates, duplicate_percentage, sample duplicate rows.
    """
    total_rows = len(df)
    if total_rows == 0:
        return {
            "duplicate_count": 0,
            "duplicate_percentage": 0.0,
            "unique_rows": 0,
            "total_rows": 0,
            "duplicate_samples": pd.DataFrame(),
            "has_duplicates": False,
        }

    # Boolean mask of rows that are duplicated (keep=False marks all duplicates)
    dup_mask_all = df.duplicated(subset=subset, keep=False)
    dup_mask_excess = df.duplicated(subset=subset, keep="first")
    
    dup_count = int(dup_mask_excess.sum())
    dup_pct = (dup_count / total_rows) * 100.0

    # Get sample duplicate rows grouped together for inspection
    sample_dups = df[dup_mask_all].copy()
    if not sample_dups.empty and subset:
        sample_dups = sample_dups.sort_values(by=subset)
    elif not sample_dups.empty:
        sample_dups = sample_dups.sort_values(by=list(df.columns[:min(3, len(df.columns))]))

    return {
        "duplicate_count": dup_count,
        "duplicate_percentage": round(dup_pct, 2),
        "unique_rows": total_rows - dup_count,
        "total_rows": total_rows,
        "duplicate_samples": sample_dups.head(50),
        "has_duplicates": dup_count > 0,
    }


def remove_duplicates(
    df: pd.DataFrame,
    subset: Optional[List[str]] = None,
    keep: str = "first",
) -> Tuple[pd.DataFrame, int, str]:
    """Remove duplicate rows from dataset, retaining first or last occurrence.
    
    Args:
        df: Input DataFrame.
        subset: Optional subset of columns.
        keep: 'first' to retain first occurrence, 'last' to retain last occurrence.

    Returns:
        (new_df, removed_count, message)
    """
    if keep not in ("first", "last"):
        raise ValueError("keep argument must be either 'first' or 'last'.")

    new_df = df.copy()
    orig_count = len(new_df)
    new_df = new_df.drop_duplicates(subset=subset, keep=keep).reset_index(drop=True)
    removed_count = orig_count - len(new_df)

    cols_desc = f"based on columns {subset}" if subset else "across all columns"
    msg = (
        f"Removed {removed_count:,} duplicate record(s) {cols_desc}, "
        f"retaining the '{keep}' occurrence. New row count: {len(new_df):,}."
    )
    return new_df, removed_count, msg
