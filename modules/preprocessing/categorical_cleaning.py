"""Categorical data standardization, label replacement, and rare category consolidation."""

from typing import Dict, Any, List, Optional, Tuple, Set
from collections import defaultdict
import pandas as pd


def get_categorical_columns(df: pd.DataFrame) -> List[str]:
    """Identify columns that are categorical, string, or low-cardinality object types."""
    cat_cols = []
    for col in df.columns:
        if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col]) or isinstance(df[col].dtype, pd.CategoricalDtype):
            cat_cols.append(col)
    return cat_cols


def detect_categorical_inconsistencies(
    df: pd.DataFrame,
    column: str,
    rare_threshold_pct: float = 3.0,
) -> Dict[str, Any]:
    """Analyze a categorical column for whitespace, casing discrepancies, and rare labels.
    
    Returns:
    Dict containing:
    - total_records
    - unique_count
    - whitespace_issues_count
    - casing_clusters: Dict[str, List[str]] where normalized lowercase key maps to multiple casing variants
    - rare_categories: List[Dict[str, Any]] categories with percentage < rare_threshold_pct
    - value_counts_df: DataFrame of categories, counts, percentages
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' does not exist.")

    series = df[column].dropna().astype(str)
    total_non_null = len(series)
    if total_non_null == 0:
        return {
            "column": column,
            "total_records": 0,
            "unique_count": 0,
            "whitespace_issues_count": 0,
            "casing_clusters": {},
            "rare_categories": [],
            "value_counts_df": pd.DataFrame(),
        }

    # 1. Whitespace padding detection
    whitespace_mask = series.str.strip() != series
    whitespace_count = int(whitespace_mask.sum())

    # 2. Case inconsistency clustering
    # Group original values by their stripped lowercase representation
    clusters = defaultdict(set)
    for val in series:
        norm = val.strip().lower()
        clusters[norm].add(val.strip())

    inconsistent_casing = {
        norm: sorted(list(variants))
        for norm, variants in clusters.items()
        if len(variants) > 1
    }

    # 3. Value counts and rare category identification
    counts = series.str.strip().value_counts()
    value_records = []
    rare_categories = []

    for val, count in counts.items():
        pct = (count / total_non_null) * 100.0
        rec = {
            "category": val,
            "count": int(count),
            "percentage": round(pct, 2),
            "is_rare": pct < rare_threshold_pct,
        }
        value_records.append(rec)
        if pct < rare_threshold_pct:
            rare_categories.append(rec)

    val_df = pd.DataFrame(value_records)

    return {
        "column": column,
        "total_records": total_non_null,
        "unique_count": len(counts),
        "whitespace_issues_count": whitespace_count,
        "casing_clusters": inconsistent_casing,
        "rare_categories": rare_categories,
        "value_counts_df": val_df,
    }


def strip_column_whitespace(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, int, str]:
    """Strip leading and trailing whitespace from specified or all string/categorical columns.
    
    Returns:
    (new_df, modified_cells_count, message)
    """
    new_df = df.copy()
    target_cols = columns if columns else get_categorical_columns(new_df)
    total_modified = 0

    for col in target_cols:
        if col in new_df.columns:
            # Check if column contains strings
            series = new_df[col]
            mask_non_null = series.notna()
            str_series = series.astype(str)
            stripped_series = str_series.str.strip()
            
            diff_mask = mask_non_null & (str_series != stripped_series)
            diff_count = int(diff_mask.sum())
            if diff_count > 0:
                # Assign stripped values preserving nulls
                new_df.loc[mask_non_null, col] = stripped_series[mask_non_null]
                total_modified += diff_count

    cols_desc = f"in {len(target_cols)} column(s)" if target_cols else "in all text columns"
    msg = f"Stripped leading and trailing whitespace from {total_modified:,} cell(s) {cols_desc}."
    return new_df, total_modified, msg


def standardize_capitalization(
    df: pd.DataFrame,
    column: str,
    case_type: str = "title",
) -> Tuple[pd.DataFrame, int, str]:
    """Standardize the casing of values in a categorical column.
    
    Args:
        df: Input DataFrame.
        column: Column to standardize.
        case_type: 'title', 'lower', 'upper', or 'capitalize'.

    Returns:
        (new_df, modified_count, message)
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' does not exist.")

    new_df = df.copy()
    mask = new_df[column].notna()
    orig_str = new_df.loc[mask, column].astype(str).str.strip()

    if case_type == "title":
        transformed = orig_str.str.title()
    elif case_type == "lower":
        transformed = orig_str.str.lower()
    elif case_type == "upper":
        transformed = orig_str.str.upper()
    elif case_type == "capitalize":
        transformed = orig_str.str.capitalize()
    else:
        raise ValueError(f"Unsupported case_type: '{case_type}'")

    diff_mask = orig_str != transformed
    modified_count = int(diff_mask.sum())

    new_df.loc[mask, column] = transformed
    msg = (
        f"Standardized casing to '{case_type}' in column '{column}'. "
        f"Modified {modified_count:,} value(s)."
    )
    return new_df, modified_count, msg


def replace_category_values(
    df: pd.DataFrame,
    column: str,
    mapping: Dict[str, str],
) -> Tuple[pd.DataFrame, int, str]:
    """Replace specific category labels in a column according to user mapping.
    
    Args:
        df: Input DataFrame.
        column: Column to modify.
        mapping: Dictionary mapping {old_value: new_value}.

    Returns:
        (new_df, modified_count, message)
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' does not exist.")

    if not mapping:
        return df.copy(), 0, "No mapping provided; column left unchanged."

    new_df = df.copy()
    mask = new_df[column].isin(list(mapping.keys()))
    modified_count = int(mask.sum())

    new_df[column] = new_df[column].replace(mapping)
    msg = (
        f"Replaced {len(mapping)} distinct category label(s) in '{column}', "
        f"affecting {modified_count:,} record(s)."
    )
    return new_df, modified_count, msg


def group_rare_categories(
    df: pd.DataFrame,
    column: str,
    threshold_pct: float = 3.0,
    other_label: str = "Other",
) -> Tuple[pd.DataFrame, int, List[str], str]:
    """Consolidate low-frequency categories into a single label ('Other').
    
    Args:
        df: Input DataFrame.
        column: Target column.
        threshold_pct: Percentage cutoff (categories below this % are grouped).
        other_label: Name for the consolidated label.

    Returns:
        (new_df, modified_count, grouped_categories_list, message)
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' does not exist.")

    new_df = df.copy()
    series = new_df[column].dropna().astype(str)
    total_non_null = len(series)
    if total_non_null == 0:
        return new_df, 0, [], f"Column '{column}' is empty."

    counts = series.value_counts()
    rare_cats = [
        val for val, count in counts.items()
        if (count / total_non_null) * 100.0 < threshold_pct
    ]

    if not rare_cats:
        return new_df, 0, [], f"No rare categories below {threshold_pct}% found in '{column}'."

    mask = new_df[column].astype(str).isin(rare_cats)
    modified_count = int(mask.sum())

    new_df.loc[mask, column] = other_label
    msg = (
        f"Grouped {len(rare_cats)} rare category/categories ({', '.join(rare_cats[:5])}"
        f"{'...' if len(rare_cats) > 5 else ''}) into '{other_label}' in column '{column}' "
        f"({modified_count:,} record(s) affected)."
    )
    return new_df, modified_count, rare_cats, msg
