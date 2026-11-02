"""Univariate analysis engine for categorical and discrete variables."""

from typing import Dict, Any, List, Optional
import pandas as pd
from utils.eda_helpers import get_eda_categorical_columns


def compute_categorical_univariate_stats(
    df: pd.DataFrame,
    column: str,
    top_n: int = 10,
    group_other: bool = False,
) -> Dict[str, Any]:
    """Calculate univariate statistics and frequency distributions for a categorical column.
    
    Args:
        df: Input DataFrame.
        column: Categorical column name.
        top_n: Number of leading categories to display.
        group_other: Whether to group categories outside top_n into 'Other'.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' does not exist in dataset.")

    total_records = len(df)
    series = df[column]
    non_null_series = series.dropna().astype(str)
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
            "unique_categories": 0,
            "most_frequent_category": None,
            "most_frequent_count": 0,
            "most_frequent_pct": 0.0,
            "least_frequent_category": None,
            "least_frequent_count": 0,
            "least_frequent_pct": 0.0,
            "frequencies_df": pd.DataFrame(columns=["Category", "Count", "Percentage"]),
            "is_high_cardinality": False,
        }

    value_counts = non_null_series.value_counts()
    unique_count = len(value_counts)

    most_freq_cat = str(value_counts.index[0])
    most_freq_cnt = int(value_counts.iloc[0])
    most_freq_pct = round((most_freq_cnt / non_null_count * 100.0), 2)

    least_freq_cat = str(value_counts.index[-1])
    least_freq_cnt = int(value_counts.iloc[-1])
    least_freq_pct = round((least_freq_cnt / non_null_count * 100.0), 2)

    # Frequency Table preparation with optional Top N and Other grouping
    if group_other and unique_count > top_n:
        top_slice = value_counts.head(top_n)
        other_cnt = int(value_counts.iloc[top_n:].sum())
        
        cats = list(top_slice.index) + ["Other"]
        counts = list(top_slice.values) + [other_cnt]
    else:
        top_slice = value_counts.head(top_n)
        cats = list(top_slice.index)
        counts = list(top_slice.values)

    freq_records = []
    for cat, cnt in zip(cats, counts):
        pct = round((cnt / non_null_count * 100.0), 2)
        freq_records.append({
            "Category": str(cat),
            "Count": int(cnt),
            "Percentage": pct,
        })
    frequencies_df = pd.DataFrame(freq_records)

    is_high_cardinality = bool(unique_count > 25 or (unique_count / max(1, total_records) > 0.4 and total_records > 50))

    return {
        "column": column,
        "total_records": total_records,
        "count": non_null_count,
        "missing_count": missing_count,
        "missing_pct": missing_pct,
        "unique_categories": unique_count,
        "most_frequent_category": most_freq_cat,
        "most_frequent_count": most_freq_cnt,
        "most_frequent_pct": most_freq_pct,
        "least_frequent_category": least_freq_cat,
        "least_frequent_count": least_freq_cnt,
        "least_frequent_pct": least_freq_pct,
        "frequencies_df": frequencies_df,
        "is_high_cardinality": is_high_cardinality,
    }


def compute_all_categorical_summaries(df: pd.DataFrame) -> pd.DataFrame:
    """Generate high-level summary DataFrame for all categorical columns."""
    cat_cols = get_eda_categorical_columns(df)
    records = []
    for col in cat_cols:
        st = compute_categorical_univariate_stats(df, col, top_n=5)
        records.append({
            "Column": col,
            "Non-Null Count": st["count"],
            "Missing %": f"{st['missing_pct']}%",
            "Unique Categories": st["unique_categories"],
            "Top Category": f"{st['most_frequent_category']} ({st['most_frequent_pct']}%)" if st["most_frequent_category"] else "N/A",
            "Least Frequent": f"{st['least_frequent_category']} ({st['least_frequent_pct']}%)" if st["least_frequent_category"] else "N/A",
            "High Cardinality": "Yes" if st["is_high_cardinality"] else "No",
        })
    return pd.DataFrame(records)

