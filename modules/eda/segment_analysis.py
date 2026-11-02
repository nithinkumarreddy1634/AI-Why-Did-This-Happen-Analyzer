"""Group and cohort segment analysis engine."""

from typing import Dict, Any, List, Optional
import pandas as pd
from utils.eda_helpers import get_eda_numerical_columns, get_eda_categorical_columns


def compute_segment_breakdown(
    df: pd.DataFrame,
    segment_col: str,
    metric_cols: Optional[List[str]] = None,
    sort_by_metric: Optional[str] = None,
    ascending: bool = False,
) -> Dict[str, Any]:
    """Group dataset by a categorical feature and summarize numerical indicators across segments."""
    if segment_col not in df.columns:
        raise ValueError(f"Segment column '{segment_col}' does not exist in dataset.")

    total_dataset_rows = len(df)
    valid_mask = df[segment_col].notna()
    subset = df[valid_mask].copy()

    all_num_cols = get_eda_numerical_columns(df)
    active_metrics = [c for c in metric_cols if c in all_num_cols] if metric_cols else all_num_cols[:min(3, len(all_num_cols))]

    if not active_metrics:
        # Categorical count only
        counts = subset[segment_col].astype(str).value_counts().reset_index()
        counts.columns = [segment_col, "Record Count"]
        counts["Share %"] = (counts["Record Count"] / total_dataset_rows * 100.0).round(2)
        return {
            "valid": True,
            "segment_col": segment_col,
            "metric_cols": [],
            "segment_summary_df": counts,
            "total_segments": len(counts),
        }

    # Build group aggregations
    agg_dict = {m: ["count", "mean", "median", "std", "sum"] for m in active_metrics}
    grouped = subset.groupby(segment_col).agg(agg_dict)

    # Flatten hierarchical columns
    flat_cols = []
    for col_name, stat in grouped.columns:
        stat_name = stat.capitalize()
        flat_cols.append(f"{col_name} ({stat_name})")
    grouped.columns = flat_cols
    grouped = grouped.reset_index()

    # Add Segment Share %
    first_count_col = f"{active_metrics[0]} (Count)"
    if first_count_col in grouped.columns:
        grouped["Segment Share %"] = (grouped[first_count_col] / total_dataset_rows * 100.0).round(2)

    # Round numeric values
    for c in grouped.columns:
        if c != segment_col:
            grouped[c] = grouped[c].round(2)

    # Sort
    if sort_by_metric and sort_by_metric in grouped.columns:
        grouped = grouped.sort_values(by=sort_by_metric, ascending=ascending).reset_index(drop=True)
    elif first_count_col in grouped.columns:
        grouped = grouped.sort_values(by=first_count_col, ascending=False).reset_index(drop=True)

    return {
        "valid": True,
        "segment_col": segment_col,
        "metric_cols": active_metrics,
        "segment_summary_df": grouped,
        "total_segments": len(grouped),
    }

