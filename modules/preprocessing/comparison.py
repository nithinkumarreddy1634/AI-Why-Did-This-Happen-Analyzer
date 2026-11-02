"""Before-and-after comparison diagnostics and Plotly visualization builders."""

from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from modules.preprocessing.transformation_history import TransformationStep


def compute_comparison_metrics(
    df_original: pd.DataFrame,
    df_processed: pd.DataFrame,
    history_steps: Optional[List[TransformationStep]] = None,
) -> Dict[str, Any]:
    """Compute comprehensive differences between original and processed datasets."""
    orig_rows, orig_cols = df_original.shape
    proc_rows, proc_cols = df_processed.shape

    # Missing values
    orig_missing = int(df_original.isna().sum().sum())
    proc_missing = int(df_processed.isna().sum().sum())
    orig_missing_pct = (orig_missing / df_original.size * 100.0) if df_original.size > 0 else 0.0
    proc_missing_pct = (proc_missing / df_processed.size * 100.0) if df_processed.size > 0 else 0.0

    # Duplicates
    orig_dups = int(df_original.duplicated().sum())
    proc_dups = int(df_processed.duplicated().sum())

    # Data type modifications
    common_cols = [c for c in df_original.columns if c in df_processed.columns]
    dtype_changes = []
    for col in common_cols:
        orig_dt = str(df_original[col].dtype)
        proc_dt = str(df_processed[col].dtype)
        if orig_dt != proc_dt:
            dtype_changes.append({"column": col, "before": orig_dt, "after": proc_dt})

    # Added & removed columns
    added_columns = [c for c in df_processed.columns if c not in df_original.columns]
    removed_columns = [c for c in df_original.columns if c not in df_processed.columns]

    # Outlier counts comparison across common numeric columns
    outlier_changes = []
    for col in common_cols:
        if pd.api.types.is_numeric_dtype(df_original[col]) and pd.api.types.is_numeric_dtype(df_processed[col]):
            s_orig = df_original[col].dropna()
            s_proc = df_processed[col].dropna()
            if len(s_orig) > 3 and len(s_proc) > 3:
                # Original IQR
                q1_o, q3_o = float(s_orig.quantile(0.25)), float(s_orig.quantile(0.75))
                iqr_o = q3_o - q1_o
                outliers_orig = int(((s_orig < q1_o - 1.5 * iqr_o) | (s_orig > q3_o + 1.5 * iqr_o)).sum())

                # Processed IQR
                q1_p, q3_p = float(s_proc.quantile(0.25)), float(s_proc.quantile(0.75))
                iqr_p = q3_p - q1_p
                outliers_proc = int(((s_proc < q1_p - 1.5 * iqr_p) | (s_proc > q3_p + 1.5 * iqr_p)).sum())

                if outliers_orig > 0 or outliers_proc > 0:
                    outlier_changes.append(
                        {
                            "column": col,
                            "before": outliers_orig,
                            "after": outliers_proc,
                            "delta": outliers_proc - outliers_orig,
                        }
                    )

    # Missing per column comparison
    missing_by_col = []
    all_cols = sorted(list(set(df_original.columns).union(df_processed.columns)))
    for col in all_cols:
        m_before = int(df_original[col].isna().sum()) if col in df_original.columns else 0
        m_after = int(df_processed[col].isna().sum()) if col in df_processed.columns else 0
        if m_before > 0 or m_after > 0:
            missing_by_col.append(
                {
                    "column": col,
                    "before": m_before,
                    "after": m_after,
                    "resolved": m_before - m_after,
                }
            )

    return {
        "rows": {
            "before": orig_rows,
            "after": proc_rows,
            "delta": proc_rows - orig_rows,
        },
        "columns": {
            "before": orig_cols,
            "after": proc_cols,
            "delta": proc_cols - orig_cols,
        },
        "missing_cells": {
            "before": orig_missing,
            "after": proc_missing,
            "delta": proc_missing - orig_missing,
            "before_pct": round(orig_missing_pct, 2),
            "after_pct": round(proc_missing_pct, 2),
        },
        "duplicates": {
            "before": orig_dups,
            "after": proc_dups,
            "delta": proc_dups - orig_dups,
        },
        "dtype_changes": dtype_changes,
        "added_columns": added_columns,
        "removed_columns": removed_columns,
        "outlier_changes": outlier_changes,
        "missing_by_col": missing_by_col,
        "total_operations_applied": len(history_steps) if history_steps else 0,
    }


def build_missing_comparison_chart(comparison_data: Dict[str, Any]) -> Optional[go.Figure]:
    """Build a Plotly grouped bar chart comparing missing values per column before vs after."""
    missing_by_col = comparison_data.get("missing_by_col", [])
    if not missing_by_col:
        return None

    # Take top 15 columns with most missing values
    sorted_cols = sorted(missing_by_col, key=lambda x: x["before"], reverse=True)[:15]
    cols = [item["column"] for item in sorted_cols]
    before_vals = [item["before"] for item in sorted_cols]
    after_vals = [item["after"] for item in sorted_cols]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=cols,
            y=before_vals,
            name="Original Missing",
            marker_color="#EF4444",
        )
    )
    fig.add_trace(
        go.Bar(
            x=cols,
            y=after_vals,
            name="Cleaned Missing",
            marker_color="#10B981",
        )
    )

    fig.update_layout(
        title="Missing Values Resolution by Column (Before vs. After)",
        barmode="group",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis_title="Column",
        yaxis_title="Missing Cell Count",
        margin=dict(l=20, r=20, t=40, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def build_outlier_comparison_chart(comparison_data: Dict[str, Any]) -> Optional[go.Figure]:
    """Build a Plotly bar chart comparing outlier counts per column before vs after."""
    outlier_changes = comparison_data.get("outlier_changes", [])
    if not outlier_changes:
        return None

    cols = [item["column"] for item in outlier_changes[:12]]
    before_vals = [item["before"] for item in outlier_changes[:12]]
    after_vals = [item["after"] for item in outlier_changes[:12]]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=cols,
            y=before_vals,
            name="Original Outliers",
            marker_color="#F59E0B",
        )
    )
    fig.add_trace(
        go.Bar(
            x=cols,
            y=after_vals,
            name="Processed Outliers",
            marker_color="#38BDF8",
        )
    )

    fig.update_layout(
        title="Numerical Outlier Counts (Before vs. After Treatment)",
        barmode="group",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis_title="Column",
        yaxis_title="Outlier Count",
        margin=dict(l=20, r=20, t=40, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def build_distribution_comparison_chart(
    df_orig: pd.DataFrame,
    df_proc: pd.DataFrame,
    column: str,
) -> Optional[go.Figure]:
    """Build side-by-side distribution histogram/KDE for a specific numerical column."""
    if column not in df_orig.columns or column not in df_proc.columns:
        return None

    s1 = df_orig[column].dropna()
    s2 = df_proc[column].dropna()
    if len(s1) == 0 or len(s2) == 0:
        return None

    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=s1,
            name="Original Distribution",
            opacity=0.6,
            marker_color="#818CF8",
        )
    )
    fig.add_trace(
        go.Histogram(
            x=s2,
            name="Cleaned Distribution",
            opacity=0.6,
            marker_color="#34D399",
        )
    )

    fig.update_layout(
        title=f"Distribution Comparison: '{column}'",
        barmode="overlay",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis_title=column,
        yaxis_title="Frequency",
        margin=dict(l=20, r=20, t=40, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig
