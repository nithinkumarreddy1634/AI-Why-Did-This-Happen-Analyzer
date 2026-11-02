"""Correlation analysis engine supporting Pearson and Spearman associations."""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
from utils.eda_helpers import get_eda_numerical_columns


def compute_correlation_matrix(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    method: str = "pearson",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Compute numerical correlation matrix excluding constant columns.
    
    Args:
        df: Input DataFrame.
        columns: Optional list of numerical columns to include.
        method: 'pearson' (linear) or 'spearman' (rank-based monotonic).
    
    Returns:
        (corr_df, metadata_dict)
    """
    if df is None or df.empty:
        return pd.DataFrame(), {"error": "Dataset is empty."}

    all_num_cols = get_eda_numerical_columns(df)
    target_cols = [c for c in columns if c in all_num_cols] if columns else all_num_cols

    # Identify constant or low-variance columns to exclude
    valid_cols = []
    excluded_cols = []
    for c in target_cols:
        series = df[c].dropna()
        if len(series) < 2 or series.nunique() <= 1:
            excluded_cols.append(c)
        else:
            valid_cols.append(c)

    if len(valid_cols) < 2:
        return pd.DataFrame(), {
            "error": "At least 2 non-constant numerical columns are required for correlation analysis.",
            "valid_columns": valid_cols,
            "excluded_constant_columns": excluded_cols,
        }

    sub_df = df[valid_cols].astype(float)
    corr_matrix = sub_df.corr(method=method).round(4)

    metadata = {
        "method": method.capitalize(),
        "total_analyzed_columns": len(valid_cols),
        "excluded_constant_columns": excluded_cols,
        "sample_size": len(df),
        "causation_disclaimer": "Important: Correlation indicates statistical association, NOT causation.",
    }

    return corr_matrix, metadata


def extract_top_correlations(
    corr_matrix: pd.DataFrame,
    threshold: float = 0.3,
    top_n: int = 10,
) -> pd.DataFrame:
    """Extract and rank the strongest positive and negative pairwise correlations.
    
    Returns a DataFrame with columns:
    ['Feature 1', 'Feature 2', 'Correlation', 'Direction', 'Strength']
    """
    if corr_matrix.empty or corr_matrix.shape[0] < 2:
        return pd.DataFrame(columns=["Feature 1", "Feature 2", "Correlation", "Direction", "Strength"])

    cols = corr_matrix.columns
    pairs = []

    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            col1 = cols[i]
            col2 = cols[j]
            val = corr_matrix.loc[col1, col2]
            if pd.isna(val):
                continue

            r_val = float(val)
            abs_val = abs(r_val)

            if abs_val >= threshold:
                direction = "Positive" if r_val > 0 else "Negative"
                if abs_val >= 0.7:
                    strength = "Strong"
                elif abs_val >= 0.5:
                    strength = "Moderate"
                else:
                    strength = "Weak"

                pairs.append({
                    "Feature 1": col1,
                    "Feature 2": col2,
                    "Correlation": r_val,
                    "AbsCorrelation": abs_val,
                    "Direction": direction,
                    "Strength": strength,
                })

    df_pairs = pd.DataFrame(pairs)
    if not df_pairs.empty:
        df_pairs = df_pairs.sort_values(by="AbsCorrelation", ascending=False).head(top_n)
        df_pairs = df_pairs.drop(columns=["AbsCorrelation"]).reset_index(drop=True)
    return df_pairs

