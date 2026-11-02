"""Multivariate feature exploration engine."""

from typing import Dict, Any, List, Optional
import pandas as pd
from utils.eda_helpers import get_eda_numerical_columns


def prepare_multivariate_matrix(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    color_column: Optional[str] = None,
    max_columns: int = 5,
) -> Dict[str, Any]:
    """Prepare a bounded subset of numerical features for multi-variable visual exploration.
    
    Caps columns to a safe maximum (default 5) to ensure responsive rendering.
    """
    if df is None or df.empty:
        return {"valid": False, "message": "Dataset is empty."}

    num_cols = get_eda_numerical_columns(df)
    if not num_cols:
        return {"valid": False, "message": "No numerical features available for multivariate analysis."}

    if columns:
        selected_cols = [c for c in columns if c in num_cols][:max_columns]
    else:
        selected_cols = num_cols[:max_columns]

    if len(selected_cols) < 2:
        return {"valid": False, "message": "Select at least 2 numerical features for multivariate matrix."}

    include_cols = list(selected_cols)
    if color_column and color_column in df.columns and color_column not in include_cols:
        include_cols.append(color_column)

    # Subsample if dataset is very large to keep browser snappy
    sample_df = df[include_cols].dropna()
    if len(sample_df) > 1000:
        sample_df = sample_df.sample(1000, random_state=42)

    return {
        "valid": True,
        "selected_columns": selected_cols,
        "color_column": color_column,
        "sample_df": sample_df,
        "sample_size": len(sample_df),
        "total_records": len(df),
    }

