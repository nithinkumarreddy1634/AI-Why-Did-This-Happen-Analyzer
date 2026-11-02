"""Target variable exploratory analysis engine."""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from utils.eda_helpers import get_eda_numerical_columns, get_eda_categorical_columns


def analyze_target_variable(
    df: pd.DataFrame,
    target_col: Optional[str] = None,
) -> Dict[str, Any]:
    """Examine target variable distribution and associations with all other features.
    
    Supports both numerical (regression target) and categorical (classification target) variables.
    """
    if df is None or df.empty:
        return {"has_target": False, "message": "Dataset is empty."}

    if not target_col or target_col not in df.columns:
        return {
            "has_target": False,
            "message": "No target variable selected. Select a target column to view outcome-specific EDA.",
            "available_columns": list(df.columns),
        }

    series = df[target_col].dropna()
    if series.empty:
        return {
            "has_target": True,
            "target_col": target_col,
            "valid": False,
            "message": f"Target column '{target_col}' contains only missing values.",
        }

    is_num = pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series)
    unique_count = series.nunique()

    # Determine if numerical target or categorical target
    if is_num and unique_count > 10:
        target_kind = "numerical"
    else:
        target_kind = "categorical"

    total_valid = len(series)
    missing_cnt = int(df[target_col].isna().sum())

    if target_kind == "numerical":
        s_float = series.astype(float)
        mean_val = float(s_float.mean())
        median_val = float(s_float.median())
        std_val = float(s_float.std(ddof=1)) if total_valid > 1 else 0.0

        # Calculate correlations with all other numerical features
        num_cols = [c for c in get_eda_numerical_columns(df) if c != target_col]
        correlations = []
        for c in num_cols:
            sub = df[[c, target_col]].dropna().astype(float)
            if len(sub) > 2 and sub[c].nunique() > 1:
                r_val = float(sub.corr().iloc[0, 1])
                if not np.isnan(r_val):
                    correlations.append({
                        "Feature": c,
                        "Correlation": round(r_val, 4),
                        "AbsCorrelation": abs(r_val),
                        "Direction": "Positive" if r_val > 0 else "Negative",
                    })

        corr_df = pd.DataFrame(correlations)
        if not corr_df.empty:
            corr_df = corr_df.sort_values(by="AbsCorrelation", ascending=False).reset_index(drop=True)

        return {
            "has_target": True,
            "valid": True,
            "target_col": target_col,
            "target_kind": "numerical",
            "count": total_valid,
            "missing_count": missing_cnt,
            "mean": round(mean_val, 2),
            "median": round(median_val, 2),
            "std": round(std_val, 2),
            "min": round(float(s_float.min()), 2),
            "max": round(float(s_float.max()), 2),
            "correlations_df": corr_df,
            "message": f"Target '{target_col}' is a continuous numerical outcome (mean={mean_val:.2f}, median={median_val:.2f}).",
        }

    else:
        # Categorical outcome
        val_counts = series.astype(str).value_counts()
        class_records = []
        for cat, cnt in val_counts.items():
            pct = round((cnt / total_valid * 100.0), 2)
            class_records.append({
                "Class": str(cat),
                "Count": int(cnt),
                "Percentage": pct,
            })
        class_df = pd.DataFrame(class_records)

        # Check class imbalance
        min_pct = class_df["Percentage"].min()
        max_pct = class_df["Percentage"].max()
        imbalance_ratio = round(max_pct / max(0.1, min_pct), 2)

        if imbalance_ratio >= 10.0:
            imbalance_status = "Severe Imbalance"
        elif imbalance_ratio >= 3.0:
            imbalance_status = "Moderate Imbalance"
        else:
            imbalance_status = "Balanced"

        return {
            "has_target": True,
            "valid": True,
            "target_col": target_col,
            "target_kind": "categorical",
            "count": total_valid,
            "missing_count": missing_cnt,
            "unique_classes": unique_count,
            "class_distribution_df": class_df,
            "imbalance_status": imbalance_status,
            "imbalance_ratio": imbalance_ratio,
            "message": f"Target '{target_col}' has {unique_count} classes ({imbalance_status}, ratio {imbalance_ratio}:1).",
        }

