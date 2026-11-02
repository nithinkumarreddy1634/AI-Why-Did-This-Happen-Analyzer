"""Automated factual insight generation engine for Exploratory Data Analysis."""

from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd

from utils.eda_helpers import (
    get_eda_numerical_columns,
    get_eda_categorical_columns,
    safe_skewness,
)
from modules.eda.correlation import compute_correlation_matrix, extract_top_correlations
from modules.eda.target_analysis import analyze_target_variable


def generate_eda_insights(
    df: pd.DataFrame,
    target_col: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Synthesize data-driven, strictly factual observations from real dataset calculations.
    
    Generates structured insights across:
    - Distribution shapes & skewness
    - Dominant categories and high-cardinality flags
    - Strongest pairwise correlations (with non-causation caveat)
    - Target variable characteristics & class imbalance
    - Data health impact on exploration
    """
    if df is None or df.empty:
        return [
            {
                "category": "Dataset",
                "badge": "Empty",
                "badge_color": "blue",
                "insight": "Dataset is empty. No analytical insights can be calculated.",
            }
        ]

    insights: List[Dict[str, Any]] = []
    n_rows, n_cols = df.shape
    num_cols = get_eda_numerical_columns(df)
    cat_cols = get_eda_categorical_columns(df)

    # 1. Dataset Dimensions Insight
    insights.append({
        "category": "Overview",
        "badge": "Dimension",
        "badge_color": "blue",
        "insight": (
            f"Dataset comprises **{n_rows:,} records** and **{n_cols} features** "
            f"({len(num_cols)} numerical, {len(cat_cols)} categorical/discrete)."
        ),
    })

    # 2. Skewness and Distribution Shapes
    skewed_features = []
    for col in num_cols:
        series = df[col].dropna()
        if len(series) > 10 and series.nunique() > 2:
            skew_val = safe_skewness(series)
            if skew_val is not None:
                if skew_val > 1.0:
                    skewed_features.append((col, "right-skewed (positive skew)", skew_val))
                elif skew_val < -1.0:
                    skewed_features.append((col, "left-skewed (negative skew)", skew_val))

    if skewed_features:
        # Sort by absolute skewness
        skewed_features.sort(key=lambda x: abs(x[2]), reverse=True)
        top_skew = skewed_features[:2]
        desc_parts = [f"`'{name}'` ({desc}, skew = {val:+.2f})" for name, desc, val in top_skew]
        insights.append({
            "category": "Distribution",
            "badge": "Skewness",
            "badge_color": "purple",
            "insight": (
                f"Significant distributional asymmetry detected in {', '.join(desc_parts)}. "
                "Consider median over mean for central tendency or log-transformations in downstream modeling."
            ),
        })

    # 3. Categorical Dominance / Monopolization
    for col in cat_cols[:5]:
        series = df[col].dropna().astype(str)
        if not series.empty:
            vc = series.value_counts(normalize=True)
            top_cat = vc.index[0]
            top_pct = vc.iloc[0] * 100.0
            if top_pct >= 60.0:
                insights.append({
                    "category": "Categorical",
                    "badge": "Dominant Class",
                    "badge_color": "amber",
                    "insight": (
                        f"In feature `'{col}'`, category **'{top_cat}'** accounts for **{top_pct:.1f}%** "
                        f"of all non-null records ({series.value_counts().iloc[0]:,} rows)."
                    ),
                })
                break

    # 4. Correlation Associations
    if len(num_cols) >= 2:
        corr_matrix, meta = compute_correlation_matrix(df, columns=num_cols, method="pearson")
        if not corr_matrix.empty:
            top_corrs = extract_top_correlations(corr_matrix, threshold=0.4, top_n=2)
            if not top_corrs.empty:
                r1 = top_corrs.iloc[0]
                f1, f2, r_val, direction, strength = (
                    r1["Feature 1"],
                    r1["Feature 2"],
                    r1["Correlation"],
                    r1["Direction"],
                    r1["Strength"],
                )
                insights.append({
                    "category": "Correlation",
                    "badge": "Association",
                    "badge_color": "blue",
                    "insight": (
                        f"**{strength} {direction} correlation** identified between `'{f1}'` and `'{f2}'` "
                        f"($r = {r_val:+.3f}$). *Note: Correlation reflects linear association, not direct causality.*"
                    ),
                })

    # 5. Target Variable Insights
    if target_col and target_col in df.columns:
        target_info = analyze_target_variable(df, target_col)
        if target_info.get("valid"):
            if target_info["target_kind"] == "categorical":
                status = target_info["imbalance_status"]
                ratio = target_info["imbalance_ratio"]
                df_cls = target_info["class_distribution_df"]
                classes_summary = ", ".join([f"'{row['Class']}': {row['Percentage']}%" for _, row in df_cls.head(3).iterrows()])
                badge_type = "emerald" if status == "Balanced" else "amber"
                insights.append({
                    "category": "Target Variable",
                    "badge": status,
                    "badge_color": badge_type,
                    "insight": (
                        f"Selected target `'{target_col}'` exhibits **{status}** ({classes_summary}). "
                        f"Imbalance ratio is **{ratio:.1f}:1**."
                    ),
                })
            else:
                insights.append({
                    "category": "Target Variable",
                    "badge": "Continuous Target",
                    "badge_color": "purple",
                    "insight": (
                        f"Target `'{target_col}'` is a continuous numerical variable ranging from **{target_info['min']}** "
                        f"to **{target_info['max']}** (Mean: {target_info['mean']}, Median: {target_info['median']})."
                    ),
                })

    # 6. Missingness Impact
    total_cells = df.size
    total_missing = int(df.isna().sum().sum())
    if total_missing > 0:
        miss_pct = round((total_missing / total_cells * 100.0), 2)
        worst_col = df.isna().sum().idxmax()
        worst_cnt = int(df[worst_col].isna().sum())
        insights.append({
            "category": "Data Health",
            "badge": "Missing Data",
            "badge_color": "amber",
            "insight": (
                f"Dataset contains **{total_missing:,} missing values** ({miss_pct}% overall). "
                f"Highest missingness occurs in `'{worst_col}'` with {worst_cnt:,} null entries."
            ),
        })

    return insights

