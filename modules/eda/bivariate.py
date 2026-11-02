"""Bivariate relationship analysis engine: Num vs Num, Cat vs Num, and Cat vs Cat."""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import stats


def analyze_num_vs_num(
    df: pd.DataFrame,
    col_x: str,
    col_y: str,
) -> Dict[str, Any]:
    """Analyze the relationship between two numerical variables."""
    if col_x not in df.columns or col_y not in df.columns:
        raise ValueError("Selected columns do not exist in dataset.")

    # Drop missing values pairwise
    valid_mask = df[col_x].notna() & df[col_y].notna()
    x = df.loc[valid_mask, col_x].astype(float)
    y = df.loc[valid_mask, col_y].astype(float)
    n_pairs = len(x)

    if n_pairs < 2 or x.nunique() <= 1 or y.nunique() <= 1:
        return {
            "valid": False,
            "col_x": col_x,
            "col_y": col_y,
            "sample_size": n_pairs,
            "pearson_corr": 0.0,
            "pearson_pvalue": 1.0,
            "spearman_corr": 0.0,
            "spearman_pvalue": 1.0,
            "covariance": 0.0,
            "slope": 0.0,
            "intercept": 0.0,
            "r_squared": 0.0,
            "message": "Insufficient variation or sample size for correlation analysis.",
        }

    # Pearson
    p_corr, p_pval = stats.pearsonr(x, y)
    if np.isnan(p_corr):
        p_corr, p_pval = 0.0, 1.0

    # Spearman
    s_corr, s_pval = stats.spearmanr(x, y)
    if np.isnan(s_corr):
        s_corr, s_pval = 0.0, 1.0

    # Covariance
    cov = float(np.cov(x, y)[0, 1])

    # Linear Regression (slope, intercept, r-squared)
    slope, intercept, r_val, p_val_reg, std_err = stats.linregress(x, y)
    r_squared = round(float(r_val ** 2), 4) if not np.isnan(r_val) else 0.0

    # Correlation interpretation
    abs_corr = abs(p_corr)
    if abs_corr >= 0.7:
        strength = "Strong"
    elif abs_corr >= 0.4:
        strength = "Moderate"
    elif abs_corr >= 0.2:
        strength = "Weak"
    else:
        strength = "Very Weak / Negligible"

    direction = "Positive" if p_corr > 0 else ("Negative" if p_corr < 0 else "None")

    return {
        "valid": True,
        "col_x": col_x,
        "col_y": col_y,
        "sample_size": n_pairs,
        "pearson_corr": round(float(p_corr), 4),
        "pearson_pvalue": float(p_pval),
        "spearman_corr": round(float(s_corr), 4),
        "spearman_pvalue": float(s_pval),
        "covariance": round(cov, 4),
        "slope": round(float(slope), 4) if not np.isnan(slope) else 0.0,
        "intercept": round(float(intercept), 4) if not np.isnan(intercept) else 0.0,
        "r_squared": r_squared,
        "strength": strength,
        "direction": direction,
        "message": f"{strength} {direction} association (Pearson r = {p_corr:.3f}, R² = {r_squared:.3f}).",
    }


def analyze_cat_vs_num(
    df: pd.DataFrame,
    cat_col: str,
    num_col: str,
) -> Dict[str, Any]:
    """Analyze a numerical measurement partitioned across categorical segments."""
    if cat_col not in df.columns or num_col not in df.columns:
        raise ValueError("Selected columns do not exist in dataset.")

    valid_mask = df[cat_col].notna() & df[num_col].notna()
    subset = df.loc[valid_mask, [cat_col, num_col]].copy()
    subset[num_col] = subset[num_col].astype(float)
    subset[cat_col] = subset[cat_col].astype(str)

    if subset.empty:
        return {
            "valid": False,
            "cat_col": cat_col,
            "num_col": num_col,
            "group_summary_df": pd.DataFrame(),
            "overall_mean": 0.0,
            "overall_median": 0.0,
            "anova_f": None,
            "anova_p": None,
            "message": "No valid observations for group analysis.",
        }

    overall_mean = float(subset[num_col].mean())
    overall_median = float(subset[num_col].median())

    # Group-wise aggregation
    grouped = subset.groupby(cat_col)[num_col]
    summary_df = grouped.agg(
        Count="count",
        Mean="mean",
        Median="median",
        Std="std",
        Min="min",
        Max="max",
        Sum="sum",
    ).reset_index()

    total_records = len(subset)
    summary_df["Share %"] = (summary_df["Count"] / total_records * 100.0).round(2)
    summary_df["Mean"] = summary_df["Mean"].round(2)
    summary_df["Median"] = summary_df["Median"].round(2)
    summary_df["Std"] = summary_df["Std"].fillna(0.0).round(2)
    summary_df["Min"] = summary_df["Min"].round(2)
    summary_df["Max"] = summary_df["Max"].round(2)

    # Sort by Count descending
    summary_df = summary_df.sort_values(by="Count", ascending=False).reset_index(drop=True)

    # ANOVA test across groups if at least 2 groups with >= 2 items
    group_arrays = [group.values for _, group in grouped if len(group) >= 2]
    f_stat, p_val = None, None
    if len(group_arrays) >= 2:
        try:
            f_stat_res, p_val_res = stats.f_oneway(*group_arrays)
            if not np.isnan(f_stat_res):
                f_stat = round(float(f_stat_res), 4)
                p_val = float(p_val_res)
        except Exception:
            pass

    return {
        "valid": True,
        "cat_col": cat_col,
        "num_col": num_col,
        "group_summary_df": summary_df,
        "overall_mean": round(overall_mean, 2),
        "overall_median": round(overall_median, 2),
        "anova_f": f_stat,
        "anova_p": p_val,
        "is_significant": (p_val < 0.05) if p_val is not None else False,
    }


def analyze_cat_vs_cat(
    df: pd.DataFrame,
    cat_col1: str,
    cat_col2: str,
) -> Dict[str, Any]:
    """Cross-tabulate two categorical variables with frequencies and percentages."""
    if cat_col1 not in df.columns or cat_col2 not in df.columns:
        raise ValueError("Selected columns do not exist in dataset.")

    valid_mask = df[cat_col1].notna() & df[cat_col2].notna()
    subset = df.loc[valid_mask, [cat_col1, cat_col2]].astype(str)

    if subset.empty:
        return {
            "valid": False,
            "cat_col1": cat_col1,
            "cat_col2": cat_col2,
            "crosstab_counts": pd.DataFrame(),
            "crosstab_pct": pd.DataFrame(),
            "chi2_stat": None,
            "chi2_p": None,
            "dof": None,
        }

    crosstab_counts = pd.crosstab(subset[cat_col1], subset[cat_col2])
    crosstab_pct = pd.crosstab(subset[cat_col1], subset[cat_col2], normalize="index") * 100.0
    crosstab_pct = crosstab_pct.round(2)

    # Chi-Square Test
    chi2_stat, chi2_p, dof = None, None, None
    if crosstab_counts.shape[0] > 1 and crosstab_counts.shape[1] > 1:
        try:
            chi2, p_val, deg_f, _ = stats.chi2_contingency(crosstab_counts)
            chi2_stat = round(float(chi2), 4)
            chi2_p = float(p_val)
            dof = int(deg_f)
        except Exception:
            pass

    return {
        "valid": True,
        "cat_col1": cat_col1,
        "cat_col2": cat_col2,
        "crosstab_counts": crosstab_counts,
        "crosstab_pct": crosstab_pct,
        "chi2_stat": chi2_stat,
        "chi2_p": chi2_p,
        "dof": dof,
        "is_significant": (chi2_p < 0.05) if chi2_p is not None else False,
    }

