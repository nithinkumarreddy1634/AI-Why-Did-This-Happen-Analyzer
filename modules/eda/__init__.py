"""Exploratory Data Analysis (EDA) package exporting all analysis engines."""

from typing import Optional, Dict, Any, List
import pandas as pd

from modules.eda.univariate import (
    compute_numerical_univariate_stats,
    compute_all_numerical_summaries,
)
from modules.eda.categorical_analysis import (
    compute_categorical_univariate_stats,
    compute_all_categorical_summaries,
)
from modules.eda.bivariate import (
    analyze_num_vs_num,
    analyze_cat_vs_num,
    analyze_cat_vs_cat,
)
from modules.eda.correlation import (
    compute_correlation_matrix,
    extract_top_correlations,
)
from modules.eda.target_analysis import (
    analyze_target_variable,
)
from modules.eda.multivariate import (
    prepare_multivariate_matrix,
)
from modules.eda.time_series import (
    detect_time_series_candidates,
    compute_time_series_aggregation,
)
from modules.eda.segment_analysis import (
    compute_segment_breakdown,
)
from modules.eda.insight_generator import (
    generate_eda_insights,
)


class EDAManager:
    """Master facade managing exploratory analysis, distributions, and correlations."""

    def __init__(self, df: Optional[pd.DataFrame] = None, target_column: Optional[str] = None) -> None:
        self.df = df
        self.target_column = target_column

    def compute_correlations(
        self,
        df: Optional[pd.DataFrame] = None,
        columns: Optional[List[str]] = None,
        method: str = "pearson",
    ) -> pd.DataFrame:
        """Compute numerical correlation matrix."""
        target_df = df if df is not None else self.df
        if target_df is None:
            raise ValueError("No dataset provided for correlation calculation.")
        corr_matrix, _ = compute_correlation_matrix(target_df, columns=columns, method=method)
        return corr_matrix

    def compute_numerical_stats(self, column: str, df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """Compute univariate stats for a numerical column."""
        target_df = df if df is not None else self.df
        if target_df is None:
            raise ValueError("No dataset provided.")
        return compute_numerical_univariate_stats(target_df, column)

    def compute_categorical_stats(
        self, column: str, df: Optional[pd.DataFrame] = None, top_n: int = 10, group_other: bool = False
    ) -> Dict[str, Any]:
        """Compute univariate stats for a categorical column."""
        target_df = df if df is not None else self.df
        if target_df is None:
            raise ValueError("No dataset provided.")
        return compute_categorical_univariate_stats(target_df, column, top_n=top_n, group_other=group_other)

    def generate_insights(self, target_column: Optional[str] = None, df: Optional[pd.DataFrame] = None) -> List[Dict[str, Any]]:
        """Generate dynamic factual observations."""
        target_df = df if df is not None else self.df
        if target_df is None:
            return []
        tgt = target_column or self.target_column
        return generate_eda_insights(target_df, target_col=tgt)


__all__ = [
    "EDAManager",
    "compute_numerical_univariate_stats",
    "compute_all_numerical_summaries",
    "compute_categorical_univariate_stats",
    "compute_all_categorical_summaries",
    "analyze_num_vs_num",
    "analyze_cat_vs_num",
    "analyze_cat_vs_cat",
    "compute_correlation_matrix",
    "extract_top_correlations",
    "analyze_target_variable",
    "prepare_multivariate_matrix",
    "detect_time_series_candidates",
    "compute_time_series_aggregation",
    "compute_segment_breakdown",
    "generate_eda_insights",
]
