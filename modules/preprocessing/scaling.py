"""Numerical feature scaling: StandardScaler, MinMaxScaler, RobustScaler."""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd


class FeatureScaler:
    """Base interface for feature scaling transformers avoiding data leakage."""

    def fit(self, df: pd.DataFrame, columns: List[str]) -> "FeatureScaler":
        raise NotImplementedError

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError

    def fit_transform(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        return self.fit(df, columns).transform(df)


class StandardScalerWrapper(FeatureScaler):
    """Z-score standardization: z = (x - mean) / std.
    
    Handles zero variance (constant columns) safely by maintaining 0.0 without division by zero.
    """

    def __init__(self) -> None:
        self.columns: List[str] = []
        self.means: Dict[str, float] = {}
        self.scales: Dict[str, float] = {}

    def fit(self, df: pd.DataFrame, columns: List[str]) -> "StandardScalerWrapper":
        self.columns = [c for c in columns if c in df.columns]
        self.means = {}
        self.scales = {}

        for col in self.columns:
            series = df[col].dropna()
            mean_val = float(series.mean()) if len(series) > 0 else 0.0
            std_val = float(series.std(ddof=0)) if len(series) > 1 else 0.0
            
            # Constant or zero-variance column protection
            if std_val == 0.0 or np.isnan(std_val):
                std_val = 1.0  # Avoid division by zero
            
            self.means[col] = mean_val
            self.scales[col] = std_val
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        new_df = df.copy()
        for col in self.columns:
            if col in new_df.columns:
                mean_val = self.means.get(col, 0.0)
                std_val = self.scales.get(col, 1.0)
                # Ensure float representation to avoid pandas 2.x in-place cast errors
                s = new_df[col].astype(float)
                new_df[col] = (s - mean_val) / std_val
        return new_df


class MinMaxScalerWrapper(FeatureScaler):
    """Min-Max normalization: z = (x - min) / (max - min) * (feature_range[1] - feature_range[0]) + feature_range[0].
    
    Handles constant columns safely by mapping to the range minimum without division by zero.
    """

    def __init__(self, feature_range: Tuple[float, float] = (0.0, 1.0)) -> None:
        self.feature_range = feature_range
        self.columns: List[str] = []
        self.mins: Dict[str, float] = {}
        self.maxs: Dict[str, float] = {}

    def fit(self, df: pd.DataFrame, columns: List[str]) -> "MinMaxScalerWrapper":
        self.columns = [c for c in columns if c in df.columns]
        self.mins = {}
        self.maxs = {}

        for col in self.columns:
            series = df[col].dropna()
            min_val = float(series.min()) if len(series) > 0 else 0.0
            max_val = float(series.max()) if len(series) > 0 else 0.0
            self.mins[col] = min_val
            self.maxs[col] = max_val
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        new_df = df.copy()
        low_target, high_target = self.feature_range
        
        for col in self.columns:
            if col in new_df.columns:
                min_val = self.mins.get(col, 0.0)
                max_val = self.maxs.get(col, 0.0)
                data_range = max_val - min_val
                s = new_df[col].astype(float)

                if data_range == 0.0:
                    # Constant column: map to lower bound of feature range
                    new_df[col] = np.where(s.notna(), low_target, np.nan)
                else:
                    norm = (s - min_val) / data_range
                    new_df[col] = norm * (high_target - low_target) + low_target
        return new_df


class RobustScalerWrapper(FeatureScaler):
    """Robust scaling using median and Interquartile Range (IQR = Q3 - Q1).
    
    Robust to heavy outlier presence.
    """

    def __init__(self, quantile_range: Tuple[float, float] = (25.0, 75.0)) -> None:
        self.quantile_range = quantile_range
        self.columns: List[str] = []
        self.medians: Dict[str, float] = {}
        self.iqrs: Dict[str, float] = {}

    def fit(self, df: pd.DataFrame, columns: List[str]) -> "RobustScalerWrapper":
        self.columns = [c for c in columns if c in df.columns]
        self.medians = {}
        self.iqrs = {}

        q_min, q_max = self.quantile_range
        for col in self.columns:
            series = df[col].dropna()
            median_val = float(series.median()) if len(series) > 0 else 0.0
            q1 = float(series.quantile(q_min / 100.0)) if len(series) > 0 else 0.0
            q3 = float(series.quantile(q_max / 100.0)) if len(series) > 0 else 0.0
            iqr_val = q3 - q1

            if iqr_val == 0.0 or np.isnan(iqr_val):
                iqr_val = 1.0  # Fallback to avoid division by zero

            self.medians[col] = median_val
            self.iqrs[col] = iqr_val
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        new_df = df.copy()
        for col in self.columns:
            if col in new_df.columns:
                median_val = self.medians.get(col, 0.0)
                iqr_val = self.iqrs.get(col, 1.0)
                s = new_df[col].astype(float)
                new_df[col] = (s - median_val) / iqr_val
        return new_df


def apply_scaling(
    df: pd.DataFrame,
    columns: List[str],
    method: str = "standard",
    target_column: Optional[str] = None,
    feature_range: Tuple[float, float] = (0.0, 1.0),
) -> Tuple[pd.DataFrame, FeatureScaler, str]:
    """Execute feature scaling on numerical columns with safety guards.
    
    Args:
        df: Input DataFrame.
        columns: List of columns to scale.
        method: 'standard', 'minmax', or 'robust'.
        target_column: Column name of target variable (excluded by default).
        feature_range: For MinMaxScaler, tuple of (min, max).

    Returns:
    (new_df, fitted_scaler, message)
    """
    safe_columns = []
    skipped_target = False
    skipped_constant = []

    for col in columns:
        if col not in df.columns:
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue
        if target_column and col == target_column:
            skipped_target = True
            continue
        # Note constant columns
        if df[col].dropna().nunique() <= 1:
            skipped_constant.append(col)
        safe_columns.append(col)

    if not safe_columns:
        return df.copy(), None, "No valid numerical feature columns selected for scaling."

    method = method.lower()
    if method in ("standard", "standard_scaler", "zscore"):
        scaler = StandardScalerWrapper()
    elif method in ("minmax", "min_max_scaler", "normalize"):
        scaler = MinMaxScalerWrapper(feature_range=feature_range)
    elif method in ("robust", "robust_scaler"):
        scaler = RobustScalerWrapper()
    else:
        raise ValueError(f"Unsupported scaling method: '{method}'")

    scaler.fit(df, safe_columns)
    new_df = scaler.transform(df)

    msg = f"Scaled {len(safe_columns)} numerical column(s) using {method.upper()}."
    if skipped_target:
        msg += f" (Protected target variable '{target_column}')."
    if skipped_constant:
        msg += f" Note: constant column(s) {', '.join(skipped_constant)} were handled safely without division by zero."

    return new_df, scaler, msg
