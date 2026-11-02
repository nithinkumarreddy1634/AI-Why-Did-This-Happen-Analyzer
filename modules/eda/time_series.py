"""Time-series exploratory analysis and temporal aggregation engine."""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from utils.eda_helpers import get_eda_datetime_columns, get_eda_numerical_columns


def detect_time_series_candidates(df: pd.DataFrame) -> List[str]:
    """Identify datetime or date-formatted columns suitable for temporal analysis."""
    return get_eda_datetime_columns(df)


def compute_time_series_aggregation(
    df: pd.DataFrame,
    date_col: str,
    metric_col: str,
    freq: str = "M",
    agg_func: str = "mean",
    rolling_window: Optional[int] = None,
) -> Dict[str, Any]:
    """Aggregate a numerical metric across temporal intervals with optional rolling average.
    
    Args:
        df: Input DataFrame.
        date_col: Datetime column.
        metric_col: Numerical column to aggregate.
        freq: Frequency string ('D', 'W', 'M', 'Q', 'Y').
        agg_func: 'mean', 'sum', 'median', 'min', 'max', 'count'.
        rolling_window: Optional window size for moving average.
    """
    if date_col not in df.columns or metric_col not in df.columns:
        raise ValueError("Selected date or metric column does not exist in dataset.")

    # Safe parsing of date column
    parsed_dates = pd.to_datetime(df[date_col], errors="coerce")
    valid_mask = parsed_dates.notna() & df[metric_col].notna()

    if valid_mask.sum() < 2:
        return {
            "valid": False,
            "message": "Fewer than 2 valid timestamped records available for time-series analysis.",
            "ts_df": pd.DataFrame(),
        }

    sub_df = pd.DataFrame({
        "timestamp": parsed_dates[valid_mask],
        "metric": df.loc[valid_mask, metric_col].astype(float),
    })

    # Sort chronologically
    sub_df = sub_df.sort_values(by="timestamp").reset_index(drop=True)

    # Standardize pandas frequency alias (avoid deprecated 'M', 'Q', 'Y' warnings)
    freq_clean = freq.upper()
    freq_map = {
        "D": "D",
        "W": "W",
        "M": "ME",
        "ME": "ME",
        "Q": "QE",
        "QE": "QE",
        "Y": "YE",
        "YE": "YE",
    }
    pd_freq = freq_map.get(freq_clean, "ME")

    # Resample
    resampler = sub_df.set_index("timestamp").resample(pd_freq)
    
    if agg_func == "sum":
        agg_series = resampler["metric"].sum()
    elif agg_func == "median":
        agg_series = resampler["metric"].median()
    elif agg_func == "min":
        agg_series = resampler["metric"].min()
    elif agg_func == "max":
        agg_series = resampler["metric"].max()
    elif agg_func == "count":
        agg_series = resampler["metric"].count()
    else:
        agg_series = resampler["metric"].mean()

    ts_df = agg_series.reset_index()
    ts_df.columns = ["Timestamp", metric_col]
    ts_df = ts_df.dropna().reset_index(drop=True)

    # Optional rolling average
    rolling_col_name = None
    if rolling_window and rolling_window > 1 and len(ts_df) >= rolling_window:
        rolling_col_name = f"{rolling_window}-Period Rolling Avg"
        ts_df[rolling_col_name] = ts_df[metric_col].rolling(rolling_window).mean().round(2)

    ts_df[metric_col] = ts_df[metric_col].round(2)

    return {
        "valid": True,
        "date_col": date_col,
        "metric_col": metric_col,
        "frequency": freq_clean,
        "aggregation_function": agg_func,
        "rolling_col": rolling_col_name,
        "ts_df": ts_df,
        "start_date": str(sub_df["timestamp"].min().date()),
        "end_date": str(sub_df["timestamp"].max().date()),
        "total_periods": len(ts_df),
    }

