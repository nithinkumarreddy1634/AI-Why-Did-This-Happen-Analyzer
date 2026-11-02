"""Time-based change, spike, and drop detection engine."""

from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np

from utils.detection_helpers import safe_pct_change
from modules.unexpected_detection.severity_classifier import (
    SeverityLevel,
    SeverityClassifier,
)


class TimeBasedChangeDetector:
    """Detects sudden period-over-period spikes, crashes, and structural trends over time."""

    def __init__(self, classifier: Optional[SeverityClassifier] = None) -> None:
        self.classifier = classifier or SeverityClassifier()

    def analyze_temporal_trend(
        self,
        df: pd.DataFrame,
        date_col: str,
        metric_col: str,
        freq: str = "M",
        agg_func: str = "mean",
        anomaly_threshold_pct: float = 30.0,
    ) -> Dict[str, Any]:
        """Aggregate metric over time and identify sudden period-over-period swings.
        
        Args:
            df: Dataset.
            date_col: Datetime column name.
            metric_col: Numerical or categorical metric column.
            freq: 'D' (Daily), 'W' (Weekly), 'ME' (Monthly), 'QE' (Quarterly).
            agg_func: 'mean', 'sum', 'count', or 'median'.
            anomaly_threshold_pct: Relative % change considered an unusual spike/drop.
        """
        # Copy and clean datetime column
        temp_df = df[[date_col, metric_col]].copy()
        temp_df[date_col] = pd.to_datetime(temp_df[date_col], errors="coerce", format="mixed")
        temp_df = temp_df.dropna(subset=[date_col])

        if temp_df.empty:
            return {
                "status": "warning",
                "message": f"Column '{date_col}' contains no valid datetime values.",
                "timeline": pd.DataFrame(),
                "spikes": pd.DataFrame(),
            }

        # Check if metric is numeric or categorical
        is_numeric = pd.api.types.is_numeric_dtype(temp_df[metric_col])
        if not is_numeric:
            # If categorical, aggregate counts or default to frequency
            agg_func = "count"

        # Standardize frequency alias for pandas 2.2+ (M -> ME, Q -> QE)
        freq_alias = freq
        if freq == "M":
            freq_alias = "ME"
        elif freq == "Q":
            freq_alias = "QE"

        # Group by datetime frequency
        temp_df = temp_df.set_index(date_col)
        
        try:
            if agg_func == "mean":
                grouped = temp_df[metric_col].resample(freq_alias).mean()
            elif agg_func == "sum":
                grouped = temp_df[metric_col].resample(freq_alias).sum()
            elif agg_func == "median":
                grouped = temp_df[metric_col].resample(freq_alias).median()
            else:
                grouped = temp_df[metric_col].resample(freq_alias).count()
        except Exception as e:
            return {
                "status": "error",
                "message": f"Resampling error with frequency '{freq}': {str(e)}",
                "timeline": pd.DataFrame(),
                "spikes": pd.DataFrame(),
            }

        timeline_df = grouped.reset_index()
        timeline_df.columns = ["period", "value"]
        timeline_df["period_str"] = timeline_df["period"].dt.strftime("%Y-%m-%d")

        # Compute period-over-period change
        timeline_df["previous_value"] = timeline_df["value"].shift(1)
        timeline_df["absolute_change"] = timeline_df["value"] - timeline_df["previous_value"]
        
        # Calculate safe percentage change
        timeline_df["pct_change"] = [
            safe_pct_change(b, c)
            for b, c in zip(timeline_df["previous_value"], timeline_df["value"])
        ]

        # Flag spikes and crashes
        timeline_df["is_spike"] = False
        timeline_df["is_drop"] = False
        timeline_df["severity"] = SeverityLevel.NORMAL.value

        spikes_list = []
        for idx, row in timeline_df.iterrows():
            pct = row["pct_change"]
            if pct is not None:
                sev, _ = self.classifier.classify_percentage_change(pct)
                timeline_df.at[idx, "severity"] = sev.value
                row_dict = row.to_dict()
                row_dict["severity"] = sev.value
                if pct >= anomaly_threshold_pct:
                    timeline_df.at[idx, "is_spike"] = True
                    row_dict["is_spike"] = True
                    spikes_list.append(row_dict)
                elif pct <= -anomaly_threshold_pct:
                    timeline_df.at[idx, "is_drop"] = True
                    row_dict["is_drop"] = True
                    spikes_list.append(row_dict)

        spikes_df = pd.DataFrame(spikes_list)
        if not spikes_df.empty:
            spikes_df = spikes_df.sort_values(by="pct_change", key=abs, ascending=False)

        # Detect overall trend direction (first valid vs last valid)
        valid_values = timeline_df["value"].dropna()
        overall_change_pct = None
        if len(valid_values) >= 2:
            first_val = float(valid_values.iloc[0])
            last_val = float(valid_values.iloc[-1])
            overall_change_pct = safe_pct_change(first_val, last_val)

        return {
            "status": "success",
            "date_col": date_col,
            "metric_col": metric_col,
            "freq": freq,
            "agg_func": agg_func,
            "periods_count": len(timeline_df),
            "unusual_periods_count": len(spikes_list),
            "overall_change_pct": overall_change_pct,
            "timeline": timeline_df,
            "spikes": spikes_df,
            "message": f"Analyzed {len(timeline_df)} periods. Found {len(spikes_list)} unusual transitions exceeding ±{anomaly_threshold_pct}%.",
        }
