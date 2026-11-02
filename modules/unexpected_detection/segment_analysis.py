"""Segment-level change and cohort deviation detector."""

from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np

from utils.detection_helpers import safe_pct_change, safe_difference
from modules.unexpected_detection.severity_classifier import (
    SeverityLevel,
    SeverityClassifier,
)


class SegmentChangeDetector:
    """Discovers which sub-demographics, categories, or tiers underwent the greatest divergence."""

    def __init__(self, classifier: Optional[SeverityClassifier] = None) -> None:
        self.classifier = classifier or SeverityClassifier()

    def analyze_segment_shifts(
        self,
        baseline_df: pd.DataFrame,
        comparison_df: pd.DataFrame,
        segment_col: str,
        metric_col: str,
        baseline_label: str = "Baseline",
        comparison_label: str = "Comparison",
    ) -> Dict[str, Any]:
        """Compute changes across segments between baseline and comparison groups.
        
        Supports:
        - Numerical metrics: Mean of metric_col per segment
        - Binary/Categorical metrics: Rate/proportion of target per segment
        """
        if segment_col not in baseline_df.columns or segment_col not in comparison_df.columns:
            return {
                "status": "error",
                "message": f"Segment column '{segment_col}' not present in both datasets.",
                "ranked_segments": pd.DataFrame(),
            }

        if metric_col not in baseline_df.columns or metric_col not in comparison_df.columns:
            return {
                "status": "error",
                "message": f"Metric column '{metric_col}' not present in both datasets.",
                "ranked_segments": pd.DataFrame(),
            }

        is_numeric = pd.api.types.is_numeric_dtype(baseline_df[metric_col])

        # Group baseline
        if is_numeric:
            b_grp = baseline_df.groupby(segment_col)[metric_col].agg(
                count="count", value="mean"
            ).reset_index()
            c_grp = comparison_df.groupby(segment_col)[metric_col].agg(
                count="count", value="mean"
            ).reset_index()
            metric_type = "Mean Value"
        else:
            # If categorical/binary, compute event rate (e.g. proportion of the dominant positive label or counts)
            # Find the most frequent label in comparison or user positive label
            top_cat = comparison_df[metric_col].mode().iloc[0] if not comparison_df[metric_col].empty else None
            
            b_grp = baseline_df.groupby(segment_col).apply(
                lambda g: pd.Series({
                    "count": len(g),
                    "value": (g[metric_col] == top_cat).mean() * 100.0,
                }),
                include_groups=False,
            ).reset_index()
            
            c_grp = comparison_df.groupby(segment_col).apply(
                lambda g: pd.Series({
                    "count": len(g),
                    "value": (g[metric_col] == top_cat).mean() * 100.0,
                }),
                include_groups=False,
            ).reset_index()
            metric_type = f"Rate of '{top_cat}' (%)"

        merged = pd.merge(
            b_grp,
            c_grp,
            on=segment_col,
            how="outer",
            suffixes=(f"_{baseline_label}", f"_{comparison_label}"),
        ).fillna({"count_" + baseline_label: 0, "count_" + comparison_label: 0})

        # Calculate differences and percentage changes
        results = []
        for _, row in merged.iterrows():
            seg_name = row[segment_col]
            b_val = row.get(f"value_{baseline_label}")
            c_val = row.get(f"value_{comparison_label}")
            b_cnt = int(row.get(f"count_{baseline_label}", 0))
            c_cnt = int(row.get(f"count_{comparison_label}", 0))

            diff = safe_difference(b_val, c_val)
            pct_chg = safe_pct_change(b_val, c_val)

            if is_numeric:
                sev, sev_reason = self.classifier.classify_percentage_change(pct_chg)
            else:
                sev, sev_reason = self.classifier.classify_percentage_point_shift(diff if diff is not None else 0.0)

            results.append({
                "Segment": seg_name,
                f"{baseline_label} Count": b_cnt,
                f"{comparison_label} Count": c_cnt,
                f"{baseline_label} {metric_type}": round(b_val, 2) if pd.notna(b_val) else None,
                f"{comparison_label} {metric_type}": round(c_val, 2) if pd.notna(c_val) else None,
                "Absolute Difference": round(diff, 2) if diff is not None else None,
                "% Change": f"{pct_chg:+.1f}%" if pct_chg is not None else "N/A",
                "Severity": sev.value,
                "Justification": sev_reason,
                "_abs_sort_key": abs(diff) if diff is not None else 0.0,
            })

        ranked_df = pd.DataFrame(results)
        if not ranked_df.empty:
            ranked_df = ranked_df.sort_values(by="_abs_sort_key", ascending=False).drop(columns=["_abs_sort_key"])

        return {
            "status": "success",
            "segment_col": segment_col,
            "metric_col": metric_col,
            "metric_type": metric_type,
            "segments_count": len(ranked_df),
            "baseline_label": baseline_label,
            "comparison_label": comparison_label,
            "ranked_segments": ranked_df,
            "message": f"Evaluated {len(ranked_df)} distinct segments on '{segment_col}'. Highest shift observed in segment '{ranked_df.iloc[0]['Segment'] if not ranked_df.empty else 'N/A'}'.",
        }
