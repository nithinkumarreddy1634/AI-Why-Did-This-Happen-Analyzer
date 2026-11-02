"""Statistical distribution shift and comparison detector."""

from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np

from utils.detection_helpers import safe_pct_change, safe_difference
from modules.unexpected_detection.severity_classifier import (
    SeverityLevel,
    SeverityClassifier,
)


class DistributionChangeDetector:
    """Detects shifts in continuous variable distributions across two cohorts."""

    def __init__(self, classifier: Optional[SeverityClassifier] = None) -> None:
        self.classifier = classifier or SeverityClassifier()

    def compare_distributions(
        self,
        baseline_series: pd.Series,
        comparison_series: pd.Series,
        column_name: str,
        baseline_label: str = "Baseline",
        comparison_label: str = "Comparison",
    ) -> Dict[str, Any]:
        """Compute comprehensive statistical distribution shift between baseline and comparison."""
        b_clean = pd.to_numeric(baseline_series, errors="coerce").dropna()
        c_clean = pd.to_numeric(comparison_series, errors="coerce").dropna()

        n_b = len(b_clean)
        n_c = len(c_clean)

        if n_b < 2 or n_c < 2:
            return {
                "column": column_name,
                "status": "warning",
                "message": f"Insufficient non-null observations (Baseline: {n_b}, Comparison: {n_c}). Minimum 2 required in both.",
                "summary": {},
            }

        # Calculate descriptive metrics for both cohorts
        b_mean = float(b_clean.mean())
        c_mean = float(c_clean.mean())
        mean_diff = safe_difference(b_mean, c_mean)
        mean_pct_change = safe_pct_change(b_mean, c_mean)

        b_median = float(b_clean.median())
        c_median = float(c_clean.median())
        median_diff = safe_difference(b_median, c_median)
        median_pct_change = safe_pct_change(b_median, c_median)

        b_std = float(b_clean.std(ddof=1)) if n_b > 1 else 0.0
        c_std = float(c_clean.std(ddof=1)) if n_c > 1 else 0.0
        std_diff = safe_difference(b_std, c_std)
        std_pct_change = safe_pct_change(b_std, c_std)

        # Quartiles
        b_q1, b_q3 = float(b_clean.quantile(0.25)), float(b_clean.quantile(0.75))
        c_q1, c_q3 = float(c_clean.quantile(0.25)), float(c_clean.quantile(0.75))
        b_iqr = b_q3 - b_q1
        c_iqr = c_q3 - c_q1

        # Calculate standard deviation shift (Cohen's d like effect size)
        pooled_std = np.sqrt(((n_b - 1) * b_std**2 + (n_c - 1) * c_std**2) / max(n_b + n_c - 2, 1))
        std_shift = (c_mean - b_mean) / pooled_std if pooled_std > 0 else 0.0

        # Determine overall distribution shift severity
        if mean_pct_change is not None:
            severity, severity_reason = self.classifier.classify_percentage_change(mean_pct_change)
        else:
            severity, severity_reason = self.classifier.classify_z_score(std_shift)

        metrics_table = [
            {
                "Metric": "Mean (Average)",
                f"{baseline_label}": round(b_mean, 3),
                f"{comparison_label}": round(c_mean, 3),
                "Absolute Difference": round(mean_diff, 3) if mean_diff is not None else None,
                "% Change": f"{mean_pct_change:+.2f}%" if mean_pct_change is not None else "N/A (zero baseline)",
            },
            {
                "Metric": "Median",
                f"{baseline_label}": round(b_median, 3),
                f"{comparison_label}": round(c_median, 3),
                "Absolute Difference": round(median_diff, 3) if median_diff is not None else None,
                "% Change": f"{median_pct_change:+.2f}%" if median_pct_change is not None else "N/A",
            },
            {
                "Metric": "Std. Deviation",
                f"{baseline_label}": round(b_std, 3),
                f"{comparison_label}": round(c_std, 3),
                "Absolute Difference": round(std_diff, 3) if std_diff is not None else None,
                "% Change": f"{std_pct_change:+.2f}%" if std_pct_change is not None else "N/A",
            },
            {
                "Metric": "Q1 (25th percentile)",
                f"{baseline_label}": round(b_q1, 3),
                f"{comparison_label}": round(c_q1, 3),
                "Absolute Difference": round(c_q1 - b_q1, 3),
                "% Change": f"{safe_pct_change(b_q1, c_q1):+.2f}%" if safe_pct_change(b_q1, c_q1) is not None else "N/A",
            },
            {
                "Metric": "Q3 (75th percentile)",
                f"{baseline_label}": round(b_q3, 3),
                f"{comparison_label}": round(c_q3, 3),
                "Absolute Difference": round(c_q3 - b_q3, 3),
                "% Change": f"{safe_pct_change(b_q3, c_q3):+.2f}%" if safe_pct_change(b_q3, c_q3) is not None else "N/A",
            },
            {
                "Metric": "IQR",
                f"{baseline_label}": round(b_iqr, 3),
                f"{comparison_label}": round(c_iqr, 3),
                "Absolute Difference": round(c_iqr - b_iqr, 3),
                "% Change": f"{safe_pct_change(b_iqr, c_iqr):+.2f}%" if safe_pct_change(b_iqr, c_iqr) is not None else "N/A",
            },
        ]

        return {
            "column": column_name,
            "status": "success",
            "baseline_label": baseline_label,
            "comparison_label": comparison_label,
            "baseline_count": n_b,
            "comparison_count": n_c,
            "mean_pct_change": mean_pct_change,
            "mean_difference": mean_diff,
            "std_shift": std_shift,
            "severity": severity.value,
            "severity_reason": severity_reason,
            "metrics_table": pd.DataFrame(metrics_table),
            "baseline_data": b_clean,
            "comparison_data": c_clean,
            "message": f"Distribution shift for '{column_name}': mean changed by {mean_diff:+.2f} ({mean_pct_change:+.1f}% if mean_pct_change else 'N/A').",
        }
