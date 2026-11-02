"""Unified Unexpected Outcome and Intelligent Change Identification Engine."""

from typing import Dict, Any, List, Optional
import pandas as pd

from utils.detection_helpers import detect_column_types
from modules.unexpected_detection.severity_classifier import (
    SeverityClassifier,
    SeverityThresholds,
    SeverityLevel,
)
from modules.unexpected_detection.numerical_anomalies import NumericalAnomalyDetector
from modules.unexpected_detection.distribution_changes import DistributionChangeDetector
from modules.unexpected_detection.categorical_changes import CategoricalChangeDetector
from modules.unexpected_detection.time_based_detection import TimeBasedChangeDetector
from modules.unexpected_detection.segment_analysis import SegmentChangeDetector
from modules.unexpected_detection.insight_generator import UnexpectedInsightGenerator


class UnexpectedDetectionEngine:
    """Master facade for detecting unexpected outcomes and structural shifts in data."""

    def __init__(self, thresholds: Optional[SeverityThresholds] = None) -> None:
        self.thresholds = thresholds or SeverityThresholds()
        self.classifier = SeverityClassifier(self.thresholds)
        self.numerical_detector = NumericalAnomalyDetector(self.classifier)
        self.distribution_detector = DistributionChangeDetector(self.classifier)
        self.categorical_detector = CategoricalChangeDetector(self.classifier)
        self.time_detector = TimeBasedChangeDetector(self.classifier)
        self.segment_detector = SegmentChangeDetector(self.classifier)
        self.insight_generator = UnexpectedInsightGenerator()

    def inspect_dataset_candidates(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """Categorize columns into candidate sets for different investigation modes."""
        col_types = detect_column_types(df)
        return {
            "numerical": [c for c, t in col_types.items() if t == "numerical"],
            "categorical": [c for c, t in col_types.items() if t == "categorical"],
            "datetime": [c for c, t in col_types.items() if t == "datetime"],
            "boolean": [c for c, t in col_types.items() if t == "boolean"],
            "constant": [c for c, t in col_types.items() if t == "constant"],
            "identifier": [c for c, t in col_types.items() if t == "identifier"],
        }

    def detect_numerical_anomalies(
        self,
        series: pd.Series,
        method: str = "Z-Score",
        threshold: float = 3.0,
    ) -> Dict[str, Any]:
        """Run numerical outlier and anomaly detection."""
        if method == "Z-Score":
            return self.numerical_detector.detect_zscore(series, threshold=threshold)
        elif method == "Modified Z-Score (MAD)":
            return self.numerical_detector.detect_modified_zscore(series, threshold=threshold)
        elif method == "IQR (Interquartile Range)":
            return self.numerical_detector.detect_iqr(series, multiplier=threshold)
        elif method == "Standard Deviation Threshold":
            return self.numerical_detector.detect_std_threshold(series, num_std=threshold)
        elif method.startswith("% Deviation"):
            ref_stat = "median" if "Median" in method else "mean"
            return self.numerical_detector.detect_pct_deviation(
                series, pct_threshold=threshold, reference_stat=ref_stat
            )
        else:
            return self.numerical_detector.detect_zscore(series, threshold=threshold)

    def compare_distributions(
        self,
        b_series: pd.Series,
        c_series: pd.Series,
        col_name: str,
        b_label: str = "Baseline",
        c_label: str = "Comparison",
    ) -> Dict[str, Any]:
        """Run statistical distribution comparison across two cohorts."""
        return self.distribution_detector.compare_distributions(
            b_series, c_series, col_name, b_label, c_label
        )

    def compare_categories(
        self,
        b_series: pd.Series,
        c_series: pd.Series,
        col_name: str,
        b_label: str = "Baseline",
        c_label: str = "Comparison",
        top_n: int = 15,
    ) -> Dict[str, Any]:
        """Run categorical frequency and proportion change detection."""
        return self.categorical_detector.compare_categories(
            b_series, c_series, col_name, b_label, c_label, top_n
        )

    def analyze_temporal_change(
        self,
        df: pd.DataFrame,
        date_col: str,
        metric_col: str,
        freq: str = "M",
        agg_func: str = "mean",
        anomaly_threshold_pct: float = 30.0,
    ) -> Dict[str, Any]:
        """Run time-series spike, drop, and transition detection."""
        return self.time_detector.analyze_temporal_trend(
            df, date_col, metric_col, freq, agg_func, anomaly_threshold_pct
        )

    def analyze_segments(
        self,
        b_df: pd.DataFrame,
        c_df: pd.DataFrame,
        segment_col: str,
        metric_col: str,
        b_label: str = "Baseline",
        c_label: str = "Comparison",
    ) -> Dict[str, Any]:
        """Run segment-level change analysis."""
        return self.segment_detector.analyze_segment_shifts(
            b_df, c_df, segment_col, metric_col, b_label, c_label
        )
