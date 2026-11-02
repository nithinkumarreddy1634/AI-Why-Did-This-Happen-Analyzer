"""Comprehensive test suite for Phase 5: Unexpected Outcome Detection Engine."""

import pytest
import pandas as pd
import numpy as np

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
from modules.unexpected_detection.detection_engine import UnexpectedDetectionEngine
from utils.detection_helpers import (
    safe_pct_change,
    safe_difference,
    detect_column_types,
    split_baseline_comparison,
)
from utils.sample_data import generate_churn_investigation_dataset


# ==============================================================================
# Helper & Safety Functions Tests
# ==============================================================================
def test_safe_pct_change_zero_and_valid():
    """Verify safe percentage change calculation with zero baseline and valid numbers."""
    assert safe_pct_change(100.0, 150.0) == 50.0
    assert safe_pct_change(100.0, 50.0) == -50.0
    assert safe_pct_change(0.0, 100.0) is None  # Zero baseline prevention
    assert safe_pct_change(None, 50.0) is None
    assert safe_pct_change(50.0, np.nan) is None


def test_safe_difference():
    """Verify safe difference calculation."""
    assert safe_difference(10.0, 25.0) == 15.0
    assert safe_difference(None, 5.0) is None


def test_column_type_detection():
    """Verify heuristic classification of columns."""
    df = pd.DataFrame({
        "num": [1.2, 3.4, 5.6, 7.8],
        "cat": ["A", "B", "A", "C"],
        "const": [1, 1, 1, 1],
        "bool_col": [True, False, True, False],
        "date_col": pd.date_range("2026-01-01", periods=4),
    })
    col_types = detect_column_types(df)
    assert col_types["num"] == "numerical"
    assert col_types["cat"] == "categorical"
    assert col_types["const"] == "constant"
    assert col_types["bool_col"] == "boolean"
    assert col_types["date_col"] == "datetime"


# ==============================================================================
# Severity Classifier Tests
# ==============================================================================
def test_severity_classification_pct():
    """Verify relative change severity thresholds."""
    classifier = SeverityClassifier()
    
    sev, msg = classifier.classify_percentage_change(5.0)
    assert sev == SeverityLevel.NORMAL
    
    sev, msg = classifier.classify_percentage_change(15.0)
    assert sev == SeverityLevel.LOW

    sev, msg = classifier.classify_percentage_change(30.0)
    assert sev == SeverityLevel.MODERATE

    sev, msg = classifier.classify_percentage_change(60.0)
    assert sev == SeverityLevel.HIGH

    sev, msg = classifier.classify_percentage_change(120.0)
    assert sev == SeverityLevel.CRITICAL


def test_severity_classification_zscore():
    """Verify z-score deviation classification."""
    classifier = SeverityClassifier()
    assert classifier.classify_z_score(0.8)[0] == SeverityLevel.NORMAL
    assert classifier.classify_z_score(1.5)[0] == SeverityLevel.LOW
    assert classifier.classify_z_score(2.5)[0] == SeverityLevel.MODERATE
    assert classifier.classify_z_score(3.5)[0] == SeverityLevel.HIGH
    assert classifier.classify_z_score(4.5)[0] == SeverityLevel.CRITICAL


def test_severity_classification_pp():
    """Verify percentage point shift classification."""
    classifier = SeverityClassifier()
    assert classifier.classify_percentage_point_shift(3.0)[0] == SeverityLevel.NORMAL
    assert classifier.classify_percentage_point_shift(8.0)[0] == SeverityLevel.LOW
    assert classifier.classify_percentage_point_shift(15.0)[0] == SeverityLevel.MODERATE
    assert classifier.classify_percentage_point_shift(30.0)[0] == SeverityLevel.HIGH
    assert classifier.classify_percentage_point_shift(45.0)[0] == SeverityLevel.CRITICAL


# ==============================================================================
# Numerical Anomaly Detection Tests
# ==============================================================================
def test_numerical_anomalies_zscore_and_iqr():
    """Test Z-score, Modified Z, and IQR detection on normal data with outliers."""
    detector = NumericalAnomalyDetector()
    
    # 100 normal points + 3 extreme outliers
    rng = np.random.default_rng(42)
    data = list(rng.normal(50, 5, 100)) + [150.0, 160.0, -50.0]
    series = pd.Series(data)

    # Z-Score
    res_z = detector.detect_zscore(series, threshold=3.0)
    assert res_z["status"] == "success"
    assert res_z["unusual_count"] >= 2
    assert res_z["total_observations"] == 103

    # Modified Z-Score
    res_mod = detector.detect_modified_zscore(series, threshold=3.5)
    assert res_mod["status"] == "success"
    assert res_mod["unusual_count"] >= 3

    # IQR
    res_iqr = detector.detect_iqr(series, multiplier=1.5)
    assert res_iqr["status"] == "success"
    assert res_iqr["unusual_count"] >= 3


def test_numerical_anomalies_edge_cases():
    """Test empty series, single value, constant series, and missing values."""
    detector = NumericalAnomalyDetector()

    # Empty
    empty_s = pd.Series([], dtype=float)
    assert detector.detect_zscore(empty_s)["status"] == "warning"

    # Constant
    const_s = pd.Series([42.0] * 50)
    assert detector.detect_zscore(const_s)["status"] == "warning"

    # Missing values
    nan_s = pd.Series([10.0, 12.0, np.nan, 11.0, 100.0, np.nan])
    res = detector.detect_zscore(nan_s, threshold=1.5)
    assert res["status"] == "success"
    assert res["missing_observations"] == 2
    assert res["valid_observations"] == 4


# ==============================================================================
# Distribution Shifts Tests
# ==============================================================================
def test_distribution_comparison():
    """Verify distribution shift between baseline and comparison groups."""
    detector = DistributionChangeDetector()
    
    b_series = pd.Series([50, 52, 48, 51, 49, 53, 47, 50])
    c_series = pd.Series([85, 90, 88, 86, 92, 89, 87, 91])  # Massive upward shift

    res = detector.compare_distributions(b_series, c_series, "charges")
    assert res["status"] == "success"
    assert res["mean_difference"] > 30.0
    assert res["mean_pct_change"] > 50.0
    assert res["severity"] in (SeverityLevel.HIGH.value, SeverityLevel.CRITICAL.value)
    assert len(res["metrics_table"]) == 6


def test_distribution_comparison_small_sample():
    """Verify behavior with insufficient samples."""
    detector = DistributionChangeDetector()
    res = detector.compare_distributions(pd.Series([1]), pd.Series([2, 3]), "metric")
    assert res["status"] == "warning"


# ==============================================================================
# Categorical Changes Tests
# ==============================================================================
def test_categorical_comparison():
    """Verify categorical proportions, emerging and disappearing classes."""
    detector = CategoricalChangeDetector()

    # Baseline: 80% Retained, 20% Churned, Service "DSL" present
    b_data = pd.Series(["Retained"] * 80 + ["Churned"] * 20 + ["DSL"] * 5)
    # Comparison: 65% Retained, 35% Churned, "DSL" gone, "Fiber" newly appeared
    c_data = pd.Series(["Retained"] * 65 + ["Churned"] * 35 + ["Fiber"] * 10)

    res = detector.compare_categories(b_data, c_data, "status")
    assert res["status"] == "success"
    assert "Fiber" in res["emerging_categories"]
    assert "DSL" in res["disappearing_categories"]
    assert res["max_percentage_point_shift"] > 10.0


# ==============================================================================
# Time-Based Changes Tests
# ==============================================================================
def test_time_based_change_detection():
    """Verify temporal spike/drop detection across periods."""
    detector = TimeBasedChangeDetector()

    dates = pd.date_range("2026-01-01", periods=12, freq="ME")
    # Values: steady at 100 for 8 months, then sudden jump to 250
    values = [100.0] * 8 + [250.0, 260.0, 270.0, 255.0]
    df = pd.DataFrame({"timestamp": dates, "monthly_charges": values})

    res = detector.analyze_temporal_trend(df, "timestamp", "monthly_charges", freq="ME", anomaly_threshold_pct=30.0)
    assert res["status"] == "success"
    assert res["unusual_periods_count"] >= 1
    spikes = res["spikes"]
    assert not spikes.empty
    assert bool(spikes["is_spike"].any())


def test_time_based_invalid_dates():
    """Verify safe handling of invalid datetime series."""
    detector = TimeBasedChangeDetector()
    df = pd.DataFrame({"not_a_date": ["foo", "bar"], "value": [10, 20]})
    res = detector.analyze_temporal_trend(df, "not_a_date", "value")
    assert res["status"] == "warning"


# ==============================================================================
# Segment-Level Change Tests
# ==============================================================================
def test_segment_analysis():
    """Verify segment disaggregation and ranking."""
    detector = SegmentChangeDetector()

    # Baseline: Equal regions
    b_df = pd.DataFrame({
        "region": ["North", "South", "West", "East"],
        "charges": [50.0, 52.0, 55.0, 51.0],
    })
    # Comparison: West has huge surge to 120
    c_df = pd.DataFrame({
        "region": ["North", "South", "West", "East"],
        "charges": [51.0, 53.0, 120.0, 52.0],
    })

    res = detector.analyze_segment_shifts(b_df, c_df, "region", "charges")
    assert res["status"] == "success"
    ranked = res["ranked_segments"]
    assert ranked.iloc[0]["Segment"] == "West"
    assert ranked.iloc[0]["Absolute Difference"] == 65.0


# ==============================================================================
# Automated Insights Tests
# ==============================================================================
def test_automated_insight_generator():
    """Verify non-causal factual insights generation."""
    gen = UnexpectedInsightGenerator()

    dummy_num_res = {
        "status": "success",
        "unusual_count": 8,
        "unusual_pct": 6.5,
        "total_observations": 120,
        "valid_observations": 120,
        "missing_observations": 0,
        "method": "Z-Score",
        "threshold": 3.0,
    }
    insights = gen.generate_numerical_anomaly_insights(dummy_num_res, "usage")
    assert len(insights) >= 1
    assert "Unusual Numerical Values Detected" in insights[0]["title"]
    assert "8" in insights[0]["evidence"]


# ==============================================================================
# Master Engine & Benchmark Integration Test
# ==============================================================================
def test_churn_investigation_benchmark():
    """Verify end-to-end detection on the synthetic churn surge dataset."""
    df = generate_churn_investigation_dataset(n_samples=600, random_state=42)
    assert len(df) == 600

    engine = UnexpectedDetectionEngine()
    candidates = engine.inspect_dataset_candidates(df)

    assert "monthly_charges" in candidates["numerical"]
    assert "region" in candidates["categorical"]
    assert "timestamp" in candidates["datetime"]

    # Cohort split test
    b_df, c_df, msg = split_baseline_comparison(df, "category", "cohort", "Baseline (H1)", "Recent Surge (H2)")
    assert len(b_df) == 300
    assert len(c_df) == 300

    # Categorical churn rate test: churn should jump significantly in recent surge
    cat_res = engine.compare_categories(b_df["churn"], c_df["churn"], "churn")
    assert cat_res["status"] == "success"
    assert cat_res["max_percentage_point_shift"] > 10.0  # Churn surged >10 pp


# ==============================================================================
# Mandated Testing Scenarios (Prompt Section 10)
# ==============================================================================

class TestMandatedScenariosPhase5:
    """Explicit tests for all 12 mandated testing scenarios specified in Section 10."""

    def test_scenario_1_normal_numerical_distribution(self):
        """Scenario 1: Normal numerical distributions without extreme outliers."""
        np.random.seed(42)
        normal_series = pd.Series(np.random.normal(100.0, 10.0, 500))
        detector = NumericalAnomalyDetector()

        # At 4.5 sigma, 0 outliers should be detected in 500 normal points
        res_z = detector.detect_zscore(normal_series, threshold=4.5)
        assert res_z["status"] == "success"
        assert res_z["unusual_count"] == 0
        assert res_z["unusual_pct"] == 0.0

        # Standard deviation threshold
        res_std = detector.detect_std_threshold(normal_series, num_std=4.5)
        assert res_std["status"] == "success"
        assert res_std["unusual_count"] == 0

    def test_scenario_2_numerical_outliers(self):
        """Scenario 2: Numerical outliers detected accurately across methods."""
        np.random.seed(42)
        data = list(np.random.normal(50.0, 5.0, 200)) + [250.0, -100.0]
        series = pd.Series(data)
        detector = NumericalAnomalyDetector()

        res_z = detector.detect_zscore(series, threshold=3.0)
        assert res_z["unusual_count"] >= 2

        res_std = detector.detect_std_threshold(series, num_std=3.0)
        assert res_std["unusual_count"] >= 2

        res_mod = detector.detect_modified_zscore(series, threshold=3.5)
        assert res_mod["unusual_count"] >= 2

        res_iqr = detector.detect_iqr(series, multiplier=1.5)
        assert res_iqr["unusual_count"] >= 2

        res_pct = detector.detect_pct_deviation(series, pct_threshold=100.0, reference_stat="median")
        assert res_pct["unusual_count"] >= 2

    def test_scenario_3_sudden_numerical_increases_and_decreases(self):
        """Scenario 3: Sudden numerical increases (spikes) and decreases (drops)."""
        detector = TimeBasedChangeDetector()
        dates = pd.date_range("2025-01-01", periods=10, freq="ME")
        # Steady at 100, then surges to 300 (+200%), then drops to 50 (-83%)
        vals = [100.0] * 5 + [300.0, 310.0, 50.0, 52.0, 50.0]
        df = pd.DataFrame({"timestamp": dates, "metric": vals})

        res = detector.analyze_temporal_trend(df, "timestamp", "metric", freq="ME", anomaly_threshold_pct=30.0)
        assert res["status"] == "success"
        assert res["unusual_periods_count"] >= 2

        spikes_df = res["spikes"]
        assert any(spikes_df["is_spike"])
        assert any(spikes_df["is_drop"])

    def test_scenario_4_categorical_distribution_changes(self):
        """Scenario 4: Categorical distribution changes, emerging and disappearing classes."""
        detector = CategoricalChangeDetector()
        b_series = pd.Series(["Standard"] * 80 + ["Premium"] * 20 + ["Legacy"] * 10)
        c_series = pd.Series(["Standard"] * 40 + ["Premium"] * 50 + ["Enterprise"] * 15)

        res = detector.compare_categories(b_series, c_series, "tier", top_n=10)
        assert res["status"] == "success"
        assert "Enterprise" in res["emerging_categories"]
        assert "Legacy" in res["disappearing_categories"]
        assert res["max_percentage_point_shift"] >= 20.0

    def test_scenario_5_time_based_changes(self):
        """Scenario 5: Time-based changes across frequencies."""
        detector = TimeBasedChangeDetector()
        dates = pd.date_range("2025-01-01", periods=60, freq="D")
        np.random.seed(42)
        df = pd.DataFrame({"date": dates, "revenue": np.random.uniform(100, 200, 60)})

        res_w = detector.analyze_temporal_trend(df, "date", "revenue", freq="W")
        assert res_w["status"] == "success"
        assert res_w["periods_count"] >= 8

        res_m = detector.analyze_temporal_trend(df, "date", "revenue", freq="ME")
        assert res_m["status"] == "success"
        assert res_m["periods_count"] >= 2

    def test_scenario_6_segment_level_differences(self):
        """Scenario 6: Segment-level differences and ranking."""
        detector = SegmentChangeDetector()
        b_df = pd.DataFrame({
            "segment": ["Enterprise", "SMB", "Consumer"],
            "csat": [4.5, 4.2, 4.0],
        })
        c_df = pd.DataFrame({
            "segment": ["Enterprise", "SMB", "Consumer"],
            "csat": [4.6, 4.1, 2.0],  # Consumer CSAT crashed by 2.0
        })

        res = detector.analyze_segment_shifts(b_df, c_df, "segment", "csat")
        assert res["status"] == "success"
        ranked = res["ranked_segments"]
        assert ranked.iloc[0]["Segment"] == "Consumer"
        assert ranked.iloc[0]["Absolute Difference"] == pytest.approx(-2.0)

    def test_scenario_7_missing_values(self):
        """Scenario 7: Dataset with missing values."""
        series_with_nans = pd.Series([10.0, np.nan, 20.0, np.nan, 30.0, 40.0, 50.0])
        detector = NumericalAnomalyDetector()
        res = detector.detect_zscore(series_with_nans, threshold=2.0)
        assert res["status"] == "success"
        assert res["missing_observations"] == 2
        assert res["valid_observations"] == 5

    def test_scenario_8_constant_columns(self):
        """Scenario 8: Constant columns (zero variance)."""
        const_series = pd.Series([99.9] * 40)
        detector = NumericalAnomalyDetector()
        res_z = detector.detect_zscore(const_series)
        assert res_z["status"] == "warning"
        assert res_z["unusual_count"] == 0

        res_std = detector.detect_std_threshold(const_series)
        assert res_std["status"] == "warning"

        res_mod = detector.detect_modified_zscore(const_series)
        assert res_mod["status"] == "warning"

    def test_scenario_9_empty_datasets(self):
        """Scenario 9: Empty datasets."""
        empty_s = pd.Series([], dtype=float)
        detector = NumericalAnomalyDetector()
        res = detector.detect_zscore(empty_s)
        assert res["status"] == "warning"
        assert res["total_observations"] == 0

        dist_detector = DistributionChangeDetector()
        res_dist = dist_detector.compare_distributions(empty_s, empty_s, "col")
        assert res_dist["status"] == "warning"

    def test_scenario_10_small_sample_sizes(self):
        """Scenario 10: Small sample sizes (2-3 observations)."""
        detector = NumericalAnomalyDetector()
        small_s = pd.Series([1.0, 2.0])
        res = detector.detect_zscore(small_s)
        assert res["status"] == "warning"
        assert "Insufficient" in res["message"]

        res_iqr = detector.detect_iqr(pd.Series([1.0, 2.0, 3.0]))
        assert res_iqr["status"] == "warning"

    def test_scenario_11_zero_baseline_values(self):
        """Scenario 11: Zero baseline values prevent division by zero."""
        assert safe_pct_change(0.0, 50.0) is None
        assert safe_pct_change(0.0, 0.0) is None

        detector = NumericalAnomalyDetector()
        zero_series = pd.Series([0.0, 0.0, 0.0, 0.0])
        res = detector.detect_pct_deviation(zero_series, pct_threshold=50.0, reference_stat="median")
        assert res["status"] == "warning"
        assert "0" in res["message"]

    def test_scenario_12_invalid_datetime_columns(self):
        """Scenario 12: Invalid datetime columns."""
        detector = TimeBasedChangeDetector()
        bad_df = pd.DataFrame({"not_a_date": ["alpha", "beta", "gamma"], "val": [1, 2, 3]})
        res = detector.analyze_temporal_trend(bad_df, "not_a_date", "val")
        assert res["status"] == "warning"
        assert "no valid datetime" in res["message"].lower()

    def test_scenario_13_dataset_without_target_variable(self):
        """Scenario 13: Dataset without a target variable runs smoothly."""
        engine = UnexpectedDetectionEngine()
        df = pd.DataFrame({
            "feature1": np.random.normal(10, 2, 50),
            "feature2": np.random.choice(["X", "Y"], 50),
        })
        candidates = engine.inspect_dataset_candidates(df)
        assert len(candidates["numerical"]) == 1
        assert len(candidates["categorical"]) == 1

        res = engine.detect_numerical_anomalies(df["feature1"], method="Z-Score")
        assert res["status"] == "success"

