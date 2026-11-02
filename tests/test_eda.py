"""Unit and integration tests for Phase 4: Exploratory Data Analysis (EDA) Engine.

Covers all 10 mandated scenarios:
1. Numerical-only dataset.
2. Categorical-only dataset.
3. Mixed numerical & categorical dataset.
4. Dataset containing missing values.
5. Dataset containing constant columns (zero variance).
6. High-cardinality categorical columns.
7. Datetime datasets.
8. Datasets without target variables.
9. Small datasets (3-4 rows).
10. Datasets containing duplicate records.

Also validates univariate, bivariate, correlation, target, multivariate, time-series,
segmentation, automated insight generation, and Plotly visualization builders.
"""

import numpy as np
import pandas as pd
import pytest
import plotly.graph_objects as go

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
from modules.eda.target_analysis import analyze_target_variable
from modules.eda.multivariate import prepare_multivariate_matrix
from modules.eda.time_series import (
    detect_time_series_candidates,
    compute_time_series_aggregation,
)
from modules.eda.segment_analysis import compute_segment_breakdown
from modules.eda.insight_generator import generate_eda_insights
from modules.eda import EDAManager
from utils.eda_helpers import (
    get_eda_numerical_columns,
    get_eda_categorical_columns,
    get_eda_datetime_columns,
    safe_skewness,
    safe_kurtosis,
)
from components.eda_visualizations import (
    plot_numerical_distribution,
    plot_categorical_distribution,
    plot_scatter_with_trend,
    plot_categorical_vs_numerical,
    plot_cross_tabulation,
    plot_correlation_heatmap,
    plot_time_series,
    plot_scatter_matrix,
)


# ==============================================================================
# Fixtures
# ==============================================================================

@pytest.fixture
def numerical_only_df() -> pd.DataFrame:
    """Fixture for Scenario 1: Numerical-only dataset."""
    np.random.seed(42)
    return pd.DataFrame({
        "feature_a": np.random.normal(50, 10, 100),
        "feature_b": np.random.uniform(0, 100, 100),
        "feature_c": np.random.exponential(5, 100),
    })


@pytest.fixture
def categorical_only_df() -> pd.DataFrame:
    """Fixture for Scenario 2: Categorical-only dataset."""
    return pd.DataFrame({
        "status": ["active", "pending", "closed", "active", "active"] * 20,
        "region": ["North", "South", "East", "West", "Central"] * 20,
        "tier": ["Gold", "Silver", "Bronze", "Silver", "Gold"] * 20,
    })


@pytest.fixture
def mixed_df() -> pd.DataFrame:
    """Fixture for Scenario 3: Mixed dataset."""
    np.random.seed(42)
    n = 120
    dept = np.random.choice(["Engineering", "Sales", "Support", "Marketing"], n)
    exp = np.random.uniform(1, 15, n)
    salary = exp * 8000 + np.random.normal(30000, 5000, n)
    return pd.DataFrame({
        "Department": dept,
        "Experience": exp,
        "Salary": salary,
        "Education": np.random.choice(["Bachelor", "Master", "PhD"], n),
        "PerformanceScore": np.random.randint(1, 6, n),
    })


@pytest.fixture
def missing_values_df() -> pd.DataFrame:
    """Fixture for Scenario 4: Dataset with missing values."""
    return pd.DataFrame({
        "num_with_nulls": [10.0, np.nan, 30.0, 40.0, np.nan, 60.0, 70.0, 80.0],
        "cat_with_nulls": ["A", "B", None, "A", "B", "C", None, "A"],
        "all_nulls": [np.nan] * 8,
    })


@pytest.fixture
def constant_columns_df() -> pd.DataFrame:
    """Fixture for Scenario 5: Dataset with constant (zero variance) columns."""
    return pd.DataFrame({
        "constant_num": [42.0] * 50,
        "constant_cat": ["Fixed"] * 50,
        "varying_num": np.linspace(1, 50, 50),
        "varying_cat": ["A", "B"] * 25,
    })


@pytest.fixture
def high_cardinality_df() -> pd.DataFrame:
    """Fixture for Scenario 6: High-cardinality categorical dataset."""
    n = 150
    return pd.DataFrame({
        "user_id": [f"USR_{i:04d}" for i in range(n)],
        "postal_code": [f"ZIP_{i % 50}" for i in range(n)],
        "metric": np.random.randn(n),
    })


@pytest.fixture
def datetime_df() -> pd.DataFrame:
    """Fixture for Scenario 7: Datetime dataset."""
    dates = pd.date_range("2025-01-01", periods=100, freq="D")
    return pd.DataFrame({
        "timestamp": dates,
        "date_str": dates.strftime("%Y-%m-%d"),
        "sales": np.random.uniform(500, 2000, 100),
        "orders": np.random.randint(5, 50, 100),
    })


@pytest.fixture
def small_df() -> pd.DataFrame:
    """Fixture for Scenario 9: Small dataset (3-4 rows)."""
    return pd.DataFrame({
        "x": [1.0, 2.0, 3.0],
        "y": [10.0, 20.0, 35.0],
        "group": ["A", "B", "A"],
    })


@pytest.fixture
def duplicate_df() -> pd.DataFrame:
    """Fixture for Scenario 10: Dataset with duplicate records."""
    base = pd.DataFrame({
        "val_a": [10, 20, 30],
        "val_b": ["alpha", "beta", "gamma"],
    })
    return pd.concat([base, base, base], ignore_index=True)


# ==============================================================================
# 1. Univariate Analysis Tests
# ==============================================================================

class TestUnivariateAnalysis:
    """Tests for numerical and categorical univariate analysis modules."""

    def test_numerical_stats_calculation(self, mixed_df):
        """Test statistical metrics calculated for a numerical column."""
        stats = compute_numerical_univariate_stats(mixed_df, "Salary")
        assert stats["column"] == "Salary"
        assert stats["count"] == len(mixed_df)
        assert stats["missing_count"] == 0
        assert stats["missing_pct"] == 0.0
        assert stats["mean"] > 0
        assert stats["std"] > 0
        assert stats["variance"] > 0
        assert stats["min"] <= stats["median"] <= stats["max"]
        assert stats["q1"] <= stats["q2"] <= stats["q3"]
        assert stats["iqr"] == pytest.approx(stats["q3"] - stats["q1"])
        assert "skewness" in stats
        assert "kurtosis" in stats

    def test_all_numerical_summaries(self, mixed_df):
        """Test batch summary of all numerical columns."""
        summary_df = compute_all_numerical_summaries(mixed_df)
        assert isinstance(summary_df, pd.DataFrame)
        assert "Salary" in summary_df["Column"].values
        assert "Experience" in summary_df["Column"].values
        assert "Mean" in summary_df.columns
        assert "IQR" in summary_df.columns

    def test_categorical_stats_calculation(self, mixed_df):
        """Test categorical frequency and percentage calculations."""
        stats = compute_categorical_univariate_stats(mixed_df, "Department")
        assert stats["column"] == "Department"
        assert stats["total_records"] == len(mixed_df)
        assert stats["unique_categories"] == 4
        assert stats["most_frequent_category"] is not None
        assert stats["least_frequent_category"] is not None
        freq_df = stats["frequencies_df"]
        assert isinstance(freq_df, pd.DataFrame)
        assert set(freq_df.columns) >= {"Category", "Count", "Percentage"}
        assert freq_df["Percentage"].sum() == pytest.approx(100.0, abs=1e-1)

    def test_categorical_top_n_group_other(self, high_cardinality_df):
        """Test top N filtering and 'Other' category grouping."""
        stats = compute_categorical_univariate_stats(
            high_cardinality_df, "postal_code", top_n=5, group_other=True
        )
        freq_df = stats["frequencies_df"]
        assert len(freq_df) <= 6  # 5 top categories + 1 Other
        assert "Other" in freq_df["Category"].values

    def test_empty_and_null_column_handling(self, missing_values_df):
        """Test safe handling of all-null columns."""
        stats = compute_numerical_univariate_stats(missing_values_df, "all_nulls")
        assert stats["count"] == 0
        assert stats["missing_count"] == 8
        assert stats["mean"] is None
        assert stats["std"] is None


# ==============================================================================
# 2. Bivariate Analysis Tests
# ==============================================================================

class TestBivariateAnalysis:
    """Tests for bivariate relationship analysis."""

    def test_num_vs_num_relationship(self, mixed_df):
        """Test Pearson, Spearman, and linear regression stats between two numericals."""
        res = analyze_num_vs_num(mixed_df, "Experience", "Salary")
        assert res["sample_size"] == len(mixed_df)
        assert res["pearson_corr"] is not None
        assert res["spearman_corr"] is not None
        # Salary is generated linearly from Experience with positive slope
        assert res["pearson_corr"] > 0.7
        assert res["slope"] > 0
        assert res["r_squared"] > 0.5
        assert "Strong" in res["strength"]

    def test_cat_vs_num_relationship(self, mixed_df):
        """Test group statistics and ANOVA between categorical and numerical."""
        res = analyze_cat_vs_num(mixed_df, "Department", "Salary")
        group_df = res["group_summary_df"]
        assert isinstance(group_df, pd.DataFrame)
        assert len(group_df) == 4
        assert "Mean" in group_df.columns
        assert "Std" in group_df.columns
        assert "anova_f" in res
        assert "anova_p" in res

    def test_cat_vs_cat_relationship(self, mixed_df):
        """Test cross-tabulation and Chi-Square test between two categoricals."""
        res = analyze_cat_vs_cat(mixed_df, "Department", "Education")
        assert isinstance(res["crosstab_counts"], pd.DataFrame)
        assert "chi2_stat" in res
        assert "chi2_p" in res


# ==============================================================================
# 3. Correlation Analysis Tests
# ==============================================================================

class TestCorrelationAnalysis:
    """Tests for correlation matrix computation and top pair extraction."""

    def test_correlation_matrix_pearson_spearman(self, mixed_df):
        """Test Pearson and Spearman correlation computation."""
        corr_p, meta_p = compute_correlation_matrix(mixed_df, method="pearson")
        assert isinstance(corr_p, pd.DataFrame)
        assert "Salary" in corr_p.columns
        assert corr_p.loc["Salary", "Salary"] == pytest.approx(1.0)
        assert len(meta_p["excluded_constant_columns"]) == 0

        corr_s, _ = compute_correlation_matrix(mixed_df, method="spearman")
        assert corr_s.shape == corr_p.shape

    def test_constant_column_exclusion_in_correlation(self, constant_columns_df):
        """Verify that zero-variance columns are excluded and flagged."""
        corr, meta = compute_correlation_matrix(constant_columns_df)
        assert "constant_num" in meta["excluded_constant_columns"]

    def test_extract_top_correlations(self, mixed_df):
        """Test extraction of top absolute correlation pairs without self-correlation."""
        corr, _ = compute_correlation_matrix(mixed_df)
        top_df = extract_top_correlations(corr, top_n=5)
        assert isinstance(top_df, pd.DataFrame)
        assert len(top_df) > 0
        for _, row in top_df.iterrows():
            assert row["Feature 1"] != row["Feature 2"]
            assert -1.0 <= row["Correlation"] <= 1.0


# ==============================================================================
# 4. Target Variable Analysis Tests
# ==============================================================================

class TestTargetVariableAnalysis:
    """Tests for target variable distribution and predictor relationships."""

    def test_numerical_target_analysis(self, mixed_df):
        """Test analysis when target variable is continuous numerical."""
        res = analyze_target_variable(mixed_df, "Salary")
        assert res["target_kind"] == "numerical"
        assert res["count"] == len(mixed_df)
        assert isinstance(res["correlations_df"], pd.DataFrame)
        assert "Experience" in res["correlations_df"]["Feature"].values

    def test_categorical_target_analysis(self, mixed_df):
        """Test analysis when target variable is categorical classification outcome."""
        res = analyze_target_variable(mixed_df, "Department")
        assert res["target_kind"] == "categorical"
        assert res["unique_classes"] == 4
        assert res["imbalance_ratio"] >= 1.0
        assert isinstance(res["class_distribution_df"], pd.DataFrame)

    def test_none_target_variable(self, mixed_df):
        """Verify safe behavior when target variable is not provided."""
        res = analyze_target_variable(mixed_df, None)
        assert res["has_target"] is False
        assert "message" in res


# ==============================================================================
# 5. Time-Series & Segment Analysis Tests
# ==============================================================================

class TestTimeSeriesAndSegmentation:
    """Tests for datetime detection, time-series aggregation, and segment breakdowns."""

    def test_detect_time_series_candidates(self, datetime_df):
        """Test automatic detection of datetime and date-string columns."""
        candidates = detect_time_series_candidates(datetime_df)
        assert "timestamp" in candidates
        assert "date_str" in candidates
        assert "sales" not in candidates

    def test_compute_time_series_aggregation(self, datetime_df):
        """Test time series aggregation with daily and rolling calculations."""
        res = compute_time_series_aggregation(
            datetime_df,
            date_col="timestamp",
            metric_col="sales",
            freq="W",
            agg_func="mean",
            rolling_window=4,
        )
        assert res["valid"] is True
        agg_df = res["ts_df"]
        assert isinstance(agg_df, pd.DataFrame)
        assert len(agg_df) > 0
        assert "sales" in agg_df.columns
        assert res["rolling_col"] in agg_df.columns

    def test_segment_breakdown(self, mixed_df):
        """Test multidimensional category segmentation with metrics and share %."""
        res = compute_segment_breakdown(
            mixed_df,
            segment_col="Department",
            metric_cols=["Salary"],
        )
        assert res["valid"] is True
        seg_df = res["segment_summary_df"]
        assert isinstance(seg_df, pd.DataFrame)
        assert "Department" in seg_df.columns
        assert "Salary (Mean)" in seg_df.columns
        assert "Segment Share %" in seg_df.columns
        assert seg_df["Segment Share %"].sum() == pytest.approx(100.0, abs=1e-1)


# ==============================================================================
# 6. Automated Insight Generation Tests
# ==============================================================================

class TestAutomatedInsights:
    """Tests dynamic, calculation-based insight synthesis without hardcoding."""

    def test_generate_insights_mixed_dataset(self, mixed_df):
        """Verify factual observations produced for mixed dataset."""
        insights = generate_eda_insights(mixed_df, target_col="Salary")
        assert isinstance(insights, list)
        assert len(insights) >= 2
        categories = [ins["category"] for ins in insights]
        assert "Overview" in categories or "Target" in categories or "Correlation" in categories
        # Check that insights do NOT claim causation
        for ins in insights:
            desc = ins["insight"].lower()
            assert "caused" not in desc or "not imply causation" in desc or "association" in desc

    def test_generate_insights_with_missing_values(self, missing_values_df):
        """Verify data quality/missingness insight generation."""
        insights = generate_eda_insights(missing_values_df)
        assert any(ins["category"] in ["Data Quality", "Overview"] for ins in insights)


# ==============================================================================
# 7. Mandated Scenarios (1 to 10)
# ==============================================================================

class TestMandatedScenarios:
    """Exhaustive tests for the 10 mandated scenarios specified in Master Prompt."""

    def test_scenario_1_numerical_only(self, numerical_only_df):
        """Scenario 1: Numerical-only dataset."""
        num_cols = get_eda_numerical_columns(numerical_only_df)
        cat_cols = get_eda_categorical_columns(numerical_only_df)
        assert len(num_cols) == 3
        assert len(cat_cols) == 0

        # Calculations must run without errors
        summaries = compute_all_numerical_summaries(numerical_only_df)
        assert len(summaries) == 3
        corr_matrix, _ = compute_correlation_matrix(numerical_only_df)
        assert corr_matrix.shape == (3, 3)

        # Categorical summary returns empty DataFrame gracefully
        cat_summaries = compute_all_categorical_summaries(numerical_only_df)
        assert len(cat_summaries) == 0

    def test_scenario_2_categorical_only(self, categorical_only_df):
        """Scenario 2: Categorical-only dataset."""
        num_cols = get_eda_numerical_columns(categorical_only_df)
        cat_cols = get_eda_categorical_columns(categorical_only_df)
        assert len(num_cols) == 0
        assert len(cat_cols) == 3

        cat_summaries = compute_all_categorical_summaries(categorical_only_df)
        assert len(cat_summaries) == 3
        num_summaries = compute_all_numerical_summaries(categorical_only_df)
        assert len(num_summaries) == 0

        corr_matrix, _ = compute_correlation_matrix(categorical_only_df)
        assert corr_matrix.empty

    def test_scenario_3_mixed_dataset(self, mixed_df):
        """Scenario 3: Mixed numerical and categorical dataset."""
        num_cols = get_eda_numerical_columns(mixed_df)
        cat_cols = get_eda_categorical_columns(mixed_df)
        assert len(num_cols) >= 2
        assert len(cat_cols) >= 2

        # Bivariate cross-types
        biv = analyze_cat_vs_num(mixed_df, cat_cols[0], num_cols[0])
        assert "group_summary_df" in biv

    def test_scenario_4_missing_values_dataset(self, missing_values_df):
        """Scenario 4: Dataset containing missing values."""
        stats = compute_numerical_univariate_stats(missing_values_df, "num_with_nulls")
        assert stats["missing_count"] == 2
        assert stats["count"] == 6

        cat_stats = compute_categorical_univariate_stats(missing_values_df, "cat_with_nulls")
        assert cat_stats["missing_count"] == 2

    def test_scenario_5_constant_columns(self, constant_columns_df):
        """Scenario 5: Dataset containing constant columns (zero variance)."""
        num_stats = compute_numerical_univariate_stats(constant_columns_df, "constant_num")
        assert num_stats["variance"] == 0.0
        assert num_stats["std"] == 0.0
        assert num_stats["skewness"] == 0.0
        assert num_stats["kurtosis"] == 0.0

        corr_matrix, meta = compute_correlation_matrix(constant_columns_df)
        assert "constant_num" in meta["excluded_constant_columns"]
        assert "constant_num" not in corr_matrix.columns

    def test_scenario_6_high_cardinality(self, high_cardinality_df):
        """Scenario 6: High-cardinality categorical columns."""
        stats = compute_categorical_univariate_stats(
            high_cardinality_df, "postal_code", top_n=10, group_other=True
        )
        assert stats["unique_categories"] == 50
        freq_df = stats["frequencies_df"]
        assert len(freq_df) == 11  # 10 categories + 1 Other

    def test_scenario_7_datetime_dataset(self, datetime_df):
        """Scenario 7: Datetime dataset."""
        candidates = detect_time_series_candidates(datetime_df)
        assert len(candidates) >= 1
        res = compute_time_series_aggregation(
            datetime_df, date_col=candidates[0], metric_col="sales", freq="ME"
        )
        assert res["valid"] is True
        agg_df = res["ts_df"]
        assert isinstance(agg_df, pd.DataFrame)
        assert not agg_df.empty

    def test_scenario_8_no_target_variable(self, mixed_df):
        """Scenario 8: Dataset without selected target variable."""
        insights = generate_eda_insights(mixed_df, target_col=None)
        assert isinstance(insights, list)
        target_res = analyze_target_variable(mixed_df, None)
        assert target_res["has_target"] is False

    def test_scenario_9_small_dataset(self, small_df):
        """Scenario 9: Small dataset (3-4 rows)."""
        num_stats = compute_numerical_univariate_stats(small_df, "x")
        assert num_stats["count"] == 3
        assert num_stats["mean"] == 2.0

        # Correlation between 3 points
        biv = analyze_num_vs_num(small_df, "x", "y")
        assert biv["sample_size"] == 3
        assert biv["pearson_corr"] is not None

    def test_scenario_10_duplicate_records(self, duplicate_df):
        """Scenario 10: Dataset containing duplicate records."""
        original_shape = duplicate_df.shape
        num_stats = compute_numerical_univariate_stats(duplicate_df, "val_a")
        assert num_stats["count"] == 9
        # Dataset must remain strictly unmodified (immutable)
        assert duplicate_df.shape == original_shape


# ==============================================================================
# 8. Plotly Visualizations Tests
# ==============================================================================

class TestPlotlyVisualizations:
    """Verify that all visualization generators construct valid Plotly Figures."""

    def test_plot_numerical_distribution(self, mixed_df):
        """Test histogram, box, and violin plot creation."""
        fig_hist = plot_numerical_distribution(mixed_df["Salary"], plot_type="histogram")
        assert isinstance(fig_hist, go.Figure)

        fig_box = plot_numerical_distribution(mixed_df["Salary"], plot_type="box")
        assert isinstance(fig_box, go.Figure)

        fig_violin = plot_numerical_distribution(mixed_df["Salary"], plot_type="violin")
        assert isinstance(fig_violin, go.Figure)

    def test_plot_categorical_distribution(self, mixed_df):
        """Test bar, horizontal bar, and pie charts."""
        stats = compute_categorical_univariate_stats(mixed_df, "Department")
        freq_df = stats["frequencies_df"]

        fig_bar = plot_categorical_distribution(freq_df, chart_type="bar")
        assert isinstance(fig_bar, go.Figure)

        fig_hbar = plot_categorical_distribution(freq_df, chart_type="horizontal_bar")
        assert isinstance(fig_hbar, go.Figure)

        fig_pie = plot_categorical_distribution(freq_df, chart_type="pie")
        assert isinstance(fig_pie, go.Figure)

    def test_plot_scatter_with_trend(self, mixed_df):
        """Test scatter plot with OLS trendline overlay."""
        fig_scatter = plot_scatter_with_trend(mixed_df, "Experience", "Salary", show_trendline=True)
        assert isinstance(fig_scatter, go.Figure)

    def test_plot_categorical_vs_numerical(self, mixed_df):
        """Test categorical comparison box and violin plots."""
        fig = plot_categorical_vs_numerical(mixed_df, "Department", "Salary", plot_type="box")
        assert isinstance(fig, go.Figure)

    def test_plot_cross_tabulation(self, mixed_df):
        """Test stacked bar and heatmap for cross-tabulation."""
        biv = analyze_cat_vs_cat(mixed_df, "Department", "Education")
        ct = biv["crosstab_counts"]

        fig_bar = plot_cross_tabulation(ct, "Department", "Education", plot_type="stacked_bar")
        assert isinstance(fig_bar, go.Figure)

        fig_heat = plot_cross_tabulation(ct, "Department", "Education", plot_type="heatmap")
        assert isinstance(fig_heat, go.Figure)

    def test_plot_correlation_heatmap(self, mixed_df):
        """Test correlation matrix heatmap."""
        corr, _ = compute_correlation_matrix(mixed_df)
        fig = plot_correlation_heatmap(corr)
        assert isinstance(fig, go.Figure)

    def test_plot_time_series(self, datetime_df):
        """Test time-series trendline and rolling average plotting."""
        res = compute_time_series_aggregation(
            datetime_df, "timestamp", "sales", freq="W", rolling_window=3
        )
        agg_df = res["ts_df"]
        fig = plot_time_series(agg_df, "Timestamp", "sales", rolling_col=res["rolling_col"])
        assert isinstance(fig, go.Figure)

    def test_plot_scatter_matrix(self, mixed_df):
        """Test scatter matrix pairwise exploration."""
        fig = plot_scatter_matrix(mixed_df, ["Experience", "Salary", "PerformanceScore"])
        assert isinstance(fig, go.Figure)


# ==============================================================================
# 9. EDAManager Facade Tests
# ==============================================================================

class TestEDAManagerFacade:
    """Test facade class integrating all exploratory modules."""

    def test_facade_methods(self, mixed_df):
        """Verify EDAManager delegation methods."""
        eda = EDAManager(df=mixed_df, target_column="Salary")

        num_stats = eda.compute_numerical_stats("Experience")
        assert num_stats["column"] == "Experience"

        cat_stats = eda.compute_categorical_stats("Department")
        assert cat_stats["column"] == "Department"

        corr_matrix = eda.compute_correlations()
        assert "Salary" in corr_matrix.columns

        insights = eda.generate_insights()
        assert len(insights) > 0
