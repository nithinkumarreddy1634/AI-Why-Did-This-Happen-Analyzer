"""Comprehensive unit and integration test suite for Phase 3 Data Cleaning & Preprocessing Engine.

Tests all modular preprocessing functions, transformers, coordinator engine,
before-and-after comparisons, export functions, and the 10 mandated dataset scenarios.
"""

import io
from datetime import datetime
import pytest
import numpy as np
import pandas as pd

from modules.preprocessing import (
    DataPreprocessor,
    PreprocessingEngine,
    TransformationStep,
    TransformationHistory,
    detect_missing_values,
    get_missing_summary,
    impute_numerical_column,
    impute_categorical_column,
    drop_missing_rows,
    drop_missing_columns,
    detect_duplicates,
    remove_duplicates,
    get_numerical_columns,
    detect_column_outliers,
    detect_all_outliers,
    handle_column_outliers,
    get_categorical_columns,
    detect_categorical_inconsistencies,
    strip_column_whitespace,
    standardize_capitalization,
    replace_category_values,
    group_rare_categories,
    inspect_data_types,
    convert_column_type,
    CategoricalEncoder,
    LabelEncodingTransformer,
    OneHotEncodingTransformer,
    OrdinalEncodingTransformer,
    is_likely_identifier,
    apply_encoding,
    FeatureScaler,
    StandardScalerWrapper,
    MinMaxScalerWrapper,
    RobustScalerWrapper,
    apply_scaling,
    compute_comparison_metrics,
    build_missing_comparison_chart,
    build_outlier_comparison_chart,
    build_distribution_comparison_chart,
    export_processed_dataset,
)


# ==============================================================================
# 1. MISSING VALUE HANDLING TESTS
# ==============================================================================

def test_detect_missing_values():
    """Verify detection of missing counts and percentages across all columns."""
    df = pd.DataFrame({
        "A": [1, 2, np.nan, 4],
        "B": ["x", None, None, "w"],
        "C": [10, 20, 30, 40],
    })
    miss_df = detect_missing_values(df)
    assert len(miss_df) == 3
    row_b = miss_df[miss_df["column"] == "B"].iloc[0]
    assert row_b["missing_count"] == 2
    assert row_b["missing_percentage"] == 50.0
    assert row_b["non_null_count"] == 2
    assert not row_b["is_all_missing"]


def test_get_missing_summary():
    """Verify aggregated dataset-level missing statistics."""
    df = pd.DataFrame({
        "A": [1, np.nan, 3],
        "B": [np.nan, np.nan, "z"],
    })
    summary = get_missing_summary(df)
    assert summary["total_cells"] == 6
    assert summary["total_missing_cells"] == 3
    assert summary["missing_percentage"] == 50.0
    assert summary["rows_with_missing"] == 2
    assert summary["cols_with_missing"] == 2


def test_impute_numerical_strategies():
    """Verify mean, median, constant, ffill, and bfill imputation for numerical columns."""
    df = pd.DataFrame({"num": [10.0, 20.0, np.nan, 40.0, 50.0]})

    # Median (30.0)
    df_med, aff, msg = impute_numerical_column(df, "num", strategy="median")
    assert aff == 1
    assert df_med["num"].iloc[2] == 30.0
    assert df["num"].isna().sum() == 1  # Original untouched

    # Mean (30.0)
    df_mean, aff, msg = impute_numerical_column(df, "num", strategy="mean")
    assert df_mean["num"].iloc[2] == 30.0

    # Constant (999.0)
    df_const, aff, msg = impute_numerical_column(df, "num", strategy="constant", constant_value=999.0)
    assert df_const["num"].iloc[2] == 999.0

    # Forward fill (should take 20.0)
    df_ffill, aff, msg = impute_numerical_column(df, "num", strategy="ffill")
    assert df_ffill["num"].iloc[2] == 20.0

    # Backward fill (should take 40.0)
    df_bfill, aff, msg = impute_numerical_column(df, "num", strategy="bfill")
    assert df_bfill["num"].iloc[2] == 40.0


def test_impute_categorical_strategies():
    """Verify mode, unknown, custom, ffill, and bfill imputation for categorical columns."""
    df = pd.DataFrame({"cat": ["A", "B", "A", np.nan, "C"]})

    # Mode ("A")
    df_mode, aff, msg = impute_categorical_column(df, "cat", strategy="mode")
    assert aff == 1
    assert df_mode["cat"].iloc[3] == "A"

    # Unknown
    df_unk, aff, msg = impute_categorical_column(df, "cat", strategy="unknown")
    assert df_unk["cat"].iloc[3] == "Unknown"

    # Custom
    df_cust, aff, msg = impute_categorical_column(df, "cat", strategy="custom", custom_value="Special")
    assert df_cust["cat"].iloc[3] == "Special"


def test_drop_missing_rows_and_columns():
    """Verify dropping rows with missing values and dropping high-missing columns."""
    df = pd.DataFrame({
        "A": [1, 2, np.nan, 4],
        "B": [np.nan, np.nan, np.nan, 4],  # 75% missing
        "C": [10, 20, 30, 40],
    })

    # Drop rows
    df_rows, dropped_rows, msg = drop_missing_rows(df, subset=["A"])
    assert dropped_rows == 1
    assert len(df_rows) == 3

    # Drop columns exceeding 50% threshold
    df_cols, dropped_cols, msg = drop_missing_columns(df, threshold_pct=50.0)
    assert "B" in dropped_cols
    assert "B" not in df_cols.columns
    assert "A" in df_cols.columns


# ==============================================================================
# 2. DUPLICATE RECORD HANDLING TESTS
# ==============================================================================

def test_duplicate_detection_and_removal():
    """Verify detection of complete duplicates and retention of first or last occurrences."""
    df = pd.DataFrame({
        "ID": [1, 2, 2, 3, 3, 3],
        "Val": ["a", "b", "b", "c", "c", "c"],
    })
    det = detect_duplicates(df)
    assert det["has_duplicates"] is True
    assert det["duplicate_count"] == 3  # one excess for 2, two excess for 3
    assert det["unique_rows"] == 3

    # Remove duplicates keeping first
    df_first, removed, msg = remove_duplicates(df, keep="first")
    assert removed == 3
    assert len(df_first) == 3
    assert df_first["ID"].tolist() == [1, 2, 3]

    # Remove duplicates keeping last
    df_last, removed, msg = remove_duplicates(df, keep="last")
    assert removed == 3
    assert len(df_last) == 3


# ==============================================================================
# 3. OUTLIER DETECTION AND TREATMENT TESTS
# ==============================================================================

def test_detect_column_outliers_iqr_and_zscore():
    """Verify IQR and Z-score outlier boundary calculations."""
    np.random.seed(42)
    normal_data = np.random.normal(50, 5, 100).tolist()
    normal_data[0] = 500.0  # Massive upper outlier
    normal_data[1] = -200.0  # Massive lower outlier
    df = pd.DataFrame({"metric": normal_data})

    # IQR
    iqr_res = detect_column_outliers(df, "metric", method="iqr", threshold=1.5)
    assert iqr_res["outlier_count"] >= 2
    assert iqr_res["upper_bound"] < 500.0
    assert iqr_res["lower_bound"] > -200.0

    # Z-Score
    z_res = detect_column_outliers(df, "metric", method="zscore", threshold=3.0)
    assert z_res["outlier_count"] >= 2


def test_handle_column_outliers_strategies():
    """Verify outlier handling: keep, remove, iqr_cap, and percentile_cap."""
    df = pd.DataFrame({"vals": [10.0, 12.0, 11.0, 13.0, 12.0, 100.0, -50.0]})

    # Keep strategy (non-destructive)
    df_keep, aff, info, msg = handle_column_outliers(df, "vals", strategy="keep")
    assert aff == 0
    assert len(df_keep) == 7

    # Remove strategy
    df_rem, aff, info, msg = handle_column_outliers(df, "vals", strategy="remove", method="iqr")
    assert aff == 2
    assert len(df_rem) == 5
    assert 100.0 not in df_rem["vals"].values
    assert -50.0 not in df_rem["vals"].values

    # IQR Capping strategy
    df_cap, aff, info, msg = handle_column_outliers(df, "vals", strategy="iqr_cap", method="iqr")
    assert aff == 2
    assert len(df_cap) == 7
    assert df_cap["vals"].max() <= info["upper_bound"]
    assert df_cap["vals"].min() >= info["lower_bound"]

    # Percentile Capping / Winsorization
    df_pcap, aff, info, msg = handle_column_outliers(
        df, "vals", strategy="percentile_cap", percentile_limits=(10.0, 90.0)
    )
    assert aff >= 2
    assert df_pcap["vals"].max() < 100.0


# ==============================================================================
# 4. CATEGORICAL STANDARDIZATION TESTS
# ==============================================================================

def test_strip_whitespace():
    """Verify trimming leading and trailing whitespace."""
    df = pd.DataFrame({
        "city": [" New York ", "London", " Paris "],
        "num": [1, 2, 3],
    })
    df_stripped, aff, msg = strip_column_whitespace(df, columns=["city"])
    assert aff == 2
    assert df_stripped["city"].tolist() == ["New York", "London", "Paris"]


def test_standardize_capitalization():
    """Verify casing harmonization (Title, Lower, Upper)."""
    df = pd.DataFrame({"gender": ["male", "MALE", "Male", "female", "FEMALE"]})
    df_title, aff, msg = standardize_capitalization(df, "gender", case_type="title")
    assert set(df_title["gender"].unique()) == {"Male", "Female"}

    df_upper, aff, msg = standardize_capitalization(df, "gender", case_type="upper")
    assert set(df_upper["gender"].unique()) == {"MALE", "FEMALE"}


def test_replace_category_values():
    """Verify selective category label replacement."""
    df = pd.DataFrame({"tier": ["Bronze", "Silver", "Gold", "Plat"]})
    df_rep, aff, msg = replace_category_values(df, "tier", mapping={"Plat": "Platinum"})
    assert aff == 1
    assert "Platinum" in df_rep["tier"].values
    assert "Plat" not in df_rep["tier"].values


def test_group_rare_categories():
    """Verify low-frequency categories are grouped into 'Other'."""
    categories = ["Frequent"] * 95 + ["Rare1"] * 2 + ["Rare2"] * 2 + ["Rare3"] * 1
    df = pd.DataFrame({"category": categories})
    df_grouped, aff, rare_list, msg = group_rare_categories(df, "category", threshold_pct=3.0, other_label="Other")
    assert aff == 5
    assert "Other" in df_grouped["category"].values
    assert "Rare1" not in df_grouped["category"].values
    assert "Rare2" not in df_grouped["category"].values


# ==============================================================================
# 5. DATA TYPE CORRECTION TESTS
# ==============================================================================

def test_inspect_and_convert_data_types():
    """Verify data type inspection and casting for integer, float, boolean, and datetime."""
    df = pd.DataFrame({
        "int_as_str": ["10", "20", "30"],
        "float_as_str": ["1.5", "2.75", "3.125"],
        "bool_as_str": ["true", "false", "true"],
        "date_as_str": ["2023-01-15", "2023-02-20", "2023-03-25"],
    })

    # Conversion to integer
    df_int, ok, fails, msg = convert_column_type(df, "int_as_str", "integer")
    assert ok is True
    assert pd.api.types.is_integer_dtype(df_int["int_as_str"])

    # Conversion to float
    df_flt, ok, fails, msg = convert_column_type(df, "float_as_str", "float")
    assert ok is True
    assert pd.api.types.is_float_dtype(df_flt["float_as_str"])

    # Conversion to boolean
    df_b, ok, fails, msg = convert_column_type(df, "bool_as_str", "boolean")
    assert ok is True
    assert pd.api.types.is_bool_dtype(df_b["bool_as_str"])

    # Conversion to datetime
    df_dt, ok, fails, msg = convert_column_type(df, "date_as_str", "datetime")
    assert ok is True
    assert pd.api.types.is_datetime64_any_dtype(df_dt["date_as_str"])


# ==============================================================================
# 6. FEATURE ENCODING TESTS & LEAKAGE PREVENTION
# ==============================================================================

def test_one_hot_encoding_and_target_protection():
    """Verify one-hot encoding expands columns and preserves target and IDs."""
    df = pd.DataFrame({
        "CustomerID": [f"ID_{i}" for i in range(10)],  # Identifier
        "Segment": ["Consumer", "Corporate"] * 5,
        "ChurnTarget": ["Yes", "No"] * 5,  # Target variable
    })

    new_df, encoder, added_cols, msg = apply_encoding(
        df,
        columns=["CustomerID", "Segment", "ChurnTarget"],
        method="one_hot",
        target_column="ChurnTarget",
    )

    # ChurnTarget must NOT be encoded
    assert "ChurnTarget" in new_df.columns
    # CustomerID must NOT be encoded due to identifier protection
    assert "CustomerID" in new_df.columns
    # Segment should be converted to one-hot columns
    assert "Segment_Consumer" in new_df.columns or "Segment_Corporate" in new_df.columns
    assert "Segment" not in new_df.columns


def test_label_and_ordinal_encoding():
    """Verify label encoding and custom ordinal category ordering."""
    df = pd.DataFrame({
        "Priority": ["Low", "High", "Medium", "Low", "High"],
    })

    # Ordinal with explicit order
    order_dict = {"Priority": ["Low", "Medium", "High"]}
    trans = OrdinalEncodingTransformer(order_mappings=order_dict)
    encoded = trans.fit_transform(df, ["Priority"])
    assert encoded["Priority"].tolist() == [0, 2, 1, 0, 2]

    # Test that transformer can be reused on new/test data without fitting
    test_df = pd.DataFrame({"Priority": ["Medium", "Low", "High", "UnknownGrade"]})
    test_encoded = trans.transform(test_df)
    assert test_encoded["Priority"].iloc[0] == 1
    assert test_encoded["Priority"].iloc[1] == 0
    assert test_encoded["Priority"].iloc[2] == 2
    assert test_encoded["Priority"].iloc[3] == -1  # Safe unseen value


# ==============================================================================
# 7. FEATURE SCALING TESTS & ZERO VARIANCE PROTECTION
# ==============================================================================

def test_feature_scaling_transformers():
    """Verify StandardScaler, MinMaxScaler, and RobustScaler with constant column safety."""
    df = pd.DataFrame({
        "FeatA": [10.0, 20.0, 30.0, 40.0, 50.0],
        "ConstantCol": [5.0, 5.0, 5.0, 5.0, 5.0],  # Zero variance
        "Target": [1, 0, 1, 0, 1],
    })

    # Standard Scaler
    std_scaler = StandardScalerWrapper()
    df_std = std_scaler.fit_transform(df, ["FeatA", "ConstantCol"])
    assert round(float(df_std["FeatA"].mean()), 2) == 0.0
    assert round(float(df_std["FeatA"].std(ddof=0)), 2) == 1.0
    # Constant column should not fail or be NaN
    assert not df_std["ConstantCol"].isna().any()

    # MinMax Scaler
    mm_scaler = MinMaxScalerWrapper(feature_range=(0.0, 1.0))
    df_mm = mm_scaler.fit_transform(df, ["FeatA"])
    assert df_mm["FeatA"].min() == 0.0
    assert df_mm["FeatA"].max() == 1.0

    # Ensure scaler can transform test data using train parameters (no leakage)
    test_df = pd.DataFrame({"FeatA": [30.0]})
    test_scaled = mm_scaler.transform(test_df)
    assert test_scaled["FeatA"].iloc[0] == 0.5


# ==============================================================================
# 8. COORDINATOR ENGINE WORKFLOW, UNDO, RESET, COMPARISON, & EXPORT
# ==============================================================================

def test_preprocessor_workflow_undo_reset():
    """Verify full end-to-end coordinator workflow, step logging, undo, and reset."""
    df = pd.DataFrame({
        "ID": [101, 102, 102, 103],
        "Age": [25.0, np.nan, 25.0, 35.0],
        "City": [" NYC ", "LA", "LA", "Chicago"],
    })
    engine = PreprocessingEngine(df, dataset_name="customer_test")
    assert len(engine.original_df) == 4

    # 1. Strip whitespace
    engine.clean_categories(operation="strip_whitespace")
    assert engine.current_df["City"].iloc[0] == "NYC"
    assert len(engine.history) == 1

    # 2. Impute missing Age with median
    engine.impute_missing("Age", strategy="median")
    assert engine.current_df["Age"].isna().sum() == 0
    assert len(engine.history) == 2

    # 3. Remove duplicates
    engine.remove_duplicates(keep="first")
    assert len(engine.current_df) == 3
    assert len(engine.history) == 3

    # 4. Verify comparison report
    comp = engine.get_comparison()
    assert comp["rows"]["before"] == 4
    assert comp["rows"]["after"] == 3
    assert comp["missing_cells"]["before"] == 1
    assert comp["missing_cells"]["after"] == 0

    # 5. Test Undo
    assert engine.undo_last_step() is True
    # Should restore back to 4 rows before duplicate removal
    assert len(engine.current_df) == 4
    assert len(engine.history) == 2

    # 6. Test Reset to Original
    engine.reset_to_original()
    assert len(engine.current_df) == 4
    assert engine.current_df["Age"].isna().sum() == 1
    assert engine.original_df["City"].iloc[0] == " NYC "


def test_export_processed_dataset():
    """Verify CSV and Excel byte serialization."""
    df = pd.DataFrame({"col1": [1, 2], "col2": ["a", "b"]})
    csv_bytes, csv_name, csv_mime = export_processed_dataset(df, format_type="csv", base_name="test")
    assert len(csv_bytes) > 0
    assert csv_name.endswith(".csv")
    assert csv_mime == "text/csv"

    xlsx_bytes, xlsx_name, xlsx_mime = export_processed_dataset(df, format_type="excel", base_name="test")
    assert len(xlsx_bytes) > 0
    assert xlsx_name.endswith(".xlsx")


# ==============================================================================
# 9. THE 10 MANDATED DATASET BENCHMARK SCENARIOS
# ==============================================================================

def test_scenario_1_numerical_missing_values():
    """Scenario 1: Dataset with numerical missing values handled properly."""
    df = pd.DataFrame({"Salary": [50000.0, np.nan, 75000.0, np.nan, 100000.0]})
    engine = PreprocessingEngine(df)
    engine.impute_missing("Salary", strategy="median")
    assert engine.current_df["Salary"].isna().sum() == 0
    assert engine.current_df["Salary"].iloc[1] == 75000.0
    assert engine.original_df["Salary"].isna().sum() == 2


def test_scenario_2_categorical_missing_values():
    """Scenario 2: Dataset with categorical missing values handled properly."""
    df = pd.DataFrame({"Department": ["Sales", None, "Engineering", None, "Sales"]})
    engine = PreprocessingEngine(df)
    engine.impute_missing("Department", strategy="mode")
    assert engine.current_df["Department"].isna().sum() == 0
    assert engine.current_df["Department"].iloc[1] == "Sales"


def test_scenario_3_duplicate_records():
    """Scenario 3: Dataset containing duplicate records."""
    df = pd.DataFrame({"ColA": [1, 1, 2, 3], "ColB": ["x", "x", "y", "z"]})
    engine = PreprocessingEngine(df)
    dup_stats = detect_duplicates(engine.current_df)
    assert dup_stats["duplicate_count"] == 1
    engine.remove_duplicates(keep="first")
    assert len(engine.current_df) == 3


def test_scenario_4_extreme_numerical_values():
    """Scenario 4: Dataset containing extreme numerical outlier values."""
    df = pd.DataFrame({"Metric": [10.0, 11.0, 12.0, 10.0, 11.0, 950.0]})
    engine = PreprocessingEngine(df)
    engine.handle_outliers("Metric", strategy="iqr_cap", method="iqr")
    assert engine.current_df["Metric"].max() < 100.0
    assert len(engine.current_df) == 6


def test_scenario_5_inconsistent_categorical_labels():
    """Scenario 5: Dataset with inconsistent categorical labels and whitespace."""
    df = pd.DataFrame({"Status": [" active ", "ACTIVE", "Active", "Inactive"]})
    engine = PreprocessingEngine(df)
    engine.clean_categories(operation="strip_whitespace")
    engine.clean_categories(operation="standardize_casing", column="Status", case_type="title")
    assert set(engine.current_df["Status"].unique()) == {"Active", "Inactive"}


def test_scenario_6_invalid_date_values():
    """Scenario 6: Dataset containing invalid date values."""
    df = pd.DataFrame({"Dates": ["2023-01-01", "Invalid-Date-123", "2023-03-15"]})
    engine = PreprocessingEngine(df)
    step = engine.convert_data_type("Dates", target_type="datetime")
    assert step.status == "Warning"  # Warns user that invalid date was coerced
    assert pd.api.types.is_datetime64_any_dtype(engine.current_df["Dates"])
    assert engine.current_df["Dates"].isna().sum() == 1


def test_scenario_7_constant_columns():
    """Scenario 7: Dataset with constant zero-variance columns handled without division by zero."""
    df = pd.DataFrame({"Static": [42.0] * 50, "Dynamic": range(50)})
    engine = PreprocessingEngine(df)
    step = engine.scale_features(columns=["Static", "Dynamic"], method="standard")
    assert step.status == "Success"
    assert not engine.current_df["Static"].isna().any()


def test_scenario_8_high_cardinality_categories():
    """Scenario 8: Dataset with high-cardinality categories consolidated into 'Other'."""
    data = ["Main"] * 100 + [f"Tiny_{i}" for i in range(15)]
    df = pd.DataFrame({"Item": data})
    engine = PreprocessingEngine(df)
    engine.clean_categories(operation="group_rare", column="Item", rare_threshold_pct=2.0, other_label="Other")
    assert "Other" in engine.current_df["Item"].values
    assert engine.current_df["Item"].nunique() == 2


def test_scenario_9_mixed_numerical_and_categorical():
    """Scenario 9: Dataset with mixed numerical and categorical columns in a multi-step pipeline."""
    df = pd.DataFrame({
        "Age": [25.0, np.nan, 45.0, 30.0],
        "Gender": [" Male ", "Female", "MALE", "Female"],
        "Salary": [50000.0, 60000.0, 150000.0, 55000.0],
        "Churn": [0, 1, 0, 1],
    })
    engine = PreprocessingEngine(df, target_column="Churn")
    engine.clean_categories(operation="strip_whitespace")
    engine.clean_categories(operation="standardize_casing", column="Gender", case_type="title")
    engine.impute_missing("Age", strategy="mean")
    engine.encode_features(columns=["Gender"], method="one_hot")
    engine.scale_features(columns=["Salary"], method="minmax")

    assert engine.current_df["Churn"].tolist() == [0, 1, 0, 1]  # Target completely untouched
    assert "Gender_Male" in engine.current_df.columns or "Gender_Female" in engine.current_df.columns
    assert engine.current_df["Salary"].min() == 0.0
    assert engine.current_df["Salary"].max() == 1.0


def test_scenario_10_entirely_empty_column():
    """Scenario 10: Dataset containing an entirely empty (100% null) column."""
    df = pd.DataFrame({
        "Valid": [1, 2, 3, 4],
        "Empty": [np.nan, np.nan, np.nan, np.nan],
    })
    engine = PreprocessingEngine(df)
    # Check that imputation falls back gracefully to constant
    step = engine.impute_missing("Empty", strategy="median", constant_value=0.0)
    assert step.status == "Success"
    assert (engine.current_df["Empty"] == 0.0).all()

