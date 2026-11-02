"""Unit and integration test suite for Phase 2 Data Ingestion & Understanding Engine."""

import io
from pathlib import Path
import pytest
import pandas as pd
import numpy as np

from config.constants import ColumnCategory, SUPPORTED_EXTENSIONS
from modules.data_loader import DataLoader
from modules.data_profiler import DataProfiler
from utils.validators import (
    validate_file_extension,
    validate_file_size,
    validate_dataset,
    validate_target_variable,
)


@pytest.fixture
def samples_path() -> Path:
    """Path to generated benchmark sample datasets."""
    return Path(__file__).resolve().parent.parent / "data" / "samples"


@pytest.fixture
def data_loader() -> DataLoader:
    """DataLoader instance."""
    return DataLoader()


@pytest.fixture
def data_profiler() -> DataProfiler:
    """DataProfiler instance."""
    return DataProfiler()


# ==============================================================================
# 1. Validation Tests
# ==============================================================================
def test_validate_file_extension():
    """Verify supported and rejected file extensions."""
    allowed = SUPPORTED_EXTENSIONS
    assert validate_file_extension("data.csv", allowed)[0] is True
    assert validate_file_extension("sheet.xlsx", allowed)[0] is True
    assert validate_file_extension("old_sheet.xls", allowed)[0] is True
    assert validate_file_extension("records.parquet", allowed)[0] is True
    
    # Rejections
    assert validate_file_extension("script.py", allowed)[0] is False
    assert validate_file_extension("malicious.exe", allowed)[0] is False
    assert validate_file_extension("no_extension", allowed)[0] is False
    assert validate_file_extension("", allowed)[0] is False


def test_validate_file_size():
    """Verify size limits."""
    assert validate_file_size(1024, max_mb=100)[0] is True
    assert validate_file_size(0, max_mb=100)[0] is False  # Empty file
    assert validate_file_size(105 * 1024 * 1024, max_mb=100)[0] is False  # Oversized


def test_validate_dataset_structure():
    """Verify DataFrame structure validation."""
    # Valid dataset
    df_valid = pd.DataFrame({"A": [1, 2, 3], "B": ["x", "y", "z"]})
    assert validate_dataset(df_valid)[0] is True

    # Empty
    assert validate_dataset(pd.DataFrame())[0] is False
    assert validate_dataset(None)[0] is False

    # Single column (needs at least 2)
    df_single = pd.DataFrame({"A": [1, 2, 3]})
    assert validate_dataset(df_single)[0] is False

    # Duplicate column names
    df_dups = pd.DataFrame([[1, 2]], columns=["col", "col"])
    assert validate_dataset(df_dups)[0] is False


def test_validate_target_variable():
    """Verify target variable requirements."""
    df = pd.DataFrame({
        "Feature": [1, 2, 3, 4],
        "BinaryTarget": ["Yes", "No", "Yes", "No"],
        "ConstantTarget": [1, 1, 1, 1],
        "NullTarget": [np.nan, np.nan, np.nan, np.nan],
    })

    assert validate_target_variable(df, "BinaryTarget")[0] is True
    assert validate_target_variable(df, "ConstantTarget")[0] is False  # Needs >= 2 values
    assert validate_target_variable(df, "NullTarget")[0] is False
    assert validate_target_variable(df, "NonExistent")[0] is False


# ==============================================================================
# 2. Data Loader Tests
# ==============================================================================
def test_load_customer_churn_csv(data_loader, samples_path):
    """Test loading standard customer churn CSV."""
    filepath = samples_path / "customer_churn.csv"
    df, meta = data_loader.load_file(filepath)

    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert df.shape[0] >= 500
    assert "CustomerID" in df.columns
    assert "Churn" in df.columns
    assert meta["rows_count"] == df.shape[0]
    assert meta["cols_count"] == df.shape[1]
    assert meta["file_size_bytes"] > 0


def test_load_numerical_sales_csv(data_loader, samples_path):
    """Test loading numerical sales CSV."""
    filepath = samples_path / "sales_performance.csv"
    df, meta = data_loader.load_file(filepath)

    assert isinstance(df, pd.DataFrame)
    assert "MonthlyRevenue" in df.columns
    assert pd.api.types.is_numeric_dtype(df["MonthlyRevenue"])


def test_load_excel_dataset(data_loader, samples_path):
    """Test loading Excel (.xlsx) dataset."""
    filepath = samples_path / "sample_data.xlsx"
    df, meta = data_loader.load_file(filepath)

    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert meta["file_extension"] == ".xlsx"


def test_load_empty_dataset_error(data_loader, samples_path):
    """Verify loading empty file raises clear ValueError."""
    filepath = samples_path / "empty.csv"
    with pytest.raises(ValueError) as excinfo:
        data_loader.load_file(filepath)
    assert "empty" in str(excinfo.value).lower()


def test_load_corrupted_dataset(data_loader, samples_path):
    """Verify loading corrupted file raises clear ValueError."""
    filepath = samples_path / "corrupted.csv"
    with pytest.raises(ValueError) as excinfo:
        data_loader.load_file(filepath)
    assert "could not parse" in str(excinfo.value).lower() or "dataset" in str(excinfo.value).lower()


def test_load_multi_encoding_csv(data_loader):
    """Verify CSV with non-utf8 encodings (latin-1) loads seamlessly."""
    latin1_text = "ID,Name,Description\n1,Café,São Paulo\n2,Renée,München\n"
    latin1_bytes = latin1_text.encode("latin-1")
    stream = io.BytesIO(latin1_bytes)
    
    df, meta = data_loader.load_file(stream, file_name="latin1_sample.csv")
    assert df.shape[0] == 2
    assert "Café" in df["Name"].values


def test_load_custom_delimiters(data_loader):
    """Verify semicolon and tab separated CSVs load seamlessly."""
    semicolon_csv = b"ColA;ColB;ColC\n10;20;30\n40;50;60\n"
    df, meta = data_loader.load_file(io.BytesIO(semicolon_csv), file_name="semi.csv")
    assert df.shape == (2, 3)
    assert meta["delimiter_used"] == ";"


# ==============================================================================
# 3. Data Profiler & Column Identification Tests
# ==============================================================================
def test_column_type_identification(data_profiler):
    """Verify robust categorization of continuous, discrete, boolean, identifier, and constant columns."""
    df = pd.DataFrame({
        "CustomerID": [f"ID-{i}" for i in range(100)],  # Identifier
        "Age": np.random.randint(20, 60, size=100),       # Numerical Continuous
        "ChildrenCount": np.random.randint(0, 4, size=100), # Numerical Discrete
        "IsActive": [True, False] * 50,                   # Boolean
        "Gender": ["Male", "Female"] * 50,                # Categorical
        "SignupDate": pd.date_range("2023-01-01", periods=100).strftime("%Y-%m-%d"), # Date & Time
        "ConstantPlatform": ["Web"] * 100,                # Constant
    })

    types = data_profiler.identify_column_types(df)
    assert types["CustomerID"] == ColumnCategory.IDENTIFIER
    assert types["IsActive"] == ColumnCategory.BOOLEAN
    assert types["Gender"] in (ColumnCategory.BOOLEAN, ColumnCategory.CATEGORICAL)
    assert types["ConstantPlatform"] == ColumnCategory.CONSTANT
    assert types["SignupDate"] == ColumnCategory.DATETIME
    assert types["Age"] in (ColumnCategory.NUMERICAL_CONTINUOUS, ColumnCategory.NUMERICAL_DISCRETE)


def test_column_profile_calculation(data_profiler):
    """Verify detailed statistical metrics are computed for numerical and categorical features."""
    df = pd.DataFrame({
        "Score": [10.0, 20.0, 30.0, 40.0, 50.0],
        "Category": ["A", "B", "A", "A", "B"],
    })

    profiles = data_profiler.generate_column_profile(df)
    score_prof = next(p for p in profiles if p["column_name"] == "Score")
    cat_prof = next(p for p in profiles if p["column_name"] == "Category")

    # Score metrics
    assert score_prof["mean"] == 30.0
    assert score_prof["min"] == 10.0
    assert score_prof["max"] == 50.0
    assert score_prof["median"] == 30.0
    assert score_prof["missing_count"] == 0

    # Category metrics
    assert cat_prof["unique_values"] == 2
    assert len(cat_prof["top_categories"]) == 2
    assert cat_prof["top_categories"][0]["category"] == "A"
    assert cat_prof["top_categories"][0]["count"] == 3


def test_data_quality_audit(data_profiler, samples_path):
    """Verify audit correctly detects missingness, duplicates, and constant columns."""
    filepath = samples_path / "missing_data.csv"
    loader = DataLoader()
    df, _ = loader.load_file(filepath)

    audit = data_profiler.audit_data_quality(df)
    assert audit["total_missing_cells"] > 0
    assert audit["missing_percentage"] > 0.0
    assert isinstance(audit["quality_score"], float)
    assert any("Missing Values" in w for w in audit["warnings"])


def test_duplicate_records_detection(data_profiler, samples_path):
    """Verify duplicate records detection."""
    filepath = samples_path / "duplicate_records.csv"
    loader = DataLoader()
    df, _ = loader.load_file(filepath)

    audit = data_profiler.audit_data_quality(df)
    assert audit["duplicate_rows"] > 0
    assert any("Duplicate Records" in w for w in audit["warnings"])


def test_target_variable_inspection(data_profiler):
    """Verify target variable analysis for categorical vs continuous."""
    df = pd.DataFrame({
        "Revenue": np.linspace(100, 1000, 100),
        "Churn": ["Yes", "No"] * 50,
    })

    # Categorical target
    churn_audit = data_profiler.inspect_target_variable(df, "Churn")
    assert churn_audit["valid"] is True
    assert churn_audit["target_type"] == "Categorical"
    assert len(churn_audit["possible_outcomes"]) == 2

    # Numerical target
    rev_audit = data_profiler.inspect_target_variable(df, "Revenue")
    assert rev_audit["valid"] is True
    assert "Numerical" in rev_audit["target_type"]
    assert any("Median" in o["label"] for o in rev_audit["possible_outcomes"])


def test_factual_insights_generation(data_profiler, samples_path):
    """Verify factual insights are dynamically generated from real data."""
    filepath = samples_path / "customer_churn.csv"
    loader = DataLoader()
    df, _ = loader.load_file(filepath)

    insights = data_profiler.generate_factual_insights(df)
    assert len(insights) >= 3
    # Check that record count matches actual dataframe
    assert f"{len(df):,} records" in insights[0]
