"""Tests for helper and validation utilities."""

import pytest
import pandas as pd
from utils.helpers import format_bytes, format_number, get_system_info
from utils.validators import (
    validate_file_extension,
    validate_file_size,
    validate_dataset,
)


def test_format_bytes():
    """Verify human-readable byte formatting."""
    assert format_bytes(500) == "500 B"
    assert format_bytes(2048) == "2.00 KB"
    assert format_bytes(1048576 * 5) == "5.00 MB"


def test_format_number():
    """Verify compact and formatted numbers."""
    assert format_number(1250) == "1.2k"
    assert format_number(2500000) == "2.5M"
    assert format_number(42.567) == "42.57"
    assert format_number(100) == "100"


def test_validate_file_extension():
    """Test allowed and disallowed file extensions."""
    allowed = [".csv", ".xlsx", ".parquet"]
    
    valid, msg = validate_file_extension("customer_churn.csv", allowed)
    assert valid is True
    
    valid, msg = validate_file_extension("sales.XLSX", allowed)
    assert valid is True

    invalid, msg = validate_file_extension("script.exe", allowed)
    assert invalid is False
    assert "Unsupported file format" in msg


def test_validate_file_size():
    """Test file size boundaries."""
    valid, msg = validate_file_size(10 * 1024 * 1024, max_mb=100)
    assert valid is True

    invalid, msg = validate_file_size(150 * 1024 * 1024, max_mb=100)
    assert invalid is False
    assert "exceeds maximum allowed size" in msg


def test_validate_dataset():
    """Test pandas DataFrame validation."""
    valid_df = pd.DataFrame({"age": [25, 30], "churn": [0, 1]})
    is_valid, msg = validate_dataset(valid_df)
    assert is_valid is True

    empty_df = pd.DataFrame()
    is_valid, msg = validate_dataset(empty_df)
    assert is_valid is False

    single_col_df = pd.DataFrame({"age": [25, 30]})
    is_valid, msg = validate_dataset(single_col_df)
    assert is_valid is False
