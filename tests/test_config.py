"""Tests for config and constants modules."""

import pytest
from config.settings import get_settings, AppSettings
from config.constants import (
    APP_NAME,
    APP_VERSION,
    SUPPORTED_EXTENSIONS,
    MAX_FILE_SIZE_MB,
    NavigationSection,
    AnalysisType,
)


def test_settings_singleton():
    """Verify settings singleton pattern."""
    settings1 = get_settings()
    settings2 = get_settings()
    assert settings1 is settings2
    assert isinstance(settings1, AppSettings)


def test_directory_creation():
    """Ensure required data directories are instantiated."""
    settings = get_settings()
    assert settings.uploads_dir.exists()
    assert settings.processed_dir.exists()
    assert settings.samples_dir.exists()


def test_constants_integrity():
    """Verify constant values and navigation definitions."""
    assert "Why Did This Happen" in APP_NAME
    assert APP_VERSION.startswith("1.0.0")
    assert ".csv" in SUPPORTED_EXTENSIONS
    assert MAX_FILE_SIZE_MB > 0
    assert len(NavigationSection) >= 10
    assert NavigationSection.WELCOME == "Welcome & Overview"
    assert AnalysisType.CLASSIFICATION is not None
