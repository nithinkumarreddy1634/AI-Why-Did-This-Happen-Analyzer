"""Configuration package for AI 'Why Did This Happen?' Analyzer."""

from config.settings import AppSettings, get_settings
from config.constants import (
    APP_NAME,
    APP_VERSION,
    APP_TAGLINE,
    APP_DESCRIPTION,
    SUPPORTED_EXTENSIONS,
    MAX_FILE_SIZE_MB,
    NavigationSection,
    AnalysisType,
    MLAlgorithm,
    StatisticalMethod,
    ExplainabilityMethod,
)

__all__ = [
    "AppSettings",
    "get_settings",
    "APP_NAME",
    "APP_VERSION",
    "APP_TAGLINE",
    "APP_DESCRIPTION",
    "SUPPORTED_EXTENSIONS",
    "MAX_FILE_SIZE_MB",
    "NavigationSection",
    "AnalysisType",
    "MLAlgorithm",
    "StatisticalMethod",
    "ExplainabilityMethod",
]
