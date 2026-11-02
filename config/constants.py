"""Constants for the AI 'Why Did This Happen?' Analyzer."""

from enum import Enum
from typing import Final, List, Dict

# ==============================================================================
# Application Branding & Metadata
# ==============================================================================
APP_NAME: Final[str] = "AI \"Why Did This Happen?\" Analyzer"
APP_SHORT_NAME: Final[str] = "WhyAnalyzer"
APP_VERSION: Final[str] = "1.0.0-phase1"
APP_TAGLINE: Final[str] = "Go Beyond Predictions. Discover the Reasons Behind the Results."
APP_DESCRIPTION: Final[str] = (
    "An intelligent data investigation platform that combines statistical analysis, "
    "machine learning, and Explainable AI to uncover patterns behind unexpected outcomes."
)
APP_ICON: Final[str] = "🔍"

SUPPORTED_EXTENSIONS: Final[List[str]] = [".csv", ".xlsx", ".xls", ".parquet"]
MAX_FILE_SIZE_MB: Final[int] = 100
DEFAULT_SAMPLE_ROWS: Final[int] = 5


# ==============================================================================
# Column Classifications & Data Health
# ==============================================================================
class ColumnCategory(str, Enum):
    """Inferred and user-assigned column semantic types."""
    NUMERICAL_CONTINUOUS = "Numerical (Continuous)"
    NUMERICAL_DISCRETE = "Numerical (Discrete)"
    CATEGORICAL = "Categorical"
    BOOLEAN = "Boolean"
    DATETIME = "Date & Time"
    IDENTIFIER = "Identifier / Key"
    TEXT = "High-Cardinality Text"
    CONSTANT = "Constant (Zero Variance)"


# ==============================================================================
# Navigation Sections
# ==============================================================================
class NavigationSection(str, Enum):
    """Primary navigation routes inside the analyzer platform."""
    WELCOME = "Welcome & Overview"
    DATASET_OVERVIEW = "Dataset Overview"
    DATA_QUALITY = "Data Quality Profiling"
    DATA_PREPROCESSING = "Data Cleaning & Preprocessing"
    EDA = "Exploratory Analysis"
    OUTCOME_INVESTIGATION = "Unexpected Outcome Detection"
    STATISTICAL_ANALYSIS = "Statistical Analysis"
    MACHINE_LEARNING = "Machine Learning Models"
    EXPLAINABLE_AI = "Explainable AI (SHAP & LIME)"
    ANOMALY_DETECTION = "Anomaly & Outlier Detection"
    ROOT_CAUSE = "Root Cause Investigation Studio"
    INVESTIGATION_REPORTS = "Investigation Reports"


NAV_ICONS: Final[Dict[NavigationSection, str]] = {
    NavigationSection.WELCOME: "🏠",
    NavigationSection.DATASET_OVERVIEW: "📊",
    NavigationSection.DATA_QUALITY: "🩺",
    NavigationSection.DATA_PREPROCESSING: "🧹",
    NavigationSection.EDA: "📈",
    NavigationSection.OUTCOME_INVESTIGATION: "🎯",
    NavigationSection.STATISTICAL_ANALYSIS: "📐",
    NavigationSection.MACHINE_LEARNING: "🤖",
    NavigationSection.EXPLAINABLE_AI: "💡",
    NavigationSection.ANOMALY_DETECTION: "⚡",
    NavigationSection.ROOT_CAUSE: "🔬",
    NavigationSection.INVESTIGATION_REPORTS: "📄",
}

# ==============================================================================
# Analytical Methods & Algorithms
# ==============================================================================
class AnalysisType(str, Enum):
    CLASSIFICATION = "Classification (Categorical Outcomes)"
    REGRESSION = "Regression (Continuous Numeric Outcomes)"
    TIME_SERIES = "Time-Series Spike/Drop Investigation"


class StatisticalMethod(str, Enum):
    T_TEST = "Student's t-Test (Two-Group Difference)"
    ANOVA = "ANOVA (Multi-Group Variance)"
    CHI_SQUARE = "Chi-Square Test of Independence"
    MANN_WHITNEY = "Mann-Whitney U Test (Non-parametric)"
    PEARSON_CORR = "Pearson Linear Correlation"
    SPEARMAN_CORR = "Spearman Rank Correlation"


class MLAlgorithm(str, Enum):
    RANDOM_FOREST = "Random Forest Classifier/Regressor"
    XGBOOST = "XGBoost Gradient Boosting"
    LOGISTIC_REGRESSION = "Logistic Regression"
    DECISION_TREE = "Interpretable Decision Tree"


class ExplainabilityMethod(str, Enum):
    SHAP_TREE = "SHAP TreeExplainer (Global & Local Impact)"
    SHAP_KERNEL = "SHAP KernelExplainer (Model Agnostic)"
    LIME_TABULAR = "LIME Tabular Explainer"
    FEATURE_IMPORTANCE = "Permutation Feature Importance"


class AnomalyMethod(str, Enum):
    ISOLATION_FOREST = "Isolation Forest"
    Z_SCORE = "Z-Score / Standard Deviation"
    IQR = "Interquartile Range (IQR)"


# ==============================================================================
# Enterprise UI Theme Colors
# ==============================================================================
THEME_COLORS = {
    "primary": "#4F46E5",        # Indigo accent
    "primary_hover": "#4338CA",
    "secondary": "#06B6D4",      # Cyan highlight
    "accent": "#F59E0B",         # Amber alert
    "success": "#10B981",        # Emerald green
    "danger": "#EF4444",         # Red error
    "warning": "#F59E0B",        # Amber warning
    "bg_dark": "#0E131F",        # Deep navy/slate dark background
    "card_bg": "#1A2234",        # Elevated dark card background
    "card_border": "#2D3748",    # Card border
    "text_main": "#F8FAFC",      # Bright clean text
    "text_muted": "#94A3B8",     # Subdued muted text
}
