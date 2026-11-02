"""Target variable and problem type detection for machine learning."""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np


class ProblemType(str, Enum):
    """Supported machine learning problem types."""
    BINARY_CLASSIFICATION = "Binary Classification"
    MULTICLASS_CLASSIFICATION = "Multiclass Classification"
    REGRESSION = "Regression"


@dataclass
class TargetSummary:
    """Detailed summary of the selected target variable and recommended problem type."""
    target_name: str
    problem_type: ProblemType
    data_type: str
    n_samples: int
    n_missing: int
    n_unique: int
    classes: Optional[List[Any]] = None
    class_distribution: Optional[Dict[Any, int]] = None
    class_proportions: Optional[Dict[Any, float]] = None
    is_imbalanced: bool = False
    imbalance_ratio: float = 1.0
    recommendation_reason: str = ""
    is_valid_for_ml: bool = True
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert target summary to dictionary."""
        return {
            "target_name": self.target_name,
            "problem_type": self.problem_type.value,
            "data_type": self.data_type,
            "n_samples": self.n_samples,
            "n_missing": self.n_missing,
            "n_unique": self.n_unique,
            "classes": self.classes,
            "class_distribution": self.class_distribution,
            "class_proportions": self.class_proportions,
            "is_imbalanced": self.is_imbalanced,
            "imbalance_ratio": self.imbalance_ratio,
            "recommendation_reason": self.recommendation_reason,
            "is_valid_for_ml": self.is_valid_for_ml,
            "error_message": self.error_message,
        }


def detect_problem_type(
    df: pd.DataFrame,
    target_col: str,
    force_type: Optional[str] = None,
    max_classification_cardinality: int = 15,
) -> TargetSummary:
    """Analyze target column to automatically detect problem type, distribution, and validity.
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataset containing the target column.
    target_col : str
        Name of target column.
    force_type : Optional[str]
        Manual override ("Binary Classification", "Multiclass Classification", "Regression").
    max_classification_cardinality : int
        Maximum number of discrete numeric unique values to consider as classification.
    """
    if target_col not in df.columns:
        return TargetSummary(
            target_name=target_col,
            problem_type=ProblemType.BINARY_CLASSIFICATION,
            data_type="unknown",
            n_samples=0,
            n_missing=0,
            n_unique=0,
            is_valid_for_ml=False,
            error_message=f"Target column '{target_col}' not found in dataset.",
        )

    series = df[target_col]
    n_samples = len(series)
    n_missing = int(series.isna().sum())
    valid_series = series.dropna()
    n_valid = len(valid_series)
    n_unique = int(valid_series.nunique())
    dtype_str = str(series.dtype)

    if n_valid < 5:
        return TargetSummary(
            target_name=target_col,
            problem_type=ProblemType.BINARY_CLASSIFICATION,
            data_type=dtype_str,
            n_samples=n_samples,
            n_missing=n_missing,
            n_unique=n_unique,
            is_valid_for_ml=False,
            error_message=f"Insufficient non-null observations in target ({n_valid} found, minimum 5 required).",
        )

    if n_unique <= 1:
        return TargetSummary(
            target_name=target_col,
            problem_type=ProblemType.BINARY_CLASSIFICATION,
            data_type=dtype_str,
            n_samples=n_samples,
            n_missing=n_missing,
            n_unique=n_unique,
            is_valid_for_ml=False,
            error_message="Target variable is constant (only 1 unique value). Supervised models cannot learn with zero variance.",
        )

    # 1. Automatic Type Determination
    auto_type: ProblemType
    reason: str

    is_bool_type = pd.api.types.is_bool_dtype(series)
    is_object_or_cat = pd.api.types.is_object_dtype(series) or isinstance(series.dtype, pd.CategoricalDtype) or pd.api.types.is_string_dtype(series)
    is_numeric = pd.api.types.is_numeric_dtype(series) and not is_bool_type

    if is_bool_type:
        auto_type = ProblemType.BINARY_CLASSIFICATION
        reason = "Boolean data type with 2 logical outcomes (True/False)."
    elif is_object_or_cat:
        if n_unique == 2:
            auto_type = ProblemType.BINARY_CLASSIFICATION
            reason = f"Categorical/text data type with exactly 2 unique outcome classes: {list(valid_series.unique()[:2])}."
        elif n_unique <= 50:
            auto_type = ProblemType.MULTICLASS_CLASSIFICATION
            reason = f"Categorical data type with {n_unique} discrete categories."
        else:
            auto_type = ProblemType.MULTICLASS_CLASSIFICATION
            reason = f"High-cardinality categorical target ({n_unique} unique values). High cardinality may impact classification generalization."
    elif is_numeric:
        unique_vals = sorted(valid_series.unique())
        # Check if strictly binary 0 and 1 or 2 unique numbers
        if n_unique == 2:
            auto_type = ProblemType.BINARY_CLASSIFICATION
            reason = f"Numerical variable with exactly 2 distinct values ({unique_vals}). Detected as binary outcome."
        elif pd.api.types.is_integer_dtype(series) and n_unique <= max_classification_cardinality:
            # Check ratio of unique values to sample size
            ratio = n_unique / n_valid
            if ratio < 0.1:
                auto_type = ProblemType.MULTICLASS_CLASSIFICATION
                reason = f"Integer values with only {n_unique} discrete levels (e.g., rating or status code), indicating discrete classes."
            else:
                auto_type = ProblemType.REGRESSION
                reason = f"Numeric discrete values with high relative variety ({n_unique} values across {n_valid} rows)."
        elif pd.api.types.is_float_dtype(series) and n_unique <= max_classification_cardinality and all(float(x).is_integer() for x in unique_vals):
            auto_type = ProblemType.MULTICLASS_CLASSIFICATION
            reason = f"Float numbers representing whole integers with only {n_unique} distinct values, indicating discrete classes."
        else:
            auto_type = ProblemType.REGRESSION
            reason = f"Continuous numeric values with {n_unique} distinct observations across a continuous scale."
    else:
        # Default fallback
        auto_type = ProblemType.BINARY_CLASSIFICATION if n_unique == 2 else ProblemType.REGRESSION
        reason = f"Assigned based on unique value count ({n_unique})."

    # 2. Check Force Override
    final_type = auto_type
    if force_type:
        for pt in ProblemType:
            if force_type.lower() in pt.value.lower() or pt.name.lower() in force_type.lower():
                final_type = pt
                if final_type != auto_type:
                    reason = f"User manually selected {final_type.value} (Auto-detected recommendation was {auto_type.value})."
                break

    # 3. Class Distribution & Imbalance
    classes = None
    class_distribution = None
    class_proportions = None
    is_imbalanced = False
    imbalance_ratio = 1.0

    if final_type in (ProblemType.BINARY_CLASSIFICATION, ProblemType.MULTICLASS_CLASSIFICATION):
        val_counts = valid_series.value_counts()
        classes = [str(c) for c in val_counts.index.tolist()]
        class_distribution = {str(k): int(v) for k, v in val_counts.items()}
        class_proportions = {str(k): float(v / n_valid) for k, v in val_counts.items()}
        
        max_c = val_counts.max()
        min_c = val_counts.min()
        imbalance_ratio = float(max_c / min_c) if min_c > 0 else 1.0
        # If ratio > 3:1 in binary or > 4:1 in multiclass, flag imbalance
        if (final_type == ProblemType.BINARY_CLASSIFICATION and imbalance_ratio >= 3.0) or \
           (final_type == ProblemType.MULTICLASS_CLASSIFICATION and imbalance_ratio >= 4.0):
            is_imbalanced = True

    return TargetSummary(
        target_name=target_col,
        problem_type=final_type,
        data_type=dtype_str,
        n_samples=n_samples,
        n_missing=n_missing,
        n_unique=n_unique,
        classes=classes,
        class_distribution=class_distribution,
        class_proportions=class_proportions,
        is_imbalanced=is_imbalanced,
        imbalance_ratio=round(imbalance_ratio, 2),
        recommendation_reason=reason,
        is_valid_for_ml=True,
    )
