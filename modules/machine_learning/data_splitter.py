"""Data splitting and validation strategies for machine learning."""

from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict, Any
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
from modules.machine_learning.problem_detector import ProblemType


@dataclass
class SplitConfig:
    """Configuration for dataset splitting."""
    test_size: float = 0.2
    random_state: int = 42
    stratify: bool = True
    time_column: Optional[str] = None
    cv_folds: int = 5


@dataclass
class SplitResult:
    """Container holding split datasets and metadata."""
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    n_train: int
    n_test: int
    split_strategy: str
    feature_columns: List[str]
    target_column: str
    cv_generator: Any
    metadata: Dict[str, Any] = field(default_factory=dict)


def prepare_and_split_data(
    df: pd.DataFrame,
    target_col: str,
    feature_columns: List[str],
    split_config: Optional[SplitConfig] = None,
    problem_type: ProblemType = ProblemType.BINARY_CLASSIFICATION,
) -> SplitResult:
    """Validate, clean target nulls, and split dataset into training and testing partitions.
    
    Parameters
    ----------
    df : pd.DataFrame
        Complete working dataset.
    target_col : str
        Target variable name.
    feature_columns : List[str]
        Features to include.
    split_config : Optional[SplitConfig]
        Splitting parameters.
    problem_type : ProblemType
        Classification or regression indicator.
    """
    cfg = split_config or SplitConfig()

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataframe.")

    if not feature_columns:
        raise ValueError("No feature columns specified for splitting.")

    # 1. Clean missing target rows
    target_series = df[target_col]
    valid_mask = target_series.notna()
    n_dropped_targets = int((~valid_mask).sum())
    
    clean_df = df.loc[valid_mask].copy()
    n_total = len(clean_df)

    if n_total < 5:
        raise ValueError(f"Dataset has only {n_total} records with non-null target. At least 5 required.")

    X = clean_df[feature_columns].copy()
    y = clean_df[target_col].copy()

    split_strategy = "Random Shuffle"
    metadata: Dict[str, Any] = {
        "n_initial_rows": len(df),
        "n_dropped_missing_targets": n_dropped_targets,
        "n_clean_rows": n_total,
        "test_ratio": cfg.test_size,
    }

    # 2. Time-Aware Chronological Splitting
    if cfg.time_column and cfg.time_column in df.columns:
        # Sort chronologically to prevent temporal leakage
        time_series = pd.to_datetime(clean_df[cfg.time_column], errors="coerce")
        sorted_indices = time_series.argsort().values
        X_sorted = X.iloc[sorted_indices]
        y_sorted = y.iloc[sorted_indices]

        split_idx = int(n_total * (1.0 - cfg.test_size))
        # Ensure at least 1 test sample
        split_idx = min(max(split_idx, 1), n_total - 1)

        X_train = X_sorted.iloc[:split_idx]
        X_test = X_sorted.iloc[split_idx:]
        y_train = y_sorted.iloc[:split_idx]
        y_test = y_sorted.iloc[split_idx:]
        split_strategy = f"Time-Aware Chronological (Sorted by '{cfg.time_column}')"
        metadata["time_column"] = cfg.time_column

    # 3. Stratified or Standard Train/Test Split
    else:
        should_stratify = False
        if cfg.stratify and problem_type in (ProblemType.BINARY_CLASSIFICATION, ProblemType.MULTICLASS_CLASSIFICATION):
            # Check minimum class frequency
            class_counts = y.value_counts()
            if (class_counts >= 2).all() and len(class_counts) > 1:
                should_stratify = True
            else:
                metadata["stratification_warning"] = (
                    "Stratification disabled: at least one class has fewer than 2 samples."
                )

        stratify_target = y if should_stratify else None
        split_strategy = "Stratified Split" if should_stratify else "Standard Random Split"

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=cfg.test_size,
            random_state=cfg.random_state,
            stratify=stratify_target,
        )

    # 4. Configure Cross-Validation Generator
    n_splits = min(cfg.cv_folds, len(y_train))
    n_splits = max(2, n_splits)
    
    if problem_type in (ProblemType.BINARY_CLASSIFICATION, ProblemType.MULTICLASS_CLASSIFICATION):
        min_class_train = y_train.value_counts().min()
        if min_class_train >= n_splits:
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=cfg.random_state)
        else:
            cv = KFold(n_splits=n_splits, shuffle=True, random_state=cfg.random_state)
    else:
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=cfg.random_state)

    return SplitResult(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        n_train=len(X_train),
        n_test=len(X_test),
        split_strategy=split_strategy,
        feature_columns=feature_columns,
        target_column=target_col,
        cv_generator=cv,
        metadata=metadata,
    )

