"""Data preparation and scikit-learn preprocessing pipeline for machine learning."""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    StandardScaler,
    RobustScaler,
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    LabelEncoder,
)

from utils.ml_helpers import detect_identifier_columns, detect_constant_columns


@dataclass
class MLPreprocessorConfig:
    """Configuration settings for the ML preprocessing pipeline."""
    numeric_imputer_strategy: str = "median"  # 'median', 'mean', 'constant'
    numeric_scaler: str = "standard"  # 'standard', 'robust', 'minmax', 'none'
    categorical_imputer_strategy: str = "most_frequent"  # 'most_frequent', 'constant'
    categorical_encoder: str = "onehot"  # 'onehot', 'ordinal'
    exclude_columns: List[str] = field(default_factory=list)
    auto_exclude_identifiers: bool = True
    auto_exclude_constants: bool = True
    max_categories_onehot: int = 50


@dataclass
class MLPreprocessingResult:
    """Output container for fitted preprocessing components."""
    preprocessor: ColumnTransformer
    feature_names_in: List[str]
    feature_names_out: List[str]
    numeric_features: List[str]
    categorical_features: List[str]
    excluded_columns: List[str]
    target_encoder: Optional[LabelEncoder] = None
    target_classes: Optional[List[str]] = None
    config: Optional[MLPreprocessorConfig] = None


def select_feature_columns(
    df: pd.DataFrame,
    target_col: str,
    config: Optional[MLPreprocessorConfig] = None,
    explicit_features: Optional[List[str]] = None,
) -> Tuple[List[str], List[str], List[str]]:
    """Determine final numerical and categorical feature lists and excluded columns.
    
    Returns
    -------
    (numeric_features, categorical_features, excluded_columns)
    """
    cfg = config or MLPreprocessorConfig()
    all_cols = [c for c in df.columns if c != target_col]
    
    excluded: List[str] = list(cfg.exclude_columns)
    
    # Auto-exclude identifiers if configured
    if cfg.auto_exclude_identifiers:
        id_cols = detect_identifier_columns(df[[c for c in all_cols if c not in excluded]])
        for c in id_cols:
            if c not in excluded:
                excluded.append(c)
                
    # Auto-exclude constant columns if configured
    if cfg.auto_exclude_constants:
        const_cols = detect_constant_columns(df[[c for c in all_cols if c not in excluded]])
        for c in const_cols:
            if c not in excluded:
                excluded.append(c)

    if explicit_features is not None:
        candidate_cols = [c for c in explicit_features if c in all_cols and c != target_col]
        # Any column in candidate_cols that was auto-excluded gets included if explicitly requested
        active_cols = candidate_cols
        excluded = [c for c in all_cols if c not in active_cols]
    else:
        active_cols = [c for c in all_cols if c not in excluded]

    num_cols = []
    cat_cols = []
    for c in active_cols:
        series = df[c]
        if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
            num_cols.append(c)
        else:
            cat_cols.append(c)

    return num_cols, cat_cols, excluded


def build_column_transformer(
    numeric_features: List[str],
    categorical_features: List[str],
    config: Optional[MLPreprocessorConfig] = None,
) -> ColumnTransformer:
    """Build a scikit-learn ColumnTransformer configured with imputers, scalers, and encoders."""
    cfg = config or MLPreprocessorConfig()
    transformers = []

    # 1. Numerical Pipeline
    if numeric_features:
        num_steps = []
        fill_val = 0.0 if cfg.numeric_imputer_strategy == "constant" else None
        num_steps.append(
            ("imputer", SimpleImputer(strategy=cfg.numeric_imputer_strategy, fill_value=fill_val))
        )
        if cfg.numeric_scaler == "standard":
            num_steps.append(("scaler", StandardScaler()))
        elif cfg.numeric_scaler == "robust":
            num_steps.append(("scaler", RobustScaler()))
        elif cfg.numeric_scaler == "minmax":
            num_steps.append(("scaler", MinMaxScaler()))
        # 'none' adds no scaler
        transformers.append(("num", Pipeline(steps=num_steps), numeric_features))

    # 2. Categorical Pipeline
    if categorical_features:
        cat_steps = []
        cat_fill = "missing" if cfg.categorical_imputer_strategy == "constant" else None
        cat_steps.append(
            ("imputer", SimpleImputer(strategy=cfg.categorical_imputer_strategy, fill_value=cat_fill))
        )
        if cfg.categorical_encoder == "onehot":
            cat_steps.append(
                (
                    "encoder",
                    OneHotEncoder(
                        handle_unknown="ignore",
                        sparse_output=False,
                        max_categories=cfg.max_categories_onehot,
                    ),
                )
            )
        else:
            cat_steps.append(
                (
                    "encoder",
                    OrdinalEncoder(
                        handle_unknown="use_encoded_value",
                        unknown_value=-1,
                    ),
                )
            )
        transformers.append(("cat", Pipeline(steps=cat_steps), categorical_features))

    if not transformers:
        raise ValueError("No valid features selected for preprocessing pipeline.")

    return ColumnTransformer(transformers=transformers, remainder="drop")


def fit_ml_preprocessor(
    X_train: pd.DataFrame,
    numeric_features: List[str],
    categorical_features: List[str],
    excluded_columns: List[str],
    config: Optional[MLPreprocessorConfig] = None,
) -> MLPreprocessingResult:
    """Fit preprocessing pipeline strictly on training feature data.
    
    Prevents data leakage by ensuring no test set statistics are seen during fitting.
    """
    cfg = config or MLPreprocessorConfig()
    preprocessor = build_column_transformer(numeric_features, categorical_features, cfg)
    
    # Fit only on the training dataframe
    features_in = numeric_features + categorical_features
    preprocessor.fit(X_train[features_in])

    # Extract output feature names
    try:
        names_out = list(preprocessor.get_feature_names_out())
        # Clean up prefixes for readable display
        clean_names = [
            n.replace("num__", "").replace("cat__", "")
            for n in names_out
        ]
    except Exception:
        clean_names = [f"feat_{i}" for i in range(len(features_in))]

    return MLPreprocessingResult(
        preprocessor=preprocessor,
        feature_names_in=features_in,
        feature_names_out=clean_names,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        excluded_columns=excluded_columns,
        config=cfg,
    )


def encode_target_series(
    y_train: pd.Series,
    y_test: Optional[pd.Series] = None,
    is_classification: bool = True,
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[LabelEncoder], Optional[List[str]]]:
    """Encode target variable safely.
    
    For classification: Uses LabelEncoder fitted strictly on y_train.
    Handles unseen categories in y_test safely.
    For regression: Casts to float.
    """
    if not is_classification:
        y_tr = pd.to_numeric(y_train, errors="coerce").fillna(0.0).to_numpy(dtype=float)
        y_te = None
        if y_test is not None:
            y_te = pd.to_numeric(y_test, errors="coerce").fillna(0.0).to_numpy(dtype=float)
        return y_tr, y_te, None, None

    # Check if target is already 0/1 integer binary
    valid_train = y_train.dropna()
    unique_vals = set(valid_train.unique())
    if unique_vals.issubset({0, 1}) or unique_vals.issubset({0.0, 1.0}):
        y_tr = y_train.astype(int).to_numpy()
        y_te = y_test.astype(int).to_numpy() if y_test is not None else None
        return y_tr, y_te, None, ["0", "1"]

    # String / Categorical / Arbitrary classes
    le = LabelEncoder()
    # Fit on training target
    y_train_str = y_train.astype(str)
    y_tr = le.fit_transform(y_train_str)
    classes = [str(c) for c in le.classes_]

    y_te = None
    if y_test is not None:
        y_test_str = y_test.astype(str)
        # Handle unseen classes by mapping to most frequent training class (class 0)
        known_classes = set(le.classes_)
        safe_test = y_test_str.apply(lambda val: val if val in known_classes else le.classes_[0])
        y_te = le.transform(safe_test)

    return y_tr, y_te, le, classes

