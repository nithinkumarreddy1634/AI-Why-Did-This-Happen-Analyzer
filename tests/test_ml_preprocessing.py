"""Tests for ML Preprocessing, Problem Detection, and Data Splitting."""

import pytest
import pandas as pd
import numpy as np

from modules.machine_learning.problem_detector import (
    ProblemType,
    detect_problem_type,
)
from modules.machine_learning.ml_preprocessing import (
    MLPreprocessorConfig,
    select_feature_columns,
    fit_ml_preprocessor,
    encode_target_series,
)
from modules.machine_learning.data_splitter import (
    SplitConfig,
    prepare_and_split_data,
)
from utils.ml_helpers import (
    detect_identifier_columns,
    detect_constant_columns,
    safe_divide,
    safe_mape,
)


def test_problem_type_detection():
    # 1. Binary Categorical
    df_binary = pd.DataFrame({
        "churn": ["Yes", "No", "Yes", "Yes", "No", "No", "Yes", "No"],
        "feature1": [1, 2, 3, 4, 5, 6, 7, 8],
    })
    res_b = detect_problem_type(df_binary, "churn")
    assert res_b.problem_type == ProblemType.BINARY_CLASSIFICATION
    assert res_b.n_unique == 2
    assert res_b.is_valid_for_ml is True

    # 2. Binary Numeric (0 and 1)
    df_num_binary = pd.DataFrame({
        "target": [0, 1, 1, 0, 1, 0, 1, 1],
        "feature1": [1, 2, 3, 4, 5, 6, 7, 8],
    })
    res_nb = detect_problem_type(df_num_binary, "target")
    assert res_nb.problem_type == ProblemType.BINARY_CLASSIFICATION

    # 3. Multiclass Categorical
    df_multi = pd.DataFrame({
        "status": ["low", "med", "high", "low", "high", "med", "critical", "low"],
        "feature1": [1, 2, 3, 4, 5, 6, 7, 8],
    })
    res_m = detect_problem_type(df_multi, "status")
    assert res_m.problem_type == ProblemType.MULTICLASS_CLASSIFICATION
    assert res_m.n_unique == 4

    # 4. Continuous Regression
    rng = np.random.RandomState(42)
    df_reg = pd.DataFrame({
        "price": rng.normal(100, 15, size=50),
        "feat": rng.normal(0, 1, size=50),
    })
    res_r = detect_problem_type(df_reg, "price")
    assert res_r.problem_type == ProblemType.REGRESSION

    # 5. Invalid / Constant Target
    df_const = pd.DataFrame({"constant_col": [1, 1, 1, 1, 1]})
    res_c = detect_problem_type(df_const, "constant_col")
    assert res_c.is_valid_for_ml is False
    assert "constant" in res_c.error_message.lower()

    # 6. Missing Target Column
    res_none = detect_problem_type(df_binary, "non_existent")
    assert res_none.is_valid_for_ml is False


def test_identifier_and_constant_exclusion():
    df = pd.DataFrame({
        "customer_id": [f"ID_{i}" for i in range(20)],
        "user_uuid": [f"uuid_{i}" for i in range(20)],
        "zero_var": [999] * 20,
        "valid_num": np.random.randn(20),
        "valid_cat": ["A", "B"] * 10,
        "target": [0, 1] * 10,
    })

    id_cols = detect_identifier_columns(df)
    assert "customer_id" in id_cols
    assert "user_uuid" in id_cols

    const_cols = detect_constant_columns(df)
    assert "zero_var" in const_cols

    num_f, cat_f, excluded = select_feature_columns(
        df,
        target_col="target",
        config=MLPreprocessorConfig(auto_exclude_identifiers=True, auto_exclude_constants=True),
    )
    assert "customer_id" in excluded
    assert "user_uuid" in excluded
    assert "zero_var" in excluded
    assert "valid_num" in num_f
    assert "valid_cat" in cat_f


def test_preprocessing_leakage_prevention_and_unseen_categories():
    # Training set
    df_train = pd.DataFrame({
        "num": [10.0, 20.0, np.nan, 40.0],
        "cat": ["Apple", "Banana", np.nan, "Apple"],
    })

    # Holdout test set with unseen category 'Cherry' and missing values
    df_test = pd.DataFrame({
        "num": [np.nan, 30.0],
        "cat": ["Cherry", "Banana"],
    })

    prep_res = fit_ml_preprocessor(
        X_train=df_train,
        numeric_features=["num"],
        categorical_features=["cat"],
        excluded_columns=[],
        config=MLPreprocessorConfig(
            numeric_imputer_strategy="median",
            categorical_encoder="onehot",
        ),
    )

    # Median of [10, 20, 40] is 20.0
    imputer = prep_res.preprocessor.named_transformers_["num"].named_steps["imputer"]
    assert imputer.statistics_[0] == 20.0

    # Transform test set - should NOT crash on unseen 'Cherry'
    X_test_trans = prep_res.preprocessor.transform(df_test)
    assert X_test_trans.shape[0] == 2
    assert not np.isnan(X_test_trans).any()


def test_target_encoding_with_unseen_categories():
    y_train = pd.Series(["cat", "dog", "dog", "cat", "bird"])
    y_test = pd.Series(["dog", "elephant"])  # elephant is unseen

    y_tr_enc, y_te_enc, le, classes = encode_target_series(y_train, y_test, is_classification=True)
    assert len(classes) == 3
    assert len(y_tr_enc) == 5
    assert len(y_te_enc) == 2
    assert set(y_te_enc).issubset({0, 1, 2})


def test_data_splitting_stratified_and_chronological():
    # Stratified test
    df = pd.DataFrame({
        "x": range(40),
        "target": [0] * 30 + [1] * 10,
        "date": pd.date_range("2024-01-01", periods=40, freq="D"),
    })

    split_res = prepare_and_split_data(
        df=df,
        target_col="target",
        feature_columns=["x"],
        split_config=SplitConfig(test_size=0.25, random_state=42, stratify=True),
        problem_type=ProblemType.BINARY_CLASSIFICATION,
    )
    assert split_res.n_train == 30
    assert split_res.n_test == 10
    # Stratification check (approximately 75% 0s, 25% 1s in test set)
    assert (split_res.y_test == 1).sum() in (2, 3)

    # Chronological test
    split_chrono = prepare_and_split_data(
        df=df,
        target_col="target",
        feature_columns=["x"],
        split_config=SplitConfig(test_size=0.2, time_column="date"),
        problem_type=ProblemType.BINARY_CLASSIFICATION,
    )
    # Check that training timestamps precede test timestamps
    train_max_date = df.loc[split_chrono.X_train.index, "date"].max()
    test_min_date = df.loc[split_chrono.X_test.index, "date"].min()
    assert train_max_date < test_min_date

