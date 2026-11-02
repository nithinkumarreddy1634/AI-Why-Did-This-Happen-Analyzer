"""Categorical feature encoding transformers: Label, One-Hot, and Ordinal."""

from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


class CategoricalEncoder:
    """Base interface for categorical encoders that prevents data leakage."""

    def fit(self, df: pd.DataFrame, columns: List[str]) -> "CategoricalEncoder":
        raise NotImplementedError

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError

    def fit_transform(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        return self.fit(df, columns).transform(df)


class LabelEncodingTransformer(CategoricalEncoder):
    """Encodes categorical levels into integer ranks [0, k-1].
    
    Maintains a mapping dictionary for each column and handles unseen categories safely.
    """

    def __init__(self, unseen_value: int = -1) -> None:
        self.unseen_value = unseen_value
        self.mappings: Dict[str, Dict[str, int]] = {}
        self.columns: List[str] = []

    def fit(self, df: pd.DataFrame, columns: List[str]) -> "LabelEncodingTransformer":
        self.columns = [c for c in columns if c in df.columns]
        self.mappings = {}
        for col in self.columns:
            unique_vals = sorted(df[col].dropna().astype(str).unique().tolist())
            self.mappings[col] = {val: idx for idx, val in enumerate(unique_vals)}
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        new_df = df.copy()
        for col in self.columns:
            if col in new_df.columns:
                mapping = self.mappings.get(col, {})
                series_str = new_df[col].astype(str)
                # Map known categories, assign unseen_value to unknown
                encoded_series = series_str.map(mapping).fillna(self.unseen_value).astype(int)
                new_df[col] = encoded_series
        return new_df


class OneHotEncodingTransformer(CategoricalEncoder):
    """Converts categorical columns into binary indicator columns (one-hot).
    
    Supports drop_first and preserves column schema across train and test sets.
    """

    def __init__(self, drop_first: bool = False, prefix_sep: str = "_") -> None:
        self.drop_first = drop_first
        self.prefix_sep = prefix_sep
        self.columns: List[str] = []
        self.category_levels: Dict[str, List[str]] = {}
        self.output_column_names: List[str] = []

    def fit(self, df: pd.DataFrame, columns: List[str]) -> "OneHotEncodingTransformer":
        self.columns = [c for c in columns if c in df.columns]
        self.category_levels = {}
        self.output_column_names = []

        for col in self.columns:
            levels = sorted(df[col].dropna().astype(str).unique().tolist())
            if self.drop_first and len(levels) > 1:
                levels = levels[1:]
            self.category_levels[col] = levels
            for lvl in levels:
                self.output_column_names.append(f"{col}{self.prefix_sep}{lvl}")
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        new_df = df.copy()
        for col in self.columns:
            if col in new_df.columns:
                levels = self.category_levels.get(col, [])
                series_str = new_df[col].astype(str)
                # Generate binary indicator columns
                for lvl in levels:
                    col_name = f"{col}{self.prefix_sep}{lvl}"
                    new_df[col_name] = (series_str == lvl).astype(int)
                # Drop original categorical column
                new_df = new_df.drop(columns=[col])
        return new_df


class OrdinalEncodingTransformer(CategoricalEncoder):
    """Encodes ordered categorical features according to explicit domain rankings.
    
    e.g., ['Low', 'Medium', 'High'] -> [0, 1, 2].
    """

    def __init__(
        self,
        order_mappings: Optional[Dict[str, List[str]]] = None,
        unseen_value: int = -1,
    ) -> None:
        self.order_mappings = order_mappings or {}
        self.unseen_value = unseen_value
        self.columns: List[str] = []
        self.level_to_rank: Dict[str, Dict[str, int]] = {}

    def fit(self, df: pd.DataFrame, columns: List[str]) -> "OrdinalEncodingTransformer":
        self.columns = [c for c in columns if c in df.columns]
        self.level_to_rank = {}

        for col in self.columns:
            custom_order = self.order_mappings.get(col)
            if not custom_order:
                # Fallback to alphanumeric sorted levels if not explicitly specified
                custom_order = sorted(df[col].dropna().astype(str).unique().tolist())
            self.level_to_rank[col] = {lvl: idx for idx, lvl in enumerate(custom_order)}
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        new_df = df.copy()
        for col in self.columns:
            if col in new_df.columns:
                rank_map = self.level_to_rank.get(col, {})
                series_str = new_df[col].astype(str)
                new_df[col] = series_str.map(rank_map).fillna(self.unseen_value).astype(int)
        return new_df


def is_likely_identifier(series: pd.Series) -> bool:
    """Heuristic check to detect if a column is a unique record identifier/key."""
    total_len = len(series)
    if total_len < 10:
        return False
    unique_count = series.nunique()
    # If 98%+ of records are unique and string or sequential integers, likely an ID
    if (unique_count / total_len) >= 0.98:
        return True
    # Column name patterns
    name_lower = str(series.name).lower()
    if any(id_kw in name_lower for id_kw in ("_id", "id_", "identifier", "uuid", "guid", "ssn", "vin", "account_no")):
        return True
    return False


def apply_encoding(
    df: pd.DataFrame,
    columns: List[str],
    method: str = "one_hot",
    target_column: Optional[str] = None,
    ordinal_orderings: Optional[Dict[str, List[str]]] = None,
    drop_first: bool = False,
) -> Tuple[pd.DataFrame, CategoricalEncoder, List[str], str]:
    """Execute categorical encoding with protection against encoding target or ID columns.
    
    Args:
        df: Input DataFrame.
        columns: List of columns to encode.
        method: 'one_hot', 'label', or 'ordinal'.
        target_column: Target variable column name (strictly skipped).
        ordinal_orderings: Custom dictionary of {column: [ordered_categories]}.
        drop_first: Whether to drop first dummy column in one-hot encoding.

    Returns:
    (new_df, fitted_encoder, new_columns_generated, message)
    """
    safe_columns = []
    skipped_target = False
    skipped_ids = []

    for col in columns:
        if col not in df.columns:
            continue
        # Check target protection
        if target_column and col == target_column:
            skipped_target = True
            continue
        # Check identifier protection
        if is_likely_identifier(df[col]):
            skipped_ids.append(col)
            continue
        safe_columns.append(col)

    if not safe_columns:
        skip_msg = []
        if skipped_target:
            skip_msg.append(f"Skipped target variable '{target_column}'.")
        if skipped_ids:
            skip_msg.append(f"Skipped identifier column(s): {', '.join(skipped_ids)}.")
        return df.copy(), None, [], f"No valid feature columns to encode. {' '.join(skip_msg)}"

    method = method.lower()
    if method in ("one_hot", "onehot", "dummy"):
        encoder = OneHotEncodingTransformer(drop_first=drop_first)
    elif method in ("label", "label_encoder"):
        encoder = LabelEncodingTransformer()
    elif method in ("ordinal", "ordinal_encoder"):
        encoder = OrdinalEncodingTransformer(order_mappings=ordinal_orderings)
    else:
        raise ValueError(f"Unsupported encoding method: '{method}'")

    encoder.fit(df, safe_columns)
    new_df = encoder.transform(df)

    # Calculate added/altered columns
    added_cols = [c for c in new_df.columns if c not in df.columns]
    
    msg = f"Applied {method.upper()} encoding to {len(safe_columns)} column(s): {', '.join(safe_columns)}."
    if added_cols:
        msg += f" Created {len(added_cols)} encoded column(s)."
    if skipped_target:
        msg += f" (Protected target '{target_column}')."
    if skipped_ids:
        msg += f" (Protected identifier columns: {', '.join(skipped_ids)})."

    return new_df, encoder, added_cols, msg
