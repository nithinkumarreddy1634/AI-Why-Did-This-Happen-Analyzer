"""Coordinator engine for managing dataset cleaning, transformations, and comparisons."""

from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from modules.preprocessing.transformation_history import (
    TransformationHistory,
    TransformationStep,
)
from modules.preprocessing.missing_values import (
    detect_missing_values,
    get_missing_summary,
    impute_numerical_column,
    impute_categorical_column,
    drop_missing_rows,
    drop_missing_columns,
)
from modules.preprocessing.duplicates import (
    detect_duplicates,
    remove_duplicates,
)
from modules.preprocessing.outliers import (
    get_numerical_columns,
    detect_column_outliers,
    detect_all_outliers,
    handle_column_outliers,
)
from modules.preprocessing.categorical_cleaning import (
    get_categorical_columns,
    detect_categorical_inconsistencies,
    strip_column_whitespace,
    standardize_capitalization,
    replace_category_values,
    group_rare_categories,
)
from modules.preprocessing.data_types import (
    inspect_data_types,
    convert_column_type,
)
from modules.preprocessing.encoding import (
    CategoricalEncoder,
    apply_encoding,
    is_likely_identifier,
)
from modules.preprocessing.scaling import (
    FeatureScaler,
    apply_scaling,
)
from modules.preprocessing.comparison import compute_comparison_metrics
from modules.preprocessing.export import export_processed_dataset


class DataPreprocessor:
    """Master coordinator orchestrating all Phase 3 preprocessing transformations."""

    def __init__(
        self,
        df: Optional[pd.DataFrame] = None,
        target_column: Optional[str] = None,
        dataset_name: str = "Dataset",
    ) -> None:
        self.dataset_name = dataset_name
        self.target_column = target_column
        self.history = TransformationHistory()
        self.fitted_encoders: Dict[str, CategoricalEncoder] = {}
        self.fitted_scalers: Dict[str, FeatureScaler] = {}
        self._history_snapshots: List[pd.DataFrame] = []

        if df is not None:
            self.set_dataset(df, dataset_name, target_column)
        else:
            self.original_df: Optional[pd.DataFrame] = None
            self.processed_df: Optional[pd.DataFrame] = None

    @property
    def current_df(self) -> Optional[pd.DataFrame]:
        """Working dataset copy after transformations."""
        return self.processed_df

    @current_df.setter
    def current_df(self, df: pd.DataFrame) -> None:
        """Update working dataset copy."""
        self.processed_df = df

    def set_dataset(
        self,
        df: pd.DataFrame,
        dataset_name: str = "Dataset",
        target_column: Optional[str] = None,
    ) -> None:
        """Load dataset, keeping an untouched original reference copy."""
        if df is None or not isinstance(df, pd.DataFrame):
            raise ValueError("Input dataset must be a valid pandas DataFrame.")

        self.dataset_name = dataset_name
        self.target_column = target_column
        # Strict deep copy to prevent mutation of original uploaded dataset
        self.original_df = df.copy(deep=True)
        self.processed_df = df.copy(deep=True)
        self.history.clear()
        self._history_snapshots.clear()
        self.fitted_encoders.clear()
        self.fitted_scalers.clear()

    def reset_to_original(self) -> None:
        """Reset the working dataset back to the uploaded original state."""
        if self.original_df is not None:
            self.processed_df = self.original_df.copy(deep=True)
            self._history_snapshots.clear()
            self.history.record_step(
                operation_name="Reset to Original",
                column_affected="Entire Dataset",
                method_selected="Revert",
                parameters_used={},
                records_affected=len(self.processed_df),
                status="Success",
                details="Dataset reset to original uploaded state. All transformations reverted.",
            )
            self.fitted_encoders.clear()
            self.fitted_scalers.clear()

    def undo_last_step(self) -> bool:
        """Revert the most recent transformation and restore preceding dataframe state."""
        if self._history_snapshots:
            self.processed_df = self._history_snapshots.pop()
            if self.history.steps:
                self.history.steps.pop()
            return True
        elif self.original_df is not None and len(self.history.steps) > 0:
            self.processed_df = self.original_df.copy(deep=True)
            self.history.steps.clear()
            return True
        return False

    def get_quality_summary(self) -> Dict[str, Any]:
        """Compute high-level data health metrics for the current processed dataset."""
        if self.processed_df is None or self.processed_df.empty:
            return {
                "total_rows": 0,
                "total_columns": 0,
                "missing_cells": 0,
                "missing_pct": 0.0,
                "duplicate_count": 0,
                "duplicate_pct": 0.0,
                "numerical_cols": 0,
                "categorical_cols": 0,
                "outlier_count": 0,
            }

        df = self.processed_df
        miss_sum = get_missing_summary(df)
        dup_sum = detect_duplicates(df)
        num_cols = get_numerical_columns(df)
        cat_cols = get_categorical_columns(df)

        total_outliers = 0
        for col in num_cols:
            out_info = detect_column_outliers(df, col, method="iqr", threshold=1.5)
            total_outliers += out_info["outlier_count"]

        return {
            "total_rows": len(df),
            "total_columns": df.shape[1],
            "missing_cells": miss_sum["total_missing_cells"],
            "missing_pct": miss_sum["missing_percentage"],
            "rows_with_missing": miss_sum["rows_with_missing"],
            "duplicate_count": dup_sum["duplicate_count"],
            "duplicate_pct": dup_sum["duplicate_percentage"],
            "numerical_cols": len(num_cols),
            "categorical_cols": len(cat_cols),
            "outlier_count": total_outliers,
        }

    # -------------------------------------------------------------------------
    # Missing Value Handling
    # -------------------------------------------------------------------------
    def impute_missing(
        self,
        column: str,
        strategy: str,
        constant_value: Optional[Union[float, str]] = None,
        fill_value: Optional[Union[float, str]] = None,
    ) -> TransformationStep:
        """Apply missing value imputation to a single column."""
        if self.processed_df is None:
            raise ValueError("No dataset loaded.")

        df = self.processed_df
        if column not in df.columns:
            raise ValueError(f"Column '{column}' does not exist.")

        actual_fill = fill_value if fill_value is not None else constant_value
        is_num = pd.api.types.is_numeric_dtype(df[column])

        # Push state snapshot before modifying
        self._history_snapshots.append(self.processed_df.copy(deep=True))

        if is_num:
            val = float(actual_fill) if actual_fill is not None else 0.0
            new_df, affected, msg = impute_numerical_column(
                df, column=column, strategy=strategy, constant_value=val
            )
        else:
            val = str(actual_fill) if actual_fill is not None else "Unknown"
            new_df, affected, msg = impute_categorical_column(
                df, column=column, strategy=strategy, custom_value=val
            )

        self.processed_df = new_df
        return self.history.record_step(
            operation_name="Impute Missing",
            column_affected=column,
            method_selected=strategy,
            parameters_used={"strategy": strategy, "constant_value": actual_fill},
            records_affected=affected,
            status="Success",
            details=msg,
        )

    def drop_missing(
        self,
        how: str = "rows",
        subset: Optional[List[str]] = None,
        threshold_pct: float = 50.0,
        axis: Optional[str] = None,
    ) -> TransformationStep:
        """Drop missing records (rows) or columns exceeding null threshold."""
        if self.processed_df is None:
            raise ValueError("No dataset loaded.")

        if axis is not None:
            how = axis

        # Push state snapshot before modifying
        self._history_snapshots.append(self.processed_df.copy(deep=True))

        if how in ("rows", "row", "index"):
            new_df, affected, msg = drop_missing_rows(self.processed_df, subset=subset)
            col_target = ", ".join(subset) if subset else "All Columns"
            op_name = "Drop Missing Rows"
        else:
            new_df, dropped_cols, msg = drop_missing_columns(
                self.processed_df, threshold_pct=threshold_pct
            )
            affected = len(dropped_cols)
            col_target = ", ".join(dropped_cols) if dropped_cols else "None"
            op_name = "Drop Missing Columns"

        self.processed_df = new_df
        return self.history.record_step(
            operation_name=op_name,
            column_affected=col_target,
            method_selected=f"{how.title()} (threshold={threshold_pct}%)" if how in ("columns", "column") else "Rows (any)",
            parameters_used={"how": how, "subset": subset, "threshold_pct": threshold_pct},
            records_affected=affected,
            status="Success" if affected > 0 else "Warning",
            details=msg,
        )

    # -------------------------------------------------------------------------
    # Duplicate Handling
    # -------------------------------------------------------------------------
    def remove_duplicate_records(
        self, subset: Optional[List[str]] = None, keep: str = "first"
    ) -> TransformationStep:
        """Remove duplicate rows with configurable keep rule."""
        if self.processed_df is None:
            raise ValueError("No dataset loaded.")

        # Push state snapshot before modifying
        self._history_snapshots.append(self.processed_df.copy(deep=True))

        new_df, affected, msg = remove_duplicates(
            self.processed_df, subset=subset, keep=keep
        )
        self.processed_df = new_df
        return self.history.record_step(
            operation_name="Remove Duplicates",
            column_affected=", ".join(subset) if subset else "All Columns",
            method_selected=f"Keep {keep.capitalize()}",
            parameters_used={"subset": subset, "keep": keep},
            records_affected=affected,
            status="Success",
            details=msg,
        )

    remove_duplicates = remove_duplicate_records

    # -------------------------------------------------------------------------
    # Outlier Treatment
    # -------------------------------------------------------------------------
    def treat_outliers(
        self,
        column: str,
        strategy: str,
        method: str = "iqr",
        threshold: float = 1.5,
        percentile_limits: Tuple[float, float] = (1.0, 99.0),
    ) -> TransformationStep:
        """Apply outlier handling strategy to a numerical column."""
        if self.processed_df is None:
            raise ValueError("No dataset loaded.")

        # Push state snapshot before modifying
        self._history_snapshots.append(self.processed_df.copy(deep=True))

        new_df, affected, info, msg = handle_column_outliers(
            self.processed_df,
            column=column,
            strategy=strategy,
            method=method,
            threshold=threshold,
            percentile_limits=percentile_limits,
        )
        self.processed_df = new_df
        return self.history.record_step(
            operation_name="Handle Outliers",
            column_affected=column,
            method_selected=f"{method.upper()} -> {strategy}",
            parameters_used={
                "method": method,
                "strategy": strategy,
                "threshold": threshold,
                "percentile_limits": percentile_limits,
            },
            records_affected=affected,
            status="Success",
            details=msg,
        )

    handle_outliers = treat_outliers

    # -------------------------------------------------------------------------
    # Categorical Standardization
    # -------------------------------------------------------------------------
    def clean_categorical_text(
        self,
        column: Optional[str] = None,
        operation: str = "strip",
        case_type: str = "title",
    ) -> TransformationStep:
        """Standardize text casing or strip whitespace."""
        if self.processed_df is None:
            raise ValueError("No dataset loaded.")

        # Push state snapshot before modifying
        self._history_snapshots.append(self.processed_df.copy(deep=True))

        if operation == "strip":
            cols = [column] if column else None
            new_df, affected, msg = strip_column_whitespace(self.processed_df, columns=cols)
            col_target = column or "All Text Columns"
            method_desc = "Strip Whitespace"
        elif operation in ("casing", "capitalize"):
            if not column:
                raise ValueError("Column name required for capitalization standardization.")
            new_df, affected, msg = standardize_capitalization(
                self.processed_df, column=column, case_type=case_type
            )
            col_target = column
            method_desc = f"Casing ({case_type})"
        else:
            raise ValueError(f"Unknown operation: '{operation}'")

        self.processed_df = new_df
        return self.history.record_step(
            operation_name="Standardize Text",
            column_affected=col_target,
            method_selected=method_desc,
            parameters_used={"operation": operation, "case_type": case_type},
            records_affected=affected,
            status="Success",
            details=msg,
        )

    def clean_categories(
        self,
        operation: str = "strip_whitespace",
        column: Optional[str] = None,
        columns_list: Optional[List[str]] = None,
        case_type: str = "title",
        mapping: Optional[Dict[str, str]] = None,
        rare_threshold_pct: float = 3.0,
        other_label: str = "Other",
    ) -> TransformationStep:
        """Universal categorical cleaning dispatcher for the UI."""
        if self.processed_df is None:
            raise ValueError("No dataset loaded.")

        op = operation.lower().strip()
        if op in ("strip", "strip_whitespace"):
            self._history_snapshots.append(self.processed_df.copy(deep=True))
            cols = columns_list or ([column] if column else None)
            new_df, affected, msg = strip_column_whitespace(self.processed_df, columns=cols)
            col_target = ", ".join(cols) if cols else "All Text Columns"
            self.processed_df = new_df
            return self.history.record_step(
                operation_name="Strip Whitespace",
                column_affected=col_target,
                method_selected="Strip Leading/Trailing Spaces",
                parameters_used={"operation": op},
                records_affected=affected,
                status="Success",
                details=msg,
            )
        elif op in ("casing", "standardize_casing", "capitalize"):
            if not column:
                raise ValueError("Column required for casing standardization.")
            return self.clean_categorical_text(column=column, operation="casing", case_type=case_type)
        elif op in ("replace", "replace_values"):
            if not column or not mapping:
                raise ValueError("Column and mapping required for value replacement.")
            return self.replace_categories(column=column, mapping=mapping)
        elif op == "group_rare":
            if not column:
                raise ValueError("Column required for rare category grouping.")
            return self.group_rare(column=column, threshold_pct=rare_threshold_pct, other_label=other_label)
        else:
            raise ValueError(f"Unsupported categorical cleaning operation: '{operation}'")

    def replace_categories(
        self, column: str, mapping: Dict[str, str]
    ) -> TransformationStep:
        """Replace specific category labels in a column."""
        if self.processed_df is None:
            raise ValueError("No dataset loaded.")

        # Push state snapshot before modifying
        self._history_snapshots.append(self.processed_df.copy(deep=True))

        new_df, affected, msg = replace_category_values(
            self.processed_df, column=column, mapping=mapping
        )
        self.processed_df = new_df
        return self.history.record_step(
            operation_name="Replace Category Values",
            column_affected=column,
            method_selected="Custom Mapping",
            parameters_used=mapping,
            records_affected=affected,
            status="Success",
            details=msg,
        )

    def group_rare(
        self, column: str, threshold_pct: float = 3.0, other_label: str = "Other"
    ) -> TransformationStep:
        """Consolidate low-frequency categories into 'Other'."""
        if self.processed_df is None:
            raise ValueError("No dataset loaded.")

        # Push state snapshot before modifying
        self._history_snapshots.append(self.processed_df.copy(deep=True))

        new_df, affected, rare_list, msg = group_rare_categories(
            self.processed_df,
            column=column,
            threshold_pct=threshold_pct,
            other_label=other_label,
        )
        self.processed_df = new_df
        return self.history.record_step(
            operation_name="Group Rare Categories",
            column_affected=column,
            method_selected=f"Threshold < {threshold_pct}% -> '{other_label}'",
            parameters_used={"threshold_pct": threshold_pct, "other_label": other_label, "rare_categories": rare_list},
            records_affected=affected,
            status="Success",
            details=msg,
        )

    # -------------------------------------------------------------------------
    # Data Type Conversion
    # -------------------------------------------------------------------------
    def convert_type(
        self, column: str, target_type: str, date_format: Optional[str] = None
    ) -> TransformationStep:
        """Safely convert a column data type."""
        if self.processed_df is None:
            raise ValueError("No dataset loaded.")

        # Push state snapshot before modifying
        self._history_snapshots.append(self.processed_df.copy(deep=True))

        new_df, success, failures, msg = convert_column_type(
            self.processed_df,
            column=column,
            target_type=target_type,
            date_format=date_format,
        )
        self.processed_df = new_df
        status = "Success" if (success and failures == 0) else ("Warning" if success else "Failed")
        return self.history.record_step(
            operation_name="Convert Data Type",
            column_affected=column,
            method_selected=f"Cast to {target_type}",
            parameters_used={"target_type": target_type, "date_format": date_format},
            records_affected=len(self.processed_df) - failures,
            status=status,
            details=msg,
        )

    convert_data_type = convert_type

    # -------------------------------------------------------------------------
    # Categorical Feature Encoding
    # -------------------------------------------------------------------------
    def encode_features(
        self,
        columns: List[str],
        method: str = "one_hot",
        ordinal_orderings: Optional[Dict[str, List[str]]] = None,
        drop_first: bool = False,
    ) -> TransformationStep:
        """Encode categorical columns, guarding against target and ID encoding."""
        if self.processed_df is None:
            raise ValueError("No dataset loaded.")

        # Push state snapshot before modifying
        self._history_snapshots.append(self.processed_df.copy(deep=True))

        new_df, encoder, added_cols, msg = apply_encoding(
            self.processed_df,
            columns=columns,
            method=method,
            target_column=self.target_column,
            ordinal_orderings=ordinal_orderings,
            drop_first=drop_first,
        )
        self.processed_df = new_df
        if encoder:
            for c in columns:
                self.fitted_encoders[c] = encoder

        return self.history.record_step(
            operation_name="Feature Encoding",
            column_affected=", ".join(columns),
            method_selected=method.upper(),
            parameters_used={"method": method, "drop_first": drop_first},
            records_affected=len(self.processed_df),
            status="Success" if encoder is not None else "Warning",
            details=msg,
        )

    # -------------------------------------------------------------------------
    # Feature Scaling
    # -------------------------------------------------------------------------
    def scale_features(
        self,
        columns: List[str],
        method: str = "standard",
        feature_range: Tuple[float, float] = (0.0, 1.0),
    ) -> TransformationStep:
        """Apply scaling to numerical features, protecting target variable."""
        if self.processed_df is None:
            raise ValueError("No dataset loaded.")

        # Push state snapshot before modifying
        self._history_snapshots.append(self.processed_df.copy(deep=True))

        new_df, scaler, msg = apply_scaling(
            self.processed_df,
            columns=columns,
            method=method,
            target_column=self.target_column,
            feature_range=feature_range,
        )
        self.processed_df = new_df
        if scaler:
            for c in columns:
                self.fitted_scalers[c] = scaler

        return self.history.record_step(
            operation_name="Feature Scaling",
            column_affected=", ".join(columns),
            method_selected=method.upper(),
            parameters_used={"method": method, "feature_range": feature_range},
            records_affected=len(self.processed_df),
            status="Success" if scaler is not None else "Warning",
            details=msg,
        )

    # -------------------------------------------------------------------------
    # Before-and-After Comparison Report
    # -------------------------------------------------------------------------
    def get_comparison(self) -> Dict[str, Any]:
        """Compute before-and-after comparison metrics using comparison module."""
        if self.original_df is None or self.processed_df is None:
            return {}
        return compute_comparison_metrics(
            self.original_df, self.processed_df, self.history.steps
        )

    def get_comparison_report(self) -> Dict[str, Any]:
        """Compute before-and-after comparison metrics (report format)."""
        return self.get_comparison()

    # -------------------------------------------------------------------------
    # Dataset Export
    # -------------------------------------------------------------------------
    def export(
        self, format_type: str = "csv", base_name: Optional[str] = None
    ) -> Tuple[bytes, str, str]:
        """Serialize processed dataset to downloadable byte buffer."""
        if self.processed_df is None:
            raise ValueError("No dataset loaded.")
        name = base_name or self.dataset_name
        return export_processed_dataset(
            self.processed_df, format_type=format_type, base_name=name
        )

    def export_dataset(self, export_format: str = "csv") -> Tuple[bytes, str, str]:
        """Serialize processed dataset to bytes for download."""
        return self.export(format_type=export_format)


# Backward compatibility and alternative aliases
Preprocessor = DataPreprocessor
PreprocessingEngine = DataPreprocessor
