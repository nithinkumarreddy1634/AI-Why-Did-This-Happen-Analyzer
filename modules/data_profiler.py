"""Data profiling and health auditing engine for the AI 'Why Did This Happen?' Analyzer.

Performs schema inference, semantic column categorization, missing value detection,
statistical distribution computation, data quality scoring, and automated insight generation.
"""

from typing import Optional, Dict, Any, List, Tuple
import re
import numpy as np
import pandas as pd

from config.constants import ColumnCategory


class DataProfiler:
    """Comprehensive dataset profiling and data quality engine."""

    ID_PATTERN = re.compile(r"(^id$|_id$|id$|^id|_id|id_|identifier|uuid|guid|key$|_key|ssn|code$)", re.IGNORECASE)
    DATE_PATTERN = re.compile(r"(date|time|timestamp|year|month|day|_at$|_on$)", re.IGNORECASE)

    # Common date-like string patterns for detecting unconverted date columns
    _DATE_VALUE_PATTERN = re.compile(
        r"^\d{4}[-/]\d{1,2}[-/]\d{1,2}"  # YYYY-MM-DD or YYYY/MM/DD
        r"|^\d{1,2}[-/]\d{1,2}[-/]\d{2,4}"  # MM-DD-YYYY or DD/MM/YYYY
        r"|^\d{1,2}\s+\w{3,9}\s+\d{4}"  # 01 Jan 2024
        r"|^\w{3,9}\s+\d{1,2},?\s+\d{4}",  # January 01, 2024
        re.IGNORECASE,
    )

    # Threshold constants for classification heuristics
    _HIGH_CARDINALITY_THRESHOLD = 50
    _HIGH_CARDINALITY_RATIO = 0.5
    _DISCRETE_INT_MAX_UNIQUE = 20
    _DISCRETE_INT_MAX_RATIO = 0.05

    def __init__(self, df: Optional[pd.DataFrame] = None) -> None:
        self.df = df

    # =========================================================================
    # 1. Automatic Column Identification
    # =========================================================================
    def identify_column_types(
        self, df: Optional[pd.DataFrame] = None
    ) -> Dict[str, ColumnCategory]:
        """Classify each column into appropriate semantic categories.

        Considers data types, cardinality ratios, unique counts, column names,
        and value distributions to distinguish true numerical measurements
        from identifiers, booleans, or categories.

        Enhanced detection includes:
        - Potential date columns stored as strings (value-level pattern matching)
        - High-cardinality columns explicitly flagged
        - Better discrete vs continuous integer distinction using ratio heuristics
        """
        target_df = df if df is not None else self.df
        if target_df is None or target_df.empty:
            return {}

        n_rows = len(target_df)
        classifications: Dict[str, ColumnCategory] = {}

        for col in target_df.columns:
            series = target_df[col]
            non_null_series = series.dropna()
            n_unique = non_null_series.nunique()
            col_str = str(col).strip()

            # Rule 1: Constant columns (only 1 unique non-null value)
            if n_unique <= 1:
                classifications[col_str] = ColumnCategory.CONSTANT
                continue

            # Rule 2: Explicit Boolean types or object columns with True/False
            if pd.api.types.is_bool_dtype(series):
                classifications[col_str] = ColumnCategory.BOOLEAN
                continue

            # Rule 3: Explicit Datetime or Time-like
            if pd.api.types.is_datetime64_any_dtype(series) or pd.api.types.is_timedelta64_dtype(series):
                classifications[col_str] = ColumnCategory.DATETIME
                continue

            # Rule 4: String/Object columns that look like dates
            # Enhanced: check both column name patterns AND value-level date patterns
            if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
                is_date_by_name = self.DATE_PATTERN.search(col_str)
                is_date_by_values = self._check_date_value_pattern(non_null_series)

                if is_date_by_name or is_date_by_values:
                    if self._check_datetime_heuristic(non_null_series):
                        classifications[col_str] = ColumnCategory.DATETIME
                        continue

            # Rule 5: Unique Identifiers / Keys
            # (e.g. customer_id, CustomerID, row_id, or cardinality == n_rows with ID naming or high uniqueness)
            cardinality_ratio = n_unique / n_rows if n_rows > 0 else 0
            if (cardinality_ratio >= 0.95 and self.ID_PATTERN.search(col_str)) or (
                cardinality_ratio == 1.0 and (pd.api.types.is_integer_dtype(series) or pd.api.types.is_object_dtype(series)) and self.ID_PATTERN.search(col_str)
            ):
                classifications[col_str] = ColumnCategory.IDENTIFIER
                continue

            # Rule 6: Binary indicators (0/1 or Yes/No) stored as integers or strings
            if n_unique == 2:
                unique_vals = set(non_null_series.unique())
                if unique_vals.issubset({0, 1, 0.0, 1.0, True, False, "0", "1", "true", "false", "yes", "no", "y", "n"}):
                    classifications[col_str] = ColumnCategory.BOOLEAN
                    continue
                else:
                    classifications[col_str] = ColumnCategory.CATEGORICAL
                    continue

            # Rule 7: Numerical Columns (Floats & Integers)
            if pd.api.types.is_numeric_dtype(series):
                # Check if it's an integer ID masquerading as number
                if pd.api.types.is_integer_dtype(series) and self.ID_PATTERN.search(col_str):
                    classifications[col_str] = ColumnCategory.IDENTIFIER
                    continue

                if pd.api.types.is_float_dtype(series):
                    classifications[col_str] = ColumnCategory.NUMERICAL_CONTINUOUS
                else:
                    # Enhanced: Discrete vs Continuous integers using ratio heuristics
                    # An integer column is discrete if it has few unique values AND
                    # a low cardinality ratio relative to dataset size
                    if n_unique <= self._DISCRETE_INT_MAX_UNIQUE:
                        classifications[col_str] = ColumnCategory.NUMERICAL_DISCRETE
                    elif cardinality_ratio <= self._DISCRETE_INT_MAX_RATIO and n_unique <= 100:
                        # Low uniqueness ratio even with moderate unique count → discrete
                        classifications[col_str] = ColumnCategory.NUMERICAL_DISCRETE
                    else:
                        classifications[col_str] = ColumnCategory.NUMERICAL_CONTINUOUS
                continue

            # Rule 8: Categorical vs High-Cardinality Free Text
            if pd.api.types.is_object_dtype(series) or isinstance(series.dtype, pd.CategoricalDtype):
                if n_unique <= self._HIGH_CARDINALITY_THRESHOLD or cardinality_ratio < 0.2:
                    classifications[col_str] = ColumnCategory.CATEGORICAL
                else:
                    classifications[col_str] = ColumnCategory.TEXT
                continue

            # Fallback
            classifications[col_str] = ColumnCategory.CATEGORICAL

        return classifications

    def _check_datetime_heuristic(self, series: pd.Series, sample_size: int = 50) -> bool:
        """Test a random sample to verify if string values parse reliably as datetimes."""
        if series.empty:
            return False
        sample = series.sample(min(len(series), sample_size), random_state=42)
        try:
            converted = pd.to_datetime(sample, errors="coerce")
            valid_ratio = converted.notna().mean()
            return valid_ratio >= 0.8
        except Exception:
            return False

    def _check_date_value_pattern(self, series: pd.Series, sample_size: int = 30) -> bool:
        """Check if string column values match common date format patterns.

        This catches date columns that don't have date-related names but contain
        date-formatted string values (e.g., a column named 'created' containing
        '2024-01-15' values).

        Args:
            series: Non-null pandas Series to inspect.
            sample_size: Number of values to sample for pattern matching.

        Returns:
            True if a majority of sampled values match date-like patterns.
        """
        if series.empty:
            return False
        try:
            sample = series.sample(min(len(series), sample_size), random_state=42)
            str_vals = sample.astype(str)
            match_count = sum(1 for v in str_vals if self._DATE_VALUE_PATTERN.match(v.strip()))
            return (match_count / len(str_vals)) >= 0.7
        except Exception:
            return False

    # =========================================================================
    # 2. Detailed Column Profiling
    # =========================================================================
    def generate_column_profile(
        self,
        df: pd.DataFrame,
        classifications: Optional[Dict[str, ColumnCategory]] = None,
    ) -> List[Dict[str, Any]]:
        """Calculate statistical metrics for every column in the dataset.

        Enhanced with:
        - Percentile values (25th, 50th, 75th) for numerical columns
        - Skewness and kurtosis for numerical columns
        - Complete category percentages for categorical columns
        """
        if df is None or df.empty:
            return []

        if classifications is None:
            classifications = self.identify_column_types(df)

        profiles: List[Dict[str, Any]] = []
        n_rows = len(df)

        for col in df.columns:
            series = df[col]
            non_null = series.dropna()
            non_null_count = int(len(non_null))
            missing_count = int(series.isna().sum())
            missing_pct = round((missing_count / n_rows) * 100, 2) if n_rows > 0 else 0.0
            n_unique = int(non_null.nunique())
            duplicate_values_count = max(0, non_null_count - n_unique)
            category = classifications.get(str(col), ColumnCategory.CATEGORICAL)

            # Compute mode (most frequent value)
            mode_val = None
            if not non_null.empty:
                mode_series = non_null.mode()
                if not mode_series.empty:
                    mode_val = mode_series.iloc[0]

            col_prof: Dict[str, Any] = {
                "column_name": str(col),
                "inferred_category": category.value if hasattr(category, "value") else str(category),
                "raw_dtype": str(series.dtype),
                "total_count": n_rows,
                "non_null_count": non_null_count,
                "missing_count": missing_count,
                "missing_percentage": missing_pct,
                "unique_values": n_unique,
                "duplicate_values": duplicate_values_count,
                "mode": str(mode_val) if mode_val is not None else "N/A",
                "min": None,
                "max": None,
                "mean": None,
                "median": None,
                "std": None,
                "percentile_25": None,
                "percentile_50": None,
                "percentile_75": None,
                "skewness": None,
                "kurtosis": None,
                "top_categories": [],
            }

            # Numerical statistics (enhanced with percentiles, skewness, kurtosis)
            if category in (ColumnCategory.NUMERICAL_CONTINUOUS, ColumnCategory.NUMERICAL_DISCRETE) or pd.api.types.is_numeric_dtype(series):
                if not non_null.empty:
                    try:
                        numeric_s = pd.to_numeric(non_null, errors="coerce").dropna()
                        if not numeric_s.empty:
                            col_prof["min"] = float(numeric_s.min())
                            col_prof["max"] = float(numeric_s.max())
                            col_prof["mean"] = round(float(numeric_s.mean()), 4)
                            col_prof["median"] = round(float(numeric_s.median()), 4)
                            col_prof["std"] = round(float(numeric_s.std(ddof=1)), 4) if len(numeric_s) > 1 else 0.0

                            # Percentiles
                            col_prof["percentile_25"] = round(float(numeric_s.quantile(0.25)), 4)
                            col_prof["percentile_50"] = round(float(numeric_s.quantile(0.50)), 4)
                            col_prof["percentile_75"] = round(float(numeric_s.quantile(0.75)), 4)

                            # Skewness and Kurtosis (require at least 3 values for meaningful computation)
                            if len(numeric_s) >= 3:
                                col_prof["skewness"] = round(float(numeric_s.skew()), 4)
                                col_prof["kurtosis"] = round(float(numeric_s.kurtosis()), 4)
                    except Exception:
                        pass

            # Categorical statistics & frequency table (enhanced with complete percentages)
            if category in (ColumnCategory.CATEGORICAL, ColumnCategory.BOOLEAN, ColumnCategory.TEXT, ColumnCategory.CONSTANT) or not pd.api.types.is_numeric_dtype(series):
                if not non_null.empty:
                    top_counts = non_null.value_counts().head(10)
                    col_prof["top_categories"] = [
                        {
                            "category": str(k),
                            "count": int(v),
                            "percentage": round((v / non_null_count) * 100, 2) if non_null_count > 0 else 0.0,
                        }
                        for k, v in top_counts.items()
                    ]

            # Datetime statistics
            if category == ColumnCategory.DATETIME:
                try:
                    dt_series = pd.to_datetime(non_null, errors="coerce").dropna()
                    if not dt_series.empty:
                        col_prof["min"] = str(dt_series.min())
                        col_prof["max"] = str(dt_series.max())
                except Exception:
                    pass

            profiles.append(col_prof)

        return profiles

    # =========================================================================
    # 3. Data Quality Analysis
    # =========================================================================
    def audit_data_quality(
        self,
        df: pd.DataFrame,
        classifications: Optional[Dict[str, ColumnCategory]] = None,
    ) -> Dict[str, Any]:
        """Perform comprehensive data quality inspection and return warnings & scores.

        Enhanced with:
        - Detection of invalid date values in datetime columns
        - Per-column missing value breakdown (top 5 columns with most missing)
        - Empty column detection (100% null)
        - Data quality score explanation breakdown
        """
        if df is None or df.empty:
            return {
                "total_records": 0,
                "total_columns": 0,
                "total_cells": 0,
                "total_missing_cells": 0,
                "missing_percentage": 0.0,
                "duplicate_rows": 0,
                "duplicate_percentage": 0.0,
                "empty_columns": [],
                "constant_columns": [],
                "identifier_columns": [],
                "high_cardinality_columns": [],
                "inconsistent_type_columns": [],
                "quality_score": 0.0,
                "quality_score_breakdown": {},
                "missing_value_breakdown": [],
                "invalid_date_columns": [],
                "warnings": ["Dataset is empty."],
            }

        if classifications is None:
            classifications = self.identify_column_types(df)

        n_rows = len(df)
        n_cols = len(df.columns)
        total_cells = n_rows * n_cols

        # 1. Missing cells
        total_missing = int(df.isna().sum().sum())
        missing_pct = round((total_missing / total_cells) * 100, 2) if total_cells > 0 else 0.0

        # 2. Duplicate rows
        dup_rows_count = int(df.duplicated().sum())
        dup_pct = round((dup_rows_count / n_rows) * 100, 2) if n_rows > 0 else 0.0

        # 3. Empty columns (100% missing)
        empty_cols = [str(c) for c in df.columns if df[c].isna().all()]

        # 4. Constant columns (1 unique non-null value)
        constant_cols = [str(c) for c in df.columns if df[c].dropna().nunique() <= 1]

        # 5. Identifier columns
        identifier_cols = [
            c for c, cat in classifications.items() if cat == ColumnCategory.IDENTIFIER
        ]

        # 6. High cardinality columns
        high_card_cols = [
            c for c, cat in classifications.items() if cat == ColumnCategory.TEXT
        ]

        # 7. Inconsistent/mixed data types
        inconsistent_cols: List[str] = []
        for c in df.columns:
            if df[c].dtype == object and df[c].dropna().size > 0:
                types_in_col = {type(val).__name__ for val in df[c].dropna().sample(min(len(df[c].dropna()), 50))}
                if len(types_in_col) > 1:
                    inconsistent_cols.append(str(c))

        # 8. Per-column missing value breakdown (top 5 columns with most missing)
        missing_per_col = df.isna().sum()
        missing_per_col = missing_per_col[missing_per_col > 0].sort_values(ascending=False)
        missing_value_breakdown: List[Dict[str, Any]] = []
        for col_name, miss_count in missing_per_col.head(5).items():
            miss_count_int = int(miss_count)
            miss_pct_col = round((miss_count_int / n_rows) * 100, 2) if n_rows > 0 else 0.0
            missing_value_breakdown.append({
                "column": str(col_name),
                "missing_count": miss_count_int,
                "missing_percentage": miss_pct_col,
            })

        # 9. Detection of invalid date values in datetime columns
        invalid_date_columns: List[Dict[str, Any]] = []
        datetime_cols = [
            c for c, cat in classifications.items() if cat == ColumnCategory.DATETIME
        ]
        for dt_col in datetime_cols:
            if dt_col in df.columns:
                col_series = df[dt_col]
                non_null_vals = col_series.dropna()
                if not non_null_vals.empty:
                    try:
                        converted = pd.to_datetime(non_null_vals, errors="coerce")
                        invalid_count = int(converted.isna().sum())
                        if invalid_count > 0:
                            invalid_pct = round((invalid_count / len(non_null_vals)) * 100, 2)
                            invalid_date_columns.append({
                                "column": dt_col,
                                "invalid_count": invalid_count,
                                "invalid_percentage": invalid_pct,
                                "total_non_null": len(non_null_vals),
                            })
                    except Exception:
                        pass

        # 10. Generate Quality Warnings
        warnings: List[str] = []
        if total_missing > 0:
            cols_with_missing = [c for c in df.columns if df[c].isna().sum() > 0]
            warnings.append(
                f"Missing Values: Found {total_missing:,} missing cells ({missing_pct}%) across {len(cols_with_missing)} columns."
            )
        if dup_rows_count > 0:
            warnings.append(
                f"Duplicate Records: Found {dup_rows_count:,} exact duplicate rows ({dup_pct}% of dataset)."
            )
        if empty_cols:
            warnings.append(
                f"Empty Columns: {len(empty_cols)} column(s) have 100% missing data ({', '.join(empty_cols)})."
            )
        if constant_cols:
            warnings.append(
                f"Constant Columns: {len(constant_cols)} column(s) have zero variance ({', '.join(constant_cols)})."
            )
        if high_card_cols:
            warnings.append(
                f"High-Cardinality Features: {len(high_card_cols)} column(s) contain distinct text ({', '.join(high_card_cols)})."
            )
        if inconsistent_cols:
            warnings.append(
                f"Mixed Types: {len(inconsistent_cols)} column(s) contain mixed object types ({', '.join(inconsistent_cols)})."
            )
        if invalid_date_columns:
            invalid_names = [entry["column"] for entry in invalid_date_columns]
            warnings.append(
                f"Invalid Dates: {len(invalid_date_columns)} datetime column(s) contain unparseable date values ({', '.join(invalid_names)})."
            )

        if not warnings:
            warnings.append("Clean Dataset: No missing values, duplicates, or constant columns detected.")

        # Compute composite quality score (0 - 100)
        # Deduct penalties for missingness, duplicates, empty columns
        penalty_missing = missing_pct * 0.4
        penalty_duplicates = dup_pct * 0.3
        penalty_empty_cols = len(empty_cols) * 5
        penalty_constant_cols = len(constant_cols) * 2
        penalty_invalid_dates = len(invalid_date_columns) * 3
        penalty_inconsistent = len(inconsistent_cols) * 2

        total_penalty = (
            penalty_missing
            + penalty_duplicates
            + penalty_empty_cols
            + penalty_constant_cols
            + penalty_invalid_dates
            + penalty_inconsistent
        )
        quality_score = max(0.0, min(100.0, round(100.0 - total_penalty, 1)))

        # Quality score explanation breakdown
        quality_score_breakdown: Dict[str, Any] = {
            "base_score": 100.0,
            "penalty_missing_values": round(penalty_missing, 2),
            "penalty_duplicate_rows": round(penalty_duplicates, 2),
            "penalty_empty_columns": round(penalty_empty_cols, 2),
            "penalty_constant_columns": round(penalty_constant_cols, 2),
            "penalty_invalid_dates": round(penalty_invalid_dates, 2),
            "penalty_inconsistent_types": round(penalty_inconsistent, 2),
            "total_penalty": round(total_penalty, 2),
            "final_score": quality_score,
        }

        return {
            "total_records": n_rows,
            "total_columns": n_cols,
            "total_cells": total_cells,
            "total_missing_cells": total_missing,
            "missing_percentage": missing_pct,
            "duplicate_rows": dup_rows_count,
            "duplicate_percentage": dup_pct,
            "empty_columns": empty_cols,
            "constant_columns": constant_cols,
            "identifier_columns": identifier_cols,
            "high_cardinality_columns": high_card_cols,
            "inconsistent_type_columns": inconsistent_cols,
            "quality_score": quality_score,
            "quality_score_breakdown": quality_score_breakdown,
            "missing_value_breakdown": missing_value_breakdown,
            "invalid_date_columns": invalid_date_columns,
            "warnings": warnings,
        }

    # =========================================================================
    # 4. Target Variable Analysis
    # =========================================================================
    def inspect_target_variable(self, df: pd.DataFrame, target_col: str) -> Dict[str, Any]:
        """Inspect the selected target variable and extract outcomes and distributions."""
        if df is None or target_col not in df.columns:
            return {"valid": False, "error": f"Target column '{target_col}' not found."}

        series = df[target_col].dropna()
        if series.empty:
            return {"valid": False, "error": f"Target column '{target_col}' has no valid values."}

        is_numeric = pd.api.types.is_numeric_dtype(series)
        unique_vals = series.unique()
        n_unique = len(unique_vals)

        if n_unique < 2:
            return {
                "valid": False,
                "error": f"Target column '{target_col}' has only 1 distinct value. Variation is required.",
            }

        # If numeric but only 2 unique values, it's effectively a binary target
        target_type = "Categorical"
        if is_numeric and n_unique > 10:
            target_type = "Numerical Continuous"
        elif is_numeric and n_unique <= 10:
            target_type = "Numerical Discrete"

        # Outcomes mapping
        possible_outcomes: List[Dict[str, Any]] = []
        if target_type.startswith("Numerical"):
            mean_val = float(series.mean())
            median_val = float(series.median())
            q25 = float(series.quantile(0.25))
            q75 = float(series.quantile(0.75))

            possible_outcomes = [
                {
                    "label": f"Above Median (> {median_val:.2f})",
                    "type": "threshold",
                    "operator": ">",
                    "value": median_val,
                },
                {
                    "label": f"Below Median (<= {median_val:.2f})",
                    "type": "threshold",
                    "operator": "<=",
                    "value": median_val,
                },
                {
                    "label": f"Above Mean (> {mean_val:.2f})",
                    "type": "threshold",
                    "operator": ">",
                    "value": mean_val,
                },
                {
                    "label": f"High Anomaly (Upper Quartile > {q75:.2f})",
                    "type": "threshold",
                    "operator": ">",
                    "value": q75,
                },
                {
                    "label": f"Low Anomaly (Lower Quartile < {q25:.2f})",
                    "type": "threshold",
                    "operator": "<",
                    "value": q25,
                },
                {
                    "label": "Continuous Numeric Values (Direct)",
                    "type": "continuous",
                    "operator": "direct",
                    "value": None,
                },
            ]
            distribution = {
                "min": float(series.min()),
                "max": float(series.max()),
                "mean": round(mean_val, 4),
                "median": round(median_val, 4),
                "std": round(float(series.std()), 4) if len(series) > 1 else 0.0,
                "values": series.tolist(),
            }
        else:
            val_counts = series.value_counts()
            for val, cnt in val_counts.items():
                pct = round((cnt / len(series)) * 100, 2)
                possible_outcomes.append(
                    {
                        "label": f"Category: '{val}' ({cnt:,} records, {pct}%)",
                        "type": "class",
                        "operator": "==",
                        "value": val,
                        "count": int(cnt),
                        "percentage": pct,
                    }
                )
            distribution = {
                "categories": [str(k) for k in val_counts.index],
                "counts": [int(v) for v in val_counts.values],
                "percentages": [round((v / len(series)) * 100, 2) for v in val_counts.values],
            }

        return {
            "valid": True,
            "target_col": target_col,
            "target_type": target_type,
            "unique_count": n_unique,
            "total_records": len(series),
            "missing_records": int(df[target_col].isna().sum()),
            "possible_outcomes": possible_outcomes,
            "distribution": distribution,
        }

    # =========================================================================
    # 5. Automatic Factual Dataset Insights
    # =========================================================================
    def generate_factual_insights(
        self,
        df: pd.DataFrame,
        classifications: Optional[Dict[str, ColumnCategory]] = None,
        target_info: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        """Generate strictly factual, data-driven observations from the actual dataset.

        Enhanced with:
        - Insights about datetime columns (date ranges, spans)
        - Insights about high-cardinality columns
        - Insights about boolean columns
        - More specific column-type counts (e.g. '3 continuous, 2 discrete numerical')
        """
        if df is None or df.empty:
            return ["Dataset is empty. No analytical insights can be calculated."]

        if classifications is None:
            classifications = self.identify_column_types(df)

        insights: List[str] = []
        n_rows, n_cols = df.shape
        insights.append(f"The dataset contains **{n_rows:,} records** and **{n_cols} columns**.")

        # Breakdown by column types
        type_counts: Dict[str, int] = {}
        for cat in classifications.values():
            label = cat.value if hasattr(cat, "value") else str(cat)
            type_counts[label] = type_counts.get(label, 0) + 1

        types_summary_parts = [f"**{count} {name}**" for name, count in sorted(type_counts.items())]
        insights.append(f"Column classification identifies {', '.join(types_summary_parts)}.")

        # Enhanced: Specific numerical breakdown (continuous vs discrete)
        n_continuous = type_counts.get(ColumnCategory.NUMERICAL_CONTINUOUS.value, 0)
        n_discrete = type_counts.get(ColumnCategory.NUMERICAL_DISCRETE.value, 0)
        if n_continuous + n_discrete > 0:
            numerical_parts: List[str] = []
            if n_continuous > 0:
                numerical_parts.append(f"{n_continuous} continuous")
            if n_discrete > 0:
                numerical_parts.append(f"{n_discrete} discrete")
            insights.append(
                f"Among numerical columns: **{', '.join(numerical_parts)}** "
                f"({n_continuous + n_discrete} total numerical features)."
            )

        # Missing values observations
        total_missing = int(df.isna().sum().sum())
        total_cells = n_rows * n_cols
        if total_missing == 0:
            insights.append("There are **0 missing cells** across the entire dataset (100% data completeness).")
        else:
            missing_pct = round((total_missing / total_cells) * 100, 2)
            cols_missing = df.columns[df.isna().any()].tolist()
            worst_col = df.isna().sum().idxmax()
            worst_col_count = int(df[worst_col].isna().sum())
            worst_col_pct = round((worst_col_count / n_rows) * 100, 1)
            insights.append(
                f"The dataset contains **{total_missing:,} missing cells** ({missing_pct}% of total values) "
                f"across {len(cols_missing)} column(s). Highest missingness is in `'{worst_col}'` "
                f"with {worst_col_count:,} missing records ({worst_col_pct}%)."
            )

        # Duplicate records observations
        dup_count = int(df.duplicated().sum())
        if dup_count > 0:
            dup_pct = round((dup_count / n_rows) * 100, 2)
            insights.append(
                f"Detected **{dup_count:,} duplicate records** ({dup_pct}% of rows). Duplicate deduplication "
                "may be evaluated in subsequent investigation phases."
            )
        else:
            insights.append("All rows are distinct (**0 duplicate records** detected).")

        # Constant or High Cardinality columns
        constant_cols = [c for c, cat in classifications.items() if cat == ColumnCategory.CONSTANT]
        if constant_cols:
            insights.append(
                f"Found **{len(constant_cols)} constant column(s)** ({', '.join(constant_cols)}) "
                "with zero variance across all records."
            )

        id_cols = [c for c, cat in classifications.items() if cat == ColumnCategory.IDENTIFIER]
        if id_cols:
            insights.append(
                f"Identified **{len(id_cols)} potential identifier/key column(s)** ({', '.join(id_cols)}) "
                "which should typically be excluded from statistical correlation."
            )

        # Enhanced: Boolean column insights
        bool_cols = [c for c, cat in classifications.items() if cat == ColumnCategory.BOOLEAN]
        if bool_cols:
            bool_details: List[str] = []
            for bc in bool_cols:
                if bc in df.columns:
                    val_counts = df[bc].dropna().value_counts()
                    if len(val_counts) == 2:
                        dominant = val_counts.index[0]
                        dominant_pct = round((val_counts.iloc[0] / val_counts.sum()) * 100, 1)
                        bool_details.append(f"`'{bc}'` ({dominant_pct}% = {dominant})")
                    else:
                        bool_details.append(f"`'{bc}'`")
            insights.append(
                f"Found **{len(bool_cols)} boolean/binary column(s)**: {', '.join(bool_details)}."
            )

        # Enhanced: High-cardinality column insights
        high_card_cols = [c for c, cat in classifications.items() if cat == ColumnCategory.TEXT]
        if high_card_cols:
            hc_details: List[str] = []
            for hc in high_card_cols:
                if hc in df.columns:
                    n_unique_hc = df[hc].dropna().nunique()
                    hc_details.append(f"`'{hc}'` ({n_unique_hc:,} unique values)")
            insights.append(
                f"Detected **{len(high_card_cols)} high-cardinality text column(s)**: "
                f"{', '.join(hc_details)}. These may need encoding or exclusion for modelling."
            )

        # Enhanced: Datetime column insights
        datetime_cols = [c for c, cat in classifications.items() if cat == ColumnCategory.DATETIME]
        if datetime_cols:
            dt_details: List[str] = []
            for dtc in datetime_cols:
                if dtc in df.columns:
                    try:
                        dt_series = pd.to_datetime(df[dtc], errors="coerce").dropna()
                        if not dt_series.empty:
                            dt_min = dt_series.min()
                            dt_max = dt_series.max()
                            span = dt_max - dt_min
                            dt_details.append(
                                f"`'{dtc}'` spans from {dt_min.strftime('%Y-%m-%d')} to "
                                f"{dt_max.strftime('%Y-%m-%d')} ({span.days:,} days)"
                            )
                        else:
                            dt_details.append(f"`'{dtc}'` (no valid dates parsed)")
                    except Exception:
                        dt_details.append(f"`'{dtc}'`")
            insights.append(
                f"Found **{len(datetime_cols)} datetime column(s)**: {', '.join(dt_details)}."
            )

        # Target variable observations if selected
        if target_info and target_info.get("valid"):
            target_col = target_info["target_col"]
            target_type = target_info["target_type"]
            n_unique = target_info["unique_count"]
            if "distribution" in target_info and "categories" in target_info["distribution"]:
                cats = target_info["distribution"]["categories"]
                pcts = target_info["distribution"]["percentages"]
                cat_summary = ", ".join([f"'{c}': {p}%" for c, p in zip(cats[:4], pcts[:4])])
                insights.append(
                    f"Selected target variable `'{target_col}'` is a **{target_type}** with "
                    f"**{n_unique} unique classes** ({cat_summary})."
                )
            elif "distribution" in target_info and "mean" in target_info["distribution"]:
                dist = target_info["distribution"]
                insights.append(
                    f"Selected target variable `'{target_col}'` is a **{target_type}** ranging from "
                    f"**{dist['min']:.2f}** to **{dist['max']:.2f}** (mean: {dist['mean']:.2f}, median: {dist['median']:.2f}, std: {dist['std']:.2f})."
                )

        return insights

    # =========================================================================
    # Full Profile Orchestrator
    # =========================================================================
    def generate_profile(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate a complete comprehensive profile dictionary."""
        if df is None or df.empty:
            raise ValueError("Cannot profile an empty or None DataFrame.")

        classifications = self.identify_column_types(df)
        column_profiles = self.generate_column_profile(df, classifications)
        quality_audit = self.audit_data_quality(df, classifications)
        insights = self.generate_factual_insights(df, classifications)

        return {
            "shape": df.shape,
            "rows_count": df.shape[0],
            "cols_count": df.shape[1],
            "classifications": classifications,
            "column_profiles": column_profiles,
            "quality_audit": quality_audit,
            "insights": insights,
        }
