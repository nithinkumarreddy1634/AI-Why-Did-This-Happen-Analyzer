"""Data type inspection, validation, and safe conversion."""

from typing import Dict, Any, List, Optional, Tuple
import re
import pandas as pd
import numpy as np


def inspect_data_types(df: pd.DataFrame) -> pd.DataFrame:
    """Inspect current data types and detect potential casting recommendations.
    
    Returns a DataFrame with columns:
    ['column', 'current_dtype', 'sample_values', 'recommended_type', 'confidence', 'reason']
    """
    records = []

    for col in df.columns:
        series = df[col]
        non_null_series = series.dropna()
        current_dtype = str(series.dtype)
        sample_vals = ", ".join(str(v) for v in non_null_series.head(3).tolist())
        
        recommended = current_dtype
        confidence = "Current"
        reason = "Appropriate data type"

        if len(non_null_series) > 0 and (pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)):
            str_vals = non_null_series.astype(str).str.strip()
            
            # 1. Check if values are boolean strings ('true'/'false', 'yes'/'no', 'y'/'n', '0'/'1')
            bool_set = {"true", "false", "yes", "no", "y", "n", "1", "0"}
            if set(str_vals.str.lower().unique()).issubset(bool_set):
                recommended = "boolean"
                confidence = "High"
                reason = "Values are boolean indicators (e.g. true/false or yes/no)"

            # 2. Check if clean integers
            elif str_vals.str.match(r"^-?\d+$").all():
                recommended = "integer"
                confidence = "High"
                reason = "All non-null values are whole integer numbers"

            # 3. Check if clean floats or numbers with comma/currency
            elif str_vals.str.match(r"^-?\$?\d+([.,]\d+)?%?$").all():
                recommended = "float"
                confidence = "High"
                reason = "Values represent numeric measurements/currency"

            # 4. Check if date / datetime format
            else:
                sample_test = str_vals.head(20)
                try:
                    parsed_dates = pd.to_datetime(sample_test, errors="coerce")
                    if parsed_dates.notna().sum() >= len(sample_test) * 0.8:
                        # Check if dates have time component
                        has_time = any(d.time() != pd.Timestamp("00:00:00").time() for d in parsed_dates.dropna())
                        recommended = "datetime" if has_time else "date"
                        confidence = "Medium"
                        reason = f"Values follow calendar {'datetime' if has_time else 'date'} patterns"
                except Exception:
                    pass

        records.append(
            {
                "column": col,
                "current_dtype": current_dtype,
                "sample_values": sample_vals if sample_vals else "[All Empty]",
                "recommended_type": recommended,
                "confidence": confidence,
                "reason": reason,
            }
        )

    return pd.DataFrame(records)


def convert_column_type(
    df: pd.DataFrame,
    column: str,
    target_type: str,
    date_format: Optional[str] = None,
) -> Tuple[pd.DataFrame, bool, int, str]:
    """Safely convert a column to a specified target data type.
    
    Supported target types:
    - 'integer': Nullable integer (Int64)
    - 'float': Standard 64-bit float
    - 'string': String/text representation
    - 'boolean': Boolean (nullable boolean)
    - 'date': Date without time component
    - 'datetime': Standard datetime64[ns]

    Returns:
    (new_df, success_flag, failed_conversions_count, message)
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' does not exist.")

    new_df = df.copy()
    original_series = new_df[column]
    orig_dtype = str(original_series.dtype)
    target_type = target_type.lower()

    try:
        if target_type in ("integer", "int", "int64"):
            # Strip currency symbols, commas, or percent signs if text
            if pd.api.types.is_object_dtype(original_series) or pd.api.types.is_string_dtype(original_series):
                cleaned = (
                    original_series.astype(str)
                    .str.replace(r"[$,% ]", "", regex=True)
                    .replace(["nan", "None", ""], np.nan)
                )
            else:
                cleaned = original_series
            
            # Numeric conversion with coerce
            numeric_vals = pd.to_numeric(cleaned, errors="coerce")
            failures = int((original_series.notna() & numeric_vals.isna()).sum())
            new_df[column] = numeric_vals.round().astype("Int64")
            msg = f"Converted '{column}' from {orig_dtype} to Nullable Integer (Int64)."
            if failures > 0:
                msg += f" Note: {failures:,} invalid value(s) could not be parsed and were coerced to Null."
            return new_df, True, failures, msg

        elif target_type in ("float", "float64"):
            if pd.api.types.is_object_dtype(original_series) or pd.api.types.is_string_dtype(original_series):
                cleaned = (
                    original_series.astype(str)
                    .str.replace(r"[$,% ]", "", regex=True)
                    .replace(["nan", "None", ""], np.nan)
                )
            else:
                cleaned = original_series

            numeric_vals = pd.to_numeric(cleaned, errors="coerce")
            failures = int((original_series.notna() & numeric_vals.isna()).sum())
            new_df[column] = numeric_vals.astype(float)
            msg = f"Converted '{column}' from {orig_dtype} to Float64."
            if failures > 0:
                msg += f" Note: {failures:,} invalid value(s) could not be parsed and were coerced to NaN."
            return new_df, True, failures, msg

        elif target_type in ("string", "str", "object"):
            new_df[column] = original_series.astype(str)
            # Restore null representation
            new_df.loc[original_series.isna(), column] = np.nan
            return new_df, True, 0, f"Converted '{column}' from {orig_dtype} to String."

        elif target_type in ("boolean", "bool"):
            bool_map = {
                "true": True, "1": True, "1.0": True, "t": True, "yes": True, "y": True,
                "false": False, "0": False, "0.0": False, "f": False, "no": False, "n": False,
            }
            str_series = original_series.astype(str).str.strip().str.lower()
            mapped = str_series.map(bool_map)
            failures = int((original_series.notna() & mapped.isna()).sum())
            new_df[column] = mapped.astype("boolean")
            msg = f"Converted '{column}' from {orig_dtype} to Boolean."
            if failures > 0:
                msg += f" Note: {failures:,} unmapped value(s) were coerced to Null."
            return new_df, True, failures, msg

        elif target_type in ("datetime", "datetime64"):
            parsed = pd.to_datetime(original_series, format=date_format, errors="coerce")
            failures = int((original_series.notna() & parsed.isna()).sum())
            new_df[column] = parsed
            msg = f"Converted '{column}' from {orig_dtype} to Datetime."
            if failures > 0:
                msg += f" Note: {failures:,} unparseable date values coerced to NaT."
            return new_df, True, failures, msg

        elif target_type == "date":
            parsed = pd.to_datetime(original_series, format=date_format, errors="coerce")
            failures = int((original_series.notna() & parsed.isna()).sum())
            new_df[column] = parsed.dt.date
            msg = f"Converted '{column}' from {orig_dtype} to Date."
            if failures > 0:
                msg += f" Note: {failures:,} unparseable date values coerced to None."
            return new_df, True, failures, msg

        else:
            return df.copy(), False, 0, f"Unsupported target type: '{target_type}'."

    except Exception as e:
        # Preserve original values when conversion fails
        return df.copy(), False, len(df), f"Conversion failed: {str(e)}. Original values preserved."
