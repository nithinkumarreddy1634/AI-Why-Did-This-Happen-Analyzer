"""Dataset export utilities for CSV and Excel formats."""

import io
from datetime import datetime
from typing import Tuple
import pandas as pd


def export_processed_dataset(
    df: pd.DataFrame,
    format_type: str = "csv",
    base_name: str = "cleaned_dataset",
) -> Tuple[bytes, str, str]:
    """Serializes DataFrame to downloadable byte buffer.
    
    Args:
        df: Processed DataFrame.
        format_type: 'csv' or 'excel'.
        base_name: Base prefix for generated filename.

    Returns:
        (bytes_content, filename, mime_type)
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    clean_base = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in base_name).strip("_")
    if not clean_base:
        clean_base = "cleaned_dataset"

    format_type = format_type.lower()
    if format_type in ("csv", "text/csv"):
        filename = f"{clean_base}_{timestamp}.csv"
        mime = "text/csv"
        # Export with utf-8 encoding
        buffer = io.BytesIO()
        df.to_csv(buffer, index=False, encoding="utf-8")
        return buffer.getvalue(), filename, mime

    elif format_type in ("excel", "xlsx", "openpyxl"):
        filename = f"{clean_base}_{timestamp}.xlsx"
        mime = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="Cleaned Data")
        return buffer.getvalue(), filename, mime

    else:
        raise ValueError(f"Unsupported export format: '{format_type}' (use 'csv' or 'excel').")
