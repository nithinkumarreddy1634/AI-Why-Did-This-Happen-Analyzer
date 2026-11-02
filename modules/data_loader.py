"""Data loading and ingestion module for tabular datasets."""

import io
import time
from pathlib import Path
from typing import Any, BinaryIO, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from config.constants import SUPPORTED_EXTENSIONS, MAX_FILE_SIZE_MB
from utils.validators import validate_file_extension, validate_file_size, validate_dataset


class DataLoader:
    """Handles dataset ingestion, validation, multi-encoding fallback, and multi-format parsing."""

    CSV_ENCODINGS = ["utf-8", "utf-8-sig", "latin1", "cp1252", "iso-8859-1"]
    CSV_DELIMITERS = [",", ";", "\t", "|"]

    def __init__(
        self,
        max_file_size_mb: int = MAX_FILE_SIZE_MB,
        allowed_extensions: Optional[List[str]] = None,
    ) -> None:
        self.max_file_size_mb = max_file_size_mb
        self.allowed_extensions = allowed_extensions or SUPPORTED_EXTENSIONS

    def load_file(
        self,
        file_source: Union[str, Path, BinaryIO, io.BytesIO],
        file_name: Optional[str] = None,
        return_metadata: bool = True,
    ) -> Union[Tuple[pd.DataFrame, Dict[str, Any]], pd.DataFrame]:
        """Load tabular data from file path or uploaded buffer.
        
        Args:
            file_source: Path or binary stream.
            file_name: Optional explicit file name.
            return_metadata: If True, returns (df, metadata). If False, returns df.
            
        Returns:
            Tuple of (DataFrame, metadata) if return_metadata=True, else DataFrame.
        """
        start_time = time.perf_counter()

        # 1. Resolve file name and size
        resolved_name = self._resolve_filename(file_source, file_name)
        file_bytes_size = self._resolve_size(file_source)

        # 2. Pre-parsing validations
        is_valid_ext, ext_msg = validate_file_extension(resolved_name, self.allowed_extensions)
        if not is_valid_ext:
            raise ValueError(ext_msg)

        # Check empty or oversized file
        if file_bytes_size == 0 and not isinstance(file_source, (str, Path)):
            raw = self._get_raw_bytes(file_source)
            if not raw or len(raw.strip()) == 0:
                raise ValueError(f"The uploaded file '{resolved_name}' is empty (0 bytes).")
            file_bytes_size = len(raw)

        if file_bytes_size > 0:
            is_valid_size, size_msg = validate_file_size(file_bytes_size, self.max_file_size_mb)
            if not is_valid_size:
                raise ValueError(size_msg)

        suffix = Path(resolved_name).suffix.lower()
        encoding_used: Optional[str] = None
        delimiter_used: Optional[str] = None

        # 3. Read Data
        if suffix == ".csv":
            df, encoding_used, delimiter_used = self._read_csv_safely(file_source)
        elif suffix in (".xlsx", ".xls"):
            df = self._read_excel_safely(file_source)
        elif suffix == ".parquet":
            df = self._read_parquet_safely(file_source)
        else:
            raise ValueError(f"Unsupported file extension: '{suffix}'.")

        # 4. DataFrame Structural Validation
        is_valid_df, df_msg = validate_dataset(df)
        if not is_valid_df:
            raise ValueError(df_msg)

        # Clean column names
        df.columns = [str(c).strip() for c in df.columns]

        # 5. Metadata compilation
        elapsed_time = round(time.perf_counter() - start_time, 3)
        metadata = {
            "filename": resolved_name,
            "file_extension": suffix,
            "file_size_bytes": file_bytes_size,
            "rows_count": int(df.shape[0]),
            "cols_count": int(df.shape[1]),
            "column_names": list(df.columns),
            "memory_bytes": int(df.memory_usage(deep=True).sum()),
            "load_duration_seconds": elapsed_time,
            "encoding_used": encoding_used,
            "delimiter_used": delimiter_used,
        }

        if return_metadata:
            return df, metadata
        return df

    def _resolve_filename(
        self,
        file_source: Union[str, Path, BinaryIO, io.BytesIO],
        file_name: Optional[str],
    ) -> str:
        """Extract filename from path or object."""
        if file_name:
            return file_name
        if isinstance(file_source, (str, Path)):
            p = Path(file_source)
            if not p.exists():
                raise FileNotFoundError(f"File not found: {file_source}")
            return p.name
        if hasattr(file_source, "name") and file_source.name:
            return str(file_source.name)
        return "uploaded_file.csv"

    def _resolve_size(self, file_source: Union[str, Path, BinaryIO, io.BytesIO]) -> int:
        """Extract size in bytes safely."""
        if isinstance(file_source, (str, Path)):
            p = Path(file_source)
            return p.stat().st_size if p.exists() else 0
        if hasattr(file_source, "size"):
            return getattr(file_source, "size", 0)
        if hasattr(file_source, "getbuffer"):
            return file_source.getbuffer().nbytes
        return 0

    def _get_raw_bytes(self, file_source: Union[str, Path, BinaryIO, io.BytesIO]) -> bytes:
        """Extract all raw bytes safely."""
        if isinstance(file_source, (str, Path)):
            with open(file_source, "rb") as f:
                return f.read()
        if hasattr(file_source, "getvalue"):
            return file_source.getvalue()
        if hasattr(file_source, "read"):
            pos = file_source.tell() if hasattr(file_source, "tell") else None
            data = file_source.read()
            if pos is not None and hasattr(file_source, "seek"):
                file_source.seek(pos)
            return data
        raise ValueError("Cannot extract bytes from file source.")

    def _read_csv_safely(
        self, file_source: Union[str, Path, BinaryIO, io.BytesIO]
    ) -> Tuple[pd.DataFrame, str, str]:
        """Safely read CSV files with encoding and delimiter fallbacks."""
        raw_bytes = self._get_raw_bytes(file_source)
        if not raw_bytes or len(raw_bytes.strip()) == 0:
            raise ValueError("The uploaded CSV file is empty (0 bytes).")

        last_error = None

        # Pass 1: Try delimiter sniffing with csv.Sniffer
        for enc in self.CSV_ENCODINGS:
            try:
                import csv
                sample_str = raw_bytes[:4096].decode(enc, errors="ignore")
                sniffed_delim = csv.Sniffer().sniff(sample_str, delimiters=",;\t|").delimiter
                stream = io.BytesIO(raw_bytes)
                df = pd.read_csv(stream, encoding=enc, sep=sniffed_delim, engine="python")
                if df is not None and df.shape[1] > 1:
                    return df, enc, sniffed_delim
            except Exception as e:
                last_error = e

        # Pass 2: Explicit delimiters with encodings
        for enc in self.CSV_ENCODINGS:
            for delim in self.CSV_DELIMITERS:
                try:
                    stream = io.BytesIO(raw_bytes)
                    df = pd.read_csv(stream, encoding=enc, sep=delim, engine="python")
                    if df is not None and df.shape[1] > 1:
                        return df, enc, delim
                except Exception as e:
                    last_error = e

        # Pass 3: Default fallback
        try:
            stream = io.BytesIO(raw_bytes)
            df = pd.read_csv(stream, encoding="utf-8", on_bad_lines="skip")
            if df is not None and not df.empty:
                return df, "utf-8", ","
        except Exception as e:
            last_error = e

        raise ValueError(f"Could not parse CSV file. Error details: {str(last_error)}")

    def _read_excel_safely(
        self, file_source: Union[str, Path, BinaryIO, io.BytesIO]
    ) -> pd.DataFrame:
        """Safely read Excel (.xlsx, .xls) files."""
        raw_bytes = self._get_raw_bytes(file_source)
        if not raw_bytes or len(raw_bytes) == 0:
            raise ValueError("The uploaded Excel file is empty.")

        stream = io.BytesIO(raw_bytes)
        try:
            df = pd.read_excel(stream, engine="openpyxl")
            return df
        except Exception:
            try:
                stream.seek(0)
                df = pd.read_excel(stream)
                return df
            except Exception as ex:
                raise ValueError(
                    f"Corrupted or invalid Excel file. Please ensure it is a valid .xlsx or .xls sheet: {str(ex)}"
                ) from ex

    def _read_parquet_safely(
        self, file_source: Union[str, Path, BinaryIO, io.BytesIO]
    ) -> pd.DataFrame:
        """Safely read Parquet files."""
        raw_bytes = self._get_raw_bytes(file_source)
        if not raw_bytes or len(raw_bytes) == 0:
            raise ValueError("The uploaded Parquet file is empty.")

        stream = io.BytesIO(raw_bytes)
        df = pd.read_parquet(stream)
        return df

    def create_benchmark_messy_dataset(self) -> pd.DataFrame:
        """Generate a realistic benchmark dataset exhibiting diverse data quality defects."""
        return generate_benchmark_dirty_dataset()


def generate_benchmark_dirty_dataset() -> pd.DataFrame:
    """Generate a realistic benchmark dataset exhibiting various data quality issues."""
    np.random.seed(42)
    n = 200

    customer_ids = [f"CUST_{1000 + i}" for i in range(n)]
    ages = np.random.normal(loc=40, scale=12, size=n).round().tolist()
    ages[5] = 135.0
    ages[18] = -5.0
    ages[45] = np.nan
    ages[88] = np.nan

    genders = np.random.choice(["Male", "male", "MALE", "Female", "female", "Other"], size=n).tolist()
    genders[10] = " Male "
    genders[25] = "Female "
    genders[33] = np.nan

    monthly_charges = np.random.normal(loc=70, scale=25, size=n).round(2).tolist()
    monthly_charges[12] = 950.0
    monthly_charges[77] = np.nan

    contract_types = np.random.choice(
        ["Month-to-month", "One year", "Two year", "Ultra_Rare_VIP"],
        p=[0.60, 0.25, 0.13, 0.02],
        size=n,
    ).tolist()

    signup_dates = [f"2023-{np.random.randint(1, 13):02d}-{np.random.randint(1, 28):02d}" for _ in range(n)]
    signup_dates[15] = "Invalid-Date-String"

    constant_col = ["Active_Platform"] * n
    tenure_months = np.random.randint(1, 72, size=n).tolist()

    churn = (
        (np.array(monthly_charges) > 80)
        & (np.array(tenure_months) < 12)
    ).astype(int).tolist()

    df = pd.DataFrame(
        {
            "customer_id": customer_ids,
            "Age": ages,
            "Gender": genders,
            "MonthlyCharges": monthly_charges,
            "Contract": contract_types,
            "SignupDate": signup_dates,
            "PlatformStatus": constant_col,
            "TenureMonths": tenure_months,
            "Churn": churn,
        }
    )

    dupes = df.iloc[:5].copy()
    df = pd.concat([df, dupes], ignore_index=True)
    return df
