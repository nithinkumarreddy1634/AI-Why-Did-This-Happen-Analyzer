"""Investigation metadata, traceability, dataset fingerprinting, and staleness detection."""

import hashlib
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd


def compute_dataset_fingerprint(df: pd.DataFrame, filename: str = "dataset") -> str:
    """Generate a deterministic SHA256 fingerprint from dataframe shape, columns, and sample hash."""
    if df is None or df.empty:
        return "empty_dataset"
    col_str = "_".join(str(c) for c in df.columns)
    dtype_str = "_".join(str(t) for t in df.dtypes)
    shape_str = f"{df.shape[0]}x{df.shape[1]}"
    raw = f"{filename}_{shape_str}_{col_str}_{dtype_str}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def check_dataset_staleness(
    current_df: Optional[pd.DataFrame],
    stored_fingerprint: str,
    current_filename: str = "dataset",
) -> Tuple[bool, str]:
    """Verify if analytical evidence is stale due to dataset modifications."""
    if current_df is None or current_df.empty:
        return True, "Active dataset is missing or empty."
    current_fp = compute_dataset_fingerprint(current_df, current_filename)
    if current_fp != stored_fingerprint:
        return True, (
            f"Dataset fingerprint changed (stored: {stored_fingerprint}, current: {current_fp}). "
            "Underlying data was modified, filtered, or reloaded."
        )
    return False, "Evidence is synchronized with active dataset."


@dataclass
class InvestigationMetadata:
    """Traceable metadata record for a root-cause investigation."""
    investigation_id: str
    title: str
    description: str
    created_at: str
    updated_at: str
    dataset_name: str
    dataset_fingerprint: str
    target_variable: str
    outcome_type: str  # 'numerical' or 'categorical'
    baseline_definition: Dict[str, Any]
    comparison_definition: Dict[str, Any]
    investigation_question: str
    status: str = "Active"  # 'Active', 'Archived', 'Draft'
    notes: str = ""
    is_stale: bool = False
    stale_reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "InvestigationMetadata":
        """Reconstruct metadata from dictionary."""
        return cls(**data)


class InvestigationHistory:
    """Manages an in-memory repository of investigations with traceability."""

    def __init__(self) -> None:
        self._investigations: Dict[str, InvestigationMetadata] = {}

    def register(self, metadata: InvestigationMetadata) -> None:
        """Register or update an investigation record."""
        self._investigations[metadata.investigation_id] = metadata

    def get(self, investigation_id: str) -> Optional[InvestigationMetadata]:
        """Retrieve an investigation by ID."""
        return self._investigations.get(investigation_id)

    def list_all(self) -> List[InvestigationMetadata]:
        """List all investigations ordered by latest updated."""
        return sorted(
            self._investigations.values(),
            key=lambda x: x.updated_at,
            reverse=True,
        )

    def delete(self, investigation_id: str) -> bool:
        """Remove an investigation record."""
        if investigation_id in self._investigations:
            del self._investigations[investigation_id]
            return True
        return False

    def clear(self) -> None:
        """Clear all stored investigations."""
        self._investigations.clear()

