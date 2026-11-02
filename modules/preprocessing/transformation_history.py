"""Transformation history tracking for preprocessing pipeline reproducibility."""

from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, Any, List, Optional, Union
import uuid
import pandas as pd


@dataclass
class TransformationStep:
    """Represents a discrete transformation operation performed on a dataset."""
    operation_name: str
    column_affected: Union[str, List[str]]
    method_selected: str
    parameters_used: Dict[str, Any] = field(default_factory=dict)
    records_affected: int = 0
    status: str = "Success"
    details: str = ""
    step_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    execution_time_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert step to dictionary."""
        return asdict(self)


# Backward compatibility alias
TransformationRecord = TransformationStep


class TransformationHistory:
    """Maintains an auditable, reproducible log of data transformations."""

    def __init__(self) -> None:
        self.steps: List[TransformationStep] = []
        self._snapshots: List[pd.DataFrame] = []

    def record_step(
        self,
        operation_name: str,
        column_affected: Union[str, List[str]],
        method_selected: str,
        parameters_used: Optional[Dict[str, Any]] = None,
        records_affected: int = 0,
        status: str = "Success",
        details: str = "",
        execution_time_ms: float = 0.0,
        snapshot: Optional[pd.DataFrame] = None,
    ) -> TransformationStep:
        """Create and append a transformation step to the log."""
        step = TransformationStep(
            operation_name=operation_name,
            column_affected=column_affected,
            method_selected=method_selected,
            parameters_used=parameters_used or {},
            records_affected=records_affected,
            status=status,
            details=details,
            execution_time_ms=execution_time_ms,
        )
        self.steps.append(step)
        if snapshot is not None:
            self._snapshots.append(snapshot.copy())
        return step

    def get_log_dataframe(self) -> pd.DataFrame:
        """Convert history of steps into a clean pandas DataFrame for display."""
        if not self.steps:
            return pd.DataFrame(
                columns=[
                    "Step #",
                    "Timestamp",
                    "Operation",
                    "Column(s)",
                    "Method",
                    "Affected Records",
                    "Status",
                    "Details",
                ]
            )

        data = []
        for idx, step in enumerate(self.steps, start=1):
            cols = (
                ", ".join(step.column_affected)
                if isinstance(step.column_affected, list)
                else str(step.column_affected)
            )
            data.append(
                {
                    "Step #": idx,
                    "Timestamp": step.timestamp,
                    "Operation": step.operation_name,
                    "Column(s)": cols,
                    "Method": step.method_selected,
                    "Affected Records": step.records_affected,
                    "Status": step.status,
                    "Details": step.details,
                }
            )
        return pd.DataFrame(data)

    def undo_last_step(self) -> Optional[pd.DataFrame]:
        """Pop the last step and return the preceding snapshot if available."""
        if not self.steps:
            return None
        self.steps.pop()
        if self._snapshots:
            return self._snapshots.pop()
        return None

    def clear(self) -> None:
        """Reset the transformation history and snapshots."""
        self.steps.clear()
        self._snapshots.clear()

    def __len__(self) -> int:
        return len(self.steps)

    def to_dict_list(self) -> List[Dict[str, Any]]:
        """Return steps as list of dicts for export/serialization."""
        return [step.to_dict() for step in self.steps]
