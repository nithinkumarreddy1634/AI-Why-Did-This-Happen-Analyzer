"""Report history tracking, archiving, and staleness detection repository."""

from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, Any, List, Optional


@dataclass
class ReportRecord:
    """Archived report record with full traceability metadata."""
    report_id: str
    title: str
    investigation_id: str
    dataset_name: str
    dataset_fingerprint: str
    generation_date: str
    formats: List[str]
    is_stale: bool = False
    stale_reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert record to dictionary."""
        return asdict(self)


class ReportHistoryRepository:
    """In-memory report catalog for browsing, downloading, and auditing generated dossiers."""

    def __init__(self) -> None:
        self._reports: Dict[str, ReportRecord] = {}

    def save_report(self, record: ReportRecord) -> None:
        """Register a generated report in history."""
        self._reports[record.report_id] = record

    def get_report(self, report_id: str) -> Optional[ReportRecord]:
        """Retrieve report metadata by ID."""
        return self._reports.get(report_id)

    def list_reports(
        self,
        search_query: Optional[str] = None,
        investigation_id: Optional[str] = None,
    ) -> List[ReportRecord]:
        """List reports sorted descending by generation timestamp, with optional search."""
        all_reps = list(self._reports.values())
        if investigation_id:
            all_reps = [r for r in all_reps if r.investigation_id == investigation_id]
        if search_query:
            q = search_query.lower()
            all_reps = [r for r in all_reps if q in r.title.lower() or q in r.dataset_name.lower()]
        return sorted(all_reps, key=lambda x: x.generation_date, reverse=True)

    def delete_report(self, report_id: str) -> bool:
        """Delete report from history."""
        if report_id in self._reports:
            del self._reports[report_id]
            return True
        return False

    def clear(self) -> None:
        """Clear all report history."""
        self._reports.clear()

