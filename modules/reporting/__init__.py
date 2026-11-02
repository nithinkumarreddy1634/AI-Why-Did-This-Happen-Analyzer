"""Automated Investigation Report Generation Engine package (Phase 13)."""

from modules.reporting.report_data_collector import CollectedReportData, ReportDataCollector
from modules.reporting.narrative_generator import ReportNarrativeSections, ReportNarrativeGenerator
from modules.reporting.pdf_generator import PDFReportGenerator
from modules.reporting.docx_generator import DOCXReportGenerator
from modules.reporting.html_generator import HTMLReportGenerator
from modules.reporting.json_generator import JSONReportGenerator
from modules.reporting.report_history import ReportRecord, ReportHistoryRepository
from modules.reporting.report_manager import AutomatedReportManager

__all__ = [
    "CollectedReportData",
    "ReportDataCollector",
    "ReportNarrativeSections",
    "ReportNarrativeGenerator",
    "PDFReportGenerator",
    "DOCXReportGenerator",
    "HTMLReportGenerator",
    "JSONReportGenerator",
    "ReportRecord",
    "ReportHistoryRepository",
    "AutomatedReportManager",
]
