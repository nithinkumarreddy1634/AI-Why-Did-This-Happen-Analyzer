"""Unified master facade coordinating report data collection, narrative generation, and exports."""

from typing import Dict, Any, List, Optional, Tuple
import uuid
from datetime import datetime
import pandas as pd

from modules.reporting.report_data_collector import CollectedReportData, ReportDataCollector
from modules.reporting.narrative_generator import ReportNarrativeSections, ReportNarrativeGenerator
from modules.reporting.pdf_generator import PDFReportGenerator
from modules.reporting.docx_generator import DOCXReportGenerator
from modules.reporting.html_generator import HTMLReportGenerator
from modules.reporting.json_generator import JSONReportGenerator
from modules.reporting.report_history import ReportRecord, ReportHistoryRepository


class AutomatedReportManager:
    """Master orchestrator for multi-format investigation report generation."""

    def __init__(self) -> None:
        self.collector = ReportDataCollector()
        self.narrative_gen = ReportNarrativeGenerator()
        self.pdf_gen = PDFReportGenerator()
        self.docx_gen = DOCXReportGenerator()
        self.html_gen = HTMLReportGenerator()
        self.json_gen = JSONReportGenerator()
        self.history = ReportHistoryRepository()

    def assemble_report(
        self,
        df: pd.DataFrame,
        dataset_name: str = "dataset.csv",
        active_investigation_pkg: Optional[Any] = None,
        ml_importance_res: Optional[Any] = None,
        ml_comparison_res: Optional[Any] = None,
        shap_global_res: Optional[Any] = None,
        lime_active_res: Optional[Any] = None,
        detection_artifacts: Optional[Any] = None,
        preprocessing_pipeline: Optional[Any] = None,
        title: str = "Automated Investigation Report",
        subtitle: str = "Multi-Engine Root Cause & Predictive Diagnostic Dossier",
        organization: str = "Analytics & Governance Group",
        author: str = "AI Why Did This Happen Analyzer",
        confidentiality: str = "Internal Business Sensitive",
    ) -> Tuple[CollectedReportData, ReportNarrativeSections]:
        """Collect data and generate deterministic narrative sections."""
        collected = self.collector.collect_from_environment(
            df=df,
            dataset_name=dataset_name,
            active_investigation_pkg=active_investigation_pkg,
            ml_importance_res=ml_importance_res,
            ml_comparison_res=ml_comparison_res,
            shap_global_res=shap_global_res,
            lime_active_res=lime_active_res,
            detection_artifacts=detection_artifacts,
            preprocessing_pipeline=preprocessing_pipeline,
            title=title,
            subtitle=subtitle,
            organization=organization,
            author=author,
            confidentiality=confidentiality,
        )
        narratives = self.narrative_gen.generate_narrative(collected)
        return collected, narratives

    def generate_pdf(
        self,
        data: CollectedReportData,
        narratives: ReportNarrativeSections,
        included_sections: Optional[List[str]] = None,
    ) -> bytes:
        """Render ReportLab PDF binary."""
        return self.pdf_gen.generate_pdf(data, narratives, included_sections)

    def generate_docx(
        self,
        data: CollectedReportData,
        narratives: ReportNarrativeSections,
        included_sections: Optional[List[str]] = None,
    ) -> bytes:
        """Render Word DOCX binary."""
        return self.docx_gen.generate_docx(data, narratives, included_sections)

    def generate_html(
        self,
        data: CollectedReportData,
        narratives: ReportNarrativeSections,
        included_sections: Optional[List[str]] = None,
    ) -> str:
        """Render HTML string."""
        return self.html_gen.generate_html(data, narratives, included_sections)

    def generate_json(
        self,
        data: CollectedReportData,
        narratives: ReportNarrativeSections,
        included_sections: Optional[List[str]] = None,
    ) -> str:
        """Render JSON string."""
        return self.json_gen.generate_json(data, narratives, included_sections)

    def log_generated_report(
        self,
        title: str,
        investigation_id: str,
        dataset_name: str,
        dataset_fingerprint: str,
        formats: List[str],
    ) -> ReportRecord:
        """Register a successfully exported report in history."""
        rep_id = f"rep_{uuid.uuid4().hex[:8]}"
        rec = ReportRecord(
            report_id=rep_id,
            title=title,
            investigation_id=investigation_id,
            dataset_name=dataset_name,
            dataset_fingerprint=dataset_fingerprint,
            generation_date=datetime.now().strftime("%Y-%m-%d %H:%M"),
            formats=formats,
        )
        self.history.save_report(rec)
        return rec

