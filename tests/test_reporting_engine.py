"""Unit and integration tests for Phase 13: Automated Investigation Report Generation Engine."""

import pytest
import json
import numpy as np
import pandas as pd

from modules.reporting.report_data_collector import ReportDataCollector, CollectedReportData
from modules.reporting.narrative_generator import ReportNarrativeGenerator, ReportNarrativeSections
from modules.reporting.pdf_generator import PDFReportGenerator
from modules.reporting.docx_generator import DOCXReportGenerator
from modules.reporting.html_generator import HTMLReportGenerator
from modules.reporting.json_generator import JSONReportGenerator
from modules.reporting.report_history import ReportRecord, ReportHistoryRepository
from modules.reporting.report_manager import AutomatedReportManager
from modules.root_cause.investigation_manager import RootCauseInvestigationManager


@pytest.fixture
def sample_report_environment():
    """Build a realistic dataset and active root-cause investigation package."""
    np.random.seed(42)
    df = pd.DataFrame({
        "contract": ["Month-to-month"] * 30 + ["Two year"] * 30,
        "tenure": np.concatenate([np.random.normal(12, 3, 30), np.random.normal(48, 8, 30)]),
        "monthly_charges": np.concatenate([np.random.normal(85, 10, 30), np.random.normal(50, 8, 30)]),
        "churn": [1] * 22 + [0] * 8 + [0] * 27 + [1] * 3,
    })

    manager = RootCauseInvestigationManager()
    pkg = manager.execute_investigation(
        df=df,
        target_col="churn",
        baseline_mask=(df["contract"] == "Two year"),
        comparison_mask=(df["contract"] == "Month-to-month"),
        title="Sample Churn Investigation",
        dataset_name="telecom_sample.csv",
        candidate_features=["tenure", "monthly_charges"],
        baseline_name="Two-Year Group",
        comparison_name="Month-to-Month Group",
    )
    return df, pkg


def test_report_data_collection(sample_report_environment):
    df, pkg = sample_report_environment
    collector = ReportDataCollector()

    collected = collector.collect_from_environment(
        df=df,
        dataset_name="telecom_sample.csv",
        active_investigation_pkg=pkg,
        title="Quarterly Churn Audit",
        organization="Data Intelligence Division",
        author="Lead Risk Analyst",
    )

    assert isinstance(collected, CollectedReportData)
    assert collected.report_title == "Quarterly Churn Audit"
    assert collected.target_variable == "churn"
    assert collected.dataset_shape == (60, 4)
    assert collected.has_root_cause is True
    assert collected.has_statistical is True
    assert collected.has_dataset_overview is True
    assert collected.root_cause_data["top_candidate"] in ["tenure", "monthly_charges"]


def test_narrative_generator(sample_report_environment):
    df, pkg = sample_report_environment
    collector = ReportDataCollector()
    collected = collector.collect_from_environment(df=df, active_investigation_pkg=pkg)

    generator = ReportNarrativeGenerator()
    narratives = generator.generate_narrative(collected)

    assert isinstance(narratives, ReportNarrativeSections)
    assert "churn" in narratives.executive_summary
    assert len(narratives.recommendations_list) >= 3
    assert len(narratives.limitations_list) >= 3
    # Check that non-causal disclaimer is present
    assert any("causality" in lim.lower() for lim in narratives.limitations_list)


def test_pdf_generation(sample_report_environment):
    df, pkg = sample_report_environment
    collector = ReportDataCollector()
    collected = collector.collect_from_environment(df=df, active_investigation_pkg=pkg)
    narratives = ReportNarrativeGenerator().generate_narrative(collected)

    pdf_gen = PDFReportGenerator()
    pdf_bytes = pdf_gen.generate_pdf(collected, narratives)

    assert isinstance(pdf_bytes, bytes)
    assert len(pdf_bytes) > 1000
    # Standard PDF magic header
    assert pdf_bytes.startswith(b"%PDF-")


def test_docx_generation(sample_report_environment):
    df, pkg = sample_report_environment
    collector = ReportDataCollector()
    collected = collector.collect_from_environment(df=df, active_investigation_pkg=pkg)
    narratives = ReportNarrativeGenerator().generate_narrative(collected)

    docx_gen = DOCXReportGenerator()
    docx_bytes = docx_gen.generate_docx(collected, narratives)

    assert isinstance(docx_bytes, bytes)
    assert len(docx_bytes) > 1000
    # Standard ZIP/DOCX magic header
    assert docx_bytes.startswith(b"PK")


def test_html_generation(sample_report_environment):
    df, pkg = sample_report_environment
    collector = ReportDataCollector()
    collected = collector.collect_from_environment(df=df, active_investigation_pkg=pkg)
    narratives = ReportNarrativeGenerator().generate_narrative(collected)

    html_gen = HTMLReportGenerator()
    html_str = html_gen.generate_html(collected, narratives)

    assert isinstance(html_str, str)
    assert "<!DOCTYPE html>" in html_str
    assert "Executive Summary" in html_str
    assert "Analytical Governance Notice" in html_str
    assert "churn" in html_str


def test_json_generation(sample_report_environment):
    df, pkg = sample_report_environment
    collector = ReportDataCollector()
    collected = collector.collect_from_environment(df=df, active_investigation_pkg=pkg)
    narratives = ReportNarrativeGenerator().generate_narrative(collected)

    json_gen = JSONReportGenerator()
    json_str = json_gen.generate_json(collected, narratives)

    assert isinstance(json_str, str)
    parsed = json.loads(json_str)
    assert "metadata" in parsed
    assert "investigation_scope" in parsed
    assert "narratives" in parsed
    assert "disclaimer" in parsed
    assert parsed["investigation_scope"]["target_variable"] == "churn"


def test_report_history_repository():
    repo = ReportHistoryRepository()
    record = ReportRecord(
        report_id="rep_test1",
        title="Q3 Risk Dossier",
        investigation_id="inv_test",
        dataset_name="data.csv",
        dataset_fingerprint="abc1234",
        generation_date="2026-11-02 18:00",
        formats=["PDF", "DOCX"],
    )
    repo.save_report(record)

    assert repo.get_report("rep_test1") is not None
    listed = repo.list_reports(search_query="Risk")
    assert len(listed) == 1
    assert listed[0].title == "Q3 Risk Dossier"

    repo.delete_report("rep_test1")
    assert repo.get_report("rep_test1") is None


def test_automated_report_manager_facade(sample_report_environment):
    df, pkg = sample_report_environment
    manager = AutomatedReportManager()

    collected, narratives = manager.assemble_report(
        df=df,
        active_investigation_pkg=pkg,
        title="End-to-End Test Report",
    )

    pdf_bytes = manager.generate_pdf(collected, narratives)
    docx_bytes = manager.generate_docx(collected, narratives)
    html_str = manager.generate_html(collected, narratives)
    json_str = manager.generate_json(collected, narratives)

    assert len(pdf_bytes) > 0
    assert len(docx_bytes) > 0
    assert len(html_str) > 0
    assert len(json_str) > 0

    rec = manager.log_generated_report(
        title="End-to-End Test Report",
        investigation_id="inv_sample",
        dataset_name="telecom.csv",
        dataset_fingerprint="fp1234",
        formats=["PDF", "DOCX", "HTML", "JSON"],
    )
    assert rec.report_id.startswith("rep_")

