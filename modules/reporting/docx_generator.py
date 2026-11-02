"""Editable Microsoft Word (.docx) report generator using python-docx."""

import io
from typing import Dict, Any, List, Optional
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT

from modules.reporting.report_data_collector import CollectedReportData
from modules.reporting.narrative_generator import ReportNarrativeSections


class DOCXReportGenerator:
    """Builds editable, styled Microsoft Word (.docx) investigation dossiers."""

    def generate_docx(
        self,
        data: CollectedReportData,
        narratives: ReportNarrativeSections,
        included_sections: Optional[List[str]] = None,
    ) -> bytes:
        """Render complete DOCX document and return raw bytes."""
        doc = docx.Document()

        # Page Margins (1 inch = 72 pt)
        sections = doc.sections
        for s in sections:
            s.top_margin = Inches(1.0)
            s.bottom_margin = Inches(1.0)
            s.left_margin = Inches(1.0)
            s.right_margin = Inches(1.0)

        inc = included_sections or ["all"]
        def should_include(sec: str) -> bool:
            return "all" in inc or sec in inc

        # ==============================================================================
        # 1. Cover & Title Header
        # ==============================================================================
        brand_p = doc.add_paragraph()
        brand_run = brand_p.add_run("AI \"WHY DID THIS HAPPEN?\" ANALYZER")
        brand_run.bold = True
        brand_run.font.size = Pt(11)
        brand_run.font.color.rgb = RGBColor(79, 70, 229)  # Indigo
        brand_p.paragraph_format.space_after = Pt(4)

        title_p = doc.add_paragraph()
        title_run = title_p.add_run(data.report_title)
        title_run.bold = True
        title_run.font.size = Pt(22)
        title_run.font.color.rgb = RGBColor(15, 23, 42)  # Dark slate
        title_p.paragraph_format.space_after = Pt(4)

        sub_p = doc.add_paragraph()
        sub_run = sub_p.add_run(data.report_subtitle)
        sub_run.font.size = Pt(12)
        sub_run.font.color.rgb = RGBColor(100, 116, 139)
        sub_p.paragraph_format.space_after = Pt(16)

        # Metadata Table
        meta_table = doc.add_table(rows=7, cols=2)
        meta_table.alignment = WD_TABLE_ALIGNMENT.CENTER
        meta_data = [
            ("Target Variable", f"{data.target_variable} ({data.outcome_type})"),
            ("Dataset Name", data.dataset_name),
            ("Records / Features", f"{data.dataset_shape[0]:,} rows × {data.dataset_shape[1]} columns"),
            ("Baseline Cohort", data.baseline_cohort_name),
            ("Comparison Cohort", data.comparison_cohort_name),
            ("Observed Shift (Δ)", f"{data.outcome_delta:+.4f}" + (f" ({data.outcome_rel_pct:+.1f}%)" if data.outcome_rel_pct is not None else "")),
            ("Reporting Date", data.generation_date),
        ]
        for idx, (lbl, val) in enumerate(meta_data):
            row = meta_table.rows[idx]
            r0 = row.cells[0].paragraphs[0].add_run(lbl)
            r0.bold = True
            r0.font.size = Pt(9.5)
            r1 = row.cells[1].paragraphs[0].add_run(val)
            r1.font.size = Pt(9.5)

        doc.add_paragraph().paragraph_format.space_after = Pt(12)

        # Governance Notice
        callout_p = doc.add_paragraph()
        callout_run = callout_p.add_run(
            "GOVERNANCE NOTICE: This document records observational analytical evidence and predictive model attributions. "
            "Findings reflect empirical associations, not confirmed counterfactual causality."
        )
        callout_run.italic = True
        callout_run.font.size = Pt(9)
        callout_run.font.color.rgb = RGBColor(67, 56, 202)
        callout_p.paragraph_format.space_after = Pt(20)

        # ==============================================================================
        # 2. Executive Summary
        # ==============================================================================
        if should_include("executive_summary"):
            h1 = doc.add_heading("1. Executive Summary", level=1)
            h1.paragraph_format.space_before = Pt(14)
            p = doc.add_paragraph(narratives.executive_summary)
            p.paragraph_format.space_after = Pt(10)

        # ==============================================================================
        # 3. Investigation Objective
        # ==============================================================================
        if should_include("investigation_objective"):
            h2 = doc.add_heading("2. Investigation Objective & Scope", level=1)
            h2.paragraph_format.space_before = Pt(14)
            p = doc.add_paragraph(narratives.investigation_objective)
            p.paragraph_format.space_after = Pt(10)

        # ==============================================================================
        # 4. Dataset Overview & Data Quality
        # ==============================================================================
        if should_include("dataset_overview"):
            h3 = doc.add_heading("3. Dataset Overview & Data Quality", level=1)
            h3.paragraph_format.space_before = Pt(14)
            doc.add_paragraph(narratives.dataset_overview_narrative)
            doc.add_paragraph(narratives.data_quality_narrative)

        # ==============================================================================
        # 5. Statistical Investigation
        # ==============================================================================
        if should_include("statistical_investigation") and data.has_statistical:
            h4 = doc.add_heading("4. Statistical Investigation & Hypothesis Testing", level=1)
            h4.paragraph_format.space_before = Pt(14)
            doc.add_paragraph(narratives.statistical_narrative)

        # ==============================================================================
        # 6. Machine Learning & Explainable AI
        # ==============================================================================
        if should_include("ml_and_xai") and (data.has_ml or data.has_shap or data.has_lime):
            h5 = doc.add_heading("5. Machine Learning & Explainability (SHAP / LIME)", level=1)
            h5.paragraph_format.space_before = Pt(14)
            if data.has_ml:
                doc.add_paragraph(narratives.ml_narrative)
            if data.has_shap or data.has_lime:
                doc.add_paragraph(narratives.shap_lime_narrative)

        # ==============================================================================
        # 7. Root Cause Priority Ranking
        # ==============================================================================
        if should_include("root_cause_ranking") and data.has_root_cause:
            h6 = doc.add_heading("6. Root Cause Candidate Priority Ranking", level=1)
            h6.paragraph_format.space_before = Pt(14)
            doc.add_paragraph(narratives.root_cause_narrative)

            cands = data.root_cause_data.get("ranked_candidates", [])
            if cands:
                tbl = doc.add_table(rows=len(cands[:6]) + 1, cols=4)
                tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
                headers = ["Rank", "Candidate Feature", "Score", "Evidence Strength"]
                for i, h in enumerate(headers):
                    r = tbl.rows[0].cells[i].paragraphs[0].add_run(h)
                    r.bold = True
                    r.font.size = Pt(9)
                for idx, c in enumerate(cands[:6], start=1):
                    row = tbl.rows[idx]
                    row.cells[0].paragraphs[0].add_run(f"#{c.get('rank', idx)}").font.size = Pt(9)
                    row.cells[1].paragraphs[0].add_run(c.get("feature", "N/A")).font.size = Pt(9)
                    row.cells[2].paragraphs[0].add_run(f"{c.get('overall_score', 0.0):.3f}").font.size = Pt(9)
                    row.cells[3].paragraphs[0].add_run(c.get("evidence_strength", "N/A")).font.size = Pt(9)

        # ==============================================================================
        # 8. Segment Decomposition
        # ==============================================================================
        if should_include("segment_analysis") and data.has_segment_analysis:
            h7 = doc.add_heading("7. Segment Contribution Decomposition", level=1)
            h7.paragraph_format.space_before = Pt(14)
            doc.add_paragraph(narratives.segment_narrative)

        # ==============================================================================
        # 9. Recommendations & Next Steps
        # ==============================================================================
        if should_include("recommendations"):
            h8 = doc.add_heading("8. Evidence-Based Recommendations & Next Steps", level=1)
            h8.paragraph_format.space_before = Pt(14)
            for rec in narratives.recommendations_list:
                p = doc.add_paragraph(style="List Bullet")
                r_cat = p.add_run(f"[{rec['category']}] ")
                r_cat.bold = True
                p.add_run(f"{rec['recommendation']} (Basis: {rec['supporting_finding']})")

        # ==============================================================================
        # 10. Analytical Limitations & Caveats
        # ==============================================================================
        if should_include("limitations"):
            h9 = doc.add_heading("9. Analytical Assumptions & Limitations", level=1)
            h9.paragraph_format.space_before = Pt(14)
            for lim in narratives.limitations_list:
                doc.add_paragraph(lim, style="List Bullet")

        buffer = io.BytesIO()
        doc.save(buffer)
        buffer.seek(0)
        return buffer.getvalue()

