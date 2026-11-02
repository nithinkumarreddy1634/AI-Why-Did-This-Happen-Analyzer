"""Professional PDF report generator using ReportLab with cover page, tables, and running headers."""

import io
from typing import Dict, Any, List, Optional
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak,
    KeepTogether,
    HRFlowable,
)
from reportlab.pdfgen import canvas

from modules.reporting.report_data_collector import CollectedReportData
from modules.reporting.narrative_generator import ReportNarrativeSections


class NumberedCanvas(canvas.Canvas):
    """Two-pass canvas to dynamically compute and stamp 'Page X of Y' and running headers."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._saved_page_states: List[Dict[str, Any]] = []

    def showPage(self) -> None:
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self) -> None:
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count: int) -> None:
        if self._pageNumber == 1:
            # Suppress header/footer on cover page
            return

        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#64748B"))

        # Running header
        self.drawString(54, 750, "AI \"Why Did This Happen?\" Analyzer — Automated Investigation Report")
        self.setStrokeColor(colors.HexColor("#CBD5E1"))
        self.setLineWidth(0.5)
        self.line(54, 742, 558, 742)

        # Running footer
        self.line(54, 45, 558, 45)
        page_str = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(558, 32, page_str)
        self.drawString(54, 32, "Confidential — Observational Analytical Findings Only")
        self.restoreState()


class PDFReportGenerator:
    """Builds structured executive PDF reports with ReportLab."""

    def generate_pdf(
        self,
        data: CollectedReportData,
        narratives: ReportNarrativeSections,
        included_sections: Optional[List[str]] = None,
    ) -> bytes:
        """Render complete PDF report document and return raw bytes."""
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=letter,
            leftMargin=54,
            rightMargin=54,
            topMargin=54,
            bottomMargin=54,
        )

        styles = getSampleStyleSheet()

        # Custom typography styles
        title_style = ParagraphStyle(
            "CoverTitle",
            parent=styles["Normal"],
            fontName="Helvetica-Bold",
            fontSize=26,
            leading=32,
            textColor=colors.HexColor("#0F172A"),
            spaceAfter=8,
        )
        subtitle_style = ParagraphStyle(
            "CoverSubtitle",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=13,
            leading=18,
            textColor=colors.HexColor("#475569"),
            spaceAfter=20,
        )
        h1_style = ParagraphStyle(
            "ReportH1",
            parent=styles["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=16,
            leading=20,
            textColor=colors.HexColor("#1E293B"),
            spaceBefore=16,
            spaceAfter=8,
        )
        h2_style = ParagraphStyle(
            "ReportH2",
            parent=styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=12,
            leading=16,
            textColor=colors.HexColor("#334155"),
            spaceBefore=10,
            spaceAfter=4,
        )
        body_style = ParagraphStyle(
            "ReportBody",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=9.5,
            leading=14,
            textColor=colors.HexColor("#1E293B"),
            spaceAfter=8,
        )
        meta_label = ParagraphStyle(
            "MetaLabel",
            parent=styles["Normal"],
            fontName="Helvetica-Bold",
            fontSize=9,
            leading=12,
            textColor=colors.HexColor("#475569"),
        )
        meta_val = ParagraphStyle(
            "MetaVal",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=9,
            leading=12,
            textColor=colors.HexColor("#0F172A"),
        )
        callout_style = ParagraphStyle(
            "Callout",
            parent=styles["Normal"],
            fontName="Helvetica-Oblique",
            fontSize=8.5,
            leading=12,
            textColor=colors.HexColor("#4338CA"),
        )

        story: List[Any] = []

        # ==============================================================================
        # 1. Cover Page
        # ==============================================================================
        story.append(Spacer(1, 40))
        story.append(Paragraph("AI \"WHY DID THIS HAPPEN?\" ANALYZER", ParagraphStyle("Brand", fontName="Helvetica-Bold", fontSize=11, textColor=colors.HexColor("#4F46E5"), spaceAfter=14)))
        story.append(Paragraph(data.report_title, title_style))
        story.append(Paragraph(data.report_subtitle, subtitle_style))
        story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor("#4F46E5"), spaceAfter=25))

        # Metadata Card
        meta_table_data = [
            [Paragraph("Target Variable:", meta_label), Paragraph(f"<b>{data.target_variable}</b> ({data.outcome_type})", meta_val)],
            [Paragraph("Dataset Name:", meta_label), Paragraph(data.dataset_name, meta_val)],
            [Paragraph("Dataset Records:", meta_label), Paragraph(f"{data.dataset_shape[0]:,} rows × {data.dataset_shape[1]} columns", meta_val)],
            [Paragraph("Reference Baseline:", meta_label), Paragraph(data.baseline_cohort_name, meta_val)],
            [Paragraph("Affected Comparison:", meta_label), Paragraph(data.comparison_cohort_name, meta_val)],
            [Paragraph("Outcome Shift (Δ):", meta_label), Paragraph(f"<b>{data.outcome_delta:+.4f}</b>" + (f" ({data.outcome_rel_pct:+.1f}%)" if data.outcome_rel_pct is not None else ""), meta_val)],
            [Paragraph("Reporting Date:", meta_label), Paragraph(data.generation_date, meta_val)],
            [Paragraph("Classification:", meta_label), Paragraph(data.confidentiality, meta_val)],
        ]
        meta_table = Table(meta_table_data, colWidths=[140, 364])
        meta_table.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F8FAFC")),
                ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#E2E8F0")),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ("LEFTPADDING", (0, 0), (-1, -1), 10),
                ("RIGHTPADDING", (0, 0), (-1, -1), 10),
            ])
        )
        story.append(meta_table)

        story.append(Spacer(1, 30))
        # Non-causal notice on cover
        notice_data = [
            [Paragraph("<b>GOVERNANCE NOTICE:</b> This dossier documents empirical associations and predictive model attributions. Observational data cannot establish unconfounded causality. All ranked factors represent candidate hypotheses for validation.", callout_style)]
        ]
        notice_table = Table(notice_data, colWidths=[504])
        notice_table.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#EEF2FF")),
                ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#C7D2FE")),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                ("LEFTPADDING", (0, 0), (-1, -1), 12),
                ("RIGHTPADDING", (0, 0), (-1, -1), 12),
            ])
        )
        story.append(notice_table)

        story.append(PageBreak())

        # ==============================================================================
        # 2. Executive Summary
        # ==============================================================================
        inc = included_sections or ["all"]
        def should_include(sec: str) -> bool:
            return "all" in inc or sec in inc

        if should_include("executive_summary"):
            story.append(Paragraph("1. Executive Summary", h1_style))
            story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#CBD5E1"), spaceAfter=10))
            story.append(Paragraph(narratives.executive_summary, body_style))
            story.append(Spacer(1, 10))

        # ==============================================================================
        # 3. Investigation Objective & Scope
        # ==============================================================================
        if should_include("investigation_objective"):
            story.append(Paragraph("2. Investigation Objective & Scope", h1_style))
            story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#CBD5E1"), spaceAfter=10))
            story.append(Paragraph(narratives.investigation_objective, body_style))
            story.append(Spacer(1, 10))

        # ==============================================================================
        # 4. Dataset Overview & Data Quality
        # ==============================================================================
        if should_include("dataset_overview"):
            story.append(Paragraph("3. Dataset Overview & Data Health", h1_style))
            story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#CBD5E1"), spaceAfter=10))
            story.append(Paragraph(narratives.dataset_overview_narrative, body_style))
            story.append(Paragraph(narratives.data_quality_narrative, body_style))
            story.append(Spacer(1, 10))

        # ==============================================================================
        # 5. Statistical Investigation
        # ==============================================================================
        if should_include("statistical_investigation") and data.has_statistical:
            story.append(Paragraph("4. Statistical Investigation & Hypothesis Testing", h1_style))
            story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#CBD5E1"), spaceAfter=10))
            story.append(Paragraph(narratives.statistical_narrative, body_style))

            top_sigs = data.statistical_data.get("top_significant", [])
            if top_sigs:
                t_rows = [[Paragraph("<b>Feature</b>", meta_label), Paragraph("<b>p-value</b>", meta_label), Paragraph("<b>Effect Size</b>", meta_label)]]
                for s in top_sigs:
                    t_rows.append([
                        Paragraph(s["feature"], body_style),
                        Paragraph(f"{s['p_value']:.4e}", body_style),
                        Paragraph(f"{s['effect_size']:.2f}" if s["effect_size"] is not None else "N/A", body_style),
                    ])
                st_tbl = Table(t_rows, colWidths=[204, 150, 150])
                st_tbl.setStyle(TableStyle([
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F1F5F9")),
                    ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#CBD5E1")),
                    ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
                    ("TOPPADDING", (0, 0), (-1, -1), 4),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ]))
                story.append(st_tbl)
            story.append(Spacer(1, 10))

        # ==============================================================================
        # 6. Machine Learning & Explainability (SHAP & LIME)
        # ==============================================================================
        if should_include("ml_and_xai") and (data.has_ml or data.has_shap or data.has_lime):
            story.append(Paragraph("5. Machine Learning & Explainable AI (SHAP & LIME)", h1_style))
            story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#CBD5E1"), spaceAfter=10))
            if data.has_ml:
                story.append(Paragraph(narratives.ml_narrative, body_style))
            if data.has_shap or data.has_lime:
                story.append(Paragraph(narratives.shap_lime_narrative, body_style))
            story.append(Spacer(1, 10))

        # ==============================================================================
        # 7. Root Cause Candidate Rankings
        # ==============================================================================
        if should_include("root_cause_ranking") and data.has_root_cause:
            story.append(Paragraph("6. Root Cause Candidate Priority Ranking", h1_style))
            story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#CBD5E1"), spaceAfter=10))
            story.append(Paragraph(narratives.root_cause_narrative, body_style))

            cands = data.root_cause_data.get("ranked_candidates", [])
            if cands:
                rc_rows = [[
                    Paragraph("<b>Rank</b>", meta_label),
                    Paragraph("<b>Candidate Feature</b>", meta_label),
                    Paragraph("<b>Score</b>", meta_label),
                    Paragraph("<b>Evidence Strength</b>", meta_label),
                    Paragraph("<b>Observed Shift</b>", meta_label),
                ]]
                for c in cands[:6]:
                    rc_rows.append([
                        Paragraph(f"#{c.get('rank', 1)}", body_style),
                        Paragraph(c.get("feature", "N/A"), body_style),
                        Paragraph(f"{c.get('overall_score', 0.0):.3f}", body_style),
                        Paragraph(c.get("evidence_strength", "N/A"), body_style),
                        Paragraph(f"{c.get('absolute_shift', 0.0):+.2f}", body_style),
                    ])
                rc_tbl = Table(rc_rows, colWidths=[45, 160, 65, 140, 94])
                rc_tbl.setStyle(TableStyle([
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F1F5F9")),
                    ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#CBD5E1")),
                    ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
                    ("TOPPADDING", (0, 0), (-1, -1), 4),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ]))
                story.append(rc_tbl)
            story.append(Spacer(1, 10))

        # ==============================================================================
        # 8. Segment Decomposition
        # ==============================================================================
        if should_include("segment_analysis") and data.has_segment_analysis:
            story.append(Paragraph("7. Segment Contribution Decomposition", h1_style))
            story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#CBD5E1"), spaceAfter=10))
            story.append(Paragraph(narratives.segment_narrative, body_style))
            story.append(Spacer(1, 10))

        # ==============================================================================
        # 9. Recommendations & Next Steps
        # ==============================================================================
        if should_include("recommendations"):
            story.append(Paragraph("8. Evidence-Based Recommendations & Next Steps", h1_style))
            story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#CBD5E1"), spaceAfter=10))
            for rec in narratives.recommendations_list:
                story.append(
                    Paragraph(
                        f"• <b>[{rec['category']}]</b> {rec['recommendation']}<br/>"
                        f"&nbsp;&nbsp;<i>Evidence Base: {rec['supporting_finding']}</i>",
                        body_style,
                    )
                )
            story.append(Spacer(1, 10))

        # ==============================================================================
        # 10. Analytical Limitations & Caveats
        # ==============================================================================
        if should_include("limitations"):
            story.append(Paragraph("9. Analytical Assumptions & Limitations", h1_style))
            story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#CBD5E1"), spaceAfter=10))
            for lim in narratives.limitations_list:
                story.append(Paragraph(f"• {lim}", body_style))
            story.append(Spacer(1, 10))

        # Build document with running headers/footers
        doc.build(story, canvasmaker=NumberedCanvas)
        buffer.seek(0)
        return buffer.getvalue()

