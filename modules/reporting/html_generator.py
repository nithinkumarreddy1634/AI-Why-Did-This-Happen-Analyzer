"""Standalone HTML report generator with clean, responsive typography and print styling."""

import html
from typing import Dict, Any, List, Optional

from modules.reporting.report_data_collector import CollectedReportData
from modules.reporting.narrative_generator import ReportNarrativeSections


class HTMLReportGenerator:
    """Builds self-contained, responsive HTML investigation dossiers."""

    def generate_html(
        self,
        data: CollectedReportData,
        narratives: ReportNarrativeSections,
        included_sections: Optional[List[str]] = None,
    ) -> str:
        """Render complete HTML report document as a string."""
        inc = included_sections or ["all"]
        def should_include(sec: str) -> bool:
            return "all" in inc or sec in inc

        pct_str = f" ({data.outcome_rel_pct:+.1f}%)" if data.outcome_rel_pct is not None else ""

        html_out = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{html.escape(data.report_title)}</title>
    <style>
        :root {{
            --primary: #4F46E5;
            --primary-light: #EEF2FF;
            --text-dark: #0F172A;
            --text-muted: #64748B;
            --border: #E2E8F0;
            --bg-card: #F8FAFC;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
            line-height: 1.6;
            color: var(--text-dark);
            background-color: #FFFFFF;
            margin: 0;
            padding: 2.5rem 1.5rem;
        }}
        .container {{
            max-width: 860px;
            margin: 0 auto;
        }}
        .brand {{
            font-size: 0.85rem;
            font-weight: 800;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: var(--primary);
            margin-bottom: 0.5rem;
        }}
        h1 {{
            font-size: 2.1rem;
            font-weight: 800;
            margin: 0 0 0.5rem 0;
            color: var(--text-dark);
            letter-spacing: -0.02em;
        }}
        .subtitle {{
            font-size: 1.1rem;
            color: var(--text-muted);
            margin-bottom: 1.5rem;
        }}
        .meta-card {{
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 1.25rem;
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 1rem;
            margin-bottom: 2rem;
        }}
        .meta-item label {{
            display: block;
            font-size: 0.75rem;
            font-weight: 700;
            text-transform: uppercase;
            color: var(--text-muted);
        }}
        .meta-item value {{
            display: block;
            font-size: 0.95rem;
            font-weight: 600;
            color: var(--text-dark);
            margin-top: 2px;
        }}
        .notice-banner {{
            background: var(--primary-light);
            border-left: 4px solid var(--primary);
            padding: 0.85rem 1.2rem;
            border-radius: 4px;
            font-size: 0.88rem;
            color: #3730A3;
            margin-bottom: 2rem;
        }}
        h2 {{
            font-size: 1.35rem;
            font-weight: 700;
            margin: 2rem 0 0.6rem 0;
            border-bottom: 1px solid var(--border);
            padding-bottom: 0.4rem;
            color: var(--text-dark);
        }}
        p {{
            margin: 0.6rem 0;
            font-size: 0.96rem;
            color: #334155;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 1rem 0;
            font-size: 0.9rem;
        }}
        th, td {{
            text-align: left;
            padding: 8px 12px;
            border-bottom: 1px solid var(--border);
        }}
        th {{
            background-color: var(--bg-card);
            color: var(--text-muted);
            font-weight: 700;
            font-size: 0.8rem;
            text-transform: uppercase;
        }}
        ul {{
            padding-left: 1.4rem;
        }}
        li {{
            margin-bottom: 0.4rem;
            font-size: 0.95rem;
        }}
        .footer {{
            margin-top: 3.5rem;
            padding-top: 1.5rem;
            border-top: 1px solid var(--border);
            font-size: 0.8rem;
            color: var(--text-muted);
            text-align: center;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="brand">AI &ldquo;Why Did This Happen?&rdquo; Analyzer</div>
        <h1>{html.escape(data.report_title)}</h1>
        <div class="subtitle">{html.escape(data.report_subtitle)}</div>

        <div class="meta-card">
            <div class="meta-item"><label>Target Metric</label><value>{html.escape(data.target_variable)} ({html.escape(data.outcome_type)})</value></div>
            <div class="meta-item"><label>Dataset</label><value>{html.escape(data.dataset_name)}</value></div>
            <div class="meta-item"><label>Baseline Cohort</label><value>{html.escape(data.baseline_cohort_name)}</value></div>
            <div class="meta-item"><label>Comparison Cohort</label><value>{html.escape(data.comparison_cohort_name)}</value></div>
            <div class="meta-item"><label>Observed Shift (Δ)</label><value>{data.outcome_delta:+.4f}{pct_str}</value></div>
            <div class="meta-item"><label>Report Date</label><value>{html.escape(data.generation_date)}</value></div>
        </div>

        <div class="notice-banner">
            <strong>Analytical Governance Notice:</strong> Findings synthesized herein represent empirical associations and predictive attributions. Observational data cannot establish unconfounded causality. All ranked factors represent priority candidates for operational review.
        </div>
"""

        # Sections
        if should_include("executive_summary"):
            html_out += f"<h2>1. Executive Summary</h2><p>{html.escape(narratives.executive_summary)}</p>"

        if should_include("investigation_objective"):
            html_out += f"<h2>2. Investigation Objective &amp; Scope</h2><p>{html.escape(narratives.investigation_objective)}</p>"

        if should_include("dataset_overview"):
            html_out += f"<h2>3. Dataset Overview &amp; Data Health</h2><p>{html.escape(narratives.dataset_overview_narrative)}</p><p>{html.escape(narratives.data_quality_narrative)}</p>"

        if should_include("statistical_investigation") and data.has_statistical:
            html_out += f"<h2>4. Statistical Investigation</h2><p>{html.escape(narratives.statistical_narrative)}</p>"

        if should_include("ml_and_xai") and (data.has_ml or data.has_shap or data.has_lime):
            html_out += f"<h2>5. Machine Learning &amp; Explainability (SHAP / LIME)</h2>"
            if data.has_ml:
                html_out += f"<p>{html.escape(narratives.ml_narrative)}</p>"
            if data.has_shap or data.has_lime:
                html_out += f"<p>{html.escape(narratives.shap_lime_narrative)}</p>"

        if should_include("root_cause_ranking") and data.has_root_cause:
            html_out += f"<h2>6. Root Cause Candidate Priority Ranking</h2><p>{html.escape(narratives.root_cause_narrative)}</p>"
            cands = data.root_cause_data.get("ranked_candidates", [])
            if cands:
                html_out += "<table><thead><tr><th>Rank</th><th>Candidate Feature</th><th>Evidence Score</th><th>Evidence Strength</th><th>Observed Shift</th></tr></thead><tbody>"
                for c in cands[:8]:
                    html_out += f"<tr><td>#{c.get('rank', 1)}</td><td><strong>{html.escape(c.get('feature', ''))}</strong></td><td>{c.get('overall_score', 0.0):.3f}</td><td>{html.escape(c.get('evidence_strength', ''))}</td><td>{c.get('absolute_shift', 0.0):+.2f}</td></tr>"
                html_out += "</tbody></table>"

        if should_include("segment_analysis") and data.has_segment_analysis:
            html_out += f"<h2>7. Segment Contribution Decomposition</h2><p>{html.escape(narratives.segment_narrative)}</p>"

        if should_include("recommendations"):
            html_out += "<h2>8. Evidence-Based Recommendations</h2><ul>"
            for rec in narratives.recommendations_list:
                html_out += f"<li><strong>[{html.escape(rec['category'])}]</strong> {html.escape(rec['recommendation'])} <em>(Supporting Finding: {html.escape(rec['supporting_finding'])})</em></li>"
            html_out += "</ul>"

        if should_include("limitations"):
            html_out += "<h2>9. Analytical Assumptions &amp; Limitations</h2><ul>"
            for lim in narratives.limitations_list:
                html_out += f"<li>{html.escape(lim)}</li>"
            html_out += "</ul>"

        html_out += f"""
        <div class="footer">
            Generated by AI &ldquo;Why Did This Happen?&rdquo; Analyzer &bull; {html.escape(data.generation_date)} &bull; {html.escape(data.confidentiality)}
        </div>
    </div>
</body>
</html>"""
        return html_out

