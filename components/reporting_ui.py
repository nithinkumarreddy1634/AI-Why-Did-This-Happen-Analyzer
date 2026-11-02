"""Interactive Streamlit UI for Automated Investigation Report Generation."""

from typing import Dict, Any, List, Optional
import streamlit as st
import pandas as pd

from modules.reporting.report_manager import AutomatedReportManager
from modules.reporting.report_data_collector import CollectedReportData
from modules.reporting.narrative_generator import ReportNarrativeSections
from modules.root_cause.investigation_history import compute_dataset_fingerprint


def render_reporting_page() -> None:
    """Render the Automated Investigation Reports Studio page."""
    # 1. Initialize Report Manager in Session State
    if "report_manager" not in st.session_state:
        st.session_state["report_manager"] = AutomatedReportManager()
    if "report_active_data" not in st.session_state:
        st.session_state["report_active_data"] = None
    if "report_active_narratives" not in st.session_state:
        st.session_state["report_active_narratives"] = None

    manager: AutomatedReportManager = st.session_state["report_manager"]

    # 2. Page Header
    st.markdown(
        """
        <div style="margin-bottom: 1.5rem;">
            <div style="display: flex; align-items: center; gap: 0.75rem;">
                <span style="font-size: 2.2rem;">📄</span>
                <div>
                    <h1 style="margin: 0; font-size: 1.85rem; font-weight: 800; color: #F8FAFC; letter-spacing: -0.02em;">
                        Automated Investigation Reports
                    </h1>
                    <p style="margin: 0.2rem 0 0 0; font-size: 0.88rem; color: #94A3B8;">
                        Multi-Format Executive Reporting (PDF, Word DOCX, Standalone HTML, JSON)
                    </p>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 3. Resolve Active Dataset & Investigation Package
    df = (
        st.session_state.get("preprocessed_dataset")
        or st.session_state.get("uploaded_dataset")
        or st.session_state.get("current_dataset")
    )
    dataset_name = st.session_state.get("dataset_filename", "active_dataset.csv")
    active_inv_pkg = st.session_state.get("root_cause_active_package")

    if df is None or df.empty:
        st.warning("⚠️ No active dataset loaded. Please upload or load a dataset in **Dataset Overview** first.")
        return

    # 4. Report Setup Form
    with st.expander("⚙️ Report Configuration & Branding", expanded=(st.session_state["report_active_data"] is None)):
        col1, col2 = st.columns(2)
        with col1:
            rep_title = st.text_input("Report Title", value="Executive Investigation Report")
            rep_subtitle = st.text_input("Subtitle", value="Multi-Engine Root Cause & Predictive Diagnostic Dossier")
            org_name = st.text_input("Organization Name", value="Analytics & Governance Group")
        with col2:
            author_name = st.text_input("Author / Prepared By", value="AI Why Did This Happen Analyzer")
            conf_level = st.selectbox(
                "Classification",
                options=["Internal Business Sensitive", "Confidential - Executive Eyes Only", "Public / General"],
                index=0,
            )

        st.markdown("<p style='font-size: 0.85rem; font-weight: 700; color: #CBD5E1; margin-top: 10px;'>Sections to Include in Report</p>", unsafe_allow_html=True)
        sec_cols = st.columns(4)
        with sec_cols[0]:
            inc_exec = st.checkbox("Executive Summary", value=True)
            inc_obj = st.checkbox("Objective & Scope", value=True)
            inc_ds = st.checkbox("Dataset Overview", value=True)
        with sec_cols[1]:
            inc_stat = st.checkbox("Statistical Investigation", value=True)
            inc_ml = st.checkbox("Machine Learning & XAI", value=True)
        with sec_cols[2]:
            inc_rc = st.checkbox("Root Cause Priority Ranking", value=True)
            inc_seg = st.checkbox("Segment Contribution", value=True)
        with sec_cols[3]:
            inc_recs = st.checkbox("Recommendations", value=True)
            inc_limits = st.checkbox("Assumptions & Limitations", value=True)

        included_sections: List[str] = []
        if inc_exec: included_sections.append("executive_summary")
        if inc_obj: included_sections.append("investigation_objective")
        if inc_ds: included_sections.append("dataset_overview")
        if inc_stat: included_sections.append("statistical_investigation")
        if inc_ml: included_sections.append("ml_and_xai")
        if inc_rc: included_sections.append("root_cause_ranking")
        if inc_seg: included_sections.append("segment_analysis")
        if inc_recs: included_sections.append("recommendations")
        if inc_limits: included_sections.append("limitations")

        build_btn = st.button("🔄 Generate / Refresh Report Dossier", type="primary", use_container_width=True)

        if build_btn or st.session_state["report_active_data"] is None:
            with st.spinner("Collecting analytical artifacts and generating deterministic narrative..."):
                # Retrieve all session artifacts
                ml_imp = st.session_state.get("ml_active_importance")
                ml_comp = st.session_state.get("ml_comparison_result")
                shap_g = st.session_state.get("shap_active_global")
                lime_l = st.session_state.get("lime_active_result")
                anom_art = st.session_state.get("detection_artifacts")
                prep_pipe = st.session_state.get("preprocessing_pipeline")

                collected, narratives = manager.assemble_report(
                    df=df,
                    dataset_name=dataset_name,
                    active_investigation_pkg=active_inv_pkg,
                    ml_importance_res=ml_imp,
                    ml_comparison_res=ml_comp,
                    shap_global_res=shap_g,
                    lime_active_res=lime_l,
                    detection_artifacts=anom_art,
                    preprocessing_pipeline=prep_pipe,
                    title=rep_title,
                    subtitle=rep_subtitle,
                    organization=org_name,
                    author=author_name,
                    confidentiality=conf_level,
                )
                st.session_state["report_active_data"] = collected
                st.session_state["report_active_narratives"] = narratives
                st.success("✅ Report generated and ready for preview & export.")

    data: CollectedReportData = st.session_state["report_active_data"]
    narratives: ReportNarrativeSections = st.session_state["report_active_narratives"]

    # 5. Live Preview & Narrative Editor Tabs
    t_preview, t_edit, t_export, t_hist = st.tabs([
        "👁️ Live Report Preview",
        "✏️ Narrative Editor",
        "📥 Multi-Format Downloads",
        "📚 Report Catalog",
    ])

    # Tab 1: Live Preview
    with t_preview:
        st.markdown(
            f"""
            <div style="background: #111827; border: 1px solid #1F2937; border-radius: 8px; padding: 1.5rem; margin-bottom: 1.5rem;">
                <div style="font-size: 0.8rem; font-weight: 700; text-transform: uppercase; color: #6366F1;">Dossier Preview</div>
                <h2 style="font-size: 1.6rem; color: #F8FAFC; margin: 0.2rem 0 0.4rem 0;">{data.report_title}</h2>
                <div style="font-size: 0.9rem; color: #94A3B8;">{data.report_subtitle}</div>
                <div style="font-size: 0.78rem; color: #64748B; margin-top: 0.5rem;">
                    Target: <strong>{data.target_variable}</strong> ({data.outcome_type}) | Cohorts: <strong>{data.baseline_cohort_name}</strong> vs <strong>{data.comparison_cohort_name}</strong> | Shift: <strong>{data.outcome_delta:+.4f}</strong>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("#### 1. Executive Summary")
        st.markdown(narratives.executive_summary)

        st.markdown("#### 2. Investigation Objective & Scope")
        st.markdown(narratives.investigation_objective)

        st.markdown("#### 3. Dataset Overview & Data Health")
        st.markdown(narratives.dataset_overview_narrative)
        st.markdown(narratives.data_quality_narrative)

        if data.has_statistical:
            st.markdown("#### 4. Statistical Findings")
            st.markdown(narratives.statistical_narrative)

        if data.has_ml or data.has_shap or data.has_lime:
            st.markdown("#### 5. Machine Learning & Explainable AI")
            if data.has_ml: st.markdown(narratives.ml_narrative)
            if data.has_shap or data.has_lime: st.markdown(narratives.shap_lime_narrative)

        if data.has_root_cause:
            st.markdown("#### 6. Root Cause Candidate Priority Ranking")
            st.markdown(narratives.root_cause_narrative)

        if data.has_segment_analysis:
            st.markdown("#### 7. Segment Contribution Decomposition")
            st.markdown(narratives.segment_narrative)

        st.markdown("#### 8. Evidence-Based Recommendations")
        for r in narratives.recommendations_list:
            st.markdown(f"- **[{r['category']}]** {r['recommendation']} *(Basis: {r['supporting_finding']})*")

        st.markdown("#### 9. Analytical Assumptions & Limitations")
        for lim in narratives.limitations_list:
            st.markdown(f"- {lim}")

    # Tab 2: Narrative Editor
    with t_edit:
        st.markdown("### ✏️ Customize & Fine-Tune Narrative Content")
        st.markdown("Review and update report text blocks prior to binary export.")

        new_exec = st.text_area("Executive Summary", value=narratives.executive_summary, height=120)
        new_obj = st.text_area("Investigation Objective", value=narratives.investigation_objective, height=90)
        new_stat = st.text_area("Statistical Narrative", value=narratives.statistical_narrative, height=90)
        new_rc = st.text_area("Root Cause Narrative", value=narratives.root_cause_narrative, height=90)

        if st.button("💾 Apply Narrative Modifications"):
            narratives.executive_summary = new_exec
            narratives.investigation_objective = new_obj
            narratives.statistical_narrative = new_stat
            narratives.root_cause_narrative = new_rc
            st.session_state["report_active_narratives"] = narratives
            st.success("✅ Narrative text updated successfully!")

    # Tab 3: Downloads
    with t_export:
        st.markdown("### 📥 Multi-Format Report Downloads")
        st.markdown("Generate auditable binaries in PDF, Word DOCX, Standalone HTML, or JSON.")

        col_dl1, col_dl2 = st.columns(2)

        with col_dl1:
            st.markdown(
                """
                <div style="background: #111827; border: 1px solid #1F2937; border-radius: 8px; padding: 1rem; margin-bottom: 1rem;">
                    <div style="font-size: 0.85rem; font-weight: 700; color: #F87171;">📕 Executive PDF Document</div>
                    <div style="font-size: 0.78rem; color: #94A3B8; margin-top: 4px;">Formatted with ReportLab, cover page, running headers, and 'Page X of Y' numbering.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            with st.spinner("Rendering PDF binary..."):
                pdf_bytes = manager.generate_pdf(data, narratives, included_sections)

            st.download_button(
                "📥 Download PDF Report (.pdf)",
                data=pdf_bytes,
                file_name=f"investigation_report_{data.target_variable}.pdf",
                mime="application/pdf",
                use_container_width=True,
                key="btn_dl_pdf",
            )

            st.markdown(
                """
                <div style="background: #111827; border: 1px solid #1F2937; border-radius: 8px; padding: 1rem; margin-top: 1rem; margin-bottom: 1rem;">
                    <div style="font-size: 0.85rem; font-weight: 700; color: #60A5FA;">📘 Editable Word Document (.docx)</div>
                    <div style="font-size: 0.78rem; color: #94A3B8; margin-top: 4px;">Standardized Microsoft Word document with editable sections and formatted tables.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            with st.spinner("Rendering Word DOCX..."):
                docx_bytes = manager.generate_docx(data, narratives, included_sections)

            st.download_button(
                "📥 Download Word Document (.docx)",
                data=docx_bytes,
                file_name=f"investigation_report_{data.target_variable}.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                use_container_width=True,
                key="btn_dl_docx",
            )

        with col_dl2:
            st.markdown(
                """
                <div style="background: #111827; border: 1px solid #1F2937; border-radius: 8px; padding: 1rem; margin-bottom: 1rem;">
                    <div style="font-size: 0.85rem; font-weight: 700; color: #34D399;">🌐 Standalone HTML Report</div>
                    <div style="font-size: 0.78rem; color: #94A3B8; margin-top: 4px;">Self-contained HTML dossier with responsive typography, viewable in any modern web browser.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            html_str = manager.generate_html(data, narratives, included_sections)
            st.download_button(
                "📥 Download Standalone HTML (.html)",
                data=html_str,
                file_name=f"investigation_report_{data.target_variable}.html",
                mime="text/html",
                use_container_width=True,
                key="btn_dl_html",
            )

            st.markdown(
                """
                <div style="background: #111827; border: 1px solid #1F2937; border-radius: 8px; padding: 1rem; margin-top: 1rem; margin-bottom: 1rem;">
                    <div style="font-size: 0.85rem; font-weight: 700; color: #FBBF24;">📦 Machine-Readable JSON</div>
                    <div style="font-size: 0.78rem; color: #94A3B8; margin-top: 4px;">Structured JSON bundle preserving metadata, numerical precision, and section breakdowns.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            json_str = manager.generate_json(data, narratives, included_sections)
            st.download_button(
                "📥 Download Complete JSON (.json)",
                data=json_str,
                file_name=f"investigation_report_{data.target_variable}.json",
                mime="application/json",
                use_container_width=True,
                key="btn_dl_json",
            )

        # Log into history upon download ready
        fp = compute_dataset_fingerprint(df, dataset_name)
        manager.log_generated_report(
            title=data.report_title,
            investigation_id="inv_active",
            dataset_name=dataset_name,
            dataset_fingerprint=fp,
            formats=["PDF", "DOCX", "HTML", "JSON"],
        )

    # Tab 4: Report Catalog & History
    with t_hist:
        st.markdown("### 📚 Report Catalog & History")
        records = manager.history.list_reports()
        if records:
            rep_df = pd.DataFrame([
                {
                    "Report ID": r.report_id,
                    "Title": r.title,
                    "Dataset": r.dataset_name,
                    "Date": r.generation_date,
                    "Formats": ", ".join(r.formats),
                    "Status": "⚠️ Stale" if r.is_stale else "✅ Synchronized",
                }
                for r in records
            ])
            st.dataframe(rep_df, use_container_width=True, hide_index=True)
        else:
            st.info("No previously exported reports found in catalog.")

