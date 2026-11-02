"""Dashboard views, redesigned overview screen, and modular section placeholders."""

from typing import Dict, Any, List, Optional
import streamlit as st
import pandas as pd
import numpy as np

from config.constants import (
    APP_NAME,
    APP_TAGLINE,
    APP_DESCRIPTION,
    NavigationSection,
    NAV_ICONS,
)
from components.metric_cards import render_metric_card
from components.dataset_upload import render_dataset_uploader
from components.dataset_preview import render_dataset_preview
from components.dataset_metrics import render_dataset_metrics


def render_hero_banner() -> None:
    """Render the enterprise hero banner."""
    st.markdown(
        f"""
        <div style="background: radial-gradient(circle at 10% 20%, rgba(79, 70, 229, 0.12) 0%, transparent 50%),
                    radial-gradient(circle at 90% 80%, rgba(6, 182, 212, 0.08) 0%, transparent 50%),
                    #0D1322;
                    border: 1px solid #1E293B; border-radius: 12px; padding: 2rem 2.2rem; margin-bottom: 1.5rem;">
            <div style="display: flex; align-items: center; gap: 0.75rem; margin-bottom: 0.4rem;">
                <span style="font-size: 1.8rem;">🔍</span>
                <span style="font-size: 1.7rem; font-weight: 800; color: #F8FAFC; letter-spacing: -0.02em;">Understand the Why Behind Your Data</span>
            </div>
            <div style="font-size: 1.05rem; font-weight: 500; color: #818CF8; margin-bottom: 0.75rem;">
                Investigate unexpected outcomes, discover hidden patterns, and identify evidence-backed contributing factors.
            </div>
            <p style="font-size: 0.88rem; color: #94A3B8; max-width: 800px; margin: 0; line-height: 1.5;">
                {APP_DESCRIPTION}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def log_activity(action: str, details: str = "") -> None:
    """Log user activity to session state timeline."""
    if "activity_log" not in st.session_state:
        st.session_state["activity_log"] = []
    from datetime import datetime
    st.session_state["activity_log"].append({
        "timestamp": datetime.now().strftime("%H:%M:%S"),
        "action": action,
        "details": details,
    })


def render_welcome_screen() -> None:
    """Render the comprehensive Phase 12 Investigation Overview Dashboard."""
    render_hero_banner()

    df = (
        st.session_state.get("preprocessed_dataset")
        or st.session_state.get("uploaded_dataset")
        or st.session_state.get("current_dataset")
    )
    dataset_name = st.session_state.get("dataset_filename")
    target_var = st.session_state.get("selected_target_variable")
    active_inv = st.session_state.get("root_cause_active_package")

    # ==============================================================================
    # 1. Dataset & Investigation KPI Ribbon
    # ==============================================================================
    col1, col2, col3, col4 = st.columns(4)

    if df is not None:
        n_rows, n_cols = df.shape
        num_cols = len(df.select_dtypes(include=[np.number]).columns)
        cat_cols = len(df.select_dtypes(exclude=[np.number]).columns)
        missing_count = int(df.isnull().sum().sum())

        with col1:
            render_metric_card(
                title="Active Dataset",
                value=f"{n_rows:,} rows",
                subtitle=f"{dataset_name} ({n_cols} cols)",
                badge_text="Loaded",
                badge_type="emerald",
            )
        with col2:
            render_metric_card(
                title="Feature Composition",
                value=f"{num_cols} Num / {cat_cols} Cat",
                subtitle=f"Missing: {missing_count:,} cells",
                badge_text="Audited",
                badge_type="blue",
            )
    else:
        with col1:
            render_metric_card(
                title="Dataset Status",
                value="No Dataset",
                subtitle="Upload in Dataset Overview",
                badge_text="Pending",
                badge_type="amber",
            )
        with col2:
            render_metric_card(
                title="Features",
                value="0 Attributes",
                subtitle="Awaiting Ingestion",
                badge_text="Inactive",
                badge_type="slate",
            )

    # Investigation metrics
    if active_inv:
        with col3:
            render_metric_card(
                title="Active Investigation",
                value=active_inv.metadata.target_variable,
                subtitle=f"Shift: {active_inv.outcome_def.absolute_change:+.3f}",
                badge_text="Active",
                badge_type="purple",
            )
        with col4:
            render_metric_card(
                title="Top Potential Cause",
                value=active_inv.ranking_result.top_candidate_name,
                subtitle=active_inv.ranking_result.ranked_candidates[0].evidence_strength if active_inv.ranking_result.ranked_candidates else "Ranked #1",
                badge_text="Priority #1",
                badge_type="emerald",
            )
    else:
        with col3:
            render_metric_card(
                title="Investigation Status",
                value="No Active Run",
                subtitle="Define in Root Cause Studio",
                badge_text="Idle",
                badge_type="slate",
            )
        with col4:
            render_metric_card(
                title="XAI & Reporting",
                value="Engines Ready",
                subtitle="SHAP, LIME, Reports",
                badge_text="Ready",
                badge_type="blue",
            )

    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

    # ==============================================================================
    # 2. Analysis Progress & Execution Tracker
    # ==============================================================================
    st.markdown(
        """
        <div style="font-size: 1.05rem; font-weight: 700; color: #F1F5F9; margin-bottom: 0.5rem;">
            Analysis Pipeline Execution Tracker
        </div>
        """,
        unsafe_allow_html=True,
    )

    phases_status = [
        ("Dataset Upload", df is not None),
        ("Data Cleaning", st.session_state.get("preprocessed_dataset") is not None),
        ("EDA Analysis", df is not None),
        ("Unexpected Detection", st.session_state.get("detection_artifacts") is not None),
        ("Statistical Tests", st.session_state.get("latest_stat_result") is not None),
        ("ML Training", st.session_state.get("ml_comparison_result") is not None),
        ("SHAP Explainability", st.session_state.get("shap_active_calc") is not None),
        ("LIME Explainability", st.session_state.get("lime_active_result") is not None),
        ("Root Cause Engine", active_inv is not None),
        ("Automated Reports", "report_active_data" in st.session_state and st.session_state["report_active_data"] is not None),
    ]

    p_cols = st.columns(5)
    for i, (p_name, is_done) in enumerate(phases_status):
        col_idx = i % 5
        badge_color = "#10B981" if is_done else "#475569"
        status_txt = "✅ Done" if is_done else "⚪ Ready"
        with p_cols[col_idx]:
            st.markdown(
                f"""
                <div style="background: #111827; border: 1px solid #1F2937; border-radius: 6px; padding: 0.6rem 0.8rem; margin-bottom: 0.6rem;">
                    <div style="font-size: 0.72rem; color: #94A3B8; text-transform: uppercase;">Stage {i+1}</div>
                    <div style="font-size: 0.82rem; font-weight: 600; color: #F8FAFC; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">{p_name}</div>
                    <div style="font-size: 0.72rem; color: {badge_color}; margin-top: 2px;">{status_txt}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

    # ==============================================================================
    # 3. Quick Actions & Recent Activity
    # ==============================================================================
    col_act, col_timeline = st.columns([1.1, 0.9])

    with col_act:
        st.markdown(
            """
            <div style="background: #111827; border: 1px solid #1F2937; border-radius: 10px; padding: 1.25rem;">
                <div style="font-size: 0.95rem; font-weight: 700; color: #F8FAFC; margin-bottom: 0.5rem;">
                    🚀 Quick Actions
                </div>
                <div style="font-size: 0.82rem; color: #94A3B8; margin-bottom: 1rem;">
                    Jump directly to key workflow stages across the investigation pipeline.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        qa1, qa2 = st.columns(2)
        with qa1:
            if st.button("📊 Explore Active Dataset", use_container_width=True):
                st.info("Navigate to **Exploratory Analysis** in the left sidebar.")
            if st.button("🤖 Train ML Models", use_container_width=True):
                st.info("Navigate to **Machine Learning Models** in the left sidebar.")
            if st.button("📄 Generate Report Dossier", use_container_width=True):
                st.info("Navigate to **Investigation Reports** in the left sidebar.")
        with qa2:
            if st.button("🎯 Detect Unexpected Changes", use_container_width=True):
                st.info("Navigate to **Unexpected Outcome Detection** in the left sidebar.")
            if st.button("🔬 Root Cause Investigation", type="primary", use_container_width=True):
                st.info("Navigate to **Root Cause Investigation Studio** in the left sidebar.")
            if st.button("💡 Explain with SHAP & LIME", use_container_width=True):
                st.info("Navigate to **Explainable AI** in the left sidebar.")

    with col_timeline:
        st.markdown(
            """
            <div style="background: #111827; border: 1px solid #1F2937; border-radius: 10px; padding: 1.25rem;">
                <div style="font-size: 0.95rem; font-weight: 700; color: #F8FAFC; margin-bottom: 0.5rem;">
                    🕒 Session Activity Log
                </div>
                <div style="font-size: 0.82rem; color: #94A3B8; margin-bottom: 0.75rem;">
                    Real-time timeline tracking actions executed in this session.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        activities = st.session_state.get("activity_log", [])
        if activities:
            for act in reversed(activities[-5:]):
                st.markdown(
                    f"""
                    <div style="font-size: 0.8rem; color: #CBD5E1; border-left: 2px solid #6366F1; padding-left: 8px; margin-bottom: 6px;">
                        <span style="color: #64748B;">[{act['timestamp']}]</span> <strong>{act['action']}</strong>: {act['details']}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.markdown(
                """
                <div style="font-size: 0.82rem; color: #64748B; font-style: italic; padding: 0.5rem 0;">
                    No activities recorded yet. Actions will populate here as you clean data, train models, or investigate root causes.
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_dataset_understanding_page() -> None:
    """Render the complete Phase 2 Dataset Upload and Data Understanding page."""
    df = render_dataset_uploader()
    if df is not None:
        log_activity("Dataset Uploaded", f"{df.shape[0]} rows × {df.shape[1]} cols")
        render_dataset_preview(df)
        render_dataset_metrics(df)


def render_placeholder_section(section: NavigationSection) -> None:
    """Render a structured placeholder screen for navigation modules."""
    icon = NAV_ICONS.get(section, "🔹")

    st.markdown(
        f"""
        <div style="background: #111827; border: 1px solid #1F2937; border-radius: 12px; padding: 2.5rem; text-align: center; margin-top: 1rem;">
            <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">{icon}</div>
            <h2 style="font-size: 1.4rem; color: #F8FAFC; margin: 0 0 0.5rem 0;">{section.value}</h2>
            <p style="font-size: 0.9rem; color: #94A3B8; max-width: 500px; margin: 0 auto;">
                This module is configured and ready for analysis. Upload a dataset or run an investigation to activate full interactive workflows.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
