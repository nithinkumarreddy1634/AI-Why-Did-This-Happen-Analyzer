"""Interactive Streamlit UI for Phase 5: Unexpected Outcome Detection Engine."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import Optional, Dict, Any, List

from modules.unexpected_detection.detection_engine import UnexpectedDetectionEngine
from modules.unexpected_detection.severity_classifier import (
    SeverityLevel,
    SeverityThresholds,
    SEVERITY_COLORS,
)
from utils.detection_helpers import split_baseline_comparison
from utils.sample_data import generate_churn_investigation_dataset


def render_unexpected_detection_page() -> None:
    """Render the master Unexpected Outcome Detection workbench."""
    st.markdown(
        """
        <div style="margin-bottom: 1.5rem;">
            <div style="font-size: 1.8rem; font-weight: 800; color: #F8FAFC; letter-spacing: -0.02em;">
                ⚡ Unexpected Outcome Detection & Intelligent Change Engine
            </div>
            <div style="font-size: 0.95rem; color: #94A3B8; margin-top: 0.25rem;">
                Empirical anomaly isolation, distribution shift quantification, and multi-dimensional cohort change identification.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 1. Dataset Selection / Activation Check
    df_cleaned = st.session_state.get("preprocessed_dataset") or st.session_state.get("uploaded_dataset")
    df_original = st.session_state.get("original_dataset")
    filename = st.session_state.get("dataset_filename", "active_dataset.csv")

    dataset_options = []
    if df_cleaned is not None:
        dataset_options.append("Cleaned / Preprocessed Dataset")
    if df_original is not None:
        dataset_options.append("Original Uploaded Dataset")
    if not dataset_options and st.session_state.get("current_dataset") is not None:
        dataset_options.append("Current Dataset")

    if not dataset_options:
        _render_empty_dataset_prompt()
        return

    # Dataset Context & Version Selector
    col_ds1, col_ds2, col_ds3 = st.columns([1.6, 1, 1])
    with col_ds1:
        if len(dataset_options) > 1:
            selected_source = st.selectbox(
                "Dataset Version to Investigate:",
                options=dataset_options,
                index=0,
                key="unexpected_dataset_source_picker",
            )
        else:
            selected_source = dataset_options[0]

        if selected_source == "Original Uploaded Dataset" and df_original is not None:
            df = df_original
            active_version_label = f"Original Upload ({filename})"
        else:
            df = df_cleaned if df_cleaned is not None else st.session_state.get("current_dataset")
            active_version_label = f"Cleaned / Working ({filename})"

        target_name = st.session_state.get("selected_target_variable") or "Unassigned"
        st.markdown(
            f"""
            <div style="background: #131B2E; border: 1px solid #1E293B; border-radius: 8px; padding: 0.5rem 0.8rem; font-size: 0.82rem; color: #CBD5E1; margin-top: 0.25rem;">
                Active: <strong style="color: #38BDF8;">{active_version_label}</strong> &nbsp;|&nbsp;
                Target: <strong style="color: #38BDF8;">{target_name}</strong> &nbsp;|&nbsp;
                Size: <strong style="color: #38BDF8;">{len(df):,} rows × {len(df.columns)} cols</strong>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_ds2:
        if st.button("🔄 Reload Demo Dataset", use_container_width=True):
            demo_df = generate_churn_investigation_dataset()
            st.session_state["current_dataset"] = demo_df
            st.session_state["uploaded_dataset"] = demo_df
            st.session_state["original_dataset"] = demo_df.copy()
            st.session_state["dataset_filename"] = "customer_churn_benchmark.csv"
            st.rerun()
    with col_ds3:
        if st.button("🧹 Clear Dataset", use_container_width=True):
            st.session_state["current_dataset"] = None
            st.session_state["uploaded_dataset"] = None
            st.session_state["original_dataset"] = None
            st.session_state["dataset_filename"] = None
            st.session_state["preprocessed_dataset"] = None
            st.session_state["selected_target_variable"] = None
            st.rerun()

    # Initialize Engine
    engine = UnexpectedDetectionEngine()
    candidates = engine.inspect_dataset_candidates(df)

    # 2. Main Investigation Tabs
    tab_overview, tab_num, tab_dist, tab_cat, tab_time, tab_seg, tab_insights, tab_export = st.tabs([
        "📌 Overview",
        "🔢 Numerical Anomalies",
        "📊 Distribution Shifts",
        "🏷️ Categorical Changes",
        "⏳ Time-Based Changes",
        "👥 Segment Disaggregation",
        "💡 Automated Insights",
        "📥 Export Results",
    ])

    # Store detection artifacts in session state for export
    if "detection_artifacts" not in st.session_state:
        st.session_state["detection_artifacts"] = {
            "anomalies": pd.DataFrame(),
            "distribution_summary": pd.DataFrame(),
            "categorical_summary": pd.DataFrame(),
            "segment_summary": pd.DataFrame(),
            "insights": [],
        }

    # TAB 1: OVERVIEW
    with tab_overview:
        _render_overview_tab(df, candidates, engine)

    # TAB 2: NUMERICAL ANOMALIES
    with tab_num:
        _render_numerical_tab(df, candidates["numerical"], engine)

    # TAB 3: DISTRIBUTION SHIFTS
    with tab_dist:
        _render_distribution_tab(df, candidates, engine)

    # TAB 4: CATEGORICAL CHANGES
    with tab_cat:
        _render_categorical_tab(df, candidates, engine)

    # TAB 5: TIME-BASED CHANGES
    with tab_time:
        _render_time_tab(df, candidates, engine)

    # TAB 6: SEGMENT-LEVEL INVESTIGATION
    with tab_seg:
        _render_segment_tab(df, candidates, engine)

    # TAB 7: AUTOMATED INSIGHTS
    with tab_insights:
        _render_insights_tab()

    # TAB 8: EXPORT RESULTS
    with tab_export:
        _render_export_tab()


def _get_active_dataframe() -> Optional[pd.DataFrame]:
    """Retrieve active DataFrame from session state, supporting preprocessed, uploaded, or current."""
    if st.session_state.get("preprocessed_dataset") is not None:
        return st.session_state["preprocessed_dataset"]
    if st.session_state.get("uploaded_dataset") is not None:
        return st.session_state["uploaded_dataset"]
    if st.session_state.get("current_dataset") is not None:
        return st.session_state["current_dataset"]
    return None


def _render_empty_dataset_prompt() -> None:
    """Prompt user to load sample dataset or upload a file."""
    st.markdown(
        """
        <div style="background: #111827; border: 1px dashed #334155; border-radius: 12px; padding: 2.5rem 2rem; text-align: center; margin-top: 1rem;">
            <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">📂</div>
            <h3 style="color: #F8FAFC; margin-bottom: 0.5rem;">No Active Dataset in Session</h3>
            <p style="color: #94A3B8; max-width: 600px; margin: 0 auto 1.5rem auto; font-size: 0.9rem;">
                To run unexpected outcome detection, either load the built-in <strong>Customer Churn Surge Benchmark</strong> (1,200 records with an unexpected spike in churn from 10% to 28%) or upload your own CSV/Excel file below.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown("#### Option 1: Load Benchmark Dataset")
        if st.button("🚀 Load Customer Churn Investigation Dataset", type="primary", use_container_width=True):
            demo_df = generate_churn_investigation_dataset()
            st.session_state["current_dataset"] = demo_df
            st.session_state["uploaded_dataset"] = demo_df
            st.session_state["dataset_filename"] = "customer_churn_benchmark.csv"
            st.rerun()

    with c2:
        st.markdown("#### Option 2: Upload Custom Data")
        uploaded_file = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx", "parquet"], label_visibility="collapsed")
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith(".csv"):
                    df = pd.read_csv(uploaded_file)
                elif uploaded_file.name.endswith(".parquet"):
                    df = pd.read_parquet(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                st.session_state["current_dataset"] = df
                st.session_state["uploaded_dataset"] = df
                st.session_state["dataset_filename"] = uploaded_file.name
                st.success(f"Successfully loaded '{uploaded_file.name}' ({len(df):,} rows).")
                st.rerun()
            except Exception as e:
                st.error(f"Failed to parse file: {str(e)}")


def _render_overview_tab(df: pd.DataFrame, candidates: Dict[str, List[str]], engine: UnexpectedDetectionEngine) -> None:
    """Render high-level detection dashboard."""
    st.markdown("### 📌 Detection System Architecture & Dataset Health")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Total Records", f"{len(df):,}")
    with c2:
        st.metric("Numerical Features", len(candidates["numerical"]))
    with c3:
        st.metric("Categorical Features", len(candidates["categorical"]))
    with c4:
        st.metric("Temporal Columns", len(candidates["datetime"]))

    st.markdown("<hr style='border: none; border-top: 1px solid #1E293B; margin: 1.5rem 0;'>", unsafe_allow_html=True)

    st.markdown("#### 🔍 Automated Rapid Scan across Numerical Features")
    st.caption("Scans numerical features using 3σ Z-Score to isolate immediate deviation candidates.")

    scan_records = []
    for col in candidates["numerical"][:8]:  # Scan up to 8 columns for performance
        res = engine.detect_numerical_anomalies(df[col], method="Z-Score", threshold=3.0)
        scan_records.append({
            "Variable": col,
            "Valid Count": f"{res['valid_observations']:,}",
            "Unusual Observations": res["unusual_count"],
            "Anomaly Rate": f"{res['unusual_pct']:.2f}%",
            "Mean": f"{res['baseline_stats'].get('mean', 0):.2f}",
            "Std Dev": f"{res['baseline_stats'].get('std', 0):.2f}",
            "Severity": SeverityLevel.HIGH.value if res["unusual_pct"] > 5 else (SeverityLevel.MODERATE.value if res["unusual_count"] > 0 else SeverityLevel.NORMAL.value),
        })

    scan_df = pd.DataFrame(scan_records)
    st.dataframe(scan_df, use_container_width=True, hide_index=True)


def _render_numerical_tab(df: pd.DataFrame, num_cols: List[str], engine: UnexpectedDetectionEngine) -> None:
    """Render interactive numerical outlier detection workbench."""
    st.markdown("### 🔢 Numerical Deviation & Anomaly Detection")
    st.caption("Identify observations exhibiting statistically improbable deviations using classical and robust estimators.")

    if not num_cols:
        st.warning("No numeric columns found in the dataset.")
        return

    c1, c2, c3 = st.columns([1.5, 1.5, 1])
    with c1:
        selected_col = st.selectbox("Select Numerical Column to Investigate", options=num_cols)
    with c2:
        method = st.selectbox(
            "Anomaly Detection Algorithm",
            options=[
                "Z-Score",
                "Standard Deviation Threshold",
                "Modified Z-Score (MAD)",
                "IQR (Interquartile Range)",
                "% Deviation from Median",
                "% Deviation from Mean",
            ],
        )
    with c3:
        if method in ("Z-Score", "Standard Deviation Threshold"):
            threshold = st.slider("Std Dev Threshold (σ)", min_value=1.5, max_value=5.0, value=3.0, step=0.1)
        elif method == "Modified Z-Score (MAD)":
            threshold = st.slider("MAD Threshold", min_value=2.0, max_value=6.0, value=3.5, step=0.1)
        elif method == "IQR (Interquartile Range)":
            threshold = st.slider("IQR Multiplier", min_value=1.0, max_value=3.5, value=1.5, step=0.25)
        else:
            threshold = st.slider("% Deviation Threshold", min_value=10.0, max_value=200.0, value=50.0, step=5.0)

    # Run detection
    results = engine.detect_numerical_anomalies(df[selected_col], method=method, threshold=threshold)

    # Display KPI metrics
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.metric("Total Records", f"{results['total_observations']:,}")
    with kpi2:
        st.metric("Unusual Records", f"{results['unusual_count']:,}")
    with kpi3:
        st.metric("Anomaly Rate", f"{results['unusual_pct']:.2f}%")
    with kpi4:
        st.metric("Missing Count", f"{results['missing_observations']:,}")

    # Plot Distribution & Anomalies
    series = df[selected_col].dropna()
    anomalies_df = results["anomalies"]
    
    # Store in session state for export
    if not anomalies_df.empty:
        export_anom = anomalies_df.copy()
        export_anom["column"] = selected_col
        export_anom["method"] = method
        st.session_state["detection_artifacts"]["anomalies"] = export_anom

    fig = px.histogram(
        series,
        x=selected_col,
        nbins=40,
        title=f"Distribution of '{selected_col}' with Detected Anomaly Bounds",
        color_discrete_sequence=["#38BDF8"],
        opacity=0.75,
    )
    fig.update_layout(
        template="plotly_dark",
        plot_bgcolor="#0E131F",
        paper_bgcolor="#0E131F",
        margin=dict(l=40, r=40, t=50, b=40),
        font=dict(color="#F8FAFC"),
    )

    # Add threshold boundary lines if present in baseline_stats
    bounds = results.get("baseline_stats", {})
    if "lower_bound" in bounds and bounds["lower_bound"] is not None:
        fig.add_vline(x=bounds["lower_bound"], line_dash="dash", line_color="#F59E0B", annotation_text="Lower Bound")
    if "upper_bound" in bounds and bounds["upper_bound"] is not None:
        fig.add_vline(x=bounds["upper_bound"], line_dash="dash", line_color="#EF4444", annotation_text="Upper Bound")

    st.plotly_chart(fig, use_container_width=True)

    # Anomaly records table
    st.markdown(f"#### Detected Anomalous Observations ({len(anomalies_df):,} points)")
    if not anomalies_df.empty:
        st.dataframe(anomalies_df.head(100), use_container_width=True)
    else:
        st.success(f"No records exceeded the {threshold} threshold under {method}.")


def _render_distribution_tab(df: pd.DataFrame, candidates: Dict[str, List[str]], engine: UnexpectedDetectionEngine) -> None:
    """Render baseline vs comparison distribution comparison."""
    st.markdown("### 📊 Statistical Distribution Change Detection")
    st.caption("Compare distributions of numeric metrics across two selected cohorts or time frames.")

    split_col, b_val, c_val, b_df, c_df = _render_cohort_splitter(df, candidates)

    if b_df is None or c_df is None or b_df.empty or c_df.empty:
        st.info("Define a valid baseline and comparison cohort above to compute distribution shifts.")
        return

    num_cols = candidates["numerical"]
    if not num_cols:
        st.warning("No numerical variables available for distribution comparison.")
        return

    comp_col = st.selectbox("Select Numerical Metric to Compare", options=num_cols)

    # Run distribution comparison
    res = engine.compare_distributions(
        b_df[comp_col], c_df[comp_col], comp_col, baseline_label=str(b_val), comparison_label=str(c_val)
    )

    if res.get("status") == "success":
        st.session_state["detection_artifacts"]["distribution_summary"] = res["metrics_table"]

        # Display severity indicator
        sev_color = SEVERITY_COLORS.get(SeverityLevel(res["severity"]), "#94A3B8")
        st.markdown(
            f"""
            <div style="background: #111827; border-left: 4px solid {sev_color}; padding: 1rem; border-radius: 6px; margin: 1rem 0;">
                <div style="font-weight: 700; color: #F8FAFC;">Shift Severity: <span style="color: {sev_color};">{res['severity']}</span></div>
                <div style="font-size: 0.85rem; color: #CBD5E1; margin-top: 0.25rem;">{res['severity_reason']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        col_t, col_p = st.columns([1, 1.2])
        with col_t:
            st.markdown("#### Comparison Metrics Table")
            st.dataframe(res["metrics_table"], use_container_width=True, hide_index=True)

        with col_p:
            st.markdown("#### Comparative Distribution Density")
            b_plot = pd.DataFrame({comp_col: res["baseline_data"], "Cohort": str(b_val)})
            c_plot = pd.DataFrame({comp_col: res["comparison_data"], "Cohort": str(c_val)})
            comb = pd.concat([b_plot, c_plot], ignore_index=True)

            fig = px.box(
                comb,
                x="Cohort",
                y=comp_col,
                color="Cohort",
                color_discrete_map={str(b_val): "#38BDF8", str(c_val): "#F59E0B"},
                points="outliers",
                title=f"Boxplot Comparison: {comp_col}",
            )
            fig.update_layout(
                template="plotly_dark",
                plot_bgcolor="#0E131F",
                paper_bgcolor="#0E131F",
                margin=dict(l=40, r=40, t=50, b=40),
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning(res.get("message", "Unable to compare distributions."))


def _render_categorical_tab(df: pd.DataFrame, candidates: Dict[str, List[str]], engine: UnexpectedDetectionEngine) -> None:
    """Render categorical outcome and frequency change workbench."""
    st.markdown("### 🏷️ Categorical Outcome Change Detection")
    st.caption("Measure proportion shifts, newly emerging categories, and vanishing categories between cohorts.")

    split_col, b_val, c_val, b_df, c_df = _render_cohort_splitter(df, candidates, key_prefix="cat_split")

    if b_df is None or c_df is None or b_df.empty or c_df.empty:
        st.info("Define baseline and comparison cohorts above to inspect categorical swings.")
        return

    cat_cols = [c for c in df.columns if c != split_col and c not in candidates.get("constant", [])]
    if not cat_cols:
        st.warning("No categorical or outcome columns found.")
        return

    target_cat_col = st.selectbox("Select Categorical / Outcome Column", options=cat_cols)

    res = engine.compare_categories(
        b_df[target_cat_col],
        c_df[target_cat_col],
        target_cat_col,
        baseline_label=str(b_val),
        comparison_label=str(c_val),
    )

    if res.get("status") == "success":
        st.session_state["detection_artifacts"]["categorical_summary"] = res["table"]

        sev_color = SEVERITY_COLORS.get(SeverityLevel(res["highest_severity"]), "#94A3B8")
        st.markdown(
            f"""
            <div style="background: #111827; border-left: 4px solid {sev_color}; padding: 1rem; border-radius: 6px; margin: 1rem 0;">
                <div style="font-weight: 700; color: #F8FAFC;">Maximum Proportion Shift: <span style="color: {sev_color};">{res['max_percentage_point_shift']:.1f} percentage points ({res['highest_severity']})</span></div>
                <div style="font-size: 0.85rem; color: #CBD5E1; margin-top: 0.25rem;">{res['highest_severity_reason']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Plotly grouped bar chart
        table_df = res["table"]
        if not table_df.empty:
            melted = table_df.melt(
                id_vars=["Category"],
                value_vars=[f"{b_val} %", f"{c_val} %"],
                var_name="Cohort",
                value_name="Percentage (%)",
            )
            fig = px.bar(
                melted,
                x="Category",
                y="Percentage (%)",
                color="Cohort",
                barmode="group",
                color_discrete_map={f"{b_val} %": "#38BDF8", f"{c_val} %": "#EF4444"},
                title=f"Proportion Comparison for '{target_cat_col}' (%)",
            )
            fig.update_layout(
                template="plotly_dark",
                plot_bgcolor="#0E131F",
                paper_bgcolor="#0E131F",
                margin=dict(l=40, r=40, t=50, b=40),
            )
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Category Shift Frequency Table")
        st.dataframe(table_df, use_container_width=True, hide_index=True)


def _render_time_tab(df: pd.DataFrame, candidates: Dict[str, List[str]], engine: UnexpectedDetectionEngine) -> None:
    """Render time-based change, spike, and drop detection."""
    st.markdown("### ⏳ Time-Based Change Detection")
    st.caption("Inspect trends, sudden period-over-period spikes, crashes, and anomalous transitions across time.")

    date_cols = candidates.get("datetime", [])
    if not date_cols:
        # Check if user wants to force-parse a column as datetime
        all_cols = list(df.columns)
        st.warning("No automatic datetime columns detected in this dataset.")
        date_col = st.selectbox("Optionally select a column to parse as Date/Time", options=["None"] + all_cols)
        if date_col == "None":
            st.info("Time-based analysis is disabled when no valid timestamp column is present.")
            return
    else:
        date_col = st.selectbox("Select Datetime Column", options=date_cols)

    c1, c2, c3 = st.columns([1.5, 1, 1])
    with c1:
        metric_col = st.selectbox(
            "Select Metric to Track",
            options=candidates["numerical"] + candidates["categorical"],
        )
    with c2:
        freq_label = st.selectbox("Aggregation Frequency", options=["Monthly", "Weekly", "Daily", "Quarterly"])
        freq_map = {"Monthly": "M", "Weekly": "W", "Daily": "D", "Quarterly": "Q"}
        freq = freq_map[freq_label]
    with c3:
        spike_thresh = st.slider("Anomaly Spike/Drop Threshold (%)", min_value=10, max_value=100, value=30, step=5)

    res = engine.analyze_temporal_change(
        df, date_col, metric_col, freq=freq, agg_func="mean", anomaly_threshold_pct=float(spike_thresh)
    )

    if res.get("status") == "success":
        timeline = res["timeline"]
        spikes = res["spikes"]

        # KPI Metrics
        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Total Time Periods", res["periods_count"])
        with m2:
            st.metric("Detected Sudden Swings", res["unusual_periods_count"])
        with m3:
            ovr = res["overall_change_pct"]
            st.metric("Net Change Over Period", f"{ovr:+.1f}%" if ovr is not None else "N/A")

        # Plot Trend Line with highlighted spikes
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=timeline["period_str"],
            y=timeline["value"],
            mode="lines+markers",
            name=f"Mean {metric_col}",
            line=dict(color="#38BDF8", width=3),
            marker=dict(size=7),
        ))

        # Add red dots for spikes/drops
        if not spikes.empty:
            fig.add_trace(go.Scatter(
                x=spikes["period_str"],
                y=spikes["value"],
                mode="markers",
                name="Anomalous Swings",
                marker=dict(color="#EF4444", size=13, symbol="triangle-up"),
            ))

        fig.update_layout(
            title=f"Temporal Trend: {metric_col} ({freq_label})",
            template="plotly_dark",
            plot_bgcolor="#0E131F",
            paper_bgcolor="#0E131F",
            margin=dict(l=40, r=40, t=50, b=40),
            xaxis_title="Period",
            yaxis_title=metric_col,
        )
        st.plotly_chart(fig, use_container_width=True)

        if not spikes.empty:
            st.markdown("#### Detected Period-Over-Period Anomalous Shifts")
            st.dataframe(
                spikes[["period_str", "previous_value", "value", "absolute_change", "pct_change", "severity"]],
                use_container_width=True,
                hide_index=True,
            )
    else:
        st.warning(res.get("message", "Unable to compute temporal trends."))


def _render_segment_tab(df: pd.DataFrame, candidates: Dict[str, List[str]], engine: UnexpectedDetectionEngine) -> None:
    """Render segment-level disaggregation and ranked divergence table."""
    st.markdown("### 👥 Segment-Level Change Detection")
    st.caption("Identify which specific sub-demographics, customer tiers, or operational units experienced the greatest shifts.")

    split_col, b_val, c_val, b_df, c_df = _render_cohort_splitter(df, candidates, key_prefix="seg_split")

    if b_df is None or c_df is None or b_df.empty or c_df.empty:
        st.info("Define baseline and comparison cohorts above to rank segment divergence.")
        return

    c1, c2 = st.columns([1, 1])
    with c1:
        seg_col = st.selectbox("Select Grouping / Segment Column", options=candidates["categorical"])
    with c2:
        metric_col = st.selectbox("Select Metric or Target Outcome", options=candidates["numerical"] + candidates["categorical"])

    res = engine.analyze_segments(
        b_df, c_df, seg_col, metric_col, baseline_label=str(b_val), comparison_label=str(c_val)
    )

    if res.get("status") == "success":
        ranked = res["ranked_segments"]
        st.session_state["detection_artifacts"]["segment_summary"] = ranked

        st.markdown(f"#### Ranked Segment Divergence Table ({len(ranked)} segments)")
        st.dataframe(ranked, use_container_width=True, hide_index=True)

        # Plot horizontal bar chart of differences
        if not ranked.empty and "Absolute Difference" in ranked.columns:
            fig = px.bar(
                ranked,
                x="Absolute Difference",
                y="Segment",
                orientation="h",
                color="Severity",
                color_discrete_map={
                    SeverityLevel.CRITICAL.value: "#EF4444",
                    SeverityLevel.HIGH.value: "#F97316",
                    SeverityLevel.MODERATE.value: "#F59E0B",
                    SeverityLevel.LOW.value: "#06B6D4",
                    SeverityLevel.NORMAL.value: "#10B981",
                },
                title=f"Shift in '{metric_col}' by '{seg_col}'",
            )
            fig.update_layout(
                template="plotly_dark",
                plot_bgcolor="#0E131F",
                paper_bgcolor="#0E131F",
                margin=dict(l=40, r=40, t=50, b=40),
            )
            st.plotly_chart(fig, use_container_width=True)


def _render_insights_tab() -> None:
    """Render synthesized factual insights backed by empirical metrics."""
    st.markdown("### 💡 Automated Unexpected Outcome Insights")
    st.caption("Factual, non-causal synthesis of detected anomalies and shifts derived directly from calculations.")

    artifacts = st.session_state.get("detection_artifacts", {})
    insights_list = []

    # 1. Numerical Anomaly Insights
    anomalies = artifacts.get("anomalies", pd.DataFrame())
    if not anomalies.empty:
        col = anomalies["column"].iloc[0]
        n_anom = len(anomalies)
        insights_list.append({
            "title": f"Extreme Deviations in '{col}'",
            "severity": SeverityLevel.HIGH.value if n_anom > 5 else SeverityLevel.MODERATE.value,
            "finding": f"Identified {n_anom:,} anomalous data points in column '{col}' violating the statistical bounds.",
            "evidence": f"Count = {n_anom:,} | Method = {anomalies['method'].iloc[0]}",
            "method": "Numerical Deviation Detection",
        })

    # 2. Distribution Shift Insights
    dist_sum = artifacts.get("distribution_summary", pd.DataFrame())
    if not dist_sum.empty:
        mean_row = dist_sum[dist_sum["Metric"] == "Mean (Average)"]
        if not mean_row.empty:
            diff = mean_row["Absolute Difference"].iloc[0]
            pct = mean_row["% Change"].iloc[0]
            insights_list.append({
                "title": "Empirical Distribution Mean Shift",
                "severity": SeverityLevel.MODERATE.value,
                "finding": f"Average value exhibited an absolute difference of {diff} ({pct}) between the inspected cohorts.",
                "evidence": f"Calculated mean difference: {diff}",
                "method": "Distribution Shift Audit",
            })

    # 3. Categorical Swing Insights
    cat_sum = artifacts.get("categorical_summary", pd.DataFrame())
    if not cat_sum.empty:
        top_cat_shift = cat_sum.sort_values(
            by="Percentage Point Shift (pp)", key=abs, ascending=False
        ).iloc[0]
        insights_list.append({
            "title": f"Category Proportion Swing: '{top_cat_shift['Category']}'",
            "severity": top_cat_shift["Severity"],
            "finding": f"Category '{top_cat_shift['Category']}' shifted by {top_cat_shift['Percentage Point Shift (pp)']} percentage points ({top_cat_shift['Relative % Change']}).",
            "evidence": f"Baseline: {top_cat_shift.iloc[3]}% vs Comparison: {top_cat_shift.iloc[4]}%",
            "method": "Categorical Outcome Change Detection",
        })

    # 4. Segment Divergence Insights
    seg_sum = artifacts.get("segment_summary", pd.DataFrame())
    if not seg_sum.empty:
        top_seg = seg_sum.iloc[0]
        insights_list.append({
            "title": f"Largest Demographic Shift in Segment '{top_seg['Segment']}'",
            "severity": top_seg["Severity"],
            "finding": f"Subgroup '{top_seg['Segment']}' experienced the largest measured divergence (Diff: {top_seg['Absolute Difference']}, Change: {top_seg['% Change']}).",
            "evidence": top_seg["Justification"],
            "method": "Segment Disaggregation",
        })

    if not insights_list:
        st.info("Run one or more detection analyses in the tabs above to populate the automated factual insights dossier.")
        return

    st.session_state["detection_artifacts"]["insights"] = insights_list

    for ins in insights_list:
        sev_color = SEVERITY_COLORS.get(SeverityLevel(ins["severity"]), "#94A3B8")
        st.markdown(
            f"""
            <div style="background: #111827; border: 1px solid #1E293B; border-left: 4px solid {sev_color}; border-radius: 8px; padding: 1.2rem; margin-bottom: 1rem;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div style="font-weight: 700; font-size: 1.05rem; color: #F8FAFC;">{ins['title']}</div>
                    <span style="font-size: 0.75rem; font-weight: 700; padding: 2px 8px; border-radius: 4px; background: rgba(255,255,255,0.08); color: {sev_color};">
                        {ins['severity']}
                    </span>
                </div>
                <div style="font-size: 0.92rem; color: #E2E8F0; margin-top: 0.5rem; line-height: 1.5;">{ins['finding']}</div>
                <div style="font-size: 0.78rem; color: #94A3B8; margin-top: 0.6rem; padding-top: 0.5rem; border-top: 1px solid #1F2937;">
                    <strong>Evidence:</strong> {ins['evidence']} &nbsp;|&nbsp; <strong>Method:</strong> {ins['method']}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def _render_export_tab() -> None:
    """Render export and download controls."""
    st.markdown("### 📥 Export Detection Results")
    st.caption("Download all empirical anomaly data tables, change summaries, and generated insight logs as CSV.")

    artifacts = st.session_state.get("detection_artifacts", {})
    anomalies = artifacts.get("anomalies", pd.DataFrame())
    dist_sum = artifacts.get("distribution_summary", pd.DataFrame())
    cat_sum = artifacts.get("categorical_summary", pd.DataFrame())
    seg_sum = artifacts.get("segment_summary", pd.DataFrame())
    insights = artifacts.get("insights", [])

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("#### Numerical Anomalies")
        if not anomalies.empty:
            csv_anom = anomalies.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download Anomalies (CSV)", csv_anom, "detected_anomalies.csv", "text/csv", use_container_width=True)
        else:
            st.caption("No numerical anomalies currently generated.")

    with c2:
        st.markdown("#### Change Summaries")
        summary_df = dist_sum if not dist_sum.empty else (cat_sum if not cat_sum.empty else seg_sum)
        if not summary_df.empty:
            csv_sum = summary_df.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download Change Summary (CSV)", csv_sum, "change_summary.csv", "text/csv", use_container_width=True)
        else:
            st.caption("No change summaries currently generated.")

    with c3:
        st.markdown("#### Automated Insights")
        if insights:
            ins_df = pd.DataFrame(insights)
            csv_ins = ins_df.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download Insights Log (CSV)", csv_ins, "unexpected_insights.csv", "text/csv", use_container_width=True)
        else:
            st.caption("No insights currently generated.")


def _render_cohort_splitter(
    df: pd.DataFrame, candidates: Dict[str, List[str]], key_prefix: str = "split"
) -> tuple[Any, Any, Any, Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """Interactive control allowing users to partition the dataset into baseline and comparison cohorts."""
    with st.expander("⚖️ Define Cohort Comparison (Baseline vs. Comparison)", expanded=True):
        split_type = st.radio(
            "Comparison Strategy",
            options=["Categorical Grouping", "Time Period Cutoff", "Numerical Threshold Cutoff"],
            horizontal=True,
            key=f"{key_prefix}_type",
        )

        if split_type == "Categorical Grouping":
            cat_cols = candidates["categorical"] + candidates["boolean"]
            if not cat_cols:
                st.warning("No categorical or boolean columns found for cohort grouping.")
                return None, None, None, None, None
            split_col = st.selectbox("Partition Column", options=cat_cols, key=f"{key_prefix}_cat_col")
            unique_vals = list(df[split_col].dropna().unique())
            if len(unique_vals) < 2:
                st.warning(f"Column '{split_col}' has fewer than 2 distinct values.")
                return None, None, None, None, None
            c1, c2 = st.columns(2)
            with c1:
                b_val = st.selectbox("Baseline Cohort Value", options=unique_vals, index=0, key=f"{key_prefix}_b_val")
            with c2:
                c_val = st.selectbox(
                    "Comparison Cohort Value",
                    options=[v for v in unique_vals if v != b_val],
                    index=0,
                    key=f"{key_prefix}_c_val",
                )
            b_df, c_df, msg = split_baseline_comparison(df, "category", split_col, b_val, c_val)
            st.caption(f"✓ {msg}")
            return split_col, b_val, c_val, b_df, c_df

        elif split_type == "Time Period Cutoff":
            date_cols = candidates.get("datetime", [])
            if not date_cols:
                st.warning("No datetime column detected in dataset.")
                return None, None, None, None, None
            date_col = st.selectbox("Datetime Column", options=date_cols, key=f"{key_prefix}_date_col")
            d_series = pd.to_datetime(df[date_col], errors="coerce").dropna()
            min_date, max_date = d_series.min().date(), d_series.max().date()
            mid_date = min_date + (max_date - min_date) / 2
            cutoff_date = st.date_input("Cutoff Date (Baseline < Cutoff <= Comparison)", value=mid_date, key=f"{key_prefix}_date_cutoff")
            b_df, c_df, msg = split_baseline_comparison(df, "datetime_cutoff", date_col, str(cutoff_date))
            st.caption(f"✓ {msg}")
            return date_col, f"Before {cutoff_date}", f"From {cutoff_date}", b_df, c_df

        else:  # Numerical Threshold Cutoff
            num_cols = candidates["numerical"]
            if not num_cols:
                st.warning("No numerical columns found for numerical splitting.")
                return None, None, None, None, None
            num_col = st.selectbox("Numerical Splitting Column", options=num_cols, key=f"{key_prefix}_num_col")
            med_val = float(df[num_col].median())
            thresh = st.number_input(f"Threshold value (Baseline <= Threshold < Comparison)", value=med_val, key=f"{key_prefix}_num_thresh")
            b_df, c_df, msg = split_baseline_comparison(df, "numerical_cutoff", num_col, thresh)
            st.caption(f"✓ {msg}")
            return num_col, f"<= {thresh}", f"> {thresh}", b_df, c_df
