"""Dataset metrics, column classification, profiling, and quality UI components."""

from typing import Dict, Any, Optional
import streamlit as st
import pandas as pd
import numpy as np

from config.constants import ColumnCategory
from modules.data_profiler import DataProfiler
from components.metric_cards import render_metric_card


def render_dataset_metrics(
    df: Optional[pd.DataFrame],
    classifications: Optional[Dict[str, ColumnCategory]] = None,
) -> None:
    """Renders the comprehensive Phase 2 Data Understanding Engine."""
    if df is None or df.empty:
        return

    profiler = DataProfiler(df)

    # Re-use or initialize column classifications
    if classifications is None:
        if "column_classifications" in st.session_state and st.session_state["column_classifications"]:
            classifications = st.session_state["column_classifications"]
        else:
            classifications = profiler.identify_column_types(df)
            st.session_state["column_classifications"] = classifications

    # Generate complete profile metrics
    column_profiles = profiler.generate_column_profile(df, classifications)
    quality_audit = profiler.audit_data_quality(df, classifications)

    st.markdown("<div style='height: 25px;'></div>", unsafe_allow_html=True)
    st.markdown(
        """
        <div style="margin-bottom: 1.25rem;">
            <h2 style="font-size: 1.35rem; font-weight: 700; color: #F8FAFC; margin-bottom: 0.25rem;">
                📊 Data Understanding & Quality Profiling
            </h2>
            <p style="font-size: 0.88rem; color: #94A3B8; margin-bottom: 0;">
                Comprehensive structural audit, statistical distribution analysis, column classifications, and target variable selection.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 1. High-Level Executive Metric Cards
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        render_metric_card(
            title="Total Records",
            value=f"{df.shape[0]:,}",
            subtitle=f"{df.shape[1]} Features",
            badge_text="Volume",
            badge_type="blue",
        )
    with col2:
        missing_cells = quality_audit["total_missing_cells"]
        missing_pct = quality_audit["missing_percentage"]
        badge_type = "emerald" if missing_cells == 0 else ("amber" if missing_pct < 10 else "purple")
        render_metric_card(
            title="Missing Cells",
            value=f"{missing_cells:,}",
            subtitle=f"{missing_pct}% of total values",
            badge_text="Clean" if missing_cells == 0 else "Missing",
            badge_type=badge_type,
        )
    with col3:
        dup_rows = quality_audit["duplicate_rows"]
        dup_pct = quality_audit["duplicate_percentage"]
        badge_type = "emerald" if dup_rows == 0 else "amber"
        render_metric_card(
            title="Duplicate Rows",
            value=f"{dup_rows:,}",
            subtitle=f"{dup_pct}% exact matches",
            badge_text="Unique" if dup_rows == 0 else "Duplicate",
            badge_type=badge_type,
        )
    with col4:
        health_score = quality_audit["quality_score"]
        badge_type = "emerald" if health_score >= 90 else ("amber" if health_score >= 70 else "purple")
        render_metric_card(
            title="Data Health Score",
            value=f"{health_score}%",
            subtitle="Composite Quality",
            badge_text="High" if health_score >= 85 else "Review",
            badge_type=badge_type,
        )
    with col5:
        target_name = st.session_state.get("selected_target_variable")
        target_display = target_name if target_name else "Unassigned"
        badge_text = "Target Set" if target_name else "Select Target"
        render_metric_card(
            title="Target Outcome",
            value=target_display[:12] + ("..." if target_name and len(target_name) > 12 else ""),
            subtitle="Investigation Guide",
            badge_text=badge_text,
            badge_type="purple" if target_name else "blue",
        )

    st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)

    # 2. Tabbed Deep-Dive Modules
    tabs = st.tabs([
        "🏷️ Column Explorer & Override",
        "📐 Column Profiling & Statistics",
        "🩺 Data Quality Audit",
        "🎯 Target Variable Selection",
        "💡 Automatic Dataset Insights",
    ])

    # TAB 1: Column Explorer
    with tabs[0]:
        _render_column_explorer_tab(df, classifications)

    # TAB 2: Profiling & Statistics
    with tabs[1]:
        _render_profiling_tab(df, column_profiles)

    # TAB 3: Data Quality Audit
    with tabs[2]:
        _render_quality_tab(quality_audit)

    # TAB 4: Target Variable Selection
    with tabs[3]:
        _render_target_selection_tab(df, profiler)

    # TAB 5: Automatic Dataset Insights
    with tabs[4]:
        _render_insights_tab(df, profiler, classifications)


def _render_column_explorer_tab(
    df: pd.DataFrame, classifications: Dict[str, ColumnCategory]
) -> None:
    """Render column classification tables with manual reclassification capability."""
    st.markdown(
        """
        <div style="margin-bottom: 0.75rem;">
            <h4 style="font-size: 1.05rem; font-weight: 700; color: #F8FAFC;">
                Automated Semantic Column Identification
            </h4>
            <p style="font-size: 0.85rem; color: #94A3B8;">
                Columns are categorized using cardinality, distributions, unique ratios, and naming patterns.
                If any column was classified incorrectly, you can adjust its classification below.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Summary count pill badges by category
    category_counts: Dict[str, int] = {}
    for cat in classifications.values():
        cname = cat.value if hasattr(cat, "value") else str(cat)
        category_counts[cname] = category_counts.get(cname, 0) + 1

    pill_html = "".join(
        f'<span style="background: #1E293B; border: 1px solid #334155; color: #38BDF8; padding: 4px 10px; border-radius: 9999px; font-size: 0.78rem; font-weight: 600; margin-right: 8px; display: inline-block; margin-bottom: 6px;">{name}: {cnt}</span>'
        for name, cnt in sorted(category_counts.items())
    )
    st.markdown(f'<div style="margin-bottom: 1rem;">{pill_html}</div>', unsafe_allow_html=True)

    # Column details table
    col_records = []
    for c in df.columns:
        cat = classifications.get(str(c), ColumnCategory.CATEGORICAL)
        val = cat.value if hasattr(cat, "value") else str(cat)
        col_records.append({
            "Column Name": c,
            "Inferred Category": val,
            "Pandas Dtype": str(df[c].dtype),
            "Non-Null Count": int(df[c].notna().sum()),
            "Unique Count": int(df[c].nunique()),
            "Cardinality Ratio": f"{df[c].nunique() / len(df):.4f}" if len(df) > 0 else "0",
        })

    col_df = pd.DataFrame(col_records)
    st.dataframe(col_df, use_container_width=True, hide_index=True)

    # Interactive Reclassification Override
    with st.expander("✏️ Reclassify Column Semantic Type", expanded=False):
        c_left, c_right, c_btn = st.columns([1.5, 1.5, 1.0])
        with c_left:
            col_to_edit = st.selectbox("Select Column to Reclassify:", options=list(df.columns), key="col_override_name")
        with c_right:
            current_cat = classifications.get(str(col_to_edit), ColumnCategory.CATEGORICAL)
            all_categories = list(ColumnCategory)
            default_idx = all_categories.index(current_cat) if current_cat in all_categories else 0
            new_cat = st.selectbox(
                "New Semantic Category:",
                options=all_categories,
                index=default_idx,
                format_func=lambda x: x.value,
                key="col_override_category",
            )
        with c_btn:
            st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
            if st.button("Apply Reclassification", key="btn_apply_reclass", use_container_width=True):
                classifications[str(col_to_edit)] = new_cat
                st.session_state["column_classifications"] = classifications
                st.success(f"Column '{col_to_edit}' reclassified to '{new_cat.value}'.")
                st.rerun()


def _render_profiling_tab(
    df: pd.DataFrame, column_profiles: list
) -> None:
    """Render statistical profiles separated by numerical vs categorical variables."""
    st.markdown(
        """
        <div style="margin-bottom: 0.75rem;">
            <h4 style="font-size: 1.05rem; font-weight: 700; color: #F8FAFC;">
                Statistical Characteristics & Distributions
            </h4>
            <p style="font-size: 0.85rem; color: #94A3B8;">
                Detailed metrics computed only where statistically appropriate for each column's mathematical nature.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 1. Numerical Variables Table
    num_profiles = [p for p in column_profiles if p.get("mean") is not None]
    if num_profiles:
        st.markdown(
            "<div style='font-size: 0.92rem; font-weight: 700; color: #E2E8F0; margin-top: 0.5rem; margin-bottom: 0.4rem;'>📈 Numerical Features (Continuous & Discrete)</div>",
            unsafe_allow_html=True,
        )
        num_rows = []
        for p in num_profiles:
            num_rows.append({
                "Column": p["column_name"],
                "Category": p["inferred_category"],
                "Non-Null": f"{p['non_null_count']:,}",
                "Missing %": f"{p['missing_percentage']}%",
                "Unique": f"{p['unique_values']:,}",
                "Min": p["min"],
                "Mean": p["mean"],
                "Median": p["median"],
                "Max": p["max"],
                "Std Dev": p["std"],
                "Mode": p["mode"],
            })
        st.dataframe(pd.DataFrame(num_rows), use_container_width=True, hide_index=True)

    # 2. Categorical & Qualitative Variables Table
    cat_profiles = [p for p in column_profiles if p.get("top_categories")]
    if cat_profiles:
        st.markdown(
            "<div style='font-size: 0.92rem; font-weight: 700; color: #E2E8F0; margin-top: 1.25rem; margin-bottom: 0.4rem;'>🔠 Categorical & Discrete Features</div>",
            unsafe_allow_html=True,
        )
        cat_rows = []
        for p in cat_profiles:
            top_cat = p["top_categories"][0] if p["top_categories"] else {}
            top_cat_summary = f"{top_cat.get('category', 'N/A')} ({top_cat.get('percentage', 0)}%)" if top_cat else "N/A"
            cat_rows.append({
                "Column": p["column_name"],
                "Category": p["inferred_category"],
                "Non-Null": f"{p['non_null_count']:,}",
                "Missing %": f"{p['missing_percentage']}%",
                "Unique Categories": f"{p['unique_values']:,}",
                "Most Frequent (Mode)": top_cat_summary,
            })
        st.dataframe(pd.DataFrame(cat_rows), use_container_width=True, hide_index=True)

        # Inspect specific categorical frequency breakdown
        with st.expander("🔍 Inspect Categorical Frequency Distributions", expanded=False):
            cat_cols = [p["column_name"] for p in cat_profiles]
            selected_cat_col = st.selectbox("Select Categorical Column:", options=cat_cols, key="inspect_cat_dropdown")
            match_prof = next((p for p in cat_profiles if p["column_name"] == selected_cat_col), None)
            if match_prof and match_prof["top_categories"]:
                freq_df = pd.DataFrame(match_prof["top_categories"])
                c1, c2 = st.columns([1.2, 1.8])
                with c1:
                    st.dataframe(freq_df, use_container_width=True, hide_index=True)
                with c2:
                    st.bar_chart(freq_df.set_index("category")["count"], color="#4F46E5")


def _render_quality_tab(quality_audit: Dict[str, Any]) -> None:
    """Render data health report and warnings."""
    st.markdown(
        """
        <div style="margin-bottom: 0.75rem;">
            <h4 style="font-size: 1.05rem; font-weight: 700; color: #F8FAFC;">
                Data Quality & Health Audit
            </h4>
            <p style="font-size: 0.85rem; color: #94A3B8;">
                Automated detection of data defects: missingness, duplicate rows, zero-variance columns, and cardinality anomalies.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    q_col1, q_col2, q_col3 = st.columns(3)
    with q_col1:
        st.markdown(
            f"""
            <div style="background: #111827; border: 1px solid #1E293B; border-radius: 8px; padding: 1rem;">
                <div style="font-size: 0.78rem; color: #94A3B8; text-transform: uppercase;">Missing Cell Count</div>
                <div style="font-size: 1.4rem; font-weight: 700; color: #F8FAFC; margin-top: 0.2rem;">
                    {quality_audit['total_missing_cells']:,} <span style="font-size: 0.85rem; color: #94A3B8;">({quality_audit['missing_percentage']}%)</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with q_col2:
        st.markdown(
            f"""
            <div style="background: #111827; border: 1px solid #1E293B; border-radius: 8px; padding: 1rem;">
                <div style="font-size: 0.78rem; color: #94A3B8; text-transform: uppercase;">Duplicate Record Rows</div>
                <div style="font-size: 1.4rem; font-weight: 700; color: #F8FAFC; margin-top: 0.2rem;">
                    {quality_audit['duplicate_rows']:,} <span style="font-size: 0.85rem; color: #94A3B8;">({quality_audit['duplicate_percentage']}%)</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with q_col3:
        st.markdown(
            f"""
            <div style="background: #111827; border: 1px solid #1E293B; border-radius: 8px; padding: 1rem;">
                <div style="font-size: 0.78rem; color: #94A3B8; text-transform: uppercase;">Constant Columns</div>
                <div style="font-size: 1.4rem; font-weight: 700; color: #F8FAFC; margin-top: 0.2rem;">
                    {len(quality_audit['constant_columns'])} <span style="font-size: 0.85rem; color: #94A3B8;">(0 Variance)</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)
    st.markdown(
        "<div style='font-size: 0.95rem; font-weight: 700; color: #F1F5F9; margin-bottom: 0.5rem;'>Data Quality Audit Findings & Warnings:</div>",
        unsafe_allow_html=True,
    )

    for warning in quality_audit["warnings"]:
        if "Clean Dataset" in warning:
            st.success(f"✅ {warning}")
        elif "Missing Values" in warning or "Duplicate" in warning or "Constant" in warning:
            st.warning(f"⚠️ {warning}")
        else:
            st.info(f"ℹ️ {warning}")


def _render_target_selection_tab(
    df: pd.DataFrame, profiler: DataProfiler
) -> None:
    """Interactive target variable selection and outcome definition."""
    st.markdown(
        """
        <div style="margin-bottom: 0.75rem;">
            <h4 style="font-size: 1.05rem; font-weight: 700; color: #F8FAFC;">
                Target Variable & Investigation Outcome Selection
            </h4>
            <p style="font-size: 0.85rem; color: #94A3B8;">
                Choose the metric, event, or phenomenon you want to investigate. The system will isolate contributing factors behind this target.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    all_columns = list(df.columns)
    saved_target = st.session_state.get("selected_target_variable")
    default_target_idx = all_columns.index(saved_target) if saved_target in all_columns else 0

    col_select, col_outcome = st.columns([1.2, 1.8])

    with col_select:
        selected_col = st.selectbox(
            "Select Target Column for Investigation:",
            options=all_columns,
            index=default_target_idx,
            key="target_col_picker",
        )

        target_audit = profiler.inspect_target_variable(df, selected_col)

        if not target_audit["valid"]:
            st.error(f"❌ Cannot select '{selected_col}': {target_audit.get('error')}")
            return

        st.session_state["selected_target_variable"] = selected_col

        st.markdown(
            f"""
            <div style="background: #111827; border: 1px solid #1E293B; border-radius: 8px; padding: 1rem; margin-top: 0.75rem;">
                <div style="font-size: 0.78rem; color: #94A3B8; text-transform: uppercase;">Target Classification</div>
                <div style="font-size: 1.1rem; font-weight: 700; color: #38BDF8; margin-top: 0.2rem;">
                    {target_audit['target_type']}
                </div>
                <div style="font-size: 0.8rem; color: #CBD5E1; margin-top: 0.35rem;">
                    Distinct values: <strong>{target_audit['unique_count']:,}</strong> | Missing: {target_audit['missing_records']:,}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_outcome:
        possible_outcomes = target_audit.get("possible_outcomes", [])
        outcome_labels = [opt["label"] for opt in possible_outcomes]

        selected_outcome_label = st.selectbox(
            "Select the Specific Outcome or Condition to Investigate:",
            options=outcome_labels,
            key="outcome_picker",
        )

        # Store chosen outcome in session
        matched_outcome = next((o for o in possible_outcomes if o["label"] == selected_outcome_label), None)
        st.session_state["selected_outcome"] = matched_outcome

        st.success(
            f"🎯 Investigation Target Configured: **'{selected_col}'** → Outcome: **'{selected_outcome_label}'**."
        )

        # Target Distribution Visualizer
        st.markdown(
            "<div style='font-size: 0.88rem; font-weight: 600; color: #E2E8F0; margin-top: 0.75rem; margin-bottom: 0.25rem;'>Target Distribution:</div>",
            unsafe_allow_html=True,
        )
        dist = target_audit.get("distribution", {})
        if "categories" in dist:
            chart_df = pd.DataFrame({
                "Category": dist["categories"],
                "Count": dist["counts"],
            }).set_index("Category")
            st.bar_chart(chart_df, color="#38BDF8", height=200)
        elif "values" in dist:
            # Numerical histogram
            vals = dist["values"]
            counts, bin_edges = np.histogram(vals, bins=20)
            hist_df = pd.DataFrame({
                "Bin Range": [f"{bin_edges[i]:.1f}-{bin_edges[i+1]:.1f}" for i in range(len(counts))],
                "Count": counts,
            }).set_index("Bin Range")
            st.bar_chart(hist_df, color="#818CF8", height=200)


def _render_insights_tab(
    df: pd.DataFrame,
    profiler: DataProfiler,
    classifications: Dict[str, ColumnCategory],
) -> None:
    """Render automatically generated factual dataset insights."""
    st.markdown(
        """
        <div style="margin-bottom: 0.75rem;">
            <h4 style="font-size: 1.05rem; font-weight: 700; color: #F8FAFC;">
                Automatic Factual Dataset Insights
            </h4>
            <p style="font-size: 0.85rem; color: #94A3B8;">
                Real-time observations derived directly from your uploaded data, highlighting dimensions, data health, and target distribution.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    target_name = st.session_state.get("selected_target_variable")
    target_info = profiler.inspect_target_variable(df, target_name) if target_name else None

    factual_insights = profiler.generate_factual_insights(
        df=df,
        classifications=classifications,
        target_info=target_info,
    )

    for i, insight in enumerate(factual_insights, 1):
        st.markdown(
            f"""
            <div style="background: #111827; border-left: 3px solid #4F46E5; border-radius: 0 8px 8px 0; padding: 0.75rem 1rem; margin-bottom: 0.6rem; font-size: 0.88rem; color: #E2E8F0;">
                <span style="color: #818CF8; font-weight: 700; margin-right: 0.5rem;">{i}.</span> {insight}
            </div>
            """,
            unsafe_allow_html=True,
        )
