"""Enterprise UI Component for Intelligent Data Cleaning & Preprocessing Engine.

Provides interactive tabs for data quality inspection, column-level treatments,
before-and-after visual comparisons, transformation history, and dataset export.
"""

from typing import Optional, List, Dict, Any
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from config.constants import APP_NAME
from modules.data_loader import DataLoader
from modules.preprocessing import (
    PreprocessingEngine,
    detect_missing_values,
    get_missing_summary,
    detect_duplicates,
    get_numerical_columns,
    detect_column_outliers,
    detect_all_outliers,
    get_categorical_columns,
    detect_categorical_inconsistencies,
    inspect_data_types,
    build_missing_comparison_chart,
    build_outlier_comparison_chart,
    build_distribution_comparison_chart,
)
from components.metric_cards import render_metric_card


def render_preprocessing_page() -> None:
    """Render the comprehensive Phase 3 Preprocessing & Data Cleaning Dashboard."""
    # Ensure session state initialization
    if "uploaded_dataset" not in st.session_state:
        st.session_state["uploaded_dataset"] = None
    if "dataset_filename" not in st.session_state:
        st.session_state["dataset_filename"] = "dataset.csv"
    if "selected_target_variable" not in st.session_state:
        st.session_state["selected_target_variable"] = None

    # Header banner
    st.markdown(
        """
        <div class="hero-banner" style="margin-bottom: 1.5rem; padding: 1.8rem 2.2rem;">
            <div class="hero-title" style="font-size: 2rem;">Intelligent Data Cleaning & Preprocessing</div>
            <div class="hero-tagline" style="font-size: 1rem; color: #38BDF8;">
                Detect quality anomalies, configure column-level treatments, and audit transformations in real time.
            </div>
            <p class="hero-description" style="font-size: 0.88rem;">
                Eliminate missing values, duplicate records, numerical outliers, casing discrepancies, and incorrect types
                while preventing target data leakage for explainable AI investigation.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    df_active = st.session_state.get("uploaded_dataset")
    filename = st.session_state.get("dataset_filename", "dataset.csv")
    target_col = st.session_state.get("selected_target_variable")

    # If no dataset loaded, provide instant benchmark dataset loader or direct uploader
    if df_active is None:
        st.info("ℹ️ **No active dataset detected in memory.** You can load our pre-configured benchmark dataset with 10 real-world quality defects or upload your own file below.")
        
        col_load, col_up = st.columns([1.2, 1.0])
        with col_load:
            st.markdown(
                """
                <div class="section-card" style="margin-top: 0; min-height: 220px;">
                    <div class="section-card-title">🧪 Load Benchmark Messy Dataset</div>
                    <p class="section-card-desc">
                        Instant 1-click dataset (258 rows) with missing values, extreme outliers, duplicated records,
                        casing inconsistencies, currency strings, invalid dates, rare categories, and constant columns.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if st.button("🚀 Load Benchmark Quality Challenge Dataset", type="primary", use_container_width=True):
                loader = DataLoader()
                bench_df = loader.create_benchmark_messy_dataset()
                st.session_state["uploaded_dataset"] = bench_df
                st.session_state["dataset_filename"] = "benchmark_quality_challenge.csv"
                st.session_state["selected_target_variable"] = "churn"
                st.session_state["preprocessing_engine"] = PreprocessingEngine(
                    bench_df, target_column="churn", dataset_name="benchmark_quality_challenge"
                )
                st.success("Loaded benchmark dataset successfully! Refreshing dashboard...")
                st.rerun()

        with col_up:
            st.markdown(
                """
                <div class="section-card" style="margin-top: 0; min-height: 220px;">
                    <div class="section-card-title">📁 Upload Custom Dataset</div>
                    <p class="section-card-desc">
                        Upload your CSV, Excel (.xlsx, .xls), or Parquet dataset for automated data cleaning.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            uploaded_file = st.file_uploader(
                "Upload dataset file",
                type=["csv", "xlsx", "xls", "parquet"],
                key="preprocessing_direct_uploader",
                label_visibility="collapsed",
            )
            if uploaded_file is not None:
                try:
                    loader = DataLoader()
                    res = loader.load_file(uploaded_file, file_name=uploaded_file.name)
                    df_custom = res[0] if isinstance(res, tuple) else res
                    st.session_state["uploaded_dataset"] = df_custom
                    st.session_state["dataset_filename"] = uploaded_file.name
                    st.session_state["preprocessing_engine"] = PreprocessingEngine(
                        df_custom, dataset_name=uploaded_file.name.split(".")[0]
                    )
                    st.success(f"Loaded '{uploaded_file.name}' ({len(df_custom):,} rows)!")
                    st.rerun()
                except Exception as ex:
                    st.error(f"Failed to parse file: {str(ex)}")
        return

    # Initialize or synchronize PreprocessingEngine
    if "preprocessing_engine" not in st.session_state or st.session_state["preprocessing_engine"] is None:
        st.session_state["preprocessing_engine"] = PreprocessingEngine(
            df_active, target_column=target_col, dataset_name=filename.split(".")[0]
        )

    engine: PreprocessingEngine = st.session_state["preprocessing_engine"]

    # Global Quick Control Bar
    top_c1, top_c2, top_c3, top_c4 = st.columns([1.5, 0.9, 0.9, 0.9])
    with top_c1:
        st.markdown(
            f"""
            <div style="font-size: 0.85rem; color: #94A3B8; padding-top: 0.4rem;">
                Active Working Copy: <strong style="color: #38BDF8;">{len(engine.current_df):,} rows × {engine.current_df.shape[1]} columns</strong>
                &nbsp;|&nbsp; Operations Applied: <strong style="color: #34D399;">{len(engine.history)}</strong>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with top_c2:
        if st.button("↩️ Undo Last Action", use_container_width=True, disabled=len(engine.history) == 0):
            if engine.undo_last_step():
                st.toast("Reverted last preprocessing transformation.", icon="↩️")
                st.rerun()
    with top_c3:
        if st.button("🔄 Reset to Original", use_container_width=True, disabled=len(engine.history) == 0):
            engine.reset_to_original()
            st.toast("Dataset reset back to original state.", icon="🔄")
            st.rerun()
    with top_c4:
        if st.button("💾 Apply as Active", use_container_width=True, type="secondary"):
            st.session_state["uploaded_dataset"] = engine.current_df.copy()
            st.toast("Cleaned dataset synced to active session memory!", icon="✅")

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

    # 11 Dedicated Preprocessing Tabs
    tabs = st.tabs(
        [
            "🩺 A. Quality Overview",
            "🩹 B. Missing Values",
            "👥 C. Duplicates",
            "⚡ D. Outlier Detection",
            "🔤 E. Categorical Cleaning",
            "🔄 F. Data Types",
            "🏷️ G. Feature Encoding",
            "⚖️ H. Feature Scaling",
            "📊 I. Before & After",
            "📜 J. History Audit",
            "💾 K. Export Dataset",
        ]
    )

    # =========================================================================
    # TAB A: DATA QUALITY OVERVIEW
    # =========================================================================
    with tabs[0]:
        st.markdown("### 🩺 Data Quality Health Overview")
        st.markdown(
            "An enterprise health diagnostic summarizing null values, duplicate records, and outliers across the active dataset."
        )

        missing_sum = get_missing_summary(engine.current_df)
        dup_info = detect_duplicates(engine.current_df)
        outliers_df = detect_all_outliers(engine.current_df)
        total_outliers = int(outliers_df["Outlier Count"].sum()) if not outliers_df.empty else 0

        # Health score calculation: 100 - penalties
        penalty_missing = min(40, missing_sum["missing_percentage"] * 1.5)
        penalty_dups = min(30, dup_info["duplicate_percentage"] * 3.0)
        penalty_outliers = min(30, (total_outliers / max(1, len(engine.current_df))) * 50)
        health_score = max(0, round(100 - (penalty_missing + penalty_dups + penalty_outliers), 1))

        # KPI Cards
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        with kpi1:
            render_metric_card(
                title="Data Quality Health",
                value=f"{health_score}%",
                subtitle="Aggregated Health Index",
                badge_text="Audited",
                badge_type="emerald" if health_score >= 80 else ("amber" if health_score >= 50 else "blue"),
            )
        with kpi2:
            render_metric_card(
                title="Missing Cells",
                value=f"{missing_sum['total_missing_cells']:,}",
                subtitle=f"{missing_sum['missing_percentage']}% of all cells",
                badge_text=f"{missing_sum['cols_with_missing']} cols affected",
                badge_type="amber" if missing_sum["total_missing_cells"] > 0 else "emerald",
            )
        with kpi3:
            render_metric_card(
                title="Duplicate Records",
                value=f"{dup_info['duplicate_count']:,}",
                subtitle=f"{dup_info['duplicate_percentage']}% of rows",
                badge_text="Exact Duplicates",
                badge_type="amber" if dup_info["duplicate_count"] > 0 else "emerald",
            )
        with kpi4:
            render_metric_card(
                title="Potential Outliers",
                value=f"{total_outliers:,}",
                subtitle="IQR bounds violations",
                badge_text=f"{len(outliers_df)} numeric cols",
                badge_type="purple" if total_outliers > 0 else "emerald",
            )

        st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)

        # Missingness & Distribution Charts
        col_chart1, col_chart2 = st.columns([1.1, 0.9])
        with col_chart1:
            miss_df = detect_missing_values(engine.current_df)
            miss_with_nulls = miss_df[miss_df["missing_count"] > 0]
            if not miss_with_nulls.empty:
                fig_miss = px.bar(
                    miss_with_nulls,
                    x="column",
                    y="missing_percentage",
                    title="Missing Values by Column (%)",
                    labels={"column": "Column", "missing_percentage": "Missing %"},
                    color="missing_percentage",
                    color_continuous_scale="Reds",
                    template="plotly_dark",
                )
                fig_miss.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    margin=dict(l=20, r=20, t=40, b=20),
                )
                st.plotly_chart(fig_miss, use_container_width=True)
            else:
                st.success("✅ Clean Dataset: Zero missing values detected across all columns.")

        with col_chart2:
            st.markdown("#### 📋 Column Quality Summary")
            st.dataframe(
                miss_df[["column", "dtype", "missing_count", "missing_percentage", "non_null_count"]],
                use_container_width=True,
                height=320,
            )

    # =========================================================================
    # TAB B: MISSING VALUE TREATMENT
    # =========================================================================
    with tabs[1]:
        st.markdown("### 🩹 Missing Value Treatment")
        st.markdown(
            "Detect missing values column-by-column and apply tailored imputation or dropping strategies."
        )

        miss_df = detect_missing_values(engine.current_df)
        columns_with_missing = miss_df[miss_df["missing_count"] > 0]["column"].tolist()

        if not columns_with_missing:
            st.success("🎉 No missing values exist in the active dataset!")
        else:
            col_sel, col_treat = st.columns([1.0, 1.2])

            with col_sel:
                st.markdown("#### 1. Select Column to Treat")
                selected_col = st.selectbox(
                    "Column with Missing Values",
                    options=columns_with_missing,
                    format_func=lambda c: f"{c} ({engine.current_df[c].isna().sum():,} nulls, {round(engine.current_df[c].isna().sum()/len(engine.current_df)*100, 1)}%)",
                )
                col_is_num = pd.api.types.is_numeric_dtype(engine.current_df[selected_col])
                null_cnt = int(engine.current_df[selected_col].isna().sum())
                null_pct = round((null_cnt / len(engine.current_df)) * 100, 2)

                st.markdown(
                    f"""
                    <div style="background: #111827; border: 1px solid #1F2937; border-radius: 8px; padding: 1rem; margin-top: 0.5rem;">
                        <div style="font-size: 0.85rem; color: #CBD5E1;">Column: <strong>{selected_col}</strong></div>
                        <div style="font-size: 0.82rem; color: #94A3B8; margin-top: 0.25rem;">Type: <code>{engine.current_df[selected_col].dtype}</code></div>
                        <div style="font-size: 0.82rem; color: #F59E0B; margin-top: 0.25rem;">Missing: <strong>{null_cnt:,} rows ({null_pct}%)</strong></div>
                        <div style="font-size: 0.82rem; color: #34D399; margin-top: 0.25rem;">Non-Null: {len(engine.current_df) - null_cnt:,} rows</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                if null_cnt == len(engine.current_df):
                    st.warning(f"⚠️ Column '{selected_col}' contains 100% missing values. Mean/Median will safely fall back to constant.")

            with col_treat:
                st.markdown("#### 2. Choose Imputation Strategy")
                if col_is_num:
                    num_strategy = st.radio(
                        "Numerical Imputation Method",
                        options=["median", "mean", "constant", "ffill", "bfill"],
                        format_func=lambda s: {
                            "median": "Median (Robust to outliers - Recommended)",
                            "mean": "Mean (Arithmetic average)",
                            "constant": "Constant Value",
                            "ffill": "Forward Fill (Time/Ordered data)",
                            "bfill": "Backward Fill (Time/Ordered data)",
                        }[s],
                    )
                    fill_val = 0.0
                    if num_strategy == "constant":
                        fill_val = st.number_input("Constant Value", value=0.0)

                    if st.button("Apply Numerical Imputation", type="primary", use_container_width=True):
                        step = engine.impute_missing(selected_col, strategy=num_strategy, fill_value=fill_val)
                        st.success(step.details)
                        st.rerun()

                else:
                    cat_strategy = st.radio(
                        "Categorical Imputation Method",
                        options=["mode", "unknown", "custom", "ffill", "bfill"],
                        format_func=lambda s: {
                            "mode": "Mode (Most frequent category)",
                            "unknown": "Label as 'Unknown'",
                            "custom": "Custom Category Label",
                            "ffill": "Forward Fill",
                            "bfill": "Backward Fill",
                        }[s],
                    )
                    fill_cat_val = "Unknown"
                    if cat_strategy == "custom":
                        fill_cat_val = st.text_input("Custom Category Value", value="Missing_Value")

                    if st.button("Apply Categorical Imputation", type="primary", use_container_width=True):
                        step = engine.impute_missing(selected_col, strategy=cat_strategy, fill_value=fill_cat_val)
                        st.success(step.details)
                        st.rerun()

            st.markdown("<hr style='border: none; border-top: 1px solid #1E293B; margin: 1.5rem 0;'>", unsafe_allow_html=True)

            # Advanced Dropping Options
            st.markdown("#### 🚨 Advanced Options: Drop Rows or Columns")
            drop_c1, drop_c2 = st.columns(2)

            with drop_c1:
                st.markdown(
                    """
                    <div style="background: rgba(239, 68, 68, 0.05); border: 1px solid rgba(239, 68, 68, 0.2); border-radius: 8px; padding: 1rem;">
                        <strong style="color: #F87171;">Drop Rows Containing Nulls</strong>
                        <p style="font-size: 0.8rem; color: #94A3B8; margin-top: 0.25rem;">
                            Removes records with missing values in selected columns. Never done automatically.
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                drop_cols_choice = st.multiselect(
                    "Columns to check for nulls (leave empty for all)",
                    options=columns_with_missing,
                    key="drop_rows_cols_selector",
                )
                confirm_row_drop = st.checkbox("I confirm dropping rows with missing values", key="confirm_drop_rows")
                if st.button("Drop Missing Rows", type="secondary", disabled=not confirm_row_drop):
                    step = engine.drop_missing(axis="rows", subset=drop_cols_choice if drop_cols_choice else None)
                    st.success(step.details)
                    st.rerun()

            with drop_c2:
                st.markdown(
                    """
                    <div style="background: rgba(239, 68, 68, 0.05); border: 1px solid rgba(239, 68, 68, 0.2); border-radius: 8px; padding: 1rem;">
                        <strong style="color: #F87171;">Drop Columns with Excessive Nulls</strong>
                        <p style="font-size: 0.8rem; color: #94A3B8; margin-top: 0.25rem;">
                            Discards columns where missing proportion exceeds a safety threshold.
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                thresh_slider = st.slider("Missing Threshold (%)", min_value=10, max_value=100, value=50, step=5)
                cols_to_drop_preview = [
                    c for c in columns_with_missing
                    if (engine.current_df[c].isna().sum() / len(engine.current_df) * 100) >= thresh_slider
                ]
                if cols_to_drop_preview:
                    st.caption(f"Columns matching threshold: **{', '.join(cols_to_drop_preview)}**")
                else:
                    st.caption("No columns exceed this threshold.")

                confirm_col_drop = st.checkbox("I confirm dropping high-missing columns", key="confirm_drop_cols")
                if st.button("Drop Columns", type="secondary", disabled=(not confirm_col_drop or len(cols_to_drop_preview) == 0)):
                    step = engine.drop_missing(axis="columns", threshold_pct=thresh_slider)
                    st.success(step.details)
                    st.rerun()

    # =========================================================================
    # TAB C: DUPLICATE RECORD HANDLING
    # =========================================================================
    with tabs[2]:
        st.markdown("### 👥 Duplicate Record Handling")
        st.markdown("Detect exact duplicate records and selectively eliminate redundant observations.")

        dup_stats = detect_duplicates(engine.current_df)
        d_c1, d_c2, d_c3 = st.columns(3)
        with d_c1:
            st.metric("Total Duplicate Rows", f"{dup_stats['duplicate_count']:,}")
        with d_c2:
            st.metric("Duplicate Percentage", f"{dup_stats['duplicate_percentage']}%")
        with d_c3:
            st.metric("Unique Records", f"{dup_stats['unique_rows']:,}")

        if not dup_stats["has_duplicates"]:
            st.success("✅ Clean Dataset: Zero duplicate records found!")
        else:
            st.warning(f"⚠️ Found **{dup_stats['duplicate_count']:,} duplicate rows** in the dataset.")
            
            with st.expander("🔍 Inspect Duplicate Records Sample", expanded=True):
                st.dataframe(dup_stats["duplicate_samples"], use_container_width=True, height=220)

            st.markdown("#### Duplicate Removal Strategy")
            col_keep, col_action = st.columns([1.0, 1.0])
            with col_keep:
                keep_strat = st.radio(
                    "Retention Strategy",
                    options=["first", "last"],
                    format_func=lambda k: "Retain First Occurrence (Drop subsequent)" if k == "first" else "Retain Last Occurrence (Drop preceding)",
                )
                confirm_dups = st.checkbox("I confirm removal of duplicate records", key="confirm_dup_remove")

            with col_action:
                st.markdown("<div style='height: 25px;'></div>", unsafe_allow_html=True)
                if st.button("Remove Duplicate Rows", type="primary", use_container_width=True, disabled=not confirm_dups):
                    step = engine.remove_duplicates(keep=keep_strat)
                    st.success(step.details)
                    st.rerun()

    # =========================================================================
    # TAB D: OUTLIER DETECTION AND HANDLING
    # =========================================================================
    with tabs[3]:
        st.markdown("### ⚡ Numerical Outlier Detection & Handling")
        st.markdown(
            "Detect statistical anomalies using Interquartile Range (IQR) or Z-score boundaries, "
            "and apply non-destructive capping or winsorization."
        )

        num_cols = get_numerical_columns(engine.current_df)
        if not num_cols:
            st.info("No continuous numerical columns available for outlier detection.")
        else:
            col_method, col_thresh = st.columns(2)
            with col_method:
                outlier_method = st.selectbox(
                    "Detection Method",
                    options=["iqr", "zscore"],
                    format_func=lambda m: "Interquartile Range (IQR) - 1.5×IQR" if m == "iqr" else "Z-Score (Standard Deviations)",
                )
            with col_thresh:
                default_thresh = 1.5 if outlier_method == "iqr" else 3.0
                outlier_threshold = st.number_input(
                    "Boundary Multiplier Threshold",
                    min_value=0.5,
                    max_value=5.0,
                    value=default_thresh,
                    step=0.25,
                )

            # Overview Table
            all_outliers = detect_all_outliers(engine.current_df, method=outlier_method, threshold=outlier_threshold)
            st.dataframe(all_outliers, use_container_width=True, height=180)

            st.markdown("<hr style='border: none; border-top: 1px solid #1E293B; margin: 1rem 0;'>", unsafe_allow_html=True)

            # Single Column Deep Dive & Treatment
            st.markdown("#### Column-Level Treatment")
            inspect_col = st.selectbox("Select Numerical Column", options=num_cols)
            col_info = detect_column_outliers(engine.current_df, inspect_col, method=outlier_method, threshold=outlier_threshold)

            c_viz, c_ctrl = st.columns([1.2, 1.0])
            with c_viz:
                # Plotly Box Plot with Scatter Points
                fig_box = px.box(
                    engine.current_df,
                    y=inspect_col,
                    points="outliers",
                    title=f"Box Plot: '{inspect_col}' with Bound Violations",
                    template="plotly_dark",
                )
                if col_info["lower_bound"] is not None:
                    fig_box.add_hline(
                        y=col_info["lower_bound"],
                        line_dash="dash",
                        line_color="#EF4444",
                        annotation_text=f"Lower Bound: {col_info['lower_bound']:.2f}",
                    )
                    fig_box.add_hline(
                        y=col_info["upper_bound"],
                        line_dash="dash",
                        line_color="#EF4444",
                        annotation_text=f"Upper Bound: {col_info['upper_bound']:.2f}",
                    )
                fig_box.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    margin=dict(l=20, r=20, t=40, b=20),
                )
                st.plotly_chart(fig_box, use_container_width=True)

            with c_ctrl:
                st.markdown(
                    f"""
                    <div style="background: #111827; border: 1px solid #1F2937; border-radius: 8px; padding: 0.9rem; margin-bottom: 0.8rem;">
                        <div style="font-size: 0.82rem; color: #94A3B8;">Outliers Detected: <strong style="color: #F59E0B;">{col_info['outlier_count']} ({col_info['outlier_pct']}%)</strong></div>
                        <div style="font-size: 0.82rem; color: #94A3B8; margin-top: 0.2rem;">Lower Bound: <code>{col_info['lower_bound']}</code> | Upper: <code>{col_info['upper_bound']}</code></div>
                        <div style="font-size: 0.82rem; color: #94A3B8; margin-top: 0.2rem;">Min: <code>{col_info['min']}</code> | Max: <code>{col_info['max']}</code></div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.info("💡 **Business Context Warning:** Outliers can represent real high-value events. Capping is recommended over row removal.")

                outlier_strat = st.radio(
                    "Outlier Treatment Strategy",
                    options=["keep", "iqr_cap", "percentile_cap", "remove"],
                    format_func=lambda s: {
                        "keep": "Keep original values (No modification)",
                        "iqr_cap": "Cap to IQR Lower/Upper Bounds",
                        "percentile_cap": "Percentile Capping (Winsorization)",
                        "remove": "Remove Outlier Records (Warning: Drops Rows)",
                    }[s],
                )

                p_low, p_high = 1.0, 99.0
                if outlier_strat == "percentile_cap":
                    p_low, p_high = st.slider("Percentile Cutoffs (Low & High)", 0.0, 100.0, (1.0, 99.0))

                confirm_outlier = True
                if outlier_strat == "remove":
                    confirm_outlier = st.checkbox(f"I confirm dropping {col_info['outlier_count']} rows", key="confirm_drop_outliers")

                if st.button("Apply Outlier Treatment", type="primary", use_container_width=True, disabled=not confirm_outlier):
                    step = engine.handle_outliers(
                        inspect_col,
                        strategy=outlier_strat,
                        method=outlier_method,
                        threshold=outlier_threshold,
                        percentile_limits=(p_low, p_high),
                    )
                    st.success(step.details)
                    st.rerun()

    # =========================================================================
    # TAB E: CATEGORICAL DATA CLEANING
    # =========================================================================
    with tabs[4]:
        st.markdown("### 🔤 Categorical Standardization & Consolidation")
        st.markdown(
            "Clean leading/trailing whitespace, resolve inconsistent capitalization, "
            "replace selected category values, and group low-frequency rare categories."
        )

        cat_cols = get_categorical_columns(engine.current_df)
        if not cat_cols:
            st.info("No categorical/text columns identified in the dataset.")
        else:
            cat_action = st.radio(
                "Categorical Operation",
                options=["strip_whitespace", "standardize_casing", "replace_values", "group_rare"],
                format_func=lambda a: {
                    "strip_whitespace": "1. Strip Leading/Trailing Whitespace",
                    "standardize_casing": "2. Standardize Inconsistent Capitalization",
                    "replace_values": "3. Replace Specific Category Labels",
                    "group_rare": "4. Group Rare Categories into 'Other'",
                }[a],
                horizontal=True,
            )

            if cat_action == "strip_whitespace":
                st.markdown("#### Strip Leading & Trailing Whitespace")
                st.markdown("Removes accidental whitespace padding (e.g. `' Male '` → `'Male'`).")
                sel_strip_cols = st.multiselect("Select columns (leave empty for all text columns)", options=cat_cols)
                if st.button("Strip Whitespace", type="primary"):
                    step = engine.clean_categories(
                        operation="strip_whitespace",
                        columns_list=sel_strip_cols if sel_strip_cols else None,
                    )
                    st.success(step.details)
                    st.rerun()

            elif cat_action == "standardize_casing":
                st.markdown("#### Standardize Inconsistent Capitalization")
                target_cat_col = st.selectbox("Select Categorical Column", options=cat_cols, key="case_cat_col")
                diag = detect_categorical_inconsistencies(engine.current_df, target_cat_col)

                if diag["casing_clusters"]:
                    st.warning(f"Detected {len(diag['casing_clusters'])} casing variation cluster(s):")
                    for norm, variants in diag["casing_clusters"].items():
                        st.markdown(f"- Normalized `'**{norm}**'` has variants: `{', '.join(variants)}`")
                else:
                    st.success(f"No conflicting casing variants found in '{target_cat_col}'.")

                case_style = st.selectbox(
                    "Standard Casing Format",
                    options=["title", "lower", "upper", "capitalize"],
                    format_func=lambda s: {"title": "Title Case (e.g. Male)", "lower": "lowercase (e.g. male)", "upper": "UPPERCASE (e.g. MALE)", "capitalize": "Capitalize"}[s],
                )

                if st.button("Apply Casing Standardization", type="primary"):
                    step = engine.clean_categories(
                        operation="standardize_casing",
                        column=target_cat_col,
                        case_type=case_style,
                    )
                    st.success(step.details)
                    st.rerun()

            elif cat_action == "replace_values":
                st.markdown("#### Replace Specific Category Values")
                target_cat_col = st.selectbox("Select Column", options=cat_cols, key="replace_cat_col")
                existing_vals = sorted(engine.current_df[target_cat_col].dropna().astype(str).unique().tolist())
                
                col_old, col_new = st.columns(2)
                with col_old:
                    val_to_replace = st.selectbox("Value to Replace", options=existing_vals)
                with col_new:
                    new_val = st.text_input("New Replacement Value", value=val_to_replace)

                if st.button("Replace Value", type="primary", disabled=(val_to_replace == new_val)):
                    step = engine.clean_categories(
                        operation="replace_values",
                        column=target_cat_col,
                        mapping={val_to_replace: new_val},
                    )
                    st.success(step.details)
                    st.rerun()

            elif cat_action == "group_rare":
                st.markdown("#### Group Rare Categories")
                target_cat_col = st.selectbox("Select Column", options=cat_cols, key="rare_cat_col")
                rare_thresh = st.slider("Rare Frequency Cutoff (%)", min_value=1.0, max_value=15.0, value=3.0, step=0.5)
                other_name = st.text_input("Label for Grouped Categories", value="Other")

                diag = detect_categorical_inconsistencies(engine.current_df, target_cat_col, rare_threshold_pct=rare_thresh)
                rare_list = diag.get("rare_categories", [])

                if rare_list:
                    st.info(f"Found **{len(rare_list)}** rare category labels representing < {rare_thresh}% of records:")
                    st.dataframe(pd.DataFrame(rare_list), use_container_width=True, height=140)
                    if st.button("Group Rare Categories", type="primary"):
                        step = engine.clean_categories(
                            operation="group_rare",
                            column=target_cat_col,
                            rare_threshold_pct=rare_thresh,
                            other_label=other_name,
                        )
                        st.success(step.details)
                        st.rerun()
                else:
                    st.success(f"No categories fall below the {rare_thresh}% threshold in '{target_cat_col}'.")

    # =========================================================================
    # TAB F: DATA TYPE CORRECTION
    # =========================================================================
    with tabs[5]:
        st.markdown("### 🔄 Data Type Validation & Casting")
        st.markdown(
            "Inspect column data types, verify format conformance, and safely convert data types."
        )

        dt_df = inspect_data_types(engine.current_df)
        st.dataframe(dt_df, use_container_width=True, height=220)

        st.markdown("#### Convert Column Data Type")
        c_dt_col, c_dt_target, c_dt_act = st.columns([1.2, 1.0, 0.8])
        
        with c_dt_col:
            cast_col = st.selectbox("Column to Convert", options=list(engine.current_df.columns))
            st.caption(f"Current type: `{(engine.current_df[cast_col].dtype)}`")
        with c_dt_target:
            target_dt = st.selectbox(
                "Target Data Type",
                options=["integer", "float", "string", "boolean", "date", "datetime"],
                format_func=lambda t: {
                    "integer": "Nullable Integer (Int64)",
                    "float": "Float (float64)",
                    "string": "String / Text",
                    "boolean": "Boolean",
                    "date": "Date (YYYY-MM-DD)",
                    "datetime": "Datetime (Timestamp)",
                }[t],
            )
        with c_dt_act:
            st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
            if st.button("Convert Type", type="primary", use_container_width=True):
                step = engine.convert_data_type(cast_col, target_type=target_dt)
                if step.status == "Success":
                    st.success(step.details)
                elif step.status == "Warning":
                    st.warning(step.details)
                else:
                    st.error(step.details)
                st.rerun()

    # =========================================================================
    # TAB G: CATEGORICAL ENCODING
    # =========================================================================
    with tabs[6]:
        st.markdown("### 🏷️ Categorical Feature Encoding")
        st.markdown(
            "Transform qualitative text features into quantitative numerical representations "
            "while strictly guarding the target variable and identifier columns."
        )

        cat_feature_cols = [
            c for c in get_categorical_columns(engine.current_df)
            if c != target_col
        ]

        if target_col:
            st.info(f"🛡️ **Target Protection Active:** Target column `'**{target_col}**'` is protected and excluded from feature encoding.")

        if not cat_feature_cols:
            st.info("No categorical feature columns available for encoding.")
        else:
            enc_cols = st.multiselect(
                "Select Categorical Columns to Encode",
                options=cat_feature_cols,
                default=cat_feature_cols[:min(3, len(cat_feature_cols))],
            )

            enc_method = st.selectbox(
                "Encoding Method",
                options=["one_hot", "label", "ordinal"],
                format_func=lambda m: {
                    "one_hot": "One-Hot Encoding (Binary indicators - Recommended for non-ordered categories)",
                    "label": "Label Encoding (Integer ranks 0, 1, 2...)",
                    "ordinal": "Ordinal Encoding (Explicit custom ranking)",
                }[m],
            )

            drop_first_chk = False
            if enc_method == "one_hot":
                drop_first_chk = st.checkbox("Drop First Dummy Column (Prevent multicollinearity)", value=False)

            ordinal_orders = None
            if enc_method == "ordinal" and enc_cols:
                st.markdown("##### Specify Category Order (From lowest rank to highest):")
                ordinal_orders = {}
                for oc in enc_cols:
                    uniq_lvls = sorted(engine.current_df[oc].dropna().astype(str).unique().tolist())
                    ord_input = st.text_input(
                        f"Order for '{oc}' (comma-separated)",
                        value=", ".join(uniq_lvls),
                    )
                    ordinal_orders[oc] = [s.strip() for s in ord_input.split(",") if s.strip()]

            if st.button("Apply Categorical Encoding", type="primary", disabled=len(enc_cols) == 0):
                step = engine.encode_features(
                    columns=enc_cols,
                    method=enc_method,
                    drop_first=drop_first_chk,
                    ordinal_orderings=ordinal_orders,
                )
                st.success(step.details)
                st.rerun()

    # =========================================================================
    # TAB H: NUMERICAL FEATURE SCALING
    # =========================================================================
    with tabs[7]:
        st.markdown("### ⚖️ Numerical Feature Scaling")
        st.markdown(
            "Normalize or standardize numerical feature scales for distance-based ML models "
            "with zero-variance constant column protection."
        )

        num_feature_cols = [
            c for c in get_numerical_columns(engine.current_df)
            if c != target_col
        ]

        if not num_feature_cols:
            st.info("No numerical feature columns available for scaling.")
        else:
            scale_cols = st.multiselect(
                "Select Numerical Columns to Scale",
                options=num_feature_cols,
                default=num_feature_cols[:min(4, len(num_feature_cols))],
            )

            scale_method = st.selectbox(
                "Scaling Method",
                options=["standard", "minmax", "robust"],
                format_func=lambda m: {
                    "standard": "StandardScaler (Mean = 0, Std = 1)",
                    "minmax": "MinMaxScaler (Bounded Range, e.g. [0, 1])",
                    "robust": "RobustScaler (Median & IQR - Outlier resistant)",
                }[m],
            )

            minmax_range = (0.0, 1.0)
            if scale_method == "minmax":
                r_low = st.number_input("Range Minimum", value=0.0)
                r_high = st.number_input("Range Maximum", value=1.0)
                minmax_range = (r_low, r_high)

            if st.button("Apply Feature Scaling", type="primary", disabled=len(scale_cols) == 0):
                step = engine.scale_features(
                    columns=scale_cols,
                    method=scale_method,
                    feature_range=minmax_range,
                )
                st.success(step.details)
                st.rerun()

    # =========================================================================
    # TAB I: BEFORE-AND-AFTER COMPARISON
    # =========================================================================
    with tabs[8]:
        st.markdown("### 📊 Before-and-After Cleaning Comparison")
        st.markdown("Inspect how the dataset has transformed across key data quality dimensions.")

        comp = engine.get_comparison()

        c_kpi1, c_kpi2, c_kpi3, c_kpi4 = st.columns(4)
        with c_kpi1:
            st.metric("Total Rows", f"{comp['rows']['after']:,}", delta=comp['rows']['delta'])
        with c_kpi2:
            st.metric("Total Columns", f"{comp['columns']['after']:,}", delta=comp['columns']['delta'])
        with c_kpi3:
            st.metric("Missing Cells", f"{comp['missing_cells']['after']:,}", delta=comp['missing_cells']['delta'], delta_color="inverse")
        with c_kpi4:
            st.metric("Duplicate Rows", f"{comp['duplicates']['after']:,}", delta=comp['duplicates']['delta'], delta_color="inverse")

        # Visualizations
        fig_m = build_missing_comparison_chart(comp)
        if fig_m:
            st.plotly_chart(fig_m, use_container_width=True)

        fig_o = build_outlier_comparison_chart(comp)
        if fig_o:
            st.plotly_chart(fig_o, use_container_width=True)

        # Numerical distribution comparison
        num_cols_common = [
            c for c in engine.original_df.columns
            if c in engine.current_df.columns and pd.api.types.is_numeric_dtype(engine.original_df[c])
        ]
        if num_cols_common:
            st.markdown("#### Distribution Shift Comparison")
            col_shift = st.selectbox("Select Column to Compare Distribution", options=num_cols_common)
            fig_d = build_distribution_comparison_chart(engine.original_df, engine.current_df, col_shift)
            if fig_d:
                st.plotly_chart(fig_d, use_container_width=True)

        # Dtype changes
        if comp["dtype_changes"]:
            st.markdown("#### Data Type Modifications")
            st.dataframe(pd.DataFrame(comp["dtype_changes"]), use_container_width=True)

    # =========================================================================
    # TAB J: PREPROCESSING HISTORY AUDIT
    # =========================================================================
    with tabs[9]:
        st.markdown("### 📜 Preprocessing Operation Audit Log")
        st.markdown(
            "Every user-initiated transformation is permanently logged with timestamps, "
            "methodology parameters, and affected record counts for reproducibility."
        )

        hist_df = engine.history.get_log_dataframe()
        if hist_df.empty:
            st.info("No transformations applied yet. Operations will appear here as they are performed.")
        else:
            st.dataframe(hist_df, use_container_width=True, height=280)
            
            c_h1, c_h2 = st.columns(2)
            with c_h1:
                if st.button("↩️ Revert Last Step", key="history_revert_btn"):
                    if engine.undo_last_step():
                        st.success("Reverted the last preprocessing operation.")
                        st.rerun()
            with c_h2:
                csv_bytes = hist_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📥 Export Audit Log (CSV)",
                    data=csv_bytes,
                    file_name=f"preprocessing_audit_log_{engine.dataset_name}.csv",
                    mime="text/csv",
                )

    # =========================================================================
    # TAB K: DOWNLOAD PROCESSED DATASET
    # =========================================================================
    with tabs[10]:
        st.markdown("### 💾 Download Sanitized & Preprocessed Dataset")
        st.markdown(
            "Export the cleaned, encoded, and standardized dataset for machine learning "
            "or downstream investigation. The original file is never overwritten."
        )

        exp_c1, exp_c2 = st.columns(2)
        with exp_c1:
            export_fmt = st.radio("Export Format", options=["csv", "excel"], format_func=lambda f: "CSV (Comma-Separated Values)" if f == "csv" else "Excel Workbook (.xlsx)")
            out_prefix = st.text_input("Filename Prefix", value=f"cleaned_{engine.dataset_name}")

        with exp_c2:
            st.markdown(
                f"""
                <div style="background: #111827; border: 1px solid #1F2937; border-radius: 8px; padding: 1.2rem;">
                    <div style="font-weight: 700; color: #38BDF8; margin-bottom: 0.5rem;">Export Specification</div>
                    <div style="font-size: 0.85rem; color: #CBD5E1; line-height: 1.6;">
                        • Rows: <strong>{len(engine.current_df):,}</strong><br>
                        • Columns: <strong>{engine.current_df.shape[1]}</strong><br>
                        • Missing Cells: <strong>{int(engine.current_df.isna().sum().sum()):,}</strong><br>
                        • Format: <strong>{export_fmt.upper()}</strong>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        try:
            file_bytes, download_filename, mime_type = engine.export(format_type=export_fmt)
            st.download_button(
                label=f"📥 Download {download_filename}",
                data=file_bytes,
                file_name=download_filename,
                mime=mime_type,
                type="primary",
                use_container_width=True,
            )
            st.success(f"✅ Ready for download: '{download_filename}' ({len(file_bytes):,} bytes).")
        except Exception as e:
            st.error(f"Export generation failed: {str(e)}")
