"""Enterprise UI Component for Exploratory Data Analysis (EDA) & Data Visualization Engine.

Provides 10 dedicated analysis tabs for dataset overview, univariate distributions,
bivariate relationships, correlation matrices, target impact, temporal patterns,
cohort segmentation, automated factual insights, and data exports.
"""

from typing import Optional, List, Dict, Any
import streamlit as st
import pandas as pd
import numpy as np

from config.constants import APP_NAME
from utils.eda_helpers import (
    get_eda_numerical_columns,
    get_eda_categorical_columns,
    get_eda_datetime_columns,
    format_stat_number,
)
from modules.eda import (
    compute_numerical_univariate_stats,
    compute_all_numerical_summaries,
    compute_categorical_univariate_stats,
    compute_all_categorical_summaries,
    analyze_num_vs_num,
    analyze_cat_vs_num,
    analyze_cat_vs_cat,
    compute_correlation_matrix,
    extract_top_correlations,
    analyze_target_variable,
    prepare_multivariate_matrix,
    detect_time_series_candidates,
    compute_time_series_aggregation,
    compute_segment_breakdown,
    generate_eda_insights,
)
from components.metric_cards import render_metric_card
from components.eda_visualizations import (
    plot_numerical_distribution,
    plot_categorical_distribution,
    plot_scatter_with_trend,
    plot_categorical_vs_numerical,
    plot_cross_tabulation,
    plot_correlation_heatmap,
    plot_time_series,
    plot_scatter_matrix,
)


def render_eda_page() -> None:
    """Render the master Exploratory Data Analysis (EDA) and Visualization Dashboard."""
    # Ensure session state initialization
    if "uploaded_dataset" not in st.session_state:
        st.session_state["uploaded_dataset"] = None
    if "original_dataset" not in st.session_state:
        st.session_state["original_dataset"] = None
    if "dataset_filename" not in st.session_state:
        st.session_state["dataset_filename"] = "dataset.csv"
    if "selected_target_variable" not in st.session_state:
        st.session_state["selected_target_variable"] = None

    # Header Banner
    st.markdown(
        """
        <div class="hero-banner" style="margin-bottom: 1.5rem; padding: 1.8rem 2.2rem;">
            <div class="hero-title" style="font-size: 2rem;">Exploratory Data Analysis (EDA) & Visualization</div>
            <div class="hero-tagline" style="font-size: 1rem; color: #38BDF8;">
                Discover patterns, examine statistical distributions, and isolate variable interactions through interactive analytics.
            </div>
            <p class="hero-description" style="font-size: 0.88rem;">
                Explore univariate feature characteristics, bivariate relationships, linear and rank correlations,
                temporal progressions, cohort segmentations, and outcome target distributions.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Dataset Source Selector: Cleaned vs. Original
    df_cleaned = st.session_state.get("uploaded_dataset")
    df_original = st.session_state.get("original_dataset")
    filename = st.session_state.get("dataset_filename", "active_dataset.csv")

    if df_cleaned is None and df_original is None:
        st.info("ℹ️ **No dataset loaded.** Please navigate to **📊 Dataset Overview** or **🧹 Data Cleaning** to upload a dataset.")
        return

    # Dataset Selection Control Bar
    ctrl_c1, ctrl_c2, ctrl_c3 = st.columns([1.2, 1.2, 1.0])

    with ctrl_c1:
        dataset_options = []
        if df_cleaned is not None:
            dataset_options.append("Cleaned / Working Dataset")
        if df_original is not None:
            dataset_options.append("Original Uploaded Dataset")

        selected_source = st.selectbox(
            "Select Dataset Version for Analysis:",
            options=dataset_options,
            index=0,
            key="eda_dataset_source_picker",
        )

    # Resolve active dataframe
    if selected_source == "Original Uploaded Dataset" and df_original is not None:
        active_df = df_original
        active_label = f"Original Upload ({filename})"
    else:
        active_df = df_cleaned if df_cleaned is not None else df_original
        active_label = f"Cleaned / Working Copy ({filename})"

    # Target variable control
    target_col = st.session_state.get("selected_target_variable")
    all_columns = list(active_df.columns)

    with ctrl_c2:
        target_options = ["None (Unassigned)"] + all_columns
        curr_target_idx = (
            target_options.index(target_col)
            if target_col in target_options
            else 0
        )
        selected_target_choice = st.selectbox(
            "Selected Target / Outcome Variable:",
            options=target_options,
            index=curr_target_idx,
            key="eda_target_picker",
        )
        if selected_target_choice == "None (Unassigned)":
            target_col = None
            st.session_state["selected_target_variable"] = None
        else:
            target_col = selected_target_choice
            st.session_state["selected_target_variable"] = target_col

    with ctrl_c3:
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
        st.caption(f"Active: **{active_label}**")

    # Column type breakdowns
    num_cols = get_eda_numerical_columns(active_df)
    cat_cols = get_eda_categorical_columns(active_df)
    dt_cols = get_eda_datetime_columns(active_df)

    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

    # High-level KPI Cards
    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    with kpi1:
        render_metric_card(
            title="Total Records",
            value=f"{len(active_df):,}",
            subtitle=f"{active_df.shape[1]} Features",
            badge_text="Volume",
            badge_type="blue",
        )
    with kpi2:
        render_metric_card(
            title="Numerical Features",
            value=f"{len(num_cols)}",
            subtitle="Continuous & Discrete",
            badge_text="Numeric",
            badge_type="purple",
        )
    with kpi3:
        render_metric_card(
            title="Categorical Features",
            value=f"{len(cat_cols)}",
            subtitle="Classes & Labels",
            badge_text="Qualitative",
            badge_type="emerald",
        )
    with kpi4:
        render_metric_card(
            title="Temporal Features",
            value=f"{len(dt_cols)}",
            subtitle="Date & Timestamps",
            badge_text="Time-Series" if dt_cols else "None",
            badge_type="blue" if dt_cols else "amber",
        )
    with kpi5:
        target_display = target_col if target_col else "Not Set"
        render_metric_card(
            title="Target Outcome",
            value=target_display[:11] + ("..." if target_col and len(target_col) > 11 else ""),
            subtitle="Investigation Anchor",
            badge_text="Assigned" if target_col else "Unassigned",
            badge_type="purple" if target_col else "blue",
        )

    st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)

    # 10 Dedicated EDA Tabs
    tabs = st.tabs([
        "📋 1. Overview",
        "📈 2. Numerical Analysis",
        "🔠 3. Categorical Analysis",
        "🔗 4. Relationships",
        "🌡️ 5. Correlations",
        "🎯 6. Target Analysis",
        "📅 7. Time-Series",
        "👥 8. Segments",
        "💡 9. Automated Insights",
        "💾 10. Export Data",
    ])

    # =========================================================================
    # TAB 1: DATASET OVERVIEW
    # =========================================================================
    with tabs[0]:
        st.markdown("### 📋 Dataset Architecture & Attribute Overview")
        st.markdown("Inspect column data types, completeness, and dimensional composition.")

        c_ov1, c_ov2 = st.columns([1.2, 0.8])
        with c_ov1:
            st.markdown("#### Feature Inventory & Data Health")
            summary_records = []
            for col in active_df.columns:
                s = active_df[col]
                non_null = int(s.notna().sum())
                nulls = int(s.isna().sum())
                pct_null = round((nulls / max(1, len(active_df)) * 100.0), 2)
                
                if col in num_cols:
                    kind = "Numerical"
                elif col in dt_cols:
                    kind = "Datetime"
                else:
                    kind = "Categorical"

                summary_records.append({
                    "Column Name": col,
                    "Semantic Kind": kind,
                    "Pandas Dtype": str(s.dtype),
                    "Non-Null Count": f"{non_null:,}",
                    "Missing Count": f"{nulls:,} ({pct_null}%)",
                    "Unique Values": f"{s.nunique():,}",
                })
            st.dataframe(pd.DataFrame(summary_records), use_container_width=True, height=360)

        with c_ov2:
            st.markdown("#### Dimensional Breakdown")
            breakdown_df = pd.DataFrame({
                "Category": ["Numerical", "Categorical", "Datetime"],
                "Count": [len(num_cols), len(cat_cols), len(dt_cols)],
            })
            fig_types = plot_categorical_distribution(
                breakdown_df.rename(columns={"Category": "Category", "Count": "Count"}).assign(
                    Percentage=lambda d: (d["Count"] / max(1, d["Count"].sum()) * 100).round(1)
                ),
                title="Feature Type Distribution",
                chart_type="pie",
            )
            st.plotly_chart(fig_types, use_container_width=True)

    # =========================================================================
    # TAB 2: NUMERICAL ANALYSIS
    # =========================================================================
    with tabs[1]:
        st.markdown("### 📈 Univariate Numerical Feature Analysis")
        st.markdown("Examine distribution shapes, parametric and non-parametric central tendencies, and tail skewness.")

        if not num_cols:
            st.info("No numerical features available in this dataset.")
        else:
            col_sel, col_mode, col_bins = st.columns([1.5, 1.2, 1.0])
            with col_sel:
                selected_num = st.selectbox("Select Numerical Feature:", options=num_cols, key="eda_num_feature_sel")
            with col_mode:
                plot_mode = st.radio(
                    "Chart Perspective:",
                    options=["histogram", "box", "violin"],
                    format_func=lambda m: {"histogram": "Histogram + Central Tendency", "box": "Box Plot (Outliers)", "violin": "Violin Plot (Density)"}[m],
                    horizontal=True,
                )
            with col_bins:
                num_bins = st.slider("Histogram Bins", min_value=10, max_value=80, value=30, step=5)

            stats = compute_numerical_univariate_stats(active_df, selected_num)

            # Key Stats Pills
            p1, p2, p3, p4, p5, p6 = st.columns(6)
            with p1:
                st.metric("Mean", format_stat_number(stats["mean"]))
            with p2:
                st.metric("Median", format_stat_number(stats["median"]))
            with p3:
                st.metric("Std Dev", format_stat_number(stats["std"]))
            with p4:
                st.metric("IQR", format_stat_number(stats["iqr"]))
            with p5:
                st.metric("Skewness", format_stat_number(stats["skewness"]))
            with p6:
                st.metric("Kurtosis", format_stat_number(stats["kurtosis"]))

            st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

            c_fig, c_stats = st.columns([1.5, 1.0])
            with c_fig:
                fig_dist = plot_numerical_distribution(
                    active_df[selected_num],
                    plot_type=plot_mode,
                    nbins=num_bins,
                )
                st.plotly_chart(fig_dist, use_container_width=True)

            with c_stats:
                st.markdown("#### Descriptive Parameter Table")
                stats_table = pd.DataFrame([
                    {"Parameter": "Record Count", "Value": format_stat_number(stats["count"])},
                    {"Parameter": "Missing Values", "Value": f"{stats['missing_count']} ({stats['missing_pct']}%)"},
                    {"Parameter": "Arithmetic Mean", "Value": format_stat_number(stats["mean"])},
                    {"Parameter": "Median (50th %)", "Value": format_stat_number(stats["median"])},
                    {"Parameter": "Mode (Most Frequent)", "Value": format_stat_number(stats["mode"])},
                    {"Parameter": "Standard Deviation", "Value": format_stat_number(stats["std"])},
                    {"Parameter": "Variance", "Value": format_stat_number(stats["variance"])},
                    {"Parameter": "Minimum", "Value": format_stat_number(stats["min"])},
                    {"Parameter": "25th Percentile (Q1)", "Value": format_stat_number(stats["q1"])},
                    {"Parameter": "75th Percentile (Q3)", "Value": format_stat_number(stats["q3"])},
                    {"Parameter": "Maximum", "Value": format_stat_number(stats["max"])},
                    {"Parameter": "Interquartile Range (IQR)", "Value": format_stat_number(stats["iqr"])},
                    {"Parameter": "Skewness (Asymmetry)", "Value": format_stat_number(stats["skewness"])},
                    {"Parameter": "Excess Kurtosis (Tails)", "Value": format_stat_number(stats["kurtosis"])},
                    {"Parameter": "Zero Values Count", "Value": f"{stats['zeros_count']} ({stats['zeros_pct']}%)"},
                ])
                st.dataframe(stats_table, use_container_width=True, height=360, hide_index=True)

            # Master Numerical Summary Expander
            with st.expander("📊 View Master Numerical Summary Table Across All Features", expanded=False):
                master_num_df = compute_all_numerical_summaries(active_df)
                st.dataframe(master_num_df, use_container_width=True, hide_index=True)

    # =========================================================================
    # TAB 3: CATEGORICAL ANALYSIS
    # =========================================================================
    with tabs[2]:
        st.markdown("### 🔠 Univariate Categorical Feature Analysis")
        st.markdown("Inspect category cardinalities, class dominance, and frequency proportions.")

        if not cat_cols:
            st.info("No categorical/discrete features available in this dataset.")
        else:
            cat_c1, cat_c2, cat_c3 = st.columns([1.5, 1.2, 1.0])
            with cat_c1:
                selected_cat = st.selectbox("Select Categorical Feature:", options=cat_cols, key="eda_cat_feature_sel")
            with cat_c2:
                cat_chart_type = st.radio(
                    "Chart Representation:",
                    options=["bar", "horizontal_bar", "pie"],
                    format_func=lambda c: {"bar": "Vertical Bar", "horizontal_bar": "Horizontal Bar", "pie": "Donut Pie (Low Cardinality)"}[c],
                    horizontal=True,
                )
            with cat_c3:
                top_n = st.slider("Display Top N Categories:", min_value=3, max_value=25, value=10)
                group_other_chk = st.checkbox("Group Remaining as 'Other'", value=False)

            cat_stats = compute_categorical_univariate_stats(
                active_df, selected_cat, top_n=top_n, group_other=group_other_chk
            )

            # Summary Metrics Pills
            cp1, cp2, cp3, cp4 = st.columns(4)
            with cp1:
                st.metric("Total Classes", f"{cat_stats['unique_categories']:,}")
            with cp2:
                top_summary = f"{cat_stats['most_frequent_category']} ({cat_stats['most_frequent_pct']}%)" if cat_stats["most_frequent_category"] else "N/A"
                st.metric("Top Category", top_summary)
            with cp3:
                least_summary = f"{cat_stats['least_frequent_category']} ({cat_stats['least_frequent_pct']}%)" if cat_stats["least_frequent_category"] else "N/A"
                st.metric("Least Frequent", least_summary)
            with cp4:
                st.metric("Missing Values", f"{cat_stats['missing_count']} ({cat_stats['missing_pct']}%)")

            st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

            c_cat_fig, c_cat_tbl = st.columns([1.5, 1.0])
            with c_cat_fig:
                fig_cat = plot_categorical_distribution(
                    cat_stats["frequencies_df"],
                    title=f"Category Frequencies: '{selected_cat}'",
                    chart_type=cat_chart_type,
                )
                st.plotly_chart(fig_cat, use_container_width=True)

            with c_cat_tbl:
                st.markdown("#### Frequency Distribution Table")
                st.dataframe(cat_stats["frequencies_df"], use_container_width=True, height=360, hide_index=True)

            with st.expander("📊 View Master Categorical Summary Table Across All Features", expanded=False):
                master_cat_df = compute_all_categorical_summaries(active_df)
                st.dataframe(master_cat_df, use_container_width=True, hide_index=True)

    # =========================================================================
    # TAB 4: RELATIONSHIP ANALYSIS (BIVARIATE)
    # =========================================================================
    with tabs[3]:
        st.markdown("### 🔗 Bivariate Relationship Analysis")
        st.markdown("Examine how pairs of variables interact: Num vs Num, Cat vs Num, and Cat vs Cat.")

        biv_type = st.radio(
            "Relationship Modality:",
            options=["num_num", "cat_num", "cat_cat"],
            format_func=lambda m: {
                "num_num": "1. Numerical vs. Numerical (Scatter & Correlation)",
                "cat_num": "2. Categorical vs. Numerical (Group Breakdowns & ANOVA)",
                "cat_cat": "3. Categorical vs. Categorical (Cross-Tabulation & Chi-Square)",
            }[m],
            horizontal=True,
        )

        st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

        if biv_type == "num_num":
            if len(num_cols) < 2:
                st.warning("At least 2 numerical features are required for Numerical vs. Numerical analysis.")
            else:
                bn_c1, bn_c2, bn_c3 = st.columns([1.2, 1.2, 1.0])
                with bn_c1:
                    x_col = st.selectbox("X-Axis Feature:", options=num_cols, index=0, key="biv_x_col")
                with bn_c2:
                    y_default = 1 if len(num_cols) > 1 else 0
                    y_col = st.selectbox("Y-Axis Feature:", options=num_cols, index=y_default, key="biv_y_col")
                with bn_c3:
                    color_options = ["None"] + cat_cols
                    color_choice = st.selectbox("Optional Color Segment:", options=color_options, index=0)
                    active_color = color_choice if color_choice != "None" else None

                res_nn = analyze_num_vs_num(active_df, x_col, y_col)

                # Metric pills
                m1, m2, m3, m4 = st.columns(4)
                with m1:
                    st.metric("Pearson Correlation (r)", f"{res_nn['pearson_corr']:+.3f}")
                with m2:
                    st.metric("Spearman Rank (ρ)", f"{res_nn['spearman_corr']:+.3f}")
                with m3:
                    st.metric("Coefficient of Det. (R²)", f"{res_nn['r_squared']:.3f}")
                with m4:
                    st.metric("Association Strength", f"{res_nn['strength']} {res_nn['direction']}")

                fig_scat = plot_scatter_with_trend(active_df, x_col, y_col, color_col=active_color)
                st.plotly_chart(fig_scat, use_container_width=True)

        elif biv_type == "cat_num":
            if not cat_cols or not num_cols:
                st.warning("Both categorical and numerical features are required.")
            else:
                cn_c1, cn_c2, cn_c3 = st.columns([1.2, 1.2, 1.0])
                with cn_c1:
                    sel_cat = st.selectbox("Categorical Feature (Groups):", options=cat_cols, key="biv_cat_col")
                with cn_c2:
                    sel_num = st.selectbox("Numerical Feature (Measurement):", options=num_cols, key="biv_num_col")
                with cn_c3:
                    plot_cat_num = st.selectbox("Chart Type:", options=["box", "violin", "bar_mean"], format_func=lambda c: {"box": "Box Plot", "violin": "Violin Plot", "bar_mean": "Mean Bar Chart"}[c])

                res_cn = analyze_cat_vs_num(active_df, sel_cat, sel_num)

                c1, c2 = st.columns([1.5, 1.0])
                with c1:
                    fig_cn = plot_categorical_vs_numerical(active_df, sel_cat, sel_num, plot_type=plot_cat_num)
                    st.plotly_chart(fig_cn, use_container_width=True)

                with c2:
                    st.markdown("#### Segment Distribution Metrics")
                    if res_cn["anova_p"] is not None:
                        sig_badge = "Significant" if res_cn["is_significant"] else "Not Significant"
                        st.caption(f"ANOVA F-Stat: **{res_cn['anova_f']}** (p = {res_cn['anova_p']:.4e} - {sig_badge})")
                    st.dataframe(res_cn["group_summary_df"], use_container_width=True, hide_index=True)

        else:
            if len(cat_cols) < 2:
                st.warning("At least 2 categorical features are required for Cross-Tabulation.")
            else:
                cc_c1, cc_c2, cc_c3 = st.columns([1.2, 1.2, 1.0])
                with cc_c1:
                    cat1 = st.selectbox("Row Category:", options=cat_cols, index=0, key="biv_cat1")
                with cc_c2:
                    c2_default = 1 if len(cat_cols) > 1 else 0
                    cat2 = st.selectbox("Column Category:", options=cat_cols, index=c2_default, key="biv_cat2")
                with cc_c3:
                    plot_cc = st.selectbox("Display Format:", options=["stacked_bar", "heatmap"], format_func=lambda x: "Stacked Bar Chart" if x == "stacked_bar" else "Cross-Tab Heatmap")

                res_cc = analyze_cat_vs_cat(active_df, cat1, cat2)

                c1, c2 = st.columns([1.4, 1.1])
                with c1:
                    fig_cc = plot_cross_tabulation(res_cc["crosstab_counts"], cat1, cat2, plot_type=plot_cc)
                    st.plotly_chart(fig_cc, use_container_width=True)

                with c2:
                    st.markdown("#### Cross-Tabulation Counts")
                    st.dataframe(res_cc["crosstab_counts"], use_container_width=True)
                    if res_cc["chi2_p"] is not None:
                        sig_txt = "Statistically Significant" if res_cc["is_significant"] else "No Significant Dependency"
                        st.caption(f"Chi-Square: **{res_cc['chi2_stat']}** (p = {res_cc['chi2_p']:.4e}, df = {res_cc['dof']}) — {sig_txt}")

    # =========================================================================
    # TAB 5: CORRELATION ANALYSIS
    # =========================================================================
    with tabs[4]:
        st.markdown("### 🌡️ Correlation Analysis & Heatmap")
        st.markdown("Compute linear (Pearson) or rank-based monotonic (Spearman) association matrices.")

        st.info("💡 **Scientific Reality Check:** Correlation measures statistical association between features. It does not establish direct cause-and-effect.")

        if len(num_cols) < 2:
            st.warning("At least 2 numerical features are required for correlation analysis.")
        else:
            cor_c1, cor_c2, cor_c3 = st.columns([1.0, 1.5, 1.0])
            with cor_c1:
                corr_method = st.selectbox("Correlation Method:", options=["pearson", "spearman"], format_func=lambda m: "Pearson (Linear)" if m == "pearson" else "Spearman (Rank Monotonic)")
            with cor_c2:
                sel_corr_cols = st.multiselect("Select Features to Correlate:", options=num_cols, default=num_cols[:min(10, len(num_cols))])
            with cor_c3:
                corr_thresh = st.slider("Association Threshold (|r| ≥):", min_value=0.1, max_value=0.8, value=0.3, step=0.05)

            corr_matrix, meta = compute_correlation_matrix(active_df, columns=sel_corr_cols, method=corr_method)

            if meta.get("excluded_constant_columns"):
                st.caption(f"ℹ️ Excluded zero-variance constant column(s): `{', '.join(meta['excluded_constant_columns'])}`")

            c_hmap, c_ranks = st.columns([1.4, 1.1])
            with c_hmap:
                fig_hm = plot_correlation_heatmap(corr_matrix, title=f"{corr_method.capitalize()} Correlation Heatmap")
                if fig_hm:
                    st.plotly_chart(fig_hm, use_container_width=True)

            with c_ranks:
                st.markdown(f"#### Strongest Associations (|r| ≥ {corr_thresh})")
                top_pairs = extract_top_correlations(corr_matrix, threshold=corr_thresh, top_n=15)
                if not top_pairs.empty:
                    st.dataframe(top_pairs, use_container_width=True, hide_index=True)
                else:
                    st.info(f"No feature pairs exceed the |r| ≥ {corr_thresh} threshold.")

    # =========================================================================
    # TAB 6: TARGET VARIABLE ANALYSIS
    # =========================================================================
    with tabs[5]:
        st.markdown("### 🎯 Target / Outcome Variable Deep Dive")
        st.markdown("Isolate relationships between the selected investigation outcome and all remaining features.")

        if not target_col:
            st.warning("No target variable currently selected. Please select a target variable in the controls bar above to activate outcome-specific exploration.")
        else:
            target_data = analyze_target_variable(active_df, target_col)
            if not target_data.get("valid"):
                st.error(target_data.get("message"))
            else:
                st.success(f"Anchor Outcome: **'{target_col}'** ({target_data['target_kind'].capitalize()} Variable)")

                if target_data["target_kind"] == "numerical":
                    t_c1, t_c2 = st.columns([1.2, 1.2])
                    with t_c1:
                        st.markdown("#### Distribution of Numerical Target")
                        fig_t_dist = plot_numerical_distribution(active_df[target_col], title=f"Target Distribution: '{target_col}'")
                        st.plotly_chart(fig_t_dist, use_container_width=True)
                    with t_c2:
                        st.markdown("#### Top Feature Correlations with Target")
                        st.dataframe(target_data["correlations_df"], use_container_width=True, hide_index=True)
                else:
                    t_c1, t_c2 = st.columns([1.2, 1.2])
                    with t_c1:
                        st.markdown("#### Outcome Class Proportions")
                        fig_t_cat = plot_categorical_distribution(
                            target_data["class_distribution_df"].rename(columns={"Class": "Category"}),
                            title=f"Target Classes: '{target_col}'",
                            chart_type="bar",
                        )
                        st.plotly_chart(fig_t_cat, use_container_width=True)
                    with t_c2:
                        st.markdown(f"#### Class Balance Status: **{target_data['imbalance_status']}**")
                        st.caption(f"Imbalance Ratio: **{target_data['imbalance_ratio']}:1**")
                        st.dataframe(target_data["class_distribution_df"], use_container_width=True, hide_index=True)

    # =========================================================================
    # TAB 7: TIME-SERIES ANALYSIS
    # =========================================================================
    with tabs[6]:
        st.markdown("### 📅 Time-Series & Temporal Analysis")
        st.markdown("Track metric fluctuations, periodicity, and moving averages over time.")

        if not dt_cols:
            st.info("ℹ️ **No temporal/datetime columns identified.** Time-series exploration is available when datasets contain date or timestamp fields.")
        elif not num_cols:
            st.info("ℹ️ **No numerical metrics available** to aggregate across time.")
        else:
            ts_c1, ts_c2, ts_c3, ts_c4 = st.columns([1.2, 1.2, 1.0, 1.0])
            with ts_c1:
                sel_date = st.selectbox("Datetime Column:", options=dt_cols, key="eda_ts_date")
            with ts_c2:
                sel_ts_metric = st.selectbox("Numerical Metric to Track:", options=num_cols, key="eda_ts_metric")
            with ts_c3:
                freq_choice = st.selectbox(
                    "Frequency:",
                    options=["D", "W", "M", "Q", "Y"],
                    format_func=lambda f: {"D": "Daily", "W": "Weekly", "M": "Monthly", "Q": "Quarterly", "Y": "Yearly"}[f],
                    index=2,
                )
            with ts_c4:
                agg_type = st.selectbox("Aggregation:", options=["mean", "sum", "median", "min", "max"])

            roll_win = st.slider("Moving Average Window (Periods):", min_value=1, max_value=12, value=3)
            use_roll = roll_win if roll_win > 1 else None

            ts_res = compute_time_series_aggregation(
                active_df, sel_date, sel_ts_metric, freq=freq_choice, agg_func=agg_type, rolling_window=use_roll
            )

            if not ts_res.get("valid"):
                st.warning(ts_res.get("message"))
            else:
                st.caption(f"Temporal Span: **{ts_res['start_date']}** to **{ts_res['end_date']}** ({ts_res['total_periods']} periods)")
                fig_ts = plot_time_series(
                    ts_res["ts_df"],
                    date_col="Timestamp",
                    metric_col=sel_ts_metric,
                    rolling_col=ts_res["rolling_col"],
                )
                st.plotly_chart(fig_ts, use_container_width=True)

    # =========================================================================
    # TAB 8: SEGMENT ANALYSIS
    # =========================================================================
    with tabs[7]:
        st.markdown("### 👥 Segment & Cohort Aggregation")
        st.markdown("Disaggregate and compare numerical metrics across customer segments, regions, or categories.")

        if not cat_cols:
            st.warning("No categorical columns available for segment grouping.")
        else:
            seg_c1, seg_c2 = st.columns([1.2, 1.8])
            with seg_c1:
                group_col = st.selectbox("Group By Segment:", options=cat_cols, key="eda_seg_group_col")
            with seg_c2:
                seg_metrics = st.multiselect("Metrics to Aggregate:", options=num_cols, default=num_cols[:min(3, len(num_cols))])

            seg_res = compute_segment_breakdown(active_df, segment_col=group_col, metric_cols=seg_metrics)
            st.dataframe(seg_res["segment_summary_df"], use_container_width=True, hide_index=True)

    # =========================================================================
    # TAB 9: AUTOMATED EDA INSIGHTS
    # =========================================================================
    with tabs[8]:
        st.markdown("### 💡 Automated Factual Dataset Insights")
        st.markdown("Observations derived algorithmically from distribution shapes, class balances, and correlations.")

        insights = generate_eda_insights(active_df, target_col=target_col)

        for idx, item in enumerate(insights, 1):
            badge = item.get("badge", "Insight")
            badge_type = item.get("badge_color", "blue")
            category = item.get("category", "General")
            text = item.get("insight", "")

            st.markdown(
                f"""
                <div style="background: #111827; border: 1px solid #1F2937; border-left: 4px solid #4F46E5; border-radius: 8px; padding: 1rem 1.25rem; margin-bottom: 0.75rem;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.35rem;">
                        <span style="font-size: 0.75rem; font-weight: 700; text-transform: uppercase; color: #94A3B8;">{category}</span>
                        <span class="metric-badge badge-{badge_type}">{badge}</span>
                    </div>
                    <div style="font-size: 0.9rem; color: #E2E8F0; line-height: 1.5;">{text}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # =========================================================================
    # TAB 10: EXPORT OPTIONS
    # =========================================================================
    with tabs[9]:
        st.markdown("### 💾 Export Exploratory Data Summaries")
        st.markdown("Download statistical matrices and summaries for reporting or offline audits.")

        ex_c1, ex_c2, ex_c3 = st.columns(3)

        with ex_c1:
            st.markdown("#### Descriptive Statistics")
            st.caption("Summary stats for all numerical columns.")
            if num_cols:
                num_summary_df = compute_all_numerical_summaries(active_df)
                num_csv = num_summary_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📥 Download Numerical Stats (CSV)",
                    data=num_csv,
                    file_name=f"numerical_stats_{filename.split('.')[0]}.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
            else:
                st.caption("No numerical features.")

        with ex_c2:
            st.markdown("#### Correlation Matrix")
            st.caption("Pairwise Pearson correlation coefficients.")
            if len(num_cols) >= 2:
                corr_df, _ = compute_correlation_matrix(active_df, method="pearson")
                corr_csv = corr_df.to_csv().encode("utf-8")
                st.download_button(
                    label="📥 Download Correlation Matrix (CSV)",
                    data=corr_csv,
                    file_name=f"correlation_matrix_{filename.split('.')[0]}.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
            else:
                st.caption("Insufficient numerical features.")

        with ex_c3:
            st.markdown("#### Categorical Summaries")
            st.caption("Cardinality and frequency tables.")
            if cat_cols:
                cat_summary_df = compute_all_categorical_summaries(active_df)
                cat_csv = cat_summary_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📥 Download Categorical Stats (CSV)",
                    data=cat_csv,
                    file_name=f"categorical_stats_{filename.split('.')[0]}.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
            else:
                st.caption("No categorical features.")

