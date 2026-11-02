"""Interactive Streamlit UI for Phase 6: Statistical Investigation & Hypothesis Testing Engine."""

from typing import Optional, Dict, Any, List
import streamlit as st
import pandas as pd
import numpy as np

from modules.statistical_analysis.statistical_engine import StatisticalEngine
from modules.statistical_analysis.hypothesis_testing import HypothesisTestResult
from components.statistical_visualizations import (
    plot_qq_plot,
    plot_mean_comparison_ci,
    plot_contingency_heatmap,
    plot_correlation_scatter,
    plot_multiple_testing_comparison,
)
from utils.eda_helpers import get_eda_numerical_columns, get_eda_categorical_columns
from utils.sample_data import generate_churn_investigation_dataset


def render_statistical_page() -> None:
    """Master render function for the Statistical Investigation & Hypothesis Testing Engine."""
    # Ensure session state initialization
    if "statistical_engine" not in st.session_state:
        st.session_state["statistical_engine"] = StatisticalEngine(default_alpha=0.05)
    if "active_stat_alpha" not in st.session_state:
        st.session_state["active_stat_alpha"] = 0.05
    if "latest_stat_result" not in st.session_state:
        st.session_state["latest_stat_result"] = None

    engine: StatisticalEngine = st.session_state["statistical_engine"]

    # Header Banner
    st.markdown(
        """
        <div style="margin-bottom: 1.5rem;">
            <div style="font-size: 1.8rem; font-weight: 800; color: #F8FAFC; letter-spacing: -0.02em;">
                🧪 Statistical Investigation & Hypothesis Testing Engine
            </div>
            <div style="font-size: 0.95rem; color: #94A3B8; margin-top: 0.25rem;">
                Rigorous empirical inference: Parametric & non-parametric tests, contingency association, effect size estimation, and multiplicity control.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 1. Dataset Resolution & Version Selector
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
        _render_empty_prompt()
        return

    # Dataset & Alpha Control Bar
    ctrl_c1, ctrl_c2, ctrl_c3 = st.columns([1.5, 1.2, 1.0])

    with ctrl_c1:
        if len(dataset_options) > 1:
            selected_source = st.selectbox(
                "Select Dataset for Statistical Testing:",
                options=dataset_options,
                index=0,
                key="stat_dataset_source_picker",
            )
        else:
            selected_source = dataset_options[0]

        if selected_source == "Original Uploaded Dataset" and df_original is not None:
            active_df = df_original
            active_label = f"Original Upload ({filename})"
        else:
            active_df = df_cleaned if df_cleaned is not None else st.session_state.get("current_dataset")
            active_label = f"Cleaned / Working ({filename})"

    with ctrl_c2:
        alpha_choice = st.selectbox(
            "Significance Level (α):",
            options=["0.05 (Standard)", "0.01 (Strict)", "0.10 (Exploratory)", "Custom"],
            index=0,
            key="stat_alpha_picker",
        )
        if alpha_choice.startswith("0.05"):
            active_alpha = 0.05
        elif alpha_choice.startswith("0.01"):
            active_alpha = 0.01
        elif alpha_choice.startswith("0.10"):
            active_alpha = 0.10
        else:
            active_alpha = st.number_input("Custom α Threshold", min_value=0.001, max_value=0.20, value=0.05, step=0.005)

        st.session_state["active_stat_alpha"] = active_alpha
        engine.default_alpha = active_alpha

    with ctrl_c3:
        target_name = st.session_state.get("selected_target_variable") or "Unassigned"
        st.markdown(
            f"""
            <div style="background: #131B2E; border: 1px solid #1E293B; border-radius: 8px; padding: 0.6rem 0.8rem; font-size: 0.8rem; color: #CBD5E1; margin-top: 0.35rem;">
                Target: <strong style="color: #38BDF8;">{target_name}</strong><br>
                Records: <strong style="color: #38BDF8;">{len(active_df):,} rows × {len(active_df.columns)} cols</strong>
            </div>
            """,
            unsafe_allow_html=True,
        )

    num_cols = get_eda_numerical_columns(active_df)
    cat_cols = get_eda_categorical_columns(active_df)

    # 10 Investigation Tabs
    tab_overview, tab_recommend, tab_num, tab_cat, tab_corr, tab_assumptions, tab_effects, tab_mult, tab_narrative, tab_export = st.tabs([
        "📌 Overview",
        "🎯 Test Recommender",
        "🔢 Numerical Tests",
        "🏷️ Categorical Tests",
        "📈 Correlation Testing",
        "🔍 Assumptions Check",
        "📐 Effect Sizes & CIs",
        "⚖️ Multiple Testing",
        "💡 Interpretation Dossier",
        "📥 Export Results",
    ])

    # TAB 1: OVERVIEW
    with tab_overview:
        _render_overview_tab(active_df, num_cols, cat_cols, engine)

    # TAB 2: TEST RECOMMENDER
    with tab_recommend:
        _render_recommender_tab(active_df, num_cols, cat_cols, engine)

    # TAB 3: NUMERICAL TESTS
    with tab_num:
        _render_numerical_tab(active_df, num_cols, cat_cols, engine, active_alpha)

    # TAB 4: CATEGORICAL TESTS
    with tab_cat:
        _render_categorical_tab(active_df, cat_cols, num_cols, engine, active_alpha)

    # TAB 5: CORRELATION TESTING
    with tab_corr:
        _render_correlation_tab(active_df, num_cols, engine, active_alpha)

    # TAB 6: ASSUMPTION CHECKING
    with tab_assumptions:
        _render_assumptions_tab(active_df, num_cols, cat_cols, engine, active_alpha)

    # TAB 7: EFFECT SIZES & CIS
    with tab_effects:
        _render_effects_tab(engine)

    # TAB 8: MULTIPLE TESTING
    with tab_mult:
        _render_multiple_testing_tab(engine, active_alpha)

    # TAB 9: INTERPRETATION DOSSIER
    with tab_narrative:
        _render_narrative_tab(engine)

    # TAB 10: EXPORT RESULTS
    with tab_export:
        _render_export_tab(engine)


def _render_empty_prompt() -> None:
    """Prompt user to load benchmark data if no dataset is active."""
    st.markdown(
        """
        <div style="background: #111827; border: 1px dashed #334155; border-radius: 12px; padding: 2.5rem 2rem; text-align: center; margin-top: 1rem;">
            <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">📊</div>
            <h3 style="color: #F8FAFC; margin-bottom: 0.5rem;">No Active Dataset in Session</h3>
            <p style="color: #94A3B8; max-width: 600px; margin: 0 auto 1.5rem auto; font-size: 0.9rem;">
                Load the benchmark investigation dataset to evaluate hypothesis tests, ANOVA, Chi-Square, and effect sizes.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button("🚀 Load Customer Churn Benchmark Dataset", type="primary"):
        demo_df = generate_churn_investigation_dataset()
        st.session_state["current_dataset"] = demo_df
        st.session_state["uploaded_dataset"] = demo_df
        st.session_state["original_dataset"] = demo_df.copy()
        st.session_state["dataset_filename"] = "customer_churn_benchmark.csv"
        st.session_state["selected_target_variable"] = "churn"
        st.rerun()


def _render_overview_tab(df: pd.DataFrame, num_cols: List[str], cat_cols: List[str], engine: StatisticalEngine) -> None:
    """Tab 1: High-level overview of statistical properties and test execution counter."""
    st.markdown("### 📌 Statistical Investigation Overview")
    st.caption("Summary of available variable distributions, hypothesis testing state, and significance tally.")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Numerical Variables", len(num_cols))
    with c2:
        st.metric("Categorical Variables", len(cat_cols))
    with c3:
        st.metric("Completed Tests Logged", len(engine.history))
    with c4:
        sig_count = sum(1 for r in engine.history if r.is_significant)
        st.metric("Statistically Significant (p < α)", sig_count)

    st.markdown("<hr style='border: none; border-top: 1px solid #1E293B; margin: 1.5rem 0;'>", unsafe_allow_html=True)

    st.markdown("#### 📜 Executed Hypothesis Tests in Current Session")
    if engine.history:
        summary_rows = []
        for idx, r in enumerate(engine.history, 1):
            summary_rows.append({
                "#": idx,
                "Test Variant": r.test_name,
                "Research Question": r.research_question,
                "Test Statistic": f"{r.statistic_name} = {r.test_statistic:.3f}",
                "P-Value": f"{r.p_value:.4f}" if r.p_value >= 0.0001 else f"{r.p_value:.2e}",
                "Decision": r.decision,
                "Effect Size": f"{r.effect_size:.3f} ({r.effect_size_name})" if r.effect_size is not None else "N/A",
            })
        st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)
        if st.button("🗑️ Clear Test History Log"):
            engine.clear_history()
            st.rerun()
    else:
        st.info("No statistical tests have been executed in this session yet. Use the tabs above to run hypothesis tests.")


def _render_recommender_tab(df: pd.DataFrame, num_cols: List[str], cat_cols: List[str], engine: StatisticalEngine) -> None:
    """Tab 2: Automated test recommendation engine."""
    st.markdown("### 🎯 Automated Statistical Test Recommender")
    st.caption("Intelligently recommends the most appropriate statistical test based on variable data types, number of groups, and study design.")

    c1, c2, c3 = st.columns([1.2, 1.2, 1.0])
    all_cols = list(df.columns)

    with c1:
        target_default = st.session_state.get("selected_target_variable")
        target_idx = all_cols.index(target_default) if target_default in all_cols else 0
        target_col = st.selectbox("Target / Outcome Variable (Y)", options=all_cols, index=target_idx)

    with c2:
        available_preds = [c for c in all_cols if c != target_col]
        predictor_col = st.selectbox("Predictor / Grouping Variable (X)", options=["None"] + available_preds)
        pred_arg = predictor_col if predictor_col != "None" else None

    with c3:
        study_design = st.selectbox("Study Design", options=["Independent Groups", "Paired / Repeated Measures"])
        design_arg = "paired" if "Paired" in study_design else "independent"

    # Evaluate recommendation
    rec = engine.recommend(df, target_col, pred_arg, design_arg)

    if rec.get("status") == "success":
        st.markdown(
            f"""
            <div style="background: #111827; border: 1px solid #1E293B; border-left: 4px solid #38BDF8; border-radius: 8px; padding: 1.2rem; margin-top: 1rem;">
                <div style="font-size: 0.8rem; font-weight: 700; color: #38BDF8; text-transform: uppercase;">Recommended Statistical Test</div>
                <div style="font-size: 1.25rem; font-weight: 800; color: #F8FAFC; margin-top: 0.25rem;">{rec['recommended_test']}</div>
                <div style="font-size: 0.92rem; color: #CBD5E1; margin-top: 0.5rem; line-height: 1.5;">{rec['justification']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        col_alt, col_assump = st.columns(2)
        with col_alt:
            st.markdown("#### 🔄 Suitable Alternatives")
            for alt in rec.get("alternative_tests", []):
                st.markdown(f"- **{alt}**")

        with col_assump:
            st.markdown("#### 📋 Core Assumptions to Verify")
            for assump in rec.get("assumptions_to_check", []):
                st.markdown(f"- {assump}")
    else:
        st.warning(rec.get("message", "Unable to formulate a recommendation."))


def _render_numerical_tab(df: pd.DataFrame, num_cols: List[str], cat_cols: List[str], engine: StatisticalEngine, alpha: float) -> None:
    """Tab 3: Parametric & Non-parametric Numerical Tests."""
    st.markdown("### 🔢 Numerical Variable Testing")
    st.caption("Compare central tendencies and distributions across independent or paired conditions.")

    test_type = st.radio(
        "Select Numerical Hypothesis Test:",
        options=[
            "Independent Samples T-Test (Two Groups)",
            "One-Way ANOVA (Three or More Groups)",
            "Paired Samples T-Test (Repeated Measures)",
            "Mann-Whitney U Test (Non-parametric Two Groups)",
            "Kruskal-Wallis Test (Non-parametric Three+ Groups)",
            "Wilcoxon Signed-Rank Test (Non-parametric Paired)",
        ],
        horizontal=True,
    )

    st.markdown("<hr style='border: none; border-top: 1px solid #1E293B; margin: 1rem 0;'>", unsafe_allow_html=True)

    if test_type in ("Independent Samples T-Test (Two Groups)", "Mann-Whitney U Test (Non-parametric Two Groups)"):
        c1, c2 = st.columns(2)
        with c1:
            num_var = st.selectbox("Numerical Metric (Outcome)", options=num_cols, key="num_t_metric")
        with c2:
            group_col = st.selectbox("Grouping Variable (Categorical)", options=cat_cols, key="num_t_group")

        unique_groups = list(df[group_col].dropna().unique())
        if len(unique_groups) < 2:
            st.warning(f"Column '{group_col}' has fewer than 2 distinct categories.")
            return

        c_g1, c_g2, c_ev = st.columns(3)
        with c_g1:
            g1_name = st.selectbox("Group 1", options=unique_groups, index=0, key="t_g1")
        with c_g2:
            g2_name = st.selectbox("Group 2", options=[g for g in unique_groups if g != g1_name], index=0, key="t_g2")
        with c_ev:
            equal_var_opt = st.selectbox("Equal Variance Assumption", options=["Auto (Levene Check)", "Assume Equal (Student's)", "Do Not Assume (Welch's)"])

        ev_arg = "auto" if "Auto" in equal_var_opt else ("true" if "Equal" in equal_var_opt else "false")

        g1_data = df.loc[df[group_col] == g1_name, num_var]
        g2_data = df.loc[df[group_col] == g2_name, num_var]

        if st.button("🚀 Run Two-Sample Test", type="primary"):
            if "Mann-Whitney" in test_type:
                res = engine.run_mann_whitney(g1_data, g2_data, str(g1_name), str(g2_name), num_var, alpha=alpha)
            else:
                res = engine.run_independent_t_test(g1_data, g2_data, str(g1_name), str(g2_name), num_var, alpha=alpha, equal_var=ev_arg)
            st.session_state["latest_stat_result"] = res
            _display_test_result(res)

            # Display CI plot if t-test
            if res.ci_lower is not None and res.ci_upper is not None:
                m1, m2 = float(g1_data.mean()), float(g2_data.mean())
                # Group-level 95% CIs
                se1 = float(g1_data.std() / np.sqrt(len(g1_data))) if len(g1_data) > 0 else 0
                se2 = float(g2_data.std() / np.sqrt(len(g2_data))) if len(g2_data) > 0 else 0
                fig = plot_mean_comparison_ci(
                    [str(g1_name), str(g2_name)],
                    [m1, m2],
                    [m1 - 1.96 * se1, m2 - 1.96 * se2],
                    [m1 + 1.96 * se1, m2 + 1.96 * se2],
                    variable_name=num_var,
                )
                st.plotly_chart(fig, use_container_width=True)

    elif test_type in ("One-Way ANOVA (Three or More Groups)", "Kruskal-Wallis Test (Non-parametric Three+ Groups)"):
        c1, c2 = st.columns(2)
        with c1:
            num_var = st.selectbox("Numerical Outcome", options=num_cols, key="anova_metric")
        with c2:
            group_col = st.selectbox("Categorical Factor", options=cat_cols, key="anova_group")

        unique_groups = list(df[group_col].dropna().unique())
        if len(unique_groups) < 3:
            st.warning(f"Grouping column '{group_col}' has {len(unique_groups)} groups. Minimum 3 groups expected for multi-group comparison.")

        groups_dict = {str(grp): df.loc[df[group_col] == grp, num_var] for grp in unique_groups}

        if st.button("🚀 Run Multi-Group Test", type="primary"):
            if "Kruskal" in test_type:
                res = engine.run_kruskal_wallis(groups_dict, num_var, group_col, alpha=alpha)
            else:
                res = engine.run_anova(groups_dict, num_var, group_col, alpha=alpha)
            st.session_state["latest_stat_result"] = res
            _display_test_result(res)

    else:  # Paired tests
        c1, c2 = st.columns(2)
        with c1:
            col1 = st.selectbox("Condition 1 / Pre Measurement", options=num_cols, key="paired_c1")
        with c2:
            col2 = st.selectbox("Condition 2 / Post Measurement", options=[c for c in num_cols if c != col1], key="paired_c2")

        if st.button("🚀 Run Paired Comparison", type="primary"):
            if "Wilcoxon" in test_type:
                res = engine.run_wilcoxon(df[col1], df[col2], f"{col2} vs {col1}", alpha=alpha)
            else:
                res = engine.run_paired_t_test(df[col1], df[col2], col1, col2, f"{col2} - {col1}", alpha=alpha)
            st.session_state["latest_stat_result"] = res
            _display_test_result(res)


def _render_categorical_tab(df: pd.DataFrame, cat_cols: List[str], num_cols: List[str], engine: StatisticalEngine, alpha: float) -> None:
    """Tab 4: Categorical Tests (Chi-Square, Fisher's Exact, Two-Proportions)."""
    st.markdown("### 🏷️ Categorical Variable Testing")
    st.caption("Investigate contingency independence, odds ratios, and event proportion differences.")

    cat_test_type = st.radio(
        "Select Categorical Test:",
        options=[
            "Chi-Square Test of Independence",
            "Fisher's Exact Test (2x2)",
            "Two-Proportion Z-Test",
        ],
        horizontal=True,
    )

    if cat_test_type == "Chi-Square Test of Independence":
        c1, c2 = st.columns(2)
        with c1:
            col1 = st.selectbox("Row Categorical Variable", options=cat_cols, key="chi_c1")
        with c2:
            col2 = st.selectbox("Column Categorical Variable", options=[c for c in cat_cols if c != col1], key="chi_c2")

        if st.button("🚀 Run Chi-Square Test", type="primary"):
            res = engine.run_chi_square(df[col1], df[col2], col1, col2, alpha=alpha)
            st.session_state["latest_stat_result"] = res
            _display_test_result(res)

            # Display Heatmap of observed vs expected
            metrics = res.additional_metrics
            if "observed_table" in metrics and "expected_table" in metrics:
                fig = plot_contingency_heatmap(metrics["observed_table"], metrics["expected_table"])
                st.plotly_chart(fig, use_container_width=True)

    elif cat_test_type == "Fisher's Exact Test (2x2)":
        c1, c2 = st.columns(2)
        with c1:
            col1 = st.selectbox("Binary Row Variable", options=cat_cols, key="fish_c1")
        with c2:
            col2 = st.selectbox("Binary Column Variable", options=[c for c in cat_cols if c != col1], key="fish_c2")

        ct = pd.crosstab(df[col1], df[col2])
        st.write("Observed 2x2 Contingency Table:")
        st.dataframe(ct, use_container_width=True)

        if ct.shape != (2, 2):
            st.warning("Selected variables do not form a 2x2 table.")
            return

        if st.button("🚀 Run Fisher's Exact Test", type="primary"):
            res = engine.run_fisher_exact(ct, col1, col2, alpha=alpha)
            st.session_state["latest_stat_result"] = res
            _display_test_result(res)

    else:  # Two-Proportion Z-Test
        c1, c2 = st.columns(2)
        with c1:
            group_col = st.selectbox("Grouping Variable (2 Cohorts)", options=cat_cols, key="prop_grp")
        with c2:
            outcome_col = st.selectbox("Binary Outcome Variable", options=cat_cols, key="prop_out")

        unique_grps = list(df[group_col].dropna().unique())
        unique_outcomes = list(df[outcome_col].dropna().unique())

        if len(unique_grps) < 2 or len(unique_outcomes) < 2:
            st.warning("Both variables must have at least 2 distinct values.")
            return

        c_g1, c_g2, c_tgt = st.columns(3)
        with c_g1:
            g1 = st.selectbox("Cohort 1", options=unique_grps, index=0)
        with c_g2:
            g2 = st.selectbox("Cohort 2", options=[g for g in unique_grps if g != g1], index=0)
        with c_tgt:
            target_event = st.selectbox("Target Event Category", options=unique_outcomes, index=0)

        sub1 = df[df[group_col] == g1]
        sub2 = df[df[group_col] == g2]

        cnt1 = int((sub1[outcome_col] == target_event).sum())
        n1 = len(sub1)
        cnt2 = int((sub2[outcome_col] == target_event).sum())
        n2 = len(sub2)

        st.caption(f"{g1}: {cnt1}/{n1} ({cnt1/n1*100:.1f}%) | {g2}: {cnt2}/{n2} ({cnt2/n2*100:.1f}%)")

        if st.button("🚀 Run Two-Proportion Test", type="primary"):
            res = StatisticalEngine.run_two_proportion_test(
                cnt1, n1, cnt2, n2, str(g1), str(g2), str(target_event), alpha=alpha
            )
            engine.record_test(res)
            st.session_state["latest_stat_result"] = res
            _display_test_result(res)


def _render_correlation_tab(df: pd.DataFrame, num_cols: List[str], engine: StatisticalEngine, alpha: float) -> None:
    """Tab 5: Pearson, Spearman, and Kendall correlation significance tests."""
    st.markdown("### 📈 Correlation Significance Testing")
    st.caption("Quantify linear and monotonic relationship strengths, p-values, and 95% Fisher z-confidence intervals.")

    if len(num_cols) < 2:
        st.warning("At least two numerical variables are required for correlation analysis.")
        return

    c1, c2, c3 = st.columns([1.2, 1.2, 1.0])
    with c1:
        x_var = st.selectbox("Variable X", options=num_cols, key="corr_x")
    with c2:
        y_var = st.selectbox("Variable Y", options=[c for c in num_cols if c != x_var], key="corr_y")
    with c3:
        corr_method = st.selectbox(
            "Correlation Algorithm",
            options=["Pearson (Linear)", "Spearman (Rank / Monotonic)", "Kendall (Tau / Small Sample)"],
        )
        method_arg = "pearson" if "Pearson" in corr_method else ("spearman" if "Spearman" in corr_method else "kendall")

    if st.button("🚀 Test Correlation Significance", type="primary"):
        res = engine.run_correlation(df[x_var], df[y_var], x_var, y_var, method=method_arg, alpha=alpha)
        st.session_state["latest_stat_result"] = res
        _display_test_result(res)

        fig = plot_correlation_scatter(df[x_var], df[y_var], x_var, y_var, res.test_statistic, res.p_value)
        st.plotly_chart(fig, use_container_width=True)


def _render_assumptions_tab(df: pd.DataFrame, num_cols: List[str], cat_cols: List[str], engine: StatisticalEngine, alpha: float) -> None:
    """Tab 6: Normality testing, Q-Q plots, and variance homoscedasticity audits."""
    st.markdown("### 🔍 Statistical Assumption Auditing")
    st.caption("Verify normality (Shapiro-Wilk, D'Agostino K²) and homogeneity of variance (Levene / Brown-Forsythe).")

    sub_t1, sub_t2 = st.tabs(["Normality Diagnostic (Q-Q Plot)", "Homogeneity of Variance (Levene)"])

    with sub_t1:
        target_num = st.selectbox("Select Numerical Feature for Normality Check", options=num_cols, key="norm_col")
        norm_res = engine.check_normality(df[target_num], alpha=alpha)

        col_n1, col_n2 = st.columns([1, 1.5])
        with col_n1:
            st.markdown("#### Test Result")
            st.markdown(
                f"""
                <div style="background: #111827; border: 1px solid #1E293B; border-left: 4px solid {'#10B981' if norm_res['is_normal'] else '#EF4444'}; border-radius: 8px; padding: 1rem;">
                    <div style="font-weight: 700; color: #F8FAFC;">{norm_res['test_name']}</div>
                    <div style="font-size: 0.9rem; color: #E2E8F0; margin-top: 0.25rem;">
                        Statistic: <strong>{norm_res.get('statistic_name', 'Stat')} = {norm_res.get('statistic', 0.0):.4f}</strong><br>
                        P-Value: <strong>{norm_res.get('p_value', 1.0):.4f}</strong> (α = {alpha})<br>
                        Skewness: <strong>{norm_res.get('skewness', 0.0)}</strong> &nbsp;|&nbsp; Kurtosis: <strong>{norm_res.get('kurtosis', 0.0)}</strong>
                    </div>
                    <div style="font-size: 0.85rem; color: #CBD5E1; margin-top: 0.5rem;">{norm_res['interpretation']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if norm_res.get("caveat"):
                st.caption(f"ℹ️ {norm_res['caveat']}")

        with col_n2:
            theo, samp, slope, intercept = engine.get_qq_plot(df[target_num])
            fig_qq = plot_qq_plot(theo, samp, slope, intercept, title=f"Q-Q Plot for '{target_num}'")
            st.plotly_chart(fig_qq, use_container_width=True)

    with sub_t2:
        c1, c2 = st.columns(2)
        with c1:
            v_metric = st.selectbox("Numerical Metric to Test Variance Across Groups", options=num_cols, key="var_col")
        with c2:
            v_group = st.selectbox("Grouping Factor", options=cat_cols, key="var_grp")

        grps = [df.loc[df[v_group] == val, v_metric] for val in df[v_group].dropna().unique()]
        var_res = engine.check_variance(grps, alpha=alpha, center="median")

        st.markdown("#### Levene's Test Result")
        st.markdown(
            f"""
            <div style="background: #111827; border: 1px solid #1E293B; border-left: 4px solid {'#10B981' if var_res.get('equal_variance') else '#EF4444'}; border-radius: 8px; padding: 1rem;">
                <div style="font-weight: 700; color: #F8FAFC;">{var_res.get('test_name')}</div>
                <div style="font-size: 0.9rem; color: #E2E8F0; margin-top: 0.25rem;">
                    Statistic: <strong>W = {var_res.get('statistic', 0.0):.4f}</strong> &nbsp;|&nbsp;
                    P-Value: <strong>{var_res.get('p_value', 1.0):.4f}</strong> (α = {alpha})
                </div>
                <div style="font-size: 0.85rem; color: #CBD5E1; margin-top: 0.5rem;">
                    {var_res.get('interpretation')}<br>
                    <strong>Guidance:</strong> {var_res.get('recommendation')}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def _render_effects_tab(engine: StatisticalEngine) -> None:
    """Tab 7: Effect Sizes and Confidence Intervals summary."""
    st.markdown("### 📐 Effect Sizes & Confidence Intervals")
    st.caption("Standardized effect magnitudes and interval estimation across executed tests.")

    if not engine.history:
        st.info("Execute tests in the previous tabs to review computed effect sizes and confidence intervals.")
        return

    records = []
    for r in engine.history:
        ci_str = f"[{r.ci_lower:+.3f}, {r.ci_upper:+.3f}]" if r.ci_lower is not None else "N/A"
        records.append({
            "Test Name": r.test_name,
            "Research Question": r.research_question,
            "Effect Metric": r.effect_size_name or "N/A",
            "Effect Magnitude": f"{r.effect_size:.4f}" if r.effect_size is not None else "N/A",
            "Interpretation": r.effect_size_interpretation or "N/A",
            "95% Confidence Interval": ci_str,
            "Significant": "✓ Yes" if r.is_significant else "✗ No",
        })

    st.dataframe(pd.DataFrame(records), use_container_width=True, hide_index=True)


def _render_multiple_testing_tab(engine: StatisticalEngine, alpha: float) -> None:
    """Tab 8: Multiple hypothesis testing adjustment."""
    st.markdown("### ⚖️ Multiple Hypothesis Testing Correction")
    st.caption("Control Family-Wise Error Rate (Bonferroni) or False Discovery Rate (Benjamini-Hochberg) across multiple concurrent tests.")

    if len(engine.history) < 2:
        st.info(f"Multiple testing correction is recommended when testing multiple hypotheses concurrently. Currently {len(engine.history)} test is recorded in history. Run at least 2 tests to activate multiplicity adjustments.")
        return

    c1, c2 = st.columns([1.5, 1])
    with c1:
        method = st.selectbox(
            "Multiplicity Correction Procedure",
            options=["Benjamini-Hochberg (False Discovery Rate - Recommended)", "Bonferroni (Family-Wise Error Rate)"],
        )
        method_arg = "bonferroni" if "Bonferroni" in method else "benjamini_hochberg"

    adj_df = engine.apply_multiple_testing_correction(method=method_arg, alpha=alpha)

    st.markdown("#### Comparison of Raw vs. Adjusted Significant Findings")
    st.dataframe(adj_df, use_container_width=True, hide_index=True)

    # Plot Comparison
    raw_p_vals = [r.p_value for r in engine.history]
    if method_arg == "bonferroni":
        adj_p_vals, _ = bonferroni_correction(raw_p_vals, alpha=alpha)
    else:
        adj_p_vals, _ = benjamini_hochberg_correction(raw_p_vals, alpha=alpha)

    labels = [f"T{i}: {r.test_name[:15]}..." for i, r in enumerate(engine.history, 1)]
    fig = plot_multiple_testing_comparison(labels, raw_p_vals, adj_p_vals, alpha=alpha)
    st.plotly_chart(fig, use_container_width=True)


def _render_narrative_tab(engine: StatisticalEngine) -> None:
    """Tab 9: Narrative interpretation dossier."""
    st.markdown("### 💡 Automated Statistical Interpretation Dossier")
    st.caption("Factual, non-causal synthesis of statistical findings and empirical conclusions.")

    active_res = st.session_state.get("latest_stat_result")
    if active_res is None:
        if engine.history:
            active_res = engine.history[-1]
        else:
            st.info("Execute a statistical test above to generate an automated interpretation narrative.")
            return

    narrative = engine.interpret(active_res)

    st.markdown(
        f"""
        <div style="background: #111827; border: 1px solid #1E293B; border-radius: 8px; padding: 1.5rem; margin-bottom: 1.5rem;">
            <div style="font-size: 1.3rem; font-weight: 800; color: #F8FAFC;">{narrative['title']}</div>
            <div style="font-size: 0.9rem; color: #38BDF8; margin-top: 0.25rem;">Research Question: {narrative['research_question']}</div>
            <hr style="border: none; border-top: 1px solid #1E293B; margin: 1rem 0;">
            <div style="font-size: 1.05rem; font-weight: 700; color: #F8FAFC;">Statistical Decision: {narrative['statistical_decision']}</div>
            <p style="color: #E2E8F0; font-size: 0.95rem; line-height: 1.6; margin-top: 0.5rem;">{narrative['decision_narrative']}</p>
            <div style="font-size: 0.95rem; color: #CBD5E1; margin-top: 0.75rem;">
                <strong>Effect Magnitude:</strong> {narrative['effect_narrative']}<br>
                <strong>Confidence Interval:</strong> {narrative['ci_narrative']}
            </div>
            <div style="background: rgba(239, 68, 68, 0.08); border-left: 3px solid #EF4444; padding: 0.75rem; border-radius: 4px; margin-top: 1rem; font-size: 0.85rem; color: #FCA5A5;">
                {narrative['non_causality_notice']}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if narrative.get("limitations"):
        st.markdown("#### ⚠️ Test Limitations & Caveats")
        for lim in narrative["limitations"]:
            st.markdown(f"- {lim}")


def _render_export_tab(engine: StatisticalEngine) -> None:
    """Tab 10: Export CSV reports for completed tests and multiple testing results."""
    st.markdown("### 📥 Export Statistical Results")
    st.caption("Download all empirical hypothesis test logs, summary parameters, and adjusted p-values as CSV.")

    if not engine.history:
        st.info("No test results to export yet. Execute statistical tests to populate the export center.")
        return

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Hypothesis Testing Summary")
        history_df = pd.DataFrame([r.to_dict() for r in engine.history])
        csv_hist = history_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Download Hypothesis Tests Log (CSV)",
            csv_hist,
            "hypothesis_testing_log.csv",
            "text/csv",
            use_container_width=True,
        )

    with c2:
        st.markdown("#### Multiple Testing Correction Summary")
        adj_df = engine.apply_multiple_testing_correction(method="benjamini_hochberg")
        csv_adj = adj_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Download Multiple Testing Adjustments (CSV)",
            csv_adj,
            "multiple_testing_adjustments.csv",
            "text/csv",
            use_container_width=True,
        )


def _display_test_result(res: HypothesisTestResult) -> None:
    """Helper component displaying formatted test output card and summary table."""
    sig_color = "#10B981" if res.is_significant else "#94A3B8"
    st.markdown(
        f"""
        <div style="background: #111827; border: 1px solid #1E293B; border-left: 4px solid {sig_color}; border-radius: 8px; padding: 1.2rem; margin: 1.2rem 0;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div style="font-weight: 700; font-size: 1.15rem; color: #F8FAFC;">{res.test_name}</div>
                <span style="font-size: 0.8rem; font-weight: 700; padding: 3px 10px; border-radius: 4px; background: rgba(255,255,255,0.08); color: {sig_color};">
                    {res.decision}
                </span>
            </div>
            <div style="font-size: 0.95rem; color: #CBD5E1; margin-top: 0.6rem; line-height: 1.5;">{res.interpretation}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.dataframe(res.summary_table(), use_container_width=True, hide_index=True)

