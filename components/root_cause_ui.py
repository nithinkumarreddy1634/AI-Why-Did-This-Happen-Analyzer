"""Interactive Streamlit UI for the Root Cause Investigation Studio."""

from typing import Dict, Any, List, Optional
import streamlit as st
import pandas as pd
import numpy as np

from modules.root_cause.investigation_manager import (
    RootCauseInvestigationManager,
    ActiveInvestigationPackage,
)
from modules.root_cause.segment_contribution import analyze_feature_segment_contributions
from modules.root_cause.interaction_analysis import analyze_feature_interactions
from components.root_cause_charts import (
    plot_root_cause_score_bar,
    plot_evidence_matrix_heatmap,
    plot_segment_contribution_waterfall,
    plot_interaction_heatmap,
)


def render_root_cause_studio_page() -> None:
    """Render the master Root Cause Investigation Studio page."""
    # 1. Initialize Root Cause Manager in Session State
    if "root_cause_manager" not in st.session_state:
        st.session_state["root_cause_manager"] = RootCauseInvestigationManager()
    if "root_cause_active_package" not in st.session_state:
        st.session_state["root_cause_active_package"] = None

    manager: RootCauseInvestigationManager = st.session_state["root_cause_manager"]
    active_pkg: Optional[ActiveInvestigationPackage] = st.session_state["root_cause_active_package"]

    # 2. Page Header
    st.markdown(
        """
        <div style="margin-bottom: 1.5rem;">
            <div style="display: flex; align-items: center; gap: 0.75rem;">
                <span style="font-size: 2.2rem;">🔬</span>
                <div>
                    <h1 style="margin: 0; font-size: 1.85rem; font-weight: 800; color: #F8FAFC; letter-spacing: -0.02em;">
                        Root Cause Investigation Studio
                    </h1>
                    <p style="margin: 0.2rem 0 0 0; font-size: 0.88rem; color: #94A3B8;">
                        Multi-Engine Evidence Aggregation, Candidate Ranking, and Segment Decomposition
                    </p>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 3. Non-Causal Methodological Banner
    st.info(
        "**Evidence-Driven Contributing Factor Engine**: All findings reflect empirical associations, predictive "
        "importance, and cross-method consensus across statistical and ML engines. They represent **potential contributing factors** "
        "— not proven counterfactual causality, which cannot be confirmed from observational data alone."
    )

    # 4. Resolve Active Dataset
    df = (
        st.session_state.get("preprocessed_dataset")
        or st.session_state.get("uploaded_dataset")
        or st.session_state.get("current_dataset")
    )
    dataset_name = st.session_state.get("dataset_filename", "active_dataset.csv")

    if df is None or df.empty:
        st.warning("⚠️ No active dataset loaded. Please upload or load a dataset in **Dataset Overview** first.")
        return

    # Check for dataset staleness if an investigation is active
    if active_pkg:
        is_stale, stale_msg = manager.verify_dataset_synchronization(df, dataset_name)
        if is_stale:
            st.warning(f"⚠️ **Dataset Synchronization Warning:** {stale_msg}")

    # 5. Top Investigation Setup & Configuration Expander
    with st.expander("🛠️ Configure or Launch New Investigation", expanded=(active_pkg is None)):
        col1, col2 = st.columns([1, 1])

        with col1:
            inv_title = st.text_input(
                "Investigation Title",
                value="Unexpected Outcome Investigation",
                help="Descriptive title for this investigation run.",
            )
            # Target Column Selection
            default_target = st.session_state.get("selected_target_variable")
            cols_list = list(df.columns)
            target_idx = cols_list.index(default_target) if default_target in cols_list else 0
            target_col = st.selectbox("Target / Outcome Variable", options=cols_list, index=target_idx)

        with col2:
            # Baseline & Comparison selection
            st.markdown("<p style='font-size: 0.85rem; font-weight: 600; color: #CBD5E1; margin-bottom: 4px;'>Cohort Segmentation Strategy</p>", unsafe_allow_html=True)
            # Find candidate categorical columns for cohort comparison
            cat_cols = [c for c in df.columns if c != target_col and (df[c].dtype == "object" or df[c].nunique() <= 10)]
            if not cat_cols:
                cat_cols = [c for c in df.columns if c != target_col]

            split_col = st.selectbox(
                "Cohort Split Column",
                options=cat_cols,
                help="Column used to partition baseline reference group vs comparison affected group.",
            )
            unique_vals = list(df[split_col].dropna().unique())
            val_strs = [str(v) for v in unique_vals]

            col_sub1, col_sub2 = st.columns(2)
            with col_sub1:
                base_val = st.selectbox("Baseline Group", options=val_strs, index=0)
            with col_sub2:
                comp_idx = 1 if len(val_strs) > 1 else 0
                comp_val = st.selectbox("Comparison Group", options=val_strs, index=comp_idx)

        # Candidate Features Selection
        all_candidate_feats = [c for c in df.columns if c not in (target_col, split_col)]
        selected_candidates = st.multiselect(
            "Candidate Features to Investigate",
            options=all_candidate_feats,
            default=all_candidate_feats[:min(12, len(all_candidate_feats))],
            help="Select features to analyze and rank as potential root cause factors.",
        )

        run_btn = st.button("🚀 Run Root Cause Investigation", type="primary", use_container_width=True)

        if run_btn:
            with st.spinner("Executing multi-engine evidence collection, statistical testing, and root-cause ranking..."):
                base_mask = (df[split_col].astype(str) == base_val)
                comp_mask = (df[split_col].astype(str) == comp_val)

                if base_mask.sum() < 3 or comp_mask.sum() < 3:
                    st.error("Each cohort must contain at least 3 records for valid statistical comparison.")
                else:
                    # Retrieve available Phase 7, 8, 9, 10 artifacts from session_state
                    ml_imp = st.session_state.get("ml_active_importance")
                    shap_g = st.session_state.get("shap_active_global")
                    shap_l = st.session_state.get("shap_active_local")
                    lime_l = st.session_state.get("lime_active_result")
                    anom_art = st.session_state.get("detection_artifacts")

                    new_pkg = manager.execute_investigation(
                        df=df,
                        target_col=target_col,
                        baseline_mask=base_mask,
                        comparison_mask=comp_mask,
                        title=inv_title,
                        dataset_name=dataset_name,
                        candidate_features=selected_candidates,
                        baseline_name=f"{split_col}={base_val}",
                        comparison_name=f"{split_col}={comp_val}",
                        ml_importance_res=ml_imp,
                        shap_global_res=shap_g,
                        shap_local_res=shap_l,
                        lime_local_res=lime_l,
                        anomaly_artifacts=anom_art,
                    )
                    st.session_state["root_cause_active_package"] = new_pkg
                    active_pkg = new_pkg
                    st.success("✅ Root Cause Investigation completed successfully!")

    if not active_pkg:
        st.info("👆 Configure your investigation parameters above and click **'Run Root Cause Investigation'** to begin.")
        return

    # 6. Overview KPI Ribbon
    outcome = active_pkg.outcome_def
    top_cand = active_pkg.ranking_result.top_candidate_name

    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    with kpi1:
        st.markdown(
            f"""
            <div style="background: #111827; border: 1px solid #1F2937; border-radius: 8px; padding: 0.85rem; text-align: center;">
                <div style="font-size: 0.72rem; color: #94A3B8; text-transform: uppercase;">Target Variable</div>
                <div style="font-size: 1.15rem; font-weight: 700; color: #F8FAFC; margin-top: 2px;">{outcome.target_variable}</div>
                <div style="font-size: 0.7rem; color: #64748B;">{outcome.outcome_type.title()}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with kpi2:
        st.markdown(
            f"""
            <div style="background: #111827; border: 1px solid #1F2937; border-radius: 8px; padding: 0.85rem; text-align: center;">
                <div style="font-size: 0.72rem; color: #94A3B8; text-transform: uppercase;">Baseline Value</div>
                <div style="font-size: 1.15rem; font-weight: 700; color: #60A5FA; margin-top: 2px;">{outcome.baseline_value:.3f}</div>
                <div style="font-size: 0.7rem; color: #64748B;">N = {outcome.baseline_count:,}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with kpi3:
        st.markdown(
            f"""
            <div style="background: #111827; border: 1px solid #1F2937; border-radius: 8px; padding: 0.85rem; text-align: center;">
                <div style="font-size: 0.72rem; color: #94A3B8; text-transform: uppercase;">Comparison Value</div>
                <div style="font-size: 1.15rem; font-weight: 700; color: #F87171; margin-top: 2px;">{outcome.comparison_value:.3f}</div>
                <div style="font-size: 0.7rem; color: #64748B;">N = {outcome.comparison_count:,}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with kpi4:
        pct_display = f"{outcome.relative_change_pct:+.1f}%" if outcome.relative_change_pct is not None else "N/A"
        st.markdown(
            f"""
            <div style="background: #111827; border: 1px solid #1F2937; border-radius: 8px; padding: 0.85rem; text-align: center;">
                <div style="font-size: 0.72rem; color: #94A3B8; text-transform: uppercase;">Observed Delta (Δ)</div>
                <div style="font-size: 1.15rem; font-weight: 700; color: #F59E0B; margin-top: 2px;">{outcome.absolute_change:+.3f}</div>
                <div style="font-size: 0.7rem; color: #94A3B8;">{pct_display}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with kpi5:
        st.markdown(
            f"""
            <div style="background: #111827; border: 1px solid rgba(16, 185, 129, 0.3); border-radius: 8px; padding: 0.85rem; text-align: center;">
                <div style="font-size: 0.72rem; color: #34D399; text-transform: uppercase; font-weight: 600;">Top Contributing Factor</div>
                <div style="font-size: 1.15rem; font-weight: 800; color: #10B981; margin-top: 2px;">{top_cand}</div>
                <div style="font-size: 0.7rem; color: #94A3B8;">Rank #1 Candidate</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

    # 7. Navigation Tabs for Investigation Studio
    t1, t2, t3, t4, t5, t6, t7 = st.tabs([
        "🏆 Root Cause Rankings",
        "🧩 Cross-Method Evidence Matrix",
        "👥 Segment Decomposition",
        "🔗 Combined Factor Interactions",
        "📐 Statistical & ML Diagnostics",
        "📑 AI Investigation Summary",
        "💾 One-Click Exports",
    ])

    # TAB 1: Root Cause Rankings
    with t1:
        st.markdown("### 🏆 Contributing Factor Priority Ranking")
        st.markdown(
            "Candidates are evaluated using normalized evidence scoring across observed magnitude, statistical significance, "
            "ML predictive importance, SHAP global/local impact, LIME surrogate rules, and outlier clustering."
        )

        col_c1, col_c2 = st.columns([1.2, 1.0])
        with col_c1:
            fig_scores = plot_root_cause_score_bar(active_pkg.ranking_result, top_n=8)
            st.plotly_chart(fig_scores, use_container_width=True)
        with col_c2:
            st.markdown("<p style='font-size: 0.85rem; font-weight: 700; color: #CBD5E1;'>Detailed Factor Ranking Table</p>", unsafe_allow_html=True)
            st.dataframe(active_pkg.ranking_result.ranking_df, use_container_width=True, hide_index=True)

        # Inspect individual candidate
        st.markdown("<hr style='border: none; border-top: 1px solid #1E293B; margin: 1.5rem 0;'>", unsafe_allow_html=True)
        st.markdown("#### 🔍 Candidate Deep Dive")
        cand_names = [c.feature for c in active_pkg.ranking_result.ranked_candidates]
        sel_c_name = st.selectbox("Select Candidate Feature to Inspect", options=cand_names, index=0)
        selected_item = next((c for c in active_pkg.ranking_result.ranked_candidates if c.feature == sel_c_name), None)

        if selected_item:
            d1, d2 = st.columns([1, 1])
            with d1:
                st.markdown(
                    f"""
                    <div style="background: #111827; border: 1px solid #1F2937; border-radius: 8px; padding: 1rem;">
                        <div style="font-size: 0.82rem; color: #38BDF8; font-weight: 700; text-transform: uppercase;">
                            Rank #{selected_item.rank}: {selected_item.feature}
                        </div>
                        <div style="margin-top: 0.5rem; font-size: 0.88rem; color: #E2E8F0;">
                            <strong>Overall Evidence Score:</strong> {selected_item.overall_score:.3f}<br>
                            <strong>Evidence Strength:</strong> <span style="color: #34D399;">{selected_item.evidence_strength}</span><br>
                            <strong>Observed Shift:</strong> {selected_item.absolute_shift:+.3f} ({selected_item.shift_direction})<br>
                            <strong>Baseline Cohort:</strong> {selected_item.baseline_display}<br>
                            <strong>Comparison Cohort:</strong> {selected_item.comparison_display}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with d2:
                st.markdown(
                    f"""
                    <div style="background: #111827; border: 1px solid #1F2937; border-radius: 8px; padding: 1rem;">
                        <div style="font-size: 0.82rem; color: #A5B4FC; font-weight: 700; text-transform: uppercase;">
                            Scoring Rationale & Limitations
                        </div>
                        <div style="margin-top: 0.5rem; font-size: 0.84rem; color: #94A3B8; line-height: 1.5;">
                            {selected_item.ranking_rationale}
                        </div>
                        <div style="margin-top: 0.5rem; font-size: 0.78rem; color: #F59E0B;">
                            ⚠️ {selected_item.confidence_limitations[0]}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    # TAB 2: Cross-Method Evidence Matrix
    with t2:
        st.markdown("### 🧩 Multi-Engine Evidence Matrix")
        st.markdown(
            "Visual heatmap displaying whether each analytical technique provides **Supporting**, **Neutral**, "
            "**Opposing**, or **Not Available** evidence for each candidate factor."
        )

        fig_matrix = plot_evidence_matrix_heatmap(active_pkg.ranking_result, top_n=10)
        st.plotly_chart(fig_matrix, use_container_width=True)

        st.dataframe(active_pkg.ranking_result.evidence_matrix_df, use_container_width=True, hide_index=True)

    # TAB 3: Segment Decomposition
    with t3:
        st.markdown("### 👥 Segment Contribution Decomposition")
        st.markdown(
            "Decomposes the total outcome delta into individual segment shares, distinguishing between high outcome rate "
            "and true population contribution."
        )

        # Feature selector for segment analysis
        seg_feat_options = [c.feature for c in active_pkg.ranking_result.ranked_candidates if c.feature in df.columns]
        current_seg_feat = active_pkg.segment_result.feature if active_pkg.segment_result else seg_feat_options[0]
        sel_seg_feat = st.selectbox(
            "Select Feature for Segment Analysis",
            options=seg_feat_options,
            index=seg_feat_options.index(current_seg_feat) if current_seg_feat in seg_feat_options else 0,
        )

        # Re-run or use active segment result
        if active_pkg.segment_result and active_pkg.segment_result.feature == sel_seg_feat:
            seg_res = active_pkg.segment_result
        else:
            base_mask = (df[split_col].astype(str) == base_val)
            comp_mask = (df[split_col].astype(str) == comp_val)
            seg_res = analyze_feature_segment_contributions(
                df=df,
                feature=sel_seg_feat,
                target_col=target_col,
                baseline_mask=base_mask,
                comparison_mask=comp_mask,
            )

        col_s1, col_s2 = st.columns([1.2, 1.0])
        with col_s1:
            fig_waterfall = plot_segment_contribution_waterfall(seg_res)
            st.plotly_chart(fig_waterfall, use_container_width=True)
        with col_s2:
            st.markdown(
                f"""
                <div style="background: #111827; border: 1px solid #1F2937; border-radius: 8px; padding: 1rem; margin-bottom: 1rem;">
                    <div style="font-size: 0.82rem; color: #10B981; font-weight: 700; text-transform: uppercase;">
                        Key Segment Findings
                    </div>
                    <div style="font-size: 0.85rem; color: #E2E8F0; margin-top: 0.5rem; line-height: 1.6;">
                        • <strong>Top Contributor:</strong> {seg_res.top_contributing_segment} (Largest driver of net change)<br>
                        • <strong>Peak Outcome Rate:</strong> {seg_res.highest_rate_segment}<br>
                        • <strong>Largest Population:</strong> {seg_res.largest_population_segment}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.dataframe(seg_res.summary_df, use_container_width=True, hide_index=True)

    # TAB 4: Combined Factor Interactions
    with t4:
        st.markdown("### 🔗 Combined Factor Interactions")
        st.markdown(
            "Investigate co-occurring pairs of features to discover non-linear risk compounding and super-additive shifts."
        )

        all_cands = [c.feature for c in active_pkg.ranking_result.ranked_candidates if c.feature in df.columns]
        if len(all_cands) >= 2:
            col_p1, col_p2 = st.columns(2)
            with col_p1:
                f1_sel = st.selectbox("Primary Feature", options=all_cands, index=0)
            with col_p2:
                f2_options = [c for c in all_cands if c != f1_sel]
                f2_sel = st.selectbox("Secondary Feature", options=f2_options, index=0 if f2_options else 0)

            base_mask = (df[split_col].astype(str) == base_val)
            comp_mask = (df[split_col].astype(str) == comp_val)

            inter_res = analyze_feature_interactions(
                df=df,
                feature1=f1_sel,
                feature2=f2_sel,
                target_col=target_col,
                baseline_mask=base_mask,
                comparison_mask=comp_mask,
            )

            fig_inter = plot_interaction_heatmap(inter_res)
            st.plotly_chart(fig_inter, use_container_width=True)

            st.info(f"💡 **Synergy Finding:** {inter_res.interaction_synergy_note}")

    # TAB 5: Statistical & ML Diagnostics
    with t5:
        st.markdown("### 📐 Statistical & ML Diagnostic Details")
        st.markdown("Examine the underlying p-values, test statistics, and ML importance scores.")

        diag_rows = []
        for e in active_pkg.evidence_list:
            diag_rows.append({
                "Feature": e.feature,
                "Stat Test": e.stat_test_name or "Not Run",
                "p-value": f"{e.p_value:.4e}" if e.p_value is not None else "N/A",
                "Significant (α=0.05)": "✅ Yes" if e.is_stat_significant else "❌ No",
                "Effect Size": f"{e.effect_size:.2f} ({e.effect_size_name})" if e.effect_size is not None else "N/A",
                "ML Rank": f"#{e.ml_importance_rank}" if e.ml_importance_rank else "N/A",
                "SHAP Rank": f"#{e.shap_global_rank}" if e.shap_global_rank else "N/A",
                "LIME Rule": e.lime_condition or "N/A",
                "Outliers": e.unusual_record_count,
            })
        st.dataframe(pd.DataFrame(diag_rows), use_container_width=True, hide_index=True)

    # TAB 6: AI Investigation Summary
    with t6:
        st.markdown("### 📑 Automated Investigation Narrative")
        rep = active_pkg.summary_report

        st.markdown(
            f"""
            <div style="background: rgba(79, 70, 229, 0.08); border: 1px solid rgba(99, 102, 241, 0.3); border-radius: 8px; padding: 1rem; margin-bottom: 1.5rem;">
                <div style="font-size: 0.85rem; color: #A5B4FC; font-weight: 700; text-transform: uppercase;">
                    Executive Synthesis
                </div>
                <div style="font-size: 0.95rem; color: #F8FAFC; margin-top: 0.4rem; line-height: 1.6;">
                    {rep.executive_narrative}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.expander("📖 View Full 10-Question Structured Answers", expanded=True):
            st.markdown(f"**1. What unexpected outcome was observed?**\n{rep.q1_observed_outcome}\n")
            st.markdown(f"**2. How large was the observed change?**\n{rep.q2_change_magnitude}\n")
            st.markdown(f"**3. Which groups were most affected?**\n{rep.q3_most_affected_groups}\n")
            st.markdown(f"**4. Which features received strongest supporting evidence?**\n{rep.q4_top_evidence_features}\n")
            st.markdown(f"**5. Which statistical findings support the investigation?**\n{rep.q5_statistical_findings}\n")
            st.markdown(f"**6. What did the predictive model identify?**\n{rep.q6_ml_model_insights}\n")
            st.markdown(f"**7. What did SHAP and LIME reveal?**\n{rep.q7_shap_lime_revelations}\n")
            st.markdown(f"**8. Were unusual observations detected?**\n{rep.q8_unusual_observations}\n")
            st.markdown(f"**9. Which findings disagree or conflict?**\n{rep.q9_conflicting_findings}\n")
            st.markdown(f"**10. What information is still missing?**\n{rep.q10_missing_information}\n")

        st.markdown("#### 🎯 Recommended Action Items")
        for rec in rep.recommendations:
            st.markdown(f"- {rec}")

    # TAB 7: One-Click Exports
    with t7:
        st.markdown("### 💾 Export Investigation Results")
        st.markdown("Download auditable CSV reports, Markdown executive summaries, and full JSON artifacts.")

        exporter = manager.exporter
        csv_ranking = exporter.export_ranking_csv(active_pkg.ranking_result)
        csv_matrix = exporter.export_evidence_matrix_csv(active_pkg.ranking_result)
        csv_segment = exporter.export_segment_analysis_csv(active_pkg.segment_result)
        md_report = exporter.export_summary_markdown(active_pkg.summary_report)
        json_bundle = exporter.export_complete_json(
            metadata=active_pkg.metadata,
            outcome_def=active_pkg.outcome_def,
            ranking_result=active_pkg.ranking_result,
            summary_report=active_pkg.summary_report,
            segment_result=active_pkg.segment_result,
            interaction_result=active_pkg.interaction_result,
        )

        col_ex1, col_ex2 = st.columns(2)
        with col_ex1:
            st.download_button(
                "📥 Download Root Cause Rankings (CSV)",
                data=csv_ranking,
                file_name="root_cause_rankings.csv",
                mime="text/csv",
                use_container_width=True,
            )
            st.download_button(
                "📥 Download Cross-Method Evidence Matrix (CSV)",
                data=csv_matrix,
                file_name="evidence_matrix.csv",
                mime="text/csv",
                use_container_width=True,
            )
            st.download_button(
                "📥 Download Segment Contribution Share (CSV)",
                data=csv_segment,
                file_name="segment_contributions.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with col_ex2:
            st.download_button(
                "📥 Download Investigation Summary Report (Markdown)",
                data=md_report,
                file_name="investigation_summary_report.md",
                mime="text/markdown",
                use_container_width=True,
            )
            st.download_button(
                "📥 Download Complete Investigation Bundle (JSON)",
                data=json_bundle,
                file_name="investigation_complete_bundle.json",
                mime="application/json",
                use_container_width=True,
            )

