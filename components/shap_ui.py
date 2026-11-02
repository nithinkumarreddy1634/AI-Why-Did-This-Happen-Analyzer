"""Streamlit User Interface for Phase 8: SHAP Explainability Studio & Model Interpretability."""

from typing import Optional, Dict, Any, List
import streamlit as st
import pandas as pd
import numpy as np

from modules.explainability.explanation_manager import SHAPExplanationManager
from modules.explainability.shap_explainer import SHAPCalculationResult
from modules.explainability.shap_global import GlobalSHAPResult
from modules.explainability.shap_local import LocalSHAPResult
from modules.machine_learning.model_trainer import MachineLearningEngine
from modules.machine_learning.problem_detector import ProblemType
from components.shap_charts import (
    plot_shap_summary_bar,
    plot_shap_beeswarm,
    plot_shap_waterfall,
    plot_shap_force_bar,
    plot_shap_dependence,
    plot_interaction_matrix_heatmap,
)
from utils.sample_data import generate_churn_investigation_dataset


def render_shap_studio_page() -> None:
    """Master render function for the SHAP Explainability Studio."""
    # Session state initialization
    if "shap_manager" not in st.session_state:
        st.session_state["shap_manager"] = SHAPExplanationManager()
    if "shap_active_calc" not in st.session_state:
        st.session_state["shap_active_calc"] = None
    if "shap_active_global" not in st.session_state:
        st.session_state["shap_active_global"] = None
    if "shap_active_local" not in st.session_state:
        st.session_state["shap_active_local"] = None
    if "shap_selected_obs_idx" not in st.session_state:
        st.session_state["shap_selected_obs_idx"] = 0

    manager: SHAPExplanationManager = st.session_state["shap_manager"]

    # Header Banner
    st.markdown(
        """
        <div style="margin-bottom: 1.5rem;">
            <div style="font-size: 1.8rem; font-weight: 800; color: #F8FAFC; letter-spacing: -0.02em;">
                💡 SHAP Explainability Studio & Model Interpretability
            </div>
            <div style="font-size: 0.95rem; color: #94A3B8; margin-top: 0.25rem;">
                Deconstruct model predictions using Game-Theoretic Shapley Additive exPlanations (SHAP). Uncover global feature drivers and local individual prediction waterfalls.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 1. Inspect Phase 7 Model Availability
    comp_res = st.session_state.get("ml_comparison_result")
    split_res = st.session_state.get("ml_active_split")
    prep_res = st.session_state.get("ml_active_prep")
    target_summary = st.session_state.get("ml_target_summary")

    # If no model trained in Phase 7, show friendly setup instructions and 1-click fallback
    if comp_res is None or split_res is None or prep_res is None:
        st.markdown(
            """
            <div style="background: rgba(79, 70, 229, 0.1); border: 1px solid rgba(99, 102, 241, 0.3); border-radius: 8px; padding: 1.2rem; margin-bottom: 1.5rem;">
                <h4 style="color: #A5B4FC; margin-top: 0;">⚡ No Trained Phase 7 Model Found in Session</h4>
                <p style="color: #CBD5E1; font-size: 0.9rem;">
                    SHAP calculates Shapley values for <em>actual trained machine learning models</em>. To explain model predictions:
                </p>
                <ol style="color: #94A3B8; font-size: 0.88rem; line-height: 1.6;">
                    <li>Navigate to <strong>"🤖 Machine Learning Models"</strong> in the sidebar.</li>
                    <li>Select features, partition data, and benchmark models.</li>
                    <li>Return here to inspect global importance, beeswarm distributions, and individual prediction waterfalls.</li>
                </ol>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("##### Quick Start Option:")
        if st.button("🚀 Auto-Train Benchmark Model on Active Dataset Now", type="primary"):
            with st.spinner("Preparing sample data, preprocessing pipeline, and training Random Forest benchmark..."):
                if "uploaded_dataset" not in st.session_state or st.session_state["uploaded_dataset"] is None:
                    sample_df = generate_churn_investigation_dataset()
                    st.session_state["uploaded_dataset"] = sample_df
                    st.session_state["preprocessed_dataset"] = sample_df.copy()
                    st.session_state["dataset_filename"] = "telecom_customer_churn_sample.csv"
                    st.session_state["selected_target_variable"] = "churn"

                df_active = st.session_state["preprocessed_dataset"]
                target_col = st.session_state.get("selected_target_variable", "churn")
                ml_engine = MachineLearningEngine()
                st.session_state["ml_engine"] = ml_engine

                t_sum = ml_engine.inspect_target(df_active, target_col)
                st.session_state["ml_target_summary"] = t_sum

                sp_res, pr_res, y_tr, y_te, t_classes = ml_engine.prepare_and_split(
                    df=df_active,
                    target_col=target_col,
                    problem_type=t_sum.problem_type,
                )
                st.session_state["ml_active_split"] = sp_res
                st.session_state["ml_active_prep"] = pr_res
                st.session_state["ml_y_train_enc"] = y_tr
                st.session_state["ml_y_test_enc"] = y_te
                st.session_state["ml_target_classes"] = t_classes

                comp = ml_engine.benchmark_models(
                    selected_models=["random_forest", "logistic_regression" if "Class" in t_sum.problem_type.value else "ridge"],
                    split_res=sp_res,
                    prep_res=pr_res,
                    y_train_enc=y_tr,
                    y_test_enc=y_te,
                    problem_type=t_sum.problem_type,
                    target_classes=t_classes,
                )
                st.session_state["ml_comparison_result"] = comp
                st.success("✅ Benchmark model successfully trained! SHAP Explainer is ready.")
                st.rerun()
        return

    # 2. Studio Tabs
    tab_model, tab_global, tab_summary, tab_waterfall, tab_dep, tab_insights, tab_export = st.tabs([
        "🧭 1. Model & Explainer",
        "🌐 2. Global Importance",
        "🐝 3. SHAP Summary (Beeswarm)",
        "🌊 4. Individual Waterfall & Force",
        "📈 5. Dependence & Interactions",
        "📝 6. Narrative Insights",
        "📥 7. Export Data",
    ])

    # =========================================================================
    # TAB 1: Model & Explainer Setup
    # =========================================================================
    with tab_model:
        st.subheader("Select Model & Initialize Explainer")
        st.markdown(
            "Select which trained model from Phase 7 to explain. The engine validates model architecture "
            "and automatically matches it with the mathematically optimal SHAP explainer."
        )

        model_keys = list(comp_res.results_by_model.keys())
        default_model = comp_res.best_model_key if comp_res.best_model_key in model_keys else model_keys[0]

        c_m1, c_m2, c_m3 = st.columns([1.5, 1.2, 1.3])
        with c_m1:
            selected_model_key = st.selectbox(
                "Trained Model to Explain",
                options=model_keys,
                index=model_keys.index(default_model),
                format_func=lambda k: comp_res.results_by_model[k]["display_name"],
                key="shap_model_selector",
            )
        
        model_obj = comp_res.results_by_model[selected_model_key]
        estimator = model_obj["estimator"]

        # Classification target class selector
        prob_type = comp_res.problem_type
        is_class = prob_type in (ProblemType.BINARY_CLASSIFICATION, ProblemType.MULTICLASS_CLASSIFICATION)
        target_classes = st.session_state.get("ml_target_classes") or ["0", "1"]

        with c_m2:
            if is_class:
                class_options = list(range(len(target_classes)))
                target_class_idx = st.selectbox(
                    "Target Outcome Class to Explain",
                    options=class_options,
                    index=min(1, len(class_options) - 1),
                    format_func=lambda idx: f"Class {idx}: '{target_classes[idx]}'",
                    key="shap_class_idx_sel",
                )
            else:
                target_class_idx = 0
                st.text_input("Output Space", value="Continuous Target", disabled=True)

        with c_m3:
            max_eval_samples = st.slider(
                "Explanation Samples (Holdout)",
                min_value=20,
                max_value=min(500, split_res.n_test),
                value=min(150, split_res.n_test),
                step=10,
                help="Limits number of holdout instances to explain for responsive interactive plotting.",
            )

        # Build explainer and compute SHAP values
        wrapper = manager.build_explainer_wrapper(
            estimator=estimator,
            feature_names=prep_res.feature_names_out,
            X_train_trans=prep_res.preprocessor.transform(split_res.X_train),
            is_classification=is_class,
            class_labels=target_classes if is_class else None,
            model_id=selected_model_key,
        )

        st.markdown(
            f"""
            <div style="background: rgba(30, 41, 59, 0.7); border: 1px solid #334155; border-radius: 8px; padding: 0.9rem; margin: 1rem 0;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="font-weight: 700; color: #38BDF8; font-size: 0.95rem;">⚙️ Active Explainer: {wrapper.metadata.explainer_type}</span>
                    <span style="font-size: 0.75rem; background: rgba(56, 189, 248, 0.15); color: #38BDF8; padding: 2px 8px; border-radius: 4px;">
                        {'Exact O(TLD^2) Calculation' if not wrapper.metadata.is_approximate else 'Sampling Approximation'}
                    </span>
                </div>
                <div style="font-size: 0.85rem; color: #CBD5E1; margin-top: 0.35rem;">{wrapper.metadata.selection_reason}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if st.button("⚡ Calculate SHAP Attributions", type="primary"):
            with st.spinner(f"Computing Shapley values with {wrapper.metadata.explainer_type}..."):
                try:
                    X_test_trans = prep_res.preprocessor.transform(split_res.X_test)
                    cache_key = f"{selected_model_key}_{target_class_idx}_{max_eval_samples}"
                    calc_res = manager.compute_explanations(
                        wrapper=wrapper,
                        X_eval_trans=X_test_trans,
                        target_class_idx=target_class_idx,
                        max_eval_samples=max_eval_samples,
                        cache_key=cache_key,
                    )
                    st.session_state["shap_active_calc"] = calc_res
                    st.session_state["shap_active_global"] = manager.extract_global_importance(calc_res)
                    st.success(f"✅ Computed SHAP values for {calc_res.n_samples} observations across {calc_res.n_features} features.")
                except Exception as ex:
                    st.error(f"SHAP calculation error: {str(ex)}")

        calc_res: Optional[SHAPCalculationResult] = st.session_state.get("shap_active_calc")
        if calc_res is not None:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Explained Samples", f"{calc_res.n_samples:,}")
            c2.metric("Input Features", f"{calc_res.n_features}")
            c3.metric("Population Base E[f(X)]", f"{calc_res.base_value:.4f}")
            c4.metric("Target Output", calc_res.target_class_name or "Continuous")

    calc_res = st.session_state.get("shap_active_calc")
    global_res: Optional[GlobalSHAPResult] = st.session_state.get("shap_active_global")

    if calc_res is None or global_res is None:
        st.info("ℹ️ Click '⚡ Calculate SHAP Attributions' in Tab 1 to compute explainability metrics.")
        return

    # =========================================================================
    # TAB 2: Global Feature Importance
    # =========================================================================
    with tab_global:
        st.subheader("Population-Level SHAP Feature Importance")
        st.markdown(
            "Global SHAP feature importance calculates the **mean absolute Shapley value** across all evaluated records. "
            "It measures the overall predictive magnitude with which each feature influences the model."
        )

        st.info(f"💡 {global_res.summary_text}")

        c_g1, c_g2 = st.columns([2, 1])
        with c_g2:
            top_k_slider = st.slider("Display Top Features", min_value=5, max_value=min(25, calc_res.n_features), value=min(12, calc_res.n_features))

        with c_g1:
            fig_bar = plot_shap_summary_bar(global_res.importance_df, max_features=top_k_slider)
            st.plotly_chart(fig_bar, use_container_width=True)

        st.markdown("##### Detailed Global Attribution Table")
        st.dataframe(global_res.importance_df.head(top_k_slider), use_container_width=True, hide_index=True)

    # =========================================================================
    # TAB 3: SHAP Summary (Beeswarm)
    # =========================================================================
    with tab_summary:
        st.subheader("SHAP Summary Beeswarm Distribution")
        st.markdown(
            "The summary beeswarm plot reveals both **feature importance** and **directional impact**. "
            "Each point represents an individual observation. Points to the right push the prediction higher; "
            "points to the left pull the prediction lower. Point colors represent feature values (Blue = Low, Red = High)."
        )

        fig_beeswarm = plot_shap_beeswarm(
            shap_values=calc_res.shap_values,
            X_df=calc_res.X_explained,
            max_features=12,
        )
        st.plotly_chart(fig_beeswarm, use_container_width=True)

    # =========================================================================
    # TAB 4: Individual Waterfall & Force Plot
    # =========================================================================
    with tab_waterfall:
        st.subheader("Local Prediction Waterfall & Force Decomposition")
        st.markdown(
            "Inspect exactly why a specific individual observation received its prediction. "
            "The waterfall chart begins at the baseline expected value $E[f(X)]$ and shows how each feature "
            "step-by-step pushes the prediction up (red) or down (cyan) to reach final model output $f(x)$."
        )

        c_w1, c_w2 = st.columns([1.5, 1])
        with c_w1:
            obs_idx = st.number_input(
                "Select Record Index to Explain",
                min_value=0,
                max_value=calc_res.n_samples - 1,
                value=st.session_state.get("shap_selected_obs_idx", 0),
                step=1,
            )
            st.session_state["shap_selected_obs_idx"] = int(obs_idx)

        # Generate local explanation
        y_test_enc = st.session_state.get("ml_y_test_enc")
        local_res = manager.extract_local_explanation(
            calc_result=calc_res,
            sample_index=int(obs_idx),
            actual_outcomes=y_test_enc,
        )
        st.session_state["shap_active_local"] = local_res

        with c_w2:
            audit = local_res.additivity_audit
            badge_color = "#10B981" if audit["is_additive"] else "#F59E0B"
            st.markdown(
                f"""
                <div style="background: rgba(15, 23, 42, 0.6); border: 1px solid #334155; border-radius: 8px; padding: 0.8rem; margin-top: 1.4rem;">
                    <div style="font-size: 0.72rem; color: #94A3B8; text-transform: uppercase;">Shapley Efficiency Axiom</div>
                    <div style="font-size: 0.95rem; font-weight: 700; color: {badge_color}; margin-top: 2px;">
                        {'✅ Additivity Verified' if audit['is_additive'] else '⚠️ Numerical Approximation'}
                    </div>
                    <div style="font-size: 0.75rem; color: #94A3B8;">Base ({audit['base_value']:.3f}) + Sum(SHAP) = {audit['expected_output']:.3f} (f(x) = {audit['actual_model_output']:.3f})</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.info(f"📝 **Record Narrative:** {local_res.narrative_summary}")

        fig_waterfall = plot_shap_waterfall(
            base_value=local_res.base_value,
            model_output=local_res.model_output,
            contributions_df=local_res.contributions_df,
            max_display=10,
        )
        st.plotly_chart(fig_waterfall, use_container_width=True)

        st.markdown("##### SHAP Force Balance")
        fig_force = plot_shap_force_bar(
            base_value=local_res.base_value,
            model_output=local_res.model_output,
            contributions_df=local_res.contributions_df,
        )
        st.plotly_chart(fig_force, use_container_width=True)

        st.markdown("##### Record Feature Attributions Table")
        st.dataframe(local_res.contributions_df.head(15), use_container_width=True, hide_index=True)

    # =========================================================================
    # TAB 5: Feature Dependence & Interactions
    # =========================================================================
    with tab_dep:
        st.subheader("Feature Dependence Curves & Non-Linear Responses")
        st.markdown(
            "Analyze how a feature's SHAP contribution changes across its observed values. "
            "Coloring by an interaction feature highlights multivariate interaction effects."
        )

        c_d1, c_d2 = st.columns(2)
        with c_d1:
            primary_feat = st.selectbox(
                "Primary Feature to Inspect",
                options=calc_res.feature_names,
                index=0,
                key="shap_dep_primary",
            )
        with c_d2:
            inter_feat = st.selectbox(
                "Interacting Feature for Color (Optional)",
                options=["Auto-Detect Strongest"] + [f for f in calc_res.feature_names if f != primary_feat],
                index=0,
                key="shap_dep_inter",
            )
            inter_arg = None if "Auto" in inter_feat else inter_feat

        dep_data = manager.extract_dependence_profile(
            calc_result=calc_res,
            primary_feature=primary_feat,
            interaction_feature=inter_arg,
        )

        st.info(f"💡 {dep_data.description}")

        fig_dep = plot_shap_dependence(
            primary_feature=dep_data.primary_feature,
            feature_values=dep_data.feature_values,
            shap_values=dep_data.shap_values,
            interaction_feature=dep_data.interaction_feature,
            interaction_values=dep_data.interaction_values,
        )
        st.plotly_chart(fig_dep, use_container_width=True)

        # Interaction Analysis Section
        st.markdown("---")
        st.subheader("Pairwise SHAP Interaction Matrix")
        if wrapper.metadata.supports_interactions:
            if st.button("🔍 Compute Top Pairwise Interactions", type="secondary"):
                with st.spinner("Calculating pairwise interaction matrix..."):
                    inter_res = manager.extract_interactions(wrapper=wrapper, calc_result=calc_res)
                    st.session_state["shap_active_interactions"] = inter_res

            inter_res = st.session_state.get("shap_active_interactions")
            if inter_res is not None and inter_res.is_supported:
                st.success(inter_res.message)
                if inter_res.interaction_matrix is not None:
                    fig_inter = plot_interaction_matrix_heatmap(inter_res.interaction_matrix)
                    st.plotly_chart(fig_inter, use_container_width=True)
                if inter_res.top_interacting_pairs:
                    st.markdown("##### Top Interacting Feature Pairs")
                    st.dataframe(pd.DataFrame(inter_res.top_interacting_pairs), use_container_width=True)
        else:
            st.info(f"ℹ️ Pairwise interaction matrices are supported for TreeExplainer. Active model uses {wrapper.metadata.explainer_type}.")

    # =========================================================================
    # TAB 6: Narrative Insights
    # =========================================================================
    with tab_insights:
        st.subheader("Automated Data-Driven SHAP Narrative Insights")
        local_res = st.session_state.get("shap_active_local")
        insights = manager.generate_narrative(global_result=global_res, local_result=local_res)

        badge_colors = {
            "success": ("#10B981", "rgba(16, 185, 129, 0.1)"),
            "warning": ("#F59E0B", "rgba(245, 158, 11, 0.1)"),
            "info": ("#38BDF8", "rgba(56, 189, 248, 0.1)"),
            "purple": ("#A855F7", "rgba(168, 85, 247, 0.1)"),
        }

        for ins in insights:
            border_col, bg_col = badge_colors.get(ins.badge_type, ("#94A3B8", "rgba(148, 163, 184, 0.1)"))
            st.markdown(
                f"""
                <div style="background: {bg_col}; border-left: 4px solid {border_col}; border-radius: 4px 8px 8px 4px; padding: 0.9rem 1.2rem; margin-bottom: 0.85rem;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div style="font-weight: 700; color: #F8FAFC; font-size: 0.95rem;">{ins.title}</div>
                        <span style="font-size: 0.72rem; color: {border_col}; text-transform: uppercase; font-weight: 700;">{ins.category}</span>
                    </div>
                    <div style="color: #CBD5E1; font-size: 0.88rem; margin-top: 0.35rem; line-height: 1.5;">{ins.description}</div>
                    <div style="color: #94A3B8; font-size: 0.78rem; margin-top: 0.4rem;">
                        <strong>Supporting Evidence:</strong> {ins.supporting_evidence}<br>
                        <strong>Caveats:</strong> {ins.caveats}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # =========================================================================
    # TAB 7: Export Data
    # =========================================================================
    with tab_export:
        st.subheader("Export SHAP Values & Attributions")
        st.markdown("Download full observation-level SHAP matrices, global rankings, and local attributions as CSV.")

        # 1. Full SHAP Matrix
        df_full_shap = pd.DataFrame(calc_res.shap_values, columns=[f"SHAP_{c}" for c in calc_res.feature_names])
        df_full_export = pd.concat([calc_res.X_explained, df_full_shap], axis=1)

        c_exp1, c_exp2 = st.columns(2)
        with c_exp1:
            st.download_button(
                label="📥 Download Full Observation SHAP Matrix (CSV)",
                data=df_full_export.to_csv(index=False),
                file_name=f"shap_matrix_{comp_res.best_model_key}.csv",
                mime="text/csv",
                type="primary",
            )
        with c_exp2:
            st.download_button(
                label="📥 Download Global Feature Importance (CSV)",
                data=global_res.importance_df.to_csv(index=False),
                file_name=f"shap_global_importance_{comp_res.best_model_key}.csv",
                mime="text/csv",
            )

