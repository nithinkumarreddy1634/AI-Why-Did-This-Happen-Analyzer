"""Streamlit User Interface for Phase 9: LIME Explainability Studio & Local Interpretability."""

from typing import Optional, Dict, Any, List
import streamlit as st
import pandas as pd
import numpy as np

from modules.explainability.lime_manager import LIMEExplanationManager
from modules.explainability.lime_explainer import LIMEConfig, LIMEExplainerWrapper
from modules.explainability.lime_local_analysis import LIMEExplanationResult
from modules.explainability.lime_shap_comparison import SHAPLIMEComparisonResult
from modules.explainability.lime_batch_analysis import BatchLIMEResult
from modules.explainability.explanation_manager import SHAPExplanationManager
from modules.machine_learning.model_trainer import MachineLearningEngine
from modules.machine_learning.problem_detector import ProblemType
from components.lime_charts import (
    plot_lime_contribution_bar,
    plot_surrogate_fidelity_comparison,
    plot_shap_lime_comparison_bar,
    plot_batch_recurring_features_bar,
)
from utils.sample_data import generate_churn_investigation_dataset


def render_lime_studio_page() -> None:
    """Master rendering function for the LIME Explainability Studio."""
    # Session state initialization
    if "lime_manager" not in st.session_state:
        st.session_state["lime_manager"] = LIMEExplanationManager()
    if "lime_active_result" not in st.session_state:
        st.session_state["lime_active_result"] = None
    if "lime_active_comparison" not in st.session_state:
        st.session_state["lime_active_comparison"] = None
    if "lime_active_batch" not in st.session_state:
        st.session_state["lime_active_batch"] = None
    if "lime_selected_obs_idx" not in st.session_state:
        st.session_state["lime_selected_obs_idx"] = 0

    lime_mgr: LIMEExplanationManager = st.session_state["lime_manager"]

    # Header Banner
    st.markdown(
        """
        <div style="margin-bottom: 1.5rem;">
            <div style="font-size: 1.8rem; font-weight: 800; color: #F8FAFC; letter-spacing: -0.02em;">
                🍋 LIME Explainability Studio & Local Model Interpretation
            </div>
            <div style="font-size: 0.95rem; color: #94A3B8; margin-top: 0.25rem;">
                Explain individual black-box predictions by perturbing local instances and fitting interpretable surrogate models. Compare local LIME attributions directly against cooperative game-theoretic SHAP values.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 1. Inspect Phase 7 Model Availability
    comp_res = st.session_state.get("ml_comparison_result")
    split_res = st.session_state.get("ml_active_split")
    prep_res = st.session_state.get("ml_active_prep")

    # If no model trained in Phase 7, show friendly instructions & 1-click fallback
    if comp_res is None or split_res is None or prep_res is None:
        st.markdown(
            """
            <div style="background: rgba(16, 185, 129, 0.1); border: 1px solid rgba(16, 185, 129, 0.3); border-radius: 8px; padding: 1.2rem; margin-bottom: 1.5rem;">
                <h4 style="color: #34D399; margin-top: 0;">⚡ No Trained Model Found in Session</h4>
                <p style="color: #CBD5E1; font-size: 0.9rem;">
                    LIME generates local perturbations and queries a <em>trained model</em> to fit an interpretable surrogate.
                    Train a model in <strong>"🤖 Machine Learning Models"</strong> or click below to launch an instant benchmark.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if st.button("🚀 Auto-Train Benchmark Model on Active Dataset Now", type="primary", key="lime_autotrain_btn"):
            with st.spinner("Training benchmark model for LIME explanations..."):
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
                st.success("✅ Benchmark model ready! LIME Studio initialized.")
                st.rerun()
        return

    # 2. Studio Tabs
    tab_model, tab_local, tab_compare, tab_batch, tab_insights, tab_export = st.tabs([
        "🧭 1. Model & Setup",
        "🔍 2. Local LIME Explanation",
        "⚖️ 3. SHAP vs. LIME Comparison",
        "👥 4. Cohort Batch Analysis",
        "📝 5. Narrative Insights",
        "📥 6. Export Attributions",
    ])

    # =========================================================================
    # TAB 1: Model & Setup
    # =========================================================================
    with tab_model:
        st.subheader("Model Selection & LIME Configuration")
        st.markdown(
            "Select which trained model to interpret and configure the perturbation neighborhood settings."
        )

        model_keys = list(comp_res.results_by_model.keys())
        default_model = comp_res.best_model_key if comp_res.best_model_key in model_keys else model_keys[0]

        c_m1, c_m2, c_m3 = st.columns(3)
        with c_m1:
            selected_model_key = st.selectbox(
                "Trained Model to Explain",
                options=model_keys,
                index=model_keys.index(default_model),
                format_func=lambda k: comp_res.results_by_model[k]["display_name"],
                key="lime_model_selector",
            )
        
        model_data = comp_res.results_by_model[selected_model_key]
        estimator = model_data["estimator"]

        prob_type = comp_res.problem_type
        is_class = prob_type in (ProblemType.BINARY_CLASSIFICATION, ProblemType.MULTICLASS_CLASSIFICATION)
        target_classes = st.session_state.get("ml_target_classes") or ["0", "1"]

        with c_m2:
            num_samples = st.slider(
                "LIME Perturbation Samples",
                min_value=200,
                max_value=3000,
                value=1000,
                step=100,
                help="Higher sample counts produce more stable surrogate fits but take longer to evaluate.",
            )
        with c_m3:
            num_features = st.slider(
                "Number of Explanation Features",
                min_value=3,
                max_value=min(20, len(prep_res.feature_names_out)),
                value=min(8, len(prep_res.feature_names_out)),
                step=1,
            )

        c_c1, c_c2 = st.columns(2)
        with c_c1:
            discretizer_choice = st.selectbox(
                "Continuous Discretization Strategy",
                options=["quartile", "decile", "entropy"],
                index=0,
                help="Method used to bin continuous features into human-readable rule conditions.",
            )
        with c_c2:
            random_seed = st.number_input("Reproducibility Random Seed", value=42, step=1)

        lime_cfg = LIMEConfig(
            num_samples=num_samples,
            num_features=num_features,
            discretize_continuous=True,
            discretizer=discretizer_choice,
            random_state=int(random_seed),
        )

        wrapper = lime_mgr.build_explainer_wrapper(
            estimator=estimator,
            feature_names=prep_res.feature_names_out,
            X_train_trans=prep_res.preprocessor.transform(split_res.X_train),
            is_classification=is_class,
            class_names=target_classes if is_class else None,
            model_id=selected_model_key,
            config=lime_cfg,
        )

        st.markdown(
            f"""
            <div style="background: rgba(30, 41, 59, 0.7); border: 1px solid #334155; border-radius: 8px; padding: 0.9rem; margin: 1rem 0;">
                <div style="font-weight: 700; color: #34D399; font-size: 0.95rem;">🍋 Active LIME Explainer Setup</div>
                <div style="font-size: 0.85rem; color: #CBD5E1; margin-top: 0.35rem;">
                    • <strong>Mode:</strong> {wrapper.metadata.mode.capitalize()}<br>
                    • <strong>Background Distribution:</strong> {wrapper.metadata.n_background_samples:,} training records<br>
                    • <strong>Feature Dimensions:</strong> {wrapper.metadata.n_features} encoded features<br>
                    • <strong>Perturbation Kernel:</strong> Exponential Euclidean distance kernel with {num_samples} local samples
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # =========================================================================
    # TAB 2: Local LIME Explanation
    # =========================================================================
    with tab_local:
        st.subheader("Individual Prediction Explanation & Surrogate Fidelity")
        st.markdown(
            "Select an observation from the holdout dataset to generate its local LIME explanation. "
            "LIME perturbs the record's features, observes the black-box model's responses, and fits a "
            "weighted ridge regression surrogate model in the local neighborhood."
        )

        X_test_trans = prep_res.preprocessor.transform(split_res.X_test)
        n_test = len(split_res.X_test)

        c_o1, c_o2 = st.columns([1.5, 1])
        with c_o1:
            obs_idx = st.number_input(
                "Select Record Index to Explain",
                min_value=0,
                max_value=n_test - 1,
                value=st.session_state.get("lime_selected_obs_idx", 0),
                step=1,
                key="lime_obs_num_input",
            )
            st.session_state["lime_selected_obs_idx"] = int(obs_idx)

        with c_o2:
            if is_class:
                class_options = list(range(len(target_classes)))
                target_class_idx = st.selectbox(
                    "Target Class to Explain",
                    options=class_options,
                    index=min(1, len(class_options) - 1),
                    format_func=lambda idx: f"Class {idx}: '{target_classes[idx]}'",
                    key="lime_class_idx_sel",
                )
            else:
                target_class_idx = 0
                st.text_input("Mode", value="Regression (Continuous)", disabled=True)

        if st.button("🍋 Generate LIME Explanation for Record", type="primary"):
            with st.spinner("Generating local perturbations and fitting surrogate model..."):
                try:
                    y_test_enc = st.session_state.get("ml_y_test_enc")
                    actual_val = y_test_enc[obs_idx] if y_test_enc is not None and len(y_test_enc) > obs_idx else None
                    cache_key = f"{selected_model_key}_{obs_idx}_{target_class_idx}_{num_samples}_{num_features}"
                    
                    lime_res = lime_mgr.explain_instance(
                        wrapper=wrapper,
                        instance_row=X_test_trans[obs_idx],
                        sample_index=int(obs_idx),
                        target_class_idx=target_class_idx,
                        num_features=num_features,
                        num_samples=num_samples,
                        actual_outcome=actual_val,
                        cache_key=cache_key,
                    )
                    st.session_state["lime_active_result"] = lime_res
                    st.success(f"✅ LIME explanation generated with surrogate fidelity R² = {lime_res.local_fidelity_r2:.3f}")
                except Exception as ex:
                    st.error(f"LIME calculation error: {str(ex)}")

        lime_res: Optional[LIMEExplanationResult] = st.session_state.get("lime_active_result")
        if lime_res is not None and lime_res.sample_index == obs_idx:
            # Metrics Row
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Predicted Outcome", str(lime_res.predicted_outcome))
            m2.metric("Local Surrogate Output", f"{lime_res.local_surrogate_prediction:.4f}")
            m3.metric("Surrogate Fidelity (R²)", f"{lime_res.local_fidelity_r2:.3f}")
            m4.metric("Approximation Gap", f"{lime_res.fidelity_gap:.4f}")

            st.info(f"📝 {lime_res.narrative_summary}")

            c_p1, c_p2 = st.columns([1.6, 1])
            with c_p1:
                fig_bar = plot_lime_contribution_bar(lime_res.contributions_df)
                st.plotly_chart(fig_bar, use_container_width=True)

            with c_p2:
                orig_score = float(lime_res.predicted_probabilities[lime_res.target_class_idx]) if lime_res.predicted_probabilities else float(lime_res.local_surrogate_prediction)
                fig_fid = plot_surrogate_fidelity_comparison(
                    original_pred=orig_score,
                    surrogate_pred=lime_res.local_surrogate_prediction,
                    fidelity_r2=lime_res.local_fidelity_r2,
                )
                st.plotly_chart(fig_fid, use_container_width=True)

            st.markdown("##### Local Rule Attribution Matrix")
            st.dataframe(lime_res.contributions_df, use_container_width=True, hide_index=True)

    # =========================================================================
    # TAB 3: SHAP vs. LIME Comparison
    # =========================================================================
    with tab_compare:
        st.subheader("Multi-Method Explainability Comparison: SHAP vs. LIME")
        st.markdown(
            "Benchmark cooperative game-theoretic attribution (**SHAP**) against local surrogate perturbation (**LIME**) "
            "for the identical observation. Features prioritized by both mathematical paradigms represent high-confidence predictive drivers."
        )

        lime_res = st.session_state.get("lime_active_result")

        if lime_res is None:
            st.info("ℹ️ Generate a LIME explanation in Tab 2 before running the SHAP vs. LIME comparison.")
        else:
            # Check if SHAP calculation exists for this instance
            shap_mgr: SHAPExplanationManager = st.session_state.get("shap_manager", SHAPExplanationManager())
            calc_res = st.session_state.get("shap_active_calc")

            if st.button("⚖️ Compare SHAP and LIME for Active Record", type="primary"):
                with st.spinner("Aligning SHAP and LIME attributions..."):
                    try:
                        # If SHAP not yet run, calculate on the fly for this model
                        if calc_res is None or calc_res.target_class_idx != lime_res.target_class_idx:
                            shap_wrapper = shap_mgr.build_explainer_wrapper(
                                estimator=estimator,
                                feature_names=prep_res.feature_names_out,
                                X_train_trans=prep_res.preprocessor.transform(split_res.X_train),
                                is_classification=is_class,
                                class_labels=target_classes if is_class else None,
                                model_id=selected_model_key,
                            )
                            calc_res = shap_mgr.compute_explanations(
                                wrapper=shap_wrapper,
                                X_eval_trans=X_test_trans,
                                target_class_idx=lime_res.target_class_idx,
                                max_eval_samples=min(100, len(X_test_trans)),
                            )
                            st.session_state["shap_active_calc"] = calc_res

                        # Extract local SHAP explanation for the same observation index
                        local_shap = shap_mgr.extract_local_explanation(
                            calc_result=calc_res,
                            sample_index=lime_res.sample_index,
                        )

                        comp_result = lime_mgr.compare_with_shap(
                            shap_result=local_shap,
                            lime_result=lime_res,
                            top_k=8,
                        )
                        st.session_state["lime_active_comparison"] = comp_result
                        st.success(f"✅ Comparison complete: {comp_result.concordance_level}")
                    except Exception as ex:
                        st.error(f"Comparison error: {str(ex)}")

            comp_res_obj: Optional[SHAPLIMEComparisonResult] = st.session_state.get("lime_active_comparison")
            if comp_res_obj is not None:
                # Metric Cards
                k1, k2, k3, k4 = st.columns(4)
                k1.metric("Concordance Level", comp_res_obj.concordance_level)
                k2.metric("Common Top Features", f"{len(comp_res_obj.common_top_features)} features")
                k3.metric("Jaccard Overlap", f"{comp_res_obj.jaccard_similarity:.1%}")
                k4.metric("Directional Agreement", f"{comp_res_obj.direction_agreement_pct:.0f}%")

                st.info(f"💡 {comp_res_obj.methodological_summary}")

                # Comparison grouped chart
                fig_comp_bar = plot_shap_lime_comparison_bar(comp_res_obj.comparison_df)
                st.plotly_chart(fig_comp_bar, use_container_width=True)

                st.markdown("##### Detailed Attribution Alignment Table")
                st.dataframe(comp_res_obj.comparison_df, use_container_width=True, hide_index=True)

    # =========================================================================
    # TAB 4: Cohort Batch Analysis
    # =========================================================================
    with tab_batch:
        st.subheader("Cohort Batch LIME Explanations")
        st.markdown(
            "Explain multiple selected holdout observations to identify recurring local patterns and rules."
        )

        available_indices = list(range(min(20, n_test)))
        selected_batch_indices = st.multiselect(
            "Select Observation Indices to Explain in Batch",
            options=available_indices,
            default=available_indices[:5],
            help="Select up to 10 instances for cohort pattern analysis.",
        )

        if st.button("👥 Run Cohort Batch LIME Explanations", type="secondary"):
            if not selected_batch_indices:
                st.error("Please select at least one observation index.")
            else:
                with st.spinner(f"Computing LIME explanations across {len(selected_batch_indices)} instances..."):
                    try:
                        y_test_enc = st.session_state.get("ml_y_test_enc")
                        batch_res = lime_mgr.explain_batch(
                            wrapper=wrapper,
                            X_eval_df=pd.DataFrame(X_test_trans, columns=prep_res.feature_names_out),
                            sample_indices=selected_batch_indices,
                            target_class_idx=target_class_idx,
                            num_features=6,
                            num_samples=500,
                            actual_outcomes=y_test_enc,
                        )
                        st.session_state["lime_active_batch"] = batch_res
                        st.success(f"✅ Batch completed: {batch_res.n_explained} explanations evaluated with mean R² = {batch_res.mean_fidelity_r2:.3f}")
                    except Exception as ex:
                        st.error(f"Batch evaluation error: {str(ex)}")

        batch_res: Optional[BatchLIMEResult] = st.session_state.get("lime_active_batch")
        if batch_res is not None:
            st.markdown(f"##### Recurring Predictive Drivers (Mean R² = {batch_res.mean_fidelity_r2:.3f})")
            fig_batch = plot_batch_recurring_features_bar(batch_res.recurring_top_features)
            st.plotly_chart(fig_batch, use_container_width=True)

            st.markdown("##### Batch Summary Table")
            st.dataframe(batch_res.summary_table, use_container_width=True, hide_index=True)

    # =========================================================================
    # TAB 5: Narrative Insights
    # =========================================================================
    with tab_insights:
        st.subheader("Automated Data-Driven LIME Insights")
        lime_res = st.session_state.get("lime_active_result")

        if lime_res is None:
            st.info("ℹ️ Generate a LIME explanation in Tab 2 to view automated narrative insights.")
        else:
            comp_obj = st.session_state.get("lime_active_comparison")
            batch_obj = st.session_state.get("lime_active_batch")
            insights = lime_mgr.generate_narrative(
                lime_result=lime_res,
                comparison_result=comp_obj,
                batch_result=batch_obj,
            )

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
    # TAB 6: Export Attributions
    # =========================================================================
    with tab_export:
        st.subheader("Export LIME Attributions & Comparison Data")
        lime_res = st.session_state.get("lime_active_result")
        comp_obj = st.session_state.get("lime_active_comparison")
        batch_obj = st.session_state.get("lime_active_batch")

        c_e1, c_e2, c_e3 = st.columns(3)
        with c_e1:
            if lime_res is not None:
                st.download_button(
                    label="📥 Download Local Explanation (CSV)",
                    data=lime_res.contributions_df.to_csv(index=False),
                    file_name=f"lime_explanation_record_{lime_res.sample_index}.csv",
                    mime="text/csv",
                    type="primary",
                )
            else:
                st.text("No active local explanation")

        with c_e2:
            if comp_obj is not None:
                st.download_button(
                    label="📥 Download SHAP vs LIME Comparison (CSV)",
                    data=comp_obj.comparison_df.to_csv(index=False),
                    file_name=f"shap_vs_lime_record_{comp_obj.sample_index}.csv",
                    mime="text/csv",
                )
            else:
                st.text("No comparison run yet")

        with c_e3:
            if batch_obj is not None:
                st.download_button(
                    label="📥 Download Cohort Batch Summary (CSV)",
                    data=batch_obj.to_csv(),
                    file_name="lime_cohort_batch_summary.csv",
                    mime="text/csv",
                )
            else:
                st.text("No batch run yet")

