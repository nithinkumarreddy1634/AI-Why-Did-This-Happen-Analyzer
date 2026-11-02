"""Streamlit User Interface for Phase 7: Machine Learning Studio & Predictive Analytics."""

from typing import Optional, Dict, Any, List
import streamlit as st
import pandas as pd
import numpy as np

from modules.machine_learning.model_trainer import MachineLearningEngine
from modules.machine_learning.problem_detector import ProblemType, TargetSummary
from modules.machine_learning.ml_preprocessing import MLPreprocessorConfig
from modules.machine_learning.data_splitter import SplitConfig
from modules.machine_learning.classification import CLASSIFICATION_MODELS_REGISTRY
from modules.machine_learning.regression import REGRESSION_MODELS_REGISTRY
from components.ml_visualizations import (
    plot_confusion_matrix,
    plot_roc_curve,
    plot_pr_curve,
    plot_actual_vs_predicted,
    plot_residual_diagnostics,
    plot_feature_importance_bars,
    plot_model_comparison_bar,
)
from utils.sample_data import generate_churn_investigation_dataset


def render_ml_studio_page() -> None:
    """Master rendering function for the Machine Learning Studio."""
    # Initialize Engine in session state
    if "ml_engine" not in st.session_state:
        st.session_state["ml_engine"] = MachineLearningEngine()
    if "ml_active_split" not in st.session_state:
        st.session_state["ml_active_split"] = None
    if "ml_active_prep" not in st.session_state:
        st.session_state["ml_active_prep"] = None
    if "ml_comparison_result" not in st.session_state:
        st.session_state["ml_comparison_result"] = None
    if "ml_active_importance" not in st.session_state:
        st.session_state["ml_active_importance"] = None
    if "ml_target_summary" not in st.session_state:
        st.session_state["ml_target_summary"] = None

    engine: MachineLearningEngine = st.session_state["ml_engine"]

    # Header Banner
    st.markdown(
        """
        <div style="margin-bottom: 1.5rem;">
            <div style="font-size: 1.8rem; font-weight: 800; color: #F8FAFC; letter-spacing: -0.02em;">
                🤖 Machine Learning Studio & Predictive Analytics
            </div>
            <div style="font-size: 0.95rem; color: #94A3B8; margin-top: 0.25rem;">
                Train, benchmark, diagnose, and interpret predictive models to extract non-linear patterns behind unexpected outcomes.
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

    col_meta1, col_meta2 = st.columns([2, 1])

    with col_meta1:
        if dataset_options:
            selected_ds_name = st.selectbox(
                "Select Dataset for Modeling",
                options=dataset_options,
                index=0,
                key="ml_dataset_selector",
                help="Switch between cleaned dataset or original raw upload.",
            )
            df = df_cleaned if "Cleaned" in selected_ds_name else (df_original if df_original is not None else st.session_state.get("current_dataset"))
        else:
            df = None

    with col_meta2:
        if df is not None:
            target_name = st.session_state.get("selected_target_variable") or "Unassigned"
            st.markdown(
                f"""
                <div style="background: rgba(30, 41, 59, 0.7); border: 1px solid #334155; border-radius: 8px; padding: 0.6rem 1rem; margin-top: 1.4rem;">
                    <div style="font-size: 0.75rem; color: #94A3B8;">Dataset Status</div>
                    <div style="font-size: 0.88rem; color: #38BDF8; font-weight: 600;">{df.shape[0]:,} rows × {df.shape[1]} cols | 🎯 Target: {target_name}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # 2. Guard for empty dataset
    if df is None:
        st.warning("⚠️ No dataset currently loaded. Please upload a dataset in 'Dataset Overview' or load our sample investigation dataset.")
        if st.button("📥 Load Sample Customer Churn Investigation Dataset", type="primary"):
            sample_df = generate_churn_investigation_dataset()
            st.session_state["uploaded_dataset"] = sample_df
            st.session_state["original_dataset"] = sample_df.copy()
            st.session_state["preprocessed_dataset"] = sample_df.copy()
            st.session_state["dataset_filename"] = "telecom_customer_churn_sample.csv"
            st.session_state["selected_target_variable"] = "churn"
            st.rerun()
        return

    # 3. Studio Navigation Tabs
    tab_overview, tab_prep, tab_train, tab_eval, tab_importance, tab_tuning, tab_predict, tab_insights = st.tabs([
        "🧭 1. Problem & Target",
        "⚙️ 2. Features & Split",
        "🚀 3. Benchmark Training",
        "📊 4. Model Evaluation",
        "💡 5. Feature Importance",
        "🔧 6. Hyperparameter Tuning",
        "🔍 7. Prediction Explorer",
        "📝 8. Narrative Insights",
    ])

    # =========================================================================
    # TAB 1: Problem & Target
    # =========================================================================
    with tab_overview:
        st.subheader("Target Variable & Objective Formulation")
        st.markdown(
            "Select the outcome variable you wish to investigate. The engine automatically inspects cardinality, "
            "data types, and class balance to recommend the optimal machine learning problem type."
        )

        all_cols = list(df.columns)
        saved_target = st.session_state.get("selected_target_variable")
        target_idx = all_cols.index(saved_target) if saved_target in all_cols else 0

        c_t1, c_t2, c_t3 = st.columns([1.5, 1.2, 1.3])
        with c_t1:
            selected_target = st.selectbox(
                "Target Variable (Outcome)",
                options=all_cols,
                index=target_idx,
                key="ml_target_select",
            )
            st.session_state["selected_target_variable"] = selected_target

        with c_t2:
            problem_override = st.selectbox(
                "Problem Type",
                options=["Auto-Detect (Recommended)", "Binary Classification", "Multiclass Classification", "Regression"],
                key="ml_prob_type_override",
            )
            force_type = None if "Auto" in problem_override else problem_override

        # Detect problem type
        target_summary: TargetSummary = engine.inspect_target(df, selected_target, force_type=force_type)
        st.session_state["ml_target_summary"] = target_summary

        with c_t3:
            badge_color = "#10B981" if target_summary.is_valid_for_ml else "#EF4444"
            st.markdown(
                f"""
                <div style="background: rgba(15, 23, 42, 0.6); border: 1px solid #334155; border-radius: 8px; padding: 0.8rem; margin-top: 1.5rem;">
                    <div style="font-size: 0.72rem; color: #94A3B8; text-transform: uppercase;">Problem Formulation</div>
                    <div style="font-size: 0.95rem; font-weight: 700; color: {badge_color}; margin-top: 2px;">
                        {target_summary.problem_type.value}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        if not target_summary.is_valid_for_ml:
            st.error(f"❌ Target Error: {target_summary.error_message}")
            return

        # Target Metrics Display
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Non-Null Samples", f"{target_summary.n_samples - target_summary.n_missing:,}")
        m2.metric("Missing Targets", f"{target_summary.n_missing:,}")
        m3.metric("Unique Outcomes", f"{target_summary.n_unique}")
        m4.metric("Imbalance Flag", "Yes (Imbalanced)" if target_summary.is_imbalanced else "No (Balanced)")

        st.info(f"💡 **Recommendation Rationale:** {target_summary.recommendation_reason}")

        if target_summary.class_distribution:
            st.markdown("##### Class Distribution Breakdown")
            df_dist = pd.DataFrame([
                {"Class Label": k, "Record Count": v, "Proportion": f"{target_summary.class_proportions[k]:.1%}"}
                for k, v in target_summary.class_distribution.items()
            ])
            st.dataframe(df_dist, use_container_width=True, hide_index=True)

    # =========================================================================
    # TAB 2: Features & Split
    # =========================================================================
    with tab_prep:
        st.subheader("Feature Engineering & Leakage-Free Data Splitting")
        st.markdown(
            "Configure preprocessing transformations. Transformations are **fitted strictly on training data** "
            "to prevent data leakage into holdout test partitions."
        )

        candidate_features = [c for c in df.columns if c != selected_target]

        c_f1, c_f2 = st.columns([1.5, 1])

        with c_f1:
            st.markdown("##### Feature Selection")
            exclude_ids = st.checkbox("Auto-Exclude Identifier & Key Columns", value=True)
            exclude_const = st.checkbox("Auto-Exclude Zero-Variance Constant Columns", value=True)
            
            selected_features = st.multiselect(
                "Predictor Features to Include in Models",
                options=candidate_features,
                default=candidate_features,
                help="Deselect any features you wish to omit from training.",
            )

        with c_f2:
            st.markdown("##### Preprocessing Configuration")
            num_imputer = st.selectbox("Numerical Imputation", ["median", "mean", "constant"], index=0)
            num_scaler = st.selectbox("Numerical Scaling", ["standard", "robust", "minmax", "none"], index=0)
            cat_imputer = st.selectbox("Categorical Imputation", ["most_frequent", "constant"], index=0)
            cat_encoder = st.selectbox("Categorical Encoding", ["onehot", "ordinal"], index=0)

        st.markdown("---")
        st.markdown("##### Train / Test Splitting & Validation")
        c_s1, c_s2, c_s3 = st.columns(3)
        with c_s1:
            test_size = st.slider("Holdout Test Size (%)", min_value=10, max_value=40, value=20, step=5)
        with c_s2:
            random_seed = st.number_input("Random Seed", value=42, step=1)
        with c_s3:
            cv_folds = st.slider("Cross-Validation Folds", min_value=2, max_value=10, value=5, step=1)

        prep_cfg = MLPreprocessorConfig(
            numeric_imputer_strategy=num_imputer,
            numeric_scaler=num_scaler,
            categorical_imputer_strategy=cat_imputer,
            categorical_encoder=cat_encoder,
            auto_exclude_identifiers=exclude_ids,
            auto_exclude_constants=exclude_const,
        )

        split_cfg = SplitConfig(
            test_size=test_size / 100.0,
            random_state=int(random_seed),
            stratify=True,
            cv_folds=cv_folds,
        )

        if st.button("⚙️ Prepare Preprocessing Pipeline & Partition Data", type="primary"):
            with st.spinner("Partitioning data and fitting preprocessing transformations on training set..."):
                try:
                    split_res, prep_res, y_tr_enc, y_te_enc, target_classes = engine.prepare_and_split(
                        df=df,
                        target_col=selected_target,
                        feature_columns=selected_features,
                        split_config=split_cfg,
                        preprocessor_config=prep_cfg,
                        problem_type=target_summary.problem_type,
                    )
                    st.session_state["ml_active_split"] = split_res
                    st.session_state["ml_active_prep"] = prep_res
                    st.session_state["ml_y_train_enc"] = y_tr_enc
                    st.session_state["ml_y_test_enc"] = y_te_enc
                    st.session_state["ml_target_classes"] = target_classes
                    st.success(f"✅ Data partitioned successfully: {split_res.n_train:,} training samples, {split_res.n_test:,} test samples.")
                except Exception as ex:
                    st.error(f"Partition Error: {str(ex)}")

        split_res = st.session_state.get("ml_active_split")
        prep_res = st.session_state.get("ml_active_prep")

        if split_res is not None and prep_res is not None:
            st.markdown(
                f"""
                <div style="background: rgba(16, 185, 129, 0.08); border: 1px solid rgba(16, 185, 129, 0.3); border-radius: 8px; padding: 0.85rem; margin-top: 1rem;">
                    <strong>Active Partition Audit:</strong><br>
                    • <strong>Strategy:</strong> {split_res.split_strategy}<br>
                    • <strong>Features (In):</strong> {len(prep_res.feature_names_in)} input columns ({len(prep_res.numeric_features)} numeric, {len(prep_res.categorical_features)} categorical)<br>
                    • <strong>Features (Out after one-hot encoding):</strong> {len(prep_res.feature_names_out)} encoded feature dimensions<br>
                    • <strong>Excluded Columns:</strong> {', '.join(prep_res.excluded_columns) if prep_res.excluded_columns else 'None'}
                </div>
                """,
                unsafe_allow_html=True,
            )

    # =========================================================================
    # TAB 3: Benchmark Training
    # =========================================================================
    with tab_train:
        st.subheader("Model Training & Multi-Algorithm Benchmark")
        st.markdown(
            "Select algorithms to train and compare side-by-side. Models are trained with cross-validation on "
            "training data and benchmarked on the identical holdout test partition."
        )

        split_res = st.session_state.get("ml_active_split")
        prep_res = st.session_state.get("ml_active_prep")
        y_tr_enc = st.session_state.get("ml_y_train_enc")
        y_te_enc = st.session_state.get("ml_y_test_enc")
        target_classes = st.session_state.get("ml_target_classes")

        if split_res is None or prep_res is None:
            st.warning("⚠️ Please partition the dataset in '2. Features & Split' before launching training.")
        else:
            prob_type = target_summary.problem_type
            is_class = prob_type in (ProblemType.BINARY_CLASSIFICATION, ProblemType.MULTICLASS_CLASSIFICATION)

            registry = CLASSIFICATION_MODELS_REGISTRY if is_class else REGRESSION_MODELS_REGISTRY
            all_model_keys = list(registry.keys())
            model_options = {k: registry[k]["display_name"] for k in all_model_keys}

            c_b1, c_b2 = st.columns([2, 1])
            with c_b1:
                selected_models = st.multiselect(
                    "Select Models to Benchmark",
                    options=all_model_keys,
                    default=all_model_keys[:4],
                    format_func=lambda k: model_options.get(k, k),
                )
            with c_b2:
                primary_metric_opts = (
                    ["F1-Score (Macro)", "Accuracy", "ROC-AUC", "Precision (Macro)", "Recall (Macro)"]
                    if is_class
                    else ["RMSE", "MAE", "R²", "Adjusted R²"]
                )
                ranking_metric = st.selectbox("Primary Ranking Metric", primary_metric_opts, index=0)
                handle_imb = st.checkbox("Apply Class Weight Balancing", value=target_summary.is_imbalanced if is_class else False)

            if st.button("🚀 Train & Benchmark Models", type="primary"):
                if not selected_models:
                    st.error("Please select at least one algorithm to train.")
                else:
                    progress_bar = st.progress(0, text="Training models...")
                    with st.spinner("Training models and running cross-validation..."):
                        try:
                            comp_res = engine.benchmark_models(
                                selected_models=selected_models,
                                split_res=split_res,
                                prep_res=prep_res,
                                y_train_enc=y_tr_enc,
                                y_test_enc=y_te_enc,
                                problem_type=prob_type,
                                target_classes=target_classes,
                                primary_metric=ranking_metric,
                                handle_imbalance=handle_imb,
                                random_state=split_cfg.random_state,
                            )
                            st.session_state["ml_comparison_result"] = comp_res
                            progress_bar.progress(100, text="Training complete!")
                            st.success(f"🏆 Top model: **{comp_res.best_model_key}** (Ranked by {ranking_metric})")
                        except Exception as ex:
                            st.error(f"Training failed: {str(ex)}")

        comp_res = st.session_state.get("ml_comparison_result")
        if comp_res is not None:
            st.markdown("---")
            st.markdown("##### Benchmark Comparison Matrix")
            st.dataframe(comp_res.comparison_table, use_container_width=True, hide_index=True)

            fig_comp = plot_model_comparison_bar(
                comparison_df=comp_res.comparison_table,
                primary_metric=comp_res.primary_metric,
            )
            st.plotly_chart(fig_comp, use_container_width=True)

    # =========================================================================
    # TAB 4: Model Evaluation
    # =========================================================================
    with tab_eval:
        st.subheader("Model Diagnostic Evaluation")
        comp_res = st.session_state.get("ml_comparison_result")
        split_res = st.session_state.get("ml_active_split")
        y_te_enc = st.session_state.get("ml_y_test_enc")
        target_classes = st.session_state.get("ml_target_classes")

        if comp_res is None:
            st.info("ℹ️ Run model training in '3. Benchmark Training' to view diagnostic curves and evaluation metrics.")
        else:
            model_keys = list(comp_res.results_by_model.keys())
            eval_model_key = st.selectbox(
                "Select Model to Inspect",
                options=model_keys,
                index=model_keys.index(comp_res.best_model_key) if comp_res.best_model_key in model_keys else 0,
                format_func=lambda k: comp_res.results_by_model[k]["display_name"],
            )

            model_data = comp_res.results_by_model[eval_model_key]
            eval_res = model_data["evaluation"]
            pipeline = model_data["pipeline"]
            prob_type = comp_res.problem_type
            is_class = prob_type in (ProblemType.BINARY_CLASSIFICATION, ProblemType.MULTICLASS_CLASSIFICATION)

            if is_class:
                e1, e2, e3, e4 = st.columns(4)
                e1.metric("Accuracy", f"{eval_res.accuracy:.1%}")
                e2.metric("Macro F1", f"{eval_res.f1_macro:.3f}")
                e3.metric("Macro Recall", f"{eval_res.recall_macro:.3f}")
                roc_str = f"{eval_res.roc_auc:.3f}" if eval_res.roc_auc is not None else "N/A"
                e4.metric("ROC-AUC", roc_str)

                c_g1, c_g2 = st.columns(2)
                with c_g1:
                    if eval_res.confusion_matrix and eval_res.labels:
                        fig_cm = plot_confusion_matrix(eval_res.confusion_matrix, eval_res.labels)
                        st.plotly_chart(fig_cm, use_container_width=True)
                with c_g2:
                    if eval_res.roc_curve_data:
                        fig_roc = plot_roc_curve(eval_res.roc_curve_data)
                        st.plotly_chart(fig_roc, use_container_width=True)
                    elif eval_res.pr_curve_data:
                        fig_pr = plot_pr_curve(eval_res.pr_curve_data)
                        st.plotly_chart(fig_pr, use_container_width=True)
                    else:
                        st.info("ROC Curve not available for multiclass or models without probability scores.")

                # Classification Report
                st.markdown("##### Detailed Classification Report")
                rep_df = pd.DataFrame(eval_res.classification_report_dict).T
                st.dataframe(rep_df, use_container_width=True)

            else:
                # Regression
                e1, e2, e3, e4 = st.columns(4)
                e1.metric("MAE", f"{eval_res.mae:.3f}")
                e2.metric("RMSE", f"{eval_res.rmse:.3f}")
                e3.metric("R² Score", f"{eval_res.r2:.3f}")
                e4.metric("Adjusted R²", f"{eval_res.adjusted_r2:.3f}")

                y_pred_eval = pipeline.predict(split_res.X_test)
                c_g1, c_g2 = st.columns(2)
                with c_g1:
                    fig_avp = plot_actual_vs_predicted(y_te_enc, y_pred_eval)
                    st.plotly_chart(fig_avp, use_container_width=True)
                with c_g2:
                    if eval_res.residuals:
                        fig_res = plot_residual_diagnostics(eval_res.residuals, y_pred_eval)
                        st.plotly_chart(fig_res, use_container_width=True)

    # =========================================================================
    # TAB 5: Feature Importance
    # =========================================================================
    with tab_importance:
        st.subheader("Feature Importance & Predictive Drivers")
        st.markdown(
            "Analyze which features contribute most strongly to model predictions. "
            "Supports tree-based impurity, linear model coefficients, and model-agnostic permutation importance."
        )

        comp_res = st.session_state.get("ml_comparison_result")
        split_res = st.session_state.get("ml_active_split")
        prep_res = st.session_state.get("ml_active_prep")
        y_te_enc = st.session_state.get("ml_y_test_enc")

        if comp_res is None or split_res is None or prep_res is None:
            st.info("ℹ️ Train models in '3. Benchmark Training' to inspect feature importance rankings.")
        else:
            model_keys = list(comp_res.results_by_model.keys())
            imp_model_key = st.selectbox(
                "Select Model for Feature Importance Analysis",
                options=model_keys,
                index=model_keys.index(comp_res.best_model_key) if comp_res.best_model_key in model_keys else 0,
                format_func=lambda k: comp_res.results_by_model[k]["display_name"],
                key="ml_imp_model_sel",
            )

            c_im1, c_im2 = st.columns([1.5, 1])
            with c_im1:
                imp_method = st.selectbox(
                    "Importance Strategy",
                    ["auto", "tree", "permutation", "coefficients"],
                    help="'auto' selects native tree importances or linear coefficients, falling back to permutation.",
                )
            with c_im2:
                n_repeats = st.slider("Permutation Repeats", min_value=3, max_value=10, value=5)

            if st.button("📊 Calculate Feature Importance", type="primary"):
                model_data = comp_res.results_by_model[imp_model_key]
                estimator = model_data["estimator"]
                with st.spinner("Computing feature importances..."):
                    try:
                        imp_res = engine.analyze_feature_importance(
                            estimator=estimator,
                            prep_res=prep_res,
                            split_res=split_res,
                            y_test_enc=y_te_enc,
                            method=imp_method,
                            model_name=model_data["display_name"],
                            n_repeats=n_repeats,
                            random_state=split_cfg.random_state,
                        )
                        st.session_state["ml_active_importance"] = imp_res
                    except Exception as ex:
                        st.error(f"Importance calculation failed: {str(ex)}")

            imp_res = st.session_state.get("ml_active_importance")
            if imp_res is not None:
                st.markdown(
                    f"""
                    <div style="background: rgba(99, 102, 241, 0.1); border: 1px solid rgba(99, 102, 241, 0.3); border-radius: 8px; padding: 0.8rem; margin: 1rem 0;">
                        <strong>Method Applied:</strong> {imp_res.method_display}<br>
                        <strong>Description:</strong> {imp_res.explanation}<br>
                        <span style="color: #F59E0B; font-size: 0.82rem;">⚠️ {imp_res.disclaimer}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                fig_imp = plot_feature_importance_bars(imp_res.importance_df, max_features=15)
                st.plotly_chart(fig_imp, use_container_width=True)

                st.markdown("##### Feature Importance Data Table")
                st.dataframe(imp_res.importance_df, use_container_width=True, hide_index=True)

    # =========================================================================
    # TAB 6: Hyperparameter Tuning
    # =========================================================================
    with tab_tuning:
        st.subheader("Hyperparameter Optimization")
        st.markdown(
            "Perform automated Grid Search or Randomized Search to fine-tune model hyperparameters strictly on training data."
        )

        comp_res = st.session_state.get("ml_comparison_result")
        split_res = st.session_state.get("ml_active_split")
        prep_res = st.session_state.get("ml_active_prep")
        y_tr_enc = st.session_state.get("ml_y_train_enc")

        if comp_res is None or split_res is None or prep_res is None:
            st.info("ℹ️ Train models first before tuning hyperparameters.")
        else:
            tune_model_key = st.selectbox(
                "Model to Optimize",
                options=list(comp_res.results_by_model.keys()),
                index=0,
                format_func=lambda k: comp_res.results_by_model[k]["display_name"],
                key="ml_tune_key",
            )

            c_t1, c_t2, c_t3 = st.columns(3)
            with c_t1:
                search_type = st.radio("Search Method", ["Grid Search", "Randomized Search"], index=0)
            with c_t2:
                tune_cv = st.slider("Tuning CV Folds", 2, 5, 3)
            with c_t3:
                tune_iter = st.slider("Random Search Iterations", 4, 15, 8)

            if st.button("🔧 Launch Hyperparameter Tuning", type="primary"):
                with st.spinner("Searching hyperparameter space..."):
                    try:
                        tune_res = engine.tune_hyperparameters(
                            model_key=tune_model_key,
                            prep_res=prep_res,
                            split_res=split_res,
                            y_train_enc=y_tr_enc,
                            problem_type=comp_res.problem_type,
                            search_type="grid" if "Grid" in search_type else "random",
                            cv_folds=tune_cv,
                            n_iter=tune_iter,
                            random_state=split_cfg.random_state,
                        )
                        st.session_state["ml_latest_tuning"] = tune_res
                        st.success(f"✅ Tuning complete! Best CV Score ({tune_res.scoring_metric}): **{tune_res.best_score:.4f}**")
                    except Exception as ex:
                        st.error(f"Tuning failed: {str(ex)}")

            tune_res = st.session_state.get("ml_latest_tuning")
            if tune_res is not None:
                st.markdown("##### Best Hyperparameters Found")
                st.json(tune_res.best_params)

                st.markdown("##### CV Results Summary")
                st.dataframe(tune_res.cv_results_summary, use_container_width=True)

    # =========================================================================
    # TAB 7: Prediction Explorer
    # =========================================================================
    with tab_predict:
        st.subheader("Holdout Prediction Explorer & Record Inspector")
        st.markdown(
            "Inspect individual record predictions, confidence scores, and residuals on the holdout test set. "
            "Filter misclassifications or large error outliers to understand model blind spots."
        )

        comp_res = st.session_state.get("ml_comparison_result")
        split_res = st.session_state.get("ml_active_split")
        y_te_enc = st.session_state.get("ml_y_test_enc")
        target_classes = st.session_state.get("ml_target_classes")

        if comp_res is None or split_res is None:
            st.info("ℹ️ Train models in '3. Benchmark Training' to explore predictions.")
        else:
            pred_model_key = st.selectbox(
                "Model to Explore Predictions",
                options=list(comp_res.results_by_model.keys()),
                index=list(comp_res.results_by_model.keys()).index(comp_res.best_model_key),
                format_func=lambda k: comp_res.results_by_model[k]["display_name"],
                key="ml_pred_model_key",
            )

            pipeline = comp_res.results_by_model[pred_model_key]["pipeline"]
            pred_res = engine.explore_predictions(
                pipeline=pipeline,
                split_res=split_res,
                y_test_enc=y_te_enc,
                problem_type=comp_res.problem_type,
                target_name=target_summary.target_name,
                target_classes=target_classes,
                model_name=comp_res.results_by_model[pred_model_key]["display_name"],
            )

            # Summary metrics
            s_cols = st.columns(len(pred_res.summary_stats))
            for col, (k, v) in zip(s_cols, pred_res.summary_stats.items()):
                col.metric(k, f"{v:,}" if isinstance(v, int) else (f"{v:.1%}" if "Accuracy" in k else f"{v}"))

            # Filtering options
            df_preds = pred_res.predictions_df
            is_class = comp_res.problem_type in (ProblemType.BINARY_CLASSIFICATION, ProblemType.MULTICLASS_CLASSIFICATION)

            if is_class:
                filter_choice = st.radio("Filter Records", ["All Records", "Misclassifications Only", "Correct Predictions Only"], horizontal=True)
                if filter_choice == "Misclassifications Only":
                    df_preds = df_preds[df_preds["Outcome_Status"] == "Misclassification"]
                elif filter_choice == "Correct Predictions Only":
                    df_preds = df_preds[df_preds["Outcome_Status"] == "Correct Prediction"]
            else:
                sort_err = st.checkbox("Sort by Largest Prediction Error First", value=True)
                if sort_err and "Abs_Error" in df_preds.columns:
                    df_preds = df_preds.sort_values(by="Abs_Error", ascending=False)

            st.dataframe(df_preds, use_container_width=True)

            csv_data = df_preds.to_csv(index=False)
            st.download_button(
                label="📥 Export Predictions to CSV",
                data=csv_data,
                file_name=f"predictions_{pred_model_key}_{target_summary.target_name}.csv",
                mime="text/csv",
            )

    # =========================================================================
    # TAB 8: Narrative Insights & Registry
    # =========================================================================
    with tab_insights:
        st.subheader("Automated Data-Driven Machine Learning Insights")
        st.markdown(
            "Empirical insights synthesized directly from model metrics, generalization curves, and feature importance."
        )

        comp_res = st.session_state.get("ml_comparison_result")
        imp_res = st.session_state.get("ml_active_importance")

        if comp_res is None:
            st.info("ℹ️ Train models to generate automated machine learning insights.")
        else:
            insights = engine.generate_narrative_insights(
                comparison_res=comp_res,
                target_summary=target_summary,
                importance_res=imp_res,
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
                            <strong>Supporting Evidence:</strong> {ins.metric_evidence}<br>
                            <strong>Limitations:</strong> {ins.limitations}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # Model Registration / Artifact Saving Section
            st.markdown("---")
            st.subheader("Model Artifact Registry & Serialization")
            st.markdown("Register top-performing trained pipelines for downstream explainability and investigation.")

            reg_col1, reg_col2 = st.columns([2, 1])
            with reg_col1:
                reg_model_key = st.selectbox(
                    "Model to Save to Registry",
                    options=list(comp_res.results_by_model.keys()),
                    format_func=lambda k: comp_res.results_by_model[k]["display_name"],
                    key="ml_reg_key",
                )
            with reg_col2:
                model_notes = st.text_input("Model Notes / Tags", value="Phase 7 Best Benchmark")

            if st.button("💾 Save Pipeline to Session Registry", type="primary"):
                model_info = comp_res.results_by_model[reg_model_key]
                artifact = engine.registry.register_model(
                    model_name=model_info["display_name"],
                    pipeline=model_info["pipeline"],
                    problem_type=comp_res.problem_type,
                    target_name=target_summary.target_name,
                    feature_names_in=prep_res.feature_names_in,
                    feature_names_out=prep_res.feature_names_out,
                    metrics=model_info["evaluation"].to_dict(),
                    target_classes=st.session_state.get("ml_target_classes"),
                    notes=model_notes,
                )
                st.session_state["active_ml_artifact"] = artifact
                st.success(f"✅ Model registered successfully with ID: `{artifact.model_id}`")

            # Registered models table
            registered_list = engine.registry.list_models()
            if registered_list:
                st.markdown("##### Registered Models in Session")
                st.dataframe(pd.DataFrame(registered_list), use_container_width=True)

