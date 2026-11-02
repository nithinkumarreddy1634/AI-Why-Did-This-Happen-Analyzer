"""Automated, data-driven machine learning insights generator."""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np

from modules.machine_learning.problem_detector import ProblemType, TargetSummary
from modules.machine_learning.model_comparison import ModelComparisonResult
from modules.machine_learning.feature_importance import FeatureImportanceResult


@dataclass
class MLInsight:
    """Structured insight item with empirical backing and analytical context."""
    title: str
    category: str  # 'Performance', 'Features', 'Imbalance', 'Generalization', 'Causality Alert'
    description: str
    metric_evidence: str
    limitations: str
    badge_type: str = "info"  # 'info', 'warning', 'success', 'purple'

    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "category": self.category,
            "description": self.description,
            "metric_evidence": self.metric_evidence,
            "limitations": self.limitations,
            "badge_type": self.badge_type,
        }


def generate_ml_insights(
    comparison_result: ModelComparisonResult,
    target_summary: TargetSummary,
    importance_result: Optional[FeatureImportanceResult] = None,
) -> List[MLInsight]:
    """Generate empirical, non-causal narrative insights from trained models and evaluation metrics."""
    insights: List[MLInsight] = []
    df_comp = comparison_result.comparison_table
    prob_type = comparison_result.problem_type
    best_key = comparison_result.best_model_key
    primary_metric = comparison_result.primary_metric

    if df_comp.empty:
        return insights

    best_row = df_comp.iloc[0]
    best_name = best_row.get("Model", best_key)

    # 1. Top Performing Algorithm Insight
    if prob_type in (ProblemType.BINARY_CLASSIFICATION, ProblemType.MULTICLASS_CLASSIFICATION):
        f1_val = best_row.get("F1-Score (Macro)", np.nan)
        acc_val = best_row.get("Accuracy", np.nan)
        roc_val = best_row.get("ROC-AUC", np.nan)
        
        evidence_parts = []
        if not np.isnan(acc_val):
            evidence_parts.append(f"Accuracy = {acc_val:.1%}")
        if not np.isnan(f1_val):
            evidence_parts.append(f"Macro F1 = {f1_val:.3f}")
        if not np.isnan(roc_val):
            evidence_parts.append(f"ROC-AUC = {roc_val:.3f}")

        evidence_str = ", ".join(evidence_parts)

        insights.append(
            MLInsight(
                title=f"Top Predictive Algorithm: {best_name}",
                category="Performance",
                description=(
                    f"Among the evaluated algorithms, **{best_name}** achieved the strongest benchmark performance "
                    f"ranked by {primary_metric} on the holdout test partition."
                ),
                metric_evidence=evidence_str,
                limitations="Performance measured on holdout split; results may vary with different random partitions.",
                badge_type="success",
            )
        )
    else:
        # Regression
        r2_val = best_row.get("R²", np.nan)
        rmse_val = best_row.get("RMSE", np.nan)
        mae_val = best_row.get("MAE", np.nan)

        evidence_parts = []
        if not np.isnan(r2_val):
            evidence_parts.append(f"R² = {r2_val:.3f}")
        if not np.isnan(rmse_val):
            evidence_parts.append(f"RMSE = {rmse_val:.3f}")
        if not np.isnan(mae_val):
            evidence_parts.append(f"MAE = {mae_val:.3f}")

        insights.append(
            MLInsight(
                title=f"Top Predictive Algorithm: {best_name}",
                category="Performance",
                description=(
                    f"**{best_name}** achieved the lowest prediction error and highest explained variance "
                    f"ranked by {primary_metric}."
                ),
                metric_evidence=", ".join(evidence_parts),
                limitations="Linear or non-linear error metrics reflect empirical fit on sample data.",
                badge_type="success",
            )
        )

    # 2. Class Imbalance Impact Warning
    if prob_type in (ProblemType.BINARY_CLASSIFICATION, ProblemType.MULTICLASS_CLASSIFICATION):
        if target_summary.is_imbalanced:
            acc = best_row.get("Accuracy", 0.0)
            f1_m = best_row.get("F1-Score (Macro)", 0.0)
            rec_m = best_row.get("Recall (Macro)", 0.0)
            
            gap = acc - f1_m
            if gap > 0.10:
                insights.append(
                    MLInsight(
                        title="Class Imbalance Distortion Detected (Accuracy Paradox)",
                        category="Imbalance",
                        description=(
                            f"The target outcome has an imbalance ratio of **{target_summary.imbalance_ratio}:1**. "
                            f"While headline Accuracy is {acc:.1%}, Macro F1-score is noticeably lower at {f1_m:.3f}. "
                            "This indicates the model performs better on majority cases than on rare/critical minority events."
                        ),
                        metric_evidence=f"Accuracy ({acc:.1%}) exceeds Macro F1 ({f1_m:.3f}) by {gap:.1%}",
                        limitations="Accuracy is not a reliable standalone metric on imbalanced data; focus on Precision, Recall, and PR-AUC.",
                        badge_type="warning",
                    )
                )

    # 3. Model Generalization / Overfitting Check
    cv_mean = best_row.get("CV Mean", np.nan)
    test_score = (
        best_row.get("F1-Score (Macro)", np.nan)
        if prob_type in (ProblemType.BINARY_CLASSIFICATION, ProblemType.MULTICLASS_CLASSIFICATION)
        else best_row.get("RMSE", np.nan)
    )

    if not np.isnan(cv_mean) and not np.isnan(test_score):
        score_diff = abs(cv_mean - test_score)
        if score_diff < 0.08:
            insights.append(
                MLInsight(
                    title="Stable Generalization Across Cross-Validation",
                    category="Generalization",
                    description=(
                        f"Cross-validation performance ({cv_mean:.3f}) closely aligns with holdout test performance ({test_score:.3f}), "
                        "suggesting stable generalization without severe overfitting or data distribution mismatch."
                    ),
                    metric_evidence=f"|CV ({cv_mean:.3f}) - Test ({test_score:.3f})| = {score_diff:.3f}",
                    limitations="Assumes training and testing data are identically and independently distributed.",
                    badge_type="info",
                )
            )
        elif score_diff > 0.15:
            insights.append(
                MLInsight(
                    title="Potential Generalization Variance",
                    category="Generalization",
                    description=(
                        f"A noticeable divergence exists between training cross-validation ({cv_mean:.3f}) "
                        f"and holdout test performance ({test_score:.3f}). Consider regularizing hyperparameters or collecting additional samples."
                    ),
                    metric_evidence=f"Score divergence: {score_diff:.3f}",
                    limitations="Variance can be caused by small test sample size or distribution shifts.",
                    badge_type="warning",
                )
            )

    # 4. Top Predictive Features Insight
    if importance_result and importance_result.top_features:
        top_3 = importance_result.top_features[:3]
        top_str = ", ".join([f"`{f}`" for f in top_3])
        method_str = importance_result.method_display
        insights.append(
            MLInsight(
                title="Dominant Predictive Signals",
                category="Features",
                description=(
                    f"Based on **{method_str}**, the strongest predictive signals associated with target `{target_summary.target_name}` "
                    f"are {top_str}."
                ),
                metric_evidence=f"Top predictor: '{top_3[0]}'",
                limitations="Predictive ranking is model-dependent; correlated features may mask each other's individual influence.",
                badge_type="purple",
            )
        )

    # 5. Methodological Non-Causality Principle (Always appended)
    insights.append(
        MLInsight(
            title="Predictive Association ≠ Causal Mechanism",
            category="Causality Alert",
            description=(
                "Supervised machine learning algorithms learn statistical predictive associations to optimize loss functions. "
                "A strong predictive relationship does **not** establish that changing a feature will causally alter the outcome in reality."
            ),
            metric_evidence="Fundamental Principle of Empirical Machine Learning",
            limitations="Causal claims require randomized controlled experiments, quasi-experimental designs, or structural causal models.",
            badge_type="warning",
        )
    )

    return insights

