"""Automated narrative generator synthesizing data-driven LIME explainability insights."""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional

from modules.explainability.lime_local_analysis import LIMEExplanationResult
from modules.explainability.lime_shap_comparison import SHAPLIMEComparisonResult
from modules.explainability.lime_batch_analysis import BatchLIMEResult


@dataclass
class LIMEInsight:
    """Individual narrative card explaining local model behavior and surrogate fidelity."""
    title: str
    category: str  # 'Local Attribution', 'Surrogate Fidelity', 'Method Comparison', 'Cohort Pattern', 'Causality Alert'
    description: str
    supporting_evidence: str
    caveats: str
    badge_type: str = "info"  # 'info', 'success', 'warning', 'purple'

    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "category": self.category,
            "description": self.description,
            "supporting_evidence": self.supporting_evidence,
            "caveats": self.caveats,
            "badge_type": self.badge_type,
        }


def generate_lime_narrative_insights(
    lime_result: LIMEExplanationResult,
    comparison_result: Optional[SHAPLIMEComparisonResult] = None,
    batch_result: Optional[BatchLIMEResult] = None,
) -> List[LIMEInsight]:
    """Synthesize empirical LIME narrative findings, fidelity diagnostics, and comparison insights."""
    insights: List[LIMEInsight] = []

    # 1. Primary Local Drivers
    idx = lime_result.sample_index
    target_str = lime_result.target_class_name
    pos = lime_result.positive_drivers
    neg = lime_result.negative_drivers

    pos_desc = f"**{pos[0]['Condition']}** ({pos[0]['Formatted_Weight']})" if pos else "None"
    neg_desc = f"**{neg[0]['Condition']}** ({neg[0]['Formatted_Weight']})" if neg else "None"

    insights.append(
        LIMEInsight(
            title=f"Primary Local Drivers for Record #{idx}",
            category="Local Attribution",
            description=(
                f"For record #{idx} ({target_str}), LIME's local surrogate identifies "
                f"{pos_desc} as the strongest factor supporting this prediction, while "
                f"{neg_desc} exerts the strongest opposing influence."
            ),
            supporting_evidence=f"Predicted = {lime_result.predicted_outcome}, Surrogate = {lime_result.local_surrogate_prediction:.3f}",
            caveats="Weights apply within the perturbed neighborhood of this specific observation.",
            badge_type="purple",
        )
    )

    # 2. Local Surrogate Fidelity (R²) Diagnostic
    r2 = lime_result.local_fidelity_r2
    gap = lime_result.fidelity_gap

    if r2 >= 0.70:
        fid_text = "high explanatory fidelity"
        fid_badge = "success"
    elif r2 >= 0.40:
        fid_text = "moderate local fidelity"
        fid_badge = "info"
    else:
        fid_text = "low local fidelity"
        fid_badge = "warning"

    insights.append(
        LIMEInsight(
            title=f"Local Surrogate Fidelity (R² = {r2:.3f})",
            category="Surrogate Fidelity",
            description=(
                f"LIME's interpretable ridge surrogate achieved an **R² = {r2:.3f}** ({fid_text}) "
                f"in approximating the complex black-box model within the local perturbed neighborhood. "
                f"The absolute prediction gap between the original model and surrogate is **{gap:.4f}**."
            ),
            supporting_evidence=f"Neighborhood R² = {r2:.3f}, |Original - Surrogate| = {gap:.4f}",
            caveats="Local fidelity measures how well the linear surrogate approximates the original model locally, NOT overall dataset test accuracy.",
            badge_type=fid_badge,
        )
    )

    # 3. SHAP vs LIME Comparison Consensus (if available)
    if comparison_result is not None:
        conc = comparison_result.concordance_level
        shared_count = len(comparison_result.common_top_features)
        jaccard = comparison_result.jaccard_similarity
        dir_agree = comparison_result.direction_agreement_pct

        insights.append(
            LIMEInsight(
                title=f"XAI Consensus: {conc}",
                category="Method Comparison",
                description=(
                    f"Comparing cooperative game theory (SHAP) with local perturbation surrogates (LIME) reveals "
                    f"**{shared_count} shared top features** ({jaccard:.1%} Jaccard overlap) with **{dir_agree:.0f}% directional agreement**. "
                    f"Features prioritized by both paradigms represent robust, multi-method predictive associations."
                ),
                supporting_evidence=f"Jaccard = {jaccard:.2f}, Rank Correlation = {comparison_result.rank_correlation or 'N/A'}, Direction Agreement = {dir_agree:.0f}%",
                caveats="Differences in weights arise naturally from SHAP's global coalition efficiency vs LIME's local neighborhood sampling.",
                badge_type="info" if "Moderate" in conc else ("success" if "High" in conc else "warning"),
            )
        )

    # 4. Cohort Recurrence Pattern (if batch available)
    if batch_result is not None and batch_result.recurring_top_features:
        top_rec = batch_result.recurring_top_features[0]
        insights.append(
            LIMEInsight(
                title=f"Cohort Recurrent Factor: {top_rec['Feature']}",
                category="Cohort Pattern",
                description=(
                    f"Across the cohort of {batch_result.n_explained} evaluated records, feature **'{top_rec['Feature']}'** "
                    f"consistently ranked among the top 3 predictive drivers in **{top_rec['Frequency_Pct']}%** of instances."
                ),
                supporting_evidence=f"Appeared in {top_rec['Appearance_Count']} of {batch_result.n_explained} explanations",
                caveats="Recurrence across instances indicates a stable localized pattern across that subgroup.",
                badge_type="purple",
            )
        )

    # 5. Mandatory Causality Alert
    insights.append(
        LIMEInsight(
            title="Non-Causality Principle: LIME Weights ≠ Real-World Causes",
            category="Causality Alert",
            description=(
                "LIME explains the decision surface of the mathematical algorithm. "
                "Even when LIME assigns a heavy positive weight to a feature, modifying that feature in real life "
                "may have zero effect on outcomes unless verified through causal DAG identification or randomized experiments."
            ),
            supporting_evidence="Core axiomatic foundation of local surrogate interpretability (Ribeiro et al., 2016)",
            caveats="Perturbation sampling around an instance does not simulate actual real-world interventions.",
            badge_type="warning",
        )
    )

    return insights

