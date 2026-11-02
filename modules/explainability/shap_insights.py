"""Automated narrative generator synthesizing data-driven SHAP insights."""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional

from modules.explainability.shap_global import GlobalSHAPResult
from modules.explainability.shap_local import LocalSHAPResult


@dataclass
class SHAPInsight:
    """Individual narrative insight card with empirical evidence and methodological caveats."""
    title: str
    category: str  # 'Global Driver', 'Local Impact', 'Directionality', 'Interaction', 'Causality Alert'
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


def generate_shap_narrative_insights(
    global_result: GlobalSHAPResult,
    local_result: Optional[LocalSHAPResult] = None,
) -> List[SHAPInsight]:
    """Synthesize global population attributions and optional single-instance local drivers."""
    insights: List[SHAPInsight] = []
    df_imp = global_result.importance_df

    # 1. Primary Global Feature Driver
    if not df_imp.empty:
        top_row = df_imp.iloc[0]
        top_name = top_row["Feature"]
        top_share = top_row["Contribution_Pct"]
        top_shap = top_row["Mean_Abs_SHAP"]

        insights.append(
            SHAPInsight(
                title=f"Primary Population Driver: {top_name}",
                category="Global Driver",
                description=(
                    f"Feature **'{top_name}'** accounts for **{top_share:.1f}%** of the model's total predictive attribution "
                    f"across all {global_result.total_explained_samples:,} evaluated observations. On average, it shifts "
                    f"predictions by ±{top_shap:.4f} units from the baseline."
                ),
                supporting_evidence=f"Mean |SHAP| = {top_shap:.4f} ({top_share:.1f}% share)",
                caveats="Attribution magnitude reflects predictive dependence inside this trained model, not real-world causality.",
                badge_type="purple",
            )
        )

    # 2. Consistent Direction Tendencies
    directional_feats = []
    if not df_imp.empty:
        for _, r in df_imp.head(5).iterrows():
            if r["Positive_Impact_Frequency"] >= 70.0:
                directional_feats.append(f"**'{r['Feature']}'** predominantly increases the model outcome (+{r['Mean_Positive_Shift']:.3f})")
            elif r["Negative_Impact_Frequency"] >= 70.0:
                directional_feats.append(f"**'{r['Feature']}'** predominantly suppresses the model outcome ({r['Mean_Negative_Shift']:.3f})")

    if directional_feats:
        insights.append(
            SHAPInsight(
                title="Consistent Asymmetric Directional Shifts",
                category="Directionality",
                description="Several top features show strong directional asymmetry across the dataset:\n• " + "\n• ".join(directional_feats),
                supporting_evidence="Observed >70% directional sign consistency in SHAP distributions",
                caveats="Directional tendencies can flip in non-linear models when interacting with other correlated features.",
                badge_type="info",
            )
        )

    # 3. Local Single-Observation Explanation (if provided)
    if local_result is not None:
        obs_idx = local_result.sample_index
        pred_out = local_result.model_output
        base_v = local_result.base_value
        delta = pred_out - base_v

        pos_names = [f"`{d['Feature']}` (+{d['SHAP_Value']:.3f})" for d in local_result.positive_drivers[:2]]
        neg_names = [f"`{d['Feature']}` ({d['SHAP_Value']:.3f})" for d in local_result.negative_drivers[:2]]

        pos_str = f"Upward drivers: {', '.join(pos_names)}. " if pos_names else ""
        neg_str = f"Downward drivers: {', '.join(neg_names)}." if neg_names else ""

        status_str = "substantially above" if delta > 0.05 else ("substantially below" if delta < -0.05 else "near")

        insights.append(
            SHAPInsight(
                title=f"Local Decomposition for Record #{obs_idx}",
                category="Local Impact",
                description=(
                    f"Observation #{obs_idx} received a model prediction of **{pred_out:.3f}**, which is {status_str} "
                    f"the population expected base value of **{base_v:.3f}** (net shift: {delta:+.3f}). "
                    f"{pos_str}{neg_str}"
                ),
                supporting_evidence=f"f(x) = {pred_out:.4f}, E[f(X)] = {base_v:.4f}, sum(SHAP) = {delta:+.4f}",
                caveats="Local attributions sum exactly to the prediction difference under the Shapley efficiency axiom.",
                badge_type="success",
            )
        )

    # 4. Mandatory Non-Causality Warning
    insights.append(
        SHAPInsight(
            title="Interpretability Principle: SHAP ≠ Causal Root Cause",
            category="Causality Alert",
            description=(
                "SHAP values answer: *'How did the model use feature X to arrive at its prediction?'* "
                "They do **not** answer: *'If we change feature X in the real world, will the outcome change?'* "
                "Interventions require causal inference DAGs or controlled randomized A/B experimentation."
            ),
            supporting_evidence="Formal mathematical foundation of Shapley additive explanations (Lundberg & Lee, 2017)",
            caveats="Confounders and multicollinearity can distribute attributions arbitrarily across correlated features.",
            badge_type="warning",
        )
    )

    return insights

