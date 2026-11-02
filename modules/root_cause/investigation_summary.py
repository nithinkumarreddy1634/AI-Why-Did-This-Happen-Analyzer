"""Automated deterministic investigation summary synthesizing multi-engine root cause findings."""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
import pandas as pd

from modules.root_cause.candidate_generator import OutcomeDefinition
from modules.root_cause.root_cause_ranker import RootCauseRankingResult, RankedRootCauseItem
from modules.root_cause.segment_contribution import SegmentAnalysisResult
from modules.root_cause.interaction_analysis import InteractionAnalysisResult


@dataclass
class InvestigationSummaryReport:
    """Comprehensive narrative summary answering the 10 mandated investigation questions."""
    q1_observed_outcome: str
    q2_change_magnitude: str
    q3_most_affected_groups: str
    q4_top_evidence_features: str
    q5_statistical_findings: str
    q6_ml_model_insights: str
    q7_shap_lime_revelations: str
    q8_unusual_observations: str
    q9_conflicting_findings: str
    q10_missing_information: str
    executive_narrative: str
    recommendations: List[str]
    disclaimer: str = (
        "Causality Disclaimer: These conclusions synthesize observational evidence and predictive attribution. "
        "No statistical technique applied here can prove counterfactual causality. "
        "All candidates should be verified via randomized A/B testing or domain-level domain interventions."
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary."""
        return {
            "q1_observed_outcome": self.q1_observed_outcome,
            "q2_change_magnitude": self.q2_change_magnitude,
            "q3_most_affected_groups": self.q3_most_affected_groups,
            "q4_top_evidence_features": self.q4_top_evidence_features,
            "q5_statistical_findings": self.q5_statistical_findings,
            "q6_ml_model_insights": self.q6_ml_model_insights,
            "q7_shap_lime_revelations": self.q7_shap_lime_revelations,
            "q8_unusual_observations": self.q8_unusual_observations,
            "q9_conflicting_findings": self.q9_conflicting_findings,
            "q10_missing_information": self.q10_missing_information,
            "executive_narrative": self.executive_narrative,
            "recommendations": self.recommendations,
            "disclaimer": self.disclaimer,
        }

    def to_markdown(self) -> str:
        """Render markdown text report."""
        return f"""# Root Cause Investigation Summary Report

> **{self.disclaimer}**

---

### 1. Observed Unexpected Outcome
{self.q1_observed_outcome}

### 2. Magnitude of Observed Change
{self.q2_change_magnitude}

### 3. Most Affected Subgroups & Demographics
{self.q3_most_affected_groups}

### 4. Features with Strongest Supporting Evidence
{self.q4_top_evidence_features}

### 5. Statistical Hypothesis Testing Findings
{self.q5_statistical_findings}

### 6. Predictive Machine Learning Insights
{self.q6_ml_model_insights}

### 7. Explainable AI (SHAP & LIME) Revelations
{self.q7_shap_lime_revelations}

### 8. Unusual Observations & Outlier Patterns
{self.q8_unusual_observations}

### 9. Conflicting or Divergent Findings
{self.q9_conflicting_findings}

### 10. Missing Information & Knowledge Gaps
{self.q10_missing_information}

---

### Executive Synthesis
{self.executive_narrative}

### Recommended Next Steps
""" + "\n".join(f"- {r}" for r in self.recommendations)


def generate_investigation_summary(
    outcome_def: OutcomeDefinition,
    ranking_res: RootCauseRankingResult,
    segment_res: Optional[SegmentAnalysisResult] = None,
    interaction_res: Optional[InteractionAnalysisResult] = None,
) -> InvestigationSummaryReport:
    """Generate deterministic narrative answering all 10 mandated investigation questions."""
    # Q1: Observed outcome
    q1 = (
        f"Investigation analyzed target variable **'{outcome_def.target_variable}'** "
        f"contrasting **'{outcome_def.baseline_name}'** (N={outcome_def.baseline_count}) "
        f"against **'{outcome_def.comparison_name}'** (N={outcome_def.comparison_count})."
    )

    # Q2: Change magnitude
    pct_str = f" ({outcome_def.relative_change_pct:+.1f}%)" if outcome_def.relative_change_pct is not None else ""
    q2 = (
        f"The outcome shifted from a baseline of **{outcome_def.baseline_value:.4f}** "
        f"to **{outcome_def.comparison_value:.4f}** in the comparison cohort, representing an absolute delta of "
        f"**{outcome_def.absolute_change:+.4f}**{pct_str}."
    )

    # Q3: Most affected groups
    if segment_res and segment_res.segments:
        q3 = (
            f"Segment decomposition of feature **'{segment_res.feature}'** reveals that cohort **'{segment_res.top_contributing_segment}'** "
            f"accounted for the largest contribution share (**{segment_res.segments[0].contribution_share_pct:.1f}%** of total shift), "
            f"while **'{segment_res.highest_rate_segment}'** registered the peak outcome rate."
        )
    else:
        q3 = "Segment contribution decomposition was not executed for a specific demographic."

    # Q4: Top evidence features
    top_cands = ranking_res.ranked_candidates[:3]
    if top_cands:
        cand_str = ", ".join(f"**{c.feature}** (Score: {c.overall_score:.2f}, {c.evidence_strength})" for c in top_cands)
        q4 = f"The highest-ranked potential contributing factors are: {cand_str}."
    else:
        q4 = "No candidate factors met minimum evidence thresholds."

    # Q5: Statistical findings
    sig_cands = [c for c in ranking_res.ranked_candidates if c.stat_significant]
    if sig_cands:
        s_names = ", ".join(f"**{c.feature}** (p={c.stat_p_value:.4f})" for c in sig_cands[:3])
        q5 = f"Independent two-sample hypothesis testing confirmed statistically significant differences for {s_names}."
    else:
        q5 = "No features demonstrated statistically significant differences at the α=0.05 threshold."

    # Q6: ML insights
    ml_cands = [c for c in ranking_res.ranked_candidates if c.ml_importance_rank is not None and c.ml_importance_rank <= 5]
    if ml_cands:
        ml_names = ", ".join(f"**{c.feature}** (Rank #{c.ml_importance_rank})" for c in ml_cands[:3])
        q6 = f"The trained machine learning model prioritized {ml_names} as top predictive drivers."
    else:
        q6 = "Machine learning model feature importance was either unexecuted or did not prioritize top candidates."

    # Q7: SHAP & LIME
    shap_cands = [c for c in ranking_res.ranked_candidates if c.shap_rank is not None and c.shap_rank <= 5]
    lime_cands = [c for c in ranking_res.ranked_candidates if c.lime_condition]
    shap_text = f"SHAP prioritized {', '.join(f'**{c.feature}**' for c in shap_cands[:2])}" if shap_cands else "SHAP global profiles uncomputed"
    lime_text = f"LIME local rules identified conditions like '{lime_cands[0].lime_condition}'" if lime_cands else "LIME local rules uncomputed"
    q7 = f"Explainable AI consensus indicates that {shap_text}, while {lime_text}."

    # Q8: Unusual observations
    anom_cands = [c for c in ranking_res.ranked_candidates if "Outlier Concentration" in " ".join(c.supporting_sources)]
    if anom_cands:
        q8 = f"Anomalous observation clustering was identified in {', '.join(f'**{c.feature}**' for c in anom_cands[:2])}."
    else:
        q8 = "No disproportionate outlier concentration was observed across comparison cohorts."

    # Q9: Conflicting findings
    conf_cands = [c for c in ranking_res.ranked_candidates if len(c.conflicting_sources) > 0]
    if conf_cands:
        q9 = (
            f"Divergence detected in **'{conf_cands[0].feature}'**: "
            + "; ".join(conf_cands[0].conflicting_sources)
            + ". Caution is advised as directional associations conflict across methods."
        )
    else:
        q9 = "All evaluated analytical engines exhibited concordant directional support without major conflicts."

    # Q10: Missing info
    missing_sources = []
    if not sig_cands and not any(c.stat_p_value is not None for c in ranking_res.ranked_candidates):
        missing_sources.append("Statistical hypothesis tests")
    if not ml_cands:
        missing_sources.append("Supervised ML model training")
    if not shap_cands:
        missing_sources.append("SHAP Game-Theoretic attributions")
    q10 = (
        f"Missing analytical context includes: {', '.join(missing_sources) if missing_sources else 'None; all core analytical engines executed'}."
    )

    # Executive narrative
    top_factor = top_cands[0] if top_cands else None
    top_feat_name = top_factor.feature if top_factor else "unidentified factors"
    top_strength = top_factor.evidence_strength if top_factor else "N/A"
    top_shift_val = f"{top_factor.absolute_shift:+.2f}" if top_factor else "0.00"

    exec_narr = (
        f"The unexpected shift in '{outcome_def.target_variable}' (Δ = {outcome_def.absolute_change:+.3f}) is most strongly "
        f"associated with **'{top_feat_name}'** ({top_strength}), showing an observed shift of "
        f"{top_shift_val}. Multi-engine evidence scoring indicates this candidate "
        f"merits primary attention for operational diagnosis."
    )

    # Recommendations
    recs = [
        f"Focus immediate domain validation on **'{top_factor.feature if top_factor else 'top factors'}'**.",
        "Perform stratified cohort checks to ensure sample sizes remain sufficient across all operational tiers.",
        "Conduct a controlled randomized intervention or A/B experiment before implementing irreversible policy changes.",
    ]
    if segment_res:
        recs.append(f"Examine operational workflows targeting segment **'{segment_res.top_contributing_segment}'**.")

    return InvestigationSummaryReport(
        q1_observed_outcome=q1,
        q2_change_magnitude=q2,
        q3_most_affected_groups=q3,
        q4_top_evidence_features=q4,
        q5_statistical_findings=q5,
        q6_ml_model_insights=q6,
        q7_shap_lime_revelations=q7,
        q8_unusual_observations=q8,
        q9_conflicting_findings=q9,
        q10_missing_information=q10,
        executive_narrative=exec_narr,
        recommendations=recs,
    )
