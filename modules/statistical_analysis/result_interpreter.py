"""Statistical result interpretation and natural language narrative synthesis."""

from typing import Dict, Any, List, Optional
from modules.statistical_analysis.hypothesis_testing import HypothesisTestResult


class StatisticalResultInterpreter:
    """Generates structured, non-causal natural language explanations from hypothesis test outputs."""

    @staticmethod
    def interpret_test(result: HypothesisTestResult) -> Dict[str, Any]:
        """Produce structured analytical narrative explaining test findings."""
        sig_str = (
            f"statistically significant (p = {result.p_value:.4f} < {result.alpha})"
            if result.is_significant
            else f"not statistically significant (p = {result.p_value:.4f} ≥ {result.alpha})"
        )

        # Decision Narrative
        decision_narrative = (
            f"Based on the observed evidence at significance level α = {result.alpha}, we "
            f"**{result.decision.lower()}**. There is sufficient statistical evidence supporting the alternative hypothesis."
            if result.is_significant
            else (
                f"At significance level α = {result.alpha}, we **{result.decision.lower()}**. "
                f"The observed variation is consistent with random sampling variability under the null hypothesis."
            )
        )

        # Practical Effect Narrative
        if result.effect_size is not None and result.effect_size_name:
            effect_narrative = (
                f"The measured magnitude of effect is **{result.effect_size:.3f}** "
                f"({result.effect_size_name}), categorized as a **{result.effect_size_interpretation or 'calculated magnitude'}**. "
                f"While statistical significance addresses whether an effect exists, the effect size indicates whether it is practically noticeable."
            )
        else:
            effect_narrative = "Effect size estimation was not calculated for this test variant."

        # Confidence Interval Narrative
        if result.ci_lower is not None and result.ci_upper is not None:
            ci_pct = int((result.confidence_level or 0.95) * 100)
            ci_narrative = (
                f"We are {ci_pct}% confident that the true population parameter lies between "
                f"**{result.ci_lower:+.3f}** and **{result.ci_upper:+.3f}**."
            )
        else:
            ci_narrative = "Confidence interval estimation is not applicable for this omnibus test statistic."

        # Standard Cautionary Limitations
        non_causality_notice = (
            "⚠️ **Causality Caveat:** Statistical testing identifies significant associations, mean differences, or rank shifts in observational data. "
            "It does NOT prove that the investigated variable caused the observed change, as unobserved confounding variables or selection bias may be present."
        )

        return {
            "title": f"Interpretation: {result.test_name}",
            "research_question": result.research_question,
            "statistical_decision": result.decision,
            "decision_narrative": decision_narrative,
            "effect_narrative": effect_narrative,
            "ci_narrative": ci_narrative,
            "plain_language_summary": result.interpretation,
            "non_causality_notice": non_causality_notice,
            "limitations": result.limitations,
        }

