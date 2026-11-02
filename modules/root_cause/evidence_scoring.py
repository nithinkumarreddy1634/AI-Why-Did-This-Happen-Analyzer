"""Configurable multi-dimensional evidence scoring engine with dynamic weight redistribution."""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
import numpy as np

from modules.root_cause.evidence_collector import CandidateEvidence
from modules.root_cause.evidence_normalizer import NormalizedEvidenceScores


@dataclass
class EvidenceScoringConfig:
    """Configurable weights and thresholds for multi-dimensional root cause ranking."""
    weight_stat: float = 0.25
    weight_magnitude: float = 0.25
    weight_ml: float = 0.15
    weight_shap: float = 0.15
    weight_lime: float = 0.10
    weight_anomaly: float = 0.10
    dampen_correlated_ml_shap: bool = True


@dataclass
class CandidateScoreDetail:
    """Detailed score breakdown and evidence strength category for a candidate."""
    feature: str
    overall_score: float
    evidence_strength: str  # 'Strong Supporting Evidence', 'Moderate Supporting Evidence', etc.
    active_sources_count: int
    dimensional_contributions: Dict[str, float]
    normalized_weights_applied: Dict[str, float]
    ranking_rationale: str
    confidence_limitations: List[str]

    def to_dict(self) -> Dict[str, Any]:
        """Convert score detail to dictionary."""
        return {
            "feature": self.feature,
            "overall_score": round(self.overall_score, 4),
            "evidence_strength": self.evidence_strength,
            "active_sources_count": self.active_sources_count,
            "dimensional_contributions": {k: round(v, 4) for k, v in self.dimensional_contributions.items()},
            "normalized_weights_applied": {k: round(v, 4) for k, v in self.normalized_weights_applied.items()},
            "ranking_rationale": self.ranking_rationale,
            "confidence_limitations": self.confidence_limitations,
        }


class EvidenceScoringEngine:
    """Computes transparent, normalized evidence scores with dynamic weight redistribution."""

    def __init__(self, config: Optional[EvidenceScoringConfig] = None) -> None:
        self.config = config or EvidenceScoringConfig()

    def score_candidates(
        self,
        evidence_list: List[CandidateEvidence],
        normalized_scores: List[NormalizedEvidenceScores],
    ) -> List[CandidateScoreDetail]:
        """Score all candidates with dynamic redistribution over active sources."""
        norm_map = {n.feature: n for n in normalized_scores}
        scored_details: List[CandidateScoreDetail] = []

        for e in evidence_list:
            n = norm_map.get(e.feature)
            if not n:
                continue

            # Determine available evidence dimensions
            active_weights: Dict[str, float] = {}
            scores_by_dim: Dict[str, float] = {}

            # Always active: Observed Magnitude
            active_weights["magnitude"] = self.config.weight_magnitude
            scores_by_dim["magnitude"] = n.magnitude_score

            if e.sources_available.get("statistical_test"):
                active_weights["statistical"] = self.config.weight_stat
                scores_by_dim["statistical"] = n.stat_score

            if e.sources_available.get("ml_importance"):
                active_weights["ml"] = self.config.weight_ml
                scores_by_dim["ml"] = n.ml_score

            if e.sources_available.get("shap_global") or e.sources_available.get("shap_local"):
                active_weights["shap"] = self.config.weight_shap
                scores_by_dim["shap"] = n.shap_score

            if e.sources_available.get("lime_local"):
                active_weights["lime"] = self.config.weight_lime
                scores_by_dim["lime"] = n.lime_score

            if e.sources_available.get("anomaly_detection"):
                active_weights["anomaly"] = self.config.weight_anomaly
                scores_by_dim["anomaly"] = n.anomaly_score

            # Anti-double-counting dampener: If both ML importance and SHAP are present
            if self.config.dampen_correlated_ml_shap and "ml" in active_weights and "shap" in active_weights:
                active_weights["ml"] *= 0.8
                active_weights["shap"] *= 0.8

            # Dynamic renormalization of weights to sum to 1.0
            total_active_w = sum(active_weights.values())
            if total_active_w > 0:
                norm_w = {k: v / total_active_w for k, v in active_weights.items()}
            else:
                norm_w = {"magnitude": 1.0}

            # Weighted sum
            weighted_sum = sum(scores_by_dim[k] * norm_w[k] for k in norm_w)

            # Apply consistency penalty for directional conflicts
            overall_score = float(np.clip(weighted_sum * n.consistency_score, 0.0, 1.0))

            # Determine strength label and rationale
            strength, rationale, limitations = self._evaluate_strength_and_rationale(
                score=overall_score,
                evidence=e,
                active_dims=list(active_weights.keys()),
                norm_scores=n,
            )

            dim_contribs = {k: scores_by_dim[k] * norm_w[k] for k in norm_w}

            scored_details.append(
                CandidateScoreDetail(
                    feature=e.feature,
                    overall_score=overall_score,
                    evidence_strength=strength,
                    active_sources_count=len(active_weights),
                    dimensional_contributions=dim_contribs,
                    normalized_weights_applied=norm_w,
                    ranking_rationale=rationale,
                    confidence_limitations=limitations,
                )
            )

        return scored_details

    def _evaluate_strength_and_rationale(
        self,
        score: float,
        evidence: CandidateEvidence,
        active_dims: List[str],
        norm_scores: NormalizedEvidenceScores,
    ) -> tuple[str, str, List[str]]:
        """Assign standardized evidence strength, explanation rationale, and limitation warnings."""
        limitations: List[str] = [
            "Observational finding only; real-world causality cannot be confirmed without controlled interventions."
        ]

        if len(evidence.conflicting_sources) > 0:
            for conf in evidence.conflicting_sources:
                limitations.append(f"Conflict detected: {conf}.")
            strength = "Conflicting Evidence"
            rationale = (
                f"Candidate exhibited divergent signals across analytical engines. "
                f"While shift magnitude was {evidence.absolute_shift:+.2f}, directional discordance was detected."
            )
            return strength, rationale, limitations

        if len(active_dims) <= 1:
            limitations.append("Only bivariate observed shift was available; advanced statistical/ML engines unexecuted.")
            strength = "Insufficient Evidence" if score < 0.3 else "Limited Supporting Evidence"
            rationale = f"Evaluated solely on observed shift (Δ = {evidence.absolute_shift:+.2f})."
            return strength, rationale, limitations

        # Determine strength
        if score >= 0.70 and len(evidence.supporting_sources) >= 2:
            strength = "Strong Supporting Evidence"
            rationale = (
                f"High consensus across {len(evidence.supporting_sources)} methods: "
                + ", ".join(evidence.supporting_sources[:3])
                + (f" with significant shift (Δ = {evidence.absolute_shift:+.2f})." if evidence.is_stat_significant else ".")
            )
        elif score >= 0.45 and len(evidence.supporting_sources) >= 2:
            strength = "Moderate Supporting Evidence"
            rationale = (
                f"Supported by {len(evidence.supporting_sources)} sources ({', '.join(evidence.supporting_sources[:2])}) "
                f"with observed shift Δ = {evidence.absolute_shift:+.2f}."
            )
        elif score >= 0.25:
            strength = "Limited Supporting Evidence"
            rationale = (
                f"Modest shift (Δ = {evidence.absolute_shift:+.2f}) with limited corroboration across analytical methods."
            )
        else:
            strength = "Insufficient Evidence"
            rationale = "Candidate exhibited minimal observed divergence and weak model/statistical importance."

        return strength, rationale, limitations

