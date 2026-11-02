"""Standardization and dimensional normalization for heterogeneous analytical evidence."""

from dataclasses import dataclass
from typing import Dict, Any, List, Optional
import numpy as np

from modules.root_cause.evidence_collector import CandidateEvidence


@dataclass
class NormalizedEvidenceScores:
    """Normalized dimension scores bounded in [0.0, 1.0] for evidence aggregation."""
    feature: str
    stat_score: float = 0.0
    magnitude_score: float = 0.0
    ml_score: float = 0.0
    shap_score: float = 0.0
    lime_score: float = 0.0
    anomaly_score: float = 0.0
    consistency_score: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert normalized scores to dictionary."""
        return {
            "feature": self.feature,
            "stat_score": round(self.stat_score, 4),
            "magnitude_score": round(self.magnitude_score, 4),
            "ml_score": round(self.ml_score, 4),
            "shap_score": round(self.shap_score, 4),
            "lime_score": round(self.lime_score, 4),
            "anomaly_score": round(self.anomaly_score, 4),
            "consistency_score": round(self.consistency_score, 4),
        }


class EvidenceNormalizer:
    """Normalizes heterogeneous analytical measurements onto comparable [0.0, 1.0] scales."""

    def normalize_candidate_evidence(
        self,
        evidence_list: List[CandidateEvidence],
        alpha: float = 0.05,
    ) -> List[NormalizedEvidenceScores]:
        """Normalize all candidate evidence profiles relative to the current investigation batch."""
        if not evidence_list:
            return []

        # Find maximum values for relative batch scaling
        max_abs_shift = max((abs(e.absolute_shift) for e in evidence_list), default=1.0)
        max_abs_shift = max_abs_shift if max_abs_shift > 0 else 1.0

        max_ml = max((e.ml_importance_score or 0.0 for e in evidence_list), default=1.0)
        max_ml = max_ml if max_ml > 0 else 1.0

        max_shap = max((abs(e.shap_global_importance or 0.0) for e in evidence_list), default=1.0)
        max_shap = max_shap if max_shap > 0 else 1.0

        max_lime = max((abs(e.lime_local_weight or 0.0) for e in evidence_list), default=1.0)
        max_lime = max_lime if max_lime > 0 else 1.0

        normalized_scores: List[NormalizedEvidenceScores] = []

        for e in evidence_list:
            # 1. Statistical Evidence Score
            stat_score = 0.0
            if e.sources_available.get("statistical_test") and e.p_value is not None:
                p = max(1e-15, float(e.p_value))
                if p < alpha:
                    # p-value strength scaled from 0.5 to 1.0
                    p_strength = min(1.0, 0.5 + 0.1 * min(5.0, -np.log10(p)))
                    # Combine with effect size if present
                    if e.effect_size is not None:
                        eff = abs(float(e.effect_size))
                        # Scale Cohen's d (0.2 small, 0.5 med, 0.8+ large)
                        eff_norm = min(1.0, eff / 0.8)
                        stat_score = 0.5 * p_strength + 0.5 * eff_norm
                    else:
                        stat_score = p_strength
                else:
                    stat_score = max(0.0, 0.2 * (alpha - p) / alpha) if p < alpha else 0.0

            # 2. Magnitude of Observed Change
            mag_score = min(1.0, abs(e.absolute_shift) / max_abs_shift)

            # 3. ML Feature Importance Score
            ml_score = 0.0
            if e.sources_available.get("ml_importance") and e.ml_importance_score is not None:
                ml_score = min(1.0, e.ml_importance_score / max_ml)

            # 4. SHAP Score
            shap_score = 0.0
            if e.sources_available.get("shap_global") and e.shap_global_importance is not None:
                shap_score = min(1.0, abs(e.shap_global_importance) / max_shap)

            # 5. LIME Score (Weighted by local surrogate fidelity)
            lime_score = 0.0
            if e.sources_available.get("lime_local") and e.lime_local_weight is not None:
                raw_lime = min(1.0, abs(e.lime_local_weight) / max_lime)
                fid = e.lime_fidelity_r2 if e.lime_fidelity_r2 is not None else 0.5
                fid_weight = max(0.2, min(1.0, fid))
                lime_score = raw_lime * fid_weight

            # 6. Anomaly Score
            anom_score = 0.0
            if e.sources_available.get("anomaly_detection") and e.anomaly_concentration_pct is not None:
                anom_score = min(1.0, e.anomaly_concentration_pct / 100.0)

            # 7. Evidence Consistency / Conflict Penalty
            cons_score = 1.0
            if len(e.conflicting_sources) > 0:
                cons_score = max(0.3, 1.0 - 0.25 * len(e.conflicting_sources))

            normalized_scores.append(
                NormalizedEvidenceScores(
                    feature=e.feature,
                    stat_score=float(np.clip(stat_score, 0.0, 1.0)),
                    magnitude_score=float(np.clip(mag_score, 0.0, 1.0)),
                    ml_score=float(np.clip(ml_score, 0.0, 1.0)),
                    shap_score=float(np.clip(shap_score, 0.0, 1.0)),
                    lime_score=float(np.clip(lime_score, 0.0, 1.0)),
                    anomaly_score=float(np.clip(anom_score, 0.0, 1.0)),
                    consistency_score=float(np.clip(cons_score, 0.0, 1.0)),
                )
            )

        return normalized_scores

