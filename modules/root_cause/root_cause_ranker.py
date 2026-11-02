"""Root cause candidate ranking engine and cross-method evidence matrix builder."""

from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np

from modules.root_cause.evidence_collector import CandidateEvidence
from modules.root_cause.evidence_scoring import CandidateScoreDetail


@dataclass
class RankedRootCauseItem:
    """Consolidated ranked root-cause candidate with full multi-engine traceability."""
    rank: int
    feature: str
    feature_type: str
    overall_score: float
    evidence_strength: str
    shift_direction: str
    absolute_shift: float
    relative_shift_pct: Optional[float]
    baseline_display: str
    comparison_display: str
    stat_significant: bool
    stat_p_value: Optional[float]
    effect_size: Optional[float]
    ml_importance_rank: Optional[int]
    shap_rank: Optional[int]
    lime_condition: Optional[str]
    supporting_sources: List[str]
    conflicting_sources: List[str]
    ranking_rationale: str
    confidence_limitations: List[str]
    method_statuses: Dict[str, str]  # 'Supporting', 'Opposing', 'Neutral', 'Not Available'

    def to_dict(self) -> Dict[str, Any]:
        """Convert ranked item to dictionary."""
        d = asdict(self)
        d["overall_score"] = round(self.overall_score, 4)
        d["absolute_shift"] = round(self.absolute_shift, 4)
        if self.relative_shift_pct is not None:
            d["relative_shift_pct"] = round(self.relative_shift_pct, 2)
        if self.stat_p_value is not None:
            d["stat_p_value"] = float(np.round(self.stat_p_value, 5))
        if self.effect_size is not None:
            d["effect_size"] = float(np.round(self.effect_size, 4))
        return d


@dataclass
class RootCauseRankingResult:
    """Master output container for root cause investigation rankings and evidence matrices."""
    ranked_candidates: List[RankedRootCauseItem]
    ranking_df: pd.DataFrame
    evidence_matrix_df: pd.DataFrame
    top_candidate_name: str
    disclaimer: str = (
        "Methodological Notice: All scores represent empirical associations and multi-engine predictive support. "
        "They do NOT establish verified causal mechanisms, which cannot be proven from observational data alone. "
        "Findings should be treated as high-priority contributing factors for further validation."
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert result container to dictionary."""
        return {
            "top_candidate": self.top_candidate_name,
            "disclaimer": self.disclaimer,
            "candidates": [c.to_dict() for c in self.ranked_candidates],
            "evidence_matrix": self.evidence_matrix_df.to_dict(orient="records"),
        }


class RootCauseRanker:
    """Ranks contributing factors and generates structured evidence matrices."""

    def rank_candidates(
        self,
        evidence_list: List[CandidateEvidence],
        score_details: List[CandidateScoreDetail],
    ) -> RootCauseRankingResult:
        """Combine evidence and score breakdowns to rank all candidates in descending score order."""
        evidence_map = {e.feature: e for e in evidence_list}
        # Sort score details descending
        sorted_scores = sorted(score_details, key=lambda s: s.overall_score, reverse=True)

        ranked_items: List[RankedRootCauseItem] = []
        matrix_rows: List[Dict[str, Any]] = []

        for rank, s in enumerate(sorted_scores, start=1):
            e = evidence_map.get(s.feature)
            if not e:
                continue

            # Compute method statuses for the evidence matrix
            m_stat = "Not Available"
            if e.sources_available.get("statistical_test"):
                m_stat = "Supporting" if e.is_stat_significant else "Neutral"

            m_ml = "Not Available"
            if e.sources_available.get("ml_importance"):
                m_ml = "Supporting" if (e.ml_importance_rank or 999) <= 5 else "Neutral"

            m_shap = "Not Available"
            if e.sources_available.get("shap_global") or e.sources_available.get("shap_local"):
                if e.conflicting_sources and any("SHAP" in c for c in e.conflicting_sources):
                    m_shap = "Opposing"
                elif (e.shap_global_rank or 999) <= 5:
                    m_shap = "Supporting"
                else:
                    m_shap = "Neutral"

            m_lime = "Not Available"
            if e.sources_available.get("lime_local"):
                m_lime = "Supporting" if e.lime_local_weight and abs(e.lime_local_weight) > 0.05 else "Neutral"

            m_anom = "Not Available"
            if e.sources_available.get("anomaly_detection"):
                m_anom = "Supporting" if e.unusual_record_count > 0 else "Neutral"

            method_statuses = {
                "Statistical": m_stat,
                "ML Importance": m_ml,
                "SHAP (Global/Local)": m_shap,
                "LIME (Local Rules)": m_lime,
                "Anomaly Concentration": m_anom,
            }

            ranked_items.append(
                RankedRootCauseItem(
                    rank=rank,
                    feature=e.feature,
                    feature_type=e.feature_type,
                    overall_score=s.overall_score,
                    evidence_strength=s.evidence_strength,
                    shift_direction=e.shift_direction,
                    absolute_shift=e.absolute_shift,
                    relative_shift_pct=e.relative_shift_pct,
                    baseline_display=e.baseline_val if isinstance(e.baseline_val, str) else f"{e.baseline_val:.2f}",
                    comparison_display=e.comparison_val if isinstance(e.comparison_val, str) else f"{e.comparison_val:.2f}",
                    stat_significant=e.is_stat_significant,
                    stat_p_value=e.p_value,
                    effect_size=e.effect_size,
                    ml_importance_rank=e.ml_importance_rank,
                    shap_rank=e.shap_global_rank,
                    lime_condition=e.lime_condition,
                    supporting_sources=e.supporting_sources,
                    conflicting_sources=e.conflicting_sources,
                    ranking_rationale=s.ranking_rationale,
                    confidence_limitations=s.confidence_limitations,
                    method_statuses=method_statuses,
                )
            )

            matrix_rows.append({
                "Rank": rank,
                "Feature": e.feature,
                "Score": round(s.overall_score, 3),
                "Strength": s.evidence_strength,
                "Statistical": m_stat,
                "ML Importance": m_ml,
                "SHAP": m_shap,
                "LIME": m_lime,
                "Anomaly": m_anom,
            })

        # Build display DataFrames
        ranking_df = pd.DataFrame([
            {
                "Rank": item.rank,
                "Feature": item.feature,
                "Score": round(item.overall_score, 3),
                "Evidence Strength": item.evidence_strength,
                "Observed Shift": f"{item.absolute_shift:+.2f}" + (f" ({item.relative_shift_pct:+.1f}%)" if item.relative_shift_pct is not None else ""),
                "Direction": item.shift_direction,
                "Statistical (p-val)": f"p={item.stat_p_value:.4f}" if item.stat_p_value is not None else "N/A",
                "ML Rank": f"#{item.ml_importance_rank}" if item.ml_importance_rank is not None else "N/A",
                "SHAP Rank": f"#{item.shap_rank}" if item.shap_rank is not None else "N/A",
                "LIME Rule": item.lime_condition if item.lime_condition else "N/A",
            }
            for item in ranked_items
        ])

        evidence_matrix_df = pd.DataFrame(matrix_rows)
        top_name = ranked_items[0].feature if ranked_items else "None"

        return RootCauseRankingResult(
            ranked_candidates=ranked_items,
            ranking_df=ranking_df,
            evidence_matrix_df=evidence_matrix_df,
            top_candidate_name=top_name,
        )

