"""Export manager generating CSV, Markdown, and JSON artifacts for root cause investigations."""

import json
from typing import Dict, Any, Optional
import pandas as pd

from modules.root_cause.investigation_history import InvestigationMetadata
from modules.root_cause.candidate_generator import OutcomeDefinition
from modules.root_cause.root_cause_ranker import RootCauseRankingResult
from modules.root_cause.segment_contribution import SegmentAnalysisResult
from modules.root_cause.interaction_analysis import InteractionAnalysisResult
from modules.root_cause.investigation_summary import InvestigationSummaryReport


class InvestigationExporter:
    """Formats and exports investigation artifacts for download and archiving."""

    @staticmethod
    def export_ranking_csv(ranking_result: RootCauseRankingResult) -> str:
        """Export candidate rankings to CSV string."""
        return ranking_result.ranking_df.to_csv(index=False)

    @staticmethod
    def export_evidence_matrix_csv(ranking_result: RootCauseRankingResult) -> str:
        """Export multi-engine evidence matrix to CSV string."""
        return ranking_result.evidence_matrix_df.to_csv(index=False)

    @staticmethod
    def export_segment_analysis_csv(segment_result: Optional[SegmentAnalysisResult]) -> str:
        """Export segment contribution table to CSV string."""
        if segment_result is None or segment_result.summary_df.empty:
            return "Segment,Comp Samples,Base Rate,Comp Rate,Change (Δ),Contribution Share\n"
        return segment_result.summary_df.to_csv(index=False)

    @staticmethod
    def export_summary_markdown(summary_report: InvestigationSummaryReport) -> str:
        """Export comprehensive narrative report to Markdown text."""
        return summary_report.to_markdown()

    @staticmethod
    def export_complete_json(
        metadata: InvestigationMetadata,
        outcome_def: OutcomeDefinition,
        ranking_result: RootCauseRankingResult,
        summary_report: InvestigationSummaryReport,
        segment_result: Optional[SegmentAnalysisResult] = None,
        interaction_result: Optional[InteractionAnalysisResult] = None,
    ) -> str:
        """Export complete investigation data bundle to a structured JSON string."""
        bundle: Dict[str, Any] = {
            "metadata": metadata.to_dict(),
            "outcome_definition": outcome_def.to_dict(),
            "ranking_summary": {
                "top_candidate": ranking_result.top_candidate_name,
                "disclaimer": ranking_result.disclaimer,
                "candidates": [c.to_dict() for c in ranking_result.ranked_candidates],
                "evidence_matrix": ranking_result.evidence_matrix_df.to_dict(orient="records"),
            },
            "investigation_summary": summary_report.to_dict(),
            "segment_analysis": segment_result.to_dict() if segment_result else None,
            "interaction_analysis": interaction_result.to_dict() if interaction_result else None,
        }
        return json.dumps(bundle, indent=2, default=str)

