"""Master orchestrator and investigation coordinator facade for Root Cause Engine."""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple, Union
import uuid
import pandas as pd

from modules.root_cause.investigation_history import (
    InvestigationMetadata,
    InvestigationHistory,
    compute_dataset_fingerprint,
    check_dataset_staleness,
)
from modules.root_cause.candidate_generator import (
    OutcomeDefinition,
    RootCauseCandidate,
    extract_outcome_definition,
    generate_root_cause_candidates,
)
from modules.root_cause.evidence_collector import (
    CandidateEvidence,
    EvidenceCollector,
)
from modules.root_cause.evidence_normalizer import (
    NormalizedEvidenceScores,
    EvidenceNormalizer,
)
from modules.root_cause.evidence_scoring import (
    CandidateScoreDetail,
    EvidenceScoringConfig,
    EvidenceScoringEngine,
)
from modules.root_cause.root_cause_ranker import (
    RankedRootCauseItem,
    RootCauseRankingResult,
    RootCauseRanker,
)
from modules.root_cause.segment_contribution import (
    SegmentContributionItem,
    SegmentAnalysisResult,
    analyze_feature_segment_contributions,
)
from modules.root_cause.interaction_analysis import (
    CombinedFactorItem,
    InteractionAnalysisResult,
    analyze_feature_interactions,
)
from modules.root_cause.investigation_summary import (
    InvestigationSummaryReport,
    generate_investigation_summary,
)
from modules.root_cause.investigation_export import InvestigationExporter


@dataclass
class ActiveInvestigationPackage:
    """Complete bundle of analytical results for an active investigation."""
    metadata: InvestigationMetadata
    outcome_def: OutcomeDefinition
    candidates: List[RootCauseCandidate]
    evidence_list: List[CandidateEvidence]
    ranking_result: RootCauseRankingResult
    summary_report: InvestigationSummaryReport
    segment_result: Optional[SegmentAnalysisResult] = None
    interaction_result: Optional[InteractionAnalysisResult] = None


class RootCauseInvestigationManager:
    """Master facade coordinating the entire Root Cause Investigation lifecycle."""

    def __init__(self, scoring_config: Optional[EvidenceScoringConfig] = None) -> None:
        self.history = InvestigationHistory()
        self.collector = EvidenceCollector()
        self.normalizer = EvidenceNormalizer()
        self.scoring_engine = EvidenceScoringEngine(config=scoring_config)
        self.ranker = RootCauseRanker()
        self.exporter = InvestigationExporter()
        self._active_package: Optional[ActiveInvestigationPackage] = None

    @property
    def active_package(self) -> Optional[ActiveInvestigationPackage]:
        """Retrieve currently active investigation package."""
        return self._active_package

    def execute_investigation(
        self,
        df: pd.DataFrame,
        target_col: str,
        baseline_mask: pd.Series,
        comparison_mask: pd.Series,
        title: str = "Investigation",
        description: str = "",
        dataset_name: str = "dataset.csv",
        candidate_features: Optional[List[str]] = None,
        target_event: Optional[Any] = None,
        baseline_name: str = "Baseline Cohort",
        comparison_name: str = "Comparison Cohort",
        primary_segment_feature: Optional[str] = None,
        interaction_pair: Optional[Tuple[str, str]] = None,
        ml_importance_res: Optional[Any] = None,
        shap_global_res: Optional[Any] = None,
        shap_local_res: Optional[Any] = None,
        lime_local_res: Optional[Any] = None,
        anomaly_artifacts: Optional[Any] = None,
    ) -> ActiveInvestigationPackage:
        """Run full root-cause investigation pipeline across all analytical dimensions."""
        # 1. Outcome definition
        outcome_def = extract_outcome_definition(
            df=df,
            target_col=target_col,
            baseline_mask=baseline_mask,
            comparison_mask=comparison_mask,
            target_event=target_event,
            baseline_name=baseline_name,
            comparison_name=comparison_name,
        )

        # 2. Metadata
        now_str = datetime.now().isoformat()
        inv_id = f"inv_{uuid.uuid4().hex[:8]}"
        fp = compute_dataset_fingerprint(df, dataset_name)

        metadata = InvestigationMetadata(
            investigation_id=inv_id,
            title=title,
            description=description,
            created_at=now_str,
            updated_at=now_str,
            dataset_name=dataset_name,
            dataset_fingerprint=fp,
            target_variable=target_col,
            outcome_type=outcome_def.outcome_type,
            baseline_definition={"name": baseline_name, "count": int(baseline_mask.sum())},
            comparison_definition={"name": comparison_name, "count": int(comparison_mask.sum())},
            investigation_question=f"Why did '{target_col}' change from '{baseline_name}' to '{comparison_name}'?",
            status="Active",
        )

        # 3. Candidate features
        feats = candidate_features or [c for c in df.columns if c != target_col]
        candidates = generate_root_cause_candidates(
            df=df,
            features=feats,
            baseline_mask=baseline_mask,
            comparison_mask=comparison_mask,
            target_col=target_col,
        )

        # 4. Multi-engine evidence collection
        evidence_list = self.collector.collect_evidence_for_candidates(
            df=df,
            candidates=candidates,
            baseline_mask=baseline_mask,
            comparison_mask=comparison_mask,
            target_col=target_col,
            ml_importance_res=ml_importance_res,
            shap_global_res=shap_global_res,
            shap_local_res=shap_local_res,
            lime_local_res=lime_local_res,
            anomaly_artifacts=anomaly_artifacts,
        )

        # 5. Normalization
        norm_scores = self.normalizer.normalize_candidate_evidence(evidence_list)

        # 6. Multi-dimensional scoring
        score_details = self.scoring_engine.score_candidates(evidence_list, norm_scores)

        # 7. Ranking & Evidence Matrix
        ranking_result = self.ranker.rank_candidates(evidence_list, score_details)

        # 8. Segment decomposition (on top candidate or requested feature)
        seg_feature = primary_segment_feature or ranking_result.top_candidate_name
        segment_result = None
        if seg_feature and seg_feature in df.columns:
            try:
                segment_result = analyze_feature_segment_contributions(
                    df=df,
                    feature=seg_feature,
                    target_col=target_col,
                    baseline_mask=baseline_mask,
                    comparison_mask=comparison_mask,
                    target_event=target_event,
                )
            except Exception:
                segment_result = None

        # 9. Interaction analysis
        interaction_result = None
        if interaction_pair and len(interaction_pair) == 2:
            f1, f2 = interaction_pair
            if f1 in df.columns and f2 in df.columns:
                try:
                    interaction_result = analyze_feature_interactions(
                        df=df,
                        feature1=f1,
                        feature2=f2,
                        target_col=target_col,
                        baseline_mask=baseline_mask,
                        comparison_mask=comparison_mask,
                        target_event=target_event,
                    )
                except Exception:
                    interaction_result = None
        elif len(ranking_result.ranked_candidates) >= 2:
            # Auto-select top 2 candidates for interaction
            f1 = ranking_result.ranked_candidates[0].feature
            f2 = ranking_result.ranked_candidates[1].feature
            if f1 in df.columns and f2 in df.columns:
                try:
                    interaction_result = analyze_feature_interactions(
                        df=df,
                        feature1=f1,
                        feature2=f2,
                        target_col=target_col,
                        baseline_mask=baseline_mask,
                        comparison_mask=comparison_mask,
                        target_event=target_event,
                    )
                except Exception:
                    interaction_result = None

        # 10. Investigation Summary
        summary_report = generate_investigation_summary(
            outcome_def=outcome_def,
            ranking_res=ranking_result,
            segment_res=segment_result,
            interaction_res=interaction_result,
        )

        package = ActiveInvestigationPackage(
            metadata=metadata,
            outcome_def=outcome_def,
            candidates=candidates,
            evidence_list=evidence_list,
            ranking_result=ranking_result,
            summary_report=summary_report,
            segment_result=segment_result,
            interaction_result=interaction_result,
        )

        self._active_package = package
        self.history.register(metadata)

        return package

    def verify_dataset_synchronization(self, current_df: pd.DataFrame, dataset_name: str = "dataset.csv") -> Tuple[bool, str]:
        """Check if active package matches current dataset state."""
        if not self._active_package:
            return False, "No active investigation exists."
        return check_dataset_staleness(
            current_df=current_df,
            stored_fingerprint=self._active_package.metadata.dataset_fingerprint,
            current_filename=dataset_name,
        )
