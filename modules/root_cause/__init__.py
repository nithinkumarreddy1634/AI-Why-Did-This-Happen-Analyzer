"""Root Cause Investigation Engine package (Phase 11)."""

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
from modules.root_cause.investigation_manager import (
    RootCauseInvestigationManager,
    ActiveInvestigationPackage,
)

__all__ = [
    "InvestigationMetadata",
    "InvestigationHistory",
    "compute_dataset_fingerprint",
    "check_dataset_staleness",
    "OutcomeDefinition",
    "RootCauseCandidate",
    "extract_outcome_definition",
    "generate_root_cause_candidates",
    "CandidateEvidence",
    "EvidenceCollector",
    "NormalizedEvidenceScores",
    "EvidenceNormalizer",
    "CandidateScoreDetail",
    "EvidenceScoringConfig",
    "EvidenceScoringEngine",
    "RankedRootCauseItem",
    "RootCauseRankingResult",
    "RootCauseRanker",
    "SegmentContributionItem",
    "SegmentAnalysisResult",
    "analyze_feature_segment_contributions",
    "CombinedFactorItem",
    "InteractionAnalysisResult",
    "analyze_feature_interactions",
    "InvestigationSummaryReport",
    "generate_investigation_summary",
    "InvestigationExporter",
    "RootCauseInvestigationManager",
    "ActiveInvestigationPackage",
]

