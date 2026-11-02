"""Unit and integration tests for Phase 11: Root Cause Investigation Engine."""

import pytest
import numpy as np
import pandas as pd

from modules.root_cause.investigation_history import (
    InvestigationMetadata,
    InvestigationHistory,
    compute_dataset_fingerprint,
    check_dataset_staleness,
)
from modules.root_cause.candidate_generator import (
    extract_outcome_definition,
    generate_root_cause_candidates,
    OutcomeDefinition,
)
from modules.root_cause.evidence_collector import (
    EvidenceCollector,
    CandidateEvidence,
)
from modules.root_cause.evidence_normalizer import (
    EvidenceNormalizer,
    NormalizedEvidenceScores,
)
from modules.root_cause.evidence_scoring import (
    EvidenceScoringEngine,
    EvidenceScoringConfig,
)
from modules.root_cause.root_cause_ranker import (
    RootCauseRanker,
    RootCauseRankingResult,
)
from modules.root_cause.segment_contribution import (
    analyze_feature_segment_contributions,
)
from modules.root_cause.interaction_analysis import (
    analyze_feature_interactions,
)
from modules.root_cause.investigation_summary import (
    generate_investigation_summary,
)
from modules.root_cause.investigation_export import (
    InvestigationExporter,
)
from modules.root_cause.investigation_manager import (
    RootCauseInvestigationManager,
)


@pytest.fixture
def synthetic_churn_df():
    """Synthetic churn dataset with known causal patterns and group shifts."""
    np.random.seed(42)
    n = 120
    contract = np.array(["Month-to-month"] * 60 + ["Two year"] * 60)
    # Month-to-month has higher tenure drops and higher monthly charges
    tenure = np.concatenate([
        np.random.normal(10, 3, 60),   # Month-to-month low tenure
        np.random.normal(50, 10, 60),  # Two-year high tenure
    ])
    monthly_charges = np.concatenate([
        np.random.normal(85, 12, 60),
        np.random.normal(45, 10, 60),
    ])
    tech_support = np.array(["No"] * 45 + ["Yes"] * 15 + ["No"] * 10 + ["Yes"] * 50)
    # Target churn is much higher in Month-to-month (80% vs 10%)
    churn = np.concatenate([
        np.random.binomial(1, 0.75, 60),
        np.random.binomial(1, 0.10, 60),
    ])

    return pd.DataFrame({
        "contract": contract,
        "tenure": np.clip(tenure, 1, 72),
        "monthly_charges": np.clip(monthly_charges, 18, 120),
        "tech_support": tech_support,
        "churn": churn,
    })


def test_fingerprinting_and_staleness(synthetic_churn_df):
    df = synthetic_churn_df
    fp1 = compute_dataset_fingerprint(df, "churn.csv")
    assert isinstance(fp1, str)
    assert len(fp1) == 16

    # Verify identical df produces identical fingerprint
    fp2 = compute_dataset_fingerprint(df.copy(), "churn.csv")
    assert fp1 == fp2

    # Check staleness on modified df
    is_stale, _ = check_dataset_staleness(df, fp1, "churn.csv")
    assert is_stale is False

    modified_df = df.iloc[:-5]
    is_stale_mod, reason = check_dataset_staleness(modified_df, fp1, "churn.csv")
    assert is_stale_mod is True
    assert "fingerprint changed" in reason.lower()


def test_outcome_definition_and_zero_baseline(synthetic_churn_df):
    df = synthetic_churn_df
    base_mask = (df["contract"] == "Two year")
    comp_mask = (df["contract"] == "Month-to-month")

    outcome = extract_outcome_definition(
        df=df,
        target_col="churn",
        baseline_mask=base_mask,
        comparison_mask=comp_mask,
        baseline_name="Two-Year Contracts",
        comparison_name="Month-to-Month Contracts",
    )

    assert isinstance(outcome, OutcomeDefinition)
    assert outcome.target_variable == "churn"
    assert outcome.outcome_type == "categorical"
    assert outcome.comparison_value > outcome.baseline_value
    assert outcome.absolute_change > 0.0
    assert outcome.relative_change_pct is not None

    # Test numerical zero baseline protection
    df_zero = pd.DataFrame({"target": [0.0, 0.0, 5.0, 10.0], "group": ["A", "A", "B", "B"]})
    out_zero = extract_outcome_definition(
        df=df_zero,
        target_col="target",
        baseline_mask=(df_zero["group"] == "A"),
        comparison_mask=(df_zero["group"] == "B"),
    )
    assert out_zero.baseline_value == 0.0
    assert out_zero.relative_change_pct is None  # Safe handling of zero division


def test_candidate_generation_and_evidence_collection(synthetic_churn_df):
    df = synthetic_churn_df
    base_mask = (df["contract"] == "Two year")
    comp_mask = (df["contract"] == "Month-to-month")

    candidates = generate_root_cause_candidates(
        df=df,
        features=["tenure", "monthly_charges", "tech_support"],
        baseline_mask=base_mask,
        comparison_mask=comp_mask,
        target_col="churn",
    )

    assert len(candidates) == 3
    tenure_cand = next(c for c in candidates if c.feature == "tenure")
    assert tenure_cand.shift_direction == "Decreased (-)"
    assert tenure_cand.absolute_shift < 0.0

    charges_cand = next(c for c in candidates if c.feature == "monthly_charges")
    assert charges_cand.shift_direction == "Increased (+)"

    # Evidence collector
    collector = EvidenceCollector(alpha=0.05)
    evidence_list = collector.collect_evidence_for_candidates(
        df=df,
        candidates=candidates,
        baseline_mask=base_mask,
        comparison_mask=comp_mask,
        target_col="churn",
    )

    assert len(evidence_list) == 3
    for ev in evidence_list:
        assert ev.stat_test_name is not None
        assert ev.p_value is not None
        assert ev.is_stat_significant is True  # All 3 have huge deliberate separation


def test_evidence_scoring_and_ranking(synthetic_churn_df):
    df = synthetic_churn_df
    base_mask = (df["contract"] == "Two year")
    comp_mask = (df["contract"] == "Month-to-month")

    candidates = generate_root_cause_candidates(
        df=df,
        features=["tenure", "monthly_charges", "tech_support"],
        baseline_mask=base_mask,
        comparison_mask=comp_mask,
        target_col="churn",
    )

    collector = EvidenceCollector()
    evidence_list = collector.collect_evidence_for_candidates(
        df=df,
        candidates=candidates,
        baseline_mask=base_mask,
        comparison_mask=comp_mask,
        target_col="churn",
    )

    normalizer = EvidenceNormalizer()
    norm_scores = normalizer.normalize_candidate_evidence(evidence_list)
    assert len(norm_scores) == 3
    for ns in norm_scores:
        assert 0.0 <= ns.stat_score <= 1.0
        assert 0.0 <= ns.magnitude_score <= 1.0

    scoring_engine = EvidenceScoringEngine()
    scored_details = scoring_engine.score_candidates(evidence_list, norm_scores)
    assert len(scored_details) == 3

    ranker = RootCauseRanker()
    ranked_res = ranker.rank_candidates(evidence_list, scored_details)
    assert isinstance(ranked_res, RootCauseRankingResult)
    assert len(ranked_res.ranked_candidates) == 3
    assert ranked_res.ranked_candidates[0].rank == 1
    assert ranked_res.ranked_candidates[0].overall_score >= ranked_res.ranked_candidates[1].overall_score
    assert not ranked_res.ranking_df.empty
    assert not ranked_res.evidence_matrix_df.empty
    assert "NOT establish verified causal" in ranked_res.disclaimer


def test_segment_contribution_analysis(synthetic_churn_df):
    df = synthetic_churn_df
    base_mask = (df["contract"] == "Two year")
    comp_mask = (df["contract"] == "Month-to-month")

    seg_res = analyze_feature_segment_contributions(
        df=df,
        feature="tenure",
        target_col="churn",
        baseline_mask=base_mask,
        comparison_mask=comp_mask,
        num_bins=3,
    )

    assert seg_res.feature == "tenure"
    assert len(seg_res.segments) > 0
    # Total contribution shares should sum approximately to 100%
    total_share = sum(s.contribution_share_pct for s in seg_res.segments)
    assert pytest.approx(total_share, 1.0) == 100.0 or total_share == 0.0
    assert seg_res.top_contributing_segment is not None
    assert not seg_res.summary_df.empty


def test_interaction_analysis(synthetic_churn_df):
    df = synthetic_churn_df
    base_mask = (df["contract"] == "Two year")
    comp_mask = (df["contract"] == "Month-to-month")

    inter_res = analyze_feature_interactions(
        df=df,
        feature1="tenure",
        feature2="monthly_charges",
        target_col="churn",
        baseline_mask=base_mask,
        comparison_mask=comp_mask,
        max_bins=2,
    )

    assert inter_res.feature1 == "tenure"
    assert inter_res.feature2 == "monthly_charges"
    assert len(inter_res.combinations) > 0
    assert not inter_res.matrix_df.empty
    assert "highest_risk_combination" in inter_res.to_dict()


def test_end_to_end_investigation_manager_workflow(synthetic_churn_df):
    df = synthetic_churn_df
    base_mask = (df["contract"] == "Two year")
    comp_mask = (df["contract"] == "Month-to-month")

    manager = RootCauseInvestigationManager()
    pkg = manager.execute_investigation(
        df=df,
        target_col="churn",
        baseline_mask=base_mask,
        comparison_mask=comp_mask,
        title="Telecom Customer Churn Surge",
        dataset_name="synthetic_churn.csv",
        candidate_features=["tenure", "monthly_charges", "tech_support"],
        baseline_name="Two-Year Contracts",
        comparison_name="Month-to-Month Contracts",
    )

    assert pkg.metadata.title == "Telecom Customer Churn Surge"
    assert pkg.ranking_result.top_candidate_name in ["tenure", "monthly_charges", "tech_support"]
    assert len(pkg.summary_report.recommendations) >= 3

    # Exporters test
    exporter = manager.exporter
    csv_rank = exporter.export_ranking_csv(pkg.ranking_result)
    assert "Candidate Feature" in csv_rank or "Feature" in csv_rank

    csv_matrix = exporter.export_evidence_matrix_csv(pkg.ranking_result)
    assert "Statistical" in csv_matrix

    json_bundle = exporter.export_complete_json(
        metadata=pkg.metadata,
        outcome_def=pkg.outcome_def,
        ranking_result=pkg.ranking_result,
        summary_report=pkg.summary_report,
        segment_result=pkg.segment_result,
        interaction_result=pkg.interaction_result,
    )
    assert "metadata" in json_bundle
    assert "investigation_summary" in json_bundle

