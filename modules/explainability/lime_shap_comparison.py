"""Multi-method explainability comparison benchmarking SHAP vs LIME local attributions."""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd

from modules.explainability.shap_local import LocalSHAPResult
from modules.explainability.lime_local_analysis import LIMEExplanationResult
from utils.lime_helpers import calculate_jaccard_similarity, compute_rank_correlation


@dataclass
class SHAPLIMEComparisonResult:
    """Side-by-side comparison matrix and agreement metrics between SHAP and LIME."""
    sample_index: int
    comparison_df: pd.DataFrame
    common_top_features: List[str]
    shap_exclusive_features: List[str]
    lime_exclusive_features: List[str]
    jaccard_similarity: float
    rank_correlation: Optional[float]
    direction_agreement_pct: float
    concordance_level: str  # 'High Agreement', 'Moderate Agreement', 'Divergent Perspectives'
    methodological_summary: str
    disclaimer: str = (
        "SHAP and LIME utilize fundamentally distinct mathematical formulations. "
        "SHAP satisfies game-theoretic efficiency across all feature coalitions, whereas LIME optimizes a local surrogate "
        "model in a perturbed sample neighborhood. Agreement between SHAP and LIME indicates robust model-level association, "
        "NOT real-world causality."
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sample_index": self.sample_index,
            "common_top_features": self.common_top_features,
            "shap_exclusive_features": self.shap_exclusive_features,
            "lime_exclusive_features": self.lime_exclusive_features,
            "jaccard_similarity": round(self.jaccard_similarity, 3),
            "rank_correlation": round(self.rank_correlation, 3) if self.rank_correlation is not None else None,
            "direction_agreement_pct": round(self.direction_agreement_pct, 1),
            "concordance_level": self.concordance_level,
            "methodological_summary": self.methodological_summary,
            "records": self.comparison_df.to_dict(orient="records"),
        }


def compare_shap_and_lime(
    shap_result: LocalSHAPResult,
    lime_result: LIMEExplanationResult,
    top_k: int = 8,
) -> SHAPLIMEComparisonResult:
    """Compare local attributions from SHAP and LIME for the identical observation instance."""
    df_shap = shap_result.contributions_df.copy()
    df_lime = lime_result.contributions_df.copy()

    # Get top K features from each method
    top_shap_feats = df_shap.head(top_k)["Feature"].tolist()
    top_lime_feats = df_lime.head(top_k)["Feature"].tolist()

    common_features = [f for f in top_shap_feats if f in top_lime_feats]
    shap_exclusive = [f for f in top_shap_feats if f not in top_lime_feats]
    lime_exclusive = [f for f in top_lime_feats if f not in top_shap_feats]

    jaccard = calculate_jaccard_similarity(top_shap_feats, top_lime_feats)

    # Ranks dictionary
    shap_ranks = {row["Feature"]: idx + 1 for idx, row in df_shap.reset_index(drop=True).iterrows()}
    lime_ranks = {row["Feature"]: idx + 1 for idx, row in df_lime.reset_index(drop=True).iterrows()}
    rank_corr = compute_rank_correlation(shap_ranks, lime_ranks)

    # Build side-by-side comparison rows for all union features
    union_features = list(dict.fromkeys(top_shap_feats + top_lime_feats))
    rows: List[Dict[str, Any]] = []
    concordant_directions = 0
    total_evaluated_directions = 0

    for fname in union_features:
        # SHAP details
        s_row = df_shap[df_shap["Feature"] == fname]
        if not s_row.empty:
            s_val = float(s_row.iloc[0]["SHAP_Value"])
            s_rank = shap_ranks.get(fname, 999)
            s_sign = 1 if s_val > 0 else (-1 if s_val < 0 else 0)
            s_dir = "Higher (+)" if s_val > 0 else "Lower (-)"
        else:
            s_val = np.nan
            s_rank = 999
            s_sign = 0
            s_dir = "Not in Top SHAP"

        # LIME details
        l_row = df_lime[df_lime["Feature"] == fname]
        if not l_row.empty:
            l_val = float(l_row.iloc[0]["LIME_Weight"])
            l_rank = lime_ranks.get(fname, 999)
            l_sign = 1 if l_val > 0 else (-1 if l_val < 0 else 0)
            l_dir = "Higher (+)" if l_val > 0 else "Lower (-)"
            condition_str = str(l_row.iloc[0].get("Condition", fname))
        else:
            l_val = np.nan
            l_rank = 999
            l_sign = 0
            l_dir = "Not in Top LIME"
            condition_str = fname

        # Direction match check
        if not np.isnan(s_val) and not np.isnan(l_val):
            total_evaluated_directions += 1
            if s_sign == l_sign:
                concordant_directions += 1
                agreement_str = "Concordant Direction (Agree)"
            else:
                agreement_str = "Discordant Direction (Disagree)"
        else:
            agreement_str = "Method-Exclusive"

        rows.append({
            "Feature": fname,
            "LIME_Condition": condition_str,
            "SHAP_Rank": s_rank if s_rank != 999 else "-",
            "LIME_Rank": l_rank if l_rank != 999 else "-",
            "SHAP_Value": s_val,
            "LIME_Weight": l_val,
            "SHAP_Direction": s_dir,
            "LIME_Direction": l_dir,
            "Agreement_Status": agreement_str,
        })

    df_comp = pd.DataFrame(rows).sort_values(
        by=["SHAP_Rank", "LIME_Rank"],
        ascending=[True, True],
    ).reset_index(drop=True)

    dir_pct = (concordant_directions / total_evaluated_directions * 100.0) if total_evaluated_directions > 0 else 0.0

    if jaccard >= 0.6 and dir_pct >= 75.0:
        concordance = "High Agreement"
    elif jaccard >= 0.35:
        concordance = "Moderate Agreement"
    else:
        concordance = "Divergent Perspectives"

    summary = (
        f"SHAP and LIME share **{len(common_features)} of the top {top_k}** most influential features "
        f"(Jaccard overlap = {jaccard:.1%}). Among shared features, **{dir_pct:.0f}%** exhibit matching directional shifts, "
        f"reflecting **{concordance}** between cooperative game-theoretic attribution and local surrogate fitting."
    )

    return SHAPLIMEComparisonResult(
        sample_index=shap_result.sample_index,
        comparison_df=df_comp,
        common_top_features=common_features,
        shap_exclusive_features=shap_exclusive,
        lime_exclusive_features=lime_exclusive,
        jaccard_similarity=jaccard,
        rank_correlation=rank_corr,
        direction_agreement_pct=dir_pct,
        concordance_level=concordance,
        methodological_summary=summary,
    )
