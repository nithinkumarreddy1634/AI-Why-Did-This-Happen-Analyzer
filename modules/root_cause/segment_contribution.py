"""Segment-level outcome decomposition and contribution share analysis."""

from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd


@dataclass
class SegmentContributionItem:
    """Individual segment breakdown for an investigated feature."""
    segment_label: str
    sample_size_base: int
    sample_size_comp: int
    total_samples: int
    baseline_outcome_rate: float
    comparison_outcome_rate: float
    absolute_change: float
    relative_change_pct: Optional[float]
    contribution_share_pct: float
    is_top_contributor: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert item to dictionary."""
        d = asdict(self)
        d["baseline_outcome_rate"] = round(self.baseline_outcome_rate, 4)
        d["comparison_outcome_rate"] = round(self.comparison_outcome_rate, 4)
        d["absolute_change"] = round(self.absolute_change, 4)
        if self.relative_change_pct is not None:
            d["relative_change_pct"] = round(self.relative_change_pct, 2)
        d["contribution_share_pct"] = round(self.contribution_share_pct, 2)
        return d


@dataclass
class SegmentAnalysisResult:
    """Consolidated segment contribution analysis for a selected feature."""
    feature: str
    feature_type: str
    segments: List[SegmentContributionItem]
    summary_df: pd.DataFrame
    top_contributing_segment: str
    highest_rate_segment: str
    largest_population_segment: str
    methodological_note: str = (
        "Contribution share decomposes the total observed change into the proportion driven by each segment. "
        "A segment can have a high outcome rate but low contribution share if its population size is small."
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert result container to dictionary."""
        return {
            "feature": self.feature,
            "feature_type": self.feature_type,
            "top_contributing_segment": self.top_contributing_segment,
            "highest_rate_segment": self.highest_rate_segment,
            "largest_population_segment": self.largest_population_segment,
            "methodological_note": self.methodological_note,
            "segments": [s.to_dict() for s in self.segments],
        }


def analyze_feature_segment_contributions(
    df: pd.DataFrame,
    feature: str,
    target_col: str,
    baseline_mask: pd.Series,
    comparison_mask: pd.Series,
    target_event: Optional[Any] = None,
    num_bins: int = 4,
) -> SegmentAnalysisResult:
    """Decompose outcome delta across segments of the investigated feature."""
    if feature not in df.columns or target_col not in df.columns:
        raise ValueError(f"Feature '{feature}' or target '{target_col}' not present in dataframe.")

    is_feat_num = pd.api.types.is_numeric_dtype(df[feature])
    is_target_num = pd.api.types.is_numeric_dtype(df[target_col])

    # Discretize feature if numeric
    if is_feat_num:
        clean_s = df[feature].dropna()
        if len(clean_s.unique()) <= num_bins:
            seg_series = df[feature].astype(str)
        else:
            try:
                seg_series = pd.qcut(df[feature], q=num_bins, duplicates="drop").astype(str)
            except Exception:
                seg_series = pd.cut(df[feature], bins=num_bins).astype(str)
        feat_type = "numerical"
    else:
        # Categorical: keep top 6 categories, group remainder into 'Other'
        top_cats = df[feature].value_counts().head(6).index.tolist()
        seg_series = df[feature].apply(lambda x: str(x) if x in top_cats else "Other")
        feat_type = "categorical"

    unique_segments = [s for s in seg_series.unique() if pd.notnull(s) and s != "nan"]

    # Compute segment deltas
    items: List[SegmentContributionItem] = []
    weighted_deltas: Dict[str, float] = {}

    for seg in unique_segments:
        seg_mask = (seg_series == seg)
        b_sub = df.loc[baseline_mask & seg_mask, target_col].dropna()
        c_sub = df.loc[comparison_mask & seg_mask, target_col].dropna()

        nb = len(b_sub)
        nc = len(c_sub)

        if is_target_num:
            b_rate = float(b_sub.mean()) if nb > 0 else 0.0
            c_rate = float(c_sub.mean()) if nc > 0 else 0.0
        else:
            event = target_event if target_event is not None else df[target_col].dropna().unique()[0]
            b_rate = float((b_sub == event).mean()) if nb > 0 else 0.0
            c_rate = float((c_sub == event).mean()) if nc > 0 else 0.0

        abs_chg = c_rate - b_rate
        rel_pct = ((abs_chg / abs(b_rate)) * 100.0) if b_rate > 0 else None

        # Weighted impact = nc * delta
        w_delta = nc * abs_chg
        weighted_deltas[seg] = w_delta

        items.append(
            SegmentContributionItem(
                segment_label=seg,
                sample_size_base=nb,
                sample_size_comp=nc,
                total_samples=nb + nc,
                baseline_outcome_rate=b_rate,
                comparison_outcome_rate=c_rate,
                absolute_change=abs_chg,
                relative_change_pct=rel_pct,
                contribution_share_pct=0.0,  # Computed below
            )
        )

    # Compute contribution shares
    total_positive_w = sum(v for v in weighted_deltas.values() if v > 0)
    for item in items:
        w = weighted_deltas.get(item.segment_label, 0.0)
        if total_positive_w > 0 and w > 0:
            item.contribution_share_pct = (w / total_positive_w) * 100.0
        else:
            item.contribution_share_pct = 0.0

    # Sort descending by contribution share
    items.sort(key=lambda x: x.contribution_share_pct, reverse=True)
    if items:
        items[0].is_top_contributor = True

    top_contrib = items[0].segment_label if items else "None"
    highest_rate = max(items, key=lambda x: x.comparison_outcome_rate).segment_label if items else "None"
    largest_pop = max(items, key=lambda x: x.sample_size_comp).segment_label if items else "None"

    summary_df = pd.DataFrame([
        {
            "Segment": item.segment_label,
            "Comp Samples": item.sample_size_comp,
            "Base Rate": f"{item.baseline_outcome_rate:.3f}",
            "Comp Rate": f"{item.comparison_outcome_rate:.3f}",
            "Change (Δ)": f"{item.absolute_change:+.3f}",
            "Rel Change (%)": f"{item.relative_change_pct:+.1f}%" if item.relative_change_pct is not None else "N/A",
            "Contribution Share": f"{item.contribution_share_pct:.1f}%",
        }
        for item in items
    ])

    return SegmentAnalysisResult(
        feature=feature,
        feature_type=feat_type,
        segments=items,
        summary_df=summary_df,
        top_contributing_segment=top_contrib,
        highest_rate_segment=highest_rate,
        largest_population_segment=largest_pop,
    )

