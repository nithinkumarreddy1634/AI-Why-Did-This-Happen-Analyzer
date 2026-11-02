"""Bivariate feature combination and interaction investigation engine."""

from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd


@dataclass
class CombinedFactorItem:
    """Outcome rate and delta for a specific feature-pair combination."""
    val1_label: str
    val2_label: str
    sample_size: int
    baseline_rate: float
    comparison_rate: float
    rate_change: float
    is_high_risk: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert combination item to dictionary."""
        d = asdict(self)
        d["baseline_rate"] = round(self.baseline_rate, 4)
        d["comparison_rate"] = round(self.comparison_rate, 4)
        d["rate_change"] = round(self.rate_change, 4)
        return d


@dataclass
class InteractionAnalysisResult:
    """Consolidated interaction and combined factor analysis between two features."""
    feature1: str
    feature2: str
    matrix_df: pd.DataFrame
    combinations: List[CombinedFactorItem]
    highest_risk_combination: str
    max_rate_change: float
    interaction_synergy_note: str
    disclaimer: str = (
        "Statistical Note: Co-occurrence of multiple factors often compounds outcome divergence. "
        "High interaction values reflect empirical non-linear association, NOT proven causal synergy."
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert interaction result to dictionary."""
        return {
            "feature1": self.feature1,
            "feature2": self.feature2,
            "highest_risk_combination": self.highest_risk_combination,
            "max_rate_change": round(self.max_rate_change, 4),
            "interaction_synergy_note": self.interaction_synergy_note,
            "disclaimer": self.disclaimer,
            "combinations": [c.to_dict() for c in self.combinations],
        }


def analyze_feature_interactions(
    df: pd.DataFrame,
    feature1: str,
    feature2: str,
    target_col: str,
    baseline_mask: pd.Series,
    comparison_mask: pd.Series,
    target_event: Optional[Any] = None,
    max_bins: int = 3,
) -> InteractionAnalysisResult:
    """Analyze outcome divergence across combinations of two features."""
    if feature1 not in df.columns or feature2 not in df.columns or target_col not in df.columns:
        raise ValueError(f"Columns '{feature1}', '{feature2}', or '{target_col}' not found.")

    is_target_num = pd.api.types.is_numeric_dtype(df[target_col])

    # Helper to bin series if numeric
    def bin_series(col_name: str) -> pd.Series:
        s = df[col_name]
        if pd.api.types.is_numeric_dtype(s):
            clean = s.dropna()
            if len(clean.unique()) <= max_bins:
                return s.astype(str)
            try:
                return pd.qcut(s, q=max_bins, duplicates="drop").astype(str)
            except Exception:
                return pd.cut(s, bins=max_bins).astype(str)
        else:
            top_cats = s.value_counts().head(4).index.tolist()
            return s.apply(lambda x: str(x) if x in top_cats else "Other")

    s1 = bin_series(feature1)
    s2 = bin_series(feature2)

    unique_v1 = [v for v in s1.unique() if pd.notnull(v) and v != "nan"]
    unique_v2 = [v for v in s2.unique() if pd.notnull(v) and v != "nan"]

    combinations: List[CombinedFactorItem] = []
    matrix_data: Dict[str, Dict[str, float]] = {v1: {} for v1 in unique_v1}

    for v1 in unique_v1:
        for v2 in unique_v2:
            pair_mask = (s1 == v1) & (s2 == v2)
            b_sub = df.loc[baseline_mask & pair_mask, target_col].dropna()
            c_sub = df.loc[comparison_mask & pair_mask, target_col].dropna()

            nb, nc = len(b_sub), len(c_sub)

            if is_target_num:
                b_r = float(b_sub.mean()) if nb > 0 else 0.0
                c_r = float(c_sub.mean()) if nc > 0 else 0.0
            else:
                event = target_event if target_event is not None else df[target_col].dropna().unique()[0]
                b_r = float((b_sub == event).mean()) if nb > 0 else 0.0
                c_r = float((c_sub == event).mean()) if nc > 0 else 0.0

            delta = c_r - b_r
            matrix_data[v1][v2] = delta

            if (nb + nc) >= 3:
                combinations.append(
                    CombinedFactorItem(
                        val1_label=f"{feature1}={v1}",
                        val2_label=f"{feature2}={v2}",
                        sample_size=nb + nc,
                        baseline_rate=b_r,
                        comparison_rate=c_r,
                        rate_change=delta,
                    )
                )

    matrix_df = pd.DataFrame(matrix_data)

    # Sort combinations by rate_change descending
    combinations.sort(key=lambda c: c.rate_change, reverse=True)
    if combinations:
        combinations[0].is_high_risk = True

    high_risk_str = f"{combinations[0].val1_label} AND {combinations[0].val2_label}" if combinations else "N/A"
    max_delta = combinations[0].rate_change if combinations else 0.0

    synergy_note = (
        f"The co-occurrence of **{high_risk_str}** resulted in the most pronounced outcome divergence "
        f"(Δ = {max_delta:+.3f}). Combined factor investigation indicates non-linear concentration."
    )

    return InteractionAnalysisResult(
        feature1=feature1,
        feature2=feature2,
        matrix_df=matrix_df,
        combinations=combinations,
        highest_risk_combination=high_risk_str,
        max_rate_change=max_delta,
        interaction_synergy_note=synergy_note,
    )

