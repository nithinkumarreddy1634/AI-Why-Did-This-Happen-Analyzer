"""Root cause candidate generation and outcome shift measurement engine."""

from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


@dataclass
class OutcomeDefinition:
    """Quantitative definition of the unexpected outcome being investigated."""
    target_variable: str
    outcome_type: str  # 'numerical' or 'categorical'
    target_event: Optional[Any] = None
    baseline_name: str = "Baseline Group"
    comparison_name: str = "Comparison Group"
    baseline_value: float = 0.0
    comparison_value: float = 0.0
    absolute_change: float = 0.0
    relative_change_pct: Optional[float] = None
    baseline_count: int = 0
    comparison_count: int = 0
    total_count: int = 0
    investigation_scope: str = "Overall Population"
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert outcome definition to dictionary."""
        d = asdict(self)
        if self.relative_change_pct is not None:
            d["relative_change_pct"] = round(self.relative_change_pct, 2)
        d["absolute_change"] = round(self.absolute_change, 4)
        d["baseline_value"] = round(self.baseline_value, 4)
        d["comparison_value"] = round(self.comparison_value, 4)
        return d


@dataclass
class RootCauseCandidate:
    """Core representation of an individual feature candidate under investigation."""
    feature: str
    feature_type: str  # 'numerical' or 'categorical'
    baseline_val: float
    comparison_val: float
    absolute_shift: float
    relative_shift_pct: Optional[float]
    shift_direction: str  # 'Increased (+)', 'Decreased (-)', 'Stable (0)'
    baseline_display: str
    comparison_display: str
    sample_size_base: int
    sample_size_comp: int
    is_valid: bool = True
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert candidate to dictionary with safe rounding."""
        d = asdict(self)
        d["baseline_val"] = round(self.baseline_val, 4)
        d["comparison_val"] = round(self.comparison_val, 4)
        d["absolute_shift"] = round(self.absolute_shift, 4)
        if self.relative_shift_pct is not None:
            d["relative_shift_pct"] = round(self.relative_shift_pct, 2)
        return d


def extract_outcome_definition(
    df: pd.DataFrame,
    target_col: str,
    baseline_mask: pd.Series,
    comparison_mask: pd.Series,
    target_event: Optional[Any] = None,
    baseline_name: str = "Baseline Cohort",
    comparison_name: str = "Comparison Cohort",
) -> OutcomeDefinition:
    """Quantify the unexpected outcome difference across baseline and comparison cohorts."""
    if df is None or df.empty:
        raise ValueError("Cannot extract outcome definition from an empty dataset.")
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataframe.")

    base_s = df.loc[baseline_mask, target_col].dropna()
    comp_s = df.loc[comparison_mask, target_col].dropna()

    is_num = pd.api.types.is_numeric_dtype(df[target_col])
    unique_vals = df[target_col].dropna().unique()
    is_binary_num = is_num and (len(unique_vals) <= 2 and set(unique_vals).issubset({0, 1}))

    if target_event is not None or is_binary_num or not is_num:
        outcome_type = "categorical"
    else:
        outcome_type = "numerical"

    if outcome_type == "numerical":
        b_val = float(base_s.mean()) if len(base_s) > 0 else 0.0
        c_val = float(comp_s.mean()) if len(comp_s) > 0 else 0.0
        abs_diff = c_val - b_val
        rel_pct = ((abs_diff / abs(b_val)) * 100.0) if b_val != 0.0 else None
        event_str = "mean value"
    else:
        # Categorical outcome rate
        unique_vals = df[target_col].dropna().unique()
        if target_event is not None:
            event = target_event
        elif set(unique_vals) == {0, 1}:
            event = 1
        elif len(unique_vals) > 1:
            event = unique_vals[1]
        else:
            event = unique_vals[0]
        target_event = event

        b_val = float((base_s == event).mean()) if len(base_s) > 0 else 0.0
        c_val = float((comp_s == event).mean()) if len(comp_s) > 0 else 0.0
        abs_diff = c_val - b_val  # difference in proportions (e.g. +0.15 = +15 percentage points)
        rel_pct = ((abs_diff / abs(b_val)) * 100.0) if b_val != 0.0 else None
        event_str = f"rate of '{event}'"

    desc = (
        f"Observed change in {event_str} from {b_val:.4f} in '{baseline_name}' "
        f"to {c_val:.4f} in '{comparison_name}' (Δ = {abs_diff:+.4f}"
        + (f", {rel_pct:+.1f}%)" if rel_pct is not None else ", baseline=0.0)")
    )

    return OutcomeDefinition(
        target_variable=target_col,
        outcome_type=outcome_type,
        target_event=target_event,
        baseline_name=baseline_name,
        comparison_name=comparison_name,
        baseline_value=b_val,
        comparison_value=c_val,
        absolute_change=abs_diff,
        relative_change_pct=rel_pct,
        baseline_count=len(base_s),
        comparison_count=len(comp_s),
        total_count=len(base_s) + len(comp_s),
        description=desc,
    )


def generate_root_cause_candidates(
    df: pd.DataFrame,
    features: List[str],
    baseline_mask: pd.Series,
    comparison_mask: pd.Series,
    target_col: Optional[str] = None,
) -> List[RootCauseCandidate]:
    """Inspect all candidate features and calculate baseline vs comparison shifts."""
    candidates: List[RootCauseCandidate] = []

    for feat in features:
        if feat == target_col or feat not in df.columns:
            continue

        base_s = df.loc[baseline_mask, feat].dropna()
        comp_s = df.loc[comparison_mask, feat].dropna()

        n_base = len(base_s)
        n_comp = len(comp_s)

        if n_base == 0 and n_comp == 0:
            continue

        is_num = pd.api.types.is_numeric_dtype(df[feat])
        feat_type = "numerical" if is_num else "categorical"

        if is_num:
            b_val = float(base_s.mean()) if n_base > 0 else 0.0
            c_val = float(comp_s.mean()) if n_comp > 0 else 0.0
            abs_shift = c_val - b_val
            rel_pct = ((abs_shift / abs(b_val)) * 100.0) if b_val != 0.0 else None

            if abs(abs_shift) < 1e-5:
                direction = "Stable (0)"
            elif abs_shift > 0:
                direction = "Increased (+)"
            else:
                direction = "Decreased (-)"

            b_disp = f"μ = {b_val:.2f}"
            c_disp = f"μ = {c_val:.2f}"
        else:
            # For categorical: measure shift in mode or category concentration
            b_mode = str(base_s.mode().iloc[0]) if not base_s.empty else "N/A"
            c_mode = str(comp_s.mode().iloc[0]) if not comp_s.empty else "N/A"

            # Top category proportion in comparison group
            top_cat = comp_s.mode().iloc[0] if not comp_s.empty else (base_s.mode().iloc[0] if not base_s.empty else "N/A")
            b_prop = float((base_s == top_cat).mean()) if n_base > 0 else 0.0
            c_prop = float((comp_s == top_cat).mean()) if n_comp > 0 else 0.0

            b_val = b_prop
            c_val = c_prop
            abs_shift = c_prop - b_prop
            rel_pct = ((abs_shift / abs(b_prop)) * 100.0) if b_prop != 0.0 else None

            if abs(abs_shift) < 1e-4:
                direction = "Stable (0)"
            elif abs_shift > 0:
                direction = f"Increased '{top_cat}' (+)"
            else:
                direction = f"Decreased '{top_cat}' (-)"

            b_disp = f"{b_mode} ({b_prop*100:.1f}%)"
            c_disp = f"{c_mode} ({c_prop*100:.1f}%)"

        candidates.append(
            RootCauseCandidate(
                feature=feat,
                feature_type=feat_type,
                baseline_val=b_val,
                comparison_val=c_val,
                absolute_shift=abs_shift,
                relative_shift_pct=rel_pct,
                shift_direction=direction,
                baseline_display=b_disp,
                comparison_display=c_disp,
                sample_size_base=n_base,
                sample_size_comp=n_comp,
            )
        )

    return candidates
