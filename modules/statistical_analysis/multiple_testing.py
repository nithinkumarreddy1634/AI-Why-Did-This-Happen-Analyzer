"""Multiple testing correction: Family-Wise Error Rate (Bonferroni) and False Discovery Rate (Benjamini-Hochberg)."""

from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import pandas as pd

from modules.statistical_analysis.hypothesis_testing import HypothesisTestResult


def bonferroni_correction(
    p_values: List[float], alpha: float = 0.05
) -> Tuple[List[float], List[bool]]:
    """Apply Bonferroni correction controlling Family-Wise Error Rate (FWER).
    
    Formula:
        p_adj = min(1.0, p * m)
    """
    m = len(p_values)
    if m == 0:
        return [], []

    adjusted_p = [float(min(1.0, p * m)) for p in p_values]
    is_sig = [bool(adj_p < alpha) for adj_p in adjusted_p]
    return adjusted_p, is_sig


def benjamini_hochberg_correction(
    p_values: List[float], alpha: float = 0.05
) -> Tuple[List[float], List[bool]]:
    """Apply Benjamini-Hochberg (BH) procedure controlling False Discovery Rate (FDR).
    
    Formula:
        Sort p_(1) <= ... <= p_(m)
        p_adj = min(1.0, (m / rank) * p) with right-to-left cumulative minimum.
    """
    m = len(p_values)
    if m == 0:
        return [], []

    # Keep track of original indices
    sorted_pairs = sorted(enumerate(p_values), key=lambda x: x[1])
    sorted_indices = [idx for idx, _ in sorted_pairs]
    sorted_p = [p for _, p in sorted_pairs]

    # Calculate raw adjusted p-values
    adj_p_sorted = [min(1.0, (m / (i + 1)) * p) for i, p in enumerate(sorted_p)]

    # Enforce monotonicity from right to left
    for i in range(m - 2, -1, -1):
        adj_p_sorted[i] = min(adj_p_sorted[i], adj_p_sorted[i + 1])

    # Reconstruct original order
    adj_p_original = [0.0] * m
    for rank, orig_idx in enumerate(sorted_indices):
        adj_p_original[orig_idx] = float(adj_p_sorted[rank])

    is_sig = [bool(p < alpha) for p in adj_p_original]
    return adj_p_original, is_sig


def adjust_test_results(
    results: List[HypothesisTestResult],
    method: str = "benjamini_hochberg",
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Adjust a batch of HypothesisTestResult objects for multiple comparisons.
    
    Args:
        results: List of HypothesisTestResult instances.
        method: 'bonferroni' or 'benjamini_hochberg' (FDR).
        alpha: Adjusted significance threshold.
    """
    if not results:
        return pd.DataFrame(
            columns=["Test Name", "Research Question", "Raw P-Value", "Adjusted P-Value", "Sig. (Raw)", "Sig. (Adjusted)", "Verdict"]
        )

    raw_p_values = [r.p_value for r in results]

    if method.lower() in ("bonferroni", "fwer"):
        adj_p_vals, is_sig_flags = bonferroni_correction(raw_p_values, alpha=alpha)
        method_name = "Bonferroni (FWER)"
    else:
        adj_p_vals, is_sig_flags = benjamini_hochberg_correction(raw_p_values, alpha=alpha)
        method_name = "Benjamini-Hochberg (FDR)"

    rows = []
    for r, adj_p, is_sig in zip(results, adj_p_vals, is_sig_flags):
        status_change = "Retained Significance" if (r.is_significant and is_sig) else (
            "Lost Significance (False Discovery)" if (r.is_significant and not is_sig) else "Non-Significant"
        )

        rows.append({
            "Test Name": r.test_name,
            "Research Question": r.research_question,
            "Test Statistic": f"{r.statistic_name} = {r.test_statistic:.3f}",
            "Raw P-Value": f"{r.p_value:.4f}" if r.p_value >= 0.0001 else f"{r.p_value:.2e}",
            "Adjusted P-Value": f"{adj_p:.4f}" if adj_p >= 0.0001 else f"{adj_p:.2e}",
            "Sig. (Raw)": "✓ Significant" if r.is_significant else "✗ Not Sig.",
            "Sig. (Adjusted)": "✓ Significant" if is_sig else "✗ Not Sig.",
            "Verdict": status_change,
        })

    df = pd.DataFrame(rows)
    df.attrs["method"] = method_name
    df.attrs["alpha"] = alpha
    return df

