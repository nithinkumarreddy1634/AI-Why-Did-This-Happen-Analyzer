"""Utility helper functions for LIME explanation parsing, formatting, and metrics."""

from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import pandas as pd


def parse_lime_feature_condition(condition_str: str) -> Dict[str, Any]:
    """Parse LIME's string condition (e.g., 'tenure <= 12.00', '45.00 < charges <= 80.00') into structured components."""
    clean_cond = condition_str.strip()
    
    # Operators to check
    operators = ["<=", ">=", "<", ">", "="]
    
    # Extract feature name (find first token that doesn't look like a number or operator)
    tokens = clean_cond.split()
    feature_name = condition_str
    for t in tokens:
        # If token is not an operator and not a float/number
        if t not in operators:
            try:
                float(t)
            except ValueError:
                feature_name = t
                break

    return {
        "raw_condition": clean_cond,
        "feature_name": feature_name,
    }


def format_lime_weight(weight: float) -> str:
    """Format LIME contribution weight with explicit sign."""
    sign = "+" if weight > 0 else ""
    return f"{sign}{weight:.4f}"


def calculate_jaccard_similarity(list_a: List[str], list_b: List[str]) -> float:
    """Compute Jaccard similarity index between two sets of feature names."""
    set_a = set(list_a)
    set_b = set(list_b)
    if not set_a and not set_b:
        return 1.0
    union = set_a | set_b
    if not union:
        return 0.0
    intersection = set_a & set_b
    return float(len(intersection) / len(union))


def compute_rank_correlation(ranks_a: Dict[str, int], ranks_b: Dict[str, int]) -> Optional[float]:
    """Compute Spearman rank correlation between two sets of feature rankings."""
    common_keys = [k for k in ranks_a if k in ranks_b]
    if len(common_keys) < 3:
        return None

    r_a = np.array([ranks_a[k] for k in common_keys], dtype=float)
    r_b = np.array([ranks_b[k] for k in common_keys], dtype=float)

    # If constant ranks
    if np.std(r_a) == 0 or np.std(r_b) == 0:
        return 0.0

    corr = np.corrcoef(r_a, r_b)[0, 1]
    return float(corr) if not np.isnan(corr) else 0.0

