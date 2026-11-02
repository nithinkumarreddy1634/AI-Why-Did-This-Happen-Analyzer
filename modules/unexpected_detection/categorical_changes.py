"""Categorical outcome and frequency change detector."""

from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np

from utils.detection_helpers import safe_pct_change, safe_difference
from modules.unexpected_detection.severity_classifier import (
    SeverityLevel,
    SeverityClassifier,
)


class CategoricalChangeDetector:
    """Detects shifts in category frequencies, proportion swings, and emerging/vanishing labels."""

    def __init__(self, classifier: Optional[SeverityClassifier] = None) -> None:
        self.classifier = classifier or SeverityClassifier()

    def compare_categories(
        self,
        baseline_series: pd.Series,
        comparison_series: pd.Series,
        column_name: str,
        baseline_label: str = "Baseline",
        comparison_label: str = "Comparison",
        top_n: int = 15,
    ) -> Dict[str, Any]:
        """Compare category proportions between baseline and comparison groups."""
        b_clean = baseline_series.dropna().astype(str)
        c_clean = comparison_series.dropna().astype(str)

        n_b = len(b_clean)
        n_c = len(c_clean)

        if n_b == 0 or n_c == 0:
            return {
                "column": column_name,
                "status": "warning",
                "message": f"One or both cohorts contain 0 valid values (Baseline: {n_b}, Comparison: {n_c}).",
                "table": pd.DataFrame(),
                "emerging_categories": [],
                "disappearing_categories": [],
            }

        b_counts = b_clean.value_counts()
        c_counts = c_clean.value_counts()

        b_props = (b_counts / n_b) * 100.0
        c_props = (c_counts / n_c) * 100.0

        all_categories = sorted(list(set(b_counts.index).union(set(c_counts.index))))
        
        # Sort categories by highest combined count
        total_counts = {cat: b_counts.get(cat, 0) + c_counts.get(cat, 0) for cat in all_categories}
        sorted_categories = sorted(all_categories, key=lambda c: total_counts[c], reverse=True)

        # Filter top N to maintain readability for high cardinality columns
        is_truncated = len(sorted_categories) > top_n
        displayed_categories = sorted_categories[:top_n]

        emerging_categories = [cat for cat in all_categories if cat not in b_counts.index and cat in c_counts.index]
        disappearing_categories = [cat for cat in all_categories if cat in b_counts.index and cat not in c_counts.index]

        rows = []
        max_shift = 0.0
        highest_severity = SeverityLevel.NORMAL
        highest_severity_reason = "No significant proportion shifts detected."

        for cat in displayed_categories:
            b_cnt = int(b_counts.get(cat, 0))
            c_cnt = int(c_counts.get(cat, 0))
            b_pr = float(b_props.get(cat, 0.0))
            c_pr = float(c_props.get(cat, 0.0))

            pp_shift = c_pr - b_pr  # Percentage points shift
            rel_change = safe_pct_change(b_pr, c_pr)

            sev, sev_reason = self.classifier.classify_percentage_point_shift(pp_shift)
            if abs(pp_shift) > max_shift:
                max_shift = abs(pp_shift)
                highest_severity = sev
                highest_severity_reason = sev_reason

            category_status = "Retained"
            if cat in emerging_categories:
                category_status = "Newly Appearing"
            elif cat in disappearing_categories:
                category_status = "Disappeared"

            rows.append({
                "Category": cat,
                f"{baseline_label} Count": b_cnt,
                f"{comparison_label} Count": c_cnt,
                f"{baseline_label} %": round(b_pr, 2),
                f"{comparison_label} %": round(c_pr, 2),
                "Percentage Point Shift (pp)": round(pp_shift, 2),
                "Relative % Change": f"{rel_change:+.1f}%" if rel_change is not None else ("New (+100%)" if c_pr > 0 else "0%"),
                "Status": category_status,
                "Severity": sev.value,
            })

        table_df = pd.DataFrame(rows)

        # Calculate concentration (Herfindahl-Hirschman index equivalent)
        b_hhi = float((b_props**2).sum())
        c_hhi = float((c_props**2).sum())
        hhi_diff = c_hhi - b_hhi

        return {
            "column": column_name,
            "status": "success",
            "baseline_label": baseline_label,
            "comparison_label": comparison_label,
            "baseline_total": n_b,
            "comparison_total": n_c,
            "unique_categories_total": len(all_categories),
            "is_truncated": is_truncated,
            "emerging_categories": emerging_categories,
            "disappearing_categories": disappearing_categories,
            "max_percentage_point_shift": max_shift,
            "highest_severity": highest_severity.value,
            "highest_severity_reason": highest_severity_reason,
            "concentration_shift": hhi_diff,
            "table": table_df,
            "raw_baseline_props": b_props,
            "raw_comparison_props": c_props,
            "message": f"Categorical analysis for '{column_name}': max shift is {max_shift:+.1f} percentage points ({highest_severity.value}).",
        }
