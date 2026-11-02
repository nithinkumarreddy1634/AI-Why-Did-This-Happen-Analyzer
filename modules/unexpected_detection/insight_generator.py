"""Automated factual insight generator for unexpected outcomes."""

from typing import Dict, Any, List, Optional
import pandas as pd
from modules.unexpected_detection.severity_classifier import SeverityLevel


class UnexpectedInsightGenerator:
    """Generates rigorous, non-causal natural language observations backed by empirical metrics."""

    @staticmethod
    def generate_numerical_anomaly_insights(
        results: Dict[str, Any], column_name: str
    ) -> List[Dict[str, Any]]:
        """Synthesize factual insights from numerical anomaly detection runs."""
        insights = []
        if results.get("status") != "success":
            return insights

        unusual_count = results.get("unusual_count", 0)
        unusual_pct = results.get("unusual_pct", 0.0)
        total = results.get("total_observations", 0)
        valid = results.get("valid_observations", 0)
        missing = results.get("missing_observations", 0)
        method = results.get("method", "Statistical Anomaly")
        threshold = results.get("threshold", 0)

        if missing > 0:
            missing_pct = (missing / total) * 100.0
            if missing_pct >= 10.0:
                insights.append({
                    "title": f"Data Completeness Notice: {column_name}",
                    "severity": SeverityLevel.LOW.value,
                    "finding": f"Column '{column_name}' contains {missing:,} missing values ({missing_pct:.1f}% of total). Anomaly statistics are computed solely on {valid:,} valid entries.",
                    "evidence": f"Missing = {missing:,} | Valid = {valid:,} | Total = {total:,}",
                    "method": "Data Integrity Audit",
                })

        if unusual_count > 0:
            sev = SeverityLevel.MODERATE.value
            if unusual_pct > 10.0:
                sev = SeverityLevel.CRITICAL.value
            elif unusual_pct > 5.0:
                sev = SeverityLevel.HIGH.value

            insights.append({
                "title": f"Unusual Numerical Values Detected in {column_name}",
                "severity": sev,
                "finding": (
                    f"A total of {unusual_count:,} observations ({unusual_pct:.2f}% of valid records) "
                    f"in '{column_name}' deviate beyond the selected {method} threshold ({threshold})."
                ),
                "evidence": f"Unusual Count = {unusual_count:,} | Rate = {unusual_pct:.2f}% | Sample Size = {valid:,}",
                "method": method,
            })
        else:
            insights.append({
                "title": f"No Extreme Numerical Anomalies in {column_name}",
                "severity": SeverityLevel.NORMAL.value,
                "finding": f"All {valid:,} valid observations in '{column_name}' remain within expected bounds under {method} (threshold: {threshold}).",
                "evidence": f"0 anomalies detected across {valid:,} valid observations.",
                "method": method,
            })

        return insights

    @staticmethod
    def generate_distribution_shift_insights(
        results: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Synthesize factual insights from distribution comparisons."""
        insights = []
        if results.get("status") != "success":
            return insights

        col = results.get("column", "Metric")
        b_label = results.get("baseline_label", "Baseline")
        c_label = results.get("comparison_label", "Comparison")
        b_count = results.get("baseline_count", 0)
        c_count = results.get("comparison_count", 0)
        mean_diff = results.get("mean_difference")
        mean_pct = results.get("mean_pct_change")
        sev = results.get("severity", SeverityLevel.NORMAL.value)
        sev_reason = results.get("severity_reason", "")

        direction = "higher" if (mean_diff is not None and mean_diff > 0) else "lower"
        pct_str = f" ({mean_pct:+.1f}%)" if mean_pct is not None else ""
        diff_str = f"{abs(mean_diff):.2f}" if mean_diff is not None else "N/A"

        insights.append({
            "title": f"Distribution Shift Observed in '{col}'",
            "severity": sev,
            "finding": (
                f"The average of '{col}' is {direction} in the {c_label} cohort compared to {b_label} "
                f"by an absolute difference of {diff_str}{pct_str}."
            ),
            "evidence": f"{b_label} (N={b_count:,}) vs. {c_label} (N={c_count:,}) | Justification: {sev_reason}",
            "method": "Two-Cohort Distribution Comparison",
        })

        return insights

    @staticmethod
    def generate_categorical_shift_insights(
        results: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Synthesize factual insights from categorical frequency changes."""
        insights = []
        if results.get("status") != "success":
            return insights

        col = results.get("column", "Category")
        table = results.get("table", pd.DataFrame())
        b_label = results.get("baseline_label", "Baseline")
        c_label = results.get("comparison_label", "Comparison")
        b_tot = results.get("baseline_total", 0)
        c_tot = results.get("comparison_total", 0)
        emerging = results.get("emerging_categories", [])
        disappearing = results.get("disappearing_categories", [])

        # Check top shifted category
        if not table.empty:
            top_shift_row = table.sort_values(
                by="Percentage Point Shift (pp)", key=abs, ascending=False
            ).iloc[0]
            
            cat_name = top_shift_row["Category"]
            pp_shift = top_shift_row["Percentage Point Shift (pp)"]
            dir_str = "increased" if pp_shift > 0 else "decreased"
            cat_sev = top_shift_row["Severity"]

            insights.append({
                "title": f"Significant Category Swing: '{cat_name}' in {col}",
                "severity": cat_sev,
                "finding": (
                    f"In column '{col}', category '{cat_name}' {dir_str} by {abs(pp_shift):.1f} percentage points "
                    f"between {b_label} ({top_shift_row[f'{b_label} %']}%) and {c_label} ({top_shift_row[f'{c_label} %']}%)."
                ),
                "evidence": f"{b_label} (N={b_tot:,}) vs. {c_label} (N={c_tot:,}) | Shift = {pp_shift:+.1f} pp",
                "method": "Proportion Shift Analysis",
            })

        if emerging:
            insights.append({
                "title": f"Newly Appearing Categories in {col}",
                "severity": SeverityLevel.MODERATE.value,
                "finding": f"Discovered {len(emerging)} newly appearing categories in {c_label} that were absent in {b_label}: {', '.join(emerging[:5])}{'...' if len(emerging) > 5 else ''}.",
                "evidence": f"New categories: {len(emerging)} | Sample size = {c_tot:,}",
                "method": "Category Cardinality Audit",
            })

        if disappearing:
            insights.append({
                "title": f"Disappearing Categories in {col}",
                "severity": SeverityLevel.LOW.value,
                "finding": f"Found {len(disappearing)} categories present in {b_label} that disappeared completely in {c_label}: {', '.join(disappearing[:5])}{'...' if len(disappearing) > 5 else ''}.",
                "evidence": f"Vanished categories: {len(disappearing)} | Baseline size = {b_tot:,}",
                "method": "Category Cardinality Audit",
            })

        return insights

    @staticmethod
    def generate_segment_insights(results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Synthesize factual insights from segment-level analysis."""
        insights = []
        if results.get("status") != "success":
            return insights

        ranked = results.get("ranked_segments", pd.DataFrame())
        seg_col = results.get("segment_col", "Segment")
        metric_col = results.get("metric_col", "Metric")
        m_type = results.get("metric_type", "Metric Value")
        b_label = results.get("baseline_label", "Baseline")
        c_label = results.get("comparison_label", "Comparison")

        if not ranked.empty:
            top_seg = ranked.iloc[0]
            diff = top_seg.get("Absolute Difference")
            pct = top_seg.get("% Change")
            sev = top_seg.get("Severity", SeverityLevel.NORMAL.value)
            seg_val = top_seg.get("Segment")

            insights.append({
                "title": f"Largest Segment Divergence: '{seg_val}' in {seg_col}",
                "severity": sev,
                "finding": (
                    f"Subgroup '{seg_val}' experienced the largest measured shift in {metric_col} "
                    f"with an absolute difference of {diff:+.2f} ({pct}) between {b_label} and {c_label}."
                ),
                "evidence": f"{b_label} Count={top_seg.get(f'{b_label} Count')}, {c_label} Count={top_seg.get(f'{c_label} Count')}",
                "method": "Segment-Level Disaggregation",
            })

        return insights
