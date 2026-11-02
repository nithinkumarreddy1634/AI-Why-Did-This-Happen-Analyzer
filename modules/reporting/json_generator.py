"""Machine-readable JSON report exporter with robust type serialization."""

import json
from typing import Dict, Any, List, Optional
import numpy as np

from modules.reporting.report_data_collector import CollectedReportData
from modules.reporting.narrative_generator import ReportNarrativeSections


class JSONReportGenerator:
    """Serializes complete report data and narratives to standardized JSON."""

    def generate_json(
        self,
        data: CollectedReportData,
        narratives: ReportNarrativeSections,
        included_sections: Optional[List[str]] = None,
    ) -> str:
        """Render complete structured JSON report string."""
        bundle: Dict[str, Any] = {
            "metadata": {
                "title": data.report_title,
                "subtitle": data.report_subtitle,
                "organization": data.organization_name,
                "author": data.author_name,
                "generation_date": data.generation_date,
                "confidentiality": data.confidentiality,
                "dataset_name": data.dataset_name,
                "dataset_shape": {"rows": data.dataset_shape[0], "columns": data.dataset_shape[1]},
            },
            "investigation_scope": {
                "target_variable": data.target_variable,
                "outcome_type": data.outcome_type,
                "baseline_cohort": data.baseline_cohort_name,
                "comparison_cohort": data.comparison_cohort_name,
                "baseline_value": data.baseline_outcome_val,
                "comparison_value": data.comparison_outcome_val,
                "delta": data.outcome_delta,
                "relative_change_pct": data.outcome_rel_pct,
            },
            "narratives": narratives.to_dict(),
            "sections_data": {
                "dataset_overview": data.dataset_overview_data,
                "data_quality": data.data_quality_data,
                "statistical": data.statistical_data,
                "machine_learning": data.ml_data,
                "shap": data.shap_data,
                "lime": data.lime_data,
                "root_cause": data.root_cause_data,
                "segment_analysis": data.segment_data,
            },
            "disclaimer": (
                "Methodological Notice: All metrics represent observational associations and predictive importance. "
                "Causality cannot be verified from observational data alone."
            ),
        }
        return json.dumps(bundle, indent=2, default=str)

