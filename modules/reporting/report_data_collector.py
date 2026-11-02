"""Report data collector consolidating analytical artifacts across Phases 1-12."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np


@dataclass
class CollectedReportData:
    """Comprehensive data bundle collected across all analytical engines for reporting."""
    report_title: str
    report_subtitle: str
    organization_name: str
    author_name: str
    generation_date: str
    confidentiality: str
    dataset_name: str
    dataset_shape: tuple[int, int]
    target_variable: str
    outcome_type: str
    baseline_cohort_name: str
    comparison_cohort_name: str
    baseline_outcome_val: float
    comparison_outcome_val: float
    outcome_delta: float
    outcome_rel_pct: Optional[float]

    # Section Data & Availability
    has_dataset_overview: bool = True
    dataset_overview_data: Dict[str, Any] = field(default_factory=dict)

    has_data_quality: bool = False
    data_quality_data: Dict[str, Any] = field(default_factory=dict)

    has_eda: bool = False
    eda_data: Dict[str, Any] = field(default_factory=dict)

    has_unexpected_detection: bool = False
    unexpected_data: Dict[str, Any] = field(default_factory=dict)

    has_statistical: bool = False
    statistical_data: Dict[str, Any] = field(default_factory=dict)

    has_ml: bool = False
    ml_data: Dict[str, Any] = field(default_factory=dict)

    has_shap: bool = False
    shap_data: Dict[str, Any] = field(default_factory=dict)

    has_lime: bool = False
    lime_data: Dict[str, Any] = field(default_factory=dict)

    has_anomaly: bool = False
    anomaly_data: Dict[str, Any] = field(default_factory=dict)

    has_root_cause: bool = False
    root_cause_data: Dict[str, Any] = field(default_factory=dict)

    has_segment_analysis: bool = False
    segment_data: Dict[str, Any] = field(default_factory=dict)


class ReportDataCollector:
    """Consolidates session state and active investigation results into a normalized report bundle."""

    def collect_from_environment(
        self,
        df: pd.DataFrame,
        dataset_name: str = "dataset.csv",
        active_investigation_pkg: Optional[Any] = None,
        ml_importance_res: Optional[Any] = None,
        ml_comparison_res: Optional[Any] = None,
        shap_global_res: Optional[Any] = None,
        lime_active_res: Optional[Any] = None,
        detection_artifacts: Optional[Any] = None,
        preprocessing_pipeline: Optional[Any] = None,
        title: str = "Automated Investigation Report",
        subtitle: str = "Multi-Engine Root Cause & Predictive Diagnostic Dossier",
        organization: str = "Analytics & Governance Group",
        author: str = "AI Why Did This Happen Analyzer",
        confidentiality: str = "Internal Business Sensitive",
    ) -> CollectedReportData:
        """Extract all available analytical metrics and metadata into a CollectedReportData bundle."""
        n_rows, n_cols = df.shape if df is not None else (0, 0)

        # Baseline & Comparison defaults
        target_var = "Target"
        out_type = "numerical"
        base_name = "Baseline Reference"
        comp_name = "Comparison Affected"
        b_val = 0.0
        c_val = 0.0
        delta = 0.0
        rel_pct = None

        if active_investigation_pkg:
            odef = active_investigation_pkg.outcome_def
            target_var = odef.target_variable
            out_type = odef.outcome_type
            base_name = odef.baseline_name
            comp_name = odef.comparison_name
            b_val = odef.baseline_value
            c_val = odef.comparison_value
            delta = odef.absolute_change
            rel_pct = odef.relative_change_pct
        elif df is not None and not df.empty:
            num_cols = df.select_dtypes(include=[np.number]).columns
            if len(num_cols) > 0:
                target_var = num_cols[-1]
                b_val = float(df[target_var].mean())
                c_val = b_val
            else:
                target_var = df.columns[-1]

        # 1. Dataset Overview
        num_c = list(df.select_dtypes(include=[np.number]).columns) if df is not None else []
        cat_c = list(df.select_dtypes(exclude=[np.number]).columns) if df is not None else []
        missing_count = int(df.isnull().sum().sum()) if df is not None else 0
        missing_pct = float(missing_count / (n_rows * n_cols) * 100.0) if (n_rows * n_cols) > 0 else 0.0

        ds_overview = {
            "n_rows": n_rows,
            "n_cols": n_cols,
            "num_features_count": len(num_c),
            "cat_features_count": len(cat_c),
            "total_missing_cells": missing_count,
            "missing_cells_pct": round(missing_pct, 2),
            "sample_columns": list(df.columns[:10]) if df is not None else [],
        }

        # 2. Data Quality
        has_quality = (missing_count > 0 or preprocessing_pipeline is not None)
        quality_data = {
            "missing_cells": missing_count,
            "duplicate_rows": int(df.duplicated().sum()) if df is not None else 0,
            "constant_columns": [c for c in df.columns if df[c].nunique() <= 1] if df is not None else [],
            "pipeline_applied": bool(preprocessing_pipeline is not None),
        }

        # 3. EDA Summary
        has_eda = len(num_c) >= 2
        eda_data = {
            "numerical_columns": num_c[:8],
            "categorical_columns": cat_c[:8],
        }

        # 4. Unexpected Detection
        has_unexp = bool(detection_artifacts and isinstance(detection_artifacts, dict))
        unexp_data = detection_artifacts if has_unexp else {}

        # 5. Statistical Evidence
        has_stat = False
        stat_data = {}
        if active_investigation_pkg:
            sig_items = [c for c in active_investigation_pkg.ranking_result.ranked_candidates if c.stat_significant]
            has_stat = len(sig_items) > 0 or any(c.stat_p_value is not None for c in active_investigation_pkg.ranking_result.ranked_candidates)
            stat_data = {
                "significant_candidates_count": len(sig_items),
                "top_significant": [
                    {"feature": c.feature, "p_value": c.stat_p_value, "effect_size": c.effect_size}
                    for c in sig_items[:5]
                ],
            }

        # 6. Machine Learning Findings
        has_ml = ml_importance_res is not None or ml_comparison_res is not None
        ml_data = {}
        if ml_importance_res and hasattr(ml_importance_res, "top_features"):
            ml_data["top_features"] = ml_importance_res.top_features[:8]
            ml_data["method"] = getattr(ml_importance_res, "method_display", "Feature Importance")
        if ml_comparison_res and hasattr(ml_comparison_res, "best_model_name"):
            ml_data["best_model"] = ml_comparison_res.best_model_name

        # 7. SHAP
        has_shap = shap_global_res is not None
        shap_data = {}
        if has_shap and hasattr(shap_global_res, "top_features"):
            shap_data["top_features"] = shap_global_res.top_features[:8]

        # 8. LIME
        has_lime = lime_active_res is not None
        lime_data = {}
        if has_lime and hasattr(lime_active_res, "contributions_df"):
            df_lime = lime_active_res.contributions_df
            lime_data["rules"] = df_lime.head(5).to_dict(orient="records") if not df_lime.empty else []
            lime_data["fidelity_r2"] = getattr(lime_active_res, "local_fidelity_r2", None)

        # 9. Anomaly
        has_anom = has_unexp and "anomalies" in unexp_data
        anom_data = unexp_data.get("anomalies", {}) if has_anom else {}

        # 10. Root Cause
        has_rc = active_investigation_pkg is not None
        rc_data = {}
        if has_rc:
            rr = active_investigation_pkg.ranking_result
            rc_data = {
                "top_candidate": rr.top_candidate_name,
                "ranked_candidates": [c.to_dict() for c in rr.ranked_candidates[:8]],
                "disclaimer": rr.disclaimer,
            }

        # 11. Segment Analysis
        has_seg = bool(active_investigation_pkg and active_investigation_pkg.segment_result)
        seg_data = active_investigation_pkg.segment_result.to_dict() if has_seg else {}

        return CollectedReportData(
            report_title=title,
            report_subtitle=subtitle,
            organization_name=organization,
            author_name=author,
            generation_date=datetime.now().strftime("%B %d, %Y - %H:%M"),
            confidentiality=confidentiality,
            dataset_name=dataset_name,
            dataset_shape=(n_rows, n_cols),
            target_variable=target_var,
            outcome_type=out_type,
            baseline_cohort_name=base_name,
            comparison_cohort_name=comp_name,
            baseline_outcome_val=b_val,
            comparison_outcome_val=c_val,
            outcome_delta=delta,
            outcome_rel_pct=rel_pct,
            has_dataset_overview=True,
            dataset_overview_data=ds_overview,
            has_data_quality=has_quality,
            data_quality_data=quality_data,
            has_eda=has_eda,
            eda_data=eda_data,
            has_unexpected_detection=has_unexp,
            unexpected_data=unexp_data,
            has_statistical=has_stat,
            statistical_data=stat_data,
            has_ml=has_ml,
            ml_data=ml_data,
            has_shap=has_shap,
            shap_data=shap_data,
            has_lime=has_lime,
            lime_data=lime_data,
            has_anomaly=has_anom,
            anomaly_data=anom_data,
            has_root_cause=has_rc,
            root_cause_data=rc_data,
            has_segment_analysis=has_seg,
            segment_data=seg_data,
        )

