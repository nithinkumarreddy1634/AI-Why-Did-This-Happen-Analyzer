"""Multi-source evidence collector aggregating statistical, ML, XAI, and segment findings."""

from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from scipy import stats

from modules.root_cause.candidate_generator import RootCauseCandidate, OutcomeDefinition


@dataclass
class CandidateEvidence:
    """Consolidated analytical evidence profile for a single potential root cause candidate."""
    feature: str
    feature_type: str  # 'numerical' or 'categorical'
    # 1. Observed Delta
    baseline_val: float
    comparison_val: float
    absolute_shift: float
    relative_shift_pct: Optional[float]
    shift_direction: str

    # 2. Statistical Evidence (Phase 6)
    stat_test_name: Optional[str] = None
    p_value: Optional[float] = None
    p_value_corrected: Optional[float] = None
    is_stat_significant: bool = False
    effect_size: Optional[float] = None
    effect_size_name: Optional[str] = None
    ci_lower: Optional[float] = None
    ci_upper: Optional[float] = None
    stat_interpretation: str = "No statistical test executed"

    # 3. Machine Learning Evidence (Phase 7)
    ml_importance_score: Optional[float] = None
    ml_importance_rank: Optional[int] = None
    ml_method: Optional[str] = None

    # 4. SHAP Evidence (Phase 8)
    shap_global_importance: Optional[float] = None
    shap_global_rank: Optional[int] = None
    shap_local_attribution: Optional[float] = None
    shap_direction: Optional[str] = None

    # 5. LIME Evidence (Phase 9)
    lime_local_weight: Optional[float] = None
    lime_condition: Optional[str] = None
    lime_direction: Optional[str] = None
    lime_fidelity_r2: Optional[float] = None

    # 6. Segment Evidence (Phase 4/5)
    max_segment_shift: Optional[float] = None
    top_affected_segment: Optional[str] = None

    # 7. Anomaly Evidence (Phase 5/10)
    anomaly_concentration_pct: Optional[float] = None
    unusual_record_count: int = 0

    # 8. Source Availability Tracking
    sources_available: Dict[str, bool] = field(default_factory=dict)
    supporting_sources: List[str] = field(default_factory=list)
    conflicting_sources: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert evidence to dictionary."""
        d = asdict(self)
        if self.p_value is not None:
            d["p_value"] = float(np.round(self.p_value, 5))
        if self.effect_size is not None:
            d["effect_size"] = float(np.round(self.effect_size, 4))
        if self.ml_importance_score is not None:
            d["ml_importance_score"] = float(np.round(self.ml_importance_score, 4))
        if self.shap_global_importance is not None:
            d["shap_global_importance"] = float(np.round(self.shap_global_importance, 4))
        if self.lime_local_weight is not None:
            d["lime_local_weight"] = float(np.round(self.lime_local_weight, 4))
        return d


class EvidenceCollector:
    """Collects and organizes empirical evidence from all prior analysis engines."""

    def __init__(self, alpha: float = 0.05) -> None:
        self.alpha = alpha

    def collect_evidence_for_candidates(
        self,
        df: pd.DataFrame,
        candidates: List[RootCauseCandidate],
        baseline_mask: pd.Series,
        comparison_mask: pd.Series,
        target_col: str,
        ml_importance_res: Optional[Any] = None,
        shap_global_res: Optional[Any] = None,
        shap_local_res: Optional[Any] = None,
        lime_local_res: Optional[Any] = None,
        anomaly_artifacts: Optional[Any] = None,
    ) -> List[CandidateEvidence]:
        """Aggregate all analytical signals across available phases for each candidate."""
        evidence_list: List[CandidateEvidence] = []

        # Extract ML importance lookup
        ml_map = self._parse_ml_importance(ml_importance_res)
        shap_global_map = self._parse_shap_global(shap_global_res)
        shap_local_map = self._parse_shap_local(shap_local_res)
        lime_map = self._parse_lime_local(lime_local_res)

        for cand in candidates:
            feat = cand.feature
            sources_avail: Dict[str, bool] = {
                "observed_shift": True,
                "statistical_test": False,
                "ml_importance": False,
                "shap_global": False,
                "shap_local": False,
                "lime_local": False,
                "anomaly_detection": False,
            }
            supporting: List[str] = ["Observed Cohort Shift"]
            conflicting: List[str] = []

            # 1. Run statistical test
            stat_res = self._run_statistical_comparison(
                df=df,
                feature=feat,
                is_numeric=(cand.feature_type == "numerical"),
                baseline_mask=baseline_mask,
                comparison_mask=comparison_mask,
            )
            if stat_res.get("executed"):
                sources_avail["statistical_test"] = True
                if stat_res.get("is_significant"):
                    supporting.append(f"Statistical Test ({stat_res['test_name']}, p={stat_res['p_value']:.4f})")

            # 2. ML feature importance
            ml_data = ml_map.get(feat, {})
            if ml_data:
                sources_avail["ml_importance"] = True
                if ml_data.get("rank", 999) <= 5:
                    supporting.append(f"ML Feature Importance (Rank #{ml_data['rank']})")

            # 3. SHAP global & local
            shap_g = shap_global_map.get(feat, {})
            if shap_g:
                sources_avail["shap_global"] = True
                if shap_g.get("rank", 999) <= 5:
                    supporting.append(f"Global SHAP Impact (Rank #{shap_g['rank']})")

            shap_l = shap_local_map.get(feat, {})
            if shap_l:
                sources_avail["shap_local"] = True

            # 4. LIME local
            lime_d = lime_map.get(feat, {})
            if lime_d:
                sources_avail["lime_local"] = True
                if lime_d.get("rank", 999) <= 3:
                    supporting.append(f"LIME Rule ({lime_d.get('condition', feat)})")

            # 5. Check directional conflict between SHAP/LIME and observed shift
            if shap_l and cand.shift_direction != "Stable (0)":
                shap_val = shap_l.get("shap_value", 0.0)
                shift_positive = cand.absolute_shift > 0
                shap_positive = shap_val > 0
                if (shift_positive and not shap_positive) or (not shift_positive and shap_positive):
                    conflicting.append("SHAP attribution opposes observed bivariate shift direction")

            # 6. Anomaly concentration
            anom_data = self._check_anomaly_concentration(df, feat, comparison_mask, anomaly_artifacts)
            if anom_data.get("available"):
                sources_avail["anomaly_detection"] = True
                if anom_data.get("unusual_count", 0) > 0:
                    supporting.append(f"Outlier Concentration ({anom_data['unusual_count']} unusual records)")

            evidence = CandidateEvidence(
                feature=feat,
                feature_type=cand.feature_type,
                baseline_val=cand.baseline_val,
                comparison_val=cand.comparison_val,
                absolute_shift=cand.absolute_shift,
                relative_shift_pct=cand.relative_shift_pct,
                shift_direction=cand.shift_direction,
                stat_test_name=stat_res.get("test_name"),
                p_value=stat_res.get("p_value"),
                is_stat_significant=stat_res.get("is_significant", False),
                effect_size=stat_res.get("effect_size"),
                effect_size_name=stat_res.get("effect_size_name"),
                ci_lower=stat_res.get("ci_lower"),
                ci_upper=stat_res.get("ci_upper"),
                stat_interpretation=stat_res.get("interpretation", ""),
                ml_importance_score=ml_data.get("score"),
                ml_importance_rank=ml_data.get("rank"),
                ml_method=ml_data.get("method"),
                shap_global_importance=shap_g.get("score"),
                shap_global_rank=shap_g.get("rank"),
                shap_local_attribution=shap_l.get("shap_value"),
                shap_direction=shap_l.get("direction"),
                lime_local_weight=lime_d.get("weight"),
                lime_condition=lime_d.get("condition"),
                lime_direction=lime_d.get("direction"),
                lime_fidelity_r2=lime_d.get("fidelity_r2"),
                anomaly_concentration_pct=anom_data.get("concentration_pct"),
                unusual_record_count=anom_data.get("unusual_count", 0),
                sources_available=sources_avail,
                supporting_sources=supporting,
                conflicting_sources=conflicting,
            )
            evidence_list.append(evidence)

        return evidence_list

    def _run_statistical_comparison(
        self,
        df: pd.DataFrame,
        feature: str,
        is_numeric: bool,
        baseline_mask: pd.Series,
        comparison_mask: pd.Series,
    ) -> Dict[str, Any]:
        """Execute independent two-sample statistical test between baseline and comparison."""
        base_s = df.loc[baseline_mask, feature].dropna()
        comp_s = df.loc[comparison_mask, feature].dropna()

        if len(base_s) < 3 or len(comp_s) < 3:
            return {"executed": False, "reason": "Insufficient observations (N < 3)"}

        try:
            if is_numeric:
                # Independent Welch's t-test
                t_stat, p_val = stats.ttest_ind(comp_s, base_s, equal_var=False)
                # Cohen's d effect size
                n1, n2 = len(comp_s), len(base_s)
                s1, s2 = float(comp_s.std(ddof=1)), float(base_s.std(ddof=1))
                pooled_sd = np.sqrt(((n1 - 1) * (s1 ** 2) + (n2 - 1) * (s2 ** 2)) / max(1, (n1 + n2 - 2)))
                d = float((comp_s.mean() - base_s.mean()) / pooled_sd) if pooled_sd > 0 else 0.0

                is_sig = bool(p_val < self.alpha)
                interp = (
                    f"Statistically significant difference (p = {p_val:.4e} < {self.alpha}, Cohen's d = {d:.2f})."
                    if is_sig else
                    f"Difference is not statistically significant (p = {p_val:.4f} >= {self.alpha})."
                )
                return {
                    "executed": True,
                    "test_name": "Welch's Two-Sample t-Test",
                    "p_value": float(p_val),
                    "is_significant": is_sig,
                    "effect_size": d,
                    "effect_size_name": "Cohen's d",
                    "ci_lower": float(comp_s.mean() - base_s.mean() - 1.96 * (pooled_sd / np.sqrt(n1 + n2))),
                    "ci_upper": float(comp_s.mean() - base_s.mean() + 1.96 * (pooled_sd / np.sqrt(n1 + n2))),
                    "interpretation": interp,
                }
            else:
                # Chi-Square test of independence
                combined = pd.DataFrame({
                    "group": ["Baseline"] * len(base_s) + ["Comparison"] * len(comp_s),
                    "val": list(base_s) + list(comp_s),
                })
                contingency = pd.crosstab(combined["group"], combined["val"])
                if contingency.shape[0] < 2 or contingency.shape[1] < 2:
                    return {"executed": False, "reason": "Single category present"}

                chi2, p_val, dof, _ = stats.chi2_contingency(contingency)
                n_total = len(combined)
                min_dim = min(contingency.shape) - 1
                cramers_v = float(np.sqrt(chi2 / (n_total * max(1, min_dim)))) if min_dim > 0 else 0.0
                is_sig = bool(p_val < self.alpha)

                return {
                    "executed": True,
                    "test_name": "Chi-Square Test of Independence",
                    "p_value": float(p_val),
                    "is_significant": is_sig,
                    "effect_size": cramers_v,
                    "effect_size_name": "Cramer's V",
                    "ci_lower": None,
                    "ci_upper": None,
                    "interpretation": f"Distribution shift {'is significant' if is_sig else 'is not significant'} (p={p_val:.4e}, V={cramers_v:.2f}).",
                }
        except Exception as err:
            return {"executed": False, "reason": str(err)}

    def _parse_ml_importance(self, ml_res: Optional[Any]) -> Dict[str, Dict[str, Any]]:
        """Extract feature importance lookup dictionary."""
        lookup: Dict[str, Dict[str, Any]] = {}
        if ml_res is None or not hasattr(ml_res, "importance_df"):
            return lookup
        df_imp = getattr(ml_res, "importance_df", None)
        if df_imp is None or not isinstance(df_imp, pd.DataFrame) or df_imp.empty:
            return lookup

        # Check column names: Feature, Importance / Normalized_Importance / Permutation_Importance
        feat_col = "Feature" if "Feature" in df_imp.columns else df_imp.columns[0]
        score_col = None
        for col in ["Normalized_Importance", "Importance", "Permutation_Importance", "Mean_Importance"]:
            if col in df_imp.columns:
                score_col = col
                break
        if not score_col:
            score_col = df_imp.columns[1] if len(df_imp.columns) > 1 else feat_col

        for rank, (_, row) in enumerate(df_imp.iterrows(), start=1):
            f = str(row[feat_col])
            val = float(row[score_col]) if pd.notnull(row[score_col]) else 0.0
            lookup[f] = {
                "score": val,
                "rank": rank,
                "method": getattr(ml_res, "method_display", "ML Importance"),
            }
        return lookup

    def _parse_shap_global(self, shap_g: Optional[Any]) -> Dict[str, Dict[str, Any]]:
        """Extract SHAP global importance lookup dictionary."""
        lookup: Dict[str, Dict[str, Any]] = {}
        if shap_g is None or not hasattr(shap_g, "importance_df"):
            return lookup
        df_shap = getattr(shap_g, "importance_df", None)
        if df_shap is None or not isinstance(df_shap, pd.DataFrame) or df_shap.empty:
            return lookup

        feat_col = "Feature" if "Feature" in df_shap.columns else df_shap.columns[0]
        val_col = "Mean_Abs_SHAP" if "Mean_Abs_SHAP" in df_shap.columns else df_shap.columns[1]

        for rank, (_, row) in enumerate(df_shap.iterrows(), start=1):
            f = str(row[feat_col])
            val = float(row[val_col]) if pd.notnull(row[val_col]) else 0.0
            lookup[f] = {"score": val, "rank": rank}
        return lookup

    def _parse_shap_local(self, shap_l: Optional[Any]) -> Dict[str, Dict[str, Any]]:
        """Extract SHAP local attribution lookup dictionary."""
        lookup: Dict[str, Dict[str, Any]] = {}
        if shap_l is None or not hasattr(shap_l, "contributions_df"):
            return lookup
        df_l = getattr(shap_l, "contributions_df", None)
        if df_l is None or not isinstance(df_l, pd.DataFrame) or df_l.empty:
            return lookup

        feat_col = "Feature" if "Feature" in df_l.columns else df_l.columns[0]
        val_col = "SHAP_Value" if "SHAP_Value" in df_l.columns else df_l.columns[1]
        dir_col = "Direction" if "Direction" in df_l.columns else None

        for _, row in df_l.iterrows():
            f = str(row[feat_col])
            val = float(row[val_col]) if pd.notnull(row[val_col]) else 0.0
            d = str(row[dir_col]) if dir_col else ("Increases" if val > 0 else "Decreases")
            lookup[f] = {"shap_value": val, "direction": d}
        return lookup

    def _parse_lime_local(self, lime_l: Optional[Any]) -> Dict[str, Dict[str, Any]]:
        """Extract LIME local explanation lookup dictionary."""
        lookup: Dict[str, Dict[str, Any]] = {}
        if lime_l is None or not hasattr(lime_l, "contributions_df"):
            return lookup
        df_lime = getattr(lime_l, "contributions_df", None)
        if df_lime is None or not isinstance(df_lime, pd.DataFrame) or df_lime.empty:
            return lookup

        feat_col = "Feature" if "Feature" in df_lime.columns else df_lime.columns[0]
        wt_col = "LIME_Weight" if "LIME_Weight" in df_lime.columns else (
            "Weight" if "Weight" in df_lime.columns else df_lime.columns[1]
        )
        cond_col = "Condition" if "Condition" in df_lime.columns else feat_col
        dir_col = "Direction" if "Direction" in df_lime.columns else None
        fid_r2 = getattr(lime_l, "local_fidelity_r2", None)

        for rank, (_, row) in enumerate(df_lime.iterrows(), start=1):
            f = str(row[feat_col])
            val = float(row[wt_col]) if pd.notnull(row[wt_col]) else 0.0
            cond = str(row[cond_col])
            d = str(row[dir_col]) if dir_col else ("Supports (+)" if val > 0 else "Opposes (-)")
            lookup[f] = {
                "weight": val,
                "condition": cond,
                "direction": d,
                "rank": rank,
                "fidelity_r2": fid_r2,
            }
        return lookup

    def _check_anomaly_concentration(
        self,
        df: pd.DataFrame,
        feature: str,
        comparison_mask: pd.Series,
        anomaly_artifacts: Optional[Any],
    ) -> Dict[str, Any]:
        """Detect whether anomalous observations cluster disproportionately in comparison group."""
        if not pd.api.types.is_numeric_dtype(df[feature]):
            return {"available": False}

        s = df[feature].dropna()
        if len(s) < 10:
            return {"available": False}

        # Check IQR fences
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        if iqr <= 0:
            return {"available": False}

        fence_low = q1 - 1.5 * iqr
        fence_high = q3 + 1.5 * iqr
        is_outlier = (df[feature] < fence_low) | (df[feature] > fence_high)

        comp_outliers = int((is_outlier & comparison_mask).sum())
        total_outliers = int(is_outlier.sum())
        concentration = (comp_outliers / total_outliers * 100.0) if total_outliers > 0 else 0.0

        return {
            "available": True,
            "unusual_count": comp_outliers,
            "total_outliers": total_outliers,
            "concentration_pct": concentration,
        }

