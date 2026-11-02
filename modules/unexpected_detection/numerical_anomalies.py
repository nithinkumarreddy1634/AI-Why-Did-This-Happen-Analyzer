"""Numerical deviation and anomaly detection engine."""

from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np

from modules.unexpected_detection.severity_classifier import (
    SeverityLevel,
    SeverityClassifier,
    SeverityThresholds,
)


class NumericalAnomalyDetector:
    """Detects statistical outliers and extreme deviations across numerical series."""

    def __init__(
        self,
        classifier: Optional[SeverityClassifier] = None,
    ) -> None:
        self.classifier = classifier or SeverityClassifier()

    def detect_zscore(
        self,
        series: pd.Series,
        threshold: float = 3.0,
    ) -> Dict[str, Any]:
        """Detect outliers using standard Z-score (number of standard deviations from mean)."""
        clean = series.dropna()
        n_total = len(series)
        n_valid = len(clean)

        if n_valid < 3:
            return self._empty_result("Z-Score", n_total, n_valid, "Insufficient observations (minimum 3 required).")

        std = float(clean.std(ddof=1))
        if std == 0 or np.isnan(std):
            return self._empty_result("Z-Score", n_total, n_valid, "Zero variance (all values are identical).")

        mean = float(clean.mean())
        z_scores = (clean - mean) / std
        anom_mask = z_scores.abs() > threshold

        anomaly_df = pd.DataFrame({
            "original_index": clean[anom_mask].index,
            "value": clean[anom_mask].values,
            "z_score": z_scores[anom_mask].values,
            "deviation_from_mean": (clean[anom_mask] - mean).values,
        })

        # Assign severity to each anomaly
        anomaly_df["severity"] = anomaly_df["z_score"].apply(
            lambda z: self.classifier.classify_z_score(z)[0].value
        )

        n_anom = int(anom_mask.sum())
        pct_anom = (n_anom / n_valid) * 100.0 if n_valid > 0 else 0.0

        return {
            "method": "Z-Score",
            "threshold": threshold,
            "total_observations": n_total,
            "valid_observations": n_valid,
            "missing_observations": n_total - n_valid,
            "unusual_count": n_anom,
            "unusual_pct": pct_anom,
            "baseline_stats": {
                "mean": mean,
                "std": std,
                "min": float(clean.min()),
                "max": float(clean.max()),
                "lower_bound": mean - threshold * std,
                "upper_bound": mean + threshold * std,
            },
            "anomalies": anomaly_df,
            "status": "success",
            "message": f"Identified {n_anom:,} anomalous points ({pct_anom:.2f}%) beyond {threshold}σ.",
        }

    def detect_modified_zscore(
        self,
        series: pd.Series,
        threshold: float = 3.5,
    ) -> Dict[str, Any]:
        """Detect outliers using Boris Iglewicz and David Hoaglin's Modified Z-score.
        
        Formula: 0.6745 * (x - median) / MAD
        Robust against heavy skewness and extreme outliers.
        """
        clean = series.dropna()
        n_total = len(series)
        n_valid = len(clean)

        if n_valid < 3:
            return self._empty_result("Modified Z-Score (MAD)", n_total, n_valid, "Insufficient observations.")

        median = float(clean.median())
        abs_deviation = (clean - median).abs()
        mad = float(abs_deviation.median())

        if mad == 0 or np.isnan(mad):
            # Fall back to mean absolute deviation or note constant median
            mad = float(abs_deviation.mean())
            if mad == 0:
                return self._empty_result("Modified Z-Score (MAD)", n_total, n_valid, "Zero MAD variance.")

        mod_z = 0.6745 * (clean - median) / mad
        anom_mask = mod_z.abs() > threshold

        anomaly_df = pd.DataFrame({
            "original_index": clean[anom_mask].index,
            "value": clean[anom_mask].values,
            "modified_z_score": mod_z[anom_mask].values,
            "deviation_from_median": (clean[anom_mask] - median).values,
        })

        anomaly_df["severity"] = anomaly_df["modified_z_score"].apply(
            lambda mz: self.classifier.classify_z_score(mz)[0].value
        )

        n_anom = int(anom_mask.sum())
        pct_anom = (n_anom / n_valid) * 100.0 if n_valid > 0 else 0.0

        return {
            "method": "Modified Z-Score (MAD)",
            "threshold": threshold,
            "total_observations": n_total,
            "valid_observations": n_valid,
            "missing_observations": n_total - n_valid,
            "unusual_count": n_anom,
            "unusual_pct": pct_anom,
            "baseline_stats": {
                "median": median,
                "mad": mad,
                "lower_bound": median - (threshold * mad / 0.6745),
                "upper_bound": median + (threshold * mad / 0.6745),
            },
            "anomalies": anomaly_df,
            "status": "success",
            "message": f"Identified {n_anom:,} anomalous points ({pct_anom:.2f}%) using robust MAD.",
        }

    def detect_iqr(
        self,
        series: pd.Series,
        multiplier: float = 1.5,
    ) -> Dict[str, Any]:
        """Detect outliers using Tukey's Interquartile Range (IQR) fence."""
        clean = series.dropna()
        n_total = len(series)
        n_valid = len(clean)

        if n_valid < 4:
            return self._empty_result("IQR Detection", n_total, n_valid, "Insufficient observations for quartiles.")

        q1 = float(clean.quantile(0.25))
        q3 = float(clean.quantile(0.75))
        iqr = q3 - q1

        lower_fence = q1 - (multiplier * iqr)
        upper_fence = q3 + (multiplier * iqr)

        anom_mask = (clean < lower_fence) | (clean > upper_fence)

        anomaly_df = pd.DataFrame({
            "original_index": clean[anom_mask].index,
            "value": clean[anom_mask].values,
            "bound_exceeded": np.where(clean[anom_mask] < lower_fence, "Below Lower Fence", "Above Upper Fence"),
            "distance_from_fence": np.where(
                clean[anom_mask] < lower_fence,
                lower_fence - clean[anom_mask],
                clean[anom_mask] - upper_fence,
            ),
        })

        # Calculate severity based on distance in IQR units
        def iqr_severity(dist: float) -> str:
            units = dist / (iqr if iqr > 0 else 1.0)
            if units >= 2.0:
                return SeverityLevel.CRITICAL.value
            elif units >= 1.0:
                return SeverityLevel.HIGH.value
            elif units >= 0.5:
                return SeverityLevel.MODERATE.value
            return SeverityLevel.LOW.value

        if not anomaly_df.empty:
            anomaly_df["severity"] = anomaly_df["distance_from_fence"].apply(iqr_severity)
        else:
            anomaly_df["severity"] = []

        n_anom = int(anom_mask.sum())
        pct_anom = (n_anom / n_valid) * 100.0 if n_valid > 0 else 0.0

        return {
            "method": "IQR (Interquartile Range)",
            "threshold": multiplier,
            "total_observations": n_total,
            "valid_observations": n_valid,
            "missing_observations": n_total - n_valid,
            "unusual_count": n_anom,
            "unusual_pct": pct_anom,
            "baseline_stats": {
                "q1": q1,
                "median": float(clean.median()),
                "q3": q3,
                "iqr": iqr,
                "lower_bound": lower_fence,
                "upper_bound": upper_fence,
            },
            "anomalies": anomaly_df,
            "status": "success",
            "message": f"Identified {n_anom:,} points ({pct_anom:.2f}%) outside [{lower_fence:.2f}, {upper_fence:.2f}].",
        }

    def detect_pct_deviation(
        self,
        series: pd.Series,
        pct_threshold: float = 50.0,
        reference_stat: str = "median",
    ) -> Dict[str, Any]:
        """Detect values deviating by more than threshold percentage from mean or median."""
        clean = series.dropna()
        n_total = len(series)
        n_valid = len(clean)

        if n_valid < 2:
            return self._empty_result(f"% Deviation ({reference_stat})", n_total, n_valid, "Insufficient observations.")

        ref_val = float(clean.median() if reference_stat == "median" else clean.mean())
        if ref_val == 0:
            return self._empty_result(
                f"% Deviation ({reference_stat})", n_total, n_valid, "Reference baseline value is exactly 0."
            )

        pct_diff = ((clean - ref_val).abs() / abs(ref_val)) * 100.0
        anom_mask = pct_diff > pct_threshold

        anomaly_df = pd.DataFrame({
            "original_index": clean[anom_mask].index,
            "value": clean[anom_mask].values,
            "pct_deviation": pct_diff[anom_mask].values,
            "direction": np.where(clean[anom_mask] > ref_val, "Positive Deviation", "Negative Deviation"),
        })

        if not anomaly_df.empty:
            anomaly_df["severity"] = anomaly_df["pct_deviation"].apply(
                lambda p: self.classifier.classify_percentage_change(p)[0].value
            )
        else:
            anomaly_df["severity"] = []

        n_anom = int(anom_mask.sum())
        pct_anom = (n_anom / n_valid) * 100.0 if n_valid > 0 else 0.0

        return {
            "method": f"% Deviation from {reference_stat.capitalize()}",
            "threshold": pct_threshold,
            "total_observations": n_total,
            "valid_observations": n_valid,
            "missing_observations": n_total - n_valid,
            "unusual_count": n_anom,
            "unusual_pct": pct_anom,
            "baseline_stats": {
                "reference_metric": reference_stat,
                "reference_value": ref_val,
                "lower_bound": ref_val - (ref_val * pct_threshold / 100.0),
                "upper_bound": ref_val + (ref_val * pct_threshold / 100.0),
            },
            "anomalies": anomaly_df,
            "status": "success",
            "message": f"Identified {n_anom:,} points ({pct_anom:.2f}%) deviating > {pct_threshold}%.",
        }

    def detect_std_threshold(
        self,
        series: pd.Series,
        num_std: float = 2.0,
    ) -> Dict[str, Any]:
        """Detect outliers beyond a configurable number of standard deviations from the mean."""
        clean = series.dropna()
        n_total = len(series)
        n_valid = len(clean)

        if n_valid < 3:
            return self._empty_result("Standard Deviation Threshold", n_total, n_valid, "Insufficient observations (minimum 3 required).")

        std = float(clean.std(ddof=1))
        if std == 0 or np.isnan(std):
            return self._empty_result("Standard Deviation Threshold", n_total, n_valid, "Zero variance (all values are identical).")

        mean = float(clean.mean())
        lower_bound = mean - (num_std * std)
        upper_bound = mean + (num_std * std)
        anom_mask = (clean < lower_bound) | (clean > upper_bound)
        z_scores = (clean - mean) / std

        anomaly_df = pd.DataFrame({
            "original_index": clean[anom_mask].index,
            "value": clean[anom_mask].values,
            "std_deviations": z_scores[anom_mask].values,
            "distance_from_mean": (clean[anom_mask] - mean).values,
            "bound_exceeded": np.where(clean[anom_mask] < lower_bound, "Below Lower Limit", "Above Upper Limit"),
        })

        if not anomaly_df.empty:
            anomaly_df["severity"] = anomaly_df["std_deviations"].apply(
                lambda z: self.classifier.classify_z_score(z)[0].value
            )
        else:
            anomaly_df["severity"] = []

        n_anom = int(anom_mask.sum())
        pct_anom = (n_anom / n_valid) * 100.0 if n_valid > 0 else 0.0

        return {
            "method": "Standard Deviation Threshold",
            "threshold": num_std,
            "total_observations": n_total,
            "valid_observations": n_valid,
            "missing_observations": n_total - n_valid,
            "unusual_count": n_anom,
            "unusual_pct": pct_anom,
            "baseline_stats": {
                "mean": mean,
                "std": std,
                "min": float(clean.min()),
                "max": float(clean.max()),
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
            },
            "anomalies": anomaly_df,
            "status": "success",
            "message": f"Identified {n_anom:,} anomalous points ({pct_anom:.2f}%) outside [{lower_bound:.2f}, {upper_bound:.2f}] ({num_std}σ).",
        }

    def _empty_result(
        self, method: str, total: int, valid: int, msg: str
    ) -> Dict[str, Any]:
        """Construct empty standard result structure."""
        return {
            "method": method,
            "threshold": 0.0,
            "total_observations": total,
            "valid_observations": valid,
            "missing_observations": total - valid,
            "unusual_count": 0,
            "unusual_pct": 0.0,
            "baseline_stats": {},
            "anomalies": pd.DataFrame(),
            "status": "warning",
            "message": msg,
        }
