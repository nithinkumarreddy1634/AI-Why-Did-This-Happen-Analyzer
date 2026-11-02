"""Configurable change severity classification system."""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Any, Optional


class SeverityLevel(str, Enum):
    """Categorized severity ranking for detected anomalies and changes."""
    NORMAL = "Normal"
    LOW = "Low Deviation"
    MODERATE = "Moderate Deviation"
    HIGH = "High Deviation"
    CRITICAL = "Critical Deviation"


# Color mapping for UI badges
SEVERITY_COLORS: Dict[SeverityLevel, str] = {
    SeverityLevel.NORMAL: "#10B981",    # Emerald
    SeverityLevel.LOW: "#06B6D4",       # Cyan
    SeverityLevel.MODERATE: "#F59E0B",  # Amber
    SeverityLevel.HIGH: "#F97316",      # Orange
    SeverityLevel.CRITICAL: "#EF4444",  # Crimson Red
}


@dataclass
class SeverityThresholds:
    """Configurable threshold boundaries for determining change severity.
    
    Attributes:
        low_pct: Minimum absolute % change for Low severity (default 10%)
        mod_pct: Minimum absolute % change for Moderate severity (default 25%)
        high_pct: Minimum absolute % change for High severity (default 50%)
        crit_pct: Minimum absolute % change for Critical severity (default 100%)
        
        low_std: Standard deviation shift for Low (default 1.0)
        mod_std: Standard deviation shift for Moderate (default 2.0)
        high_std: Standard deviation shift for High (default 3.0)
        crit_std: Standard deviation shift for Critical (default 4.0)
        
        low_pct_points: Absolute percentage point shift for proportions (e.g. 5%)
        mod_pct_points: Absolute percentage point shift for proportions (e.g. 15%)
        high_pct_points: Absolute percentage point shift for proportions (e.g. 25%)
        crit_pct_points: Absolute percentage point shift for proportions (e.g. 40%)
    """
    low_pct: float = 10.0
    mod_pct: float = 25.0
    high_pct: float = 50.0
    crit_pct: float = 100.0

    low_std: float = 1.0
    mod_std: float = 2.0
    high_std: float = 3.0
    crit_std: float = 4.0

    low_pct_points: float = 5.0
    mod_pct_points: float = 12.0
    high_pct_points: float = 25.0
    crit_pct_points: float = 40.0


class SeverityClassifier:
    """Evaluates metrics and assigns severity levels with explicit justification."""

    def __init__(self, thresholds: Optional[SeverityThresholds] = None) -> None:
        self.thresholds = thresholds or SeverityThresholds()

    def classify_percentage_change(
        self, abs_pct_change: Optional[float]
    ) -> tuple[SeverityLevel, str]:
        """Classify severity based on relative percentage change."""
        if abs_pct_change is None:
            return SeverityLevel.NORMAL, "Relative percentage change cannot be determined (zero/undefined baseline)."
        
        val = abs(abs_pct_change)
        t = self.thresholds

        if val >= t.crit_pct:
            return (
                SeverityLevel.CRITICAL,
                f"Absolute change of {val:.1f}% exceeds Critical threshold (>= {t.crit_pct}%).",
            )
        elif val >= t.high_pct:
            return (
                SeverityLevel.HIGH,
                f"Absolute change of {val:.1f}% exceeds High threshold (>= {t.high_pct}%).",
            )
        elif val >= t.mod_pct:
            return (
                SeverityLevel.MODERATE,
                f"Absolute change of {val:.1f}% exceeds Moderate threshold (>= {t.mod_pct}%).",
            )
        elif val >= t.low_pct:
            return (
                SeverityLevel.LOW,
                f"Absolute change of {val:.1f}% exceeds Low threshold (>= {t.low_pct}%).",
            )
        else:
            return (
                SeverityLevel.NORMAL,
                f"Absolute change of {val:.1f}% is below Low threshold (< {t.low_pct}%).",
            )

    def classify_z_score(self, abs_z_score: float) -> tuple[SeverityLevel, str]:
        """Classify severity based on standardized Z-score deviations."""
        val = abs(abs_z_score)
        t = self.thresholds

        if val >= t.crit_std:
            return (
                SeverityLevel.CRITICAL,
                f"Z-score |{val:.2f}| exceeds Critical standard deviation threshold (>= {t.crit_std}σ).",
            )
        elif val >= t.high_std:
            return (
                SeverityLevel.HIGH,
                f"Z-score |{val:.2f}| exceeds High standard deviation threshold (>= {t.high_std}σ).",
            )
        elif val >= t.mod_std:
            return (
                SeverityLevel.MODERATE,
                f"Z-score |{val:.2f}| exceeds Moderate threshold (>= {t.mod_std}σ).",
            )
        elif val >= t.low_std:
            return (
                SeverityLevel.LOW,
                f"Z-score |{val:.2f}| exceeds Low threshold (>= {t.low_std}σ).",
            )
        else:
            return (
                SeverityLevel.NORMAL,
                f"Z-score |{val:.2f}| conforms to normal expected variance (< {t.low_std}σ).",
            )

    def classify_percentage_point_shift(
        self, abs_pp_shift: float
    ) -> tuple[SeverityLevel, str]:
        """Classify proportion shifts (e.g., churn rate 10% -> 25% = 15 percentage points)."""
        val = abs(abs_pp_shift)
        t = self.thresholds

        if val >= t.crit_pct_points:
            return (
                SeverityLevel.CRITICAL,
                f"Proportion shifted by {val:.1f} percentage points, exceeding Critical limit (>= {t.crit_pct_points} pp).",
            )
        elif val >= t.high_pct_points:
            return (
                SeverityLevel.HIGH,
                f"Proportion shifted by {val:.1f} percentage points, exceeding High limit (>= {t.high_pct_points} pp).",
            )
        elif val >= t.mod_pct_points:
            return (
                SeverityLevel.MODERATE,
                f"Proportion shifted by {val:.1f} percentage points, exceeding Moderate limit (>= {t.mod_pct_points} pp).",
            )
        elif val >= t.low_pct_points:
            return (
                SeverityLevel.LOW,
                f"Proportion shifted by {val:.1f} percentage points, exceeding Low limit (>= {t.low_pct_points} pp).",
            )
        else:
            return (
                SeverityLevel.NORMAL,
                f"Proportion shift of {val:.1f} percentage points is within normal expected variance (< {t.low_pct_points} pp).",
            )
