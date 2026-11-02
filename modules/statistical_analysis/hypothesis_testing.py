"""Core Hypothesis Testing framework and result structures."""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd


@dataclass
class HypothesisTestResult:
    """Standardized representation for all statistical hypothesis test outcomes.
    
    Ensures every test provides a research question, hypotheses, statistics,
    decision, effect size, confidence interval, plain-language interpretation,
    and explicit assumptions/limitations.
    """
    test_name: str
    research_question: str
    h0: str
    h1: str
    test_statistic: float
    statistic_name: str
    p_value: float
    alpha: float = 0.05
    is_significant: bool = False
    decision: str = "Fail to reject H0"
    effect_size: Optional[float] = None
    effect_size_name: Optional[str] = None
    effect_size_interpretation: Optional[str] = None
    ci_lower: Optional[float] = None
    ci_upper: Optional[float] = None
    confidence_level: Optional[float] = 0.95
    interpretation: str = ""
    assumptions_met: Dict[str, Any] = field(default_factory=dict)
    limitations: List[str] = field(default_factory=list)
    sample_sizes: Dict[str, int] = field(default_factory=dict)
    additional_metrics: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Derive significance and decision if not explicitly set."""
        if not np.isnan(self.p_value):
            self.is_significant = bool(self.p_value < self.alpha)
            self.decision = "Reject H0 (Statistically Significant)" if self.is_significant else "Fail to Reject H0 (Not Statistically Significant)"

    def to_dict(self) -> Dict[str, Any]:
        """Convert test result to a dictionary representation."""
        return {
            "test_name": self.test_name,
            "research_question": self.research_question,
            "h0": self.h0,
            "h1": self.h1,
            "statistic_name": self.statistic_name,
            "test_statistic": round(self.test_statistic, 4) if not np.isnan(self.test_statistic) else None,
            "p_value": self.p_value,
            "alpha": self.alpha,
            "is_significant": self.is_significant,
            "decision": self.decision,
            "effect_size": round(self.effect_size, 4) if self.effect_size is not None and not np.isnan(self.effect_size) else None,
            "effect_size_name": self.effect_size_name,
            "effect_size_interpretation": self.effect_size_interpretation,
            "ci_lower": round(self.ci_lower, 4) if self.ci_lower is not None and not np.isnan(self.ci_lower) else None,
            "ci_upper": round(self.ci_upper, 4) if self.ci_upper is not None and not np.isnan(self.ci_upper) else None,
            "confidence_level": self.confidence_level,
            "interpretation": self.interpretation,
            "assumptions_met": self.assumptions_met,
            "limitations": self.limitations,
            "sample_sizes": self.sample_sizes,
            "additional_metrics": self.additional_metrics,
        }

    def summary_table(self) -> pd.DataFrame:
        """Return formatted summary table for UI rendering."""
        ci_str = (
            f"[{self.ci_lower:.3f}, {self.ci_upper:.3f}]"
            if self.ci_lower is not None and self.ci_upper is not None
            else "N/A"
        )
        eff_str = (
            f"{self.effect_size:.3f} ({self.effect_size_interpretation or 'Computed'})"
            if self.effect_size is not None
            else "N/A"
        )
        records = [
            {"Parameter": "Statistical Test", "Value": self.test_name},
            {"Parameter": "Research Question", "Value": self.research_question},
            {"Parameter": "Null Hypothesis (H0)", "Value": self.h0},
            {"Parameter": "Alternative Hypothesis (H1)", "Value": self.h1},
            {"Parameter": f"Test Statistic ({self.statistic_name})", "Value": f"{self.test_statistic:.4f}"},
            {"Parameter": "P-Value", "Value": f"{self.p_value:.4e}" if self.p_value < 0.0001 else f"{self.p_value:.4f}"},
            {"Parameter": "Significance Level (α)", "Value": str(self.alpha)},
            {"Parameter": "Statistical Decision", "Value": self.decision},
            {"Parameter": f"Effect Size ({self.effect_size_name or 'N/A'})", "Value": eff_str},
            {"Parameter": f"Confidence Interval ({int((self.confidence_level or 0.95)*100)}%)", "Value": ci_str},
        ]
        return pd.DataFrame(records)

