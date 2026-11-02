"""Unified Statistical Engine facade managing hypothesis testing and statistical investigations."""

from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np

from modules.statistical_analysis.hypothesis_testing import HypothesisTestResult
from modules.statistical_analysis.numerical_tests import (
    independent_t_test,
    paired_t_test,
    one_way_anova,
    mann_whitney_u_test,
    wilcoxon_signed_rank_test,
    kruskal_wallis_test,
)
from modules.statistical_analysis.categorical_tests import (
    chi_square_independence_test,
    fisher_exact_test,
    two_proportion_test,
)
from modules.statistical_analysis.correlation_tests import (
    compute_pearson_correlation,
    compute_spearman_correlation,
    compute_kendall_correlation,
)
from modules.statistical_analysis.assumption_checks import (
    check_normality,
    check_equal_variance,
    generate_qq_plot_data,
)
from modules.statistical_analysis.multiple_testing import (
    adjust_test_results,
    bonferroni_correction,
    benjamini_hochberg_correction,
)
from modules.statistical_analysis.test_recommender import recommend_statistical_test
from modules.statistical_analysis.result_interpreter import StatisticalResultInterpreter


class StatisticalEngine:
    """Master facade coordinating hypothesis testing, effect sizing, and multiple comparison adjustments."""

    def __init__(self, default_alpha: float = 0.05) -> None:
        self.default_alpha = default_alpha
        self.history: List[HypothesisTestResult] = []

    def record_test(self, result: HypothesisTestResult) -> HypothesisTestResult:
        """Log a completed hypothesis test to engine history."""
        self.history.append(result)
        return result

    def clear_history(self) -> None:
        """Clear test history log."""
        self.history.clear()

    # 1. Recommendation
    def recommend(
        self,
        df: pd.DataFrame,
        target_col: str,
        predictor_col: Optional[str] = None,
        study_design: str = "independent",
    ) -> Dict[str, Any]:
        """Provide automatic statistical test recommendation."""
        return recommend_statistical_test(df, target_col, predictor_col, study_design)

    # 2. Numerical Tests
    def run_independent_t_test(
        self,
        group1: pd.Series,
        group2: pd.Series,
        group1_name: str = "Group 1",
        group2_name: str = "Group 2",
        variable_name: str = "Metric",
        alpha: Optional[float] = None,
        equal_var: str = "auto",
    ) -> HypothesisTestResult:
        """Execute independent two-sample t-test."""
        res = independent_t_test(
            group1, group2, group1_name, group2_name, variable_name, alpha or self.default_alpha, equal_var
        )
        return self.record_test(res)

    def run_paired_t_test(
        self,
        before_series: pd.Series,
        after_series: pd.Series,
        before_name: str = "Pre",
        after_name: str = "Post",
        variable_name: str = "Metric",
        alpha: Optional[float] = None,
    ) -> HypothesisTestResult:
        """Execute paired samples t-test."""
        res = paired_t_test(
            before_series, after_series, before_name, after_name, variable_name, alpha or self.default_alpha
        )
        return self.record_test(res)

    def run_anova(
        self,
        groups_dict: Dict[str, pd.Series],
        outcome_name: str = "Metric",
        group_col_name: str = "Group",
        alpha: Optional[float] = None,
    ) -> HypothesisTestResult:
        """Execute One-Way ANOVA."""
        res = one_way_anova(groups_dict, outcome_name, group_col_name, alpha or self.default_alpha)
        return self.record_test(res)

    def run_mann_whitney(
        self,
        group1: pd.Series,
        group2: pd.Series,
        group1_name: str = "Group 1",
        group2_name: str = "Group 2",
        variable_name: str = "Metric",
        alpha: Optional[float] = None,
    ) -> HypothesisTestResult:
        """Execute non-parametric Mann-Whitney U test."""
        res = mann_whitney_u_test(
            group1, group2, group1_name, group2_name, variable_name, alpha or self.default_alpha
        )
        return self.record_test(res)

    def run_wilcoxon(
        self,
        before_series: pd.Series,
        after_series: pd.Series,
        variable_name: str = "Metric",
        alpha: Optional[float] = None,
    ) -> HypothesisTestResult:
        """Execute non-parametric Wilcoxon Signed-Rank test."""
        res = wilcoxon_signed_rank_test(before_series, after_series, variable_name, alpha or self.default_alpha)
        return self.record_test(res)

    def run_kruskal_wallis(
        self,
        groups_dict: Dict[str, pd.Series],
        outcome_name: str = "Metric",
        group_col_name: str = "Group",
        alpha: Optional[float] = None,
    ) -> HypothesisTestResult:
        """Execute non-parametric Kruskal-Wallis test."""
        res = kruskal_wallis_test(groups_dict, outcome_name, group_col_name, alpha or self.default_alpha)
        return self.record_test(res)

    # 3. Categorical Tests
    def run_chi_square(
        self,
        series1: pd.Series,
        series2: pd.Series,
        col1_name: str = "Variable 1",
        col2_name: str = "Variable 2",
        alpha: Optional[float] = None,
    ) -> HypothesisTestResult:
        """Execute Chi-Square Test of Independence."""
        res = chi_square_independence_test(series1, series2, col1_name, col2_name, alpha or self.default_alpha)
        return self.record_test(res)

    def run_fisher_exact(
        self,
        table_2x2: pd.DataFrame,
        col1_name: str = "Row",
        col2_name: str = "Col",
        alpha: Optional[float] = None,
    ) -> HypothesisTestResult:
        """Execute Fisher's Exact Test."""
        res = fisher_exact_test(table_2x2, col1_name, col2_name, alpha or self.default_alpha)
        return self.record_test(res)

    def run_two_proportion_test(
        count1: int,
        nobs1: int,
        count2: int,
        nobs2: int,
        group1_name: str = "Group 1",
        group2_name: str = "Group 2",
        outcome_label: str = "Rate",
        alpha: Optional[float] = None,
    ) -> HypothesisTestResult:
        """Execute Two-Proportion Z-Test."""
        return two_proportion_test(
            count1, nobs1, count2, nobs2, group1_name, group2_name, outcome_label, alpha or 0.05
        )

    # 4. Correlation Tests
    def run_correlation(
        self,
        x: pd.Series,
        y: pd.Series,
        x_name: str = "X",
        y_name: str = "Y",
        method: str = "pearson",
        alpha: Optional[float] = None,
    ) -> HypothesisTestResult:
        """Execute correlation significance test (pearson, spearman, or kendall)."""
        a = alpha or self.default_alpha
        if method.lower() == "spearman":
            res = compute_spearman_correlation(x, y, x_name, y_name, a)
        elif method.lower() == "kendall":
            res = compute_kendall_correlation(x, y, x_name, y_name, a)
        else:
            res = compute_pearson_correlation(x, y, x_name, y_name, a)
        return self.record_test(res)

    # 5. Assumption Checks
    def check_normality(self, series: pd.Series, alpha: Optional[float] = None) -> Dict[str, Any]:
        """Test normality using Shapiro-Wilk or D'Agostino."""
        return check_normality(series, alpha or self.default_alpha)

    def check_variance(self, groups: List[pd.Series], alpha: Optional[float] = None, center: str = "median") -> Dict[str, Any]:
        """Test homogeneity of variance using Levene's test."""
        return check_equal_variance(groups, alpha or self.default_alpha, center=center)

    def get_qq_plot(self, series: pd.Series) -> Tuple[np.ndarray, np.ndarray, float, float]:
        """Generate Q-Q plot theoretical and empirical quantiles."""
        return generate_qq_plot_data(series)

    # 6. Multiple Comparisons
    def apply_multiple_testing_correction(
        self, method: str = "benjamini_hochberg", alpha: Optional[float] = None
    ) -> pd.DataFrame:
        """Apply Bonferroni or Benjamini-Hochberg adjustment to recorded tests."""
        return adjust_test_results(self.history, method=method, alpha=alpha or self.default_alpha)

    # 7. Narrative Interpretation
    def interpret(self, result: HypothesisTestResult) -> Dict[str, Any]:
        """Generate structured narrative explanation for a test result."""
        return StatisticalResultInterpreter.interpret_test(result)
