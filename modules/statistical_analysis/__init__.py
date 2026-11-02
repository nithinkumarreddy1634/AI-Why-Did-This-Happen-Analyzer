"""Statistical Investigation and Hypothesis Testing Engine (Phase 6)."""

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
from modules.statistical_analysis.effect_sizes import (
    compute_cohens_d,
    compute_hedges_g,
    compute_paired_cohens_d,
    compute_eta_squared,
    compute_cramers_v,
    interpret_cohens_d,
    interpret_eta_squared,
    interpret_cramers_v,
)
from modules.statistical_analysis.confidence_intervals import (
    ci_difference_in_means,
    ci_paired_difference,
    ci_difference_in_proportions,
    ci_pearson_correlation,
)
from modules.statistical_analysis.multiple_testing import (
    bonferroni_correction,
    benjamini_hochberg_correction,
    adjust_test_results,
)
from modules.statistical_analysis.test_recommender import recommend_statistical_test
from modules.statistical_analysis.result_interpreter import StatisticalResultInterpreter
from modules.statistical_analysis.statistical_engine import StatisticalEngine

__all__ = [
    "StatisticalEngine",
    "HypothesisTestResult",
    "independent_t_test",
    "paired_t_test",
    "one_way_anova",
    "mann_whitney_u_test",
    "wilcoxon_signed_rank_test",
    "kruskal_wallis_test",
    "chi_square_independence_test",
    "fisher_exact_test",
    "two_proportion_test",
    "compute_pearson_correlation",
    "compute_spearman_correlation",
    "compute_kendall_correlation",
    "check_normality",
    "check_equal_variance",
    "generate_qq_plot_data",
    "compute_cohens_d",
    "compute_hedges_g",
    "compute_paired_cohens_d",
    "compute_eta_squared",
    "compute_cramers_v",
    "interpret_cohens_d",
    "interpret_eta_squared",
    "interpret_cramers_v",
    "ci_difference_in_means",
    "ci_paired_difference",
    "ci_difference_in_proportions",
    "ci_pearson_correlation",
    "bonferroni_correction",
    "benjamini_hochberg_correction",
    "adjust_test_results",
    "recommend_statistical_test",
    "StatisticalResultInterpreter",
]
