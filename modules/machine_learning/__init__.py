"""Machine Learning Engine package for Phase 7 predictive analytics."""

from modules.machine_learning.problem_detector import (
    ProblemType,
    TargetSummary,
    detect_problem_type,
)
from modules.machine_learning.ml_preprocessing import (
    MLPreprocessorConfig,
    MLPreprocessingResult,
    select_feature_columns,
    build_column_transformer,
    fit_ml_preprocessor,
    encode_target_series,
)
from modules.machine_learning.data_splitter import (
    SplitConfig,
    SplitResult,
    prepare_and_split_data,
)
from modules.machine_learning.classification import (
    CLASSIFICATION_MODELS_REGISTRY,
    get_classification_model,
    HAS_XGBOOST,
)
from modules.machine_learning.regression import (
    REGRESSION_MODELS_REGISTRY,
    get_regression_model,
)
from modules.machine_learning.model_evaluator import (
    ClassificationMetrics,
    RegressionMetrics,
    evaluate_classification_model,
    evaluate_regression_model,
)
from modules.machine_learning.model_comparison import (
    ModelComparisonResult,
    run_model_comparison,
)
from modules.machine_learning.feature_importance import (
    FeatureImportanceResult,
    extract_feature_importance,
)
from modules.machine_learning.hyperparameter_tuning import (
    TuningResult,
    tune_model_hyperparameters,
    DEFAULT_CLASSIFICATION_PARAM_GRIDS,
    DEFAULT_REGRESSION_PARAM_GRIDS,
)
from modules.machine_learning.prediction_engine import (
    PredictionExplorationResult,
    generate_prediction_exploration,
)
from modules.machine_learning.model_manager import (
    ModelArtifact,
    ModelRegistry,
)
from modules.machine_learning.ml_insights import (
    MLInsight,
    generate_ml_insights,
)
from modules.machine_learning.model_trainer import MachineLearningEngine

__all__ = [
    "ProblemType",
    "TargetSummary",
    "detect_problem_type",
    "MLPreprocessorConfig",
    "MLPreprocessingResult",
    "select_feature_columns",
    "build_column_transformer",
    "fit_ml_preprocessor",
    "encode_target_series",
    "SplitConfig",
    "SplitResult",
    "prepare_and_split_data",
    "CLASSIFICATION_MODELS_REGISTRY",
    "REGRESSION_MODELS_REGISTRY",
    "get_classification_model",
    "get_regression_model",
    "HAS_XGBOOST",
    "ClassificationMetrics",
    "RegressionMetrics",
    "evaluate_classification_model",
    "evaluate_regression_model",
    "ModelComparisonResult",
    "run_model_comparison",
    "FeatureImportanceResult",
    "extract_feature_importance",
    "TuningResult",
    "tune_model_hyperparameters",
    "DEFAULT_CLASSIFICATION_PARAM_GRIDS",
    "DEFAULT_REGRESSION_PARAM_GRIDS",
    "PredictionExplorationResult",
    "generate_prediction_exploration",
    "ModelArtifact",
    "ModelRegistry",
    "MLInsight",
    "generate_ml_insights",
    "MachineLearningEngine",
]
