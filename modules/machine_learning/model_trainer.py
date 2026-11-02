"""Master orchestrator facade for the Machine Learning Engine."""

from typing import Dict, Any, Optional, List, Tuple
import pandas as pd
import numpy as np

from modules.machine_learning.problem_detector import (
    ProblemType,
    TargetSummary,
    detect_problem_type,
)
from modules.machine_learning.ml_preprocessing import (
    MLPreprocessorConfig,
    MLPreprocessingResult,
    select_feature_columns,
    fit_ml_preprocessor,
    encode_target_series,
)
from modules.machine_learning.data_splitter import (
    SplitConfig,
    SplitResult,
    prepare_and_split_data,
)
from modules.machine_learning.model_comparison import (
    run_model_comparison,
    ModelComparisonResult,
)
from modules.machine_learning.feature_importance import (
    extract_feature_importance,
    FeatureImportanceResult,
)
from modules.machine_learning.hyperparameter_tuning import (
    tune_model_hyperparameters,
    TuningResult,
)
from modules.machine_learning.prediction_engine import (
    generate_prediction_exploration,
    PredictionExplorationResult,
)
from modules.machine_learning.ml_insights import (
    generate_ml_insights,
    MLInsight,
)
from modules.machine_learning.model_manager import ModelRegistry, ModelArtifact


class MachineLearningEngine:
    """Unified facade orchestrating end-to-end machine learning workflows."""

    def __init__(self) -> None:
        self.registry = ModelRegistry()

    def inspect_target(
        self,
        df: pd.DataFrame,
        target_col: str,
        force_type: Optional[str] = None,
    ) -> TargetSummary:
        """Inspect target column and detect problem type."""
        return detect_problem_type(df=df, target_col=target_col, force_type=force_type)

    def prepare_and_split(
        self,
        df: pd.DataFrame,
        target_col: str,
        feature_columns: Optional[List[str]] = None,
        split_config: Optional[SplitConfig] = None,
        preprocessor_config: Optional[MLPreprocessorConfig] = None,
        problem_type: ProblemType = ProblemType.BINARY_CLASSIFICATION,
    ) -> Tuple[SplitResult, MLPreprocessingResult, np.ndarray, np.ndarray, Optional[List[str]]]:
        """Orchestrate feature selection, splitting, and strictly leak-free preprocessing.
        
        Returns
        -------
        (split_result, preprocessor_result, y_train_enc, y_test_enc, target_classes)
        """
        cfg_pre = preprocessor_config or MLPreprocessorConfig()
        cfg_split = split_config or SplitConfig()

        # 1. Feature column selection
        num_cols, cat_cols, excluded = select_feature_columns(
            df=df,
            target_col=target_col,
            config=cfg_pre,
            explicit_features=feature_columns,
        )
        active_features = num_cols + cat_cols

        # 2. Split dataset
        split_res = prepare_and_split_data(
            df=df,
            target_col=target_col,
            feature_columns=active_features,
            split_config=cfg_split,
            problem_type=problem_type,
        )

        # 3. Fit preprocessor strictly on training data
        prep_res = fit_ml_preprocessor(
            X_train=split_res.X_train,
            numeric_features=num_cols,
            categorical_features=cat_cols,
            excluded_columns=excluded,
            config=cfg_pre,
        )

        # 4. Target encoding
        is_classification = problem_type in (
            ProblemType.BINARY_CLASSIFICATION,
            ProblemType.MULTICLASS_CLASSIFICATION,
        )
        y_tr_enc, y_te_enc, le, target_classes = encode_target_series(
            y_train=split_res.y_train,
            y_test=split_res.y_test,
            is_classification=is_classification,
        )

        return split_res, prep_res, y_tr_enc, y_te_enc, target_classes

    def benchmark_models(
        self,
        selected_models: List[str],
        split_res: SplitResult,
        prep_res: MLPreprocessingResult,
        y_train_enc: np.ndarray,
        y_test_enc: np.ndarray,
        problem_type: ProblemType,
        target_classes: Optional[List[str]] = None,
        primary_metric: Optional[str] = None,
        handle_imbalance: bool = False,
        random_state: int = 42,
    ) -> ModelComparisonResult:
        """Benchmark multiple machine learning models."""
        return run_model_comparison(
            selected_model_keys=selected_models,
            X_train=split_res.X_train,
            y_train=y_train_enc,
            X_test=split_res.X_test,
            y_test=y_test_enc,
            preprocessor=prep_res.preprocessor,
            problem_type=problem_type,
            labels=target_classes,
            cv_generator=split_res.cv_generator,
            primary_metric=primary_metric,
            handle_imbalance=handle_imbalance,
            random_state=random_state,
        )

    def analyze_feature_importance(
        self,
        estimator: Any,
        prep_res: MLPreprocessingResult,
        split_res: SplitResult,
        y_test_enc: np.ndarray,
        method: str = "auto",
        model_name: str = "Model",
        n_repeats: int = 5,
        random_state: int = 42,
    ) -> FeatureImportanceResult:
        """Compute feature importance scores."""
        X_test_trans = prep_res.preprocessor.transform(split_res.X_test)
        return extract_feature_importance(
            estimator=estimator,
            feature_names=prep_res.feature_names_out,
            X_test=X_test_trans,
            y_test=y_test_enc,
            preferred_method=method,
            model_name=model_name,
            n_repeats=n_repeats,
            random_state=random_state,
        )

    def tune_hyperparameters(
        self,
        model_key: str,
        prep_res: MLPreprocessingResult,
        split_res: SplitResult,
        y_train_enc: np.ndarray,
        problem_type: ProblemType,
        search_type: str = "grid",
        param_grid: Optional[Dict[str, List[Any]]] = None,
        cv_folds: int = 3,
        n_iter: int = 8,
        random_state: int = 42,
    ) -> TuningResult:
        """Tune model hyperparameters."""
        X_train_trans = prep_res.preprocessor.transform(split_res.X_train)
        return tune_model_hyperparameters(
            model_key=model_key,
            X_train_trans=X_train_trans,
            y_train=y_train_enc,
            problem_type=problem_type,
            search_type=search_type,
            custom_param_grid=param_grid,
            cv_folds=cv_folds,
            n_iter=n_iter,
            random_state=random_state,
        )

    def explore_predictions(
        self,
        pipeline: Any,
        split_res: SplitResult,
        y_test_enc: np.ndarray,
        problem_type: ProblemType,
        target_name: str,
        target_classes: Optional[List[str]] = None,
        model_name: str = "Trained Model",
    ) -> PredictionExplorationResult:
        """Generate prediction exploration table."""
        return generate_prediction_exploration(
            pipeline=pipeline,
            X_test=split_res.X_test,
            y_test=y_test_enc,
            target_name=target_name,
            problem_type=problem_type,
            labels=target_classes,
            model_name=model_name,
        )

    def generate_narrative_insights(
        self,
        comparison_res: ModelComparisonResult,
        target_summary: TargetSummary,
        importance_res: Optional[FeatureImportanceResult] = None,
    ) -> List[MLInsight]:
        """Synthesize automated narrative findings."""
        return generate_ml_insights(
            comparison_result=comparison_res,
            target_summary=target_summary,
            importance_result=importance_res,
        )

