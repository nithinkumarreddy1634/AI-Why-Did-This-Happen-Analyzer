"""Model comparison engine for multi-algorithm benchmarking."""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
import time
import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score

from modules.machine_learning.problem_detector import ProblemType
from modules.machine_learning.classification import (
    get_classification_model,
    CLASSIFICATION_MODELS_REGISTRY,
)
from modules.machine_learning.regression import (
    get_regression_model,
    REGRESSION_MODELS_REGISTRY,
)
from modules.machine_learning.model_evaluator import (
    evaluate_classification_model,
    evaluate_regression_model,
    ClassificationMetrics,
    RegressionMetrics,
)


@dataclass
class ModelComparisonResult:
    """Container holding multi-model benchmark results and comparison matrix."""
    comparison_table: pd.DataFrame
    results_by_model: Dict[str, Any]
    best_model_key: str
    primary_metric: str
    problem_type: ProblemType
    models_evaluated: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "best_model_key": self.best_model_key,
            "primary_metric": self.primary_metric,
            "problem_type": self.problem_type.value,
            "models_evaluated": self.models_evaluated,
            "summary_records": self.comparison_table.to_dict(orient="records"),
        }


def run_model_comparison(
    selected_model_keys: List[str],
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    preprocessor: Any,
    problem_type: ProblemType,
    labels: Optional[List[str]] = None,
    cv_generator: Optional[Any] = None,
    primary_metric: Optional[str] = None,
    handle_imbalance: bool = False,
    random_state: int = 42,
) -> ModelComparisonResult:
    """Train, evaluate, and benchmark multiple selected machine learning models.
    
    Parameters
    ----------
    selected_model_keys : List[str]
        Keys of models to benchmark.
    X_train, y_train : Training features and targets.
    X_test, y_test : Testing features and targets.
    preprocessor : Fitted scikit-learn ColumnTransformer.
    problem_type : Binary/Multiclass Classification or Regression.
    labels : Optional class label names for classification.
    cv_generator : Cross-validation split generator.
    primary_metric : Metric used to rank models.
    handle_imbalance : Flag to adjust class weights.
    random_state : Random seed for reproducibility.
    """
    is_classification = problem_type in (
        ProblemType.BINARY_CLASSIFICATION,
        ProblemType.MULTICLASS_CLASSIFICATION,
    )

    if not selected_model_keys:
        default_keys = (
            ["logistic_regression", "random_forest", "decision_tree"]
            if is_classification
            else ["linear_regression", "ridge", "random_forest"]
        )
        selected_model_keys = default_keys

    # Default ranking metric
    if not primary_metric:
        primary_metric = "F1-Score (Macro)" if is_classification else "RMSE"

    results_by_model: Dict[str, Any] = {}
    table_rows: List[Dict[str, Any]] = []

    # Pre-transform data with the fitted preprocessor for speed during CV
    X_tr_trans = preprocessor.transform(X_train)
    X_te_trans = preprocessor.transform(X_test)
    n_features = X_tr_trans.shape[1]

    for model_key in selected_model_keys:
        t0 = time.time()
        
        # 1. Instantiate model
        if is_classification:
            estimator = get_classification_model(
                model_key=model_key,
                random_state=random_state,
                handle_imbalance=handle_imbalance,
            )
            display_name = CLASSIFICATION_MODELS_REGISTRY.get(model_key, {}).get("display_name", model_key)
            scoring_cv = "f1_macro" if is_classification else "neg_root_mean_squared_error"
        else:
            estimator = get_regression_model(
                model_key=model_key,
                random_state=random_state,
            )
            display_name = REGRESSION_MODELS_REGISTRY.get(model_key, {}).get("display_name", model_key)
            scoring_cv = "neg_root_mean_squared_error"

        # 2. Cross-Validation on training data
        cv_scores = None
        if cv_generator is not None:
            try:
                raw_cv = cross_val_score(estimator, X_tr_trans, y_train, cv=cv_generator, scoring=scoring_cv, n_jobs=-1)
                cv_scores = np.abs(raw_cv) if not is_classification else raw_cv
            except Exception:
                cv_scores = None

        # 3. Fit estimator
        estimator.fit(X_tr_trans, y_train)
        duration = time.time() - t0

        # 4. Predict on holdout test set
        y_pred = estimator.predict(X_te_trans)

        # 5. Build full end-to-end Pipeline for easy inference
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", estimator),
        ])

        # 6. Evaluate
        if is_classification:
            y_proba = None
            if hasattr(estimator, "predict_proba"):
                try:
                    y_proba = estimator.predict_proba(X_te_trans)
                except Exception:
                    y_proba = None

            eval_res = evaluate_classification_model(
                y_true=y_test,
                y_pred=y_pred,
                y_proba=y_proba,
                labels=labels,
                training_duration_sec=duration,
                cv_scores=cv_scores,
            )
            row = {
                "Model Key": model_key,
                "Model": display_name,
                "Accuracy": eval_res.accuracy,
                "F1-Score (Macro)": eval_res.f1_macro,
                "F1-Score (Weighted)": eval_res.f1_weighted,
                "Precision (Macro)": eval_res.precision_macro,
                "Recall (Macro)": eval_res.recall_macro,
                "ROC-AUC": eval_res.roc_auc if eval_res.roc_auc is not None else np.nan,
                "PR-AUC": eval_res.pr_auc if eval_res.pr_auc is not None else np.nan,
                "CV Mean": eval_res.cv_mean if eval_res.cv_mean is not None else np.nan,
                "CV Std": eval_res.cv_std if eval_res.cv_std is not None else np.nan,
                "Training Time (s)": round(duration, 3),
            }
        else:
            eval_res = evaluate_regression_model(
                y_true=y_test,
                y_pred=y_pred,
                n_features=n_features,
                training_duration_sec=duration,
                cv_scores=cv_scores,
            )
            row = {
                "Model Key": model_key,
                "Model": display_name,
                "MAE": eval_res.mae,
                "RMSE": eval_res.rmse,
                "MSE": eval_res.mse,
                "R²": eval_res.r2,
                "Adjusted R²": eval_res.adjusted_r2,
                "MAPE (%)": eval_res.mape if eval_res.mape is not None else np.nan,
                "CV Mean": eval_res.cv_mean if eval_res.cv_mean is not None else np.nan,
                "CV Std": eval_res.cv_std if eval_res.cv_std is not None else np.nan,
                "Training Time (s)": round(duration, 3),
            }

        table_rows.append(row)
        results_by_model[model_key] = {
            "model_key": model_key,
            "display_name": display_name,
            "pipeline": pipeline,
            "estimator": estimator,
            "evaluation": eval_res,
            "duration": duration,
        }

    df_comp = pd.DataFrame(table_rows)

    # 7. Sort and determine best model
    # For error metrics (MAE, RMSE, MSE), lower is better. For others, higher is better.
    lower_is_better = primary_metric in ["MAE", "RMSE", "MSE", "MAPE (%)"]
    
    if primary_metric in df_comp.columns:
        df_comp = df_comp.sort_values(
            by=primary_metric,
            ascending=lower_is_better,
            na_position="last",
        ).reset_index(drop=True)
        best_model_key = df_comp.iloc[0]["Model Key"]
    else:
        # Fallback sorting
        fallback_col = "F1-Score (Macro)" if is_classification else "RMSE"
        ascending = False if is_classification else True
        df_comp = df_comp.sort_values(by=fallback_col, ascending=ascending).reset_index(drop=True)
        best_model_key = df_comp.iloc[0]["Model Key"]

    return ModelComparisonResult(
        comparison_table=df_comp,
        results_by_model=results_by_model,
        best_model_key=best_model_key,
        primary_metric=primary_metric,
        problem_type=problem_type,
        models_evaluated=selected_model_keys,
    )

