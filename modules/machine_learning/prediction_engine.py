"""Prediction exploration and record-level inference inspector."""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from modules.machine_learning.problem_detector import ProblemType


@dataclass
class PredictionExplorationResult:
    """Container holding record-level test predictions and summary diagnostics."""
    predictions_df: pd.DataFrame
    problem_type: ProblemType
    model_name: str
    n_samples: int
    n_correct: Optional[int] = None
    n_errors: Optional[int] = None
    mean_error: Optional[float] = None
    summary_stats: Dict[str, Any] = field(default_factory=dict)

    def to_csv(self) -> str:
        """Export predictions table to CSV string."""
        return self.predictions_df.to_csv(index=False)


def generate_prediction_exploration(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    target_name: str,
    problem_type: ProblemType,
    labels: Optional[List[str]] = None,
    model_name: str = "Trained Model",
) -> PredictionExplorationResult:
    """Generate row-by-row prediction comparisons with probabilities and residual error."""
    is_classification = problem_type in (
        ProblemType.BINARY_CLASSIFICATION,
        ProblemType.MULTICLASS_CLASSIFICATION,
    )

    y_pred = pipeline.predict(X_test)
    n_samples = len(X_test)

    # Base DataFrame copying feature values
    df_out = X_test.copy().reset_index(drop=True)

    if is_classification:
        # Check probabilities
        y_proba = None
        if hasattr(pipeline, "predict_proba"):
            try:
                y_proba = pipeline.predict_proba(X_test)
            except Exception:
                y_proba = None

        if labels is not None and len(labels) > 0:
            actual_labels = [labels[int(y)] if int(y) < len(labels) else str(y) for y in y_test]
            pred_labels = [labels[int(y)] if int(y) < len(labels) else str(y) for y in y_pred]
        else:
            actual_labels = [str(y) for y in y_test]
            pred_labels = [str(y) for y in y_pred]

        df_out[f"Actual_{target_name}"] = actual_labels
        df_out[f"Predicted_{target_name}"] = pred_labels
        df_out["Is_Correct"] = (y_test == y_pred)
        df_out["Outcome_Status"] = np.where(df_out["Is_Correct"], "Correct Prediction", "Misclassification")

        if y_proba is not None:
            # Confidence is max predicted probability
            max_prob = np.max(y_proba, axis=1)
            df_out["Confidence"] = np.round(max_prob, 4)

            # Class-specific probability columns
            if labels is not None and len(labels) == y_proba.shape[1]:
                for idx, lbl in enumerate(labels):
                    df_out[f"Prob_{lbl}"] = np.round(y_proba[:, idx], 4)
            else:
                for idx in range(y_proba.shape[1]):
                    df_out[f"Prob_Class_{idx}"] = np.round(y_proba[:, idx], 4)

        n_correct = int(np.sum(y_test == y_pred))
        n_errors = n_samples - n_correct
        summary_stats = {
            "Total Holdout Samples": n_samples,
            "Correct Predictions": n_correct,
            "Misclassifications": n_errors,
            "Empirical Accuracy": round(n_correct / n_samples, 4) if n_samples > 0 else 0.0,
        }

        return PredictionExplorationResult(
            predictions_df=df_out,
            problem_type=problem_type,
            model_name=model_name,
            n_samples=n_samples,
            n_correct=n_correct,
            n_errors=n_errors,
            summary_stats=summary_stats,
        )

    else:
        # Regression
        actuals = np.asarray(y_test, dtype=float)
        preds = np.asarray(y_pred, dtype=float)
        residuals = actuals - preds
        abs_errors = np.abs(residuals)
        
        df_out[f"Actual_{target_name}"] = np.round(actuals, 4)
        df_out[f"Predicted_{target_name}"] = np.round(preds, 4)
        df_out["Residual_Error"] = np.round(residuals, 4)
        df_out["Abs_Error"] = np.round(abs_errors, 4)

        # Percentage error where actual is non-zero
        safe_actual = np.where(np.abs(actuals) < 1e-6, np.nan, actuals)
        pct_err = np.abs(residuals / safe_actual) * 100.0
        df_out["Pct_Error"] = np.round(pct_err, 2)

        mean_mae = float(np.mean(abs_errors))
        summary_stats = {
            "Total Holdout Samples": n_samples,
            "Mean Absolute Error (MAE)": round(mean_mae, 4),
            "Max Absolute Error": round(float(np.max(abs_errors)), 4),
            "Min Absolute Error": round(float(np.min(abs_errors)), 4),
        }

        return PredictionExplorationResult(
            predictions_df=df_out,
            problem_type=problem_type,
            model_name=model_name,
            n_samples=n_samples,
            mean_error=mean_mae,
            summary_stats=summary_stats,
        )

