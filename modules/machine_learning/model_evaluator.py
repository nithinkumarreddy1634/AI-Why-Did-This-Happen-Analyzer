"""Model evaluation metrics, diagnostic curves, and report generation."""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Union
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from utils.ml_helpers import safe_mape, safe_divide


@dataclass
class ClassificationMetrics:
    """Evaluation metrics container for classification models."""
    accuracy: float
    precision_macro: float
    precision_weighted: float
    recall_macro: float
    recall_weighted: float
    f1_macro: float
    f1_weighted: float
    roc_auc: Optional[float] = None
    pr_auc: Optional[float] = None
    confusion_matrix: Optional[List[List[int]]] = None
    labels: List[str] = field(default_factory=list)
    classification_report_dict: Dict[str, Any] = field(default_factory=dict)
    roc_curve_data: Optional[Dict[str, Any]] = None
    pr_curve_data: Optional[Dict[str, Any]] = None
    cv_mean: Optional[float] = None
    cv_std: Optional[float] = None
    training_duration_sec: float = 0.0
    metric_explanations: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "Accuracy": self.accuracy,
            "F1-Score (Macro)": self.f1_macro,
            "F1-Score (Weighted)": self.f1_weighted,
            "Precision (Macro)": self.precision_macro,
            "Recall (Macro)": self.recall_macro,
            "ROC-AUC": self.roc_auc,
            "PR-AUC": self.pr_auc,
            "CV Mean": self.cv_mean,
            "CV Std": self.cv_std,
            "Training Time (s)": round(self.training_duration_sec, 3),
        }


@dataclass
class RegressionMetrics:
    """Evaluation metrics container for regression models."""
    mae: float
    mse: float
    rmse: float
    r2: float
    adjusted_r2: float
    mape: Optional[float] = None
    residuals: Optional[List[float]] = None
    cv_mean: Optional[float] = None
    cv_std: Optional[float] = None
    training_duration_sec: float = 0.0
    metric_explanations: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        res = {
            "MAE": self.mae,
            "RMSE": self.rmse,
            "MSE": self.mse,
            "R²": self.r2,
            "Adjusted R²": self.adjusted_r2,
            "CV Mean": self.cv_mean,
            "CV Std": self.cv_std,
            "Training Time (s)": round(self.training_duration_sec, 3),
        }
        if self.mape is not None:
            res["MAPE (%)"] = self.mape
        return res


def evaluate_classification_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    labels: Optional[List[str]] = None,
    training_duration_sec: float = 0.0,
    cv_scores: Optional[np.ndarray] = None,
) -> ClassificationMetrics:
    """Compute comprehensive classification evaluation metrics and curve data."""
    acc = float(accuracy_score(y_true, y_pred))
    p_macro = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    p_weighted = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
    r_macro = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    r_weighted = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))
    f1_m = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    f1_w = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    cm = confusion_matrix(y_true, y_pred)
    cm_list = cm.tolist()

    unique_classes = np.unique(np.concatenate([y_true, y_pred]))
    if labels is None:
        class_labels = [str(c) for c in unique_classes]
    else:
        class_labels = [str(labels[int(c)]) if int(c) < len(labels) else str(c) for c in unique_classes]

    report = classification_report(
        y_true, y_pred, target_names=class_labels, output_dict=True, zero_division=0
    )

    roc_auc = None
    pr_auc = None
    roc_curve_data = None
    pr_curve_data = None

    if y_proba is not None:
        try:
            n_classes = len(unique_classes)
            if n_classes == 2:
                # Binary classification
                proba_pos = y_proba[:, 1] if y_proba.ndim > 1 else y_proba
                roc_auc = float(roc_auc_score(y_true, proba_pos))
                pr_auc = float(average_precision_score(y_true, proba_pos))

                fpr, tpr, _ = roc_curve(y_true, proba_pos)
                prec_c, rec_c, _ = precision_recall_curve(y_true, proba_pos)

                roc_curve_data = {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "auc": roc_auc}
                pr_curve_data = {
                    "precision": prec_c.tolist(),
                    "recall": rec_c.tolist(),
                    "auc": pr_auc,
                    "baseline": float(np.mean(y_true)),
                }
            elif n_classes > 2 and y_proba.shape[1] >= n_classes:
                # Multiclass One-vs-Rest ROC-AUC
                roc_auc = float(roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro"))
        except Exception:
            roc_auc = None
            pr_auc = None

    cv_mean = float(np.mean(cv_scores)) if cv_scores is not None and len(cv_scores) > 0 else None
    cv_std = float(np.std(cv_scores)) if cv_scores is not None and len(cv_scores) > 0 else None

    explanations = {
        "Accuracy": "Fraction of overall correct predictions. Caution: Can be misleadingly optimistic when classes are highly imbalanced.",
        "Precision": "Proportion of positive identifications that were actually correct (minimizes false alarms).",
        "Recall": "Proportion of actual positives identified correctly (minimizes missed cases).",
        "F1-Score": "Harmonic mean of precision and recall; balanced measure recommended for imbalanced datasets.",
        "ROC-AUC": "Degree of separability across all possible decision thresholds (0.5 = random guess, 1.0 = perfect separation).",
        "PR-AUC": "Area under the precision-recall curve; especially reliable for detecting rare events / minority classes.",
    }

    return ClassificationMetrics(
        accuracy=acc,
        precision_macro=p_macro,
        precision_weighted=p_weighted,
        recall_macro=r_macro,
        recall_weighted=r_weighted,
        f1_macro=f1_m,
        f1_weighted=f1_w,
        roc_auc=roc_auc,
        pr_auc=pr_auc,
        confusion_matrix=cm_list,
        labels=class_labels,
        classification_report_dict=report,
        roc_curve_data=roc_curve_data,
        pr_curve_data=pr_curve_data,
        cv_mean=cv_mean,
        cv_std=cv_std,
        training_duration_sec=training_duration_sec,
        metric_explanations=explanations,
    )


def evaluate_regression_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_features: int,
    training_duration_sec: float = 0.0,
    cv_scores: Optional[np.ndarray] = None,
) -> RegressionMetrics:
    """Compute comprehensive regression evaluation metrics and residual diagnostics."""
    mae = float(mean_absolute_error(y_true, y_pred))
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_true, y_pred))

    n = len(y_true)
    p = n_features
    # Adjusted R-squared
    if n > p + 1:
        adj_r2 = float(1.0 - (1.0 - r2) * ((n - 1) / (n - p - 1)))
    else:
        adj_r2 = r2

    mape = safe_mape(y_true, y_pred)
    residuals = (y_true - y_pred).tolist()

    cv_mean = float(np.mean(cv_scores)) if cv_scores is not None and len(cv_scores) > 0 else None
    cv_std = float(np.std(cv_scores)) if cv_scores is not None and len(cv_scores) > 0 else None

    explanations = {
        "MAE": "Mean Absolute Error: Average magnitude of the errors without direction; expressed in target variable units.",
        "RMSE": "Root Mean Squared Error: Penalizes larger deviations more heavily than MAE; expressed in target variable units.",
        "MSE": "Mean Squared Error: Average squared differences; heavily sensitive to extreme outlier predictions.",
        "R²": "Coefficient of determination: Proportion of variance in the target explained by the model features.",
        "Adjusted R²": "R² penalized for the number of predictors; prevents artificial inflation from adding irrelevant variables.",
        "MAPE": "Mean Absolute Percentage Error: Relative percentage difference. Reported only when actual targets are non-zero.",
    }

    return RegressionMetrics(
        mae=mae,
        mse=mse,
        rmse=rmse,
        r2=r2,
        adjusted_r2=adj_r2,
        mape=mape,
        residuals=residuals,
        cv_mean=cv_mean,
        cv_std=cv_std,
        training_duration_sec=training_duration_sec,
        metric_explanations=explanations,
    )

