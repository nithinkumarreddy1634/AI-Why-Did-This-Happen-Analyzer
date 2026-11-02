"""Plotly visualizations for Machine Learning Engine evaluation and diagnostics."""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from config.constants import THEME_COLORS


DARK_LAYOUT_TEMPLATE = dict(
    plot_bgcolor="rgba(15, 23, 42, 0.6)",
    paper_bgcolor="rgba(15, 23, 42, 0.8)",
    font=dict(color="#F8FAFC", family="Inter, system-ui, sans-serif"),
    xaxis=dict(
        gridcolor="rgba(51, 65, 85, 0.4)",
        zerolinecolor="rgba(148, 163, 184, 0.3)",
    ),
    yaxis=dict(
        gridcolor="rgba(51, 65, 85, 0.4)",
        zerolinecolor="rgba(148, 163, 184, 0.3)",
    ),
    margin=dict(l=40, r=40, t=50, b=40),
)


def plot_confusion_matrix(
    cm: List[List[int]],
    labels: List[str],
    title: str = "Confusion Matrix Heatmap",
) -> go.Figure:
    """Render an interactive Confusion Matrix Heatmap."""
    cm_arr = np.array(cm)
    
    # Text annotations for cells
    total = np.sum(cm_arr)
    annotations = []
    text_matrix = []
    
    for i in range(len(labels)):
        row_text = []
        for j in range(len(labels)):
            val = cm_arr[i, j]
            pct = (val / total) * 100 if total > 0 else 0
            row_text.append(f"{val:,}<br>({pct:.1f}%)")
        text_matrix.append(row_text)

    fig = go.Figure(
        data=go.Heatmap(
            z=cm_arr,
            x=[f"Pred: {lbl}" for lbl in labels],
            y=[f"True: {lbl}" for lbl in labels],
            text=text_matrix,
            texttemplate="%{text}",
            textfont={"size": 13, "color": "#F8FAFC"},
            colorscale=[[0, "#1E293B"], [0.5, "#4F46E5"], [1.0, "#06B6D4"]],
            showscale=True,
            colorbar=dict(title="Count"),
        )
    )

    fig.update_layout(
        title=dict(text=title, font=dict(size=15, color="#F8FAFC")),
        xaxis_title="Predicted Label",
        yaxis_title="Actual True Label",
        yaxis=dict(autorange="reversed"),
        **DARK_LAYOUT_TEMPLATE,
    )
    return fig


def plot_roc_curve(roc_curve_data: Dict[str, Any], title: str = "ROC Curve (Receiver Operating Characteristic)") -> go.Figure:
    """Render ROC Curve with 45-degree no-discrimination baseline."""
    fpr = roc_curve_data.get("fpr", [])
    tpr = roc_curve_data.get("tpr", [])
    auc = roc_curve_data.get("auc", 0.0)

    fig = go.Figure()

    # ROC Line
    fig.add_trace(
        go.Scatter(
            x=fpr,
            y=tpr,
            mode="lines",
            name=f"Model ROC (AUC = {auc:.3f})",
            line=dict(color="#06B6D4", width=3),
        )
    )

    # Baseline random chance
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Random Guess (AUC = 0.500)",
            line=dict(color="#EF4444", width=2, dash="dash"),
        )
    )

    fig.update_layout(
        title=dict(text=title, font=dict(size=15, color="#F8FAFC")),
        xaxis=dict(title="False Positive Rate (1 - Specificity)", range=[0, 1]),
        yaxis=dict(title="True Positive Rate (Recall / Sensitivity)", range=[0, 1.05]),
        legend=dict(x=0.55, y=0.15, bgcolor="rgba(30, 41, 59, 0.8)"),
        **DARK_LAYOUT_TEMPLATE,
    )
    return fig


def plot_pr_curve(pr_curve_data: Dict[str, Any], title: str = "Precision-Recall Curve") -> go.Figure:
    """Render Precision-Recall curve with baseline class prevalence line."""
    precision = pr_curve_data.get("precision", [])
    recall = pr_curve_data.get("recall", [])
    auc = pr_curve_data.get("auc", 0.0)
    baseline = pr_curve_data.get("baseline", 0.5)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=recall,
            y=precision,
            mode="lines",
            name=f"Model PR (AP = {auc:.3f})",
            line=dict(color="#10B981", width=3),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[baseline, baseline],
            mode="lines",
            name=f"Class Prevalence ({baseline:.1%})",
            line=dict(color="#F59E0B", width=2, dash="dash"),
        )
    )

    fig.update_layout(
        title=dict(text=title, font=dict(size=15, color="#F8FAFC")),
        xaxis=dict(title="Recall", range=[0, 1]),
        yaxis=dict(title="Precision", range=[0, 1.05]),
        legend=dict(x=0.55, y=0.85, bgcolor="rgba(30, 41, 59, 0.8)"),
        **DARK_LAYOUT_TEMPLATE,
    )
    return fig


def plot_actual_vs_predicted(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Actual vs. Predicted Target Values",
) -> go.Figure:
    """Render regression Actual vs Predicted scatter plot with 45-degree identity line."""
    fig = go.Figure()

    # Scatter points
    fig.add_trace(
        go.Scatter(
            x=y_true,
            y=y_pred,
            mode="markers",
            marker=dict(color="#6366F1", size=6, opacity=0.7),
            name="Observations",
        )
    )

    # 45-degree line
    min_val = min(float(np.min(y_true)), float(np.min(y_pred)))
    max_val = max(float(np.max(y_true)), float(np.max(y_pred)))

    fig.add_trace(
        go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode="lines",
            name="Ideal Fit (y = ŷ)",
            line=dict(color="#10B981", width=2, dash="dash"),
        )
    )

    fig.update_layout(
        title=dict(text=title, font=dict(size=15, color="#F8FAFC")),
        xaxis_title="Actual Target",
        yaxis_title="Predicted Target",
        **DARK_LAYOUT_TEMPLATE,
    )
    return fig


def plot_residual_diagnostics(
    residuals: List[float],
    y_pred: np.ndarray,
) -> go.Figure:
    """Render residuals vs. predicted plot to inspect heteroscedasticity and variance."""
    res_arr = np.array(residuals)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=y_pred,
            y=res_arr,
            mode="markers",
            marker=dict(
                color=res_arr,
                colorscale="Bluered",
                size=6,
                opacity=0.75,
                showscale=True,
                colorbar=dict(title="Residual"),
            ),
            name="Residuals",
        )
    )

    # Zero error reference line
    fig.add_hline(
        y=0,
        line=dict(color="#F8FAFC", width=1.5, dash="dash"),
        annotation_text="Zero Residual Line",
        annotation_position="bottom right",
    )

    fig.update_layout(
        title=dict(text="Residuals vs. Predicted (Heteroscedasticity Check)", font=dict(size=15, color="#F8FAFC")),
        xaxis_title="Predicted Value (ŷ)",
        yaxis_title="Residual (y - ŷ)",
        **DARK_LAYOUT_TEMPLATE,
    )
    return fig


def plot_feature_importance_bars(
    importance_df: pd.DataFrame,
    max_features: int = 15,
    title: str = "Relative Feature Importance",
) -> go.Figure:
    """Render horizontal bar chart of top predictive features."""
    plot_df = importance_df.head(max_features).sort_values(by="Importance", ascending=True)

    fig = go.Figure(
        go.Bar(
            x=plot_df["Importance"],
            y=plot_df["Feature"],
            orientation="h",
            marker=dict(
                color=plot_df["Importance"],
                colorscale=[[0, "#312E81"], [0.5, "#4F46E5"], [1.0, "#06B6D4"]],
            ),
            text=[f"{v:.3f}" for v in plot_df["Importance"]],
            textposition="auto",
        )
    )

    fig.update_layout(
        title=dict(text=title, font=dict(size=15, color="#F8FAFC")),
        xaxis_title="Importance Score",
        yaxis_title="Feature Name",
        **DARK_LAYOUT_TEMPLATE,
    )
    return fig


def plot_model_comparison_bar(
    comparison_df: pd.DataFrame,
    primary_metric: str,
    title: str = "Algorithm Performance Benchmark",
) -> go.Figure:
    """Render comparison bar chart ranking models by primary metric."""
    df_sorted = comparison_df.sort_values(by=primary_metric, ascending=True)

    fig = go.Figure(
        go.Bar(
            x=df_sorted[primary_metric],
            y=df_sorted["Model"],
            orientation="h",
            marker=dict(
                color=df_sorted[primary_metric],
                colorscale=[[0, "#4F46E5"], [1.0, "#10B981"]],
            ),
            text=[f"{v:.3f}" if isinstance(v, (int, float)) else str(v) for v in df_sorted[primary_metric]],
            textposition="auto",
        )
    )

    fig.update_layout(
        title=dict(text=f"{title} ({primary_metric})", font=dict(size=15, color="#F8FAFC")),
        xaxis_title=primary_metric,
        yaxis_title="Model Algorithm",
        **DARK_LAYOUT_TEMPLATE,
    )
    return fig

