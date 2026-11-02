"""Interactive Plotly visualizations for LIME local explanations and SHAP comparisons."""

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


def plot_lime_contribution_bar(
    contributions_df: pd.DataFrame,
    title: str = "LIME Local Feature Contributions",
) -> go.Figure:
    """Render horizontal bar chart of LIME weights colored by direction (+ green/red, - cyan/red)."""
    # Sort ascending for horizontal bar
    df_sorted = contributions_df.sort_values(by="LIME_Weight", ascending=True)

    colors = [
        "#10B981" if w > 0 else "#EF4444"
        for w in df_sorted["LIME_Weight"]
    ]

    fig = go.Figure(
        go.Bar(
            x=df_sorted["LIME_Weight"],
            y=df_sorted["Condition"],
            orientation="h",
            marker=dict(color=colors),
            text=[f"{w:+.4f}" for w in df_sorted["LIME_Weight"]],
            textposition="auto",
        )
    )

    # Vertical zero line
    fig.add_vline(x=0, line=dict(color="#94A3B8", width=1.5, dash="dash"))

    fig.update_layout(
        title=dict(text=title, font=dict(size=15, color="#F8FAFC")),
        xaxis_title="LIME Local Surrogate Weight (Impact on Target)",
        yaxis_title="Feature Condition / Threshold",
        **DARK_LAYOUT_TEMPLATE,
    )
    return fig


def plot_surrogate_fidelity_comparison(
    original_pred: float,
    surrogate_pred: float,
    fidelity_r2: float,
    title: str = "Local Surrogate Fidelity Comparison",
) -> go.Figure:
    """Render side-by-side bar comparing black-box model output with LIME surrogate prediction."""
    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=["Original Model Output", "LIME Local Surrogate"],
            y=[original_pred, surrogate_pred],
            marker=dict(color=["#4F46E5", "#06B6D4"]),
            text=[f"{original_pred:.4f}", f"{surrogate_pred:.4f}"],
            textposition="auto",
            width=0.45,
        )
    )

    gap = abs(original_pred - surrogate_pred)

    fig.update_layout(
        title=dict(text=f"{title} (R² = {fidelity_r2:.3f}, Gap = {gap:.4f})", font=dict(size=15, color="#F8FAFC")),
        yaxis_title="Prediction Score / Value",
        **DARK_LAYOUT_TEMPLATE,
    )
    return fig


def plot_shap_lime_comparison_bar(
    comparison_df: pd.DataFrame,
    title: str = "SHAP vs. LIME Feature Ranking Comparison",
) -> go.Figure:
    """Render grouped bar chart comparing SHAP and LIME relative rankings or magnitudes."""
    plot_df = comparison_df.head(10).sort_values(by="Feature", ascending=True)

    fig = go.Figure()

    # SHAP Bars
    shap_vals = [s if not np.isnan(s) else 0.0 for s in plot_df["SHAP_Value"]]
    fig.add_trace(
        go.Bar(
            y=plot_df["Feature"],
            x=shap_vals,
            name="SHAP Attribution",
            orientation="h",
            marker=dict(color="#6366F1"),
            text=[f"{v:+.3f}" if v != 0 else "N/A" for v in shap_vals],
            textposition="auto",
        )
    )

    # LIME Bars
    lime_vals = [l if not np.isnan(l) else 0.0 for l in plot_df["LIME_Weight"]]
    fig.add_trace(
        go.Bar(
            y=plot_df["Feature"],
            x=lime_vals,
            name="LIME Local Weight",
            orientation="h",
            marker=dict(color="#10B981"),
            text=[f"{v:+.3f}" if v != 0 else "N/A" for v in lime_vals],
            textposition="auto",
        )
    )

    fig.update_layout(
        title=dict(text=title, font=dict(size=15, color="#F8FAFC")),
        barmode="group",
        xaxis_title="Attribution Metric (SHAP Value vs LIME Weight)",
        yaxis_title="Feature",
        legend=dict(orientation="h", y=-0.15, bgcolor="rgba(30, 41, 59, 0.8)"),
        **DARK_LAYOUT_TEMPLATE,
    )
    return fig


def plot_batch_recurring_features_bar(
    recurring_features: List[Dict[str, Any]],
    title: str = "Cohort Recurring Predictive Drivers",
) -> go.Figure:
    """Render horizontal bar chart of recurring top drivers across multiple explained records."""
    df_rec = pd.DataFrame(recurring_features).head(10).sort_values(by="Frequency_Pct", ascending=True)

    fig = go.Figure(
        go.Bar(
            x=df_rec["Frequency_Pct"],
            y=df_rec["Feature"],
            orientation="h",
            marker=dict(
                color=df_rec["Frequency_Pct"],
                colorscale=[[0, "#4F46E5"], [1.0, "#A855F7"]],
            ),
            text=[f"{pct:.1f}% ({cnt} obs)" for pct, cnt in zip(df_rec["Frequency_Pct"], df_rec["Appearance_Count"])],
            textposition="auto",
        )
    )

    fig.update_layout(
        title=dict(text=title, font=dict(size=15, color="#F8FAFC")),
        xaxis_title="Frequency in Top 3 Drivers (%)",
        yaxis_title="Feature",
        **DARK_LAYOUT_TEMPLATE,
    )
    return fig

