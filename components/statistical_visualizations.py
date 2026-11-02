"""Plotly visualizations for statistical hypothesis testing, distributions, and assumption checks."""

from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

DARK_THEME_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Plus Jakarta Sans, sans-serif", color="#F8FAFC"),
    margin=dict(l=35, r=35, t=50, b=35),
)


def plot_qq_plot(
    theoretical_quantiles: np.ndarray,
    sample_quantiles: np.ndarray,
    slope: float,
    intercept: float,
    title: str = "Normal Q-Q Plot (Quantile-Quantile Diagnostic)",
) -> go.Figure:
    """Render interactive Q-Q plot comparing empirical quantiles against theoretical normal distribution."""
    fig = go.Figure()

    if len(theoretical_quantiles) > 0 and len(sample_quantiles) > 0:
        # Empirical points
        fig.add_trace(
            go.Scatter(
                x=theoretical_quantiles,
                y=sample_quantiles,
                mode="markers",
                name="Sample Quantiles",
                marker=dict(color="#38BDF8", size=6, opacity=0.8),
            )
        )

        # Theoretical reference line
        line_x = np.linspace(theoretical_quantiles.min(), theoretical_quantiles.max(), 50)
        line_y = slope * line_x + intercept
        fig.add_trace(
            go.Scatter(
                x=line_x,
                y=line_y,
                mode="lines",
                name="Normal Reference Line",
                line=dict(color="#EF4444", width=2.5, dash="dash"),
            )
        )

    fig.update_layout(
        **DARK_THEME_LAYOUT,
        title=title,
        xaxis_title="Theoretical Normal Quantiles",
        yaxis_title="Ordered Sample Values",
    )
    return fig


def plot_mean_comparison_ci(
    group_names: List[str],
    means: List[float],
    ci_lowers: List[float],
    ci_uppers: List[float],
    variable_name: str = "Mean Value",
    title: str = "Group Means with 95% Confidence Intervals",
) -> go.Figure:
    """Render point estimates with error bars indicating confidence intervals."""
    fig = go.Figure()

    errors_plus = [up - m for up, m in zip(ci_uppers, means)]
    errors_minus = [m - low for low, m in zip(ci_lowers, means)]

    fig.add_trace(
        go.Scatter(
            x=group_names,
            y=means,
            mode="markers+text",
            text=[f"{m:.2f}" for m in means],
            textposition="top center",
            error_y=dict(
                type="data",
                symmetric=False,
                array=errors_plus,
                arrayminus=errors_minus,
                color="#F59E0B",
                thickness=2.5,
                width=8,
            ),
            marker=dict(color="#38BDF8", size=12),
            name="Group Mean",
        )
    )

    fig.update_layout(
        **DARK_THEME_LAYOUT,
        title=title,
        xaxis_title="Cohort / Group",
        yaxis_title=variable_name,
    )
    return fig


def plot_contingency_heatmap(
    observed_df: pd.DataFrame,
    expected_df: pd.DataFrame,
    title: str = "Contingency Table: Observed (Expected) Counts",
) -> go.Figure:
    """Render heatmap displaying observed counts alongside expected frequencies."""
    r_labels = list(observed_df.index)
    c_labels = list(observed_df.columns)

    text_matrix = []
    for r in r_labels:
        row_text = []
        for c in c_labels:
            obs = observed_df.loc[r, c]
            exp = expected_df.loc[r, c] if r in expected_df.index and c in expected_df.columns else 0.0
            row_text.append(f"Obs: {obs:,}<br>Exp: {exp:.1f}")
        text_matrix.append(row_text)

    fig = go.Figure(
        data=go.Heatmap(
            z=observed_df.values,
            x=c_labels,
            y=r_labels,
            colorscale="Viridis",
            text=text_matrix,
            texttemplate="%{text}",
            colorbar=dict(title="Observed Count"),
        )
    )

    fig.update_layout(
        **DARK_THEME_LAYOUT,
        title=title,
        xaxis_title=observed_df.columns.name or "Column Category",
        yaxis_title=observed_df.index.name or "Row Category",
    )
    return fig


def plot_correlation_scatter(
    x: pd.Series,
    y: pd.Series,
    x_name: str,
    y_name: str,
    r_val: float,
    p_val: float,
    title: Optional[str] = None,
) -> go.Figure:
    """Scatter plot with OLS linear trendline and statistical annotation."""
    clean_df = pd.DataFrame({"x": x, "y": y}).dropna()
    clean_df = clean_df.astype(float)

    fig = px.scatter(
        clean_df,
        x="x",
        y="y",
        trendline="ols",
        template="plotly_dark",
        title=title or f"Linear Correlation: '{x_name}' vs '{y_name}' (r = {r_val:.3f}, p = {p_val:.4f})",
        labels={"x": x_name, "y": y_name},
    )
    fig.update_traces(marker=dict(color="#38BDF8", size=6, opacity=0.75))
    fig.update_layout(
        **DARK_THEME_LAYOUT,
        xaxis_title=x_name,
        yaxis_title=y_name,
    )
    return fig


def plot_multiple_testing_comparison(
    labels: List[str],
    raw_p: List[float],
    adj_p: List[float],
    alpha: float = 0.05,
    title: str = "P-Value Adjustment Comparison (Raw vs. Corrected)",
) -> go.Figure:
    """Side-by-side grouped bar chart comparing raw and multiplicity-adjusted p-values."""
    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            name="Raw P-Value",
            x=labels,
            y=raw_p,
            marker_color="#38BDF8",
        )
    )
    fig.add_trace(
        go.Bar(
            name="Adjusted P-Value",
            x=labels,
            y=adj_p,
            marker_color="#F59E0B",
        )
    )

    # Add alpha threshold line
    fig.add_hline(
        y=alpha,
        line_dash="dash",
        line_color="#EF4444",
        annotation_text=f"Significance Threshold (α = {alpha})",
        annotation_position="top right",
    )

    fig.update_layout(
        **DARK_THEME_LAYOUT,
        barmode="group",
        title=title,
        yaxis_title="P-Value",
        xaxis_title="Hypothesis Test",
        yaxis=dict(range=[0, min(1.0, max(max(raw_p, default=0.1), max(adj_p, default=0.1)) * 1.15)]),
    )
    return fig

