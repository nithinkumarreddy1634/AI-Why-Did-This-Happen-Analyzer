"""Plotly visualization builders for the Exploratory Data Analysis (EDA) engine.

Provides dark-themed, enterprise-grade charts for univariate distributions,
bivariate interactions, correlation heatmaps, time-series trends, and segment breakdowns.
"""

from typing import Optional, List, Dict, Any
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px


DARK_THEME_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Plus Jakarta Sans, sans-serif", color="#F8FAFC"),
    margin=dict(l=25, r=25, t=45, b=25),
)


def plot_numerical_distribution(
    series: pd.Series,
    title: Optional[str] = None,
    plot_type: str = "histogram",
    color: str = "#4F46E5",
    nbins: int = 30,
) -> go.Figure:
    """Create interactive histogram, box plot, or violin plot for a numerical column."""
    clean_series = series.dropna().astype(float)
    col_name = str(series.name) if series.name else "Value"
    plot_title = title or f"Distribution of '{col_name}'"

    fig = go.Figure()

    if plot_type == "box":
        fig.add_trace(
            go.Box(
                y=clean_series,
                name=col_name,
                marker_color=color,
                boxpoints="outliers",
                jitter=0.3,
                pointpos=-1.8,
            )
        )
        fig.update_layout(
            **DARK_THEME_LAYOUT,
            title=plot_title,
            yaxis_title=col_name,
        )
    elif plot_type == "violin":
        fig.add_trace(
            go.Violin(
                y=clean_series,
                name=col_name,
                box_visible=True,
                meanline_visible=True,
                fillcolor=color,
                line_color="#818CF8",
                opacity=0.7,
            )
        )
        fig.update_layout(
            **DARK_THEME_LAYOUT,
            title=plot_title,
            yaxis_title=col_name,
        )
    else:
        # Histogram with KDE approximation
        fig.add_trace(
            go.Histogram(
                x=clean_series,
                nbinsx=nbins,
                name="Frequency",
                marker_color=color,
                opacity=0.75,
            )
        )
        # Add mean and median vertical indicator lines
        mean_v = float(clean_series.mean())
        median_v = float(clean_series.median())
        fig.add_vline(
            x=mean_v,
            line_dash="dash",
            line_color="#38BDF8",
            annotation_text=f"Mean: {mean_v:.2f}",
            annotation_position="top right",
        )
        fig.add_vline(
            x=median_v,
            line_dash="dot",
            line_color="#34D399",
            annotation_text=f"Median: {median_v:.2f}",
            annotation_position="top left",
        )
        fig.update_layout(
            **DARK_THEME_LAYOUT,
            title=plot_title,
            xaxis_title=col_name,
            yaxis_title="Record Count",
            bargap=0.08,
        )

    return fig


def plot_categorical_distribution(
    freq_df: pd.DataFrame,
    title: Optional[str] = None,
    chart_type: str = "bar",
    color: str = "#06B6D4",
) -> go.Figure:
    """Render bar, horizontal bar, or pie chart for category frequency distributions."""
    fig = go.Figure()
    plot_title = title or "Categorical Distribution"

    if chart_type == "pie" and len(freq_df) <= 8:
        fig.add_trace(
            go.Pie(
                labels=freq_df["Category"],
                values=freq_df["Count"],
                hole=0.4,
                textinfo="label+percent",
                marker=dict(colors=["#4F46E5", "#06B6D4", "#10B981", "#F59E0B", "#EC4899", "#8B5CF6", "#64748B", "#F97316"]),
            )
        )
        fig.update_layout(
            **DARK_THEME_LAYOUT,
            title=plot_title,
            legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
        )
    elif chart_type == "horizontal_bar":
        # Sort ascending for horizontal bar so top category is at the top
        df_sorted = freq_df.sort_values(by="Count", ascending=True)
        fig.add_trace(
            go.Bar(
                x=df_sorted["Count"],
                y=df_sorted["Category"],
                orientation="h",
                marker_color=color,
                text=[f"{p}%" for p in df_sorted["Percentage"]],
                textposition="auto",
            )
        )
        fig.update_layout(
            **DARK_THEME_LAYOUT,
            title=plot_title,
            xaxis_title="Record Count",
            yaxis_title="Category",
        )
    else:
        fig.add_trace(
            go.Bar(
                x=freq_df["Category"],
                y=freq_df["Count"],
                marker_color=color,
                text=[f"{p}%" for p in freq_df["Percentage"]],
                textposition="outside",
            )
        )
        fig.update_layout(
            **DARK_THEME_LAYOUT,
            title=plot_title,
            xaxis_title="Category",
            yaxis_title="Record Count",
        )

    return fig


def plot_scatter_with_trend(
    df: pd.DataFrame,
    col_x: str,
    col_y: str,
    color_col: Optional[str] = None,
    show_trendline: bool = True,
) -> go.Figure:
    """Render interactive scatter plot with optional OLS trendline and categorical coloring."""
    sub_df = df[[col_x, col_y] + ([color_col] if color_col else [])].dropna()
    sub_df[col_x] = sub_df[col_x].astype(float)
    sub_df[col_y] = sub_df[col_y].astype(float)

    if color_col:
        fig = px.scatter(
            sub_df,
            x=col_x,
            y=col_y,
            color=color_col,
            title=f"Scatter: '{col_x}' vs. '{col_y}' (Colored by '{color_col}')",
            template="plotly_dark",
        )
    else:
        fig = px.scatter(
            sub_df,
            x=col_x,
            y=col_y,
            title=f"Scatter: '{col_x}' vs. '{col_y}'",
            template="plotly_dark",
            color_discrete_sequence=["#38BDF8"],
        )

    # Compute and overlay manual OLS trendline if requested and sample permits
    if show_trendline and len(sub_df) >= 3:
        x_vals = sub_df[col_x].values
        y_vals = sub_df[col_y].values
        if np.var(x_vals) > 0:
            m, b = np.polyfit(x_vals, y_vals, 1)
            x_line = np.linspace(x_vals.min(), x_vals.max(), 50)
            y_line = m * x_line + b
            fig.add_trace(
                go.Scatter(
                    x=x_line,
                    y=y_line,
                    mode="lines",
                    name="Linear Trend (OLS)",
                    line=dict(color="#F43F5E", width=2.5, dash="dash"),
                )
            )

    fig.update_layout(
        **DARK_THEME_LAYOUT,
        xaxis_title=col_x,
        yaxis_title=col_y,
    )
    return fig


def plot_categorical_vs_numerical(
    df: pd.DataFrame,
    cat_col: str,
    num_col: str,
    plot_type: str = "box",
) -> go.Figure:
    """Render grouped box plot, violin plot, or mean bar chart comparing a metric across classes."""
    valid_df = df[[cat_col, num_col]].dropna().copy()
    valid_df[num_col] = valid_df[num_col].astype(float)
    valid_df[cat_col] = valid_df[cat_col].astype(str)

    if plot_type == "violin":
        fig = px.violin(
            valid_df,
            x=cat_col,
            y=num_col,
            box=True,
            points="outliers",
            color=cat_col,
            title=f"Distribution of '{num_col}' by '{cat_col}'",
            template="plotly_dark",
        )
    elif plot_type == "bar_mean":
        mean_df = valid_df.groupby(cat_col)[num_col].mean().reset_index()
        fig = px.bar(
            mean_df,
            x=cat_col,
            y=num_col,
            color=cat_col,
            title=f"Average '{num_col}' by '{cat_col}'",
            template="plotly_dark",
        )
    else:
        # Default: Box plot
        fig = px.box(
            valid_df,
            x=cat_col,
            y=num_col,
            color=cat_col,
            title=f"Box Plot: '{num_col}' across '{cat_col}' Groups",
            template="plotly_dark",
        )

    fig.update_layout(
        **DARK_THEME_LAYOUT,
        xaxis_title=cat_col,
        yaxis_title=num_col,
        showlegend=False,
    )
    return fig


def plot_cross_tabulation(
    crosstab_df: pd.DataFrame,
    cat_col1: str,
    cat_col2: str,
    plot_type: str = "stacked_bar",
) -> go.Figure:
    """Render stacked bar chart or heatmap comparing two categorical distributions."""
    if plot_type == "heatmap":
        fig = px.imshow(
            crosstab_df,
            text_auto=True,
            aspect="auto",
            color_continuous_scale="Viridis",
            title=f"Cross-Tabulation Matrix: '{cat_col1}' × '{cat_col2}'",
            template="plotly_dark",
        )
    else:
        # Stacked bar
        fig = go.Figure()
        colors = ["#4F46E5", "#06B6D4", "#10B981", "#F59E0B", "#EC4899", "#8B5CF6", "#64748B", "#F97316"]
        for idx, col in enumerate(crosstab_df.columns):
            fig.add_trace(
                go.Bar(
                    x=crosstab_df.index,
                    y=crosstab_df[col],
                    name=str(col),
                    marker_color=colors[idx % len(colors)],
                )
            )
        fig.update_layout(
            **DARK_THEME_LAYOUT,
            barmode="stack",
            title=f"Stacked Distribution: '{cat_col1}' by '{cat_col2}'",
            xaxis_title=cat_col1,
            yaxis_title="Record Count",
            legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
        )

    return fig


def plot_correlation_heatmap(
    corr_matrix: pd.DataFrame,
    title: str = "Correlation Matrix (Pearson r)",
) -> Optional[go.Figure]:
    """Render annotated correlation heatmap with symmetric diverging colorscale [-1, 1]."""
    if corr_matrix.empty or corr_matrix.shape[0] < 2:
        return None

    z_vals = corr_matrix.values
    x_cols = list(corr_matrix.columns)
    y_cols = list(corr_matrix.index)

    # Format annotations
    text_vals = np.empty_like(z_vals, dtype=object)
    for i in range(len(y_cols)):
        for j in range(len(x_cols)):
            v = z_vals[i, j]
            text_vals[i, j] = f"{v:.2f}" if not np.isnan(v) else ""

    fig = go.Figure(
        data=go.Heatmap(
            z=z_vals,
            x=x_cols,
            y=y_cols,
            zmin=-1.0,
            zmax=1.0,
            colorscale="RdBu_r",
            text=text_vals,
            texttemplate="%{text}",
            colorbar=dict(title="Correlation"),
        )
    )

    fig.update_layout(
        **DARK_THEME_LAYOUT,
        title=title,
        height=max(400, min(800, len(x_cols) * 45)),
    )
    return fig


def plot_time_series(
    ts_df: pd.DataFrame,
    date_col: str,
    metric_col: str,
    title: Optional[str] = None,
    rolling_col: Optional[str] = None,
) -> go.Figure:
    """Render temporal trend line chart with optional moving average overlay."""
    fig = go.Figure()
    plot_title = title or f"Time-Series Trend of '{metric_col}'"

    # Primary metric line
    fig.add_trace(
        go.Scatter(
            x=ts_df[date_col],
            y=ts_df[metric_col],
            mode="lines+markers",
            name=metric_col,
            line=dict(color="#38BDF8", width=2.5),
            marker=dict(size=5),
        )
    )

    # Rolling average line if present
    if rolling_col and rolling_col in ts_df.columns:
        fig.add_trace(
            go.Scatter(
                x=ts_df[date_col],
                y=ts_df[rolling_col],
                mode="lines",
                name=rolling_col,
                line=dict(color="#F59E0B", width=2.5, dash="dash"),
            )
        )

    fig.update_layout(
        **DARK_THEME_LAYOUT,
        title=plot_title,
        xaxis_title=date_col,
        yaxis_title=metric_col,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def plot_scatter_matrix(
    sample_df: pd.DataFrame,
    columns: List[str],
    color_col: Optional[str] = None,
) -> go.Figure:
    """Render pairwise scatter plot matrix (SPLOM) bounded to max 5 features for performance."""
    if color_col and color_col in sample_df.columns:
        fig = px.scatter_matrix(
            sample_df,
            dimensions=columns,
            color=color_col,
            title="Pairwise Scatter Plot Matrix",
            template="plotly_dark",
        )
    else:
        fig = px.scatter_matrix(
            sample_df,
            dimensions=columns,
            title="Pairwise Scatter Plot Matrix",
            template="plotly_dark",
            color_discrete_sequence=["#818CF8"],
        )

    fig.update_traces(diagonal_visible=False, showupperhalf=False, marker=dict(size=4, opacity=0.7))
    fig.update_layout(
        **DARK_THEME_LAYOUT,
        height=650,
    )
    return fig

