"""Interactive Plotly visualizations for SHAP global and local model explainability."""

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


def plot_shap_summary_bar(
    importance_df: pd.DataFrame,
    max_features: int = 15,
    title: str = "Global SHAP Feature Importance (Mean |SHAP|)",
) -> go.Figure:
    """Render horizontal bar chart of mean absolute SHAP values."""
    plot_df = importance_df.head(max_features).sort_values(by="Mean_Abs_SHAP", ascending=True)

    fig = go.Figure(
        go.Bar(
            x=plot_df["Mean_Abs_SHAP"],
            y=plot_df["Feature"],
            orientation="h",
            marker=dict(
                color=plot_df["Mean_Abs_SHAP"],
                colorscale=[[0, "#312E81"], [0.5, "#4F46E5"], [1.0, "#06B6D4"]],
            ),
            text=[f"{v:.4f}" for v in plot_df["Mean_Abs_SHAP"]],
            textposition="auto",
        )
    )

    fig.update_layout(
        title=dict(text=title, font=dict(size=15, color="#F8FAFC")),
        xaxis_title="Mean |SHAP Value| (Average Impact Magnitude)",
        yaxis_title="Feature",
        **DARK_LAYOUT_TEMPLATE,
    )
    return fig


def plot_shap_beeswarm(
    shap_values: np.ndarray,
    X_df: pd.DataFrame,
    max_features: int = 12,
    title: str = "SHAP Summary Beeswarm Distribution",
) -> go.Figure:
    """Render an interactive Plotly-based beeswarm/strip scatter plot colored by normalized feature values."""
    n_samples, n_features = shap_values.shape
    mean_abs = np.mean(np.abs(shap_values), axis=0)
    top_indices = np.argsort(mean_abs)[::-1][:max_features]

    fig = go.Figure()

    # Iterate over features in ascending importance so top feature is at the top
    for row_idx, f_idx in enumerate(reversed(top_indices)):
        fname = X_df.columns[f_idx]
        shaps = shap_values[:, f_idx]
        vals = X_df[fname].values

        # Normalize feature values to [0, 1] for unified color scale (Low = Cyan, High = Crimson)
        try:
            v_min = np.nanmin(vals)
            v_max = np.nanmax(vals)
            norm_vals = (vals - v_min) / (v_max - v_min) if v_max > v_min else np.full_like(vals, 0.5)
        except Exception:
            norm_vals = np.full_like(shaps, 0.5)

        # Add jitter on y-axis for beeswarm effect
        rng = np.random.RandomState(42 + f_idx)
        jitter = rng.normal(0, 0.08, size=len(shaps))
        y_pos = row_idx + jitter

        fig.add_trace(
            go.Scatter(
                x=shaps,
                y=y_pos,
                mode="markers",
                marker=dict(
                    color=norm_vals,
                    colorscale=[[0, "#06B6D4"], [0.5, "#A855F7"], [1.0, "#EF4444"]],
                    size=6,
                    opacity=0.75,
                    showscale=(row_idx == 0),
                    colorbar=dict(
                        title="Feature Value",
                        tickvals=[0, 1],
                        ticktext=["Low", "High"],
                        len=0.75,
                    ) if row_idx == 0 else None,
                ),
                text=[f"Val: {v:.2f}<br>SHAP: {s:.3f}" if isinstance(v, (int, float)) else f"Val: {v}<br>SHAP: {s:.3f}" for v, s in zip(vals, shaps)],
                hoverinfo="text",
                name=fname,
                showlegend=False,
            )
        )

    ordered_names = [X_df.columns[i] for i in reversed(top_indices)]

    # Vertical zero line
    fig.add_vline(x=0, line=dict(color="#94A3B8", width=1.5, dash="dash"))

    fig.update_layout(
        title=dict(text=title, font=dict(size=15, color="#F8FAFC")),
        xaxis_title="SHAP Value (Impact on Model Prediction)",
        yaxis=dict(
            tickmode="array",
            tickvals=list(range(len(ordered_names))),
            ticktext=ordered_names,
            gridcolor="rgba(51, 65, 85, 0.4)",
        ),
        **DARK_LAYOUT_TEMPLATE,
    )
    return fig


def plot_shap_waterfall(
    base_value: float,
    model_output: float,
    contributions_df: pd.DataFrame,
    max_display: int = 10,
    title: str = "SHAP Waterfall Attribution for Selected Record",
) -> go.Figure:
    """Render an interactive Plotly Waterfall Chart decomposing baseline to final prediction."""
    sorted_df = contributions_df.sort_values(by="Abs_SHAP", ascending=False).reset_index(drop=True)
    top_df = sorted_df.head(max_display)
    remainder_df = sorted_df.iloc[max_display:]

    x_labels = ["Base Value E[f(X)]"]
    y_values = [base_value]
    measure = ["absolute"]
    text_labels = [f"{base_value:.3f}"]

    for _, r in top_df.iterrows():
        val_str = f"{r['Feature_Value']:.2f}" if isinstance(r['Feature_Value'], (int, float)) else str(r['Feature_Value'])
        x_labels.append(f"{r['Feature']} = {val_str}")
        y_values.append(r["SHAP_Value"])
        measure.append("relative")
        sign = "+" if r["SHAP_Value"] > 0 else ""
        text_labels.append(f"{sign}{r['SHAP_Value']:.3f}")

    if not remainder_df.empty:
        rem_sum = remainder_df["SHAP_Value"].sum()
        x_labels.append(f"Other {len(remainder_df)} Features")
        y_values.append(rem_sum)
        measure.append("relative")
        sign = "+" if rem_sum > 0 else ""
        text_labels.append(f"{sign}{rem_sum:.3f}")

    x_labels.append("Final Model Output f(x)")
    y_values.append(model_output)
    measure.append("total")
    text_labels.append(f"{model_output:.3f}")

    fig = go.Figure(
        go.Waterfall(
            name="SHAP Shift",
            orientation="v",
            measure=measure,
            x=x_labels,
            textposition="outside",
            text=text_labels,
            y=y_values,
            connector=dict(line=dict(color="rgba(148, 163, 184, 0.4)")),
            decreasing=dict(marker=dict(color="#06B6D4")),  # Cyan for negative
            increasing=dict(marker=dict(color="#EF4444")),  # Red for positive
            totals=dict(marker=dict(color="#4F46E5")),      # Indigo for totals
        )
    )

    fig.update_layout(
        title=dict(text=title, font=dict(size=15, color="#F8FAFC")),
        xaxis=dict(tickangle=-35),
        yaxis_title="Model Output Space",
        waterfallgap=0.3,
        **DARK_LAYOUT_TEMPLATE,
    )
    return fig


def plot_shap_force_bar(
    base_value: float,
    model_output: float,
    contributions_df: pd.DataFrame,
    title: str = "SHAP Force Contribution Bar",
) -> go.Figure:
    """Render a horizontal stacked force bar showing pushing and pulling attributions."""
    pos_df = contributions_df[contributions_df["SHAP_Value"] > 0].sort_values(by="SHAP_Value", ascending=False)
    neg_df = contributions_df[contributions_df["SHAP_Value"] < 0].sort_values(by="SHAP_Value", ascending=True)

    pos_sum = pos_df["SHAP_Value"].sum() if not pos_df.empty else 0.0
    neg_sum = abs(neg_df["SHAP_Value"].sum()) if not neg_df.empty else 0.0

    fig = go.Figure()

    # Base Value Marker
    fig.add_trace(
        go.Bar(
            y=["Attribution Balance"],
            x=[base_value],
            name=f"Base E[f(X)]: {base_value:.3f}",
            orientation="h",
            marker=dict(color="#64748B"),
        )
    )

    # Positive Force (Increasing)
    fig.add_trace(
        go.Bar(
            y=["Attribution Balance"],
            x=[pos_sum],
            name=f"Upward Push: +{pos_sum:.3f}",
            orientation="h",
            marker=dict(color="#EF4444"),
        )
    )

    # Negative Force (Decreasing)
    fig.add_trace(
        go.Bar(
            y=["Attribution Balance"],
            x=[-neg_sum],
            name=f"Downward Push: -{neg_sum:.3f}",
            orientation="h",
            marker=dict(color="#06B6D4"),
        )
    )

    fig.update_layout(
        title=dict(text=f"{title} (f(x) = {model_output:.3f})", font=dict(size=15, color="#F8FAFC")),
        barmode="relative",
        xaxis_title="Net Shift from Baseline",
        legend=dict(orientation="h", y=-0.2, bgcolor="rgba(30, 41, 59, 0.8)"),
        **DARK_LAYOUT_TEMPLATE,
    )
    return fig


def plot_shap_dependence(
    primary_feature: str,
    feature_values: List[Any],
    shap_values: List[float],
    interaction_feature: Optional[str] = None,
    interaction_values: Optional[List[Any]] = None,
    title: Optional[str] = None,
) -> go.Figure:
    """Render SHAP dependence scatter plot with optional interaction color mapping."""
    plot_title = title or f"SHAP Dependence: '{primary_feature}'"

    fig = go.Figure()

    if interaction_feature and interaction_values:
        fig.add_trace(
            go.Scatter(
                x=feature_values,
                y=shap_values,
                mode="markers",
                marker=dict(
                    color=interaction_values,
                    colorscale="Viridis",
                    size=7,
                    opacity=0.8,
                    showscale=True,
                    colorbar=dict(title=interaction_feature),
                ),
                text=[f"{primary_feature}: {x}<br>SHAP: {s:.3f}<br>{interaction_feature}: {i}" for x, s, i in zip(feature_values, shap_values, interaction_values)],
                hoverinfo="text",
                name="Observations",
            )
        )
    else:
        fig.add_trace(
            go.Scatter(
                x=feature_values,
                y=shap_values,
                mode="markers",
                marker=dict(color="#06B6D4", size=7, opacity=0.8),
                name="Observations",
            )
        )

    # Zero line
    fig.add_hline(y=0, line=dict(color="#94A3B8", width=1.5, dash="dash"))

    fig.update_layout(
        title=dict(text=plot_title, font=dict(size=15, color="#F8FAFC")),
        xaxis_title=f"Feature Value: {primary_feature}",
        yaxis_title=f"SHAP Attribution for {primary_feature}",
        **DARK_LAYOUT_TEMPLATE,
    )
    return fig


def plot_interaction_matrix_heatmap(
    interaction_matrix: pd.DataFrame,
    title: str = "SHAP Pairwise Interaction Strength Matrix",
) -> go.Figure:
    """Render pairwise interaction strength heatmap."""
    fig = go.Figure(
        go.Heatmap(
            z=interaction_matrix.values,
            x=interaction_matrix.columns.tolist(),
            y=interaction_matrix.index.tolist(),
            colorscale=[[0, "#1E293B"], [0.5, "#4F46E5"], [1.0, "#A855F7"]],
            showscale=True,
            colorbar=dict(title="Mean |Interaction|"),
        )
    )

    fig.update_layout(
        title=dict(text=title, font=dict(size=15, color="#F8FAFC")),
        xaxis=dict(tickangle=-30),
        yaxis=dict(autorange="reversed"),
        **DARK_LAYOUT_TEMPLATE,
    )
    return fig

