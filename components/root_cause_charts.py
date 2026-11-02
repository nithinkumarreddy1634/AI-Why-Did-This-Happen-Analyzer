"""Enterprise Plotly visualizations for Root Cause Investigation Studio."""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from modules.root_cause.root_cause_ranker import RootCauseRankingResult
from modules.root_cause.segment_contribution import SegmentAnalysisResult
from modules.root_cause.interaction_analysis import InteractionAnalysisResult

DARK_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#F8FAFC", family="Plus Jakarta Sans, sans-serif"),
    margin=dict(l=40, r=40, t=50, b=40),
)


def plot_root_cause_score_bar(ranking_result: RootCauseRankingResult, top_n: int = 8) -> go.Figure:
    """Horizontal bar chart showing multi-dimensional evidence scores for top candidates."""
    top_items = ranking_result.ranked_candidates[:top_n]
    if not top_items:
        fig = go.Figure()
        fig.update_layout(title="No candidates to display", **DARK_LAYOUT)
        return fig

    features = [item.feature for item in reversed(top_items)]
    scores = [item.overall_score for item in reversed(top_items)]
    strengths = [item.evidence_strength for item in reversed(top_items)]

    # Dynamic color mapping based on strength
    color_map = {
        "Strong Supporting Evidence": "#10B981",    # Emerald
        "Moderate Supporting Evidence": "#6366F1",  # Indigo
        "Limited Supporting Evidence": "#F59E0B",   # Amber
        "Conflicting Evidence": "#EC4899",          # Pink
        "Insufficient Evidence": "#64748B",         # Slate
    }
    bar_colors = [color_map.get(s, "#6366F1") for s in strengths]

    fig = go.Figure(
        go.Bar(
            x=scores,
            y=features,
            orientation="h",
            marker=dict(color=bar_colors, line=dict(color="rgba(255,255,255,0.15)", width=1)),
            text=[f"{s:.2f}" for s in scores],
            textposition="auto",
            hovertemplate="<b>%{y}</b><br>Evidence Score: %{x:.3f}<extra></extra>",
        )
    )

    fig.update_layout(
        title="<b>Potential Root Cause Candidates Ranked by Multi-Engine Evidence</b>",
        xaxis=dict(title="Consolidated Evidence Score [0.0 - 1.0]", range=[0, 1.05], gridcolor="#1E293B"),
        yaxis=dict(gridcolor="#1E293B"),
        height=max(320, 45 * len(features)),
        **DARK_LAYOUT,
    )
    return fig


def plot_evidence_matrix_heatmap(ranking_result: RootCauseRankingResult, top_n: int = 10) -> go.Figure:
    """Visual heatmap showing method-level support (Supporting, Neutral, Opposing, N/A) across candidates."""
    top_items = ranking_result.ranked_candidates[:top_n]
    if not top_items:
        fig = go.Figure()
        fig.update_layout(title="No candidates available", **DARK_LAYOUT)
        return fig

    methods = ["Statistical", "ML Importance", "SHAP", "LIME", "Anomaly"]
    features = [item.feature for item in top_items]

    # Map string status to numeric value for heatmap coloring
    # 2: Supporting (Green), 0: Neutral (Blue/Gray), -1: Opposing (Red/Pink), -2: Not Available (Dark Gray)
    val_map = {"Supporting": 2, "Neutral": 0, "Opposing": -1, "Not Available": -2}
    z_matrix = []
    text_matrix = []

    for item in top_items:
        row_z = []
        row_txt = []
        for m in methods:
            # Map method name to key
            key = (
                "Statistical" if m == "Statistical"
                else "ML Importance" if m == "ML Importance"
                else "SHAP (Global/Local)" if m == "SHAP"
                else "LIME (Local Rules)" if m == "LIME"
                else "Anomaly Concentration"
            )
            st = item.method_statuses.get(key, "Not Available")
            row_z.append(val_map.get(st, -2))
            row_txt.append(st)
        z_matrix.append(row_z)
        text_matrix.append(row_txt)

    colorscale = [
        [0.0, "#0F172A"],   # -2: Not Available (very dark slate)
        [0.25, "#EF4444"],  # -1: Opposing (Red)
        [0.5, "#334155"],   # 0: Neutral (Medium slate)
        [1.0, "#10B981"],   # 2: Supporting (Emerald green)
    ]

    fig = go.Figure(
        data=go.Heatmap(
            z=z_matrix,
            x=methods,
            y=features,
            text=text_matrix,
            texttemplate="%{text}",
            textfont=dict(size=11, color="#F8FAFC"),
            colorscale=colorscale,
            showscale=False,
            hoverongaps=False,
        )
    )

    fig.update_layout(
        title="<b>Cross-Method Evidence Matrix</b> (Multi-Engine Corroboration)",
        xaxis=dict(side="top"),
        yaxis=dict(autorange="reversed"),
        height=max(300, 36 * len(features)),
        **DARK_LAYOUT,
    )
    return fig


def plot_segment_contribution_waterfall(segment_result: SegmentAnalysisResult) -> go.Figure:
    """Bar chart decomposing the total outcome change share across segments."""
    segments = segment_result.segments
    if not segments:
        fig = go.Figure()
        fig.update_layout(title="No segment data available", **DARK_LAYOUT)
        return fig

    labels = [s.segment_label for s in segments]
    shares = [s.contribution_share_pct for s in segments]
    changes = [s.absolute_change for s in segments]

    colors = ["#10B981" if c >= 0 else "#EF4444" for c in changes]

    fig = go.Figure(
        go.Bar(
            x=labels,
            y=shares,
            marker=dict(color=colors, line=dict(color="rgba(255,255,255,0.15)", width=1)),
            text=[f"{sh:.1f}% (Δ={ch:+.2f})" for sh, ch in zip(shares, changes)],
            textposition="auto",
            hovertemplate="<b>Segment: %{x}</b><br>Contribution Share: %{y:.1f}%<extra></extra>",
        )
    )

    fig.update_layout(
        title=f"<b>Segment Contribution Share to Total Delta for '{segment_result.feature}'</b>",
        xaxis=dict(title="Segment / Cohort", gridcolor="#1E293B"),
        yaxis=dict(title="Contribution Share (%)", gridcolor="#1E293B"),
        height=380,
        **DARK_LAYOUT,
    )
    return fig


def plot_interaction_heatmap(interaction_result: InteractionAnalysisResult) -> go.Figure:
    """Heatmap showing outcome rate changes across combinations of two features."""
    matrix_df = interaction_result.matrix_df
    if matrix_df.empty:
        fig = go.Figure()
        fig.update_layout(title="No interaction data available", **DARK_LAYOUT)
        return fig

    fig = px.imshow(
        matrix_df,
        labels=dict(x=interaction_result.feature2, y=interaction_result.feature1, color="Outcome Delta (Δ)"),
        color_continuous_scale="Viridis",
        text_auto=".2f",
    )

    fig.update_layout(
        title=f"<b>Combined Factor Interaction: {interaction_result.feature1} × {interaction_result.feature2}</b>",
        xaxis=dict(gridcolor="#1E293B"),
        yaxis=dict(gridcolor="#1E293B"),
        height=400,
        **DARK_LAYOUT,
    )
    return fig

