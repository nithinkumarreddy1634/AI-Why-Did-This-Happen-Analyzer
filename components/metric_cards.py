"""Metric card UI components for high-level statistics."""

import streamlit as st
from typing import Optional


def render_metric_card(
    title: str,
    value: str,
    subtitle: str = "",
    badge_text: Optional[str] = None,
    badge_type: str = "blue",
) -> None:
    """Render a modern dark-themed enterprise metric card."""
    badge_html = (
        f'<span class="metric-badge badge-{badge_type}">{badge_text}</span>'
        if badge_text
        else ""
    )
    
    card_html = f"""
    <div class="metric-card">
        <div style="display: flex; justify-content: space-between; align-items: flex-start;">
            <div class="metric-title">{title}</div>
            {badge_html}
        </div>
        <div class="metric-value">{value}</div>
        <div class="metric-subtitle">{subtitle}</div>
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)
