"""Helper utility functions for the analyzer application."""

import sys
import platform
import streamlit as st
from typing import Dict, Any


def format_bytes(size_bytes: int) -> str:
    """Format file size in bytes to a human-readable string."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.2f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.2f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.2f} GB"


def format_number(val: float | int) -> str:
    """Format large numbers with compact or comma separated strings."""
    if isinstance(val, (int, float)):
        if abs(val) >= 1_000_000:
            return f"{val / 1_000_000:.1f}M"
        if abs(val) >= 1_000:
            return f"{val / 1_000:.1f}k"
        if isinstance(val, float):
            return f"{val:.2f}"
        return f"{val:,}"
    return str(val)


def get_system_info() -> Dict[str, Any]:
    """Retrieve runtime environment details."""
    return {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "streamlit_version": st.__version__,
        "architecture": platform.architecture()[0],
    }


def render_custom_css() -> None:
    """Inject custom enterprise dark dashboard styling into Streamlit."""
    custom_css = """
    <style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }

    /* Main background & Container padding */
    .stApp {
        background: radial-gradient(circle at 15% 15%, rgba(79, 70, 229, 0.08) 0%, transparent 40%),
                    radial-gradient(circle at 85% 85%, rgba(6, 182, 212, 0.06) 0%, transparent 40%),
                    #0B0F19;
        color: #F8FAFC;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #0E131F !important;
        border-right: 1px solid #1E293B !important;
    }

    section[data-testid="stSidebar"] div[class*="stSelectbox"] label,
    section[data-testid="stSidebar"] div[class*="stRadio"] label {
        color: #94A3B8 !important;
        font-weight: 500 !important;
    }

    /* Enterprise Hero Banner */
    .hero-banner {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.9) 100%);
        border: 1px solid #334155;
        border-radius: 14px;
        padding: 2.2rem 2.4rem;
        margin-bottom: 2rem;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.4), 0 8px 10px -6px rgba(0, 0, 0, 0.3);
        position: relative;
        overflow: hidden;
    }

    .hero-banner::after {
        content: "";
        position: absolute;
        top: 0;
        right: 0;
        width: 280px;
        height: 100%;
        background: radial-gradient(circle, rgba(79, 70, 229, 0.25) 0%, transparent 70%);
        pointer-events: none;
    }

    .hero-title {
        font-size: 2.25rem;
        font-weight: 800;
        letter-spacing: -0.025em;
        background: linear-gradient(135deg, #FFFFFF 30%, #818CF8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }

    .hero-tagline {
        font-size: 1.15rem;
        font-weight: 600;
        color: #38BDF8;
        margin-bottom: 0.85rem;
    }

    .hero-description {
        font-size: 0.95rem;
        line-height: 1.6;
        color: #CBD5E1;
        max-width: 850px;
        margin-bottom: 0;
    }

    /* Enterprise Metric Card */
    .metric-card {
        background: #131B2E;
        border: 1px solid #1E293B;
        border-radius: 12px;
        padding: 1.25rem 1.4rem;
        transition: all 0.25s ease-in-out;
        position: relative;
        height: 100%;
    }

    .metric-card:hover {
        border-color: #4F46E5;
        transform: translateY(-2px);
        box-shadow: 0 12px 20px -8px rgba(79, 70, 229, 0.25);
    }

    .metric-title {
        font-size: 0.82rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #94A3B8;
        font-weight: 600;
        margin-bottom: 0.35rem;
    }

    .metric-value {
        font-size: 1.75rem;
        font-weight: 800;
        color: #F8FAFC;
        margin-bottom: 0.25rem;
    }

    .metric-subtitle {
        font-size: 0.8rem;
        color: #64748B;
    }

    .metric-badge {
        display: inline-block;
        padding: 0.2rem 0.55rem;
        border-radius: 9999px;
        font-size: 0.72rem;
        font-weight: 600;
    }

    .badge-blue { background: rgba(56, 189, 248, 0.15); color: #38BDF8; }
    .badge-purple { background: rgba(168, 85, 247, 0.15); color: #C084FC; }
    .badge-emerald { background: rgba(16, 185, 129, 0.15); color: #34D399; }
    .badge-amber { background: rgba(245, 158, 11, 0.15); color: #FBBF24; }

    /* Section Placeholder Card */
    .section-card {
        background: #111827;
        border: 1px solid #1F2937;
        border-radius: 12px;
        padding: 1.75rem 2rem;
        margin-top: 1rem;
        margin-bottom: 1.5rem;
    }

    .section-card-title {
        font-size: 1.25rem;
        font-weight: 700;
        color: #F1F5F9;
        margin-bottom: 0.5rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    .section-card-desc {
        color: #94A3B8;
        font-size: 0.92rem;
        line-height: 1.6;
        margin-bottom: 1.25rem;
    }

    .phase-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 600;
        background: rgba(79, 70, 229, 0.2);
        color: #818CF8;
        border: 1px solid rgba(99, 102, 241, 0.3);
    }

    /* Code block styling */
    code {
        font-family: 'JetBrains Mono', monospace !important;
        background-color: #1E293B !important;
        color: #38BDF8 !important;
        padding: 0.15rem 0.4rem !important;
        border-radius: 4px !important;
    }
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)
