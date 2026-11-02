"""Enterprise sidebar component for navigation and system status."""

import streamlit as st
from config.constants import (
    APP_NAME,
    APP_VERSION,
    APP_TAGLINE,
    APP_ICON,
    NavigationSection,
    NAV_ICONS,
)
from utils.helpers import get_system_info


def render_sidebar() -> NavigationSection:
    """Renders the enterprise sidebar and returns the active navigation selection."""
    with st.sidebar:
        # App Branding Header
        st.markdown(
            f"""
            <div style="padding: 0.5rem 0 1.5rem 0; border-bottom: 1px solid #1E293B;">
                <div style="display: flex; align-items: center; gap: 0.75rem; margin-bottom: 0.35rem;">
                    <span style="font-size: 1.8rem;">{APP_ICON}</span>
                    <span style="font-size: 1.15rem; font-weight: 800; color: #F8FAFC; letter-spacing: -0.01em;">{APP_NAME}</span>
                </div>
                <div style="font-size: 0.75rem; color: #94A3B8; line-height: 1.4;">{APP_TAGLINE}</div>
                <div style="margin-top: 0.6rem;">
                    <span style="font-size: 0.7rem; font-weight: 600; background: rgba(79, 70, 229, 0.2); color: #A5B4FC; padding: 2px 8px; border-radius: 9999px; border: 1px solid rgba(99, 102, 241, 0.3);">
                        v{APP_VERSION}
                    </span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
        st.markdown(
            "<p style='font-size: 0.75rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; color: #64748B; margin-bottom: 8px;'>Investigation Modules</p>",
            unsafe_allow_html=True,
        )

        # Navigation Options formatted with icons
        nav_options = list(NavigationSection)
        formatted_labels = {
            section: f"{NAV_ICONS.get(section, '🔹')}  {section.value}"
            for section in nav_options
        }

        selected_section = st.radio(
            label="Navigation",
            options=nav_options,
            format_func=lambda s: formatted_labels[s],
            label_visibility="collapsed",
        )

        st.markdown("<hr style='border: none; border-top: 1px solid #1E293B; margin: 1.5rem 0;'>", unsafe_allow_html=True)

        # Dataset Status Indicator
        uploaded_df = st.session_state.get("uploaded_dataset")
        filename = st.session_state.get("dataset_filename")
        target_col = st.session_state.get("selected_target_variable")

        if uploaded_df is not None:
            n_rows, n_cols = uploaded_df.shape
            target_badge = (
                f"<div style='font-size: 0.73rem; color: #818CF8; margin-top: 0.3rem;'>🎯 Target: <strong>{target_col}</strong></div>"
                if target_col
                else "<div style='font-size: 0.72rem; color: #94A3B8; margin-top: 0.3rem;'>Target: Not selected</div>"
            )
            st.markdown(
                f"""
                <div style="background: rgba(16, 185, 129, 0.08); border: 1px solid rgba(16, 185, 129, 0.3); border-radius: 8px; padding: 0.85rem; margin-bottom: 1rem;">
                    <div style="font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.05em; color: #34D399; font-weight: 700;">Active Dataset</div>
                    <div style="display: flex; align-items: center; gap: 0.5rem; margin-top: 0.35rem;">
                        <span style="height: 8px; width: 8px; background-color: #10B981; border-radius: 50%; display: inline-block;"></span>
                        <span style="font-size: 0.82rem; color: #F8FAFC; font-weight: 600; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">{filename}</span>
                    </div>
                    <div style="font-size: 0.72rem; color: #94A3B8; margin-top: 0.25rem;">
                        {n_rows:,} rows × {n_cols} columns
                    </div>
                    {target_badge}
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
                <div style="background: #111827; border: 1px solid #1F2937; border-radius: 8px; padding: 0.85rem; margin-bottom: 1rem;">
                    <div style="font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.05em; color: #94A3B8; font-weight: 600;">Dataset Status</div>
                    <div style="display: flex; align-items: center; gap: 0.5rem; margin-top: 0.35rem;">
                        <span style="height: 8px; width: 8px; background-color: #F59E0B; border-radius: 50%; display: inline-block;"></span>
                        <span style="font-size: 0.82rem; color: #E2E8F0; font-weight: 500;">No Dataset Loaded</span>
                    </div>
                    <div style="font-size: 0.72rem; color: #64748B; margin-top: 0.3rem;">Upload CSV or Excel in Dataset Overview.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Environment Details Expander
        with st.expander("⚙️ System Environment", expanded=False):
            info = get_system_info()
            st.markdown(
                f"""
                <div style="font-size: 0.75rem; color: #94A3B8; line-height: 1.6;">
                    <strong>Python:</strong> {info['python_version']}<br>
                    <strong>Streamlit:</strong> {info['streamlit_version']}<br>
                    <strong>Platform:</strong> {info['platform'].split('-')[0]}<br>
                    <strong>Arch:</strong> {info['architecture']}
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Phase Status
        st.markdown(
            """
            <div style="text-align: center; font-size: 0.7rem; color: #38BDF8; margin-top: 1.5rem; font-weight: 600;">
                Phase 13: Automated Investigation Reporting Active
            </div>
            """,
            unsafe_allow_html=True,
        )

    return selected_section
