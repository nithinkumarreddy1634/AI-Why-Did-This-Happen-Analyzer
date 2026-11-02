"""Dataset preview component for interactive data exploration."""

from typing import Optional
import streamlit as st
import pandas as pd
from utils.helpers import format_bytes


def render_dataset_preview(df: Optional[pd.DataFrame]) -> None:
    """Renders the interactive preview module for an uploaded dataset."""
    if df is None or df.empty:
        st.info("ℹ️ No dataset loaded. Please upload a dataset in the upload section above.")
        return

    n_rows, n_cols = df.shape
    memory_footprint = df.memory_usage(deep=True).sum()

    st.markdown(
        """
        <div style="margin-top: 1.5rem; margin-bottom: 0.75rem;">
            <h3 style="font-size: 1.2rem; font-weight: 700; color: #F8FAFC;">
                🔍 Dataset Preview & Interactive Exploration
            </h3>
            <p style="font-size: 0.85rem; color: #94A3B8;">
                Explore raw records, verify schema parsing, and inspect sample records across head, tail, and random slices.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Dimensional summary pill row
    dim_col1, dim_col2, dim_col3, dim_col4 = st.columns(4)
    with dim_col1:
        st.metric("Total Records", f"{n_rows:,}")
    with dim_col2:
        st.metric("Total Columns", f"{n_cols}")
    with dim_col3:
        st.metric("Dataset Shape", f"({n_rows:,}, {n_cols})")
    with dim_col4:
        st.metric("Memory Footprint", format_bytes(memory_footprint))

    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

    # Preview controls
    ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([1.5, 1.5, 3])
    with ctrl_col1:
        preview_mode = st.radio(
            "Slice Perspective:",
            options=["First Records (Head)", "Last Records (Tail)", "Random Sample"],
            horizontal=True,
            key="preview_slice_mode",
        )
    with ctrl_col2:
        num_records = st.select_slider(
            "Records to display:",
            options=[5, 10, 20, 50, 100],
            value=10,
            key="preview_num_records",
        )

    # Slice the dataframe based on selected mode
    records_to_take = min(num_records, n_rows)
    if "Head" in preview_mode:
        slice_df = df.head(records_to_take)
        slice_label = f"First {records_to_take} rows (Index 0 to {records_to_take - 1})"
    elif "Tail" in preview_mode:
        slice_df = df.tail(records_to_take)
        slice_label = f"Last {records_to_take} rows (Index {max(0, n_rows - records_to_take)} to {n_rows - 1})"
    else:
        slice_df = df.sample(records_to_take, random_state=42) if n_rows > 0 else df
        slice_label = f"Random sample of {records_to_take} rows"

    st.caption(f"Displaying: **{slice_label}**")
    st.dataframe(
        slice_df,
        use_container_width=True,
        height=340,
    )

    # Quick Column Summary Expander
    with st.expander("📑 View Full Column Names & Raw Pandas Data Types", expanded=False):
        types_df = pd.DataFrame({
            "Column Index": range(len(df.columns)),
            "Column Name": df.columns,
            "Pandas Dtype": [str(t) for t in df.dtypes],
            "Non-Null Count": [f"{df[c].notna().sum():,} / {n_rows:,}" for c in df.columns],
            "Null Count": [f"{df[c].isna().sum():,}" for c in df.columns],
            "Unique Values": [f"{df[c].nunique():,}" for c in df.columns],
        })
        st.dataframe(types_df, use_container_width=True, hide_index=True)
