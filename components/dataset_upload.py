"""Dataset upload component for the AI 'Why Did This Happen?' Analyzer.

Handles drag-and-drop file upload, format and size validation, sample dataset
loading, and session state synchronization.
"""

from pathlib import Path
from typing import Optional, Callable
import streamlit as st
import pandas as pd

from config.constants import SUPPORTED_EXTENSIONS, MAX_FILE_SIZE_MB
from modules.data_loader import DataLoader
from modules.data_profiler import DataProfiler


def render_dataset_uploader(on_dataset_loaded: Optional[Callable] = None) -> Optional[pd.DataFrame]:
    """Renders the drag-and-drop upload widget and sample dataset picker."""
    st.markdown(
        """
        <div style="margin-bottom: 1.25rem;">
            <h2 style="font-size: 1.35rem; font-weight: 700; color: #F8FAFC; margin-bottom: 0.25rem;">
                📥 Dataset Ingestion & Source Selection
            </h2>
            <p style="font-size: 0.88rem; color: #94A3B8; margin-bottom: 0;">
                Upload your structured dataset (CSV, Excel) to begin automated profiling, data quality auditing, and root-cause analysis.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    loader = DataLoader(max_file_size_mb=MAX_FILE_SIZE_MB)
    profiler = DataProfiler()

    # Upload options: User Upload vs Built-in Sample Datasets
    upload_tab, sample_tab = st.tabs(["📁 Upload Your File", "🧪 Load Built-in Sample Datasets"])

    with upload_tab:
        col_upload, col_guidelines = st.columns([1.6, 1.0])

        with col_upload:
            uploaded_file = st.file_uploader(
                label="Choose a CSV or Excel file",
                type=["csv", "xlsx", "xls", "parquet"],
                help=f"Supported extensions: {', '.join(SUPPORTED_EXTENSIONS)}. Max size: {MAX_FILE_SIZE_MB}MB.",
                key="primary_file_uploader",
                label_visibility="collapsed",
            )

            if uploaded_file is not None:
                # Check if this is a newly uploaded file compared to session state
                current_name = st.session_state.get("dataset_filename")
                file_has_changed = current_name != uploaded_file.name

                if file_has_changed or st.session_state.get("uploaded_dataset") is None:
                    with st.spinner(f"Ingesting and parsing '{uploaded_file.name}'..."):
                        try:
                            df, metadata = loader.load_file(uploaded_file, file_name=uploaded_file.name)
                            classifications = profiler.identify_column_types(df)

                            # Secure session state storage
                            _store_dataset_in_session(df, uploaded_file.name, metadata, classifications)
                            st.success(f"Successfully uploaded and verified **{uploaded_file.name}**!")
                            if on_dataset_loaded:
                                on_dataset_loaded()
                            st.rerun()

                        except ValueError as val_err:
                            st.error(f"⚠️ Validation Error: {str(val_err)}")
                        except Exception as err:
                            st.error(f"❌ Failed to parse uploaded file: {str(err)}")

        with col_guidelines:
            st.markdown(
                f"""
                <div style="background: #111827; border: 1px solid #1F2937; border-radius: 10px; padding: 1.1rem; height: 100%;">
                    <div style="font-size: 0.8rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em; color: #38BDF8; margin-bottom: 0.5rem;">
                        📋 Ingestion Specifications
                    </div>
                    <ul style="font-size: 0.8rem; color: #94A3B8; margin-bottom: 0; padding-left: 1.2rem; line-height: 1.7;">
                        <li><strong>Formats:</strong> CSV, Excel (.xlsx, .xls), Parquet</li>
                        <li><strong>Size Threshold:</strong> Up to {MAX_FILE_SIZE_MB} MB</li>
                        <li><strong>Encoding:</strong> Auto-detected (UTF-8, Latin-1, CP1252)</li>
                        <li><strong>Delimiters:</strong> Auto-detected comma, tab, semicolon</li>
                        <li><strong>Privacy:</strong> Uploads stay in memory during your session</li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with sample_tab:
        st.markdown(
            "<p style='font-size: 0.85rem; color: #94A3B8;'>Select an enterprise benchmark scenario to explore the analyzer immediately:</p>",
            unsafe_allow_html=True,
        )
        sample_col1, sample_col2, sample_col3 = st.columns(3)

        with sample_col1:
            st.markdown(
                """
                <div style="background: #131B2E; border: 1px solid #1E293B; border-radius: 8px; padding: 1rem; margin-bottom: 0.75rem;">
                    <div style="font-size: 0.9rem; font-weight: 700; color: #F8FAFC;">👥 Customer Churn Benchmark</div>
                    <div style="font-size: 0.78rem; color: #94A3B8; margin-top: 0.3rem;">
                        1,000 subscriber accounts with demographics, contract length, monthly charges, and churn outcome.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if st.button("Load Churn Dataset", key="btn_load_churn", use_container_width=True):
                _load_prebuilt_sample("churn")
                st.rerun()

        with sample_col2:
            st.markdown(
                """
                <div style="background: #131B2E; border: 1px solid #1E293B; border-radius: 8px; padding: 1rem; margin-bottom: 0.75rem;">
                    <div style="font-size: 0.9rem; font-weight: 700; color: #F8FAFC;">📈 Sales & Revenue Dataset</div>
                    <div style="font-size: 0.78rem; color: #94A3B8; margin-top: 0.3rem;">
                        800 commercial transactions with deal size, discounts, rep experience, and final sales revenue.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if st.button("Load Sales Dataset", key="btn_load_sales", use_container_width=True):
                _load_prebuilt_sample("sales")
                st.rerun()

        with sample_col3:
            st.markdown(
                """
                <div style="background: #131B2E; border: 1px solid #1E293B; border-radius: 8px; padding: 1rem; margin-bottom: 0.75rem;">
                    <div style="font-size: 0.9rem; font-weight: 700; color: #F8FAFC;">🩺 Data Quality Stress Test</div>
                    <div style="font-size: 0.78rem; color: #94A3B8; margin-top: 0.3rem;">
                        500 records containing missing cells, duplicate rows, constant columns, and date formats.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if st.button("Load Quality Stress Test", key="btn_load_stress", use_container_width=True):
                _load_prebuilt_sample("stress_test")
                st.rerun()

    # Active Dataset Status Bar
    if st.session_state.get("uploaded_dataset") is not None:
        df = st.session_state["uploaded_dataset"]
        fname = st.session_state.get("dataset_filename", "active_dataset.csv")
        meta = st.session_state.get("dataset_metadata", {})
        
        st.markdown(
            f"""
            <div style="margin-top: 1rem; background: rgba(16, 185, 129, 0.08); border: 1px solid rgba(16, 185, 129, 0.3); border-radius: 8px; padding: 0.85rem 1.2rem; display: flex; justify-content: space-between; align-items: center;">
                <div style="display: flex; align-items: center; gap: 0.75rem;">
                    <span style="font-size: 1.25rem;">✅</span>
                    <div>
                        <span style="font-weight: 700; color: #34D399; font-size: 0.92rem;">Active Dataset: {fname}</span>
                        <span style="color: #94A3B8; font-size: 0.82rem; margin-left: 0.75rem;">
                            ({df.shape[0]:,} rows × {df.shape[1]} columns | {meta.get('load_duration_seconds', 0)}s parse)
                        </span>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        col_left, col_right = st.columns([4, 1])
        with col_right:
            if st.button("🔄 Unload / Reset Dataset", key="btn_reset_dataset", use_container_width=True):
                _clear_dataset_session()
                st.rerun()

    return st.session_state.get("uploaded_dataset")


def _store_dataset_in_session(
    df: pd.DataFrame,
    filename: str,
    metadata: dict,
    classifications: dict,
) -> None:
    """Store dataset and clear stale state safely."""
    st.session_state["uploaded_dataset"] = df
    st.session_state["original_dataset"] = df.copy(deep=True)
    st.session_state["dataset_filename"] = filename
    st.session_state["dataset_metadata"] = metadata
    st.session_state["column_classifications"] = classifications
    
    # Reset target selections when a new dataset is loaded
    st.session_state["selected_target_variable"] = None
    st.session_state["selected_outcome"] = None


def _clear_dataset_session() -> None:
    """Clear all dataset-related state variables."""
    keys_to_clear = [
        "uploaded_dataset",
        "original_dataset",
        "dataset_filename",
        "dataset_metadata",
        "column_classifications",
        "selected_target_variable",
        "selected_outcome",
    ]
    for k in keys_to_clear:
        if k in st.session_state:
            del st.session_state[k]


def _load_prebuilt_sample(sample_type: str) -> None:
    """Load or generate benchmark datasets for demonstration."""
    sample_dir = Path(__file__).resolve().parent.parent / "data" / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)
    loader = DataLoader()
    profiler = DataProfiler()

    filepath = sample_dir / f"{sample_type}.csv"
    if not filepath.exists():
        from utils.sample_generator import generate_sample_dataset
        generate_sample_dataset(sample_type, filepath)

    df, metadata = loader.load_file(filepath, file_name=f"{sample_type}.csv")
    classifications = profiler.identify_column_types(df)
    _store_dataset_in_session(df, f"{sample_type}.csv", metadata, classifications)
    st.success(f"Benchmark dataset '{sample_type}.csv' loaded successfully.")
