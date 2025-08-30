import streamlit as st
import pandas as pd
import traceback
import os
from utils.data_loader import CSVProcessor

def load_file_data(uploaded_file):
    """Load data from uploaded file using CSVProcessor."""
    try:
        processor = CSVProcessor()
        data = processor.load_and_validate(uploaded_file)
        return data
    except Exception as e:
        st.error(f"Error loading {uploaded_file.name}: {str(e)}")
        return None


def multi_file_upload_section():
    """Handle multiple file uploads and configure joins."""
    st.header("📁 Upload Your Datasets")

    uploaded_files = st.file_uploader(
        "Choose CSV files",
        type=['csv'],
        accept_multiple_files=True,
        help="Upload one or more CSV files. If multiple files are uploaded, you'll configure joins next."
    )

    if uploaded_files:
        st.subheader("Uploaded Files Overview")
        for file in uploaded_files:
            if file.name not in st.session_state.uploaded_files:
                data = load_file_data(file)
                if data is not None:
                    st.session_state.uploaded_files[file.name] = data

        # Show summary
        for filename, data in st.session_state.uploaded_files.items():
            with st.expander(f"📄 {filename} ({len(data):,} rows, {len(data.columns)} cols)"):
                col1, col2, col3, col4 = st.columns(4)
                with col1: st.metric("Rows", f"{len(data):,}")
                with col2: st.metric("Columns", len(data.columns))
                with col3: st.metric("Missing Values", f"{data.isnull().sum().sum():,}")
                with col4: st.metric("Memory", f"{data.memory_usage(deep=True).sum()/(1024*1024):.1f} MB")
                st.dataframe(data.head(3), use_container_width=True)

        # If multiple files → configure joins
        if len(st.session_state.uploaded_files) > 1:
            configure_joins()
        else:
            filename = list(st.session_state.uploaded_files.keys())[0]
            st.session_state.joined_data = st.session_state.uploaded_files[filename]
            st.session_state.original_data = st.session_state.joined_data.copy()
            if st.button("Proceed with Single File", type="primary"):
                st.session_state.current_phase = 1
                st.rerun()


def configure_joins():
    """Configure joins between multiple files."""
    st.markdown("---")
    st.subheader("🔗 Configure Table Joins")

    file_names = list(st.session_state.uploaded_files.keys())
    primary_table = st.selectbox("Select Primary Table:", file_names)
    other_tables = [f for f in file_names if f != primary_table]
    join_configs = []

    for i, table_name in enumerate(other_tables):
        st.write(f"**Join Configuration for: {table_name}**")
        col1, col2, col3 = st.columns(3)

        with col1:
            join_type = st.selectbox(
                "Join Type:", ["left", "inner", "right", "outer"], key=f"join_type_{i}"
            )
        with col2:
            primary_key = st.selectbox(
                "Primary Table Key:",
                st.session_state.uploaded_files[primary_table].columns.tolist(),
                key=f"primary_key_{i}",
            )
        with col3:
            secondary_key = st.selectbox(
                "Secondary Table Key:",
                st.session_state.uploaded_files[table_name].columns.tolist(),
                key=f"secondary_key_{i}",
            )

        # ✅ Default to all available columns (except secondary key)
        available_cols = [
            c for c in st.session_state.uploaded_files[table_name].columns
            if c != secondary_key
        ]
        selected_columns = st.multiselect(
            f"Columns from {table_name}:",
            options=available_cols,
            default=available_cols,  # 👈 auto-select all
            key=f"columns_{i}",
        )

        join_configs.append({
            'secondary_table': table_name,
            'join_type': join_type,
            'primary_key': primary_key,
            'secondary_key': secondary_key,
            'selected_columns': selected_columns,
        })

    st.session_state.join_config = {
        'primary_table': primary_table,
        'joins': join_configs
    }

    if st.button("Preview Join Result", type="secondary"):
        preview_join()
    if st.button("Execute Joins & Proceed", type="primary"):
        execute_joins()


def preview_join():
    """Preview join results (first 100 rows)."""
    try:
        joined_data = perform_joins(sample_size=100)
        st.subheader("Join Preview (First 100 rows)")
        st.dataframe(joined_data.head(10), use_container_width=True)
    except Exception as e:
        st.error(f"Error previewing join: {str(e)}")
        st.code(traceback.format_exc())


def perform_joins(sample_size=None):
    """Perform joins based on session join config."""
    config = st.session_state.join_config
    result_data = st.session_state.uploaded_files[config['primary_table']].copy()
    if sample_size:
        result_data = result_data.head(sample_size)

    for join in config['joins']:
        secondary_data = st.session_state.uploaded_files[join['secondary_table']].copy()
        cols = [join['secondary_key']] + join['selected_columns']
        secondary_data = secondary_data[cols]

        result_data = pd.merge(
            result_data,
            secondary_data,
            left_on=join['primary_key'],
            right_on=join['secondary_key'],
            how=join['join_type'],
        )
    return result_data


def execute_joins():
    """Execute joins and move to Phase 1."""
    try:
        joined_data = perform_joins()
        st.session_state.joined_data = joined_data
        st.session_state.original_data = joined_data.copy()
        st.session_state.current_phase = 1
        st.success(f"✅ Successfully joined! {len(joined_data):,} rows, {len(joined_data.columns)} cols")
        st.rerun()
    except Exception as e:
        st.error(f"Error executing joins: {str(e)}")
        st.code(traceback.format_exc())


def display_join_summary():
    """Display summary of joins performed."""
    if st.session_state.join_config and len(st.session_state.uploaded_files) > 1:
        st.subheader("Join Summary")
        config = st.session_state.join_config
        st.write(f"**Primary Table:** {config['primary_table']}")
        for join in config['joins']:
            st.write(f"- {join['join_type']} join with {join['secondary_table']} on {join['primary_key']}={join['secondary_key']}")
