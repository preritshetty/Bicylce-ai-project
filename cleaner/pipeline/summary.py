import streamlit as st
import pandas as pd

def enhanced_data_comparison_tab():
    """Compare dataset transformations across phases (Original → Phase1 → Phase3)."""
    st.subheader("📊 Data Transformation Summary")

    comparison_data = []

    # --- Original Data ---
    if st.session_state.original_data is not None:
        original = st.session_state.original_data
        comparison_data.append({
            "Phase": "Original",
            "Rows": len(original),
            "Columns": len(original.columns),
            "Missing Values": original.isnull().sum().sum(),
            "Memory (MB)": f"{original.memory_usage(deep=True).sum() / (1024*1024):.2f}",
        })

    # --- Phase 1 Cleaned ---
    if st.session_state.phase1_data is not None:
        phase1 = st.session_state.phase1_data
        comparison_data.append({
            "Phase": "Phase 1 (Cleaned)",
            "Rows": len(phase1),
            "Columns": len(phase1.columns),
            "Missing Values": phase1.isnull().sum().sum(),
            "Memory (MB)": f"{phase1.memory_usage(deep=True).sum() / (1024*1024):.2f}",
        })

    # --- Phase 3 Renamed ---
    if hasattr(st.session_state, "phase3_renamed_data"):
        renamed = st.session_state.phase3_renamed_data
        comparison_data.append({
            "Phase": "Phase 3 (Renamed)",
            "Rows": len(renamed),
            "Columns": len(renamed.columns),
            "Missing Values": renamed.isnull().sum().sum(),
            "Memory (MB)": f"{renamed.memory_usage(deep=True).sum() / (1024*1024):.2f}",
        })

    # --- Phase 3 Flagged ---
    if st.session_state.phase3_results:
        try:
            flagged_data = pd.read_csv(st.session_state.phase3_results['flagged_data_file'])
            flagged_rows = (flagged_data['flag_status'] > 0).sum()
            comparison_data.append({
                "Phase": "Phase 3 (Flagged)",
                "Rows": len(flagged_data),
                "Columns": len(flagged_data.columns),
                "Missing Values": flagged_data.drop('flag_status', axis=1).isnull().sum().sum(),
                "Memory (MB)": f"{flagged_data.memory_usage(deep=True).sum() / (1024*1024):.2f}",
                "Flagged Rows": flagged_rows,
            })
        except Exception:
            pass

    # --- Show Table ---
    if comparison_data:
        st.dataframe(pd.DataFrame(comparison_data), use_container_width=True)

    # --- Show Samples ---
    cols = st.columns(3)
    with cols[0]:
        st.write("**Original Data Sample**")
        if st.session_state.original_data is not None:
            st.dataframe(st.session_state.original_data.head(), use_container_width=True)
    with cols[1]:
        st.write("**Phase 1 Cleaned Sample**")
        if st.session_state.phase1_data is not None:
            st.dataframe(st.session_state.phase1_data.head(), use_container_width=True)
    with cols[2]:
        st.write("**Phase 3 Renamed Sample**")
        if hasattr(st.session_state, "phase3_renamed_data"):
            st.dataframe(st.session_state.phase3_renamed_data.head(), use_container_width=True)
        else:
            st.info("No column renaming applied")
