import streamlit as st
import os
import pandas as pd
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# --- Local imports ---
from pipeline.session import initialize_session_state, get_final_clean_df
from pipeline.joins import multi_file_upload_section
from pipeline.phase1_cleaning import modified_phase1_processing
from pipeline.phase2_ai import phase2_processing, display_phase2_results
from pipeline.phase3_codegen import phase3_processing, display_phase3_results
from pipeline.summary import enhanced_data_comparison_tab
from pipeline.downloads import create_final_downloads
from analyzer.app import render_analysis   # Analysis tab from your existing analyzer/
FINAL_OUT_PATH = "data/final_cleaned.csv"

# -------------------- CLEANER TAB --------------------

def render_cleaner_tab():
    """Render the full pipeline (Upload → Phase1 → Phase2 → Phase3 → Results)."""

    # Sidebar progress
    st.sidebar.title("Pipeline Progress")
    phases = [
        "Upload & Join Data",
        "Phase 1: Cleaning",
        "Phase 2: AI Analysis",
        "Phase 3: Code Generation",
        "Results",
    ]
    for i, phase in enumerate(phases):
        if i <= st.session_state.current_phase:
            st.sidebar.success(f"✅ {phase}")
        elif i == st.session_state.current_phase + 1:
            st.sidebar.info(f"➡️ {phase}")
        else:
            st.sidebar.write(f"⏳ {phase}")

    # Sidebar uploaded files summary
    if st.session_state.uploaded_files:
        st.sidebar.markdown("---")
        st.sidebar.write("**Uploaded Files:**")
        for f in st.session_state.uploaded_files.keys():
            st.sidebar.write(f"📄 {f}")

    # --- State machine for pipeline ---
    if st.session_state.current_phase == 0:
        multi_file_upload_section()

    elif st.session_state.current_phase == 1:
        modified_phase1_processing()

    elif st.session_state.current_phase == 2:
        if st.session_state.phase1_data is not None:
            phase2_processing(st.session_state.phase1_data)
        else:
            st.error("Phase 1 data not available. Please restart the process.")
            if st.button("Restart"):
                for k in list(st.session_state.keys()):
                    del st.session_state[k]
                st.rerun()

    elif st.session_state.current_phase == 3:
        display_phase2_results()
        st.markdown("---")
        phase3_processing()

    elif st.session_state.current_phase >= 4:
        tab1, tab2, tab3, tab4 = st.tabs(["Phase 2 Results", "Phase 3 Results", "Comparison", "Downloads"])
        with tab1: display_phase2_results()
        with tab2: display_phase3_results()
        with tab3: enhanced_data_comparison_tab()
        with tab4: create_final_downloads()

    # Sidebar restart button
    st.sidebar.markdown("---")
    if st.sidebar.button("Start New Analysis"):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        try:
            if os.path.exists(FINAL_OUT_PATH):
                os.remove(FINAL_OUT_PATH)
            flag_map_path = "data/flag_mapping.json"
            if os.path.exists(flag_map_path):
                os.remove(flag_map_path)
        except Exception as e:
            st.warning(f"Could not clear old data: {e}")
        st.rerun()

# -------------------- MAIN ENTRY --------------------

def main():
    st.set_page_config(page_title="Data Quality Pipeline", page_icon="🔍", layout="wide")
    st.title("🔍 LLM Powered Analysis")
    st.markdown("**Transform your data through AI-powered quality detection and cleaning**")

    initialize_session_state()

    # Check if final cleaned dataset exists
    final_df_available = get_final_clean_df() is not None
    analyze_label = "📊 Analyze" if final_df_available else "📊 Analyze (locked)"

    tab_clean, tab_analyze = st.tabs(["🧹 Clean & Export", analyze_label])

    with tab_clean:
        render_cleaner_tab()

    with tab_analyze:
        df_for_analysis = get_final_clean_df()
        if df_for_analysis is None:
            st.info("🔒 Analyzer unlocks after cleaning is complete.")
            st.stop()
        render_analysis(df_for_analysis)

if __name__ == "__main__":
    main()
