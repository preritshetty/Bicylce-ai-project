import streamlit as st
import os
import pandas as pd

FINAL_OUT_PATH = "data/final_cleaned.csv"

def initialize_session_state():
    """Initialize Streamlit session state variables with defaults."""
    defaults = {
        "current_phase": 0,
        "uploaded_files": {},
        "join_config": {},
        "joined_data": None,
        "original_data": None,
        "phase1_data": None,
        "phase2_results": None,
        "phase3_results": None,
        "phase1_reports": {},
        "processing_complete": False,
        "cleaned_df": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def get_final_clean_df() -> pd.DataFrame | None:
    """Return the final cleaned dataset, preferring in-memory, else from disk."""
    df = st.session_state.get("cleaned_df")
    if df is not None and not df.empty:
        return df
    if os.path.exists(FINAL_OUT_PATH):
        try:
            df = pd.read_csv(FINAL_OUT_PATH)
            return df if not df.empty else None
        except Exception:
            return None
    return None
