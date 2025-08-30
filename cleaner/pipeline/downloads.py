import streamlit as st
import pandas as pd
import json
import os
import tempfile
import zipfile
from io import StringIO
from datetime import datetime

# -------------------- HELPERS --------------------

def create_download_link(data, filename, file_type="csv"):
    """Generic download button for CSV/JSON data."""
    if file_type == "csv":
        content = data.to_csv(index=False)
        mime = "text/csv"
    elif file_type == "json":
        content = json.dumps(data, indent=2)
        mime = "application/json"
    else:
        content = data
        mime = "application/octet-stream"

    return st.download_button(
        label=f"Download {filename}",
        data=content.encode() if isinstance(content, str) else content,
        file_name=filename,
        mime=mime,
    )

# -------------------- CREATE DOWNLOADS --------------------

def create_final_downloads():
    """Create download links for all outputs (Original, Phase1, Phase2, Phase3)."""
    st.subheader("📥 Download Results")

    col1, col2, col3 = st.columns(3)

    # --- Original ---
    with col1:
        st.write("**Original Data**")
        if st.session_state.original_data is not None:
            create_download_link(st.session_state.original_data, "original_data.csv")

    # --- Phase 1 ---
    with col2:
        st.write("**Phase 1 Cleaned Data**")
        if st.session_state.phase1_data is not None:
            create_download_link(st.session_state.phase1_data, "phase1_cleaned_data.csv")

    # --- Phase 2 ---
    with col3:
        st.write("**Phase 2 Analysis**")
        if st.session_state.phase2_results:
            create_download_link(st.session_state.phase2_results, "phase2_analysis.json", "json")

    # --- Phase 3 ---
    if st.session_state.phase3_results:
        st.write("**Phase 3 Results**")
        col1, col2, col3, col4 = st.columns(4)

        # Renamed data
        with col1:
            if hasattr(st.session_state, "phase3_renamed_data"):
                create_download_link(st.session_state.phase3_renamed_data, "phase3_renamed_data.csv")

        # Flagged data
        with col2:
            flagged_file = st.session_state.phase3_results.get("flagged_data_file")
            if flagged_file and os.path.exists(flagged_file):
                flagged = pd.read_csv(flagged_file)
                create_download_link(flagged, "flagged_data.csv")

        # Execution report
        with col3:
            report_file = st.session_state.phase3_results.get("execution_report_file")
            if report_file and os.path.exists(report_file):
                with open(report_file, "r") as f:
                    report = json.load(f)
                create_download_link(report, "execution_report.json", "json")

        # Flag mapping
        with col4:
            mapping_file = st.session_state.phase3_results.get("flag_mapping_file")
            if mapping_file and os.path.exists(mapping_file):
                with open(mapping_file, "r") as f:
                    fmap = json.load(f)
                create_download_link(fmap, "flag_mapping.json", "json")

        # Renaming report
        if st.session_state.phase3_results.get("applied_renamings"):
            st.write("**Column Renaming Report**")
            renaming_report = {
                "applied_renamings": st.session_state.phase3_results["applied_renamings"],
                "timestamp": datetime.now().isoformat(),
                "total_columns_renamed": len(st.session_state.phase3_results["applied_renamings"]),
            }
            create_download_link(renaming_report, "column_renaming_report.json", "json")

    # --- Zip Package ---
    if st.button("Create Complete Results Package"):
        create_results_package_with_renaming()

# -------------------- CREATE PACKAGE --------------------

def create_results_package_with_renaming():
    """Create a ZIP file with all results (Original → Phase3)."""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmp_file:
            with zipfile.ZipFile(tmp_file.name, "w") as zipf:

                # Original
                if st.session_state.original_data is not None:
                    buf = StringIO()
                    st.session_state.original_data.to_csv(buf, index=False)
                    zipf.writestr("original_data.csv", buf.getvalue())

                # Phase 1
                if st.session_state.phase1_data is not None:
                    buf = StringIO()
                    st.session_state.phase1_data.to_csv(buf, index=False)
                    zipf.writestr("phase1_cleaned_data.csv", buf.getvalue())

                # Phase 2
                if st.session_state.phase2_results:
                    zipf.writestr("phase2_analysis.json", json.dumps(st.session_state.phase2_results, indent=2))

                # Phase 3 Renamed
                if hasattr(st.session_state, "phase3_renamed_data"):
                    buf = StringIO()
                    st.session_state.phase3_renamed_data.to_csv(buf, index=False)
                    zipf.writestr("phase3_renamed_data.csv", buf.getvalue())

                # Phase 3 Files
                results = st.session_state.phase3_results
                if results:
                    if os.path.exists(results.get("flagged_data_file", "")):
                        zipf.write(results["flagged_data_file"], "phase3_flagged_data.csv")
                    if os.path.exists(results.get("execution_report_file", "")):
                        zipf.write(results["execution_report_file"], "phase3_execution_report.json")
                    if os.path.exists(results.get("flag_mapping_file", "")):
                        zipf.write(results["flag_mapping_file"], "phase3_flag_mapping.json")

                    # Renaming report
                    if results.get("applied_renamings"):
                        renaming_report = {
                            "applied_renamings": results["applied_renamings"],
                            "timestamp": datetime.now().isoformat(),
                            "total_columns_renamed": len(results["applied_renamings"]),
                        }
                        zipf.writestr("column_renaming_report.json", json.dumps(renaming_report, indent=2))

            with open(tmp_file.name, "rb") as f:
                zip_data = f.read()

            st.download_button(
                label="⬇️ Download Complete Results Package",
                data=zip_data,
                file_name=f"data_quality_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                mime="application/zip",
            )

        os.unlink(tmp_file.name)
    except Exception as e:
        st.error(f"Error creating results package: {str(e)}")
