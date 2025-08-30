import streamlit as st
import pandas as pd
import os
import json
import traceback
from pathlib import Path
from datetime import datetime

from phase3.flag_mapper import FlagMapper
from phase3.code_generator import CodeGenerator,CodeGenConfig
from phase3.code_executor import CodeExecutor
from pipeline.phase2_ai import update_issues_with_renamed_columns

# -------------------- PHASE 3 PROCESSING --------------------

def phase3_processing():
    """Phase 3: Code Generation and Execution with Column Renaming."""
    st.markdown('<div class="phase-header phase-3">Phase 3: Automated Code Generation & Column Renaming</div>', unsafe_allow_html=True)

    if not st.session_state.phase2_results:
        st.error("Phase 2 results not available. Please complete Phase 2 first.")
        return

    results = st.session_state.phase2_results
    st.info(f"Found {results['total_issues']} issues and "
            f"{len([k for k,v in results.get('column_renaming',{}).items() if k!=v and not v.startswith('Use clear')])} columns to rename")

    # --- Column Renaming Section ---
    st.subheader("Column Renaming Configuration")
    column_renaming = results.get('column_renaming', {})
    valid_renamings = {
        old: new for old, new in column_renaming.items()
        if old != new and not new.startswith("Use clear") and len(new.strip()) > 0
    }

    selected_renamings = {}
    if valid_renamings:
        st.write("**Suggested Renamings:**")
        for old, new in valid_renamings.items():
            col1, col2, col3 = st.columns([1,2,2])
            with col1: apply = st.checkbox("Apply", value=True, key=f"rename_{old}")
            with col2: st.write(f"**{old}**")
            with col3:
                final_name = st.text_input("New name:", value=new, key=f"new_{old}", label_visibility="collapsed")
            if apply and final_name.strip():
                selected_renamings[old] = final_name.strip()
    else:
        st.info("No renaming suggestions.")
    st.session_state.selected_renamings = selected_renamings

    # --- Config ---
    st.subheader("Code Generation Configuration")
    col1, col2 = st.columns(2)
    with col1:
        code_model = st.selectbox("Code Model", ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"], index=0)
        code_temperature = st.slider("Temperature", 0.0, 0.5, 0.1, 0.05)
    with col2:
        max_code_tokens = st.number_input("Max Tokens", 500, 2000, 1000, 100)
        execution_timeout = st.number_input("Execution Timeout (s)", 10, 120, 30, 10)

    # --- Generate & Execute ---
    if st.button("Apply Renaming & Generate Detection Codes", type="primary"):
        with st.spinner("Applying renaming and generating detection codes..."):
            try:
                working_data = st.session_state.phase1_data.copy() if st.session_state.phase1_data is not None else st.session_state.original_data.copy()

                if selected_renamings:
                    working_data = working_data.rename(columns=selected_renamings)
                    updated_issues = update_issues_with_renamed_columns(results['issues'], selected_renamings)
                    st.session_state.phase3_renamed_data = working_data
                    st.success(f"Applied {len(selected_renamings)} renamings")
                else:
                    updated_issues = results['issues']
                    st.session_state.phase3_renamed_data = working_data

                # Step 1: Flag Mapping
                mapper = FlagMapper()
                flag_mapping = mapper.create_flag_mapping(updated_issues)
                flag_mapping_file = mapper.save_mapping_to_file()

                # Step 2: Code Generation
                config = CodeGenConfig(
                    model_name=code_model,
                    temperature=code_temperature,
                    max_tokens=max_code_tokens
                )
                generator = CodeGenerator(config)

                sample_info = f"Sample data structure: {len(working_data.head(10))} rows, columns: {list(working_data.columns)}"
                generation_results = generator.generate_all_detection_codes(flag_mapping, sample_info)
                codes_file = generator.save_detection_codes(generation_results, "phase_outputs")

                # Step 3: Execute Codes
                os.makedirs("phase_outputs", exist_ok=True)
                temp_file = "phase_outputs/phase3_renamed_data.csv"
                working_data.to_csv(temp_file, index=False)
                executor = CodeExecutor(temp_file)
                execution_results = executor.run_complete_pipeline(codes_file)

                # Store results
                st.session_state.phase3_results = {
                    "flag_mapping_file": flag_mapping_file,
                    "codes_file": codes_file,
                    "execution_results": execution_results,
                    "output_files": execution_results.get("output_files", {}),
                    "flagged_data_file": execution_results["output_files"]["flagged_data"],
                    "execution_report_file": execution_results["output_files"]["execution_report"],
                    "renamed_data_file": temp_file,
                    "applied_renamings": selected_renamings,
                }

                # Save final cleaned dataset
                Path("data").mkdir(exist_ok=True)
                flagged = pd.read_csv(st.session_state.phase3_results["flagged_data_file"])
                flagged.to_csv("data/final_cleaned.csv", index=False)
                st.session_state["cleaned_df"] = flagged.copy()

                # Save flag mapping
                try:
                    with open(flag_mapping_file, "r") as f: fmap = json.load(f)
                    with open("data/flag_mapping.json", "w") as f2: json.dump(fmap, f2, indent=2)
                except Exception as e:
                    st.warning(f"Could not save flag mapping: {e}")

                st.success("✅ Phase 3 completed successfully!")
                st.session_state.current_phase = 4
                st.rerun()

            except Exception as e:
                st.error(f"Error in Phase 3: {str(e)}")
                st.code(traceback.format_exc())

# -------------------- DISPLAY RESULTS --------------------

def display_phase3_results():
    """Display Phase 3 results (renaming + execution)."""
    results = st.session_state.phase3_results
    execution = results["execution_results"]
    summary = execution["execution_summary"]

    # Show renamings
    applied = results.get("applied_renamings", {})
    if applied:
        st.subheader("Applied Column Renamings")
        df = pd.DataFrame([{"Original": old, "New": new} for old, new in applied.items()])
        st.dataframe(df, use_container_width=True)

    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric("Total Rows", f"{summary['total_rows']:,}")
    with col2: st.metric("Flagged Rows", f"{summary['flagged_rows']:,}")
    with col3: st.metric("Clean Rows", f"{summary['clean_rows']:,}")
    with col4: st.metric("Flag Coverage", f"{summary['flagged_percentage']}%")

    col1, col2, col3 = st.columns(3)
    with col1: st.metric("Successful Codes", f"{summary['successful_codes']}/{summary['successful_codes']+summary['failed_codes']}")
    with col2: st.metric("Success Rate", f"{summary['success_rate']}%")
    with col3: st.metric("Execution Time", f"{summary['execution_time_seconds']}s")

    # Individual results
    st.subheader("Detection Code Results")
    res_df = pd.DataFrame([
        {
            "Flag": r["flag_value"],
            "Status": "✅ Success" if r["success"] else "❌ Failed",
            "Rows Detected": r["rows_detected"],
            "Description": r["explanation"][:80] + "..." if len(r["explanation"]) > 80 else r["explanation"],
            "Error": r.get("error", "")[:50] + "..." if r.get("error") and len(r["error"]) > 50 else r.get("error", "")
        }
        for r in execution["individual_results"]
    ])
    st.dataframe(res_df, use_container_width=True)

    # Flag breakdown
    if execution.get("flag_breakdown", {}).get("flag_combinations"):
        st.subheader("Flag Combinations Found")
        comb = execution["flag_breakdown"]["flag_combinations"]
        comb_df = pd.DataFrame([
            {
                "Flag Status": c["flag_status"],
                "Individual Flags": "+".join(map(str, c["individual_flags"])),
                "Description": c["flag_description"],
                "Row Count": c["row_count"],
                "Binary": c["binary_representation"],
            }
            for c in comb
        ])
        st.dataframe(comb_df, use_container_width=True)
