import streamlit as st
import pandas as pd
import traceback
import os
from utils.data_sampler import DataSampler

# Handles: Phase 2 (AI-powered detection), results display, renaming updates

def phase2_processing(data):
    """Phase 2: LLM-powered Issue Detection"""
    st.markdown('<div class="phase-header phase-2">Phase 2: AI-Powered Issue Detection</div>', unsafe_allow_html=True)

    # --- Data summary ---
    st.subheader("Phase 1 Cleaned Data")
    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric("Rows", f"{len(data):,}")
    with col2: st.metric("Columns", len(data.columns))
    with col3: st.metric("Missing Values", f"{data.isnull().sum().sum():,}")
    with col4: st.metric("Memory", f"{data.memory_usage(deep=True).sum()/(1024*1024):.1f} MB")
    st.dataframe(data.head(10), use_container_width=True)

    # --- Phase 1 reports ---
    if st.session_state.phase1_reports:
        with st.expander("Phase 1 Cleaning Reports"):
            if 'cleaning' in st.session_state.phase1_reports:
                cleaning_report = st.session_state.phase1_reports['cleaning']
                if not cleaning_report.empty:
                    st.subheader("Cleaning Operations")
                    st.dataframe(cleaning_report, use_container_width=True)
            if 'missing_values' in st.session_state.phase1_reports:
                missing_report = st.session_state.phase1_reports['missing_values']
                if not missing_report.empty:
                    st.subheader("Missing Values Imputation")
                    st.dataframe(missing_report, use_container_width=True)

    # --- LLM Config ---
    st.subheader("AI Analysis Configuration")
    api_key_present = bool(os.getenv('OPENAI_API_KEY'))
    if not api_key_present:
        st.error("❌ OpenAI API key not found. Please set OPENAI_API_KEY in .env or env variables.")
        return

    col1, col2 = st.columns(2)
    with col1:
        model_choice = st.selectbox("AI Model", ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"], index=0)
        temperature = st.slider("Temperature", 0.0, 1.0, 0.1, 0.1)
    with col2:
        max_tokens = st.number_input("Max Tokens", 500, 4000, 1500, 100)
        sample_size = st.number_input("Sample Size for Analysis", 50, 500, 150, 25)

    # --- Data Sampling ---
    st.subheader("Data Sampling for Analysis")
    sampler = DataSampler(max_rows=sample_size)
    sample_info = sampler.create_sample(data)

    col1, col2, col3 = st.columns(3)
    with col1: st.metric("Original Rows", f"{sample_info['original_rows']:,}")
    with col2: st.metric("Sample Rows", f"{sample_info['sampled_rows']:,}")
    with col3: st.metric("Sampling Ratio", f"{sample_info['sampling_ratio']:.1%}")
    st.info(f"Sampling Strategy: {sample_info['sampling_strategy'].replace('_', ' ').title()}")

    with st.expander("View Sample Data"):
        st.dataframe(sample_info['sampled_data'].head(10), use_container_width=True)

    # --- Run Analysis ---
    if st.button("Run Phase 2 AI Analysis", type="primary"):
        with st.spinner("Running AI analysis..."):
            try:
                from utils.llm_interface_langchain import LLMInterfaceLangChain, LLMConfig
                config = LLMConfig(model_name=model_choice, temperature=temperature, max_tokens=max_tokens)
                llm_interface = LLMInterfaceLangChain(config)

                column_info = sampler.extract_column_info(sample_info['sampled_data'])
                analysis_results = llm_interface.analyze_data_quality(
                    sample_info['sampled_data'], column_info
                )

                st.session_state.phase2_results = analysis_results
                st.session_state.current_phase = 3
                st.success("✅ Phase 2 analysis completed!")
                st.rerun()

            except Exception as e:
                st.error(f"Error in Phase 2: {str(e)}")
                st.code(traceback.format_exc())

# -------------------- DISPLAY RESULTS --------------------

def display_phase2_results():
    """Display Phase 2 AI Analysis results"""
    results = st.session_state.phase2_results

    # Overall metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric("Issues Found", results['total_issues'])
    with col2: st.metric("Quality Score", f"{results.get('data_quality_score', 'N/A')}/100")
    with col3:
        high_severity = sum(1 for i in results['issues'] if i['severity'].lower() == 'high')
        st.metric("High Severity", high_severity)
    with col4:
        categories = set(issue['category'] for issue in results['issues'])
        st.metric("Categories", len(categories))

    # Issues table
    if results['issues']:
        st.subheader("Detected Issues")
        issues_df = pd.DataFrame([
            {
                'Category': issue['category'].title(),
                'Severity': issue['severity'].title(),
                'Description': issue['description'][:100] + "..." if len(issue['description']) > 100 else issue['description'],
                'Affected Columns': ', '.join(issue['affected_columns']),
                'Fix Approach': issue['fix_approach'][:80] + "..." if len(issue['fix_approach']) > 80 else issue['fix_approach']
            }
            for issue in results['issues']
        ])
        st.dataframe(issues_df, use_container_width=True)

    # Column renaming
    if results.get('column_renaming'):
        st.subheader("Column Renaming Suggestions")
        renaming_df = pd.DataFrame([
            {'Current Name': old, 'Suggested Name': new}
            for old, new in results['column_renaming'].items()
            if old != new and not new.startswith('Use clear')
        ])
        if not renaming_df.empty:
            st.dataframe(renaming_df, use_container_width=True)

    # Recommendations
    if results.get('recommendations'):
        st.subheader("Recommendations")
        for i, rec in enumerate(results['recommendations'], 1):
            st.write(f"{i}. {rec}")

# -------------------- COLUMN NAME UPDATES --------------------

def update_issues_with_renamed_columns(issues, column_mappings):
    """Update issue references to match renamed columns."""
    updated_issues = []
    for issue in issues:
        updated_issue = issue.copy()
        updated_issue['affected_columns'] = [
            column_mappings.get(col, col) for col in issue['affected_columns']
        ]

        description = issue['description']
        fix_approach = issue['fix_approach']
        for old, new in column_mappings.items():
            description = description.replace(old, new)
            fix_approach = fix_approach.replace(old, new)

        updated_issue['description'] = description
        updated_issue['fix_approach'] = fix_approach
        updated_issues.append(updated_issue)

    return updated_issues
