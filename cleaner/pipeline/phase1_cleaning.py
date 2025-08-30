import streamlit as st
import pandas as pd
import traceback
from utils.basic_cleaning import BasicCleaner
from utils.missing_values import MissingValueCleaner

# --------------------- HELPERS ---------------------

def display_data_summary(data, title: str):
    """Display data summary with metrics and preview."""
    st.subheader(title)
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Rows", f"{len(data):,}")
    with col2:
        st.metric("Columns", len(data.columns))
    with col3:
        st.metric("Missing Values", f"{data.isnull().sum().sum():,}")
    with col4:
        memory_mb = data.memory_usage(deep=True).sum() / (1024 * 1024)
        st.metric("Memory Usage", f"{memory_mb:.1f} MB")

    st.dataframe(data.head(10), use_container_width=True)


def analyze_missing_values(data):
    """Analyze missing values in the dataset."""
    cleaner = MissingValueCleaner()
    return cleaner.analyze_missing_patterns(data)

# --------------------- PHASE 1 ---------------------

def modified_phase1_processing():
    """Phase 1: Basic Data Cleaning (for joined data)."""
    st.markdown('<div class="phase-header phase-1">Phase 1: Basic Data Cleaning</div>', unsafe_allow_html=True)

    if st.session_state.original_data is None:
        st.error("No data available for processing.")
        return None

    try:
        data = st.session_state.original_data
        st.success(f"Data ready for processing: {len(data)} rows, {len(data.columns)} columns")
        display_data_summary(data, "Dataset for Cleaning")

        # --- Cleaning Options ---
        st.subheader("Cleaning Options")
        col1, col2 = st.columns(2)
        with col1:
            remove_duplicates = st.checkbox("Remove duplicate rows", value=True)
            clean_text = st.checkbox("Clean text columns (trim whitespace)", value=True)
        with col2:
            case_standardization = st.selectbox(
                "Text case standardization", ["none", "title", "upper", "lower"], index=1
            )

        # --- Missing Values Handling ---
        st.subheader("Missing Values Handling")
        missing_analysis = analyze_missing_values(data)

        if missing_analysis['total_missing'] > 0:
            st.warning(f"Found {missing_analysis['total_missing']:,} missing values")

            if missing_analysis['missing_by_column']:
                missing_df = pd.DataFrame([
                    {
                        'Column': col,
                        'Missing Count': info['missing_count'],
                        'Missing %': f"{info['missing_percentage']:.1f}%",
                        'Suggested Strategy': info.get('suggested_strategy', 'mode')
                    }
                    for col, info in missing_analysis['missing_by_column'].items()
                ])
                st.dataframe(missing_df, use_container_width=True)

            handle_missing = st.checkbox("Handle missing values automatically", value=True)
        else:
            handle_missing = False
            st.success("No missing values found!")

        # --- Run Phase 1 Processing ---
        if st.button("Run Phase 1 Processing", type="primary"):
            with st.spinner("Processing Phase 1..."):
                cleaner = BasicCleaner()
                cleaned_data = cleaner.perform_basic_cleaning(
                    data,
                    remove_duplicates=remove_duplicates,
                    clean_text=clean_text,
                    case_type=case_standardization,
                )

                if handle_missing and missing_analysis['total_missing'] > 0:
                    missing_cleaner = MissingValueCleaner()
                    cleaned_data = missing_cleaner.handle_all_missing_values(cleaned_data)
                    missing_report = missing_cleaner.get_imputation_summary()
                    st.session_state.phase1_reports['missing_values'] = missing_report

                st.session_state.phase1_data = cleaned_data
                st.session_state.phase1_reports['cleaning'] = cleaner.get_cleaning_report()
                st.session_state.current_phase = 2

                st.success("Phase 1 completed successfully!")
                st.rerun()
        return data

    except Exception as e:
        st.error(f"Error in Phase 1: {str(e)}")
        st.code(traceback.format_exc())
        return None


def phase1_processing(uploaded_file):
    """Phase 1: Basic Data Cleaning (single uploaded file)."""
    st.markdown('<div class="phase-header phase-1">Phase 1: Basic Data Cleaning</div>', unsafe_allow_html=True)

    try:
        from utils.data_loader import CSVProcessor
        processor = CSVProcessor()
        data = processor.load_and_validate(uploaded_file)
        st.session_state.original_data = data.copy()
        st.success(f"Data loaded successfully: {len(data)} rows, {len(data.columns)} columns")

        display_data_summary(data, "Original Data")

        # Cleaning options
        st.subheader("Cleaning Options")
        col1, col2 = st.columns(2)
        with col1:
            remove_duplicates = st.checkbox("Remove duplicate rows", value=True)
            clean_text = st.checkbox("Clean text columns (trim whitespace)", value=True)
        with col2:
            case_standardization = st.selectbox(
                "Text case standardization", ["none", "title", "upper", "lower"], index=1
            )

        # Missing values
        st.subheader("Missing Values Handling")
        missing_analysis = analyze_missing_values(data)
        if missing_analysis['total_missing'] > 0:
            st.warning(f"Found {missing_analysis['total_missing']:,} missing values")
            if missing_analysis['missing_by_column']:
                missing_df = pd.DataFrame([
                    {
                        'Column': col,
                        'Missing Count': info['missing_count'],
                        'Missing %': f"{info['missing_percentage']:.1f}%",
                        'Suggested Strategy': info.get('suggested_strategy', 'mode')
                    }
                    for col, info in missing_analysis['missing_by_column'].items()
                ])
                st.dataframe(missing_df, use_container_width=True)
            handle_missing = st.checkbox("Handle missing values automatically", value=True)
        else:
            handle_missing = False
            st.success("No missing values found!")

        # Run Phase 1
        if st.button("Run Phase 1 Processing", type="primary"):
            with st.spinner("Processing Phase 1..."):
                cleaner = BasicCleaner()
                cleaned_data = cleaner.perform_basic_cleaning(
                    data,
                    remove_duplicates=remove_duplicates,
                    clean_text=clean_text,
                    case_type=case_standardization,
                )

                if handle_missing and missing_analysis['total_missing'] > 0:
                    missing_cleaner = MissingValueCleaner()
                    cleaned_data = missing_cleaner.handle_all_missing_values(cleaned_data)
                    missing_report = missing_cleaner.get_imputation_summary()
                    st.session_state.phase1_reports['missing_values'] = missing_report

                st.session_state.phase1_data = cleaned_data
                st.session_state.phase1_reports['cleaning'] = cleaner.get_cleaning_report()
                st.session_state.current_phase = 2

                st.success("Phase 1 completed successfully!")
                st.rerun()
        return data

    except Exception as e:
        st.error(f"Error in Phase 1: {str(e)}")
        st.code(traceback.format_exc())
        return None
