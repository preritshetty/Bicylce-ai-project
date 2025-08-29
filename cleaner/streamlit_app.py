import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))  # add project root

import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import sys
from datetime import datetime
from io import StringIO, BytesIO
import traceback
import tempfile
import zipfile
from pathlib import Path
_current = Path(__file__).resolve()
_repo_root = _current.parents[1]          # project root
_analyzer_dir = _repo_root / "analyzer"
_modules_dir = _analyzer_dir / "modules"

for p in (str(_repo_root), str(_analyzer_dir), str(_modules_dir)):
    if p not in sys.path:
        sys.path.insert(0, p)
from analyzer.app import render_analysis   # re-use the full analysis UI

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    st.warning("python-dotenv not installed. Please install it with: pip install python-dotenv")
    st.info("Or set OPENAI_API_KEY as a system environment variable.")

# Add utils and phase3 directories to Python path
current_dir = Path(__file__).parent
utils_dir = current_dir / "utils"
phase3_dir = current_dir / "phase3"

# Add both directories to Python path
for directory in [str(utils_dir), str(phase3_dir)]:
    if directory not in sys.path:
        sys.path.insert(0, directory)

# Import your custom modules with proper error handling
try:
    from utils.data_loader import CSVProcessor
    from utils.basic_cleaning import BasicCleaner
    from utils.missing_values import MissingValueCleaner, ImputationStrategy
    from utils.data_sampler import DataSampler
    # Try to import the available LLM interface
    try:
        from utils.llm_interface_langchain import LLMInterfaceLangChain, LLMConfig
    except ImportError:
        from utils.llm_interface_simple import LLMInterface as LLMInterfaceLangChain
        # Create a dummy LLMConfig for compatibility
        class LLMConfig:
            def __init__(self):
                pass
    
    from phase3.flag_mapper import FlagMapper, IssueFlag
    from phase3.code_generator import CodeGenerator
    # Try to import CodeGenConfig if it exists
    try:
        from phase3.code_generator import CodeGenConfig
    except ImportError:
        # Create a dummy CodeGenConfig for compatibility
        class CodeGenConfig:
            def __init__(self):
                pass
    
    # Try to import CodeExecutor
    try:
        from phase3.code_executor import CodeExecutor
    except ImportError:
        try:
            from phase3.executor import CodeExecutor
        except ImportError:
            # Create a dummy CodeExecutor for compatibility
            class CodeExecutor:
                def __init__(self):
                    pass
                def execute_codes(self, *args, **kwargs):
                    return {"success": False, "message": "CodeExecutor not available"}
    
    MODULES_LOADED = True
except ImportError as e:
    st.error(f"Error importing modules: {e}")
    st.error("Attempting fallback imports...")
    
    # Fallback - try importing without path prefixes
    try:
        from data_loader import CSVProcessor
        from basic_cleaning import BasicCleaner
        from missing_values import MissingValueCleaner, ImputationStrategy
        from data_sampler import DataSampler
        try:
            from llm_interface_langchain import LLMInterfaceLangChain, LLMConfig
        except ImportError:
            from llm_interface_simple import LLMInterface as LLMInterfaceLangChain
            class LLMConfig:
                def __init__(self):
                    pass
        
        from flag_mapper import FlagMapper, IssueFlag
        from code_generator import CodeGenerator
        try:
            from code_generator import CodeGenConfig
        except ImportError:
            class CodeGenConfig:
                def __init__(self):
                    pass
        
        try:
            from code_executor import CodeExecutor
        except ImportError:
            try:
                from executor import CodeExecutor
            except ImportError:
                class CodeExecutor:
                    def __init__(self):
                        pass
                    def execute_codes(self, *args, **kwargs):
                        return {"success": False, "message": "CodeExecutor not available"}
        
        MODULES_LOADED = True
        st.success("✅ Fallback imports successful")
    except ImportError as e2:
        st.error(f"Fallback import failed: {e2}")
        st.error("**Available files check:**")
        
        # Show what files are actually available
        if utils_dir.exists():
            st.write(f"**Utils directory ({utils_dir}):**")
            utils_files = list(utils_dir.glob("*.py"))
            for file in utils_files:
                st.write(f"  - {file.name}")
        else:
            st.error(f"Utils directory not found: {utils_dir}")
            
        if phase3_dir.exists():
            st.write(f"**Phase3 directory ({phase3_dir}):**")
            phase3_files = list(phase3_dir.glob("*.py"))
            for file in phase3_files:
                st.write(f"  - {file.name}")
        else:
            st.error(f"Phase3 directory not found: {phase3_dir}")
        
        st.error("Please ensure all required modules are in the utils/ and phase3/ directories")
        MODULES_LOADED = False

# Check if modules loaded successfully before proceeding
if not MODULES_LOADED:
    st.stop()

# --------------------- PAGE CONFIG & STYLES ---------------------
st.set_page_config(
    page_title="Data Quality Pipeline",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constant path where final cleaned CSV is saved (Phase 3)
FINAL_OUT_PATH = "data/final_cleaned.csv"

# Custom CSS for better styling
st.markdown("""
<style>
.phase-header {
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 1rem;
    border-radius: 10px;
    margin: 1rem 0;
    text-align: center;
    font-size: 1.5rem;
    font-weight: bold;
}
.phase-1 { background: linear-gradient(90deg, #ff9a9e 0%, #fecfef 100%); }
.phase-2 { background: linear-gradient(90deg, #a18cd1 0%, #fbc2eb 100%); }
.phase-3 { background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); }
.phase-4 { background: linear-gradient(90deg, #ffecd2 0%, #fcb69f 100%); }

.metric-card {
    background: #f8f9fa;
    padding: 1rem;
    border-radius: 8px;
    border-left: 4px solid #667eea;
    margin: 0.5rem 0;
}
.success-card { border-left-color: #28a745; }
.warning-card { border-left-color: #ffc107; }
.error-card { border-left-color: #dc3545; }
</style>
""", unsafe_allow_html=True)

# --------------------- HELPERS & PIPELINE FUNCS ---------------------
def initialize_session_state():
    """Initialize session state variables"""
    if 'current_phase' not in st.session_state:
        st.session_state.current_phase = 0
    if 'uploaded_files' not in st.session_state:
        st.session_state.uploaded_files = {}
    if 'join_config' not in st.session_state:
        st.session_state.join_config = {}
    if 'joined_data' not in st.session_state:
        st.session_state.joined_data = None
    if 'original_data' not in st.session_state:
        st.session_state.original_data = None
    if 'phase1_data' not in st.session_state:
        st.session_state.phase1_data = None
    if 'phase2_results' not in st.session_state:
        st.session_state.phase2_results = None
    if 'phase3_results' not in st.session_state:
        st.session_state.phase3_results = None
    if 'phase1_reports' not in st.session_state:
        st.session_state.phase1_reports = {}
    if 'processing_complete' not in st.session_state:
        st.session_state.processing_complete = False
    # NEW: in-memory cleaned df for Analyze tab
    if 'cleaned_df' not in st.session_state:
        st.session_state.cleaned_df = None
FINAL_OUT_PATH = "data/final_cleaned.csv"  # keep this single source of truth

def get_final_clean_df() -> pd.DataFrame | None:
    """Return the final cleaned dataset, preferring in-memory, else on-disk."""
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

# New functions for multi-file handling and joining
def load_file_data(uploaded_file):
    """Load data from uploaded file"""
    try:
        processor = CSVProcessor()
        data = processor.load_and_validate(uploaded_file)
        return data
    except Exception as e:
        st.error(f"Error loading {uploaded_file.name}: {str(e)}")
        return None

def multi_file_upload_section():
    """Handle multiple file uploads"""
    st.header("📁 Upload Your Datasets")
    
    uploaded_files = st.file_uploader(
        "Choose CSV files",
        type=['csv'],
        accept_multiple_files=True,
        help="Upload one or more CSV files. If multiple files are uploaded, you'll configure joins next."
    )
    
    if uploaded_files:
        st.subheader("Uploaded Files Overview")
        
        # Process and store uploaded files
        files_data = {}
        for file in uploaded_files:
            if file.name not in st.session_state.uploaded_files:
                data = load_file_data(file)
                if data is not None:
                    files_data[file.name] = data
                    st.session_state.uploaded_files[file.name] = data
        
        # Show file summaries
        for filename, data in st.session_state.uploaded_files.items():
            with st.expander(f"📄 {filename} ({len(data):,} rows, {len(data.columns)} cols)"):
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Rows", f"{len(data):,}")
                with col2:
                    st.metric("Columns", len(data.columns))
                with col3:
                    missing_count = data.isnull().sum().sum()
                    st.metric("Missing Values", f"{missing_count:,}")
                with col4:
                    memory_mb = data.memory_usage(deep=True).sum() / (1024 * 1024)
                    st.metric("Memory", f"{memory_mb:.1f} MB")
                
                st.write("**Column Names:**")
                st.write(", ".join(data.columns.tolist()[:10]) + ("..." if len(data.columns) > 10 else ""))
                
                st.write("**Sample Data:**")
                st.dataframe(data.head(3), use_container_width=True)
        
        # If multiple files, configure joins
        if len(st.session_state.uploaded_files) > 1:
            configure_joins()
        else:
            # Single file - proceed directly
            filename = list(st.session_state.uploaded_files.keys())[0]
            st.session_state.joined_data = st.session_state.uploaded_files[filename]
            st.session_state.original_data = st.session_state.joined_data.copy()
            
            if st.button("Proceed with Single File", type="primary"):
                st.session_state.current_phase = 1
                st.rerun()

def configure_joins():
    """Configure joins between multiple files"""
    st.markdown("---")
    st.subheader("🔗 Configure Table Joins")
    
    file_names = list(st.session_state.uploaded_files.keys())
    
    # Select primary table
    primary_table = st.selectbox(
        "Select Primary Table (main dataset):",
        file_names,
        help="This will be the base table that other tables join to"
    )
    
    # Configure joins for other tables
    join_configs = []
    other_tables = [f for f in file_names if f != primary_table]
    
    if other_tables:
        st.write("**Configure Joins for Additional Tables:**")
        
        for i, table_name in enumerate(other_tables):
            st.write(f"**Join Configuration for: {table_name}**")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                join_type = st.selectbox(
                    "Join Type:",
                    ["left", "inner", "right", "outer"],
                    index=0,
                    key=f"join_type_{i}",
                    help="Left: keep all primary table rows. Inner: only matching rows."
                )
            
            with col2:
                primary_columns = st.session_state.uploaded_files[primary_table].columns.tolist()
                primary_key = st.selectbox(
                    "Primary Table Key:",
                    primary_columns,
                    key=f"primary_key_{i}"
                )
            
            with col3:
                secondary_columns = st.session_state.uploaded_files[table_name].columns.tolist()
                secondary_key = st.selectbox(
                    "Secondary Table Key:",
                    secondary_columns,
                    key=f"secondary_key_{i}"
                )
            
            # Column selection
            st.write(f"**Select Columns to Include from {table_name}:**")
            available_columns = [col for col in secondary_columns if col != secondary_key]
            
            if available_columns:
                selected_columns = st.multiselect(
                    f"Columns from {table_name}:",
                    available_columns,
                    default=available_columns[:5] if len(available_columns) > 5 else available_columns,
                    key=f"columns_{i}"
                )
            else:
                selected_columns = []
                st.warning(f"No additional columns available from {table_name}")
            
            join_configs.append({
                'secondary_table': table_name,
                'join_type': join_type,
                'primary_key': primary_key,
                'secondary_key': secondary_key,
                'selected_columns': selected_columns
            })
        
        # Store join configuration
        st.session_state.join_config = {
            'primary_table': primary_table,
            'joins': join_configs
        }
        
        # Preview join
        if st.button("Preview Join Result", type="secondary"):
            preview_join()
        
        # Execute join
        if st.button("Execute Joins & Proceed", type="primary"):
            execute_joins()

def preview_join():
    """Show a preview of the join result with enhanced error handling"""
    try:
        joined_data = perform_joins(sample_size=100)  # Preview with smaller sample
        
        st.subheader("Join Preview (First 100 rows)")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Rows", f"{len(joined_data):,}")
        with col2:
            st.metric("Columns", len(joined_data.columns))
        with col3:
            missing_count = joined_data.isnull().sum().sum()
            st.metric("Missing Values", f"{missing_count:,}")
        with col4:
            memory_mb = joined_data.memory_usage(deep=True).sum() / (1024 * 1024)
            st.metric("Memory", f"{memory_mb:.1f} MB")
        
        st.dataframe(joined_data.head(10), use_container_width=True)
        
        # Show data type information
        st.subheader("Column Data Types After Join")
        dtype_info = pd.DataFrame({
            'Column': joined_data.columns,
            'Data Type': [str(dtype) for dtype in joined_data.dtypes],
            'Non-Null Count': [joined_data[col].count() for col in joined_data.columns]
        })
        st.dataframe(dtype_info, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error previewing join: {str(e)}")
        st.code(traceback.format_exc())

def perform_joins(sample_size=None):
    """Perform the actual joins with data type handling"""
    config = st.session_state.join_config
    primary_table = config['primary_table']
    
    # Start with primary table
    result_data = st.session_state.uploaded_files[primary_table].copy()
    
    if sample_size:
        result_data = result_data.head(sample_size)
    
    # Perform joins sequentially
    for join_config in config['joins']:
        secondary_table = join_config['secondary_table']
        secondary_data = st.session_state.uploaded_files[secondary_table].copy()
        
        # Select only needed columns from secondary table
        columns_to_include = [join_config['secondary_key']] + join_config['selected_columns']
        secondary_data = secondary_data[columns_to_include]
        
        # Handle data type mismatches for join keys
        primary_key = join_config['primary_key']
        secondary_key = join_config['secondary_key']
        
        # Get the data types of join columns
        primary_dtype = result_data[primary_key].dtype
        secondary_dtype = secondary_data[secondary_key].dtype
        
        # Convert data types to match if they're different
        if primary_dtype != secondary_dtype:
            st.warning(f"Data type mismatch detected for join key '{primary_key}' vs '{secondary_key}': {primary_dtype} vs {secondary_dtype}")
            
            try:
                # Try to convert both to string first (most compatible)
                result_data[primary_key] = result_data[primary_key].astype(str)
                secondary_data[secondary_key] = secondary_data[secondary_key].astype(str)
                st.info(f"Converted both join keys to string type for compatibility")
                
            except Exception as e1:
                try:
                    # If string conversion fails, try numeric conversion
                    if 'int' in str(primary_dtype).lower() or 'float' in str(primary_dtype).lower():
                        # Primary is numeric, convert secondary to numeric
                        secondary_data[secondary_key] = pd.to_numeric(secondary_data[secondary_key], errors='coerce')
                        st.info(f"Converted secondary key to numeric type")
                    else:
                        # Primary is not numeric, convert primary to match secondary
                        result_data[primary_key] = pd.to_numeric(result_data[primary_key], errors='coerce')
                        st.info(f"Converted primary key to numeric type")
                        
                except Exception as e2:
                    st.error(f"Could not resolve data type mismatch: {str(e2)}")
                    st.error("Attempting string conversion as fallback...")
                    # Final fallback - force string conversion
                    result_data[primary_key] = result_data[primary_key].fillna('').astype(str)
                    secondary_data[secondary_key] = secondary_data[secondary_key].fillna('').astype(str)
        
        # Handle duplicate column names
        overlapping_columns = set(result_data.columns) & set(secondary_data.columns)
        overlapping_columns.discard(primary_key)  # Keep join key
        overlapping_columns.discard(secondary_key)
        
        if overlapping_columns:
            # Add suffix to secondary table columns
            suffix_mapping = {col: f"{col}_{secondary_table.split('.')[0]}" for col in overlapping_columns}
            secondary_data = secondary_data.rename(columns=suffix_mapping)
        
        # Show join preview info
        st.write(f"Joining {len(result_data)} rows with {len(secondary_data)} rows on {primary_key}={secondary_key}")
        
        # Check for potential join issues
        primary_unique = result_data[primary_key].nunique()
        secondary_unique = secondary_data[secondary_key].nunique()
        primary_nulls = result_data[primary_key].isnull().sum()
        secondary_nulls = secondary_data[secondary_key].isnull().sum()
        
        if primary_nulls > 0 or secondary_nulls > 0:
            st.warning(f"Found null values in join keys: Primary={primary_nulls}, Secondary={secondary_nulls}")
        
        # Perform the join
        try:
            result_data = pd.merge(
                result_data,
                secondary_data,
                left_on=primary_key,
                right_on=secondary_key,
                how=join_config['join_type'],
                suffixes=('', f'_{secondary_table.split(".")[0]}')
            )
            
            st.success(f"Join successful: {len(result_data)} rows after joining {secondary_table}")
            
        except Exception as e:
            st.error(f"Join failed even after data type conversion: {str(e)}")
            
            # Show diagnostic information
            st.write("Diagnostic Information:")
            st.write(f"Primary key '{primary_key}' sample values:", result_data[primary_key].head().tolist())
            st.write(f"Secondary key '{secondary_key}' sample values:", secondary_data[secondary_key].head().tolist())
            st.write(f"Primary key data type:", result_data[primary_key].dtype)
            st.write(f"Secondary key data type:", secondary_data[secondary_key].dtype)
            
            # Re-raise the exception to stop processing
            raise e
        
        # Remove duplicate join key if different names were used
        if primary_key != secondary_key:
            if secondary_key in result_data.columns:
                result_data = result_data.drop(columns=[secondary_key])
    
    return result_data

def execute_joins():
    """Execute joins and proceed to Phase 1 with enhanced error handling"""
    try:
        with st.spinner("Executing joins..."):
            joined_data = perform_joins()
            
            st.session_state.joined_data = joined_data
            st.session_state.original_data = joined_data.copy()
            st.session_state.current_phase = 1
            
            st.success(f"Successfully joined tables! Final dataset: {len(joined_data):,} rows, {len(joined_data.columns)} columns")
            
            # Show final data type summary
            with st.expander("Final Dataset Summary"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Data Types:**")
                    dtype_counts = joined_data.dtypes.value_counts()
                    for dtype, count in dtype_counts.items():
                        st.write(f"- {dtype}: {count} columns")
                
                with col2:
                    st.write("**Sample Data:**")
                    st.dataframe(joined_data.head(3))
            
            st.rerun()
            
    except Exception as e:
        st.error(f"Error executing joins: {str(e)}")
        st.code(traceback.format_exc())
        
        # Provide troubleshooting suggestions
        st.subheader("Troubleshooting Suggestions:")
        st.write("1. Check that join keys contain matching values")
        st.write("2. Ensure join keys don't have too many null values")
        st.write("3. Consider using string keys if numeric conversion fails")
        st.write("4. Check for special characters or formatting issues in key columns")

def display_join_summary():
    """Display summary of joins performed"""
    if st.session_state.join_config and len(st.session_state.uploaded_files) > 1:
        st.subheader("Join Summary")
        
        config = st.session_state.join_config
        st.write(f"**Primary Table:** {config['primary_table']}")
        
        for i, join in enumerate(config['joins'], 1):
            st.write(f"**Join {i}:** {join['secondary_table']}")
            st.write(f"- Type: {join['join_type'].title()} Join")
            st.write(f"- Keys: {join['primary_key']} = {join['secondary_key']}")
            st.write(f"- Added Columns: {', '.join(join['selected_columns']) if join['selected_columns'] else 'None'}")

# Modified phase1_processing function to work with joined data
def modified_phase1_processing():
    """Phase 1: Basic Data Cleaning (modified for joined data)"""
    st.markdown('<div class="phase-header phase-1">Phase 1: Basic Data Cleaning</div>', unsafe_allow_html=True)
    
    if st.session_state.original_data is None:
        st.error("No data available for processing.")
        return None
    
    try:
        data = st.session_state.original_data
        
        st.success(f"Data ready for processing: {len(data)} rows, {len(data.columns)} columns")
        
        # Show join summary if applicable
        display_join_summary()
        
        # Display data summary
        display_data_summary(data, "Dataset for Cleaning")
        
        # Basic cleaning options
        st.subheader("Cleaning Options")
        
        col1, col2 = st.columns(2)
        
        with col1:
            remove_duplicates = st.checkbox("Remove duplicate rows", value=True)
            clean_text = st.checkbox("Clean text columns (trim whitespace)", value=True)
        
        with col2:
            case_standardization = st.selectbox(
                "Text case standardization",
                ["none", "title", "upper", "lower"],
                index=1
            )
        
        # Missing values handling
        st.subheader("Missing Values Handling")
        
        missing_analysis = analyze_missing_values(data)
        
        if missing_analysis['total_missing'] > 0:
            st.warning(f"Found {missing_analysis['total_missing']:,} missing values")
            
            # Show missing values by column
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
        
        # Process Phase 1
        if st.button("Run Phase 1 Processing", type="primary"):
            with st.spinner("Processing Phase 1..."):
                # Basic cleaning
                cleaner = BasicCleaner()
                cleaned_data = cleaner.perform_basic_cleaning(
                    data,
                    remove_duplicates=remove_duplicates,
                    clean_text=clean_text,
                    case_type=case_standardization
                )
                
                # Handle missing values
                if handle_missing and missing_analysis['total_missing'] > 0:
                    missing_cleaner = MissingValueCleaner()
                    cleaned_data = missing_cleaner.handle_all_missing_values(cleaned_data)
                    missing_report = missing_cleaner.get_imputation_summary()
                    st.session_state.phase1_reports['missing_values'] = missing_report
                
                st.session_state.phase1_data = cleaned_data
                st.session_state.phase1_reports['cleaning'] = cleaner.get_cleaning_report()
                st.session_state.current_phase = 2  # Move to Phase 2
                
                st.success("Phase 1 completed successfully!")
                st.rerun()
        
        return data
        
    except Exception as e:
        st.error(f"Error in Phase 1: {str(e)}")
        st.code(traceback.format_exc())
        return None

def create_download_link(data, filename, file_type="csv"):
    """Create download link for data"""
    if file_type == "csv":
        csv = data.to_csv(index=False)
        b64 = StringIO(csv).getvalue().encode()
    elif file_type == "json":
        json_str = json.dumps(data, indent=2)
        b64 = json_str.encode()
    else:
        b64 = data
    
    return st.download_button(
        label=f"Download {filename}",
        data=b64,
        file_name=filename,
        mime="text/csv" if file_type == "csv" else "application/json" if file_type == "json" else "application/octet-stream"
    )

def display_data_summary(data, title):
    """Display data summary in a nice format"""
    st.subheader(title)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Rows", f"{len(data):,}")
    
    with col2:
        st.metric("Columns", len(data.columns))
    
    with col3:
        missing_count = data.isnull().sum().sum()
        st.metric("Missing Values", f"{missing_count:,}")
    
    with col4:
        memory_mb = data.memory_usage(deep=True).sum() / (1024 * 1024)
        st.metric("Memory Usage", f"{memory_mb:.1f} MB")
    
    # Show data preview
    st.dataframe(data.head(10), use_container_width=True)

def phase1_processing(uploaded_file):
    """Phase 1: Basic Data Cleaning"""
    st.markdown('<div class="phase-header phase-1">Phase 1: Basic Data Cleaning</div>', unsafe_allow_html=True)
    
    try:
        # Load data
        processor = CSVProcessor()
        data = processor.load_and_validate(uploaded_file)
        st.session_state.original_data = data.copy()
        
        st.success(f"Data loaded successfully: {len(data)} rows, {len(data.columns)} columns")
        
        # Display original data summary
        display_data_summary(data, "Original Data")
        
        # Basic cleaning options
        st.subheader("Cleaning Options")
        
        col1, col2 = st.columns(2)
        
        with col1:
            remove_duplicates = st.checkbox("Remove duplicate rows", value=True)
            clean_text = st.checkbox("Clean text columns (trim whitespace)", value=True)
        
        with col2:
            case_standardization = st.selectbox(
                "Text case standardization",
                ["none", "title", "upper", "lower"],
                index=1
            )
        
        # Missing values handling
        st.subheader("Missing Values Handling")
        
        missing_analysis = analyze_missing_values(data)
        
        if missing_analysis['total_missing'] > 0:
            st.warning(f"Found {missing_analysis['total_missing']:,} missing values")
            
            # Show missing values by column
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
        
        # Process Phase 1
        if st.button("Run Phase 1 Processing", type="primary"):
            with st.spinner("Processing Phase 1..."):
                # Basic cleaning
                cleaner = BasicCleaner()
                cleaned_data = cleaner.perform_basic_cleaning(
                    data,
                    remove_duplicates=remove_duplicates,
                    clean_text=clean_text,
                    case_type=case_standardization
                )
                
                # Handle missing values
                if handle_missing and missing_analysis['total_missing'] > 0:
                    missing_cleaner = MissingValueCleaner()
                    cleaned_data = missing_cleaner.handle_all_missing_values(cleaned_data)
                    missing_report = missing_cleaner.get_imputation_summary()
                    st.session_state.phase1_reports['missing_values'] = missing_report
                
                st.session_state.phase1_data = cleaned_data
                st.session_state.phase1_reports['cleaning'] = cleaner.get_cleaning_report()
                st.session_state.current_phase = 1
                
                st.success("Phase 1 completed successfully!")
                st.rerun()
        
        return data
        
    except Exception as e:
        st.error(f"Error in Phase 1: {str(e)}")
        st.code(traceback.format_exc())
        return None

def analyze_missing_values(data):
    """Analyze missing values in the data"""
    cleaner = MissingValueCleaner()
    return cleaner.analyze_missing_patterns(data)

def phase2_processing(data):
    """Phase 2: LLM-powered Issue Detection"""
    st.markdown('<div class="phase-header phase-2">Phase 2: AI-Powered Issue Detection</div>', unsafe_allow_html=True)
    
    # Display cleaned data summary
    display_data_summary(data, "Phase 1 Cleaned Data")
    
    # Show Phase 1 reports
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
    
    # LLM Configuration
    st.subheader("AI Analysis Configuration")
    
    # Check for API key
    api_key_present = bool(os.getenv('OPENAI_API_KEY'))
    
    if not api_key_present:
        st.error("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
        st.info("You can add it to a .env file or set it as an environment variable.")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        model_choice = st.selectbox(
            "AI Model",
            ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"],
            index=0
        )
        temperature = st.slider("Temperature", 0.0, 1.0, 0.1, 0.1)
    
    with col2:
        max_tokens = st.number_input("Max Tokens", 500, 4000, 1500, 100)
        sample_size = st.number_input("Sample Size for Analysis", 50, 500, 150, 25)
    
    # Data sampling
    st.subheader("Data Sampling for Analysis")
    
    sampler = DataSampler(max_rows=sample_size)
    sample_info = sampler.create_sample(data)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Original Rows", f"{sample_info['original_rows']:,}")
    
    with col2:
        st.metric("Sample Rows", f"{sample_info['sampled_rows']:,}")
    
    with col3:
        st.metric("Sampling Ratio", f"{sample_info['sampling_ratio']:.1%}")
    
    st.info(f"Sampling Strategy: {sample_info['sampling_strategy'].replace('_', ' ').title()}")
    
    # Show sample preview
    with st.expander("View Sample Data"):
        st.dataframe(sample_info['sampled_data'].head(10), use_container_width=True)
    
    # Run Phase 2 Analysis
    if st.button("Run Phase 2 AI Analysis", type="primary"):
        with st.spinner("Running AI analysis... This may take a few minutes."):
            try:
                # Setup LLM interface
                config = LLMConfig(
                    model_name=model_choice,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                
                llm_interface = LLMInterfaceLangChain(config)
                
                # Extract column info
                column_info = sampler.extract_column_info(sample_info['sampled_data'])
                
                # Run analysis
                analysis_results = llm_interface.analyze_data_quality(
                    sample_info['sampled_data'],
                    column_info
                )
                
                st.session_state.phase2_results = analysis_results
                st.session_state.current_phase = 3
                
                st.success("Phase 2 analysis completed!")
                st.rerun()
                
            except Exception as e:
                st.error(f"Error in Phase 2: {str(e)}")
                st.code(traceback.format_exc())

def display_phase2_results():
    """Display Phase 2 analysis results"""
    results = st.session_state.phase2_results
    
    # Overall metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Issues Found", results['total_issues'])
    
    with col2:
        st.metric("Quality Score", f"{results.get('data_quality_score', 'N/A')}/100")
    
    with col3:
        severity_counts = {}
        for issue in results['issues']:
            severity = issue['severity'].strip().lower()
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        high_severity = severity_counts.get('high', 0)
        st.metric("High Severity", high_severity)
    
    with col4:
        categories = set(issue['category'] for issue in results['issues'])
        st.metric("Categories", len(categories))
    
    # Issues breakdown
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
    
    # Column renaming suggestions
    if results.get('column_renaming'):
        st.subheader("Column Renaming Suggestions")
        
        renaming_df = pd.DataFrame([
            {'Current Name': old_name, 'Suggested Name': new_name}
            for old_name, new_name in results['column_renaming'].items()
            if old_name != new_name and not new_name.startswith('Use clear')
        ])
        
        if not renaming_df.empty:
            st.dataframe(renaming_df, use_container_width=True)
    
    # Recommendations
    if results.get('recommendations'):
        st.subheader("Recommendations")
        for i, rec in enumerate(results['recommendations'], 1):
            st.write(f"{i}. {rec}")

def update_issues_with_renamed_columns(issues, column_mappings):
    """Update issue column references to match renamed columns"""
    updated_issues = []
    
    for issue in issues:
        updated_issue = issue.copy()
        
        # Update affected_columns with new names
        updated_columns = []
        for col in issue['affected_columns']:
            if col in column_mappings:
                updated_columns.append(column_mappings[col])
            else:
                updated_columns.append(col)
        
        updated_issue['affected_columns'] = updated_columns
        
        # Update description to reference new column names
        description = issue['description']
        for old_name, new_name in column_mappings.items():
            description = description.replace(old_name, new_name)
        updated_issue['description'] = description
        
        # Update fix_approach to reference new column names
        fix_approach = issue['fix_approach']
        for old_name, new_name in column_mappings.items():
            fix_approach = fix_approach.replace(old_name, new_name)
        updated_issue['fix_approach'] = fix_approach
        
        updated_issues.append(updated_issue)
    
    return updated_issues

def phase3_processing():
    """Phase 3: Code Generation and Execution with Column Renaming"""
    st.markdown('<div class="phase-header phase-3">Phase 3: Automated Code Generation & Column Renaming</div>', unsafe_allow_html=True)
    
    if not st.session_state.phase2_results:
        st.error("Phase 2 results not available. Please complete Phase 2 first.")
        return
    
    # Show Phase 2 summary
    results = st.session_state.phase2_results
    st.info(f"Found {results['total_issues']} issues to address and {len([k for k, v in results.get('column_renaming', {}).items() if k != v and not v.startswith('Use clear')])} columns to rename")
    
    # Column Renaming Section
    st.subheader("Column Renaming Configuration")
    
    # Show suggested renamings
    column_renaming = results.get('column_renaming', {})
    valid_renamings = {
        old_name: new_name 
        for old_name, new_name in column_renaming.items() 
        if old_name != new_name and not new_name.startswith('Use clear') and len(new_name.strip()) > 0
    }
    
    if valid_renamings:
        st.write("**Suggested Column Renamings:**")
        
        # Create checkboxes for each renaming suggestion
        selected_renamings = {}
        
        for old_name, new_name in valid_renamings.items():
            col1, col2, col3 = st.columns([1, 2, 2])
            
            with col1:
                apply_rename = st.checkbox(f"Apply", key=f"rename_{old_name}", value=True)
            
            with col2:
                st.write(f"**{old_name}**")
            
            with col3:
                # Allow user to edit the suggested name
                final_name = st.text_input(
                    "New name:", 
                    value=new_name, 
                    key=f"newname_{old_name}",
                    label_visibility="collapsed"
                )
            
            if apply_rename and final_name.strip():
                selected_renamings[old_name] = final_name.strip()
        
        st.session_state.selected_renamings = selected_renamings
    else:
        st.info("No column renaming suggestions available.")
        st.session_state.selected_renamings = {}
    
    # Configuration
    st.subheader("Code Generation Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        code_model = st.selectbox(
            "Code Generation Model",
            ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"],
            index=0
        )
        code_temperature = st.slider("Code Generation Temperature", 0.0, 0.5, 0.1, 0.05)
    
    with col2:
        max_code_tokens = st.number_input("Max Tokens per Code", 500, 2000, 1000, 100)
        execution_timeout = st.number_input("Execution Timeout (seconds)", 10, 120, 30, 10)
    
    # Generate flags mapping and apply column renaming
    if st.button("Apply Column Renaming & Generate Detection Codes", type="primary"):
        with st.spinner("Applying column renaming and generating detection codes..."):
            try:
                # Step 0: Apply column renaming to Phase 1 data
                working_data = st.session_state.phase1_data.copy() if st.session_state.phase1_data is not None else st.session_state.original_data.copy()
                
                if hasattr(st.session_state, 'selected_renamings') and st.session_state.selected_renamings:
                    st.write("Step 0: Applying column renaming...")
                    
                    # Apply the renamings
                    working_data = working_data.rename(columns=st.session_state.selected_renamings)
                    
                    # Update column names in issues to match renamed columns
                    updated_issues = update_issues_with_renamed_columns(results['issues'], st.session_state.selected_renamings)
                    
                    st.success(f"Applied {len(st.session_state.selected_renamings)} column renamings")
                    
                    # Store the renamed data
                    st.session_state.phase3_renamed_data = working_data
                else:
                    updated_issues = results['issues']
                    st.session_state.phase3_renamed_data = working_data
                
                # Step 1: Create flag mapping with updated column names
                st.write("Step 1: Creating flag mapping...")
                mapper = FlagMapper()
                flag_mapping = mapper.create_flag_mapping(updated_issues)
                
                # Save flag mapping
                flag_mapping_file = mapper.save_mapping_to_file()
                st.success(f"Flag mapping created with {len(flag_mapping)} flags")
                
                # Step 2: Generate detection codes
                st.write("Step 2: Generating detection codes...")
                
                config = CodeGenConfig(
                    model_name=code_model,
                    temperature=code_temperature,
                    max_tokens=max_code_tokens
                )
                
                code_generator = CodeGenerator(config)
                
                # Prepare sample data info with renamed columns
                sample_data = working_data.head(10)
                sample_info = f"Sample data structure: {len(sample_data)} rows, columns: {list(sample_data.columns)}"
                
                # Generate codes
                generation_results = code_generator.generate_all_detection_codes(flag_mapping, sample_info)
                
                # Save codes
                codes_file = code_generator.save_detection_codes(generation_results, "phase outputs")
                
                st.success(f"Generated {generation_results.total_codes} detection codes")
                
                # Step 3: Execute codes on renamed data
                st.write("Step 3: Executing detection codes on renamed data...")
                
                # Save renamed data temporarily for execution
                temp_data_file = "phase outputs/phase3_renamed_data.csv"
                os.makedirs("phase outputs", exist_ok=True)
                
                working_data.to_csv(temp_data_file, index=False)
                
                # Execute detection codes
                executor = CodeExecutor(temp_data_file)
                execution_results = executor.run_complete_pipeline(codes_file)
                
                st.session_state.phase3_results = {
                    'flag_mapping_file': flag_mapping_file,
                    'codes_file': codes_file,
                    'execution_results': execution_results,
                    'output_files': execution_results.get('output_files', {}),
                    'flagged_data_file': execution_results['output_files']['flagged_data'],
                    'execution_report_file': execution_results['output_files']['execution_report'],
                    'renamed_data_file': temp_data_file,
                    'applied_renamings': st.session_state.selected_renamings if hasattr(st.session_state, 'selected_renamings') else {}
                }
                Path("data").mkdir(exist_ok=True)

                # Save final cleaned dataset (flagged_data is the Phase 3 output)
                final_out = Path(FINAL_OUT_PATH)
                flagged_data = pd.read_csv(st.session_state.phase3_results['flagged_data_file'])
                flagged_data.to_csv(final_out, index=False)

                # >>> IMPORTANT: make the cleaned frame available to the Analyze tab immediately
                st.session_state["cleaned_df"] = flagged_data.copy()

                # Save flag mapping (if available)
                try:
                    with open(st.session_state.phase3_results['flag_mapping_file'], "r") as f:
                        fmap = json.load(f)
                    with open("data/flag_mapping.json", "w") as f2:
                        json.dump(fmap, f2, indent=2)
                except Exception as e:
                    st.warning(f"Could not save flag mapping: {e}")

                st.success(f"✅ Final cleaned dataset saved for Analyzer → {final_out}")
                st.session_state.current_phase = 4
                st.success("Phase 3 completed successfully with column renaming!")
                st.rerun()
                
            except Exception as e:
                st.error(f"Error in Phase 3: {str(e)}")
                st.code(traceback.format_exc())

def display_phase3_results():
    """Display Phase 3 execution results with column renaming info"""
    results = st.session_state.phase3_results
    execution_results = results['execution_results']
    summary = execution_results['execution_summary']
    
    # Show column renaming results if applied
    applied_renamings = results.get('applied_renamings', {})
    if applied_renamings:
        st.subheader("Applied Column Renamings")
        renaming_df = pd.DataFrame([
            {'Original Column': old_name, 'New Column': new_name}
            for old_name, new_name in applied_renamings.items()
        ])
        st.dataframe(renaming_df, use_container_width=True)
        st.success(f"Successfully renamed {len(applied_renamings)} columns")
    
    # Execution metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Rows", f"{summary['total_rows']:,}")
    
    with col2:
        st.metric("Flagged Rows", f"{summary['flagged_rows']:,}")
    
    with col3:
        st.metric("Clean Rows", f"{summary['clean_rows']:,}")
    
    with col4:
        st.metric("Flag Coverage", f"{summary['flagged_percentage']}%")
    
    # Execution details
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Successful Codes", f"{summary['successful_codes']}/{summary['successful_codes'] + summary['failed_codes']}")
    
    with col2:
        st.metric("Success Rate", f"{summary['success_rate']}%")
    
    with col3:
        st.metric("Execution Time", f"{summary['execution_time_seconds']}s")
    
    # Individual code results
    st.subheader("Detection Code Results")
    
    individual_results = execution_results['individual_results']
    results_df = pd.DataFrame([
        {
            'Flag': result['flag_value'],
            'Status': '✅ Success' if result['success'] else '❌ Failed',
            'Rows Detected': result['rows_detected'],
            'Description': result['explanation'][:80] + "..." if len(result['explanation']) > 80 else result['explanation'],
            'Error': result.get('error', '')[:50] + "..." if result.get('error', '') and len(result.get('error', '')) > 50 else result.get('error', '')
        }
        for result in individual_results
    ])
    
    st.dataframe(results_df, use_container_width=True)
    
    # Flag breakdown
    if execution_results.get('flag_breakdown', {}).get('flag_combinations'):
        st.subheader("Flag Combinations Found")
        
        combinations = execution_results['flag_breakdown']['flag_combinations']
        combinations_df = pd.DataFrame([
            {
                'Flag Status': combo['flag_status'],
                'Individual Flags': '+'.join(map(str, combo['individual_flags'])),
                'Description': combo['flag_description'],
                'Row Count': combo['row_count'],
                'Binary': combo['binary_representation']
            }
            for combo in combinations
        ])
        
        st.dataframe(combinations_df, use_container_width=True)

def create_final_downloads():
    """Create download links for all phase outputs including renamed data"""
    st.subheader("Download Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write("**Original Data**")
        if st.session_state.original_data is not None:
            create_download_link(
                st.session_state.original_data,
                "original_data.csv"
            )
    
    with col2:
        st.write("**Phase 1 Cleaned Data**")
        if st.session_state.phase1_data is not None:
            create_download_link(
                st.session_state.phase1_data,
                "phase1_cleaned_data.csv"
            )
    
    with col3:
        st.write("**Phase 2 Analysis**")
        if st.session_state.phase2_results:
            create_download_link(
                st.session_state.phase2_results,
                "phase2_analysis.json",
                "json"
            )
    
    if st.session_state.phase3_results:
        st.write("**Phase 3 Results**")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            # Renamed data (if column renaming was applied)
            if hasattr(st.session_state, 'phase3_renamed_data'):
                create_download_link(
                    st.session_state.phase3_renamed_data,
                    "phase3_renamed_data.csv"
                )
        
        with col2:
            # Flagged data
            if os.path.exists(st.session_state.phase3_results['flagged_data_file']):
                flagged_data = pd.read_csv(st.session_state.phase3_results['flagged_data_file'])
                create_download_link(
                    flagged_data,
                    "flagged_data.csv"
                )
        
        with col3:
            # Execution report
            if os.path.exists(st.session_state.phase3_results['execution_report_file']):
                with open(st.session_state.phase3_results['execution_report_file'], 'r') as f:
                    execution_report = json.load(f)
                create_download_link(
                    execution_report,
                    "execution_report.json",
                    "json"
                )
        
        with col4:
            # Flag mapping
            if os.path.exists(st.session_state.phase3_results['flag_mapping_file']):
                with open(st.session_state.phase3_results['flag_mapping_file'], 'r') as f:
                    flag_mapping = json.load(f)
                create_download_link(
                    flag_mapping,
                    "flag_mapping.json",
                    "json"
                )
        
        # Column renaming report
        if st.session_state.phase3_results.get('applied_renamings'):
            st.write("**Column Renaming Report**")
            renaming_report = {
                'applied_renamings': st.session_state.phase3_results['applied_renamings'],
                'timestamp': datetime.now().isoformat(),
                'total_columns_renamed': len(st.session_state.phase3_results['applied_renamings'])
            }
            create_download_link(
                renaming_report,
                "column_renaming_report.json",
                "json"
            )
    
    # Create complete results package
    if st.button("Create Complete Results Package"):
        create_results_package_with_renaming()

def create_results_package_with_renaming():
    """Create a ZIP file with all results including renamed data"""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as tmp_file:
            with zipfile.ZipFile(tmp_file.name, 'w') as zipf:
                # Add original data
                if st.session_state.original_data is not None:
                    csv_buffer = StringIO()
                    st.session_state.original_data.to_csv(csv_buffer, index=False)
                    zipf.writestr("original_data.csv", csv_buffer.getvalue())
                
                # Add Phase 1 data
                if st.session_state.phase1_data is not None:
                    csv_buffer = StringIO()
                    st.session_state.phase1_data.to_csv(csv_buffer, index=False)
                    zipf.writestr("phase1_cleaned_data.csv", csv_buffer.getvalue())
                
                # Add Phase 2 results
                if st.session_state.phase2_results:
                    zipf.writestr(
                        "phase2_analysis.json",
                        json.dumps(st.session_state.phase2_results, indent=2)
                    )
                
                # Add Phase 3 renamed data (if available)
                if hasattr(st.session_state, 'phase3_renamed_data'):
                    csv_buffer = StringIO()
                    st.session_state.phase3_renamed_data.to_csv(csv_buffer, index=False)
                    zipf.writestr("phase3_renamed_data.csv", csv_buffer.getvalue())
                
                # Add Phase 3 results
                if st.session_state.phase3_results:
                    if os.path.exists(st.session_state.phase3_results['flagged_data_file']):
                        zipf.write(
                            st.session_state.phase3_results['flagged_data_file'],
                            "phase3_flagged_data.csv"
                        )
                    
                    if os.path.exists(st.session_state.phase3_results['execution_report_file']):
                        zipf.write(
                            st.session_state.phase3_results['execution_report_file'],
                            "phase3_execution_report.json"
                        )
                    
                    if os.path.exists(st.session_state.phase3_results['flag_mapping_file']):
                        zipf.write(
                            st.session_state.phase3_results['flag_mapping_file'],
                            "phase3_flag_mapping.json"
                        )
                    
                    if st.session_state.phase3_results.get('applied_renamings'):
                        renaming_report = {
                            'applied_renamings': st.session_state.phase3_results['applied_renamings'],
                            'timestamp': datetime.now().isoformat(),
                            'total_columns_renamed': len(st.session_state.phase3_results['applied_renamings'])
                        }
                        zipf.writestr(
                            "column_renaming_report.json",
                            json.dumps(renaming_report, indent=2)
                        )
            
            # Read the zip file for download
            with open(tmp_file.name, 'rb') as f:
                zip_data = f.read()
            
            st.download_button(
                label="Download Complete Results Package",
                data=zip_data,
                file_name=f"data_quality_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                mime="application/zip"
            )
        
        os.unlink(tmp_file.name)
    except Exception as e:
        st.error(f"Error creating results package: {str(e)}")

def enhanced_data_comparison_tab():
    """Enhanced data comparison tab that includes renamed data"""
    # Data comparison between phases
    st.subheader("Data Transformation Summary")

    comparison_data = []

    if st.session_state.original_data is not None:
        original = st.session_state.original_data
        comparison_data.append({
            "Phase": "Original",
            "Rows": len(original),
            "Columns": len(original.columns),
            "Missing Values": original.isnull().sum().sum(),
            "Memory (MB)": f"{original.memory_usage(deep=True).sum() / (1024*1024):.2f}"
        })

    if st.session_state.phase1_data is not None:
        phase1 = st.session_state.phase1_data
        comparison_data.append({
            "Phase": "Phase 1 (Cleaned)",
            "Rows": len(phase1),
            "Columns": len(phase1.columns),
            "Missing Values": phase1.isnull().sum().sum(),
            "Memory (MB)": f"{phase1.memory_usage(deep=True).sum() / (1024*1024):.2f}"
        })

    if hasattr(st.session_state, 'phase3_renamed_data'):
        renamed_data = st.session_state.phase3_renamed_data
        comparison_data.append({
            "Phase": "Phase 3 (Renamed)",
            "Rows": len(renamed_data),
            "Columns": len(renamed_data.columns),
            "Missing Values": renamed_data.isnull().sum().sum(),
            "Memory (MB)": f"{renamed_data.memory_usage(deep=True).sum() / (1024*1024):.2f}"
        })

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
                "Flagged Rows": flagged_rows
            })
        except Exception:
            pass

    if comparison_data:
        st.dataframe(pd.DataFrame(comparison_data), use_container_width=True)

    # Show sample data comparisons
    cols = st.columns(3)
    with cols[0]:
        st.write("**Original Data Sample**")
        if st.session_state.original_data is not None:
            st.dataframe(st.session_state.original_data.head(), use_container_width=True)
    with cols[1]:
        st.write("**Phase 1 Cleaned Data Sample**")
        if st.session_state.phase1_data is not None:
            st.dataframe(st.session_state.phase1_data.head(), use_container_width=True)
    with cols[2]:
        st.write("**Phase 3 Renamed Data Sample**")
        if hasattr(st.session_state, 'phase3_renamed_data'):
            st.dataframe(st.session_state.phase3_renamed_data.head(), use_container_width=True)
        else:
            st.info("No column renaming applied")
def render_cleaner_tab():
    """Render the original cleaner state-machine UI inside the Clean & Export tab."""
    # Sidebar progress
    st.sidebar.title("Pipeline Progress")
    phases = ["Upload & Join Data", "Phase 1: Cleaning", "Phase 2: AI Analysis", "Phase 3: Code Generation", "Results"]
    for i, phase in enumerate(phases):
        if i <= st.session_state.current_phase:
            st.sidebar.success(f"✅ {phase}")
        elif i == st.session_state.current_phase + 1:
            st.sidebar.info(f"➡️ {phase}")
        else:
            st.sidebar.write(f"⏳ {phase}")

    # Sidebar: uploaded files + join config summary
    if st.session_state.uploaded_files:
        st.sidebar.markdown("---")
        st.sidebar.write("**Uploaded Files:**")
        for filename in st.session_state.uploaded_files.keys():
            st.sidebar.write(f"📄 {filename}")
        if len(st.session_state.uploaded_files) > 1 and st.session_state.join_config:
            st.sidebar.write("**Join Configuration:**")
            st.sidebar.write(f"Primary: {st.session_state.join_config['primary_table']}")
            for join in st.session_state.join_config['joins']:
                st.sidebar.write(f"+ {join['secondary_table']} ({join['join_type']})")

    # Main content by phase (this mirrors your old main())
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
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.rerun()

    elif st.session_state.current_phase == 3:
        display_phase2_results()
        st.markdown("---")
        phase3_processing()

    elif st.session_state.current_phase >= 4:
        tab1, tab2, tab3, tab4 = st.tabs(["Phase 2 Results", "Phase 3 Results", "Data Comparison", "Downloads"])
        with tab1:
            display_phase2_results()
        with tab2:
            display_phase3_results()
        with tab3:
            enhanced_data_comparison_tab()
        with tab4:
            create_final_downloads()

    # Sidebar: restart button
    st.sidebar.markdown("---")
    if st.sidebar.button("Start New Analysis"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        try:
            if os.path.exists(FINAL_OUT_PATH):
                os.remove(FINAL_OUT_PATH)
            flag_map_path = "data/flag_mapping.json"
            if os.path.exists(flag_map_path):
                os.remove(flag_map_path)
        except Exception as e:
            st.warning(f"Could not remove old final dataset: {e}")

        st.rerun()

    # Footer
    st.markdown("---")
    st.markdown("""
    **Data Quality Pipeline** - Powered by AI

    This application processes your data through multiple phases:
    - **Upload & Join**: Upload multiple files and configure table joins
    - **Phase 1**: Traditional cleaning (duplicates, missing values, text standardization)  
    - **Phase 2**: AI-powered issue detection and business logic analysis
    - **Phase 3**: Automated code generation and execution for quality flagging

    All processing is done locally with your OpenAI API key for analysis.
    """)

def main():
    st.title("🔍 LLM Powered Analysis")
    st.markdown("**Transform your data through AI-powered quality detection and cleaning**")
    initialize_session_state()

    # Check availability of a final cleaned dataset
    final_df_available = get_final_clean_df() is not None
    analyze_label = "📊 Analyze" if final_df_available else "📊 Analyze (locked)"

    tab_clean, tab_analyze = st.tabs(["🧹 Clean & Export", analyze_label])

    with tab_clean:
        # render your full cleaner UI (state machine)
        render_cleaner_tab()

    with tab_analyze:
        df_for_analysis = get_final_clean_df()   # ONLY from memory/disk final_cleaned.csv
        if df_for_analysis is None:
            st.info("🔒 Analyzer unlocks after cleaning is complete.")
            st.stop()

        # ✅ pass concrete dataframe into analyzer
        render_analysis(df_for_analysis)


if __name__ == "__main__":
    main()
