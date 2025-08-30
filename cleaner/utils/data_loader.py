import pandas as pd
import os

class CSVProcessor:
    """Simple CSV loader/validator for the cleaning pipeline."""
    
    def __init__(self):
        """Initialize CSV processor"""
        pass
        
    def load_and_validate(self, uploaded_file):
        """Load and validate a CSV file."""
        try:
            if isinstance(uploaded_file, str):
                # if a path is passed
                df = pd.read_csv(uploaded_file)
            else:
                # if it's a Streamlit UploadedFile
                df = pd.read_csv(uploaded_file)
            
            # Basic sanity check
            if df.empty:
                raise ValueError("CSV file is empty")
            
            return df
        except Exception as e:
            raise RuntimeError(f"Failed to load CSV: {e}")

