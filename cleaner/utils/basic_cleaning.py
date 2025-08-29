import pandas as pd
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

class BasicCleaner:
    """Simple data hygiene operations for CSV data"""
    
    def __init__(self):
        self.cleaning_log = []
        
    def remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove exact duplicate rows"""
        df = df.copy()
        original_count = len(df)
        
        df = df.drop_duplicates()
        duplicates_removed = original_count - len(df)
        
        self.cleaning_log.append({
            'operation': 'remove_duplicates',
            'details': f"Removed {duplicates_removed} duplicate rows",
            'success': True
        })
        
        return df
    
    def clean_text_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean text columns by removing extra whitespace"""
        df = df.copy()
        text_columns = df.select_dtypes(include=['object', 'string']).columns
        
        cleaned_columns = []
        
        for column in text_columns:
            try:
                # Only clean non-null values
                mask = df[column].notna()
                if mask.any():
                    # Strip whitespace and normalize spaces
                    df.loc[mask, column] = df.loc[mask, column].astype(str).str.strip()
                    df.loc[mask, column] = df.loc[mask, column].str.replace(r'\s+', ' ', regex=True)
                    cleaned_columns.append(column)
            except Exception:
                pass
        
        if cleaned_columns:
            self.cleaning_log.append({
                'operation': 'clean_text',
                'details': f"Cleaned text in {len(cleaned_columns)} columns",
                'success': True
            })
        
        return df
    def remove_uniform_prefixes(self, df: pd.DataFrame, threshold: float = 0.95) -> pd.DataFrame:
        """
        Detect and remove uniform prefixes like 'Name_', 'Pilot_', 'Crew_' 
        that appear in >= threshold fraction of rows for a given column.
        """
        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            return df

        df = df.copy()
        changed_cols = []

        for col in df.select_dtypes(include=["object", "string"]).columns:
            s = df[col].astype(str).str.strip()

            # Extract leading token (letters only) followed by underscore
            tokens = s.str.extract(r'^([A-Za-z]+)_', expand=False)
            has_prefix = tokens.notna()

            if has_prefix.mean() >= threshold:  # enough rows share the prefix
                top = tokens.mode(dropna=True)
                if not top.empty:
                    token = top.iloc[0]
                    # Remove this prefix from the start of the string
                    df[col] = s.str.replace(rf'^{token}_', '', regex=True)
                    changed_cols.append((col, f"{token}_", round(100 * has_prefix.mean(), 2)))

        if changed_cols:
            self.cleaning_log.append({
                "operation": "remove_uniform_prefixes",
                "details": [
                    {"column": c, "removed_prefix": p, "%rows": pct} 
                    for c, p, pct in changed_cols
                ],
                "success": True
            })

        return df   

    def standardize_case(self, df: pd.DataFrame, case_type: str = 'title') -> pd.DataFrame:
        """Standardize text case for text columns"""
        df = df.copy()
        text_columns = df.select_dtypes(include=['object', 'string']).columns
        
        processed_columns = []
        
        for column in text_columns:
            try:
                # Only process non-null values
                mask = df[column].notna()
                if mask.any():
                    if case_type == 'title':
                        df.loc[mask, column] = df.loc[mask, column].astype(str).str.title()
                    elif case_type == 'upper':
                        df.loc[mask, column] = df.loc[mask, column].astype(str).str.upper()
                    elif case_type == 'lower':
                        df.loc[mask, column] = df.loc[mask, column].astype(str).str.lower()
                    
                    processed_columns.append(column)
            except Exception:
                pass
        
        if processed_columns:
            self.cleaning_log.append({
                'operation': 'standardize_case',
                'details': f"Standardized case in {len(processed_columns)} columns to {case_type}",
                'success': True
            })
        
        return df
    
    def perform_basic_cleaning(
        self,
        df: pd.DataFrame,
        remove_duplicates: bool = True,
        clean_text: bool = True,
        case_type: str = "title"
    ) -> pd.DataFrame:
        """Perform all basic cleaning operations safely"""
        # Guard against None or invalid type
        if df is None or not isinstance(df, pd.DataFrame):
            return df

        # Guard against empty dataframe
        if df.empty:
            return df

        df = df.copy()
        self.cleaning_log = []  # Reset log

        # Step 1: Remove duplicates
        if remove_duplicates:
            df = self.remove_duplicates(df)

        # Step 2: Clean text
        if clean_text:
            df = self.clean_text_columns(df)

        # Step 2.5: Remove uniform prefixes like Name_/Pilot_/Crew_
        df = self.remove_uniform_prefixes(df)

        # Step 3: Standardize case
        if case_type and case_type.lower() != "none":
            df = self.standardize_case(df, case_type)

        return df

    
    def get_cleaning_report(self) -> pd.DataFrame:
        """Get a report of cleaning operations"""
        if not self.cleaning_log:
            return pd.DataFrame()
        
        return pd.DataFrame(self.cleaning_log)
    
    def reset_log(self):
        """Reset the cleaning log"""
        self.cleaning_log = []