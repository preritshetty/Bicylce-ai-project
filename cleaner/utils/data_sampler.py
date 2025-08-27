import pandas as pd
import numpy as np
import random
from typing import Dict, Any

class DataSampler:
    """Enhanced data sampler for LLM analysis with better issue detection"""
    
    def __init__(self, max_rows: int = 150):  # Increased from 50 to 150
        self.max_rows = max_rows
        random.seed(42)  # For reproducible sampling
    
    def create_sample(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Create intelligent sample of the dataset"""
        
        # If small dataset, return everything (increased threshold)
        if len(df) <= self.max_rows:
            return {
                'sampled_data': df.copy(),
                'sampling_strategy': 'full_dataset',
                'original_rows': len(df),
                'sampled_rows': len(df),
                'sampling_ratio': 1.0
            }
        
        # Smart sampling for larger datasets
        sample_indices = set()
        
        # 1. Random baseline (30% of sample) - reduced to make room for targeted sampling
        random_count = int(self.max_rows * 0.3)
        random_indices = random.sample(range(len(df)), random_count)
        sample_indices.update(random_indices)
        
        # 2. Rows with potential data quality issues (40% of sample)
        issue_count = int(self.max_rows * 0.4)
        issue_indices = self._find_potential_issues(df, issue_count)
        sample_indices.update(issue_indices)
        
        # 3. NEW: Rows with contradictory combinations (30% of sample)
        contradiction_count = self.max_rows - len(sample_indices)
        contradiction_indices = self._find_contradictory_patterns(df, contradiction_count)
        sample_indices.update(contradiction_indices)
        
        # Create final sample
        final_indices = list(sample_indices)[:self.max_rows]
        sampled_df = df.iloc[final_indices].copy().reset_index(drop=True)
        
        return {
            'sampled_data': sampled_df,
            'sampling_strategy': 'smart_sampling',
            'original_rows': len(df),
            'sampled_rows': len(sampled_df),
            'sampling_ratio': len(sampled_df) / len(df)
        }
    
    def _find_potential_issues(self, df: pd.DataFrame, count: int) -> list:
        """Find rows that might have data quality issues"""
        issue_indices = set()
        
        # Look for rows with many missing values (top priority)
        missing_counts = df.isnull().sum(axis=1)
        if len(missing_counts) > 0:
            threshold = len(df.columns) * 0.2  # 20% missing
            high_missing = df[missing_counts > threshold].index.tolist()
            issue_indices.update(high_missing[:count//2])
        
        # Look for numeric outliers
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if len(issue_indices) >= count:
                break
            try:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                if IQR > 0:  # Avoid division issues
                    outliers = df[(df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)].index.tolist()
                    issue_indices.update(outliers[:count//4])
            except:
                continue
        
        # Fill remaining with random if we don't have enough
        remaining = count - len(issue_indices)
        if remaining > 0:
            available = set(range(len(df))) - issue_indices
            if available:
                additional = random.sample(list(available), min(remaining, len(available)))
                issue_indices.update(additional)
        
        return list(issue_indices)
    
    # def _find_contradictory_patterns(self, df: pd.DataFrame, count: int) -> list:
    #     """Find rows with contradictory data combinations (e.g., window AND aisle seat)"""
    #     contradiction_indices = set()
        
    #     # Look for boolean/binary contradictions
    #     boolean_cols = []
    #     for col in df.columns:
    #         if df[col].dtype == 'bool' or (df[col].dtype == 'object' and 
    #                                      set(df[col].dropna().unique()).issubset({'True', 'False', True, False, 'Y', 'N', 'Yes', 'No', 1, 0})):
    #             boolean_cols.append(col)
        
    #     # Find seat-related contradictions (window + aisle)
    #     seat_patterns = [
    #         (['window', 'aisle'], ['Window', 'Aisle']),  # Both selected
    #         (['window_seat', 'aisle_seat'], ['true', 'True', 'Y', 'Yes', 1]),
    #         (['seat_type'], ['Window,Aisle', 'window,aisle', 'Both'])  # Combined in one field
    #     ]
        
    #     for pattern in seat_patterns:
    #         cols_to_check = pattern[0]
    #         positive_values = pattern[1]
            
    #         # Find columns that match our pattern
    #         matching_cols = [col for col in df.columns 
    #                        if any(pattern_col.lower() in col.lower() for pattern_col in cols_to_check)]
            
    #         if len(matching_cols) >= 2:
    #             # Look for rows where multiple exclusive options are selected
    #             for idx in df.index:
    #                 if len(contradiction_indices) >= count:
    #                     break
                    
    #                 true_count = 0
    #                 for col in matching_cols[:2]:  # Check first two matching columns
    #                     cell_value = df.loc[idx, col]
    #                     if pd.notna(cell_value) and str(cell_value).strip() in [str(v) for v in positive_values]:
    #                         true_count += 1
                    
    #                 if true_count > 1:  # Multiple exclusive options selected
    #                     contradiction_indices.add(idx)
        
    #     # Look for status contradictions (e.g., cancelled AND confirmed)
    #     status_patterns = [
    #         (['status', 'booking_status', 'flight_status'], ['cancelled', 'confirmed']),
    #         (['active', 'inactive'], ['true', 'false']),
    #         (['available', 'sold'], ['yes', 'no'])
    #     ]
        
    #     for pattern in status_patterns:
    #         field_patterns = pattern[0]
    #         conflicting_values = pattern[1]
            
    #         status_cols = [col for col in df.columns 
    #                       if any(status_word in col.lower() for status_word in field_patterns)]
            
    #         if len(status_cols) >= 1:
    #             for col in status_cols:
    #                 if len(contradiction_indices) >= count:
    #                     break
                    
    #                 # Look for rows with contradictory status values
    #                 col_values = df[col].astype(str).str.lower()
    #                 for idx in df.index:
    #                     if len(contradiction_indices) >= count:
    #                         break
                        
    #                     value = col_values.iloc[idx]
    #                     # Check if value contains conflicting terms
    #                     contains_conflicts = sum(1 for conflict in conflicting_values if conflict in value)
    #                     if contains_conflicts > 1:
    #                         contradiction_indices.add(idx)
        
    #     # Look for numerical contradictions (negative prices, impossible dates)
    #     numeric_cols = df.select_dtypes(include=[np.number]).columns
    #     for col in numeric_cols:
    #         if len(contradiction_indices) >= count:
    #             break
            
    #         col_name_lower = col.lower()
            
    #         # Negative values where they shouldn't exist
    #         if any(keyword in col_name_lower for keyword in ['price', 'fare', 'cost', 'amount', 'duration', 'distance', 'age']):
    #             negative_rows = df[df[col] < 0].index.tolist()
    #             contradiction_indices.update(negative_rows[:min(10, count//4)])
            
    #         # Extremely high values (potential data entry errors)
    #         if any(keyword in col_name_lower for keyword in ['price', 'fare', 'cost']):
    #             try:
    #                 Q3 = df[col].quantile(0.75)
    #                 high_threshold = Q3 * 10  # 10x higher than Q3
    #                 extreme_rows = df[df[col] > high_threshold].index.tolist()
    #                 contradiction_indices.update(extreme_rows[:min(5, count//6)])
    #             except:
    #                 pass
        
    #     # Fill remaining with random suspicious patterns if we don't have enough
    #     remaining = count - len(contradiction_indices)
    #     if remaining > 0:
    #         # Look for rows with unusual text patterns (mixed case, special characters)
    #         text_cols = df.select_dtypes(include=['object']).columns
    #         unusual_rows = set()
            
    #         for col in text_cols:
    #             if len(unusual_rows) >= remaining:
    #                 break
                
    #             text_series = df[col].astype(str)
    #             for idx in df.index:
    #                 if len(unusual_rows) >= remaining:
    #                     break
                    
    #                 value = text_series.iloc[idx]
    #                 # Look for mixed patterns, multiple spaces, special chars
    #                 if (len(value) > 1 and 
    #                     (value != value.upper() and value != value.lower() and value != value.title()) or
    #                     '  ' in value or 
    #                     any(char in value for char in ['@', '#', '$', '%', '&', '*'])):
    #                     unusual_rows.add(idx)
            
    #         contradiction_indices.update(list(unusual_rows)[:remaining])
        
    #     return list(contradiction_indices)

    def _find_contradictory_patterns(self, df: pd.DataFrame, count: int) -> list:
        """Find rows with contradictory data combinations (e.g., window AND aisle seat)"""
        contradiction_indices = set()
        
        # Look for boolean/binary contradictions
        boolean_cols = []
        for col in df.columns:
            if df[col].dtype == 'bool' or (df[col].dtype == 'object' and 
                                        set(df[col].dropna().unique()).issubset({'True', 'False', True, False, 'Y', 'N', 'Yes', 'No', 1, 0})):
                boolean_cols.append(col)
        
        # Find seat-related contradictions (window + aisle)
        seat_patterns = [
            (['window', 'aisle'], ['Window', 'Aisle']),  # Both selected
            (['window_seat', 'aisle_seat'], ['true', 'True', 'Y', 'Yes', 1]),
            (['seat_type'], ['Window,Aisle', 'window,aisle', 'Both'])  # Combined in one field
        ]
        
        for pattern in seat_patterns:
            cols_to_check = pattern[0]
            positive_values = pattern[1]
            
            # Find columns that match our pattern
            matching_cols = [col for col in df.columns 
                        if any(pattern_col.lower() in col.lower() for pattern_col in cols_to_check)]
            
            if len(matching_cols) >= 2:
                # Look for rows where multiple exclusive options are selected
                for idx in df.index:
                    if len(contradiction_indices) >= count:
                        break
                    
                    true_count = 0
                    for col in matching_cols[:2]:  # Check first two matching columns
                        cell_value = df.loc[idx, col]
                        if pd.notna(cell_value) and str(cell_value).strip() in [str(v) for v in positive_values]:
                            true_count += 1
                    
                    if true_count > 1:  # Multiple exclusive options selected
                        contradiction_indices.add(idx)
        
        # Look for status contradictions (e.g., cancelled AND confirmed)
        status_patterns = [
            (['status', 'booking_status', 'flight_status'], ['cancelled', 'confirmed']),
            (['active', 'inactive'], ['true', 'false']),
            (['available', 'sold'], ['yes', 'no'])
        ]
        
        for pattern in status_patterns:
            field_patterns = pattern[0]
            conflicting_values = pattern[1]
            
            status_cols = [col for col in df.columns 
                        if any(status_word in col.lower() for status_word in field_patterns)]
            
            if len(status_cols) >= 1:
                for col in status_cols:
                    if len(contradiction_indices) >= count:
                        break
                    
                    # Look for rows with contradictory status values
                    col_values = df[col].astype(str).str.lower()
                    for idx in df.index:
                        if len(contradiction_indices) >= count:
                            break
                        
                        # FIX: Use .loc instead of .iloc with the actual index
                        value = col_values.loc[idx]
                        # Check if value contains conflicting terms
                        contains_conflicts = sum(1 for conflict in conflicting_values if conflict in value)
                        if contains_conflicts > 1:
                            contradiction_indices.add(idx)
        
        # Look for numerical contradictions (negative prices, impossible dates)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if len(contradiction_indices) >= count:
                break
            
            col_name_lower = col.lower()
            
            # Negative values where they shouldn't exist
            if any(keyword in col_name_lower for keyword in ['price', 'fare', 'cost', 'amount', 'duration', 'distance', 'age']):
                negative_rows = df[df[col] < 0].index.tolist()
                contradiction_indices.update(negative_rows[:min(10, count//4)])
            
            # Extremely high values (potential data entry errors)
            if any(keyword in col_name_lower for keyword in ['price', 'fare', 'cost']):
                try:
                    Q3 = df[col].quantile(0.75)
                    high_threshold = Q3 * 10  # 10x higher than Q3
                    extreme_rows = df[df[col] > high_threshold].index.tolist()
                    contradiction_indices.update(extreme_rows[:min(5, count//6)])
                except:
                    pass
        
        # Fill remaining with random suspicious patterns if we don't have enough
        remaining = count - len(contradiction_indices)
        if remaining > 0:
            # Look for rows with unusual text patterns (mixed case, special characters)
            text_cols = df.select_dtypes(include=['object']).columns
            unusual_rows = set()
            
            for col in text_cols:
                if len(unusual_rows) >= remaining:
                    break
                
                text_series = df[col].astype(str)
                # FIX: Use position-based iteration instead of index-based
                for position in range(len(text_series)):
                    if len(unusual_rows) >= remaining:
                        break
                    
                    try:
                        value = text_series.iloc[position]  # Now using position correctly
                        # Get the actual index for this position
                        actual_idx = text_series.index[position]
                        
                        # Look for mixed patterns, multiple spaces, special chars
                        if (len(value) > 1 and 
                            (value != value.upper() and value != value.lower() and value != value.title()) or
                            '  ' in value or 
                            any(char in value for char in ['@', '#', '$', '%', '&', '*'])):
                            unusual_rows.add(actual_idx)
                            
                    except (IndexError, KeyError):
                        # Skip if there's any indexing issue
                        continue
            
            contradiction_indices.update(list(unusual_rows)[:remaining])
        
        return list(contradiction_indices)
    
    def extract_column_info(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Extract column metadata for LLM - keep it simple"""
        column_info = {}
        
        for col in df.columns:
            info = {
                'dtype': str(df[col].dtype),
                'null_count': int(df[col].isnull().sum()),
                'unique_count': int(df[col].nunique())
            }
            
            # Add type-specific info only if needed
            if pd.api.types.is_numeric_dtype(df[col]) and not df[col].empty:
                try:
                    info.update({
                        'min': float(df[col].min()),
                        'max': float(df[col].max()),
                        'mean': float(df[col].mean())
                    })
                except:
                    pass
            
            elif df[col].dtype == 'object' and not df[col].empty:
                # Just get a few sample values
                unique_vals = df[col].dropna().unique()
                if len(unique_vals) > 0:
                    info['sample_values'] = [str(val) for val in unique_vals[:3]]
            
            column_info[col] = info
        
        return column_info