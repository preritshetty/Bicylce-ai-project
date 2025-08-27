import pandas as pd
import numpy as np
import streamlit as st
from typing import Dict, List, Optional, Union, Tuple
from dataclasses import dataclass
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

class ImputationStrategy(Enum):
    MEAN = "mean"
    MEDIAN = "median"
    MODE = "mode"
    FORWARD_FILL = "forward_fill"
    BACKWARD_FILL = "backward_fill"
    INTERPOLATE = "interpolate"
    CONSTANT = "constant"
    DROP = "drop"
    SKIP = "skip"

@dataclass
class MissingValueReport:
    """Report on missing values before and after imputation"""
    column: str
    original_missing_count: int
    original_missing_percentage: float
    imputed_missing_count: int
    strategy_used: str
    imputation_value: Union[str, float, int, None] = None
    success: bool = True
    error_message: Optional[str] = None

class MissingValueCleaner:
    """Comprehensive missing value handler for various data types"""
    
    def __init__(self, default_strategies: Optional[Dict[str, ImputationStrategy]] = None):
        """
        Initialize with default strategies for different data types
        
        Args:
            default_strategies: Override default imputation strategies
        """
        self.default_strategies = default_strategies or {
            'numeric': ImputationStrategy.MEDIAN,
            'categorical': ImputationStrategy.MODE,
            'boolean': ImputationStrategy.CONSTANT,
            'datetime': ImputationStrategy.FORWARD_FILL,
            'text': ImputationStrategy.CONSTANT
        }
        
        self.imputation_reports = []
        self.column_strategies = {}
        
    def analyze_missing_patterns(self, df: pd.DataFrame) -> Dict:
        """
        Analyze missing data patterns and suggest optimal strategies
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Dict with missing data analysis
        """
        analysis = {
            'total_missing': df.isnull().sum().sum(),
            'missing_by_column': {},
            'missing_patterns': {},
            'recommendations': {}
        }
        
        # Analyze each column
        for column in df.columns:
            missing_count = df[column].isnull().sum()
            missing_percentage = (missing_count / len(df)) * 100
            
            if missing_count > 0:
                column_info = {
                    'missing_count': missing_count,
                    'missing_percentage': missing_percentage,
                    'dtype': str(df[column].dtype),
                    'unique_values': df[column].nunique(),
                    'sample_values': df[column].dropna().head(5).tolist()
                }
                
                # Suggest strategy based on missing percentage and data type
                suggested_strategy = self._suggest_strategy(df[column], missing_percentage)
                column_info['suggested_strategy'] = suggested_strategy
                
                analysis['missing_by_column'][column] = column_info
        
        # Analyze missing patterns across columns
        analysis['missing_patterns'] = self._analyze_missing_patterns(df)
        
        return analysis
    
    def _suggest_strategy(self, series: pd.Series, missing_percentage: float) -> str:
        """Suggest optimal imputation strategy for a column"""
        
        # High missing percentage (>50%) - consider dropping or constant
        if missing_percentage > 50:
            return "Consider dropping column or using constant value"
        
        # Very low missing percentage (<5%) - any strategy should work
        if missing_percentage < 5:
            if pd.api.types.is_numeric_dtype(series):
                return "median (low missing rate)"
            else:
                return "mode (low missing rate)"
        
        # Medium missing percentage - choose based on data type
        if pd.api.types.is_numeric_dtype(series):
            # Check for skewness to choose mean vs median
            non_null_data = series.dropna()
            if len(non_null_data) > 3:
                skewness = non_null_data.skew()
                if abs(skewness) < 0.5:
                    return "mean (normally distributed)"
                else:
                    return "median (skewed distribution)"
            return "median (safe choice)"
        
        elif pd.api.types.is_categorical_dtype(series) or series.dtype == 'object':
            unique_ratio = series.nunique() / len(series.dropna())
            if unique_ratio < 0.1:  # Low cardinality
                return "mode (categorical)"
            else:
                return "constant value or consider advanced imputation"
        
        elif pd.api.types.is_datetime64_any_dtype(series):
            return "forward_fill or interpolate"
        
        return "mode (default)"
    
    def _analyze_missing_patterns(self, df: pd.DataFrame) -> Dict:
        """Analyze patterns in missing data across columns"""
        patterns = {}
        
        # Find columns that are missing together
        missing_df = df.isnull()
        
        # Correlation of missing values
        if len(missing_df.columns) > 1:
            missing_corr = missing_df.corr()
            high_corr_pairs = []
            
            for i, col1 in enumerate(missing_corr.columns):
                for j, col2 in enumerate(missing_corr.columns[i+1:], i+1):
                    corr_value = missing_corr.iloc[i, j]
                    if abs(corr_value) > 0.5:  # High correlation threshold
                        high_corr_pairs.append({
                            'column1': col1,
                            'column2': col2,
                            'correlation': corr_value
                        })
            
            patterns['correlated_missing'] = high_corr_pairs
        
        # Find rows with multiple missing values
        rows_with_multiple_missing = (missing_df.sum(axis=1) > 1).sum()
        patterns['rows_multiple_missing'] = rows_with_multiple_missing
        
        # Completely empty rows
        completely_empty_rows = (missing_df.sum(axis=1) == len(df.columns)).sum()
        patterns['completely_empty_rows'] = completely_empty_rows
        
        return patterns
    
    def handle_numeric_missing(self, df: pd.DataFrame, columns: List[str] = None, 
                             strategy: ImputationStrategy = ImputationStrategy.MEDIAN) -> pd.DataFrame:
        """
        Handle missing values in numeric columns
        
        Args:
            df: DataFrame to process
            columns: List of columns to process (if None, auto-detect numeric)
            strategy: Imputation strategy to use
            
        Returns:
            DataFrame with imputed values
        """
        df = df.copy()
        
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        for column in columns:
            if column not in df.columns:
                continue
                
            original_missing = df[column].isnull().sum()
            if original_missing == 0:
                continue
            
            try:
                if strategy == ImputationStrategy.MEAN:
                    imputation_value = df[column].mean()
                    df[column].fillna(imputation_value, inplace=True)
                    
                elif strategy == ImputationStrategy.MEDIAN:
                    imputation_value = df[column].median()
                    df[column].fillna(imputation_value, inplace=True)
                    
                elif strategy == ImputationStrategy.FORWARD_FILL:
                    df[column].fillna(method='ffill', inplace=True)
                    imputation_value = "forward_fill"
                    
                elif strategy == ImputationStrategy.BACKWARD_FILL:
                    df[column].fillna(method='bfill', inplace=True)
                    imputation_value = "backward_fill"
                    
                elif strategy == ImputationStrategy.INTERPOLATE:
                    df[column].interpolate(inplace=True)
                    imputation_value = "interpolated"
                    
                elif strategy == ImputationStrategy.CONSTANT:
                    # Use 0 as default for numeric
                    imputation_value = 0
                    df[column].fillna(imputation_value, inplace=True)
                
                # Record the imputation
                self._record_imputation(column, original_missing, strategy.value, 
                                      imputation_value, len(df))
                                      
            except Exception as e:
                self._record_imputation(column, original_missing, strategy.value, 
                                      None, len(df), success=False, error=str(e))
        
        return df
    
    def handle_categorical_missing(self, df: pd.DataFrame, columns: List[str] = None,
                                 strategy: ImputationStrategy = ImputationStrategy.MODE,
                                 constant_value: str = "Unknown") -> pd.DataFrame:
        """
        Handle missing values in categorical columns
        
        Args:
            df: DataFrame to process
            columns: List of columns to process (if None, auto-detect categorical)
            strategy: Imputation strategy to use
            constant_value: Value to use for constant strategy
            
        Returns:
            DataFrame with imputed values
        """
        df = df.copy()
        
        if columns is None:
            columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        for column in columns:
            if column not in df.columns:
                continue
                
            original_missing = df[column].isnull().sum()
            if original_missing == 0:
                continue
            
            try:
                if strategy == ImputationStrategy.MODE:
                    mode_value = df[column].mode()
                    if len(mode_value) > 0:
                        imputation_value = mode_value.iloc[0]
                        df[column].fillna(imputation_value, inplace=True)
                    else:
                        # Fallback to constant if no mode
                        imputation_value = constant_value
                        df[column].fillna(imputation_value, inplace=True)
                        
                elif strategy == ImputationStrategy.CONSTANT:
                    imputation_value = constant_value
                    df[column].fillna(imputation_value, inplace=True)
                    
                elif strategy == ImputationStrategy.FORWARD_FILL:
                    df[column].fillna(method='ffill', inplace=True)
                    imputation_value = "forward_fill"
                    
                elif strategy == ImputationStrategy.BACKWARD_FILL:
                    df[column].fillna(method='bfill', inplace=True)
                    imputation_value = "backward_fill"
                
                # Record the imputation
                self._record_imputation(column, original_missing, strategy.value,
                                      imputation_value, len(df))
                                      
            except Exception as e:
                self._record_imputation(column, original_missing, strategy.value,
                                      None, len(df), success=False, error=str(e))
        
        return df
    
    def handle_boolean_missing(self, df: pd.DataFrame, columns: List[str] = None,
                             default_value: bool = False) -> pd.DataFrame:
        """
        Handle missing values in boolean columns
        
        Args:
            df: DataFrame to process
            columns: List of columns to process
            default_value: Default boolean value to use
            
        Returns:
            DataFrame with imputed values
        """
        df = df.copy()
        
        if columns is None:
            # Try to identify boolean columns
            columns = []
            for col in df.columns:
                if df[col].dtype == 'bool':
                    columns.append(col)
                else:
                    # Check if object column has boolean-like values
                    unique_vals = set(str(val).upper() for val in df[col].dropna().unique())
                    boolean_patterns = [
                        {'TRUE', 'FALSE'}, {'YES', 'NO'}, {'Y', 'N'}, {'1', '0'},
                        {'AVAILABLE', 'NOT AVAILABLE'}, {'ENABLED', 'DISABLED'}
                    ]
                    if any(unique_vals.issubset(pattern) or pattern.issubset(unique_vals) 
                           for pattern in boolean_patterns):
                        columns.append(col)
        
        for column in columns:
            if column not in df.columns:
                continue
                
            original_missing = df[column].isnull().sum()
            if original_missing == 0:
                continue
            
            try:
                df[column].fillna(default_value, inplace=True)
                
                # Record the imputation
                self._record_imputation(column, original_missing, "constant",
                                      default_value, len(df))
                                      
            except Exception as e:
                self._record_imputation(column, original_missing, "constant",
                                      None, len(df), success=False, error=str(e))
        
        return df
    
    def handle_datetime_missing(self, df: pd.DataFrame, columns: List[str] = None,
                              method: str = 'forward_fill') -> pd.DataFrame:
        """
        Handle missing values in datetime columns
        
        Args:
            df: DataFrame to process
            columns: List of columns to process
            method: Method to use ('forward_fill', 'backward_fill', 'interpolate')
            
        Returns:
            DataFrame with imputed values
        """
        df = df.copy()
        
        if columns is None:
            columns = df.select_dtypes(include=['datetime64']).columns.tolist()
            # Also check for potential datetime columns
            for col in df.select_dtypes(include=['object']).columns:
                try:
                    pd.to_datetime(df[col].dropna().head(10))
                    columns.append(col)
                except:
                    pass
        
        for column in columns:
            if column not in df.columns:
                continue
                
            original_missing = df[column].isnull().sum()
            if original_missing == 0:
                continue
            
            try:
                # Ensure column is datetime
                if not pd.api.types.is_datetime64_any_dtype(df[column]):
                    df[column] = pd.to_datetime(df[column], errors='coerce')
                
                if method == 'forward_fill':
                    df[column].fillna(method='ffill', inplace=True)
                    imputation_value = "forward_fill"
                elif method == 'backward_fill':
                    df[column].fillna(method='bfill', inplace=True)
                    imputation_value = "backward_fill"
                elif method == 'interpolate':
                    df[column].interpolate(method='time', inplace=True)
                    imputation_value = "interpolated"
                
                # Record the imputation
                self._record_imputation(column, original_missing, method,
                                      imputation_value, len(df))
                                      
            except Exception as e:
                self._record_imputation(column, original_missing, method,
                                      None, len(df), success=False, error=str(e))
        
        return df
    
    def handle_all_missing_values(self, df: pd.DataFrame, 
                                custom_strategies: Optional[Dict[str, ImputationStrategy]] = None) -> pd.DataFrame:
        """
        Comprehensive missing value handling for all columns
        
        Args:
            df: DataFrame to process
            custom_strategies: Custom strategies for specific columns
            
        Returns:
            DataFrame with all missing values handled
        """
        df = df.copy()
        self.imputation_reports = []  # Reset reports
        
        # Apply custom strategies first
        if custom_strategies:
            for column, strategy in custom_strategies.items():
                if column in df.columns and df[column].isnull().sum() > 0:
                    df = self._apply_strategy_to_column(df, column, strategy)
        
        # Handle remaining missing values by type
        remaining_missing_cols = df.columns[df.isnull().any()].tolist()
        
        if remaining_missing_cols:
            # Numeric columns
            numeric_cols = df[remaining_missing_cols].select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:
                df = self.handle_numeric_missing(df, numeric_cols, self.default_strategies['numeric'])
            
            # Categorical columns
            categorical_cols = df[remaining_missing_cols].select_dtypes(include=['object']).columns.tolist()
            if categorical_cols:
                df = self.handle_categorical_missing(df, categorical_cols, self.default_strategies['categorical'])
            
            # Boolean columns
            boolean_cols = [col for col in remaining_missing_cols if self._is_boolean_column(df[col])]
            if boolean_cols:
                df = self.handle_boolean_missing(df, boolean_cols)
            
            # Datetime columns
            datetime_cols = df[remaining_missing_cols].select_dtypes(include=['datetime64']).columns.tolist()
            if datetime_cols:
                df = self.handle_datetime_missing(df, datetime_cols)
        
        return df
    
    def _apply_strategy_to_column(self, df: pd.DataFrame, column: str, 
                                strategy: ImputationStrategy) -> pd.DataFrame:
        """Apply specific strategy to a single column"""
        
        if pd.api.types.is_numeric_dtype(df[column]):
            df = self.handle_numeric_missing(df, [column], strategy)
        elif pd.api.types.is_datetime64_any_dtype(df[column]):
            df = self.handle_datetime_missing(df, [column], strategy.value)
        else:
            df = self.handle_categorical_missing(df, [column], strategy)
        
        return df
    
    def _is_boolean_column(self, series: pd.Series) -> bool:
        """Check if series contains boolean-like values"""
        unique_vals = set(str(val).upper() for val in series.dropna().unique())
        boolean_patterns = [
            {'TRUE', 'FALSE'}, {'YES', 'NO'}, {'Y', 'N'}, {'1', '0'},
            {'AVAILABLE', 'NOT AVAILABLE'}, {'ENABLED', 'DISABLED'}
        ]
        return any(unique_vals.issubset(pattern) or pattern.issubset(unique_vals) 
                   for pattern in boolean_patterns)
    
    def _record_imputation(self, column: str, original_missing: int, strategy: str,
                         imputation_value, total_rows: int, success: bool = True,
                         error: str = None):
        """Record imputation details for reporting"""
        
        report = MissingValueReport(
            column=column,
            original_missing_count=original_missing,
            original_missing_percentage=(original_missing / total_rows) * 100,
            imputed_missing_count=0,  # Will be updated after imputation
            strategy_used=strategy,
            imputation_value=imputation_value,
            success=success,
            error_message=error
        )
        
        self.imputation_reports.append(report)
    
    def get_imputation_summary(self) -> pd.DataFrame:
        """Get summary of all imputations performed"""
        
        if not self.imputation_reports:
            return pd.DataFrame()
        
        summary_data = []
        for report in self.imputation_reports:
            summary_data.append({
                'Column': report.column,
                'Original Missing': report.original_missing_count,
                'Missing %': f"{report.original_missing_percentage:.1f}%",
                'Strategy': report.strategy_used,
                'Imputation Value': report.imputation_value,
                'Success': "✅" if report.success else "❌",
                'Error': report.error_message or ""
            })
        
        return pd.DataFrame(summary_data)
    
    def validate_imputation_quality(self, original_df: pd.DataFrame, 
                                  imputed_df: pd.DataFrame) -> Dict:
        """
        Validate the quality of imputation
        
        Args:
            original_df: Original DataFrame before imputation
            imputed_df: DataFrame after imputation
            
        Returns:
            Dict with validation results
        """
        validation = {
            'total_missing_before': original_df.isnull().sum().sum(),
            'total_missing_after': imputed_df.isnull().sum().sum(),
            'missing_reduction': 0,
            'columns_fully_imputed': [],
            'columns_partially_imputed': [],
            'data_integrity_issues': []
        }
        
        validation['missing_reduction'] = validation['total_missing_before'] - validation['total_missing_after']
        
        # Check each column
        for column in original_df.columns:
            missing_before = original_df[column].isnull().sum()
            missing_after = imputed_df[column].isnull().sum()
            
            if missing_before > 0:
                if missing_after == 0:
                    validation['columns_fully_imputed'].append(column)
                elif missing_after < missing_before:
                    validation['columns_partially_imputed'].append({
                        'column': column,
                        'before': missing_before,
                        'after': missing_after
                    })
        
        # Check for data integrity issues
        for column in original_df.columns:
            if pd.api.types.is_numeric_dtype(original_df[column]):
                # Check if imputed values are within reasonable range
                original_min = original_df[column].min()
                original_max = original_df[column].max()
                imputed_min = imputed_df[column].min()
                imputed_max = imputed_df[column].max()
                
                if imputed_min < original_min or imputed_max > original_max:
                    validation['data_integrity_issues'].append({
                        'column': column,
                        'issue': 'Imputed values outside original range',
                        'original_range': f"{original_min} - {original_max}",
                        'imputed_range': f"{imputed_min} - {imputed_max}"
                    })
        
        return validation