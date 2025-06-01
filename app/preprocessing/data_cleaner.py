"""
Data Cleaning Module

Handles data cleaning operations including:
- Missing value detection and handling
- Data validation
- Outlier detection and treatment
- Data type conversion
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class DataCleaner:
    """
    Handles data cleaning operations for survey data.
    """
    
    def __init__(self, 
                 missing_threshold: float = 0.3,
                 outlier_method: str = 'iqr',
                 outlier_factor: float = 1.5):
        """
        Initialize DataCleaner.
        
        Args:
            missing_threshold: Threshold for dropping columns with too many missing values
            outlier_method: Method for outlier detection ('iqr', 'zscore')
            outlier_factor: Factor for outlier detection
        """
        self.missing_threshold = missing_threshold
        self.outlier_method = outlier_method
        self.outlier_factor = outlier_factor
        self.cleaning_report = {}
        
    def clean_survey_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Main method to clean survey data.
        
        Args:
            data: Raw survey data DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        logger.info("Starting data cleaning process")
        
        # Create a copy to avoid modifying original data
        cleaned_data = data.copy()
        
        # Track original shape
        original_shape = cleaned_data.shape
        self.cleaning_report['original_shape'] = original_shape
        
        # Step 1: Remove duplicate responses
        cleaned_data = self._remove_duplicates(cleaned_data)
        
        # Step 2: Handle missing values
        cleaned_data = self._handle_missing_values(cleaned_data)
        
        # Step 3: Validate and clean data types
        cleaned_data = self._clean_data_types(cleaned_data)
        
        # Step 4: Handle outliers
        cleaned_data = self._handle_outliers(cleaned_data)
        
        # Step 5: Clean text responses
        cleaned_data = self._clean_text_responses(cleaned_data)
        
        # Track final shape
        final_shape = cleaned_data.shape
        self.cleaning_report['final_shape'] = final_shape
        self.cleaning_report['rows_removed'] = original_shape[0] - final_shape[0]
        self.cleaning_report['columns_removed'] = original_shape[1] - final_shape[1]
        
        logger.info(f"Data cleaning completed. Shape changed from {original_shape} to {final_shape}")
        
        return cleaned_data
    
    def _remove_duplicates(self, data: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate responses."""
        initial_count = len(data)
        
        # Remove exact duplicates
        data = data.drop_duplicates()
        
        # Remove duplicates based on user_id and survey_id combination
        if 'user_id' in data.columns and 'survey_id' in data.columns:
            data = data.drop_duplicates(subset=['user_id', 'survey_id'], keep='last')
        
        duplicates_removed = initial_count - len(data)
        self.cleaning_report['duplicates_removed'] = duplicates_removed
        
        if duplicates_removed > 0:
            logger.info(f"Removed {duplicates_removed} duplicate responses")
        
        return data
    
    def _handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset."""
        missing_info = {}
        
        # Calculate missing value percentages
        missing_percentages = (data.isnull().sum() / len(data)) * 100
        
        # Drop columns with too many missing values
        columns_to_drop = missing_percentages[missing_percentages > (self.missing_threshold * 100)].index.tolist()
        
        if columns_to_drop:
            data = data.drop(columns=columns_to_drop)
            missing_info['columns_dropped'] = columns_to_drop
            logger.info(f"Dropped columns with >30% missing values: {columns_to_drop}")
        
        # Handle remaining missing values
        for column in data.columns:
            missing_count = data[column].isnull().sum()
            if missing_count > 0:
                missing_info[column] = {
                    'missing_count': missing_count,
                    'missing_percentage': (missing_count / len(data)) * 100
                }
                
                # Different strategies for different data types
                if data[column].dtype in ['object', 'string']:
                    # For categorical/text data, use mode or 'Unknown'
                    mode_value = data[column].mode()
                    fill_value = mode_value[0] if not mode_value.empty else 'Unknown'
                    data[column] = data[column].fillna(fill_value)
                    missing_info[column]['fill_method'] = f'mode ({fill_value})'
                    
                elif data[column].dtype in ['int64', 'float64']:
                    # For numerical data, use median
                    median_value = data[column].median()
                    data[column] = data[column].fillna(median_value)
                    missing_info[column]['fill_method'] = f'median ({median_value})'
                    
                elif pd.api.types.is_datetime64_any_dtype(data[column]):
                    # For datetime data, use forward fill
                    data[column] = data[column].fillna(method='ffill')
                    missing_info[column]['fill_method'] = 'forward_fill'
        
        self.cleaning_report['missing_values'] = missing_info
        return data
    
    def _clean_data_types(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate data types."""
        type_changes = {}
        
        for column in data.columns:
            original_dtype = data[column].dtype
            
            # Convert string representations of numbers
            if data[column].dtype == 'object':
                # Try to convert to numeric if it looks like numbers
                try:
                    numeric_data = pd.to_numeric(data[column], errors='coerce')
                    # If most values are convertible, use numeric
                    if numeric_data.notna().sum() / len(data) > 0.8:
                        data[column] = numeric_data
                        type_changes[column] = f'{original_dtype} -> numeric'
                except:
                    pass
            
            # Handle specific survey response patterns
            if 'rating' in column.lower() or 'score' in column.lower():
                # Ensure rating columns are numeric
                data[column] = pd.to_numeric(data[column], errors='coerce')
                if original_dtype != data[column].dtype:
                    type_changes[column] = f'{original_dtype} -> numeric (rating)'
            
            # Clean yes/no responses
            if data[column].dtype == 'object':
                unique_values = data[column].str.lower().unique() if hasattr(data[column], 'str') else []
                if len(unique_values) <= 5 and any(val in ['yes', 'no', 'true', 'false', 'y', 'n'] 
                                                  for val in unique_values if pd.notna(val)):
                    # Convert to boolean
                    data[column] = data[column].str.lower().map({
                        'yes': True, 'no': False, 'true': True, 'false': False, 'y': True, 'n': False
                    })
                    type_changes[column] = f'{original_dtype} -> boolean'
        
        if type_changes:
            self.cleaning_report['type_changes'] = type_changes
            logger.info(f"Made {len(type_changes)} data type changes")
        
        return data
    
    def _handle_outliers(self, data: pd.DataFrame) -> pd.DataFrame:
        """Detect and handle outliers in numerical columns."""
        outlier_info = {}
        
        numerical_columns = data.select_dtypes(include=[np.number]).columns
        
        for column in numerical_columns:
            if self.outlier_method == 'iqr':
                outliers = self._detect_outliers_iqr(data[column])
            elif self.outlier_method == 'zscore':
                outliers = self._detect_outliers_zscore(data[column])
            else:
                continue
            
            if outliers.sum() > 0:
                outlier_info[column] = {
                    'outlier_count': outliers.sum(),
                    'outlier_percentage': (outliers.sum() / len(data)) * 100
                }
                
                # Cap outliers instead of removing (preserve data)
                if self.outlier_method == 'iqr':
                    Q1 = data[column].quantile(0.25)
                    Q3 = data[column].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - self.outlier_factor * IQR
                    upper_bound = Q3 + self.outlier_factor * IQR
                    
                    data[column] = data[column].clip(lower=lower_bound, upper=upper_bound)
                    outlier_info[column]['treatment'] = f'capped to [{lower_bound:.2f}, {upper_bound:.2f}]'
        
        if outlier_info:
            self.cleaning_report['outliers'] = outlier_info
            logger.info(f"Handled outliers in {len(outlier_info)} columns")
        
        return data
    
    def _detect_outliers_iqr(self, series: pd.Series) -> pd.Series:
        """Detect outliers using IQR method."""
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - self.outlier_factor * IQR
        upper_bound = Q3 + self.outlier_factor * IQR
        return (series < lower_bound) | (series > upper_bound)
    
    def _detect_outliers_zscore(self, series: pd.Series, threshold: float = 3) -> pd.Series:
        """Detect outliers using Z-score method."""
        z_scores = np.abs((series - series.mean()) / series.std())
        return z_scores > threshold
    
    def _clean_text_responses(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clean text responses."""
        text_cleaning_info = {}
        
        text_columns = data.select_dtypes(include=['object']).columns
        
        for column in text_columns:
            if data[column].dtype == 'object':
                original_unique = data[column].nunique()
                
                # Remove leading/trailing whitespace
                data[column] = data[column].astype(str).str.strip()
                
                # Remove extra whitespace
                data[column] = data[column].str.replace(r'\s+', ' ', regex=True)
                
                # Handle empty strings
                data[column] = data[column].replace('', np.nan)
                data[column] = data[column].replace('nan', np.nan)
                
                # Convert to lowercase for consistency (if it's categorical)
                if data[column].nunique() < 50:  # Assuming categorical if few unique values
                    data[column] = data[column].str.lower()
                
                final_unique = data[column].nunique()
                
                if original_unique != final_unique:
                    text_cleaning_info[column] = {
                        'original_unique': original_unique,
                        'final_unique': final_unique
                    }
        
        if text_cleaning_info:
            self.cleaning_report['text_cleaning'] = text_cleaning_info
        
        return data
    
    def get_cleaning_report(self) -> Dict[str, Any]:
        """Get the cleaning report."""
        return self.cleaning_report
    
    def save_cleaning_report(self, filepath: str):
        """Save cleaning report to a file."""
        import json
        
        # Convert numpy types to native Python types for JSON serialization
        def convert_numpy_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        # Create a JSON-serializable version of the report
        serializable_report = {}
        for key, value in self.cleaning_report.items():
            if isinstance(value, dict):
                serializable_report[key] = {k: convert_numpy_types(v) for k, v in value.items()}
            else:
                serializable_report[key] = convert_numpy_types(value)
        
        with open(filepath, 'w') as f:
            json.dump(serializable_report, f, indent=2)
        
        logger.info(f"Cleaning report saved to {filepath}") 