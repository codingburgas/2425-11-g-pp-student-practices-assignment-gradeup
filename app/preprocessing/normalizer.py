"""
Data Normalization Module

Handles data normalization and scaling operations including:
- StandardScaler normalization (z-score)
- MinMaxScaler normalization 
- RobustScaler normalization
- Unit vector scaling
- Custom scaling methods
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, Normalizer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import joblib

logger = logging.getLogger(__name__)


class DataNormalizer:
    """
    Handles data normalization operations for survey data.
    """
    
    def __init__(self, 
                 numerical_method: str = 'standard',
                 categorical_method: str = 'onehot',
                 save_scalers: bool = True):
        """
        Initialize DataNormalizer.
        
        Args:
            numerical_method: Method for numerical data ('standard', 'minmax', 'robust', 'unit')
            categorical_method: Method for categorical data ('onehot', 'label', 'target')
            save_scalers: Whether to save fitted scalers for later use
        """
        self.numerical_method = numerical_method
        self.categorical_method = categorical_method
        self.save_scalers = save_scalers
        self.scalers = {}
        self.encoders = {}
        self.normalization_report = {}
        
    def normalize_data(self, data: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Main method to normalize survey data.
        
        Args:
            data: Input DataFrame to normalize
            fit: Whether to fit scalers (True for training, False for new data)
            
        Returns:
            Normalized DataFrame
        """
        logger.info("Starting data normalization process")
        
        # Create a copy to avoid modifying original data
        normalized_data = data.copy()
        
        # Track original info
        original_shape = normalized_data.shape
        self.normalization_report['original_shape'] = original_shape
        self.normalization_report['original_columns'] = list(normalized_data.columns)
        
        # Step 1: Normalize numerical columns
        normalized_data = self._normalize_numerical_data(normalized_data, fit=fit)
        
        # Step 2: Encode categorical columns
        normalized_data = self._encode_categorical_data(normalized_data, fit=fit)
        
        # Step 3: Handle boolean columns
        normalized_data = self._handle_boolean_data(normalized_data)
        
        # Track final info
        final_shape = normalized_data.shape
        self.normalization_report['final_shape'] = final_shape
        self.normalization_report['final_columns'] = list(normalized_data.columns)
        self.normalization_report['columns_added'] = final_shape[1] - original_shape[1]
        
        logger.info(f"Data normalization completed. Shape changed from {original_shape} to {final_shape}")
        
        return normalized_data
    
    def _normalize_numerical_data(self, data: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Normalize numerical columns."""
        numerical_columns = data.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove boolean columns (they should be handled separately)
        boolean_columns = []
        for col in numerical_columns:
            if data[col].nunique() == 2 and set(data[col].unique()) <= {0, 1, True, False}:
                boolean_columns.append(col)
        
        numerical_columns = [col for col in numerical_columns if col not in boolean_columns]
        
        if not numerical_columns:
            logger.info("No numerical columns found for normalization")
            return data
        
        normalization_info = {}
        
        for column in numerical_columns:
            if fit:
                # Initialize scaler based on method
                if self.numerical_method == 'standard':
                    scaler = StandardScaler()
                elif self.numerical_method == 'minmax':
                    scaler = MinMaxScaler()
                elif self.numerical_method == 'robust':
                    scaler = RobustScaler()
                elif self.numerical_method == 'unit':
                    scaler = Normalizer()
                else:
                    logger.warning(f"Unknown normalization method: {self.numerical_method}. Using standard.")
                    scaler = StandardScaler()
                
                # Fit and transform
                data[[column]] = scaler.fit_transform(data[[column]])
                
                # Save scaler
                if self.save_scalers:
                    self.scalers[column] = scaler
                
                # Record info
                normalization_info[column] = {
                    'method': self.numerical_method,
                    'original_mean': float(data[column].mean()) if self.numerical_method == 'unit' else None,
                    'original_std': float(data[column].std()) if self.numerical_method == 'unit' else None,
                    'scaler_params': self._get_scaler_params(scaler)
                }
            else:
                # Transform using existing scaler
                if column in self.scalers:
                    data[[column]] = self.scalers[column].transform(data[[column]])
                    normalization_info[column] = {
                        'method': self.numerical_method,
                        'transformed': True
                    }
                else:
                    logger.warning(f"No scaler found for column {column}. Skipping normalization.")
        
        if normalization_info:
            self.normalization_report['numerical_normalization'] = normalization_info
            logger.info(f"Normalized {len(normalization_info)} numerical columns using {self.numerical_method} method")
        
        return data
    
    def _encode_categorical_data(self, data: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Encode categorical columns."""
        categorical_columns = data.select_dtypes(include=['object']).columns.tolist()
        
        if not categorical_columns:
            logger.info("No categorical columns found for encoding")
            return data
        
        encoding_info = {}
        
        for column in categorical_columns:
            original_column_count = len(data.columns)
            
            if fit:
                if self.categorical_method == 'onehot':
                    # One-hot encoding
                    encoded_df = pd.get_dummies(data[column], prefix=column, drop_first=True)
                    
                    # Store the columns created for this feature
                    self.encoders[column] = {
                        'method': 'onehot',
                        'columns': encoded_df.columns.tolist(),
                        'original_categories': data[column].unique().tolist()
                    }
                    
                    # Drop original column and add encoded columns
                    data = data.drop(columns=[column])
                    data = pd.concat([data, encoded_df], axis=1)
                    
                elif self.categorical_method == 'label':
                    # Label encoding
                    encoder = LabelEncoder()
                    data[column] = encoder.fit_transform(data[column].astype(str))
                    
                    # Save encoder
                    if self.save_scalers:
                        self.encoders[column] = {
                            'method': 'label',
                            'encoder': encoder,
                            'classes': encoder.classes_.tolist()
                        }
                
                elif self.categorical_method == 'target':
                    # Target encoding (requires target variable - placeholder for now)
                    logger.warning(f"Target encoding not implemented. Using label encoding for {column}")
                    encoder = LabelEncoder()
                    data[column] = encoder.fit_transform(data[column].astype(str))
                    
                    if self.save_scalers:
                        self.encoders[column] = {
                            'method': 'label',
                            'encoder': encoder,
                            'classes': encoder.classes_.tolist()
                        }
                
                # Record encoding info
                new_column_count = len(data.columns)
                encoding_info[column] = {
                    'method': self.categorical_method,
                    'original_unique_values': len(self.encoders[column].get('original_categories', [])),
                    'columns_created': new_column_count - original_column_count + 1,  # +1 for dropped original
                    'new_columns': self.encoders[column].get('columns', [column])
                }
                
            else:
                # Transform using existing encoder
                if column in self.encoders:
                    encoder_info = self.encoders[column]
                    
                    if encoder_info['method'] == 'onehot':
                        # Apply one-hot encoding with same structure
                        encoded_df = pd.get_dummies(data[column], prefix=column, drop_first=True)
                        
                        # Ensure all expected columns are present
                        for expected_col in encoder_info['columns']:
                            if expected_col not in encoded_df.columns:
                                encoded_df[expected_col] = 0
                        
                        # Keep only the expected columns
                        encoded_df = encoded_df[encoder_info['columns']]
                        
                        # Drop original and add encoded
                        data = data.drop(columns=[column])
                        data = pd.concat([data, encoded_df], axis=1)
                        
                    elif encoder_info['method'] == 'label':
                        # Apply label encoding
                        encoder = encoder_info['encoder']
                        # Handle unseen categories
                        data[column] = data[column].astype(str)
                        unseen_categories = set(data[column]) - set(encoder.classes_)
                        if unseen_categories:
                            logger.warning(f"Unseen categories in {column}: {unseen_categories}. Assigning -1.")
                            data[column] = data[column].map(lambda x: x if x in encoder.classes_ else 'UNKNOWN')
                            # Add 'UNKNOWN' to encoder classes temporarily
                            if 'UNKNOWN' not in encoder.classes_:
                                encoder.classes_ = np.append(encoder.classes_, 'UNKNOWN')
                        
                        data[column] = encoder.transform(data[column])
                    
                    encoding_info[column] = {
                        'method': encoder_info['method'],
                        'transformed': True
                    }
                else:
                    logger.warning(f"No encoder found for column {column}. Skipping encoding.")
        
        if encoding_info:
            self.normalization_report['categorical_encoding'] = encoding_info
            logger.info(f"Encoded {len(encoding_info)} categorical columns using {self.categorical_method} method")
        
        return data
    
    def _handle_boolean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Handle boolean columns."""
        boolean_info = {}
        
        for column in data.columns:
            if data[column].dtype == 'bool':
                # Convert boolean to 0/1
                data[column] = data[column].astype(int)
                boolean_info[column] = 'converted to int (0/1)'
            elif data[column].nunique() == 2 and set(data[column].unique()) <= {0, 1}:
                # Already in 0/1 format
                boolean_info[column] = 'already in 0/1 format'
        
        if boolean_info:
            self.normalization_report['boolean_handling'] = boolean_info
            logger.info(f"Handled {len(boolean_info)} boolean columns")
        
        return data
    
    def _get_scaler_params(self, scaler) -> Dict[str, Any]:
        """Get parameters from a fitted scaler."""
        params = {}
        
        if hasattr(scaler, 'mean_'):
            params['mean'] = scaler.mean_.tolist() if hasattr(scaler.mean_, 'tolist') else float(scaler.mean_)
        if hasattr(scaler, 'scale_'):
            params['scale'] = scaler.scale_.tolist() if hasattr(scaler.scale_, 'tolist') else float(scaler.scale_)
        if hasattr(scaler, 'data_min_'):
            params['data_min'] = scaler.data_min_.tolist() if hasattr(scaler.data_min_, 'tolist') else float(scaler.data_min_)
        if hasattr(scaler, 'data_max_'):
            params['data_max'] = scaler.data_max_.tolist() if hasattr(scaler.data_max_, 'tolist') else float(scaler.data_max_)
        if hasattr(scaler, 'center_'):
            params['center'] = scaler.center_.tolist() if hasattr(scaler.center_, 'tolist') else float(scaler.center_)
        
        return params
    
    def inverse_transform_numerical(self, data: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Inverse transform normalized numerical data.
        
        Args:
            data: Normalized data
            columns: Specific columns to inverse transform (None for all)
            
        Returns:
            DataFrame with inverse transformed data
        """
        if columns is None:
            columns = [col for col in data.columns if col in self.scalers]
        
        inverse_data = data.copy()
        
        for column in columns:
            if column in self.scalers and column in inverse_data.columns:
                inverse_data[[column]] = self.scalers[column].inverse_transform(inverse_data[[column]])
        
        return inverse_data
    
    def save_scalers(self, filepath: str):
        """Save all scalers and encoders to a file."""
        scalers_data = {
            'scalers': self.scalers,
            'encoders': self.encoders,
            'numerical_method': self.numerical_method,
            'categorical_method': self.categorical_method
        }
        
        joblib.dump(scalers_data, filepath)
        logger.info(f"Scalers and encoders saved to {filepath}")
    
    def load_scalers(self, filepath: str):
        """Load scalers and encoders from a file."""
        scalers_data = joblib.load(filepath)
        
        self.scalers = scalers_data['scalers']
        self.encoders = scalers_data['encoders']
        self.numerical_method = scalers_data['numerical_method']
        self.categorical_method = scalers_data['categorical_method']
        
        logger.info(f"Scalers and encoders loaded from {filepath}")
    
    def get_normalization_report(self) -> Dict[str, Any]:
        """Get the normalization report."""
        return self.normalization_report
    
    def save_normalization_report(self, filepath: str):
        """Save normalization report to a file."""
        import json
        
        # Convert numpy types to native Python types for JSON serialization
        def convert_numpy_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (list, tuple)):
                return [convert_numpy_types(item) for item in obj]
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            return obj
        
        # Create a JSON-serializable version of the report
        serializable_report = convert_numpy_types(self.normalization_report)
        
        with open(filepath, 'w') as f:
            json.dump(serializable_report, f, indent=2)
        
        logger.info(f"Normalization report saved to {filepath}") 