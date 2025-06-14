# Data preprocessing pipeline for machine learning training

import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
import json
from .validators import DataValidator


class StandardScaler:
    """Standard scaler for feature normalization (z-score normalization)."""
    
    def __init__(self):
        """Initialize standard scaler."""
        self.mean_ = None
        self.std_ = None
        self.is_fitted = False
        
    def fit(self, X: np.ndarray) -> 'StandardScaler':
        """
        Compute mean and standard deviation for scaling.
        
        Args:
            X: Input data to fit the scaler on
            
        Returns:
            Self for method chaining
        """
        validator = DataValidator()
        validator.validate_input(X)
        
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)
        
        # Handle zero standard deviation
        self.std_[self.std_ == 0] = 1.0
        
        self.is_fitted = True
        return self
        
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Scale features using previously computed mean and std.
        
        Args:
            X: Input data to transform
            
        Returns:
            Scaled data
        """
        if not self.is_fitted:
            raise ValueError("Scaler has not been fitted yet")
            
        validator = DataValidator()
        validator.validate_input(X)
        
        return (X - self.mean_) / self.std_
        
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(X).transform(X)
        
    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Undo the scaling transformation.
        
        Args:
            X: Scaled data to inverse transform
            
        Returns:
            Original scale data
        """
        if not self.is_fitted:
            raise ValueError("Scaler has not been fitted yet")
            
        return X * self.std_ + self.mean_


class MinMaxScaler:
    """Min-max scaler for feature normalization to [0, 1] range."""
    
    def __init__(self, feature_range: Tuple[float, float] = (0.0, 1.0)):
        """
        Initialize min-max scaler.
        
        Args:
            feature_range: Target range for scaling
        """
        self.feature_range = feature_range
        self.min_ = None
        self.max_ = None
        self.scale_ = None
        self.is_fitted = False
        
    def fit(self, X: np.ndarray) -> 'MinMaxScaler':
        """
        Compute min and max values for scaling.
        
        Args:
            X: Input data to fit the scaler on
            
        Returns:
            Self for method chaining
        """
        validator = DataValidator()
        validator.validate_input(X)
        
        self.min_ = np.min(X, axis=0)
        self.max_ = np.max(X, axis=0)
        
        # Compute scale
        data_range = self.max_ - self.min_
        data_range[data_range == 0] = 1.0  # Handle constant features
        
        feature_min, feature_max = self.feature_range
        self.scale_ = (feature_max - feature_min) / data_range
        
        self.is_fitted = True
        return self
        
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Scale features to the specified range."""
        if not self.is_fitted:
            raise ValueError("Scaler has not been fitted yet")
            
        validator = DataValidator()
        validator.validate_input(X)
        
        feature_min = self.feature_range[0]
        X_scaled = (X - self.min_) * self.scale_ + feature_min
        
        return X_scaled
        
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(X).transform(X)
        
    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Undo the min-max scaling transformation."""
        if not self.is_fitted:
            raise ValueError("Scaler has not been fitted yet")
            
        feature_min = self.feature_range[0]
        X_original = (X - feature_min) / self.scale_ + self.min_
        
        return X_original


class RobustScaler:
    """Robust scaler using median and interquartile range."""
    
    def __init__(self):
        """Initialize robust scaler."""
        self.median_ = None
        self.iqr_ = None
        self.is_fitted = False
        
    def fit(self, X: np.ndarray) -> 'RobustScaler':
        """
        Compute median and IQR for scaling.
        
        Args:
            X: Input data to fit the scaler on
            
        Returns:
            Self for method chaining
        """
        validator = DataValidator()
        validator.validate_input(X)
        
        self.median_ = np.median(X, axis=0)
        q1 = np.percentile(X, 25, axis=0)
        q3 = np.percentile(X, 75, axis=0)
        self.iqr_ = q3 - q1
        
        # Handle zero IQR
        self.iqr_[self.iqr_ == 0] = 1.0
        
        self.is_fitted = True
        return self
        
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Scale features using median and IQR."""
        if not self.is_fitted:
            raise ValueError("Scaler has not been fitted yet")
            
        validator = DataValidator()
        validator.validate_input(X)
        
        return (X - self.median_) / self.iqr_
        
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(X).transform(X)


class LabelEncoder:
    """Encode categorical labels as integers."""
    
    def __init__(self):
        """Initialize label encoder."""
        self.classes_ = None
        self.class_to_index = None
        self.is_fitted = False
        
    def fit(self, y: np.ndarray) -> 'LabelEncoder':
        """
        Fit label encoder on categorical labels.
        
        Args:
            y: Categorical labels to encode
            
        Returns:
            Self for method chaining
        """
        if len(y.shape) != 1:
            raise ValueError("Label encoder expects 1D array")
            
        self.classes_ = np.unique(y)
        self.class_to_index = {cls: idx for idx, cls in enumerate(self.classes_)}
        self.is_fitted = True
        
        return self
        
    def transform(self, y: np.ndarray) -> np.ndarray:
        """
        Transform categorical labels to integer encoding.
        
        Args:
            y: Categorical labels to transform
            
        Returns:
            Integer encoded labels
        """
        if not self.is_fitted:
            raise ValueError("Label encoder has not been fitted yet")
            
        if len(y.shape) != 1:
            raise ValueError("Label encoder expects 1D array")
            
        # Check for unknown classes
        unknown_classes = set(y) - set(self.classes_)
        if unknown_classes:
            raise ValueError(f"Unknown classes found: {unknown_classes}")
            
        return np.array([self.class_to_index[cls] for cls in y])
        
    def fit_transform(self, y: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(y).transform(y)
        
    def inverse_transform(self, y: np.ndarray) -> np.ndarray:
        """
        Transform integer encoded labels back to original labels.
        
        Args:
            y: Integer encoded labels
            
        Returns:
            Original categorical labels
        """
        if not self.is_fitted:
            raise ValueError("Label encoder has not been fitted yet")
            
        if len(y.shape) != 1:
            raise ValueError("Expected 1D array")
            
        # Check for invalid indices
        max_index = len(self.classes_) - 1
        if np.any(y < 0) or np.any(y > max_index):
            raise ValueError(f"Invalid label indices. Expected range [0, {max_index}]")
            
        return self.classes_[y]


class OneHotEncoder:
    """One-hot encode categorical features."""
    
    def __init__(self, sparse: bool = False):
        """
        Initialize one-hot encoder.
        
        Args:
            sparse: Whether to return sparse representation (not implemented)
        """
        self.sparse = sparse
        self.categories_ = None
        self.is_fitted = False
        
    def fit(self, X: np.ndarray) -> 'OneHotEncoder':
        """
        Fit one-hot encoder on categorical data.
        
        Args:
            X: Categorical data to fit encoder on
            
        Returns:
            Self for method chaining
        """
        if len(X.shape) != 2:
            raise ValueError("OneHotEncoder expects 2D array")
            
        self.categories_ = []
        for i in range(X.shape[1]):
            unique_values = np.unique(X[:, i])
            self.categories_.append(unique_values)
            
        self.is_fitted = True
        return self
        
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform categorical data to one-hot encoding.
        
        Args:
            X: Categorical data to transform
            
        Returns:
            One-hot encoded data
        """
        if not self.is_fitted:
            raise ValueError("OneHotEncoder has not been fitted yet")
            
        if len(X.shape) != 2:
            raise ValueError("OneHotEncoder expects 2D array")
            
        if X.shape[1] != len(self.categories_):
            raise ValueError(f"Expected {len(self.categories_)} features, got {X.shape[1]}")
            
        encoded_features = []
        
        for i in range(X.shape[1]):
            feature_values = X[:, i]
            categories = self.categories_[i]
            
            # Create one-hot encoding for this feature
            feature_encoded = np.zeros((X.shape[0], len(categories)))
            
            for j, category in enumerate(categories):
                mask = feature_values == category
                feature_encoded[mask, j] = 1
                
            encoded_features.append(feature_encoded)
            
        return np.concatenate(encoded_features, axis=1)
        
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(X).transform(X)
        
    def get_feature_names(self, input_features: Optional[List[str]] = None) -> List[str]:
        """
        Get feature names for one-hot encoded features.
        
        Args:
            input_features: Names of input features
            
        Returns:
            List of feature names for encoded features
        """
        if not self.is_fitted:
            raise ValueError("OneHotEncoder has not been fitted yet")
            
        if input_features is None:
            input_features = [f'feature_{i}' for i in range(len(self.categories_))]
            
        feature_names = []
        for i, categories in enumerate(self.categories_):
            for category in categories:
                feature_names.append(f'{input_features[i]}_{category}')
                
        return feature_names


class DataPreprocessor:
    """Complete data preprocessing pipeline."""
    
    def __init__(self, 
                 scaling_method: str = 'standard',
                 handle_missing: str = 'mean',
                 encode_categorical: bool = True,
                 remove_outliers: bool = False,
                 outlier_method: str = 'iqr'):
        """
        Initialize data preprocessor.
        
        Args:
            scaling_method: Method for scaling ('standard', 'minmax', 'robust', 'none')
            handle_missing: How to handle missing values ('mean', 'median', 'drop', 'none')
            encode_categorical: Whether to encode categorical features
            remove_outliers: Whether to remove outliers
            outlier_method: Method for outlier detection ('iqr', 'zscore')
        """
        self.scaling_method = scaling_method
        self.handle_missing = handle_missing
        self.encode_categorical = encode_categorical
        self.remove_outliers = remove_outliers
        self.outlier_method = outlier_method
        
        # Initialize components
        self.scaler = None
        self.label_encoder = None
        self.categorical_encoders = {}
        self.missing_values = {}
        self.is_fitted = False
        
    def _initialize_scaler(self):
        """Initialize the appropriate scaler."""
        if self.scaling_method == 'standard':
            self.scaler = StandardScaler()
        elif self.scaling_method == 'minmax':
            self.scaler = MinMaxScaler()
        elif self.scaling_method == 'robust':
            self.scaler = RobustScaler()
        elif self.scaling_method == 'none':
            self.scaler = None
        else:
            raise ValueError(f"Unknown scaling method: {self.scaling_method}")
            
    def _handle_missing_values(self, X: np.ndarray, is_training: bool = True) -> np.ndarray:
        """Handle missing values in the data."""
        if self.handle_missing == 'none':
            return X
            
        X_processed = X.copy()
        
        for i in range(X.shape[1]):
            feature_data = X_processed[:, i]
            missing_mask = np.isnan(feature_data)
            
            if not np.any(missing_mask):
                continue
                
            if is_training:
                if self.handle_missing == 'mean':
                    fill_value = np.nanmean(feature_data)
                elif self.handle_missing == 'median':
                    fill_value = np.nanmedian(feature_data)
                else:
                    continue
                    
                self.missing_values[i] = fill_value
            else:
                fill_value = self.missing_values.get(i, 0)
                
            X_processed[missing_mask, i] = fill_value
            
        return X_processed
        
    def _remove_outliers_from_data(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Remove outliers from the data."""
        validator = DataValidator()
        outlier_mask = validator.detect_outliers(X, method=self.outlier_method)
        
        # Keep non-outlier samples
        clean_indices = ~outlier_mask
        
        return X[clean_indices], y[clean_indices]
        
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'DataPreprocessor':
        """
        Fit the preprocessor on training data.
        
        Args:
            X: Training features
            y: Training labels (optional)
            
        Returns:
            Self for method chaining
        """
        validator = DataValidator()
        validator.validate_input(X)
        
        X_processed = X.copy()
        
        # Handle missing values
        X_processed = self._handle_missing_values(X_processed, is_training=True)
        
        # Remove outliers if requested
        if self.remove_outliers and y is not None:
            X_processed, y_processed = self._remove_outliers_from_data(X_processed, y)
        
        # Initialize and fit scaler
        self._initialize_scaler()
        if self.scaler is not None:
            self.scaler.fit(X_processed)
            
        # Fit label encoder if provided
        if y is not None and self.encode_categorical:
            if len(y.shape) == 1 and not np.issubdtype(y.dtype, np.number):
                self.label_encoder = LabelEncoder()
                self.label_encoder.fit(y)
                
        self.is_fitted = True
        return self
        
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform data using fitted preprocessor.
        
        Args:
            X: Data to transform
            
        Returns:
            Transformed data
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor has not been fitted yet")
            
        validator = DataValidator()
        validator.validate_input(X)
        
        X_processed = X.copy()
        
        # Handle missing values
        X_processed = self._handle_missing_values(X_processed, is_training=False)
        
        # Apply scaling
        if self.scaler is not None:
            X_processed = self.scaler.transform(X_processed)
            
        return X_processed
        
    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Fit and transform in one step."""
        self.fit(X, y)
        X_transformed = self.transform(X)
        
        if y is not None and self.label_encoder is not None:
            y_transformed = self.label_encoder.transform(y)
            return X_transformed, y_transformed
            
        return X_transformed
        
    def get_preprocessing_info(self) -> Dict[str, Any]:
        """Get information about the preprocessing pipeline."""
        info = {
            'scaling_method': self.scaling_method,
            'handle_missing': self.handle_missing,
            'encode_categorical': self.encode_categorical,
            'remove_outliers': self.remove_outliers,
            'is_fitted': self.is_fitted
        }
        
        if self.is_fitted:
            info['missing_values_found'] = len(self.missing_values)
            info['has_label_encoder'] = self.label_encoder is not None
            
            if self.scaler is not None:
                info['scaler_type'] = type(self.scaler).__name__
                
        return info 