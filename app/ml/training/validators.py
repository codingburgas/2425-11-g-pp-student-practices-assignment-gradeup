# Data validation utilities for the training system

import numpy as np
from typing import List, Dict, Any, Tuple, Optional, Union
from .base import BaseValidator


class DataValidator(BaseValidator):
    """Comprehensive data validator for machine learning training."""
    
    def __init__(self, 
                 min_samples: int = 10,
                 max_features: int = 1000,
                 allow_missing: bool = False,
                 missing_threshold: float = 0.1):
        """
        Initialize data validator.
        
        Args:
            min_samples: Minimum number of samples required
            max_features: Maximum number of features allowed
            allow_missing: Whether to allow missing values
            missing_threshold: Maximum fraction of missing values allowed
        """
        self.min_samples = min_samples
        self.max_features = max_features
        self.allow_missing = allow_missing
        self.missing_threshold = missing_threshold
        
    def validate_input(self, data: np.ndarray) -> bool:
        """
        Validate input data array.
        
        Args:
            data: Input data array to validate
            
        Returns:
            bool: True if data is valid, False otherwise
            
        Raises:
            ValueError: If data fails validation with detailed error message
        """
        if not isinstance(data, np.ndarray):
            raise ValueError("Input data must be a numpy array")
            
        if data.size == 0:
            raise ValueError("Input data cannot be empty")
            
        if len(data.shape) != 2:
            raise ValueError(f"Input data must be 2D, got shape {data.shape}")
            
        n_samples, n_features = data.shape
        
        if n_samples < self.min_samples:
            raise ValueError(f"Not enough samples: {n_samples} < {self.min_samples}")
            
        if n_features > self.max_features:
            raise ValueError(f"Too many features: {n_features} > {self.max_features}")
            
        # Check for missing values
        if not self.allow_missing and np.any(np.isnan(data)):
            raise ValueError("Missing values detected but not allowed")
            
        if self.allow_missing:
            missing_ratio = np.sum(np.isnan(data)) / data.size
            if missing_ratio > self.missing_threshold:
                raise ValueError(f"Too many missing values: {missing_ratio:.2%} > {self.missing_threshold:.2%}")
                
        # Check for infinite values
        if np.any(np.isinf(data)):
            raise ValueError("Infinite values detected in input data")
            
        # Check data type
        if not np.issubdtype(data.dtype, np.number):
            raise ValueError(f"Data must be numeric, got {data.dtype}")
            
        return True
        
    def validate_labels(self, labels: np.ndarray) -> bool:
        """
        Validate label data array.
        
        Args:
            labels: Label array to validate
            
        Returns:
            bool: True if labels are valid, False otherwise
            
        Raises:
            ValueError: If labels fail validation with detailed error message
        """
        if not isinstance(labels, np.ndarray):
            raise ValueError("Labels must be a numpy array")
            
        if labels.size == 0:
            raise ValueError("Labels cannot be empty")
            
        if len(labels.shape) > 2:
            raise ValueError(f"Labels must be 1D or 2D, got shape {labels.shape}")
            
        # Check for missing values
        if np.any(np.isnan(labels)):
            raise ValueError("Missing values detected in labels")
            
        # Check for infinite values
        if np.any(np.isinf(labels)):
            raise ValueError("Infinite values detected in labels")
            
        return True
        
    def validate_data_consistency(self, X: np.ndarray, y: np.ndarray) -> bool:
        """
        Validate consistency between input data and labels.
        
        Args:
            X: Input data array
            y: Label array
            
        Returns:
            bool: True if data is consistent, False otherwise
            
        Raises:
            ValueError: If data is inconsistent
        """
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"Sample count mismatch: X has {X.shape[0]} samples, y has {y.shape[0]}")
            
        return True
        
    def detect_outliers(self, data: np.ndarray, method: str = 'iqr', threshold: float = 3.0) -> np.ndarray:
        """
        Detect outliers in the data.
        
        Args:
            data: Input data array
            method: Outlier detection method ('iqr', 'zscore')
            threshold: Threshold for outlier detection
            
        Returns:
            np.ndarray: Boolean mask indicating outliers
        """
        if method == 'iqr':
            Q1 = np.percentile(data, 25, axis=0)
            Q3 = np.percentile(data, 75, axis=0)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = (data < lower_bound) | (data > upper_bound)
            return np.any(outliers, axis=1)
            
        elif method == 'zscore':
            z_scores = np.abs((data - np.mean(data, axis=0)) / np.std(data, axis=0))
            return np.any(z_scores > threshold, axis=1)
            
        else:
            raise ValueError(f"Unknown outlier detection method: {method}")
            
    def get_data_summary(self, data: np.ndarray) -> Dict[str, Any]:
        """
        Get comprehensive summary statistics for the data.
        
        Args:
            data: Input data array
            
        Returns:
            Dict containing summary statistics
        """
        summary = {
            'shape': data.shape,
            'dtype': str(data.dtype),
            'memory_usage_mb': data.nbytes / (1024 * 1024),
            'missing_values': int(np.sum(np.isnan(data))),
            'missing_percentage': float(np.sum(np.isnan(data)) / data.size * 100),
            'infinite_values': int(np.sum(np.isinf(data))),
            'features': {}
        }
        
        # Per-feature statistics
        for i in range(data.shape[1]):
            feature_data = data[:, i]
            valid_data = feature_data[~np.isnan(feature_data) & ~np.isinf(feature_data)]
            
            if len(valid_data) > 0:
                summary['features'][f'feature_{i}'] = {
                    'mean': float(np.mean(valid_data)),
                    'std': float(np.std(valid_data)),
                    'min': float(np.min(valid_data)),
                    'max': float(np.max(valid_data)),
                    'median': float(np.median(valid_data)),
                    'unique_values': int(len(np.unique(valid_data))),
                    'missing_count': int(np.sum(np.isnan(feature_data)))
                }
                
        return summary
        
    def check_class_distribution(self, labels: np.ndarray) -> Dict[str, Any]:
        """
        Check class distribution for classification problems.
        
        Args:
            labels: Label array
            
        Returns:
            Dict containing class distribution information
        """
        if len(labels.shape) == 1:
            # For 1D labels (single-class classification)
            unique_labels, counts = np.unique(labels, return_counts=True)
            total_samples = len(labels)
            
            distribution = {
                'num_classes': len(unique_labels),
                'class_counts': dict(zip(unique_labels.astype(str), counts.astype(int))),
                'class_percentages': dict(zip(unique_labels.astype(str), (counts / total_samples * 100).astype(float))),
                'is_balanced': np.std(counts) / np.mean(counts) < 0.1,  # CV < 10% considered balanced
                'imbalance_ratio': float(np.max(counts) / np.min(counts))
            }
            
        else:
            # For 2D labels (multi-class or multi-label classification)
            distribution = {
                'num_classes': labels.shape[1],
                'samples_per_class': {},
                'class_percentages': {},
                'is_balanced': True,
                'imbalance_ratio': 1.0
            }
            
            for i in range(labels.shape[1]):
                class_samples = np.sum(labels[:, i])
                class_percentage = class_samples / labels.shape[0] * 100
                distribution['samples_per_class'][f'class_{i}'] = int(class_samples)
                distribution['class_percentages'][f'class_{i}'] = float(class_percentage)
                
        return distribution


class ParameterValidator:
    """Validator for training parameters and hyperparameters."""
    
    @staticmethod
    def validate_learning_rate(lr: float) -> bool:
        """Validate learning rate parameter."""
        if not isinstance(lr, (int, float)):
            raise ValueError("Learning rate must be numeric")
        if lr <= 0 or lr > 1:
            raise ValueError(f"Learning rate must be in (0, 1], got {lr}")
        return True
        
    @staticmethod
    def validate_epochs(epochs: int) -> bool:
        """Validate number of epochs."""
        if not isinstance(epochs, int):
            raise ValueError("Epochs must be an integer")
        if epochs < 1:
            raise ValueError(f"Epochs must be positive, got {epochs}")
        if epochs > 10000:
            raise ValueError(f"Too many epochs: {epochs} > 10000")
        return True
        
    @staticmethod
    def validate_batch_size(batch_size: int, total_samples: int) -> bool:
        """Validate batch size."""
        if not isinstance(batch_size, int):
            raise ValueError("Batch size must be an integer")
        if batch_size < 1:
            raise ValueError(f"Batch size must be positive, got {batch_size}")
        if batch_size > total_samples:
            raise ValueError(f"Batch size ({batch_size}) cannot be larger than total samples ({total_samples})")
        return True
        
    @staticmethod
    def validate_split_ratio(ratio: float, name: str = "split ratio") -> bool:
        """Validate data split ratio."""
        if not isinstance(ratio, (int, float)):
            raise ValueError(f"{name} must be numeric")
        if ratio < 0 or ratio >= 1:
            raise ValueError(f"{name} must be in [0, 1), got {ratio}")
        return True
        
    @staticmethod
    def validate_random_seed(seed: Optional[int]) -> bool:
        """Validate random seed."""
        if seed is not None:
            if not isinstance(seed, int):
                raise ValueError("Random seed must be an integer or None")
            if seed < 0:
                raise ValueError(f"Random seed must be non-negative, got {seed}")
        return True 