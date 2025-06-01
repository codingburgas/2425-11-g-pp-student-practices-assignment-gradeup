"""
ML Utility Functions

This module contains utility functions for data generation, feature extraction,
and other helper functions.
"""

import numpy as np
from typing import Tuple, Dict, Any


def create_sample_dataset(n_samples: int = 1000, n_features: int = 10, n_classes: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    """Create a sample dataset for testing the ML pipeline."""
    np.random.seed(42)
    
    # Generate synthetic features
    X = np.random.randn(n_samples, n_features)
    
    # Create synthetic target based on feature combinations
    weights = np.random.randn(n_features)
    linear_combination = np.dot(X, weights)
    
    # Convert to class labels
    if n_classes == 2:
        y = (linear_combination > 0).astype(int)
    else:
        percentiles = [100 * i / n_classes for i in range(1, n_classes)]
        thresholds = np.percentile(linear_combination, percentiles)
        y = np.zeros(n_samples, dtype=int)
        for i, threshold in enumerate(thresholds):
            y[linear_combination >= threshold] = i + 1
    
    return X, y


def extract_features_from_survey_response(survey_response: Dict[str, Any]) -> np.ndarray:
    """
    Extract numerical features from survey response for ML model.
    
    Args:
        survey_response: Dictionary containing survey answers
        
    Returns:
        Numpy array of numerical features
    """
    features = []
    
    # Example feature extraction (customize based on actual survey structure)
    for key, value in survey_response.items():
        if isinstance(value, (int, float)):
            features.append(value)
        elif isinstance(value, str):
            # Convert categorical to numerical (simple hash-based encoding)
            features.append(hash(value) % 100)
        elif isinstance(value, bool):
            features.append(int(value))
        elif isinstance(value, list):
            # For multiple choice questions
            features.append(len(value))
    
    return np.array(features).reshape(1, -1) 