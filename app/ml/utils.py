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
    
    
    X = np.random.randn(n_samples, n_features)
    
    
    weights = np.random.randn(n_features)
    linear_combination = np.dot(X, weights)
    
    
    if n_classes == 2:
        y = (linear_combination > 0).astype(int)
    else:
        percentiles = [100 * i / n_classes for i in range(1, n_classes)]
        thresholds = np.percentile(linear_combination, percentiles)
        y = np.zeros(n_samples, dtype=int)
        for i, threshold in enumerate(thresholds):
            y[linear_combination >= threshold] = i + 1
    
    return X, y


def extract_features_from_survey_response(survey_response: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract features from survey response for ML model.
    
    Args:
        survey_response: Dictionary containing survey answers
        
    Returns:
        Dictionary of extracted features
    """
    features = {}
    
    # Extract features from survey response
    for key, value in survey_response.items():
        if isinstance(value, (int, float)):
            features[key] = value
        elif isinstance(value, str):
            # Convert string to numerical representation
            features[f"{key}_hash"] = hash(value) % 100
            features[f"{key}_length"] = len(value)
        elif isinstance(value, bool):
            features[key] = int(value)
        elif isinstance(value, list):
            # List-based features
            features[f"{key}_count"] = len(value)
            # Convert list items to features if they are strings
            for i, item in enumerate(value[:5]):  # Limit to first 5 items
                if isinstance(item, str):
                    features[f"{key}_item_{i}_hash"] = hash(str(item)) % 100
        else:
            # For other types, try to convert to string then hash
            features[f"{key}_hash"] = hash(str(value)) % 100
    
    return features


def extract_features_as_array(survey_response: Dict[str, Any]) -> np.ndarray:
    """
    Extract features as numpy array (for ML models that need arrays).
    
    Args:
        survey_response: Dictionary containing survey answers
        
    Returns:
        Numpy array of numerical features
    """
    features_dict = extract_features_from_survey_response(survey_response)
    features_list = list(features_dict.values())
    
    if not features_list:
        return np.array([]).reshape(1, -1)
    
    return np.array(features_list).reshape(1, -1) 