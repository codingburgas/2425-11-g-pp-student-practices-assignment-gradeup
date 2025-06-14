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


"""
Utility functions for the recommendation engine
"""

def normalize_score(score: float, min_score: float = 0.0, max_score: float = 1.0) -> float:
    """Normalize a score to a specific range"""
    return max(min_score, min(max_score, score))

def calculate_weighted_average(scores: list, weights: list) -> float:
    """Calculate weighted average of scores"""
    if not scores or not weights or len(scores) != len(weights):
        return 0.0
    
    total_weight = sum(weights)
    if total_weight == 0:
        return 0.0
    
    weighted_sum = sum(score * weight for score, weight in zip(scores, weights))
    return weighted_sum / total_weight

def format_match_percentage(score: float) -> str:
    """Format match score as percentage"""
    return f"{int(score * 100)}%"

def map_survey_data_to_recommendation_format(survey_data):
    """
    Map actual survey responses to the format expected by the recommendation engine.
    
    Args:
        survey_data: Dict with keys '1', '2', etc. from actual survey
        
    Returns:
        Dict with standardized field names for recommendation engine
    """
    
    # Initialize with default values
    mapped_data = {
        'math_interest': 5,  # Default neutral
        'science_interest': 5,
        'art_interest': 5,
        'sports_interest': 5,
        'study_hours_per_day': 4,
        'preferred_subject': 'General',
        'career_goal': 'Professional',
        'extracurricular': True,
        'leadership_experience': False,
        'team_preference': True,
        'languages_spoken': ['Bulgarian', 'English'],
        'grades_average': 5.0
    }
    
    try:
        # Q1: What subjects do you enjoy the most?
        if '1' in survey_data:
            subject = survey_data['1']
            mapped_data['preferred_subject'] = subject
            
            # Map subject interest to numeric scores
            if subject in ['Mathematics', 'Physics', 'Computer Science']:
                mapped_data['math_interest'] = 8
                mapped_data['science_interest'] = 8
            elif subject in ['Chemistry', 'Biology']:
                mapped_data['science_interest'] = 8
                mapped_data['math_interest'] = 6
            elif subject in ['Arts', 'Literature']:
                mapped_data['art_interest'] = 8
                mapped_data['science_interest'] = 3
            elif subject == 'Physics':
                mapped_data['science_interest'] = 9
                mapped_data['math_interest'] = 8
        
        # Q2: What type of career are you interested in?
        if '2' in survey_data:
            career = survey_data['2']
            mapped_data['career_goal'] = career
            
            # Adjust interests based on career choice
            if career in ['Technology', 'Engineering']:
                mapped_data['math_interest'] = max(mapped_data['math_interest'], 7)
                mapped_data['science_interest'] = max(mapped_data['science_interest'], 7)
            elif career in ['Science', 'Medicine']:
                mapped_data['science_interest'] = max(mapped_data['science_interest'], 8)
            elif career in ['Arts']:
                mapped_data['art_interest'] = max(mapped_data['art_interest'], 8)
            elif career == 'Education':
                # Education career - moderate interests across subjects
                mapped_data['math_interest'] = 6
                mapped_data['science_interest'] = 6
                mapped_data['art_interest'] = 6
        
        # Q8: Average grades
        if '8' in survey_data:
            grades = survey_data['8']
            if grades == 'Below 4.0':
                mapped_data['grades_average'] = 3.5
            elif grades == '4.0-4.5':
                mapped_data['grades_average'] = 4.25
            elif grades == '4.5-5.0':
                mapped_data['grades_average'] = 4.75
            elif grades == '5.0-5.5':
                mapped_data['grades_average'] = 5.25
            elif grades == '5.5-6.0':
                mapped_data['grades_average'] = 5.75
        
        # Q9: Extracurricular activities
        if '9' in survey_data:
            activities = survey_data['9'] if isinstance(survey_data['9'], list) else [survey_data['9']]
            mapped_data['extracurricular'] = len(activities) > 0 and 'None' not in activities
            
            # Determine interests from activities
            if 'Sports' in activities:
                mapped_data['sports_interest'] = 8
            if 'Art' in activities or 'Music' in activities:
                mapped_data['art_interest'] = max(mapped_data['art_interest'], 7)
            if 'Programming' in activities:
                mapped_data['math_interest'] = max(mapped_data['math_interest'], 8)
                mapped_data['science_interest'] = max(mapped_data['science_interest'], 7)
            if 'Student Government' in activities or 'Debate' in activities:
                mapped_data['leadership_experience'] = True
        
        # Q10: Skills to develop
        if '10' in survey_data:
            skills = survey_data['10'] if isinstance(survey_data['10'], list) else [survey_data['10']]
            if 'Leadership' in skills:
                mapped_data['leadership_experience'] = True
            if 'Teamwork' in skills:
                mapped_data['team_preference'] = True
            if 'Technical Skills' in skills:
                mapped_data['math_interest'] = max(mapped_data['math_interest'], 7)
                mapped_data['science_interest'] = max(mapped_data['science_interest'], 7)
        
        # Q7: Teaching language
        if '7' in survey_data:
            language = survey_data['7']
            if language == 'English':
                mapped_data['languages_spoken'] = ['Bulgarian', 'English']
            elif language == 'Bulgarian':
                mapped_data['languages_spoken'] = ['Bulgarian']
            else:
                mapped_data['languages_spoken'] = ['Bulgarian', 'English', 'Other']
        
        # Estimate study hours based on grades and activities
        if mapped_data['grades_average'] >= 5.0 and mapped_data['extracurricular']:
            mapped_data['study_hours_per_day'] = 6
        elif mapped_data['grades_average'] >= 5.0:
            mapped_data['study_hours_per_day'] = 5
        elif mapped_data['extracurricular']:
            mapped_data['study_hours_per_day'] = 4
        else:
            mapped_data['study_hours_per_day'] = 3
            
    except Exception as e:
        print(f"Warning: Error mapping survey data: {e}")
        # Return default data if mapping fails
        pass
    
    return mapped_data 