"""
Confidence Scoring for Program Recommendations

This module implements the improved percentage-based confidence scoring system
to provide more realistic and varied recommendation confidence scores.
"""

import random
from typing import Dict, Any, List


def calculate_confidence_score(base_score: float) -> int:
    """
    Calculate a confidence score as a percentage from a base score.
    
    Args:
        base_score: Raw score from keyword matching
        
    Returns:
        Integer confidence percentage (0-100)
    """
    # Normalize to percentage
    confidence_decimal = min(0.95, max(0.1, base_score / 15))
    confidence_percent = round(confidence_decimal * 100)
    
    # Add slight variance
    confidence_variance = random.randint(-3, 3)
    final_confidence = max(10, min(95, confidence_percent + confidence_variance))
    
    return final_confidence


def apply_confidence_scoring(program_scores: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Apply confidence scoring to a list of program scores.
    
    Args:
        program_scores: List of program scores
        
    Returns:
        List with updated confidence scores
    """
    result = []
    
    for item in program_scores:
        program = item['program']
        confidence = item['confidence']
        
        # Apply variance for more realistic results
        final_confidence = calculate_confidence_score(confidence)
        
        prediction = {
            'program_id': program['id'],
            'program_name': program['name'],
            'school_name': program['school_name'],
            'confidence': final_confidence,
            'match_score': final_confidence,  # Added for compatibility
            'degree_type': program.get('degree_type', 'Bachelor')
        }
        
        result.append(prediction)
    
    return result 