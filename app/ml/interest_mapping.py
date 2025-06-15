"""
Interest Mapping for Recommendation Engine

This module maps user interests to program keywords for improved matching.
"""

def get_interest_map(survey_data):
    """
    Create a mapping of interest keywords to interest levels based on survey data.
    
    Args:
        survey_data: Dictionary containing survey responses
        
    Returns:
        Dictionary mapping interest keywords to interest levels
    """
    math_interest = survey_data.get('math_interest', 0)
    science_interest = survey_data.get('science_interest', 0)
    art_interest = survey_data.get('art_interest', 0)
    sports_interest = survey_data.get('sports_interest', 0)
    
    # Map interests to related keywords
    return {
        'math': math_interest,
        'technology': math_interest,
        'programming': math_interest,
        'science': science_interest,
        'biology': science_interest,
        'physics': science_interest,
        'art': art_interest,
        'design': art_interest,
        'creative': art_interest,
        'sports': sports_interest,
        'physical': sports_interest
    } 