"""
Match Reason Generation for Program Recommendations

This module provides functions to generate informative match reasons
for program recommendations based on user interests and program attributes.
"""

from typing import Dict, List, Any


def generate_match_reasons(program: Dict[str, Any], 
                          user_interests: List[str], 
                          interest_map: Dict[str, int]) -> List[str]:
    """
    Generate personalized match reasons for a program recommendation.
    
    Args:
        program: Program information dictionary
        user_interests: List of user's expressed interests
        interest_map: Mapping of interest keywords to interest levels
        
    Returns:
        List of match reasons for this program
    """
    reasons = []
    
    # Check for keyword matches in interests
    for keyword in program.get('keywords', []):
        for interest in user_interests:
            if keyword in interest or interest in keyword:
                reasons.append(f"Matches your interest in {keyword.title()}")
                break
    
    # Add interest-based reasons
    high_interests = [(k, v) for k, v in interest_map.items() if v >= 7]
    for interest, level in high_interests:
        if any(interest in kw.lower() for kw in program.get('keywords', [])):
            reasons.append(f"Matches your high interest in {interest.title()}")
    
    # Add program-specific reasons
    program_name = program.get('name', '').lower()
    
    if 'computer science' in program_name or 'software' in program_name:
        reasons.append("Suitable for high-achieving students")
    
    if 'engineering' in program_name:
        reasons.append("Suitable for high-achieving students")
    
    if 'medicine' in program_name:
        reasons.append("Suitable for high-achieving students")
    
    # Ensure we have at least one reason
    if not reasons:
        reasons.append("Suitable for high-achieving students")
    
    return reasons[:3]  # Limit to top 3 reasons 