"""
Demo Prediction Service

This service provides mock predictions when no trained ML models are available.
Perfect for testing and demonstration purposes.
"""

import random
import numpy as np
from typing import Dict, List, Any
from datetime import datetime


class DemoPredictionService:
    """Demo service that generates realistic mock predictions."""
    
    def __init__(self):
        self.demo_programs = [
            {
                'id': 1,
                'name': 'Computer Science',
                'school_name': 'Sofia University St. Kliment Ohridski',
                'degree_type': 'Bachelor',
                'keywords': ['technology', 'programming', 'software', 'computer', 'math', 'science']
            },
            {
                'id': 2,
                'name': 'Electrical Engineering',
                'school_name': 'Technical University of Sofia',
                'degree_type': 'Bachelor',
                'keywords': ['engineering', 'electricity', 'physics', 'technology', 'math', 'science']
            },
            {
                'id': 3,
                'name': 'Medicine',
                'school_name': 'Medical University of Sofia',
                'degree_type': 'Master',
                'keywords': ['medicine', 'biology', 'health', 'science', 'healthcare']
            },
            {
                'id': 4,
                'name': 'Business Administration',
                'school_name': 'New Bulgarian University',
                'degree_type': 'Bachelor',
                'keywords': ['business', 'management', 'leadership', 'economics', 'finance']
            },
            {
                'id': 5,
                'name': 'Psychology',
                'school_name': 'Plovdiv University Paisii Hilendarski',
                'degree_type': 'Bachelor',
                'keywords': ['psychology', 'mental_health', 'research', 'helping', 'social']
            },
            {
                'id': 6,
                'name': 'Economics',
                'school_name': 'University of National and World Economy',
                'degree_type': 'Bachelor',
                'keywords': ['economics', 'finance', 'business', 'mathematics', 'statistics']
            },
            {
                'id': 7,
                'name': 'Fine Arts',
                'school_name': 'National Academy of Arts',
                'degree_type': 'Bachelor',
                'keywords': ['art', 'design', 'creative', 'drawing', 'painting', 'sculpture']
            },
            {
                'id': 8,
                'name': 'Mechanical Engineering',
                'school_name': 'Technical University of Sofia',
                'degree_type': 'Bachelor',
                'keywords': ['engineering', 'mechanical', 'design', 'technology', 'physics']
            },
            {
                'id': 9,
                'name': 'Law',
                'school_name': 'Sofia University St. Kliment Ohridski',
                'degree_type': 'Master',
                'keywords': ['law', 'legal', 'justice', 'social', 'politics']
            },
            {
                'id': 10,
                'name': 'Architecture',
                'school_name': 'University of Architecture, Civil Engineering and Geodesy',
                'degree_type': 'Master',
                'keywords': ['architecture', 'design', 'civil engineering', 'construction', 'art']
            },
            {
                'id': 11,
                'name': 'Computer Science',
                'school_name': 'New Bulgarian University',
                'degree_type': 'Bachelor',
                'keywords': ['technology', 'programming', 'software', 'computer', 'math', 'science']
            },
            {
                'id': 12,
                'name': 'Computer Science',
                'school_name': 'Plovdiv University Paisii Hilendarski',
                'degree_type': 'Bachelor',
                'keywords': ['technology', 'programming', 'software', 'computer', 'math', 'science']
            },
            {
                'id': 13,
                'name': 'Computer Science',
                'school_name': 'American University in Bulgaria',
                'degree_type': 'Bachelor',
                'keywords': ['technology', 'programming', 'software', 'computer', 'math', 'science']
            },
            {
                'id': 14,
                'name': 'Informatics and Computer Science',
                'school_name': 'Burgas Free University',
                'degree_type': 'Bachelor',
                'keywords': ['technology', 'programming', 'software', 'computer', 'math', 'science', 'informatics']
            },
            {
                'id': 15,
                'name': 'Computer Systems and Technologies',
                'school_name': 'Ruse University Angel Kanchev',
                'degree_type': 'Bachelor',
                'keywords': ['technology', 'programming', 'software', 'computer', 'systems', 'science']
            }
        ]
    
    def predict_programs(self, survey_data: Dict[str, Any], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Generate demo predictions based on survey data.
        
        Args:
            survey_data: Survey response data
            top_k: Number of predictions to return
            
        Returns:
            List of mock predictions with realistic confidence scores
        """
        predictions = []
        
        # Extract interests from survey data
        career_interests = survey_data.get('career_interests', [])
        favorite_subjects = survey_data.get('favorite_subjects', [])
        career_goals = survey_data.get('career_goals', '').lower()
        math_interest = survey_data.get('math_interest', 0)
        science_interest = survey_data.get('science_interest', 0)
        art_interest = survey_data.get('art_interest', 0)
        sports_interest = survey_data.get('sports_interest', 0)
        preferred_subject = survey_data.get('preferred_subject', '').lower()
        
        # Combine all interests for matching
        all_interests = []
        if isinstance(career_interests, list):
            all_interests.extend([str(interest).lower() for interest in career_interests])
        elif career_interests:
            all_interests.append(str(career_interests).lower())
            
        if isinstance(favorite_subjects, list):
            all_interests.extend([str(subject).lower() for subject in favorite_subjects])
        elif favorite_subjects:
            all_interests.append(str(favorite_subjects).lower())
        
        if preferred_subject:
            all_interests.append(preferred_subject)
            
        if career_goals:
            all_interests.append(career_goals)
        
        # Add interest values to adjust scoring
        interest_map = {
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
        
        # Score each program based on keyword matching and interests
        program_scores = []
        for program in self.demo_programs:
            score = 0
            
            # Check keyword matches
            for keyword in program['keywords']:
                # Add points for matching keywords in interests
                for interest in all_interests:
                    if keyword in interest or interest in keyword:
                        score += 2
                
                # Add points based on interest levels
                for interest_keyword, interest_level in interest_map.items():
                    if keyword == interest_keyword or interest_keyword in keyword:
                        score += (interest_level / 10) * 3  # Scale 0-10 to 0-3 bonus points
            
            # Base score should never be zero to avoid all-identical matches
            base_score = max(1, score)
            
            # Add some randomness for variety (but less than before)
            score = base_score + random.uniform(0, 1)
            
            # Normalize score to confidence percentage (max 95%)
            confidence_decimal = min(0.95, max(0.1, score / 15))
            confidence_percent = round(confidence_decimal * 100)
            
            program_scores.append({
                'program': program,
                'confidence': confidence_percent
            })
        
        # Sort by confidence and take top K
        program_scores.sort(key=lambda x: x['confidence'], reverse=True)
        
        for i, item in enumerate(program_scores[:top_k]):
            program = item['program']
            confidence = item['confidence']
            
            # Add slight variance to confidence to make it more realistic
            confidence_variance = random.randint(-3, 3)
            final_confidence = max(10, min(95, confidence + confidence_variance))
            
            prediction = {
                'program_id': program['id'],
                'program_name': program['name'],
                'school_name': program['school_name'],
                'confidence': final_confidence,
                'match_score': final_confidence,  # Added for compatibility
                'rank': i + 1,
                'degree_type': program['degree_type'],
                'match_reasons': self._generate_match_reasons(program, all_interests, interest_map)
            }
            
            predictions.append(prediction)
        
        return predictions
    
    def _generate_match_reasons(self, program: Dict, user_interests: List[str], interest_map: Dict[str, int]) -> List[str]:
        """Generate realistic match reasons for a program."""
        reasons = []
        
        # Check for keyword matches
        for keyword in program['keywords']:
            for interest in user_interests:
                if keyword in interest or interest in keyword:
                    reasons.append(f"Matches your interest in {keyword.title()}")
                    break
        
        # Add interest-based reasons
        high_interests = [(k, v) for k, v in interest_map.items() if v >= 7]
        for interest, level in high_interests:
            if interest in [kw.lower() for kw in program['keywords']]:
                reasons.append(f"Matches your high interest in {interest.title()}")
        
        # Add some specific reasons
        if program['name'] in ['Computer Science', 'Software Engineering', 'Informatics and Computer Science']:
            reasons.append("Suitable for high-achieving students")
        
        if 'engineering' in program['name'].lower():
            reasons.append("Suitable for high-achieving students")
        
        if 'medicine' in program['name'].lower():
            reasons.append("Suitable for high-achieving students")
        
        # Ensure we have at least one reason
        if not reasons:
            reasons.append(f"Suitable for high-achieving students")
        
        return reasons[:3]  # Limit to 3 reasons
    
    def is_available(self) -> bool:
        """Check if demo service is available (always True)."""
        return True
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the demo service."""
        return {
            'service_type': 'demo',
            'model_version': 'demo_v1.0',
            'available_programs': len(self.demo_programs),
            'description': 'Demo prediction service for testing without trained models'
        }


# Global demo service instance
demo_prediction_service = DemoPredictionService() 