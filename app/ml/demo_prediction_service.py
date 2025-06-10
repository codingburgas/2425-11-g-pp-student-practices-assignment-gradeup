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
                'school_name': 'Tech University',
                'degree_type': 'Bachelor',
                'keywords': ['technology', 'programming', 'software', 'computer']
            },
            {
                'id': 2,
                'name': 'Software Engineering',
                'school_name': 'Innovation Institute',
                'degree_type': 'Bachelor',
                'keywords': ['technology', 'programming', 'software', 'engineering']
            },
            {
                'id': 3,
                'name': 'Data Science',
                'school_name': 'Analytics College',
                'degree_type': 'Master',
                'keywords': ['data', 'analytics', 'mathematics', 'statistics']
            },
            {
                'id': 4,
                'name': 'Business Administration',
                'school_name': 'Business School',
                'degree_type': 'Bachelor',
                'keywords': ['business', 'management', 'leadership', 'economics']
            },
            {
                'id': 5,
                'name': 'Digital Marketing',
                'school_name': 'Marketing Academy',
                'degree_type': 'Bachelor',
                'keywords': ['marketing', 'business', 'digital', 'communications']
            },
            {
                'id': 6,
                'name': 'Nursing',
                'school_name': 'Health Sciences University',
                'degree_type': 'Bachelor',
                'keywords': ['healthcare', 'nursing', 'medical', 'caring']
            },
            {
                'id': 7,
                'name': 'Psychology',
                'school_name': 'Liberal Arts College',
                'degree_type': 'Bachelor',
                'keywords': ['psychology', 'mental_health', 'research', 'helping']
            },
            {
                'id': 8,
                'name': 'Mechanical Engineering',
                'school_name': 'Engineering Institute',
                'degree_type': 'Bachelor',
                'keywords': ['engineering', 'mechanical', 'design', 'technology']
            },
            {
                'id': 9,
                'name': 'Graphic Design',
                'school_name': 'Art & Design School',
                'degree_type': 'Bachelor',
                'keywords': ['design', 'creative', 'visual', 'art']
            },
            {
                'id': 10,
                'name': 'Finance',
                'school_name': 'Business University',
                'degree_type': 'Bachelor',
                'keywords': ['finance', 'business', 'economics', 'money']
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
            
        all_interests.append(career_goals)
        
        # Score each program based on keyword matching
        program_scores = []
        for program in self.demo_programs:
            score = 0
            
            # Check keyword matches
            for keyword in program['keywords']:
                for interest in all_interests:
                    if keyword in interest or interest in keyword:
                        score += 1
            
            # Add some randomness for variety
            score += random.uniform(0, 2)
            
            # Normalize score to confidence (0-1)
            confidence = min(0.95, max(0.1, score / 10))
            
            program_scores.append({
                'program': program,
                'confidence': confidence
            })
        
        # Sort by confidence and take top K
        program_scores.sort(key=lambda x: x['confidence'], reverse=True)
        
        for i, item in enumerate(program_scores[:top_k]):
            program = item['program']
            base_confidence = item['confidence']
            
            # Add some variance to make it realistic
            confidence_variance = random.uniform(-0.1, 0.1)
            final_confidence = max(0.1, min(0.95, base_confidence + confidence_variance))
            
            prediction = {
                'program_id': program['id'],
                'program_name': program['name'],
                'school_name': program['school_name'],
                'confidence': round(final_confidence, 3),
                'rank': i + 1,
                'degree_type': program['degree_type'],
                'match_reasons': self._generate_match_reasons(program, all_interests)
            }
            
            predictions.append(prediction)
        
        return predictions
    
    def _generate_match_reasons(self, program: Dict, user_interests: List[str]) -> List[str]:
        """Generate realistic match reasons for a program."""
        reasons = []
        
        # Check for keyword matches
        for keyword in program['keywords']:
            for interest in user_interests:
                if keyword in interest or interest in keyword:
                    reasons.append(f"Strong match with your interest in {keyword}")
                    break
        
        # Add some generic reasons
        if program['name'] in ['Computer Science', 'Software Engineering']:
            if any('tech' in interest for interest in user_interests):
                reasons.append("High demand field with excellent career prospects")
        
        if 'business' in program['keywords']:
            reasons.append("Versatile degree with multiple career paths")
        
        if 'healthcare' in program['keywords']:
            reasons.append("Meaningful career helping others")
        
        # Ensure we have at least one reason
        if not reasons:
            reasons.append(f"Good general fit based on your profile")
        
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