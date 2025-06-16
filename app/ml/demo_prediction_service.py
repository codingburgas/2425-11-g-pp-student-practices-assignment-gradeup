"""
Demo Prediction Service

This service provides mock predictions when no trained ML models are available.
Perfect for testing and demonstration purposes.
"""

import random
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging


class DemoPredictionService:
    """Demo service that generates realistic mock predictions."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
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
        
        # Add more diverse programs for better recommendations
        self.diverse_programs = [
            {
                'id': 16,
                'name': 'Mathematics',
                'school_name': 'Sofia University St. Kliment Ohridski',
                'degree_type': 'Bachelor',
                'keywords': ['mathematics', 'algebra', 'geometry', 'statistics', 'science']
            },
            {
                'id': 17,
                'name': 'Graphic Design',
                'school_name': 'National Academy of Arts',
                'degree_type': 'Bachelor',
                'keywords': ['design', 'art', 'creative', 'visual', 'graphics', 'digital']
            },
            {
                'id': 18,
                'name': 'Sports Science',
                'school_name': 'National Sports Academy',
                'degree_type': 'Bachelor',
                'keywords': ['sports', 'physical', 'fitness', 'exercise', 'coaching']
            },
            {
                'id': 19,
                'name': 'Tourism Management',
                'school_name': 'University of Economics - Varna',
                'degree_type': 'Bachelor',
                'keywords': ['tourism', 'management', 'business', 'hospitality']
            },
            {
                'id': 20,
                'name': 'International Relations',
                'school_name': 'Sofia University St. Kliment Ohridski',
                'degree_type': 'Bachelor',
                'keywords': ['politics', 'international', 'diplomacy', 'languages']
            },
            {
                'id': 21,
                'name': 'Journalism',
                'school_name': 'Sofia University St. Kliment Ohridski',
                'degree_type': 'Bachelor',
                'keywords': ['media', 'writing', 'communication', 'reporting']
            },
            {
                'id': 22,
                'name': 'History',
                'school_name': 'Sofia University St. Kliment Ohridski',
                'degree_type': 'Bachelor',
                'keywords': ['history', 'humanities', 'research', 'culture']
            },
            {
                'id': 23,
                'name': 'Chemistry',
                'school_name': 'Sofia University St. Kliment Ohridski',
                'degree_type': 'Bachelor',
                'keywords': ['chemistry', 'science', 'laboratory', 'research']
            }
        ]
        
        # Combine regular and diverse programs
        self.demo_programs.extend(self.diverse_programs)
    
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
        
        # Enhanced scoring: Directly boost programs matching preferred subject
        preferred_subject_boost = 8  # Strong boost for preferred subject match
        career_goal_boost = 10      # Very strong boost for career goal match
        
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
            
            # Direct match to preferred subject - major boost
            if preferred_subject and any(preferred_subject.lower() in kw.lower() for kw in program['keywords']):
                score += preferred_subject_boost
                
            # Direct match to career goals - highest boost
            if career_goals and any(career_goals.lower() in kw.lower() for kw in program['keywords']):
                score += career_goal_boost
            
            # Special match for "Computer Science" - only boost if the user has high math interest
            if program['name'] == 'Computer Science' and math_interest < 6:
                score -= 3  # Penalty for users with low math interest
            
            # Base score should never be zero to avoid all-identical matches
            base_score = max(1, score)
            
            # Add unique variance based on program ID and name to avoid identical scores
            # Hash the program name to get a consistent but unique modifier
            name_hash = hash(program['name']) % 100
            id_modifier = program['id'] % 10
            
            # Scale by program degree type (Bachelor/Master)
            degree_bonus = 1.0 if program['degree_type'] == 'Bachelor' else 1.2
            
            # Create a unique variance factor for each program (between 0-2.5)
            unique_variance = ((name_hash / 40) + (id_modifier / 4)) * degree_bonus
            
            score = base_score + unique_variance
            
            # Improved normalization for better score distribution
            # Ensure scores have better variance and meaningful differentiation
            if score >= 10:
                confidence_decimal = min(0.95, 0.75 + (score - 10) * 0.02)
            elif score >= 5:
                confidence_decimal = min(0.75, 0.5 + (score - 5) * 0.05) 
            else:
                confidence_decimal = min(0.5, max(0.3, score * 0.06))
            
            # Ensure each program gets a significantly different score
            # Add a larger unique offset based on program ID and name
            program_id_factor = (program['id'] % 20) * 0.01  # 0-0.19 variation
            name_factor = (hash(program['name']) % 25) * 0.01  # 0-0.24 variation
            
            # Apply the unique factors to the confidence score
            confidence_decimal = min(0.95, confidence_decimal + program_id_factor + name_factor)
            
            # Convert to percentage (0-100)
            final_confidence = int(confidence_decimal * 100)
            
            # Ensure we don't have all recommendations with the same score
            # Add position-based variance for more diversity
            position_variance = (i % 5) * 3  # 0, 3, 6, 9, 12 variance
            final_confidence = max(20, min(95, final_confidence - position_variance))
            
            # Add much stronger position-based variance to ensure different scores
            # First item gets highest score, others get progressively lower scores
            position_factor = i * 5  # 0, 5, 10, 15, 20, etc.
            if i > 0:  # Keep the top recommendation at its original score
                final_confidence = max(25, min(90, final_confidence - position_factor))
            
            # Reset the random seed after use
            random.seed()
            
            prediction = {
                'program_id': program['id'],
                'program_name': program['name'],
                'school_name': program['school_name'],
                'confidence': final_confidence,
                'match_score': final_confidence / 100.0,  # Convert to decimal
                'rank': i + 1,
                'degree_type': program['degree_type'],
                'match_reasons': self._generate_match_reasons(program, all_interests, interest_map)
            }
            
            predictions.append(prediction)
        
        # Check if we have any matches
        if not program_scores:
            # Ensure we always have some recommendations by adding default programs
            self.logger.warning("No matching programs found, adding default recommendations")
            default_programs = [
                {
                    'program': self.demo_programs[0],  # Computer Science
                    'confidence': 65
                },
                {
                    'program': self.demo_programs[1],  # Software Engineering
                    'confidence': 60
                },
                {
                    'program': self.demo_programs[2],  # Informatics
                    'confidence': 55
                }
            ]
            program_scores.extend(default_programs)
        
        # Sort by confidence and take top K
        program_scores.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Ensure diversity in top recommendations
        # If top 3 recommendations are all computer science related, swap in a different field
        computer_science_count = 0
        for item in program_scores[:3]:
            program_name = item['program']['name'].lower()
            if 'computer' in program_name or 'software' in program_name or 'informatics' in program_name:
                computer_science_count += 1
                
        # If all top recommendations are computer science, promote diversity
        if computer_science_count >= 2 and len(program_scores) > 3:
            self.logger.info("Too many computer science programs in top recommendations, promoting diversity")
            # Find a non-computer science program to promote
            for i, item in enumerate(program_scores[3:], start=3):
                program_name = item['program']['name'].lower()
                if not ('computer' in program_name or 'software' in program_name or 'informatics' in program_name):
                    # Swap this program with the second position
                    program_scores[1], program_scores[i] = program_scores[i], program_scores[1]
                    break
        
        # Create predictions with fixed decreasing scores for better variance
        predictions = []
        for i, item in enumerate(program_scores[:top_k]):
            program = item['program']
            
            # Fixed decreasing scores: 65%, 55%, 45%, 35%, etc.
            fixed_score = 65 - (i * 10)
            
            # Add small random variation to avoid exact same scores
            # But keep the first recommendation at exactly 65%
            if i > 0:
                # Small random variation +/- 2%
                random.seed(hash(program['name']) + i)
                variation = random.randint(-2, 2)
                fixed_score = max(20, min(95, fixed_score + variation))
            
            prediction = {
                'program_id': program['id'],
                'program_name': program['name'],
                'school_name': program['school_name'],
                'confidence': fixed_score,
                'match_score': fixed_score / 100.0,  # Convert to decimal
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