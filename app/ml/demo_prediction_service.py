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
        # NO HARDCODED PROGRAMS! Use ONLY database programs
        self.demo_programs = []
        self._load_programs_from_database()

    def _load_programs_from_database(self):
        """Load all programs from the actual database - NO FAKE PROGRAMS!"""
        try:
            from app.models import Program, School
            programs = Program.query.join(School).all()
            
            for program in programs:
                # Generate keywords based on actual program name
                keywords = self._generate_keywords_from_name(program.name)
                
                self.demo_programs.append({
                    'id': program.id,  # REAL database ID
                    'name': program.name,  # REAL database program name
                    'school_name': program.school.name,  # REAL database school name
                    'degree_type': program.degree_type or 'Bachelor',
                    'keywords': keywords
                })
                
            self.logger.info(f"Loaded {len(self.demo_programs)} real programs from database")
            
        except Exception as e:
            self.logger.error(f"Error loading programs from database: {e}")
            # If database fails, create empty list - NO FAKE FALLBACKS!
            self.demo_programs = []

    def _generate_keywords_from_name(self, program_name: str) -> List[str]:
        """Generate keywords based on the actual program name"""
        name_lower = program_name.lower()
        keywords = []
        
        # Add program name words as keywords
        keywords.extend(name_lower.split())
        
        # Add relevant keywords based on program type
        if 'computer' in name_lower or 'informatics' in name_lower:
            keywords.extend(['technology', 'programming', 'software', 'math', 'science'])
        elif 'engineering' in name_lower:
            keywords.extend(['engineering', 'technology', 'physics', 'math', 'science'])
        elif 'business' in name_lower or 'administration' in name_lower:
            keywords.extend(['management', 'leadership', 'economics', 'finance'])
        elif 'medicine' in name_lower or 'medical' in name_lower:
            keywords.extend(['biology', 'health', 'science', 'healthcare'])
        elif 'psychology' in name_lower:
            keywords.extend(['mental_health', 'research', 'helping', 'social'])
        elif 'economics' in name_lower:
            keywords.extend(['finance', 'business', 'mathematics', 'statistics'])
        elif 'law' in name_lower:
            keywords.extend(['legal', 'justice', 'social', 'politics'])
        elif 'communication' in name_lower or 'mass' in name_lower:
            keywords.extend(['media', 'writing', 'reporting'])
        
        return list(set(keywords))  # Remove duplicates
    
    def predict_programs(self, survey_data: Dict[str, Any], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Generate demo predictions based on survey data.
        
        Args:
            survey_data: Survey response data
            top_k: Number of predictions to return
            
        Returns:
            List of mock predictions with realistic confidence scores
        """
        # Ensure programs are loaded
        if not self.demo_programs:
            self.logger.info("No programs loaded, attempting to reload from database")
            self._load_programs_from_database()
            
        # If still no programs, return empty list
        if not self.demo_programs:
            self.logger.error("No programs available in database")
            return []
            
        predictions = []
        
        # Extract interests from survey data
        career_interests = survey_data.get('career_interests', [])
        favorite_subjects = survey_data.get('favorite_subjects', [])
        career_goals = survey_data.get('career_goals', '').lower()
        math_interest = survey_data.get('math_interest', 5)
        science_interest = survey_data.get('science_interest', 5)
        art_interest = survey_data.get('art_interest', 5)
        sports_interest = survey_data.get('sports_interest', 5)
        preferred_subject = survey_data.get('preferred_subject', '').lower()
        
        # Add more survey fields for better matching
        career_goal = survey_data.get('career_goal', '').lower()
        if career_goal and not career_goals:
            career_goals = career_goal
        
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
        
        # Add interest values to adjust scoring with better mapping
        interest_map = {
            'math': math_interest,
            'mathematics': math_interest,
            'technology': math_interest * 0.8,  # Reduce tech bias
            'programming': math_interest * 0.8,
            'computer': math_interest * 0.8,
            'science': science_interest,
            'biology': science_interest,
            'physics': science_interest,
            'chemistry': science_interest,
            'medicine': science_interest,
            'healthcare': science_interest,
            'art': art_interest,
            'design': art_interest,
            'creative': art_interest,
            'drawing': art_interest,
            'painting': art_interest,
            'graphics': art_interest,
            'sports': sports_interest,
            'physical': sports_interest,
            'fitness': sports_interest,
            'business': 7,  # Base business interest
            'management': 7,
            'economics': 6,
            'finance': 6,
            'law': 6,
            'legal': 6,
            'politics': 5,
            'social': 6,
            'psychology': 6,
            'history': 5,
            'journalism': 6,
            'media': 6,
            'tourism': 6,
            'international': 5
        }
        
        # Enhanced scoring: Directly boost programs matching preferred subject
        preferred_subject_boost = 8  # Strong boost for preferred subject match
        career_goal_boost = 10      # Very strong boost for career goal match
        
        # Score each program based on keyword matching and interests
        program_scores = []
        for i, program in enumerate(self.demo_programs):
            score = 0
            program_name_lower = program['name'].lower()
            
            # Calculate base interest alignment
            interest_alignment = 0
            keyword_matches = 0
            
            # Check keyword matches with weighted scoring
            for keyword in program['keywords']:
                keyword_lower = keyword.lower()
                
                # Add points for matching keywords in interests
                for interest in all_interests:
                    if keyword_lower in interest or interest in keyword_lower:
                        keyword_matches += 1
                        score += 1.5
                
                # Add points based on interest levels with domain-specific weighting
                if keyword_lower in interest_map:
                    interest_level = interest_map[keyword_lower]
                    if isinstance(interest_level, (int, float)):
                        weight = interest_level / 10.0
                        score += weight * 2.5
                        interest_alignment += weight
            
            # Specific interest domain matching with reduced CS bias
            if 'computer' in program_name_lower or 'informatics' in program_name_lower:
                # Computer Science programs require HIGH math interest to get boost
                if math_interest >= 8:
                    tech_score = (math_interest + science_interest) / 2
                    score += (tech_score / 10) * 2.5  # Reduced from 3
                elif math_interest >= 6:
                    score += 1.0  # Minimal boost for moderate math
                else:
                    score -= 2.0  # Penalty for low math interest
                    
                # Strong penalty if user has low math/science but high art
                if math_interest < 6 and art_interest > 6:
                    score -= 5
                    
            elif 'engineering' in program_name_lower:
                # Engineering programs favor math and science
                if math_interest >= 7 or science_interest >= 7:
                    eng_score = (math_interest + science_interest) / 2
                    score += (eng_score / 10) * 3.5
                else:
                    score += 1.0  # Minimal boost if low math/science
                    
            elif 'art' in program_name_lower or 'design' in program_name_lower:
                # Art programs favor creativity - strong boost
                score += (art_interest / 10) * 5  # Increased from 4
                if art_interest < 5:
                    score -= 3  # Stronger penalty
                else:
                    score += 2  # Additional boost for high art interest
                    
            elif 'medicine' in program_name_lower:
                # Medicine favors science and helping
                if science_interest >= 7:
                    score += (science_interest / 10) * 4  # Increased from 3.5
                else:
                    score += 1.5  # Some boost even for moderate science
                    
            elif 'business' in program_name_lower or 'economics' in program_name_lower:
                # Business programs get strong consistent scoring
                score += 4.0  # Increased from 3.5
                # Boost for communication-oriented students
                if art_interest >= 6 or science_interest >= 6:
                    score += 1.5
                    
            elif 'psychology' in program_name_lower:
                # Psychology programs - boost for social/helping oriented students
                social_score = (art_interest + science_interest) / 2
                score += 3.0 + (social_score / 10) * 2
                
            elif 'sports' in program_name_lower:
                # Sports programs
                score += (sports_interest / 10) * 5  # Increased from 4
                if sports_interest < 5:
                    score -= 3  # Stronger penalty
                else:
                    score += 2  # Additional boost
                    
            elif 'law' in program_name_lower:
                # Law programs - favor communication and social interests
                score += 3.5 + (art_interest / 15) * 2
                
            elif 'history' in program_name_lower or 'journalism' in program_name_lower:
                # Humanities programs
                score += 3.0 + (art_interest / 10) * 2
            
            # Direct match to preferred subject - major boost
            if preferred_subject and any(preferred_subject in kw.lower() for kw in program['keywords']):
                score += preferred_subject_boost
                
            # Direct match to career goals - highest boost with more specific matching
            if career_goals:
                career_match_found = False
                for kw in program['keywords']:
                    if career_goals in kw.lower() or kw.lower() in career_goals:
                        career_match_found = True
                        break
                
                if career_match_found:
                    score += career_goal_boost
                    
                # Specific career goal matching
                if 'technology' in career_goals and 'computer' in program_name_lower:
                    if math_interest >= 7:  # Only boost if high math interest
                        score += 1.5  # Reduced boost
                elif 'business' in career_goals and 'business' in program_name_lower:
                    score += 4  # Strong boost for business
                elif 'health' in career_goals and 'medicine' in program_name_lower:
                    score += 4  # Strong boost for healthcare
                elif 'art' in career_goals and 'art' in program_name_lower:
                    score += 4  # Strong boost for arts
                elif 'sports' in career_goals and 'sports' in program_name_lower:
                    score += 4  # Strong boost for sports
            
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
            
            # Keep as decimal (0-1.0) for consistency
            final_confidence = confidence_decimal
            
            # Ensure we don't have all recommendations with the same score
            # Add position-based variance for more diversity (in decimal format)
            position_variance = (i % 5) * 0.03  # 0, 0.03, 0.06, 0.09, 0.12 variance
            final_confidence = max(0.2, min(0.95, final_confidence - position_variance))
            
            # Add much stronger position-based variance to ensure different scores
            # First item gets highest score, others get progressively lower scores
            position_factor = i * 0.05  # 0, 0.05, 0.10, 0.15, 0.20, etc.
            if i > 0:  # Keep the top recommendation at its original score
                final_confidence = max(0.25, min(0.9, final_confidence - position_factor))
            
            # Reset the random seed after use
            random.seed()
            
            prediction = {
                'program_id': program['id'],
                'program_name': program['name'],
                'school_name': program['school_name'],
                'confidence': final_confidence,
                'match_score': final_confidence,  # Already decimal
                'rank': i + 1,
                'degree_type': program['degree_type'],
                'match_reasons': self._generate_match_reasons(program, all_interests, interest_map),
                'program': program  # Add program reference for sorting
            }
            
            program_scores.append(prediction)
        
        # Check if we have any matches
        if not program_scores:
            # Ensure we always have some recommendations by adding diverse default programs
            self.logger.warning("No matching programs found, adding default recommendations")
            
            # Safely select default programs from available programs
            default_programs = []
            if len(self.demo_programs) >= 1:
                default_programs.append({
                    'program': self.demo_programs[0],
                    'confidence': 0.65
                })
            if len(self.demo_programs) >= 2:
                default_programs.append({
                    'program': self.demo_programs[1],
                    'confidence': 0.60
                })
            if len(self.demo_programs) >= 3:
                default_programs.append({
                    'program': self.demo_programs[2],
                    'confidence': 0.55
                })
            
            # If we still don't have enough programs, create some basic ones
            if len(default_programs) < 3 and len(self.demo_programs) > 0:
                for i in range(len(default_programs), min(3, len(self.demo_programs))):
                    default_programs.append({
                        'program': self.demo_programs[i],
                        'confidence': 0.5 - (i * 0.05)
                    })
            
            program_scores.extend(default_programs)
        
        # Sort by confidence and take top K
        program_scores.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Ensure diversity in top recommendations - prevent any single field from dominating
        if len(program_scores) > 5:
            # Count programs by field in top 5
            field_counts = {}
            top_programs = program_scores[:5]
            
            for item in top_programs:
                program_name = item['program']['name'].lower()
                field = 'other'
                
                if 'computer' in program_name or 'informatics' in program_name:
                    field = 'computer_science'
                elif 'engineering' in program_name:
                    field = 'engineering'
                elif 'business' in program_name or 'economics' in program_name:
                    field = 'business'
                elif 'medicine' in program_name:
                    field = 'medicine'
                elif 'art' in program_name or 'design' in program_name:
                    field = 'arts'
                elif 'psychology' in program_name:
                    field = 'psychology'
                    
                field_counts[field] = field_counts.get(field, 0) + 1
            
            # If any field has more than 2 programs in top 5, promote diversity
            max_field_count = max(field_counts.values()) if field_counts else 0
            if max_field_count > 2:
                self.logger.info(f"Field imbalance detected (max: {max_field_count}), promoting diversity")
                
                # Find the dominant field
                dominant_field = max(field_counts.items(), key=lambda x: x[1])[0]
                
                # Find programs from other fields to promote
                for i in range(5, min(len(program_scores), 15)):
                    candidate = program_scores[i]
                    candidate_name = candidate['program']['name'].lower()
                    candidate_field = 'other'
                    
                    if 'computer' in candidate_name or 'informatics' in candidate_name:
                        candidate_field = 'computer_science'
                    elif 'engineering' in candidate_name:
                        candidate_field = 'engineering'
                    elif 'business' in candidate_name or 'economics' in candidate_name:
                        candidate_field = 'business'
                    elif 'medicine' in candidate_name:
                        candidate_field = 'medicine'
                    elif 'art' in candidate_name or 'design' in candidate_name:
                        candidate_field = 'arts'
                    elif 'psychology' in candidate_name:
                        candidate_field = 'psychology'
                    
                    # If this candidate is from a different field, promote it
                    if candidate_field != dominant_field:
                        # Find a program from the dominant field to replace (start from position 2)
                        for j in range(2, 5):
                            target_name = program_scores[j]['program']['name'].lower()
                            target_field = 'computer_science' if ('computer' in target_name or 'informatics' in target_name) else 'other'
                            
                            if target_field == dominant_field:
                                self.logger.info(f"Swapping {target_name} with {candidate_name} for diversity")
                                program_scores[j], program_scores[i] = program_scores[i], program_scores[j]
                                break
                        break
        
        # Return the top predictions
        final_predictions = []
        for i, item in enumerate(program_scores[:top_k]):
            # Clean up the prediction and ensure proper format
            prediction = {
                'program_id': item['program_id'],
                'program_name': item['program_name'],
                'school_name': item['school_name'],
                'confidence': item['confidence'],
                'match_score': item['match_score'],
                'rank': i + 1,
                'degree_type': item.get('degree_type', 'Bachelor'),
                'match_reasons': item['match_reasons']
            }
            final_predictions.append(prediction)
        
        return final_predictions
    
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