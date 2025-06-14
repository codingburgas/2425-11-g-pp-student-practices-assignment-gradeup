"""
Advanced Recommendation Engine for GradeUp School Recommendation System

This module implements a comprehensive recommendation engine with the following components:
1. University matching algorithm based on location and program preferences
2. Program recommendation system using interest alignment and career goals
3. Personalized suggestions including trending programs and completion suggestions
4. Recommendation history tracking and pattern analysis

Author: GradeUp Development Team
Version: 1.0
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import json
import logging
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from flask import current_app

from app import db
from app.models import User, School, Program, Survey, SurveyResponse, Recommendation, Favorite, PredictionHistory
from .utils import extract_features_from_survey_response


class RecommendationEngine:
    """
    Advanced recommendation engine for university and program recommendations.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        # Create console handler if not exists
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        
        self.scaler = StandardScaler()
        self.is_initialized = False
        
    def initialize(self):
        """Initialize the recommendation engine."""
        try:
            self.is_initialized = True
            self.logger.info("Recommendation engine initialized successfully")
        except Exception as e:
            self.logger.error(f"Error initializing recommendation engine: {e}")
            self.is_initialized = False

    def match_universities(self, user_preferences: Dict[str, Any], 
                          survey_data: Optional[Dict[str, Any]] = None,
                          top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Advanced university matching algorithm.
        
        Args:
            user_preferences: User preference data
            survey_data: Optional survey response data
            top_k: Number of top matches to return
            
        Returns:
            List of matched universities with scores
        """
        try:
            if not self.is_initialized:
                self.initialize()
                
            # Get all active schools
            schools = School.query.all()
            if not schools:
                return []
                
            matches = []
            
            for school in schools:
                score = self._calculate_university_match_score(
                    school, user_preferences, survey_data
                )
                
                matches.append({
                    'school_id': school.id,
                    'school_name': school.name,
                    'location': school.location,
                    'description': school.description,
                    'website': school.website,
                    'match_score': score,
                    'match_reasons': self._get_match_reasons(school, user_preferences, survey_data)
                })
                
            # Sort by match score
            matches.sort(key=lambda x: x['match_score'], reverse=True)
            
            return matches[:top_k]
            
        except Exception as e:
            self.logger.error(f"Error in university matching: {e}")
            return []
            
    def _calculate_university_match_score(self, school: School, 
                                        user_preferences: Dict[str, Any],
                                        survey_data: Optional[Dict[str, Any]] = None) -> float:
        """Calculate match score for a university."""
        score = 0.0
        
        # Location preference matching
        if user_preferences.get('preferred_location'):
            if school.location and user_preferences['preferred_location'].lower() in school.location.lower():
                score += 0.3
                
        # Program diversity scoring
        program_count = school.programs.count()
        if program_count > 10:
            score += 0.2
        elif program_count > 5:
            score += 0.1
            
        # Survey data matching
        if survey_data:
            score += self._calculate_survey_alignment_score(school, survey_data)
            
        return min(score, 1.0)  # Cap at 1.0
        
    def _calculate_survey_alignment_score(self, school: School, survey_data: Dict[str, Any]) -> float:
        """Calculate how well school aligns with survey responses."""
        alignment_score = 0.0
        
        # Interest-based scoring
        interests = {
            'math_interest': ['Mathematics', 'Engineering', 'Computer Science'],
            'science_interest': ['Physics', 'Chemistry', 'Biology', 'Medicine'],
            'art_interest': ['Art', 'Design', 'Literature', 'Music'],
            'sports_interest': ['Sports Science', 'Physical Education']
        }
        
        for interest, related_fields in interests.items():
            if survey_data.get(interest, 0) >= 7:  # High interest
                for program in school.programs:
                    if any(field.lower() in program.name.lower() for field in related_fields):
                        alignment_score += 0.1
                        break
                        
        return min(alignment_score, 0.4)  # Cap contribution at 0.4
        
    def _get_match_reasons(self, school: School, user_preferences: Dict[str, Any],
                          survey_data: Optional[Dict[str, Any]] = None) -> List[str]:
        """Get reasons why this school matches the user."""
        reasons = []
        
        if user_preferences.get('preferred_location'):
            if school.location and user_preferences['preferred_location'].lower() in school.location.lower():
                reasons.append(f"Located in preferred area: {school.location}")
                
        program_count = school.programs.count()
        if program_count > 10:
            reasons.append(f"Offers diverse programs ({program_count} programs available)")
            
        if survey_data:
            high_interests = [k for k, v in survey_data.items() if k.endswith('_interest') and v >= 7]
            if high_interests:
                reasons.append("Programs align with your interests")
                
        return reasons

    def recommend_programs(self, user_id: int, survey_data: Dict[str, Any],
                          user_preferences: Optional[Dict[str, Any]] = None,
                          top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Advanced program recommendation system.
        
        Args:
            user_id: User ID for personalization
            survey_data: Survey response data
            user_preferences: Optional user preferences
            top_k: Number of top recommendations to return
            
        Returns:
            List of recommended programs with scores and reasons
        """
        try:
            if not self.is_initialized:
                self.initialize()
                
            # Get all programs
            programs = Program.query.join(School).all()
            if not programs:
                return []
                
            # Get user's past favorites for personalization
            user_favorites = self._get_user_favorites(user_id)
            
            recommendations = []
            
            for program in programs:
                score = self._calculate_program_match_score(
                    program, survey_data, user_preferences, user_favorites
                )
                
                if score > 0.2:  # Only include programs with reasonable match
                    recommendations.append({
                        'program_id': program.id,
                        'program_name': program.name,
                        'school_id': program.school.id,
                        'school_name': program.school.name,
                        'degree_type': program.degree_type,
                        'duration': program.duration,
                        'tuition_fee': program.tuition_fee,
                        'description': program.description,
                        'match_score': score,
                        'recommendation_reasons': self._get_program_match_reasons(
                            program, survey_data, user_preferences
                        )
                    })
                    
            # Sort by match score
            recommendations.sort(key=lambda x: x['match_score'], reverse=True)
            
            # Apply diversity filtering to avoid too many similar programs
            diverse_recommendations = self._apply_diversity_filter(recommendations, top_k)
            
            return diverse_recommendations
            
        except Exception as e:
            self.logger.error(f"Error in program recommendation: {e}")
            return []
            
    def _calculate_program_match_score(self, program: Program, survey_data: Dict[str, Any],
                                     user_preferences: Optional[Dict[str, Any]] = None,
                                     user_favorites: List[int] = None) -> float:
        """Calculate match score for a program."""
        score = 0.0
        
        # Interest alignment scoring
        score += self._calculate_interest_alignment(program, survey_data)
        
        # Career goal alignment
        score += self._calculate_career_alignment(program, survey_data)
        
        # Academic performance alignment
        score += self._calculate_academic_alignment(program, survey_data)
        
        # User preference alignment
        if user_preferences:
            score += self._calculate_preference_alignment(program, user_preferences)
            
        # Favorite schools bonus
        if user_favorites and program.school_id in user_favorites:
            score += 0.1
            
        return min(score, 1.0)  # Cap at 1.0
        
    def _calculate_interest_alignment(self, program: Program, survey_data: Dict[str, Any]) -> float:
        """Calculate how well program aligns with user interests."""
        program_name_lower = program.name.lower()
        alignment_score = 0.0
        
        # Define interest mappings
        interest_mappings = {
            'math_interest': {
                'keywords': ['math', 'engineering', 'computer', 'physics', 'statistics', 'data'],
                'weight': 0.3
            },
            'science_interest': {
                'keywords': ['biology', 'chemistry', 'physics', 'medicine', 'research', 'lab'],
                'weight': 0.3
            },
            'art_interest': {
                'keywords': ['art', 'design', 'creative', 'music', 'literature', 'media'],
                'weight': 0.3
            },
            'sports_interest': {
                'keywords': ['sport', 'physical', 'fitness', 'health', 'recreation'],
                'weight': 0.2
            }
        }
        
        for interest_key, mapping in interest_mappings.items():
            interest_level = survey_data.get(interest_key, 0)
            if interest_level >= 5:  # Moderate to high interest
                for keyword in mapping['keywords']:
                    if keyword in program_name_lower:
                        # Scale by interest level (1-10) and mapping weight
                        alignment_score += (interest_level / 10.0) * mapping['weight']
                        break
                        
        return min(alignment_score, 0.4)  # Cap contribution
        
    def _calculate_career_alignment(self, program: Program, survey_data: Dict[str, Any]) -> float:
        """Calculate career goal alignment."""
        career_goal = survey_data.get('career_goal', '').lower()
        if not career_goal:
            return 0.0
            
        program_name_lower = program.name.lower()
        career_mappings = {
            'engineer': ['engineering', 'computer', 'software', 'mechanical', 'electrical'],
            'doctor': ['medicine', 'health', 'biology', 'chemistry'],
            'teacher': ['education', 'teaching', 'pedagogy'],
            'business': ['business', 'management', 'economics', 'finance'],
            'artist': ['art', 'design', 'creative', 'music'],
            'scientist': ['science', 'research', 'physics', 'chemistry', 'biology']
        }
        
        for career, keywords in career_mappings.items():
            if career in career_goal:
                for keyword in keywords:
                    if keyword in program_name_lower:
                        return 0.3
                        
        return 0.0
        
    def _calculate_academic_alignment(self, program: Program, survey_data: Dict[str, Any]) -> float:
        """Calculate academic performance alignment."""
        grades_average = survey_data.get('grades_average', 0)
        study_hours = survey_data.get('study_hours_per_day', 0)
        
        alignment_score = 0.0
        
        # High achievers might prefer challenging programs
        if grades_average >= 5.0 and study_hours >= 4:
            challenging_keywords = ['advanced', 'honors', 'research', 'master', 'phd']
            program_name_lower = program.name.lower()
            if any(keyword in program_name_lower for keyword in challenging_keywords):
                alignment_score += 0.2
                
        return alignment_score
        
    def _calculate_preference_alignment(self, program: Program, user_preferences: Dict[str, Any]) -> float:
        """Calculate alignment with user preferences."""
        alignment_score = 0.0
        
        # Budget considerations
        if user_preferences.get('max_tuition') and program.tuition_fee:
            if program.tuition_fee <= user_preferences['max_tuition']:
                alignment_score += 0.1
                
        # Duration preferences
        preferred_duration = user_preferences.get('preferred_duration')
        if preferred_duration and program.duration:
            if preferred_duration.lower() in program.duration.lower():
                alignment_score += 0.1
                
        return alignment_score
        
    def _get_program_match_reasons(self, program: Program, survey_data: Dict[str, Any],
                                 user_preferences: Optional[Dict[str, Any]] = None) -> List[str]:
        """Get reasons why this program matches the user."""
        reasons = []
        
        # Interest-based reasons
        program_name_lower = program.name.lower()
        high_interests = {k: v for k, v in survey_data.items() if k.endswith('_interest') and v >= 7}
        
        for interest_key, level in high_interests.items():
            interest_name = interest_key.replace('_interest', '').title()
            if interest_name.lower() in program_name_lower:
                reasons.append(f"Matches your high interest in {interest_name}")
                
        # Career alignment
        career_goal = survey_data.get('career_goal', '')
        if career_goal and career_goal.lower() in program_name_lower:
            reasons.append(f"Aligns with your career goal: {career_goal}")
            
        # Academic level
        grades_average = survey_data.get('grades_average', 0)
        if grades_average >= 5.0:
            reasons.append("Suitable for high-achieving students")
            
        return reasons[:3]  # Limit to top 3 reasons
        
    def _apply_diversity_filter(self, recommendations: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
        """Apply diversity filtering to recommendations."""
        if len(recommendations) <= top_k:
            return recommendations
            
        diverse_recs = []
        program_types_seen = set()
        schools_seen = set()
        
        # First pass: get diverse program types and schools
        for rec in recommendations:
            program_name = rec['program_name'].lower()
            school_id = rec['school_id']
            
            # Simple program type classification
            program_type = self._classify_program_type(program_name)
            
            if len(diverse_recs) < top_k:
                if program_type not in program_types_seen or school_id not in schools_seen:
                    diverse_recs.append(rec)
                    program_types_seen.add(program_type)
                    schools_seen.add(school_id)
                    
        # Fill remaining slots with highest scoring programs
        remaining_slots = top_k - len(diverse_recs)
        for rec in recommendations:
            if rec not in diverse_recs and remaining_slots > 0:
                diverse_recs.append(rec)
                remaining_slots -= 1
                
        return diverse_recs
        
    def _classify_program_type(self, program_name: str) -> str:
        """Classify program into broad categories."""
        name_lower = program_name.lower()
        
        if any(word in name_lower for word in ['engineering', 'computer', 'software']):
            return 'engineering'
        elif any(word in name_lower for word in ['business', 'management', 'economics']):
            return 'business'
        elif any(word in name_lower for word in ['medicine', 'health', 'biology']):
            return 'health'
        elif any(word in name_lower for word in ['art', 'design', 'creative']):
            return 'creative'
        elif any(word in name_lower for word in ['science', 'physics', 'chemistry']):
            return 'science'
        else:
            return 'other'
            
    def _get_user_favorites(self, user_id: int) -> List[int]:
        """Get list of user's favorite school IDs."""
        try:
            favorites = Favorite.query.filter_by(user_id=user_id).all()
            return [fav.school_id for fav in favorites]
        except Exception:
            return []

    def get_personalized_suggestions(self, user_id: int, limit: int = 5) -> Dict[str, Any]:
        """
        Generate personalized suggestions based on user behavior and preferences.
        
        Args:
            user_id: User ID
            limit: Maximum number of suggestions per category
            
        Returns:
            Dictionary with different types of personalized suggestions
        """
        try:
            user = User.query.get(user_id)
            if not user:
                return {}
                
            suggestions = {
                'trending_programs': self._get_trending_programs(limit),
                'similar_user_favorites': self._get_similar_user_recommendations(user_id, limit),
                'completion_suggestions': self._get_completion_suggestions(user_id, limit),
                'seasonal_recommendations': self._get_seasonal_recommendations(limit)
            }
            
            return suggestions
            
        except Exception as e:
            self.logger.error(f"Error generating personalized suggestions: {e}")
            return {}
            
    def _get_trending_programs(self, limit: int) -> List[Dict[str, Any]]:
        """Get currently trending programs based on user interactions."""
        try:
            # Get programs with most favorites in last 30 days - SQL Server compatible
            trending = db.session.query(Program.id, Program.name, School.name.label('school_name'), 
                                      db.func.count(Favorite.id).label('favorite_count'))\
                .select_from(Program)\
                .join(School, Program.school_id == School.id)\
                .outerjoin(Favorite, Favorite.school_id == School.id)\
                .filter(db.or_(
                    Favorite.created_at >= datetime.utcnow() - timedelta(days=30),
                    Favorite.created_at.is_(None)
                ))\
                .group_by(Program.id, Program.name, School.name)\
                .order_by(db.desc('favorite_count'))\
                .limit(limit).all()
                
            results = []
            for program_id, program_name, school_name, count in trending:
                results.append({
                    'program_id': program_id,
                    'program_name': program_name,
                    'school_name': school_name,
                    'trend_score': count,
                    'reason': f"Popular choice - {count} students favorited this month"
                })
                
            return results
            
        except Exception as e:
            self.logger.error(f"Error getting trending programs: {e}")
            # Fallback: return random programs
            try:
                fallback_programs = Program.query.join(School).limit(limit).all()
                results = []
                for program in fallback_programs:
                    results.append({
                        'program_id': program.id,
                        'program_name': program.name,
                        'school_name': program.school.name,
                        'trend_score': 0,
                        'reason': "Featured program"
                    })
                return results
            except Exception:
                return []
            
    def _get_similar_user_recommendations(self, user_id: int, limit: int) -> List[Dict[str, Any]]:
        """Get recommendations based on similar users' choices."""
        try:
            # Find users with similar favorites
            user_favorites = set(fav.school_id for fav in Favorite.query.filter_by(user_id=user_id).all())
            
            if not user_favorites:
                return []
                
            similar_users = []
            all_favorites = Favorite.query.filter(Favorite.user_id != user_id).all()
            
            # Group by user
            user_fav_map = {}
            for fav in all_favorites:
                if fav.user_id not in user_fav_map:
                    user_fav_map[fav.user_id] = set()
                user_fav_map[fav.user_id].add(fav.school_id)
                
            # Calculate similarity (Jaccard similarity)
            for other_user_id, other_favorites in user_fav_map.items():
                intersection = len(user_favorites & other_favorites)
                union = len(user_favorites | other_favorites)
                if union > 0:
                    similarity = intersection / union
                    if similarity > 0.2:  # At least 20% similarity
                        similar_users.append((other_user_id, similarity, other_favorites))
                        
            # Get recommendations from most similar users
            recommendations = []
            for other_user_id, similarity, other_favorites in sorted(similar_users, key=lambda x: x[1], reverse=True)[:3]:
                # Find schools they like that current user hasn't favorited
                new_recommendations = other_favorites - user_favorites
                for school_id in list(new_recommendations)[:limit]:
                    school = School.query.get(school_id)
                    if school:
                        recommendations.append({
                            'school_id': school.id,
                            'school_name': school.name,
                            'similarity_score': similarity,
                            'reason': f"Students with similar interests also liked this school"
                        })
                        
            return recommendations[:limit]
            
        except Exception as e:
            self.logger.error(f"Error getting similar user recommendations: {e}")
            return []
            
    def _get_completion_suggestions(self, user_id: int, limit: int) -> List[Dict[str, Any]]:
        """Get suggestions to help user complete their profile or take actions."""
        suggestions = []
        user = User.query.get(user_id)
        
        if not user:
            return []
            
        # Check survey completion
        completed_surveys = SurveyResponse.query.filter_by(user_id=user_id).count()
        total_surveys = Survey.query.filter_by(is_active=True).count()
        
        if completed_surveys < total_surveys:
            suggestions.append({
                'type': 'survey',
                'title': 'Complete Your Profile Survey',
                'description': f'You have {total_surveys - completed_surveys} surveys remaining',
                'action_url': '/survey',
                'priority': 1
            })
            
        # Check if user has favorites
        favorites_count = Favorite.query.filter_by(user_id=user_id).count()
        if favorites_count == 0:
            suggestions.append({
                'type': 'favorites',
                'title': 'Start Building Your Favorites',
                'description': 'Save schools you\'re interested in to get better recommendations',
                'action_url': '/universities',
                'priority': 2
            })
            
        # Check profile completeness
        if not user.bio or not user.location:
            suggestions.append({
                'type': 'profile',
                'title': 'Complete Your Profile',
                'description': 'Add more details to improve your recommendations',
                'action_url': '/auth/profile',
                'priority': 3
            })
            
        return sorted(suggestions, key=lambda x: x['priority'])[:limit]
        
    def _get_seasonal_recommendations(self, limit: int) -> List[Dict[str, Any]]:
        """Get seasonal or time-based recommendations."""
        recommendations = []
        current_month = datetime.utcnow().month
        
        # Application season recommendations (assuming fall applications)
        if current_month in [9, 10, 11, 12]:  # Fall application season
            recommendations.append({
                'type': 'seasonal',
                'title': 'Application Deadlines Approaching',
                'description': 'Many universities have application deadlines coming up',
                'season': 'fall_applications'
            })
        elif current_month in [1, 2, 3]:  # Spring consideration season
            recommendations.append({
                'type': 'seasonal',
                'title': 'Spring Semester Options',
                'description': 'Consider programs with spring start dates',
                'season': 'spring_options'
            })
            
        return recommendations[:limit]
        
    def store_recommendation_history(self, user_id: int, survey_response_id: Optional[int],
                                   recommendations: List[Dict[str, Any]], 
                                   recommendation_type: str = 'program') -> bool:
        """
        Store recommendation history for analysis and improvement.
        
        Args:
            user_id: User ID
            survey_response_id: Optional survey response ID
            recommendations: List of recommendations
            recommendation_type: Type of recommendation ('program', 'university', etc.)
            
        Returns:
            Success status
        """
        try:
            # Store individual recommendations in the database
            # Only store if we have a valid survey_response_id (required by schema)
            if survey_response_id is not None:
                for rec in recommendations:
                    if recommendation_type == 'program' and 'program_id' in rec:
                        recommendation = Recommendation(
                            survey_response_id=survey_response_id,
                            program_id=rec['program_id'],
                            score=rec.get('match_score', 0.0)
                        )
                        db.session.add(recommendation)
                    
            # Store in prediction history for advanced analytics
            if survey_response_id:
                survey_response = SurveyResponse.query.get(survey_response_id)
                if survey_response:
                    prediction_history = PredictionHistory(
                        user_id=user_id,
                        survey_response_id=survey_response_id,
                        input_features=json.dumps(survey_response.get_answers()),
                        predictions=json.dumps([{
                            'id': rec.get('program_id' if recommendation_type == 'program' else 'school_id'),
                            'name': rec.get('program_name' if recommendation_type == 'program' else 'school_name'),
                            'score': rec.get('match_score', 0.0)
                        } for rec in recommendations[:10]]),
                        confidence_scores=json.dumps([rec.get('match_score', 0.0) for rec in recommendations[:10]]),
                        model_version='recommendation_engine_v1.0',
                        prediction_type=recommendation_type
                    )
                    db.session.add(prediction_history)
                    
            db.session.commit()
            return True
            
        except Exception as e:
            self.logger.error(f"Error storing recommendation history: {e}")
            db.session.rollback()
            return False
            
    def get_recommendation_history(self, user_id: int, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Get user's recommendation history.
        
        Args:
            user_id: User ID
            limit: Maximum number of history records to return
            
        Returns:
            List of recommendation history records
        """
        try:
            history = PredictionHistory.query.filter_by(user_id=user_id)\
                .order_by(PredictionHistory.created_at.desc())\
                .limit(limit).all()
                
            results = []
            for record in history:
                results.append({
                    'id': record.id,
                    'created_at': record.created_at,
                    'prediction_type': record.prediction_type,
                    'model_version': record.model_version,
                    'predictions': record.get_predictions(),
                    'confidence_scores': record.get_confidence_scores(),
                    'input_features': record.get_input_features()
                })
                
            return results
            
        except Exception as e:
            self.logger.error(f"Error getting recommendation history: {e}")
            return []
            
    def analyze_recommendation_patterns(self, user_id: int) -> Dict[str, Any]:
        """
        Analyze user's recommendation patterns and preferences.
        
        Args:
            user_id: User ID
            
        Returns:
            Analysis results
        """
        try:
            history = PredictionHistory.query.filter_by(user_id=user_id).all()
            
            if not history:
                return {'message': 'No recommendation history available'}
                
            # Analyze preferred program types
            program_types = {}
            total_recommendations = 0
            
            for record in history:
                predictions = record.get_predictions()
                for pred in predictions:
                    program_name = pred.get('name', '').lower()
                    program_type = self._classify_program_type(program_name)
                    program_types[program_type] = program_types.get(program_type, 0) + 1
                    total_recommendations += 1
                    
            # Calculate preferences
            preferences = {}
            if total_recommendations > 0:
                for prog_type, count in program_types.items():
                    preferences[prog_type] = {
                        'count': count,
                        'percentage': (count / total_recommendations) * 100
                    }
                    
            # Get average confidence scores
            all_scores = []
            for record in history:
                all_scores.extend(record.get_confidence_scores())
                
            avg_confidence = np.mean(all_scores) if all_scores else 0
            
            return {
                'total_recommendations': total_recommendations,
                'program_type_preferences': preferences,
                'average_confidence': avg_confidence,
                'recommendation_sessions': len(history),
                'latest_session': history[0].created_at if history else None
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing recommendation patterns: {e}")
            return {'error': str(e)}


# Global instance
recommendation_engine = RecommendationEngine() 