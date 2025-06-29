"""
Advanced Prediction System for School Recommendations

This module provides enhanced prediction capabilities including:
- Advanced confidence scoring
- Batch prediction processing  
- Prediction history storage
- Pipeline optimization
"""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Any, Optional, Tuple
import json
import logging
from datetime import datetime
from flask import current_app

from app import db
from app.models import PredictionHistory, User, SurveyResponse, Program, School
from .service import MLModelService
from .utils import extract_features_from_survey_response
from .demo_prediction_service import demo_prediction_service


class AdvancedPredictionSystem:
    """
    Advanced prediction system with enhanced capabilities for school recommendations.
    """
    
    def __init__(self):
        self.ml_service = MLModelService()
        self.model_version = "v2.0"
        self.confidence_threshold = 0.3
        self.logger = logging.getLogger(__name__)
        
    def initialize(self, instance_path: str):
        """Initialize the prediction system."""
        self.ml_service.initialize(instance_path)
        
    def predict_with_confidence(self, 
                              survey_data: Dict[str, Any], 
                              user_id: int,
                              survey_response_id: Optional[int] = None,
                              top_k: int = 5,
                              store_history: bool = True) -> Dict[str, Any]:
        """
        Generate predictions with advanced confidence scoring.
        
        Args:
            survey_data: Survey response data
            user_id: ID of the user requesting predictions
            survey_response_id: Optional ID of survey response
            top_k: Number of top predictions to return
            store_history: Whether to store prediction in history
            
        Returns:
            Dictionary containing predictions with confidence analysis
        """
        try:
            # Extract features
            features = extract_features_from_survey_response(survey_data)
            
            # Get base predictions from ML service or demo service
            base_predictions = []
            
            # Try ML service first
            if self.ml_service.is_trained:
                try:
                    self.logger.info("ML service is trained, attempting to use trained model")
                    base_predictions = self.ml_service.predict_programs(survey_data, top_k * 2)
                    self.logger.info(f"ML service returned {len(base_predictions)} predictions")
                except Exception as e:
                    self.logger.warning(f"ML service failed, falling back to demo: {e}")
            else:
                self.logger.info("ML service is not trained, using demo service")
            
            # Fall back to demo service if no ML predictions
            if len(base_predictions) == 0:
                try:
                    demo_predictions = demo_prediction_service.predict_programs(survey_data, top_k * 2)
                    self.logger.info("Using demo prediction service")
                    
                    # Improve variance in demo predictions to prevent 24% match scores
                    for pred in demo_predictions:
                        # Generate more realistic confidence scores based on interest alignment
                        interest_score = 0
                        
                        # Extract interest scores from survey data
                        math_interest = survey_data.get('math_interest', 5)
                        science_interest = survey_data.get('science_interest', 5)
                        art_interest = survey_data.get('art_interest', 5)
                        sports_interest = survey_data.get('sports_interest', 5)
                        
                        # Match program to interests
                        program_name = pred.get('program_name', '').lower()
                        if any(word in program_name for word in ['computer', 'software', 'engineering', 'math']):
                            interest_score += math_interest * 1.5
                        elif any(word in program_name for word in ['science', 'physics', 'chemistry', 'biology']):
                            interest_score += science_interest * 1.5
                        elif any(word in program_name for word in ['art', 'design', 'music', 'literature']):
                            interest_score += art_interest * 1.5
                        elif any(word in program_name for word in ['sport', 'physical']):
                            interest_score += sports_interest * 1.5
                        
                        # Add some variance based on career goals
                        career_goal = survey_data.get('career_goal', '').lower()
                        if career_goal and career_goal in program_name:
                            interest_score += 15
                            
                        # Normalize to decimal (0-1) with better distribution
                        confidence_decimal = min(0.95, max(0.30, interest_score * 0.03))
                        pred['confidence'] = confidence_decimal
                        pred['match_score'] = confidence_decimal
                        
                    base_predictions = demo_predictions
                except Exception as e:
                    self.logger.error(f"Demo service also failed: {e}")
            
            if len(base_predictions) == 0:
                return self._empty_prediction_result(error="No prediction service available")
            
            # Apply advanced confidence scoring
            enhanced_predictions = self._enhance_predictions_with_confidence(
                base_predictions, features, survey_data
            )
            
            # Filter by confidence threshold
            filtered_predictions = [
                pred for pred in enhanced_predictions 
                if pred['enhanced_confidence'] >= self.confidence_threshold
            ]
            
            # Ensure we have at least some results
            if len(filtered_predictions) == 0 and len(enhanced_predictions) > 0:
                filtered_predictions = enhanced_predictions[:max(1, top_k // 2)]
            
            # If we still have no predictions, create fallback recommendations
            if len(filtered_predictions) == 0:
                self.logger.warning("No predictions available, creating fallback recommendations")
                filtered_predictions = self._create_fallback_recommendations(survey_data, top_k)
            
            # Take top K results
            final_predictions = filtered_predictions[:top_k]
            
            # Calculate overall confidence metrics
            confidence_metrics = self._calculate_confidence_metrics(final_predictions)
            
            # Prepare result
            result = {
                'predictions': final_predictions,
                'confidence_metrics': confidence_metrics,
                'model_version': self.model_version,
                'timestamp': datetime.utcnow().isoformat(),
                'total_candidates': len(base_predictions),
                'filtered_count': len(filtered_predictions)
            }
            
            # Store prediction history if requested
            if store_history:
                self._store_prediction_history(
                    user_id=user_id,
                    survey_response_id=survey_response_id,
                    input_features=features,
                    predictions=final_predictions,
                    confidence_scores=[pred['enhanced_confidence'] for pred in final_predictions],
                    prediction_type='individual'
                )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in predict_with_confidence: {e}")
            return self._empty_prediction_result(error=str(e))
    
    def batch_predict(self, 
                     prediction_requests: List[Dict[str, Any]],
                     store_history: bool = True) -> List[Dict[str, Any]]:
        """
        Process multiple predictions efficiently in batch.
        
        Args:
            prediction_requests: List of prediction request dictionaries
                Each should contain: survey_data, user_id, survey_response_id (optional), top_k (optional)
            store_history: Whether to store predictions in history
            
        Returns:
            List of prediction results
        """
        results = []
        batch_start_time = datetime.utcnow()
        
        try:
            self.logger.info(f"Starting batch prediction for {len(prediction_requests)} requests")
            
            # Process each request
            for i, request in enumerate(prediction_requests):
                try:
                    survey_data = request.get('survey_data', {})
                    user_id = request.get('user_id')
                    survey_response_id = request.get('survey_response_id')
                    top_k = request.get('top_k', 5)
                    
                    if not user_id:
                        results.append(self._empty_prediction_result(error="Missing user_id"))
                        continue
                    
                    # Get prediction with individual history storage disabled
                    # We'll handle batch storage separately
                    result = self.predict_with_confidence(
                        survey_data=survey_data,
                        user_id=user_id,
                        survey_response_id=survey_response_id,
                        top_k=top_k,
                        store_history=False
                    )
                    
                    result['batch_index'] = i
                    results.append(result)
                    
                    # Store individual prediction history if requested
                    if store_history and result.get('predictions'):
                        features = extract_features_from_survey_response(survey_data)
                        self._store_prediction_history(
                            user_id=user_id,
                            survey_response_id=survey_response_id,
                            input_features=features,
                            predictions=result['predictions'],
                            confidence_scores=[pred['enhanced_confidence'] 
                                             for pred in result['predictions']],
                            prediction_type='batch'
                        )
                    
                except Exception as e:
                    self.logger.error(f"Error processing batch request {i}: {e}")
                    results.append(self._empty_prediction_result(error=str(e)))
            
            # Calculate batch statistics
            batch_duration = (datetime.utcnow() - batch_start_time).total_seconds()
            successful_predictions = len([r for r in results if r.get('predictions')])
            
            self.logger.info(f"Batch prediction completed: {successful_predictions}/{len(prediction_requests)} successful in {batch_duration:.2f}s")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in batch_predict: {e}")
            return [self._empty_prediction_result(error=str(e)) for _ in prediction_requests]
    
    def get_prediction_history(self, 
                              user_id: int, 
                              limit: int = 10,
                              prediction_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Retrieve prediction history for a user.
        
        Args:
            user_id: ID of the user
            limit: Maximum number of records to retrieve
            prediction_type: Optional filter by prediction type ('individual' or 'batch')
            
        Returns:
            List of prediction history records
        """
        try:
            query = PredictionHistory.query.filter_by(user_id=user_id)
            
            if prediction_type:
                query = query.filter_by(prediction_type=prediction_type)
            
            history_records = query.order_by(PredictionHistory.created_at.desc()).limit(limit).all()
            
            history_list = []
            for record in history_records:
                history_list.append({
                    'id': record.id,
                    'created_at': record.created_at.isoformat(),
                    'prediction_type': record.prediction_type,
                    'model_version': record.model_version,
                    'predictions': record.get_predictions(),
                    'confidence_scores': record.get_confidence_scores(),
                    'input_features': record.get_input_features(),
                    'survey_response_id': record.survey_response_id
                })
            
            return history_list
            
        except Exception as e:
            self.logger.error(f"Error retrieving prediction history: {e}")
            return []
    
    def analyze_prediction_patterns(self, user_id: int) -> Dict[str, Any]:
        """
        Analyze prediction patterns and trends for a user.
        
        Args:
            user_id: ID of the user
            
        Returns:
            Dictionary containing pattern analysis
        """
        try:
            history_records = PredictionHistory.query.filter_by(user_id=user_id).all()
            
            if not history_records:
                return {'status': 'No prediction history found'}
            
            # Extract data for analysis
            all_predictions = []
            confidence_scores = []
            timestamps = []
            
            for record in history_records:
                predictions = record.get_predictions()
                scores = record.get_confidence_scores()
                
                all_predictions.extend(predictions)
                confidence_scores.extend(scores)
                timestamps.append(record.created_at)
            
            # Calculate statistics
            avg_confidence = np.mean(confidence_scores) if confidence_scores else 0
            confidence_trend = self._calculate_confidence_trend(confidence_scores, timestamps)
            
            # Find most recommended programs/schools
            program_counts = {}
            school_counts = {}
            
            for pred in all_predictions:
                program_name = pred.get('program_name', 'Unknown')
                school_name = pred.get('school_name', 'Unknown')
                
                program_counts[program_name] = program_counts.get(program_name, 0) + 1
                school_counts[school_name] = school_counts.get(school_name, 0) + 1
            
            # Sort by frequency
            top_programs = sorted(program_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            top_schools = sorted(school_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            
            return {
                'total_predictions': len(history_records),
                'total_recommendations': len(all_predictions),
                'average_confidence': round(avg_confidence, 3),
                'confidence_trend': confidence_trend,
                'top_recommended_programs': top_programs,
                'top_recommended_schools': top_schools,
                'prediction_frequency': len(history_records) / max(1, (datetime.utcnow() - min(timestamps)).days) if timestamps else 0
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing prediction patterns: {e}")
            return {'error': str(e)}
    
    def _enhance_predictions_with_confidence(self, 
                                           base_predictions: List[Dict], 
                                           features: Dict,
                                           survey_data: Dict) -> List[Dict]:
        """
        Enhance predictions with advanced confidence scoring.
        """
        enhanced_predictions = []
        
        for pred in base_predictions:
            # Start with base confidence
            base_confidence = pred.get('confidence', 0.5)
            
            # Apply multiple confidence factors
            confidence_factors = []
            
            # Factor 1: Base model confidence
            confidence_factors.append(('base_model', base_confidence, 0.4))
            
            # Factor 2: Feature completeness (more complete surveys = higher confidence)
            feature_completeness = self._calculate_feature_completeness(features)
            confidence_factors.append(('feature_completeness', feature_completeness, 0.2))
            
            # Factor 3: Program popularity (more popular programs might be safer bets)
            popularity_score = self._estimate_program_popularity(pred)
            confidence_factors.append(('popularity', popularity_score, 0.15))
            
            # Factor 4: Consistency with user preferences
            preference_alignment = self._calculate_preference_alignment(pred, survey_data)
            confidence_factors.append(('preference_alignment', preference_alignment, 0.15))
            
            # Factor 5: Domain expertise boost for clear career paths
            domain_clarity = self._assess_domain_clarity(survey_data)
            confidence_factors.append(('domain_clarity', domain_clarity, 0.1))
            
            # Calculate weighted confidence score
            enhanced_confidence = sum(score * weight for _, score, weight in confidence_factors)
            
            # Add detailed confidence breakdown
            pred['enhanced_confidence'] = round(enhanced_confidence, 3)
            pred['original_confidence'] = base_confidence
            pred['confidence_factors'] = {
                name: {'score': round(score, 3), 'weight': weight}
                for name, score, weight in confidence_factors
            }
            
            enhanced_predictions.append(pred)
        
        # Sort by enhanced confidence
        enhanced_predictions.sort(key=lambda x: x['enhanced_confidence'], reverse=True)
        
        return enhanced_predictions
    
    def _calculate_feature_completeness(self, features: Dict) -> float:
        """Calculate how complete the feature set is."""
        if not features:
            return 0.0
        
        # Count non-null, non-zero features
        total_features = len(features)
        complete_features = 0
        
        for v in features.values():
            # Handle different types of values
            if v is None or v == '' or v == 0:
                continue
            elif isinstance(v, (list, tuple)) and len(v) == 0:
                continue
            elif hasattr(v, '__len__') and len(v) == 0:
                continue
            else:
                complete_features += 1
        
        return complete_features / max(1, total_features)
    
    def _estimate_program_popularity(self, prediction: Dict) -> float:
        """Estimate program popularity based on general knowledge."""
        program_name = prediction.get('program_name', '').lower()
        
        # Popular programs get higher scores
        popular_programs = {
            'computer science': 0.9,
            'business': 0.8,
            'engineering': 0.8,
            'medicine': 0.7,
            'psychology': 0.6,
            'marketing': 0.6,
            'finance': 0.7
        }
        
        for prog, score in popular_programs.items():
            if prog in program_name:
                return score
        
        return 0.5  # Default for unknown programs
    
    def _calculate_preference_alignment(self, prediction: Dict, survey_data: Dict) -> float:
        """Calculate how well the prediction aligns with stated preferences."""
        # This is a simplified implementation
        # In practice, you'd analyze specific survey answers
        
        alignment_score = 0.5  # Default neutral score
        
        # Check if program type matches career interests (if available in survey)
        if 'career_interests' in survey_data:
            interests = survey_data['career_interests']
            program_name = prediction.get('program_name', '').lower()
            
            # Simple keyword matching
            for interest in interests if isinstance(interests, list) else [interests]:
                if str(interest).lower() in program_name:
                    alignment_score += 0.2
        
        return min(1.0, alignment_score)
    
    def _assess_domain_clarity(self, survey_data: Dict) -> float:
        """Assess how clear the user is about their domain/career path."""
        clarity_indicators = 0
        total_indicators = 0
        
        # Check for specific career goals
        if 'career_goals' in survey_data:
            total_indicators += 1
            if survey_data['career_goals'] and len(str(survey_data['career_goals'])) > 10:
                clarity_indicators += 1
        
        # Check for subject preferences
        if 'favorite_subjects' in survey_data:
            total_indicators += 1
            subjects = survey_data['favorite_subjects']
            if subjects and (isinstance(subjects, list) and len(subjects) > 0):
                clarity_indicators += 1
        
        return clarity_indicators / max(1, total_indicators)
    
    def _calculate_confidence_metrics(self, predictions: List[Dict]) -> Dict[str, Any]:
        """Calculate overall confidence metrics for a set of predictions."""
        if not predictions:
            return {'status': 'No predictions available'}
        
        confidences = [pred['enhanced_confidence'] for pred in predictions]
        
        return {
            'average_confidence': round(np.mean(confidences), 3),
            'max_confidence': round(max(confidences), 3),
            'min_confidence': round(min(confidences), 3),
            'confidence_std': round(np.std(confidences), 3),
            'high_confidence_count': len([c for c in confidences if c >= 0.7]),
            'medium_confidence_count': len([c for c in confidences if 0.4 <= c < 0.7]),
            'low_confidence_count': len([c for c in confidences if c < 0.4])
        }
    
    def _calculate_confidence_trend(self, confidence_scores: List[float], timestamps: List) -> str:
        """Calculate trend in confidence scores over time."""
        if len(confidence_scores) < 2:
            return 'insufficient_data'
        
        # Simple trend calculation
        recent_scores = confidence_scores[-5:]  # Last 5 predictions
        earlier_scores = confidence_scores[:-5] if len(confidence_scores) > 5 else confidence_scores[:-2]
        
        if not earlier_scores:
            return 'insufficient_data'
        
        recent_avg = np.mean(recent_scores)
        earlier_avg = np.mean(earlier_scores)
        
        if recent_avg > earlier_avg + 0.05:
            return 'improving'
        elif recent_avg < earlier_avg - 0.05:
            return 'declining'
        else:
            return 'stable'
    
    def _store_prediction_history(self, 
                                user_id: int,
                                survey_response_id: Optional[int],
                                input_features: Dict,
                                predictions: List[Dict],
                                confidence_scores: List[float],
                                prediction_type: str):
        """Store prediction in history database."""
        try:
            history_record = PredictionHistory(
                user_id=user_id,
                survey_response_id=survey_response_id,
                model_version=self.model_version,
                prediction_type=prediction_type
            )
            
            history_record.set_input_features(input_features)
            history_record.set_predictions(predictions)
            history_record.set_confidence_scores(confidence_scores)
            
            db.session.add(history_record)
            db.session.commit()
            
            self.logger.info(f"Stored prediction history for user {user_id}")
            
        except Exception as e:
            self.logger.error(f"Error storing prediction history: {e}")
            db.session.rollback()
    
    def _empty_prediction_result(self, error: Optional[str] = None) -> Dict[str, Any]:
        """Return empty prediction result structure."""
        result = {
            'predictions': [],
            'confidence_metrics': {'status': 'No predictions available'},
            'model_version': self.model_version,
            'timestamp': datetime.utcnow().isoformat(),
            'total_candidates': 0,
            'filtered_count': 0
        }
        
        if error:
            result['error'] = error
        
        return result

    def _create_fallback_recommendations(self, survey_data: Dict[str, Any], top_k: int) -> List[Dict[str, Any]]:
        """Create fallback recommendations when no predictions are available."""
        try:
            # Get all programs from database
            programs = Program.query.join(School).limit(top_k).all()
            
            if not programs:
                # Create hardcoded recommendations if no programs in database
                return [
                    {
                        'program_id': 1,
                        'program_name': 'Computer Science',
                        'school_name': 'Sofia University',
                        'match_score': 0.65,
                        'enhanced_confidence': 0.65,
                        'confidence': 65,
                        'recommendation_reasons': ['Default recommendation based on popularity']
                    },
                    {
                        'program_id': 2,
                        'program_name': 'Business Administration',
                        'school_name': 'University of National and World Economy',
                        'match_score': 0.60,
                        'enhanced_confidence': 0.60,
                        'confidence': 60,
                        'recommendation_reasons': ['Default recommendation based on popularity']
                    },
                    {
                        'program_id': 3,
                        'program_name': 'Medicine',
                        'school_name': 'Medical University of Sofia',
                        'match_score': 0.55,
                        'enhanced_confidence': 0.55,
                        'confidence': 55,
                        'recommendation_reasons': ['Default recommendation based on popularity']
                    }
                ]
            
            # Create recommendations from database programs
            fallback_recommendations = []
            for i, program in enumerate(programs):
                # Calculate a reasonable score that decreases with index
                base_score = 0.7 - (i * 0.05)
                score = max(0.3, min(0.9, base_score))
                
                fallback_recommendations.append({
                    'program_id': program.id,
                    'program_name': program.name,
                    'school_name': program.school.name if program.school else 'Unknown University',
                    'match_score': score,
                    'enhanced_confidence': score,
                    'confidence': int(score * 100),
                    'recommendation_reasons': ['Default recommendation based on program availability']
                })
            
            return fallback_recommendations
            
        except Exception as e:
            self.logger.error(f"Error creating fallback recommendations: {e}")
            # Return minimal hardcoded recommendation
            return [{
                'program_id': 1,
                'program_name': 'General Studies',
                'school_name': 'University',
                'match_score': 0.5,
                'enhanced_confidence': 0.5,
                'confidence': 50,
                'recommendation_reasons': ['Emergency fallback recommendation']
            }]


# Global instance
advanced_prediction_system = AdvancedPredictionSystem()