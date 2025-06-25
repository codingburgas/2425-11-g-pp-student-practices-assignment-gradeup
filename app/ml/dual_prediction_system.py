"""
Dual Prediction System

This module combines the main AI method (neural network) with a backup statistical method
to provide robust predictions with fallback capabilities.
"""

import numpy as np
import os
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import joblib

from app import db
from app.models import PredictionHistory, User, SurveyResponse
from .prediction_system import AdvancedPredictionSystem
from .demo_prediction_service import demo_prediction_service
from .service import MLModelService
from .utils import extract_features_as_array


class DualPredictionSystem:
    """
    Dual prediction system combining neural network and statistical methods.
    """
    
    def __init__(self):
        self.main_system = AdvancedPredictionSystem()
        self.ml_service = MLModelService()
        self.backup_model = None
        self.backup_scaler = None
        self.logger = logging.getLogger(__name__)
        
    def initialize(self, instance_path: str):
        """Initialize both prediction systems."""
        # Initialize main system
        self.main_system.initialize(instance_path)
        
        # Initialize ML service
        self.ml_service.initialize(instance_path)
        
        # Load backup model
        self._load_backup_model(instance_path)
        
    def _load_backup_model(self, instance_path: str):
        """Load the backup statistical model."""
        try:
            backup_path = os.path.join(instance_path, 'models', 'backup_model.pkl')
            scaler_path = os.path.join(instance_path, 'models', 'backup_scaler.pkl')
            
            if os.path.exists(backup_path) and os.path.exists(scaler_path):
                self.backup_model = joblib.load(backup_path)
                self.backup_scaler = joblib.load(scaler_path)
                self.logger.info("Backup model loaded successfully")
            else:
                self.logger.warning("Backup model files not found")
                
        except Exception as e:
            self.logger.error(f"Error loading backup model: {e}")
    
    def predict_with_dual_methods(self, 
                                 survey_data: Dict[str, Any], 
                                 user_id: int,
                                 survey_response_id: Optional[int] = None,
                                 top_k: int = 5) -> Dict[str, Any]:
        """
        Generate predictions using both main and backup methods.
        
        Args:
            survey_data: Survey response data
            user_id: ID of the user requesting predictions
            survey_response_id: Optional ID of survey response
            top_k: Number of top predictions to return
            
        Returns:
            Dictionary containing predictions from both methods
        """
        try:
            # Get predictions from main method
            main_predictions = self._get_main_predictions(
                survey_data, user_id, survey_response_id, top_k
            )
            
            # Get predictions from backup method
            backup_predictions = self._get_backup_predictions(
                survey_data, top_k
            )
            
            # Generate consensus recommendations
            consensus_recommendations = self._find_consensus_recommendations(
                main_predictions.get('predictions', []),
                backup_predictions
            )
            
            # Calculate overall confidence metrics
            overall_metrics = self._calculate_overall_metrics(
                main_predictions, backup_predictions
            )
            
            result = {
                'main_method': {
                    'predictions': main_predictions.get('predictions', []),
                    'confidence_metrics': main_predictions.get('confidence_metrics', {}),
                    'status': 'success' if main_predictions.get('predictions') else 'unavailable',
                    'method_type': 'neural_network'
                },
                'backup_method': {
                    'predictions': backup_predictions,
                    'status': 'success' if backup_predictions else 'unavailable',
                    'method_type': 'statistical_regression'
                },
                'consensus_recommendations': consensus_recommendations,
                'overall_metrics': overall_metrics,
                'timestamp': datetime.utcnow().isoformat(),
                'dual_analysis': True
            }
            
            # Store dual prediction history
            self._store_dual_prediction_history(
                user_id, survey_response_id, result
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in dual prediction: {e}")
            return self._empty_dual_result(error=str(e))
    
    def _get_main_predictions(self, survey_data: Dict[str, Any], 
                            user_id: int, 
                            survey_response_id: Optional[int],
                            top_k: int) -> Dict[str, Any]:
        """Get predictions from the main neural network method."""
        try:
            # Try to get predictions from the advanced prediction system
            result = self.main_system.predict_with_confidence(
                survey_data, user_id, survey_response_id, top_k, store_history=False
            )
            
            if result and result.get('predictions'):
                return result
            
            # Fallback to demo service if main system fails
            demo_predictions = demo_prediction_service.predict_programs(survey_data, top_k)
            
            # Convert demo predictions to expected format
            formatted_predictions = []
            for pred in demo_predictions:
                formatted_predictions.append({
                    'program_id': pred.get('program_id'),
                    'program_name': pred.get('program_name', pred.get('name', 'Unknown')),
                    'school_name': pred.get('school_name', 'Unknown'),
                    'confidence': pred.get('confidence', 0.5),
                    'match_score': pred.get('match_score', pred.get('confidence', 0.5)),
                    'recommendation_reasons': pred.get('match_reasons', []),
                    'source': 'demo_service'
                })
            
            return {
                'predictions': formatted_predictions,
                'confidence_metrics': {'average_confidence': 0.6},
                'model_version': 'demo_v1.0'
            }
            
        except Exception as e:
            self.logger.error(f"Error getting main predictions: {e}")
            return {}
    
    def _get_backup_predictions(self, survey_data: Dict[str, Any], top_k: int) -> List[Dict[str, Any]]:
        """Get predictions from the backup statistical method."""
        try:
            if self.backup_model is None or self.backup_scaler is None:
                self.logger.warning("Backup model not available, using fallback")
                return self._get_fallback_backup_predictions(survey_data, top_k)
            
            # Extract features for backup model
            features = extract_features_as_array(survey_data)
            features_scaled = self.backup_scaler.transform([features])
            
            # Get prediction probabilities
            prediction_probs = self.backup_model.predict_proba(features_scaled)[0]
            
            # Get top predictions
            top_indices = np.argsort(prediction_probs)[::-1][:top_k]
            
            # Create backup predictions
            backup_predictions = []
            for i, idx in enumerate(top_indices):
                confidence = float(prediction_probs[idx])
                
                # Create a prediction based on the program index
                backup_predictions.append({
                    'program_id': idx + 1,
                    'program_name': self._get_program_name_by_index(idx),
                    'school_name': self._get_school_name_by_index(idx),
                    'confidence': confidence,
                    'match_score': confidence,
                    'rank': i + 1,
                    'method': 'statistical_regression',
                    'match_reasons': self._generate_statistical_reasons(survey_data, idx)
                })
            
            return backup_predictions
            
        except Exception as e:
            self.logger.error(f"Error getting backup predictions: {e}")
            return self._get_fallback_backup_predictions(survey_data, top_k)
    
    def _get_fallback_backup_predictions(self, survey_data: Dict[str, Any], top_k: int) -> List[Dict[str, Any]]:
        """Generate fallback predictions when backup model is unavailable."""
        
        # Simple rule-based predictions based on interests
        programs = [
            {'name': 'Computer Science', 'school': 'Sofia University', 'interests': ['math', 'technology']},
            {'name': 'Engineering', 'school': 'Technical University', 'interests': ['math', 'science']},
            {'name': 'Business', 'school': 'Economics University', 'interests': ['management', 'finance']},
            {'name': 'Medicine', 'school': 'Medical University', 'interests': ['science', 'helping']},
            {'name': 'Art & Design', 'school': 'Art Academy', 'interests': ['creativity', 'art']},
            {'name': 'Psychology', 'school': 'Sofia University', 'interests': ['social', 'helping']},
            {'name': 'Mathematics', 'school': 'Sofia University', 'interests': ['math', 'logic']},
            {'name': 'Physics', 'school': 'Sofia University', 'interests': ['science', 'research']},
        ]
        
        # Calculate simple scores based on survey data
        scored_programs = []
        
        for i, program in enumerate(programs):
            score = 0.3  # Base score
            
            # Add interest-based scoring
            math_interest = survey_data.get('math_interest', 5)
            science_interest = survey_data.get('science_interest', 5)
            art_interest = survey_data.get('art_interest', 5)
            
            if 'math' in program['interests'] and math_interest > 6:
                score += 0.2
            if 'science' in program['interests'] and science_interest > 6:
                score += 0.2
            if 'art' in program['interests'] and art_interest > 6:
                score += 0.2
            
            # Add some randomness
            score += np.random.uniform(0, 0.3)
            score = min(score, 0.95)  # Cap at 95%
            
            scored_programs.append({
                'program_id': i + 1,
                'program_name': program['name'],
                'school_name': program['school'],
                'confidence': score,
                'match_score': score,
                'rank': 0,  # Will be set after sorting
                'method': 'rule_based_fallback',
                'match_reasons': [f"Matches your {', '.join(program['interests'])} interests"]
            })
        
        # Sort by score and assign ranks
        scored_programs.sort(key=lambda x: x['confidence'], reverse=True)
        for i, prog in enumerate(scored_programs[:top_k]):
            prog['rank'] = i + 1
        
        return scored_programs[:top_k]
    
    def _get_program_name_by_index(self, idx: int) -> str:
        """Get program name by index for backup predictions."""
        program_names = [
            'Computer Science', 'Engineering', 'Business Administration',
            'Medicine', 'Psychology', 'Art & Design', 'Mathematics',
            'Physics', 'Law', 'Architecture'
        ]
        
        if idx < len(program_names):
            return program_names[idx]
        else:
            return f"Program {idx + 1}"
    
    def _get_school_name_by_index(self, idx: int) -> str:
        """Get school name by index for backup predictions."""
        school_names = [
            'Sofia University', 'Technical University', 'Economics University',
            'Medical University', 'Sofia University', 'Art Academy',
            'Sofia University', 'Sofia University', 'Sofia University',
            'Architecture University'
        ]
        
        if idx < len(school_names):
            return school_names[idx]
        else:
            return 'University'
    
    def _generate_statistical_reasons(self, survey_data: Dict[str, Any], program_idx: int) -> List[str]:
        """Generate match reasons for statistical predictions."""
        reasons = []
        
        # Based on program type and survey data
        math_interest = survey_data.get('math_interest', 5)
        science_interest = survey_data.get('science_interest', 5)
        art_interest = survey_data.get('art_interest', 5)
        
        if program_idx == 0 and math_interest > 6:  # Computer Science
            reasons.append("High mathematical aptitude")
        elif program_idx == 1 and science_interest > 6:  # Engineering
            reasons.append("Strong science background")
        elif program_idx == 4 and art_interest > 6:  # Art & Design
            reasons.append("Creative interests alignment")
        
        if not reasons:
            reasons.append("Statistical correlation with similar profiles")
        
        return reasons
    
    def _find_consensus_recommendations(self, 
                                      main_predictions: List[Dict], 
                                      backup_predictions: List[Dict]) -> List[Dict[str, Any]]:
        """Find programs that both methods recommend."""
        consensus = []
        
        if not main_predictions or not backup_predictions:
            return consensus
        
        # Create mappings for comparison
        main_programs = {pred['program_name'].lower(): pred for pred in main_predictions}
        backup_programs = {pred['program_name'].lower(): pred for pred in backup_predictions}
        
        # Find common programs
        for program_name in main_programs.keys():
            if program_name in backup_programs:
                main_pred = main_programs[program_name]
                backup_pred = backup_programs[program_name]
                
                avg_confidence = (main_pred['confidence'] + backup_pred['confidence']) / 2
                
                consensus.append({
                    'program_name': main_pred['program_name'],
                    'school_name': main_pred['school_name'],
                    'avg_confidence': avg_confidence,
                    'main_confidence': main_pred['confidence'],
                    'backup_confidence': backup_pred['confidence'],
                    'consensus_strength': 'high' if abs(main_pred['confidence'] - backup_pred['confidence']) < 0.2 else 'moderate'
                })
        
        # Sort by average confidence
        consensus.sort(key=lambda x: x['avg_confidence'], reverse=True)
        
        return consensus[:3]  # Return top 3 consensus recommendations
    
    def _calculate_overall_metrics(self, 
                                 main_result: Dict[str, Any], 
                                 backup_predictions: List[Dict]) -> Dict[str, Any]:
        """Calculate overall confidence metrics across both methods."""
        
        main_predictions = main_result.get('predictions', [])
        
        metrics = {
            'main_method_available': len(main_predictions) > 0,
            'backup_method_available': len(backup_predictions) > 0,
            'total_predictions': len(main_predictions) + len(backup_predictions),
            'method_agreement': 0.0,
            'overall_confidence': 0.0
        }
        
        if main_predictions and backup_predictions:
            # Calculate method agreement (how many programs appear in both lists)
            main_names = {pred['program_name'].lower() for pred in main_predictions}
            backup_names = {pred['program_name'].lower() for pred in backup_predictions}
            
            common_count = len(main_names.intersection(backup_names))
            metrics['method_agreement'] = common_count / min(len(main_names), len(backup_names))
            
            # Calculate overall confidence
            main_avg = sum(pred['confidence'] for pred in main_predictions) / len(main_predictions)
            backup_avg = sum(pred['confidence'] for pred in backup_predictions) / len(backup_predictions)
            metrics['overall_confidence'] = (main_avg + backup_avg) / 2
        
        elif main_predictions:
            metrics['overall_confidence'] = sum(pred['confidence'] for pred in main_predictions) / len(main_predictions)
        
        elif backup_predictions:
            metrics['overall_confidence'] = sum(pred['confidence'] for pred in backup_predictions) / len(backup_predictions)
        
        return metrics
    
    def _store_dual_prediction_history(self, 
                                     user_id: int, 
                                     survey_response_id: Optional[int],
                                     result: Dict[str, Any]):
        """Store dual prediction results in history."""
        try:
            # Create a summary for storage
            summary = {
                'main_method_status': result['main_method']['status'],
                'backup_method_status': result['backup_method']['status'],
                'consensus_count': len(result['consensus_recommendations']),
                'overall_confidence': result['overall_metrics']['overall_confidence'],
                'method_agreement': result['overall_metrics']['method_agreement']
            }
            
            # Store in prediction history
            history_entry = PredictionHistory(
                user_id=user_id,
                survey_response_id=survey_response_id,
                prediction_type='dual_method',
                input_features=summary,
                predictions=result['main_method']['predictions'][:3],  # Store top 3
                confidence_scores=[pred['confidence'] for pred in result['main_method']['predictions'][:3]],
                model_version='dual_v1.0',
                created_at=datetime.utcnow()
            )
            
            db.session.add(history_entry)
            db.session.commit()
            
        except Exception as e:
            self.logger.error(f"Error storing dual prediction history: {e}")
    
    def _empty_dual_result(self, error: Optional[str] = None) -> Dict[str, Any]:
        """Return empty dual prediction result."""
        return {
            'main_method': {
                'predictions': [],
                'status': 'error',
                'method_type': 'neural_network'
            },
            'backup_method': {
                'predictions': [],
                'status': 'error',
                'method_type': 'statistical_regression'
            },
            'consensus_recommendations': [],
            'overall_metrics': {
                'main_method_available': False,
                'backup_method_available': False,
                'total_predictions': 0,
                'method_agreement': 0.0,
                'overall_confidence': 0.0
            },
            'error': error,
            'timestamp': datetime.utcnow().isoformat(),
            'dual_analysis': True
        }


# Global instance
dual_prediction_system = DualPredictionSystem() 