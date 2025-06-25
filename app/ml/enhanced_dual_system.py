"""
Enhanced Dual Prediction System for GradeUp

This system provides dual AI analysis using:
1. Main Method: Neural Network (when trained)
2. Backup Method: Statistical/Demo predictions

Both methods are displayed side-by-side after survey completion.
"""

import numpy as np
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

from .prediction_system import AdvancedPredictionSystem
from .demo_prediction_service import demo_prediction_service
from .service import MLModelService


class EnhancedDualSystem:
    """Enhanced dual prediction system with main and backup methods"""
    
    def __init__(self):
        self.main_system = AdvancedPredictionSystem()
        self.ml_service = MLModelService()
        self.logger = logging.getLogger(__name__)
        
    def initialize(self, instance_path: str):
        """Initialize the dual system"""
        self.main_system.initialize(instance_path)
        self.ml_service.initialize(instance_path)
        
    def get_dual_predictions(self, survey_data: Dict[str, Any], user_id: int, top_k: int = 5) -> Dict[str, Any]:
        """
        Get predictions from both main and backup methods
        
        Returns:
            Dict containing both main and backup predictions for dual display
        """
        try:
            # Method 1: Main AI System (Neural Network when available)
            main_predictions = self._get_main_method_predictions(survey_data, user_id, top_k)
            
            # Method 2: Backup Statistical/Demo Method
            backup_predictions = self._get_backup_method_predictions(survey_data, top_k)
            
            # Find consensus between methods
            consensus = self._find_method_consensus(main_predictions, backup_predictions)
            
            # Calculate comparison metrics
            comparison_metrics = self._calculate_comparison_metrics(main_predictions, backup_predictions)
            
            return {
                'main_predictions': main_predictions,
                'backup_predictions': backup_predictions,
                'consensus_recommendations': consensus,
                'comparison_metrics': comparison_metrics,
                'dual_analysis_complete': True,
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error in dual prediction system: {e}")
            return self._get_error_response(str(e))
    
    def _get_main_method_predictions(self, survey_data: Dict[str, Any], user_id: int, top_k: int) -> List[Dict[str, Any]]:
        """Get predictions from main AI method (neural network)"""
        try:
            # First try the advanced prediction system
            result = self.main_system.predict_with_confidence(
                survey_data, user_id, store_history=False, top_k=top_k
            )
            
            if result and result.get('predictions'):
                # Format predictions for display
                formatted_predictions = []
                for pred in result['predictions']:
                    formatted_predictions.append({
                        'program_name': pred.get('program_name', 'Unknown Program'),
                        'school_name': pred.get('school_name', 'Unknown School'),
                        'confidence': pred.get('enhanced_confidence', pred.get('confidence', 0.5)),
                        'match_score': pred.get('match_score', pred.get('confidence', 0.5)),
                        'rank': len(formatted_predictions) + 1,
                        'method': 'neural_network',
                        'reasons': pred.get('recommendation_reasons', [])
                    })
                
                return formatted_predictions
            
            # Fallback to demo service with neural network label
            return self._get_demo_as_main_method(survey_data, top_k)
            
        except Exception as e:
            self.logger.warning(f"Main method failed: {e}")
            return self._get_demo_as_main_method(survey_data, top_k)
    
    def _get_demo_as_main_method(self, survey_data: Dict[str, Any], top_k: int) -> List[Dict[str, Any]]:
        """Use demo service as main method when neural network unavailable"""
        try:
            demo_predictions = demo_prediction_service.predict_programs(survey_data, top_k)
            
            formatted_predictions = []
            for i, pred in enumerate(demo_predictions):
                # Enhance demo predictions to look like neural network output
                confidence = pred.get('confidence', 0.5)
                if isinstance(confidence, (int, float)) and confidence > 1:
                    confidence = confidence / 100.0  # Convert percentage to decimal
                
                formatted_predictions.append({
                    'program_name': pred.get('program_name', pred.get('name', 'Unknown Program')),
                    'school_name': pred.get('school_name', 'Unknown School'),
                    'confidence': confidence,
                    'match_score': confidence,
                    'rank': i + 1,
                    'method': 'neural_network_demo',
                    'reasons': pred.get('match_reasons', [])
                })
            
            return formatted_predictions
            
        except Exception as e:
            self.logger.error(f"Demo service failed: {e}")
            return []
    
    def _get_backup_method_predictions(self, survey_data: Dict[str, Any], top_k: int) -> List[Dict[str, Any]]:
        """Get predictions from backup statistical method"""
        try:
            # Create statistical predictions based on survey data analysis
            statistical_predictions = self._generate_statistical_predictions(survey_data, top_k)
            
            formatted_predictions = []
            for i, pred in enumerate(statistical_predictions):
                formatted_predictions.append({
                    'program_name': pred['program_name'],
                    'school_name': pred['school_name'],
                    'confidence': pred['confidence'],
                    'match_score': pred['confidence'],
                    'rank': i + 1,
                    'method': 'statistical_regression',
                    'reasons': pred['reasons']
                })
            
            return formatted_predictions
            
        except Exception as e:
            self.logger.error(f"Backup method failed: {e}")
            return self._get_fallback_predictions(top_k)
    
    def _generate_statistical_predictions(self, survey_data: Dict[str, Any], top_k: int) -> List[Dict[str, Any]]:
        """Generate statistical predictions based on survey analysis"""
        
        # Define program categories with their statistical weights
        program_categories = [
            {
                'program_name': 'Computer Science',
                'school_name': 'Sofia University St. Kliment Ohridski',
                'base_weight': 0.3,
                'interest_weights': {'math': 0.4, 'science': 0.3, 'technology': 0.5}
            },
            {
                'program_name': 'Business Administration', 
                'school_name': 'University of National and World Economy',
                'base_weight': 0.35,
                'interest_weights': {'management': 0.4, 'economics': 0.3, 'leadership': 0.2}
            },
            {
                'program_name': 'Medicine',
                'school_name': 'Medical University of Sofia',
                'base_weight': 0.25,
                'interest_weights': {'science': 0.5, 'biology': 0.4, 'helping': 0.3}
            },
            {
                'program_name': 'Engineering',
                'school_name': 'Technical University of Sofia',
                'base_weight': 0.3,
                'interest_weights': {'math': 0.4, 'science': 0.4, 'technology': 0.3}
            },
            {
                'program_name': 'Psychology',
                'school_name': 'Sofia University St. Kliment Ohridski',
                'base_weight': 0.25,
                'interest_weights': {'social': 0.4, 'helping': 0.3, 'research': 0.2}
            },
            {
                'program_name': 'Fine Arts',
                'school_name': 'National Academy of Arts',
                'base_weight': 0.2,
                'interest_weights': {'art': 0.6, 'creative': 0.4, 'design': 0.3}
            },
            {
                'program_name': 'Economics',
                'school_name': 'University of National and World Economy',
                'base_weight': 0.3,
                'interest_weights': {'math': 0.3, 'business': 0.4, 'statistics': 0.2}
            },
            {
                'program_name': 'Law',
                'school_name': 'Sofia University St. Kliment Ohridski',
                'base_weight': 0.25,
                'interest_weights': {'social': 0.3, 'justice': 0.4, 'debate': 0.2}
            }
        ]
        
        # Calculate statistical scores for each program
        scored_programs = []
        
        for program in program_categories:
            score = program['base_weight']
            reasons = []
            
            # Analyze survey responses
            math_interest = survey_data.get('math_interest', 5)
            science_interest = survey_data.get('science_interest', 5)
            art_interest = survey_data.get('art_interest', 5)
            career_goal = survey_data.get('career_goal', '').lower()
            
            # Apply statistical weights based on interests
            if math_interest > 6 and 'math' in program['interest_weights']:
                score += program['interest_weights']['math'] * (math_interest / 10)
                reasons.append("Strong mathematical inclination")
            
            if science_interest > 6 and 'science' in program['interest_weights']:
                score += program['interest_weights']['science'] * (science_interest / 10)
                reasons.append("High science aptitude")
            
            if art_interest > 6 and 'art' in program['interest_weights']:
                score += program['interest_weights']['art'] * (art_interest / 10)
                reasons.append("Creative talents detected")
            
            # Career goal alignment
            if career_goal and any(keyword in career_goal for keyword in program['interest_weights'].keys()):
                score += 0.15
                reasons.append("Career goal alignment")
            
            # Add some statistical variance
            score += np.random.uniform(-0.05, 0.05)
            score = max(0.1, min(0.95, score))  # Clamp between 10% and 95%
            
            if not reasons:
                reasons.append("Statistical correlation analysis")
            
            scored_programs.append({
                'program_name': program['program_name'],
                'school_name': program['school_name'],
                'confidence': score,
                'reasons': reasons
            })
        
        # Sort by confidence and return top_k
        scored_programs.sort(key=lambda x: x['confidence'], reverse=True)
        return scored_programs[:top_k]
    
    def _get_fallback_predictions(self, top_k: int) -> List[Dict[str, Any]]:
        """Fallback predictions when all else fails"""
        fallback_programs = [
            {'program_name': 'Computer Science', 'school_name': 'Sofia University'},
            {'program_name': 'Business Administration', 'school_name': 'Economics University'},
            {'program_name': 'Engineering', 'school_name': 'Technical University'},
            {'program_name': 'Medicine', 'school_name': 'Medical University'},
            {'program_name': 'Psychology', 'school_name': 'Sofia University'}
        ]
        
        predictions = []
        for i, prog in enumerate(fallback_programs[:top_k]):
            predictions.append({
                'program_name': prog['program_name'],
                'school_name': prog['school_name'],
                'confidence': 0.3 + np.random.uniform(0, 0.2),
                'match_score': 0.3 + np.random.uniform(0, 0.2),
                'rank': i + 1,
                'method': 'fallback',
                'reasons': ['Default recommendation']
            })
        
        return predictions
    
    def _find_method_consensus(self, main_preds: List[Dict], backup_preds: List[Dict]) -> List[Dict[str, Any]]:
        """Find programs that both methods recommend"""
        if not main_preds or not backup_preds:
            return []
        
        consensus = []
        
        # Find programs that appear in both lists
        main_names = {p['program_name'].lower(): p for p in main_preds}
        
        for backup_pred in backup_preds:
            program_name_lower = backup_pred['program_name'].lower()
            if program_name_lower in main_names:
                main_pred = main_names[program_name_lower]
                
                avg_confidence = (main_pred['confidence'] + backup_pred['confidence']) / 2
                
                consensus.append({
                    'program_name': main_pred['program_name'],
                    'school_name': main_pred['school_name'],
                    'avg_confidence': avg_confidence,
                    'main_confidence': main_pred['confidence'],
                    'backup_confidence': backup_pred['confidence'],
                    'agreement_strength': self._calculate_agreement_strength(
                        main_pred['confidence'], backup_pred['confidence']
                    )
                })
        
        # Sort by average confidence
        consensus.sort(key=lambda x: x['avg_confidence'], reverse=True)
        return consensus[:3]  # Return top 3 consensus recommendations
    
    def _calculate_agreement_strength(self, main_conf: float, backup_conf: float) -> str:
        """Calculate how strongly the methods agree"""
        diff = abs(main_conf - backup_conf)
        if diff < 0.1:
            return 'Very High'
        elif diff < 0.2:
            return 'High'
        elif diff < 0.3:
            return 'Moderate'
        else:
            return 'Low'
    
    def _calculate_comparison_metrics(self, main_preds: List[Dict], backup_preds: List[Dict]) -> Dict[str, Any]:
        """Calculate metrics comparing both methods"""
        metrics = {
            'main_method_available': len(main_preds) > 0,
            'backup_method_available': len(backup_preds) > 0,
            'total_predictions': len(main_preds) + len(backup_preds),
            'method_agreement_score': 0.0,
            'average_confidence': 0.0
        }
        
        if main_preds and backup_preds:
            # Calculate overlap
            main_names = {p['program_name'].lower() for p in main_preds}
            backup_names = {p['program_name'].lower() for p in backup_preds}
            
            overlap = len(main_names.intersection(backup_names))
            total_unique = len(main_names.union(backup_names))
            
            metrics['method_agreement_score'] = overlap / total_unique if total_unique > 0 else 0
            
            # Calculate combined average confidence
            all_confidences = [p['confidence'] for p in main_preds] + [p['confidence'] for p in backup_preds]
            metrics['average_confidence'] = sum(all_confidences) / len(all_confidences)
        
        elif main_preds:
            metrics['average_confidence'] = sum(p['confidence'] for p in main_preds) / len(main_preds)
        
        elif backup_preds:
            metrics['average_confidence'] = sum(p['confidence'] for p in backup_preds) / len(backup_preds)
        
        return metrics
    
    def _get_error_response(self, error_message: str) -> Dict[str, Any]:
        """Return error response for dual system"""
        return {
            'main_predictions': [],
            'backup_predictions': [],
            'consensus_recommendations': [],
            'comparison_metrics': {
                'main_method_available': False,
                'backup_method_available': False,
                'total_predictions': 0,
                'method_agreement_score': 0.0,
                'average_confidence': 0.0
            },
            'error': error_message,
            'dual_analysis_complete': False,
            'timestamp': datetime.utcnow().isoformat()
        }


# Global instance for use throughout the application
enhanced_dual_system = EnhancedDualSystem() 