"""
ML Model Service

This module contains the Flask service integration for ML model operations.
"""

from flask import current_app
import numpy as np
import os
import json
from typing import Dict, List, Any, Optional

from .pipeline import MLTrainingPipeline
from .utils import extract_features_from_survey_response


class MLModelService:
    """Service class for ML model operations."""
    
    def __init__(self):
        self.model_path = None
        self.model = None
        self.is_trained = False
        
    def initialize(self, instance_path: str):
        """Initialize the service with Flask app instance path."""
        self.model_path = os.path.join(instance_path, 'models', 'recommendation_model.pkl')
        
    def load_model(self) -> bool:
        """Load the trained model from disk."""
        try:
            if self.model_path and os.path.exists(self.model_path):
                self.model = MLTrainingPipeline.load_pipeline(self.model_path)
                self.is_trained = True
                current_app.logger.info(f"Model loaded successfully from {self.model_path}")
                return True
            else:
                current_app.logger.warning(f"Model file not found at {self.model_path}")
                return False
        except Exception as e:
            current_app.logger.error(f"Error loading model: {e}")
            return False
    
    def save_model(self) -> bool:
        """Save the current model to disk."""
        try:
            if self.model is not None and self.model_path:
                os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
                self.model.save_pipeline(self.model_path)
                current_app.logger.info(f"Model saved successfully to {self.model_path}")
                return True
            else:
                current_app.logger.warning("No model to save or path not set")
                return False
        except Exception as e:
            current_app.logger.error(f"Error saving model: {e}")
            return False
    
    def train_model(self, survey_responses: List, force_retrain: bool = False) -> Dict[str, Any]:
        """
        Train the ML model using survey response data.
        
        Args:
            survey_responses: List of survey responses from the database
            force_retrain: Whether to retrain even if model exists
            
        Returns:
            Dictionary with training results and metrics
        """
        if self.is_trained and not force_retrain:
            return {"status": "Model already trained", "retrained": False}
        
        try:
            
            X, y, program_mapping = self._prepare_training_data(survey_responses)
            
            if len(X) < 10:  
                return {
                    "status": "Insufficient training data", 
                    "samples": len(X),
                    "minimum_required": 10
                }
            
            
            self.model = MLTrainingPipeline(random_seed=42)
            
            training_results = self.model.train_model(
                X, y,
                hidden_sizes=[32, 16],
                activation='relu',
                learning_rate=0.01,
                epochs=100,
                batch_size=min(32, len(X) // 4),
                test_size=0.2,
                val_size=0.15,
                verbose=False
            )
            
            self.is_trained = True
            
            
            self.save_model()
            
            
            self._save_program_mapping(program_mapping)
            
            return {
                "status": "Training completed successfully",
                "retrained": True,
                "training_samples": len(X),
                "test_accuracy": training_results['test_metrics']['accuracy'],
                "test_f1_score": training_results['test_metrics']['f1_macro'],
                "program_mapping": program_mapping
            }
            
        except Exception as e:
            current_app.logger.error(f"Error training model: {e}")
            return {"status": f"Training failed: {str(e)}"}
    
    def predict_programs(self, survey_data: Dict[str, Any], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Predict suitable programs for a given survey response.
        
        Args:
            survey_data: Survey response data
            top_k: Number of top predictions to return
            
        Returns:
            List of predicted programs with confidence scores
        """
        if not self.is_trained or self.model is None:
            if not self.load_model():
                return []
        
        try:
            
            features = extract_features_from_survey_response(survey_data)
            
            
            predictions = self.model.predict(features)
            class_predictions = self.model.predict_classes(features)
            
            
            program_mapping = self._load_program_mapping()
            
            if program_mapping is None:
                current_app.logger.warning("Program mapping not found")
                return []
            
            
            recommendations = []
            
            if len(predictions.shape) > 1 and predictions.shape[1] > 1:
                
                prediction_probs = predictions[0]
                sorted_indices = np.argsort(prediction_probs)[::-1]
                
                for i, idx in enumerate(sorted_indices[:top_k]):
                    if idx < len(program_mapping):
                        program_info = program_mapping[idx]
                        recommendations.append({
                            'program_id': program_info['id'],
                            'program_name': program_info['name'],
                            'school_name': program_info['school_name'],
                            'confidence': float(prediction_probs[idx]),
                            'rank': i + 1
                        })
            else:
                
                confidence = float(predictions[0][0])
                predicted_class = int(class_predictions[0])
                
                if predicted_class < len(program_mapping):
                    program_info = program_mapping[predicted_class]
                    recommendations.append({
                        'program_id': program_info['id'],
                        'program_name': program_info['name'],
                        'school_name': program_info['school_name'],
                        'confidence': confidence,
                        'rank': 1
                    })
            
            return recommendations
            
        except Exception as e:
            current_app.logger.error(f"Error making predictions: {e}")
            return []
    
    def _prepare_training_data(self, survey_responses: List) -> tuple:
        """Prepare training data from survey responses."""
        X = []
        y = []
        program_mapping = {}
        
        
        
        
        programs = [
            {'id': 1, 'name': 'Computer Science', 'school_name': 'Technical University'},
            {'id': 2, 'name': 'Engineering', 'school_name': 'Technical University'},
            {'id': 3, 'name': 'Arts', 'school_name': 'University of Arts'},
            {'id': 4, 'name': 'Business', 'school_name': 'Business School'}
        ]
        
        for i, program in enumerate(programs):
            program_mapping[i] = program
        
        
        if len(survey_responses) == 0:
            X, y = self._create_synthetic_training_data(program_mapping)
        else:
            
            for response in survey_responses:
                try:
                    
                    answers = response.get_answers() if hasattr(response, 'get_answers') else {}
                    features = extract_features_from_survey_response(answers)
                    X.append(features.flatten())
                    
                    
                    y.append(0)  
                    
                except Exception as e:
                    current_app.logger.warning(f"Error processing survey response: {e}")
                    continue
        
        return np.array(X), np.array(y), program_mapping
    
    def _create_synthetic_training_data(self, program_mapping: Dict) -> tuple:
        """Create synthetic training data when no real data is available."""
        current_app.logger.info("Creating synthetic training data")
        
        X = []
        y = []
        np.random.seed(42)
        
        
        for program_idx, program_info in program_mapping.items():
            program_name = program_info['name'].lower()
            
            
            for _ in range(50):  
                if 'computer' in program_name or 'engineering' in program_name:
                    
                    survey = {
                        'math_interest': np.random.normal(8, 1),
                        'science_interest': np.random.normal(7, 1),
                        'art_interest': np.random.normal(3, 1),
                        'sports_interest': np.random.normal(5, 1.5),
                        'study_hours_per_day': np.random.normal(5, 1),
                        'preferred_subject': 'Mathematics',
                        'career_goal': 'Engineer',
                        'extracurricular': np.random.choice([True, False]),
                        'leadership_experience': np.random.choice([True, False]),
                        'team_preference': np.random.choice([True, False]),
                        'languages_spoken': ['Bulgarian', 'English'],
                        'grades_average': np.random.normal(5.5, 0.3)
                    }
                elif 'arts' in program_name:
                    
                    survey = {
                        'math_interest': np.random.normal(4, 1),
                        'science_interest': np.random.normal(5, 1),
                        'art_interest': np.random.normal(9, 0.5),
                        'sports_interest': np.random.normal(6, 1),
                        'study_hours_per_day': np.random.normal(4, 1),
                        'preferred_subject': 'Art',
                        'career_goal': 'Designer',
                        'extracurricular': True,
                        'leadership_experience': np.random.choice([True, False]),
                        'team_preference': True,
                        'languages_spoken': ['Bulgarian', 'English'],
                        'grades_average': np.random.normal(5.2, 0.4)
                    }
                elif 'business' in program_name:
                    
                    survey = {
                        'math_interest': np.random.normal(6, 1),
                        'science_interest': np.random.normal(5, 1),
                        'art_interest': np.random.normal(5, 1),
                        'sports_interest': np.random.normal(6, 1),
                        'study_hours_per_day': np.random.normal(4, 1),
                        'preferred_subject': 'Economics',
                        'career_goal': 'Business',
                        'extracurricular': True,
                        'leadership_experience': True,
                        'team_preference': True,
                        'languages_spoken': ['Bulgarian', 'English'],
                        'grades_average': np.random.normal(5.3, 0.3)
                    }
                else:
                    
                    survey = {
                        'math_interest': np.random.normal(6, 2),
                        'science_interest': np.random.normal(6, 2),
                        'art_interest': np.random.normal(6, 2),
                        'sports_interest': np.random.normal(6, 2),
                        'study_hours_per_day': np.random.normal(4, 1),
                        'preferred_subject': 'General',
                        'career_goal': 'General',
                        'extracurricular': np.random.choice([True, False]),
                        'leadership_experience': np.random.choice([True, False]),
                        'team_preference': np.random.choice([True, False]),
                        'languages_spoken': ['Bulgarian', 'English'],
                        'grades_average': np.random.normal(5.0, 0.5)
                    }
                
                
                for key, value in survey.items():
                    if isinstance(value, (int, float)) and key not in ['grades_average']:
                        survey[key] = max(1, min(10, value))
                    elif key == 'grades_average':
                        survey[key] = max(2.0, min(6.0, value))
                
                features = extract_features_from_survey_response(survey)
                X.append(features.flatten())
                y.append(program_idx)
        
        return X, y
    
    def _save_program_mapping(self, program_mapping: Dict):
        """Save program mapping to a JSON file."""
        try:
            if self.model_path:
                mapping_path = os.path.join(os.path.dirname(self.model_path), 'program_mapping.json')
                os.makedirs(os.path.dirname(mapping_path), exist_ok=True)
                
                with open(mapping_path, 'w') as f:
                    json.dump(program_mapping, f)
        except Exception as e:
            current_app.logger.error(f"Error saving program mapping: {e}")
    
    def _load_program_mapping(self) -> Optional[Dict]:
        """Load program mapping from JSON file."""
        try:
            if self.model_path:
                mapping_path = os.path.join(os.path.dirname(self.model_path), 'program_mapping.json')
                if os.path.exists(mapping_path):
                    with open(mapping_path, 'r') as f:
                        mapping = json.load(f)
                    
                    return {int(k): v for k, v in mapping.items()}
            return None
        except Exception as e:
            current_app.logger.error(f"Error loading program mapping: {e}")
            return None 