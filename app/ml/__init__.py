"""
ML Module for University Recommendation System

This module contains all machine learning functionality including:
- Custom neural network implementation
- Training pipelines
- Model evaluation
- Flask integration
- Model persistence
"""

from .models import CustomNeuralNetwork, ActivationFunctions, LossFunctions
from .pipeline import MLTrainingPipeline
from .evaluator import ModelEvaluator
from .service import MLModelService
from .utils import create_sample_dataset, extract_features_from_survey_response

__version__ = "1.0.0"
__author__ = "University Recommendation System Team"

__all__ = [
    'CustomNeuralNetwork',
    'ActivationFunctions', 
    'LossFunctions',
    'MLTrainingPipeline',
    'ModelEvaluator',
    'MLModelService',
    'create_sample_dataset',
    'extract_features_from_survey_response'
] 