"""
Model Evaluation Module

This module contains comprehensive model evaluation metrics and tools.
"""

import numpy as np
from typing import Tuple, Dict, Any
from .models import CustomNeuralNetwork


class ModelEvaluator:
    """
    Comprehensive model evaluation metrics and visualization tools.
    """
    
    @staticmethod
    def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
        """
        Calculate confusion matrix.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            num_classes: Number of classes
            
        Returns:
            Confusion matrix
        """
        cm = np.zeros((num_classes, num_classes), dtype=int)
        for i in range(len(y_true)):
            cm[y_true[i], y_pred[i]] += 1
        return cm
    
    @staticmethod
    def precision_recall_f1(y_true: np.ndarray, y_pred: np.ndarray, 
                           num_classes: int, average: str = 'macro') -> Tuple[float, float, float]:
        """
        Calculate precision, recall, and F1-score.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            num_classes: Number of classes
            average: Averaging method ('macro', 'micro', 'weighted')
            
        Returns:
            Tuple of (precision, recall, f1_score)
        """
        cm = ModelEvaluator.confusion_matrix(y_true, y_pred, num_classes)
        
        precisions = []
        recalls = []
        f1_scores = []
        
        for i in range(num_classes):
            tp = cm[i, i]
            fp = np.sum(cm[:, i]) - tp
            fn = np.sum(cm[i, :]) - tp
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            precisions.append(precision)
            recalls.append(recall)
            f1_scores.append(f1)
        
        if average == 'macro':
            return np.mean(precisions), np.mean(recalls), np.mean(f1_scores)
        elif average == 'micro':
            tp_total = np.sum(np.diag(cm))
            fp_total = np.sum(cm) - tp_total
            fn_total = fp_total  # For multi-class, fp_total == fn_total
            
            precision = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0
            recall = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            return precision, recall, f1
        elif average == 'weighted':
            weights = np.sum(cm, axis=1)
            total_weight = np.sum(weights)
            
            weighted_precision = np.sum(np.array(precisions) * weights) / total_weight
            weighted_recall = np.sum(np.array(recalls) * weights) / total_weight
            weighted_f1 = np.sum(np.array(f1_scores) * weights) / total_weight
            
            return weighted_precision, weighted_recall, weighted_f1
        
        return np.array(precisions), np.array(recalls), np.array(f1_scores)
    
    @staticmethod
    def evaluate_model(model: CustomNeuralNetwork, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """
        Comprehensive model evaluation.
        
        Args:
            model: Trained neural network model
            X_test: Test input data
            y_test: Test labels
            
        Returns:
            Dictionary containing all evaluation metrics
        """
        # Make predictions
        y_pred_proba = model.predict(X_test)
        y_pred = model.predict_classes(X_test)
        
        # Convert y_test to class labels if one-hot encoded
        if y_test.ndim > 1 and y_test.shape[1] > 1:
            y_test_labels = np.argmax(y_test, axis=1)
            num_classes = y_test.shape[1]
        else:
            y_test_labels = y_test.flatten().astype(int)
            num_classes = len(np.unique(y_test_labels))
        
        # Calculate metrics
        accuracy = np.mean(y_pred == y_test_labels)
        
        # Confusion matrix
        cm = ModelEvaluator.confusion_matrix(y_test_labels, y_pred, num_classes)
        
        # Precision, Recall, F1-score
        precision_macro, recall_macro, f1_macro = ModelEvaluator.precision_recall_f1(
            y_test_labels, y_pred, num_classes, 'macro'
        )
        precision_micro, recall_micro, f1_micro = ModelEvaluator.precision_recall_f1(
            y_test_labels, y_pred, num_classes, 'micro'
        )
        precision_weighted, recall_weighted, f1_weighted = ModelEvaluator.precision_recall_f1(
            y_test_labels, y_pred, num_classes, 'weighted'
        )
        
        # Per-class metrics
        precision_per_class, recall_per_class, f1_per_class = ModelEvaluator.precision_recall_f1(
            y_test_labels, y_pred, num_classes, 'none'
        )
        
        return {
            'accuracy': accuracy,
            'confusion_matrix': cm,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'f1_macro': f1_macro,
            'precision_micro': precision_micro,
            'recall_micro': recall_micro,
            'f1_micro': f1_micro,
            'precision_weighted': precision_weighted,
            'recall_weighted': recall_weighted,
            'f1_weighted': f1_weighted,
            'precision_per_class': precision_per_class,
            'recall_per_class': recall_per_class,
            'f1_per_class': f1_per_class,
            'predictions': y_pred,
            'prediction_probabilities': y_pred_proba
        } 