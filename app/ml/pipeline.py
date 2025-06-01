"""
ML Training Pipeline Module

This module contains the complete ML training pipeline with data preprocessing,
model training, and evaluation.
"""

import numpy as np
import pickle
import os
from typing import List, Tuple, Dict, Any
from .models import CustomNeuralNetwork
from .evaluator import ModelEvaluator


class MLTrainingPipeline:
    """
    Complete ML training pipeline with data preprocessing, model training, and evaluation.
    """
    
    def __init__(self, random_seed: int = 42):
        """Initialize the training pipeline."""
        self.random_seed = random_seed
        self.model = None
        self.scaler_params = None
        self.label_encoder_params = None
        
    def preprocess_data(self, X: np.ndarray, y: np.ndarray = None, 
                       fit_scalers: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess the input data.
        
        Args:
            X: Input features
            y: Target labels (optional)
            fit_scalers: Whether to fit scalers (True for training, False for prediction)
            
        Returns:
            Preprocessed X and y
        """
        
        if fit_scalers:
            self.scaler_params = {
                'mean': np.mean(X, axis=0),
                'std': np.std(X, axis=0)
            }
        
        if self.scaler_params is not None:
            
            std_safe = np.where(self.scaler_params['std'] == 0, 1, self.scaler_params['std'])
            X_scaled = (X - self.scaler_params['mean']) / std_safe
        else:
            X_scaled = X
        
        
        if y is not None:
            if fit_scalers:
                unique_labels = np.unique(y)
                self.label_encoder_params = {
                    'classes': unique_labels,
                    'num_classes': len(unique_labels)
                }
            
            if self.label_encoder_params is not None and len(self.label_encoder_params['classes']) > 2:
                
                y_encoded = np.zeros((len(y), self.label_encoder_params['num_classes']))
                for i, label in enumerate(y):
                    class_idx = np.where(self.label_encoder_params['classes'] == label)[0][0]
                    y_encoded[i, class_idx] = 1
            else:
                
                y_encoded = y.reshape(-1, 1).astype(float)
            
            return X_scaled, y_encoded
        
        return X_scaled, None
    
    def split_data(self, X: np.ndarray, y: np.ndarray, 
                   test_size: float = 0.2, val_size: float = 0.2) -> Tuple[np.ndarray, ...]:
        """
        Split data into train, validation, and test sets.
        
        Args:
            X: Input features
            y: Target labels
            test_size: Proportion of data for testing
            val_size: Proportion of remaining data for validation
            
        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        np.random.seed(self.random_seed)
        n_samples = X.shape[0]
        
        
        indices = np.random.permutation(n_samples)
        
        
        test_split = int(n_samples * (1 - test_size))
        val_split = int(test_split * (1 - val_size))
        
        
        train_indices = indices[:val_split]
        val_indices = indices[val_split:test_split]
        test_indices = indices[test_split:]
        
        
        X_train, X_val, X_test = X[train_indices], X[val_indices], X[test_indices]
        y_train, y_val, y_test = y[train_indices], y[val_indices], y[test_indices]
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def train_model(self, X: np.ndarray, y: np.ndarray, 
                   hidden_sizes: List[int] = [64, 32], 
                   activation: str = 'relu',
                   learning_rate: float = 0.01,
                   epochs: int = 100,
                   batch_size: int = 32,
                   test_size: float = 0.2,
                   val_size: float = 0.2,
                   verbose: bool = True) -> Dict[str, Any]:
        """
        Complete model training pipeline.
        
        Args:
            X: Input features
            y: Target labels
            hidden_sizes: List of hidden layer sizes
            activation: Activation function
            learning_rate: Learning rate
            epochs: Number of training epochs
            batch_size: Batch size
            test_size: Test set size
            val_size: Validation set size
            verbose: Whether to print progress
            
        Returns:
            Training results and evaluation metrics
        """
        
        X_processed, y_processed = self.preprocess_data(X, y, fit_scalers=True)
        
        
        X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(
            X_processed, y_processed, test_size, val_size
        )
        
        
        input_size = X_train.shape[1]
        output_size = y_train.shape[1] if y_train.ndim > 1 else 1
        output_activation = 'softmax' if output_size > 2 else 'sigmoid'
        
        
        self.model = CustomNeuralNetwork(
            input_size=input_size,
            hidden_sizes=hidden_sizes,
            output_size=output_size,
            activation=activation,
            output_activation=output_activation,
            learning_rate=learning_rate,
            random_seed=self.random_seed
        )
        
        if verbose:
            print(f"Training neural network with architecture: {input_size} -> {' -> '.join(map(str, hidden_sizes))} -> {output_size}")
            print(f"Activation: {activation}, Output activation: {output_activation}")
            print(f"Training samples: {X_train.shape[0]}, Validation samples: {X_val.shape[0]}, Test samples: {X_test.shape[0]}")
        
        
        training_history = self.model.train(
            X_train, y_train, X_val, y_val,
            epochs=epochs, batch_size=batch_size, verbose=verbose
        )
        
        
        test_metrics = ModelEvaluator.evaluate_model(self.model, X_test, y_test)
        
        if verbose:
            print(f"\nTest Results:")
            print(f"Accuracy: {test_metrics['accuracy']:.4f}")
            print(f"Precision (macro): {test_metrics['precision_macro']:.4f}")
            print(f"Recall (macro): {test_metrics['recall_macro']:.4f}")
            print(f"F1-score (macro): {test_metrics['f1_macro']:.4f}")
        
        return {
            'model': self.model,
            'training_history': training_history,
            'test_metrics': test_metrics,
            'data_splits': {
                'X_train': X_train, 'X_val': X_val, 'X_test': X_test,
                'y_train': y_train, 'y_val': y_val, 'y_test': y_test
            }
        }
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions on new data."""
        if self.model is None:
            raise ValueError("Model not trained yet. Call train_model first.")
        
        X_processed, _ = self.preprocess_data(X, fit_scalers=False)
        return self.model.predict(X_processed)
    
    def predict_classes(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for new data."""
        if self.model is None:
            raise ValueError("Model not trained yet. Call train_model first.")
        
        X_processed, _ = self.preprocess_data(X, fit_scalers=False)
        return self.model.predict_classes(X_processed)
    
    def save_pipeline(self, filepath: str):
        """Save the entire training pipeline."""
        if self.model is None:
            raise ValueError("Model not trained yet. Call train_model first.")
        
        pipeline_data = {
            'model': self.model,
            'scaler_params': self.scaler_params,
            'label_encoder_params': self.label_encoder_params,
            'random_seed': self.random_seed
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(pipeline_data, f)
    
    @classmethod
    def load_pipeline(cls, filepath: str) -> 'MLTrainingPipeline':
        """Load a trained pipeline."""
        with open(filepath, 'rb') as f:
            pipeline_data = pickle.load(f)
        
        pipeline = cls(random_seed=pipeline_data['random_seed'])
        pipeline.model = pipeline_data['model']
        pipeline.scaler_params = pipeline_data['scaler_params']
        pipeline.label_encoder_params = pipeline_data['label_encoder_params']
        
        return pipeline 