# Unified model training interface for the training system

import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union, Type
import time
import json
from pathlib import Path

from .base import BaseTrainer, TrainingConfig, TrainingResult
from .validators import DataValidator, ParameterValidator
from .preprocessing import DataPreprocessor
from .data_splitter import DataSplitter
from .cross_validator import CrossValidator, CrossValidationResult
from .linear_models import LinearRegression, LogisticRegression
from .optimizers import OptimizerFactory, LearningRateScheduler


class ModelTrainer:
    """Unified interface for training machine learning models."""
    
    def __init__(self, 
                 model_class: Type[BaseTrainer],
                 model_params: Optional[Dict[str, Any]] = None,
                 preprocessing_config: Optional[Dict[str, Any]] = None,
                 training_config: Optional[TrainingConfig] = None,
                 random_seed: Optional[int] = 42):
        """
        Initialize model trainer.
        
        Args:
            model_class: Class of model to train
            model_params: Parameters for model initialization
            preprocessing_config: Configuration for data preprocessing
            training_config: Training configuration parameters
            random_seed: Random seed for reproducibility
        """
        self.model_class = model_class
        self.model_params = model_params or {}
        self.preprocessing_config = preprocessing_config or {}
        self.training_config = training_config or TrainingConfig()
        self.random_seed = random_seed
        
        # Initialize components
        self.preprocessor = None
        self.model = None
        self.data_splitter = DataSplitter(random_seed=random_seed)
        self.cross_validator = CrossValidator(random_seed=random_seed)
        
        # Training state
        self.is_fitted = False
        self.training_history = {}
        self.evaluation_results = {}
        
    def _initialize_preprocessor(self):
        """Initialize data preprocessor with configuration."""
        if not self.preprocessing_config:
            self.preprocessor = None
            return
            
        self.preprocessor = DataPreprocessor(**self.preprocessing_config)
        
    def _validate_training_inputs(self, X: np.ndarray, y: np.ndarray):
        """Validate training inputs."""
        validator = DataValidator()
        validator.validate_input(X)
        validator.validate_labels(y)
        validator.validate_data_consistency(X, y)
        
        # Validate training configuration
        ParameterValidator.validate_learning_rate(self.training_config.learning_rate)
        ParameterValidator.validate_epochs(self.training_config.epochs)
        ParameterValidator.validate_batch_size(self.training_config.batch_size, X.shape[0])
        
    def _prepare_data(self, X: np.ndarray, y: np.ndarray, 
                     is_training: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data using preprocessing pipeline."""
        if self.preprocessor is None:
            return X, y
            
        if is_training:
            if len(y.shape) == 1:
                # For 1D labels, try to fit and transform both X and y
                try:
                    result = self.preprocessor.fit_transform(X, y)
                    if isinstance(result, tuple):
                        X_processed, y_processed = result
                    else:
                        X_processed, y_processed = result, y
                except:
                    # Fallback: just process X
                    X_processed = self.preprocessor.fit_transform(X)
                    y_processed = y
            else:
                # For multi-dimensional labels, just process X
                X_processed = self.preprocessor.fit_transform(X)
                y_processed = y
        else:
            # For prediction/evaluation, only transform X
            X_processed = self.preprocessor.transform(X)
            y_processed = y
            
        return X_processed, y_processed
        
    def train(self, 
              X: np.ndarray, 
              y: np.ndarray,
              validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
              verbose: bool = True) -> TrainingResult:
        """
        Train the model with comprehensive training pipeline.
        
        Args:
            X: Training features
            y: Training targets
            validation_data: Optional validation data (X_val, y_val)
            verbose: Whether to print training progress
            
        Returns:
            TrainingResult object with training history and metrics
        """
        if verbose:
            print("Starting model training...")
            print(f"Model: {self.model_class.__name__}")
            print(f"Training samples: {X.shape[0]}")
            print(f"Features: {X.shape[1]}")
            
        start_time = time.time()
        
        # Validate inputs
        self._validate_training_inputs(X, y)
        
        # Initialize preprocessor
        self._initialize_preprocessor()
        
        # Prepare training data
        X_processed, y_processed = self._prepare_data(X, y, is_training=True)
        
        # Handle validation data
        X_val_processed, y_val_processed = None, None
        if validation_data is not None:
            X_val, y_val = validation_data
            X_val_processed, y_val_processed = self._prepare_data(X_val, y_val, is_training=False)
        elif self.training_config.validation_split > 0:
            # Create validation split
            X_train, X_val, y_train, y_val = self.data_splitter.train_test_split(
                X_processed, y_processed, 
                test_size=self.training_config.validation_split,
                shuffle=True
            )
            X_processed, y_processed = X_train, y_train
            X_val_processed, y_val_processed = X_val, y_val
            
        # Initialize model
        model_params = self.model_params.copy()
        model_params.update({
            'learning_rate': self.training_config.learning_rate,
            'max_iterations': self.training_config.epochs,
            'random_seed': self.random_seed
        })
        
        self.model = self.model_class(**model_params)
        
        if verbose:
            print(f"Training with {X_processed.shape[0]} samples...")
            if X_val_processed is not None:
                print(f"Validation with {X_val_processed.shape[0]} samples...")
        
        # Training loop with early stopping
        best_val_loss = float('inf')
        patience_counter = 0
        training_result = TrainingResult()
        
        for epoch in range(self.training_config.epochs):
            epoch_start = time.time()
            
            # Train for one epoch
            train_metrics = self.model.fit(X_processed, y_processed, verbose=False)
            
            # Evaluate on training data
            train_eval = self.model.evaluate(X_processed, y_processed)
            for metric_name, value in train_eval.items():
                training_result.add_epoch_metric(f'train_{metric_name}', value)
                
            # Evaluate on validation data if available
            if X_val_processed is not None:
                val_eval = self.model.evaluate(X_val_processed, y_val_processed)
                for metric_name, value in val_eval.items():
                    training_result.add_epoch_metric(f'val_{metric_name}', value)
                    
                # Early stopping logic
                val_loss = val_eval.get('mse', val_eval.get('log_loss', float('inf')))
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    training_result.best_epoch = epoch
                else:
                    patience_counter += 1
                    
                if (self.training_config.early_stopping and 
                    patience_counter >= self.training_config.patience):
                    if verbose:
                        print(f"Early stopping at epoch {epoch + 1}")
                    break
                    
            epoch_time = time.time() - epoch_start
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{self.training_config.epochs} - "
                      f"Train loss: {train_eval.get('mse', train_eval.get('log_loss', 0)):.4f}")
                if X_val_processed is not None:
                    print(f"  Val loss: {val_eval.get('mse', val_eval.get('log_loss', 0)):.4f}")
                    
        # Finalize training result
        total_time = time.time() - start_time
        training_result.training_time = total_time
        
        # Set final metrics
        final_train_metrics = self.model.evaluate(X_processed, y_processed)
        for metric_name, value in final_train_metrics.items():
            training_result.set_final_metric(f'final_train_{metric_name}', value)
            
        if X_val_processed is not None:
            final_val_metrics = self.model.evaluate(X_val_processed, y_val_processed)
            for metric_name, value in final_val_metrics.items():
                training_result.set_final_metric(f'final_val_{metric_name}', value)
        
        self.is_fitted = True
        self.training_history = training_result.history
        
        if verbose:
            print(f"Training completed in {total_time:.2f} seconds")
            print(f"Final training metrics: {final_train_metrics}")
            if X_val_processed is not None:
                print(f"Final validation metrics: {final_val_metrics}")
                
        return training_result
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions on new data."""
        if not self.is_fitted:
            raise ValueError("Model has not been trained yet")
            
        X_processed, _ = self._prepare_data(X, np.zeros(X.shape[0]), is_training=False)
        return self.model.predict(X_processed)
        
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Evaluate model on test data."""
        if not self.is_fitted:
            raise ValueError("Model has not been trained yet")
            
        X_processed, y_processed = self._prepare_data(X, y, is_training=False)
        return self.model.evaluate(X_processed, y_processed)
        
    def cross_validate(self, 
                      X: np.ndarray, 
                      y: np.ndarray,
                      cv_strategy: str = 'k_fold',
                      k: int = 5,
                      stratified: bool = False,
                      verbose: bool = True) -> CrossValidationResult:
        """
        Perform cross-validation on the model.
        
        Args:
            X: Feature data
            y: Target data
            cv_strategy: CV strategy ('k_fold', 'time_series')
            k: Number of folds
            stratified: Whether to use stratified CV
            verbose: Whether to print progress
            
        Returns:
            CrossValidationResult object
        """
        # Prepare data
        self._initialize_preprocessor()
        if self.preprocessor is not None:
            X_processed = self.preprocessor.fit_transform(X)
            y_processed = y  # Don't transform labels for CV
        else:
            X_processed, y_processed = X, y
            
        # Prepare model configuration
        model_params = self.model_params.copy()
        model_params.update({
            'learning_rate': self.training_config.learning_rate,
            'max_iterations': self.training_config.epochs,
            'random_seed': self.random_seed
        })
        
        if cv_strategy == 'k_fold':
            return self.cross_validator.k_fold_cv(
                self.model_class, model_params, X_processed, y_processed,
                k=k, stratified=stratified, verbose=verbose
            )
        elif cv_strategy == 'time_series':
            return self.cross_validator.time_series_cv(
                self.model_class, model_params, X_processed, y_processed,
                n_splits=k, verbose=verbose
            )
        else:
            raise ValueError(f"Unknown CV strategy: {cv_strategy}")
            
    def get_feature_importance(self) -> Optional[np.ndarray]:
        """Get feature importance if available."""
        if not self.is_fitted:
            raise ValueError("Model has not been trained yet")
            
        if hasattr(self.model, 'weights'):
            return np.abs(self.model.weights)
        return None
        
    def save_model(self, filepath: str):
        """Save trained model to file."""
        if not self.is_fitted:
            raise ValueError("Model has not been trained yet")
            
        model_data = {
            'model_class': self.model_class.__name__,
            'model_params': self.model_params,
            'training_config': self.training_config.to_dict(),
            'preprocessing_config': self.preprocessing_config,
            'weights': self.model.weights.tolist() if self.model.weights is not None else None,
            'bias': float(self.model.bias) if self.model.bias is not None else None,
            'training_history': self.training_history,
            'is_fitted': self.is_fitted
        }
        
        # Save preprocessor state if available
        if self.preprocessor is not None:
            model_data['preprocessor_info'] = self.preprocessor.get_preprocessing_info()
            
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(model_data, f, indent=2)
            
    @classmethod
    def load_model(cls, filepath: str) -> 'ModelTrainer':
        """Load trained model from file."""
        with open(filepath, 'r') as f:
            model_data = json.load(f)
            
        # Reconstruct model class
        model_class_name = model_data['model_class']
        if model_class_name == 'LinearRegression':
            model_class = LinearRegression
        elif model_class_name == 'LogisticRegression':
            model_class = LogisticRegression
        else:
            raise ValueError(f"Unknown model class: {model_class_name}")
            
        # Create trainer
        training_config = TrainingConfig.from_dict(model_data['training_config'])
        trainer = cls(
            model_class=model_class,
            model_params=model_data['model_params'],
            preprocessing_config=model_data['preprocessing_config'],
            training_config=training_config
        )
        
        # Restore model state
        trainer.model = model_class(**model_data['model_params'])
        if model_data['weights'] is not None:
            trainer.model.weights = np.array(model_data['weights'])
        if model_data['bias'] is not None:
            trainer.model.bias = model_data['bias']
        trainer.model.is_fitted = model_data['is_fitted']
        
        trainer.is_fitted = model_data['is_fitted']
        trainer.training_history = model_data['training_history']
        
        return trainer
        
    def get_training_summary(self) -> Dict[str, Any]:
        """Get summary of training configuration and results."""
        summary = {
            'model_class': self.model_class.__name__,
            'model_params': self.model_params,
            'training_config': self.training_config.to_dict(),
            'preprocessing_config': self.preprocessing_config,
            'is_fitted': self.is_fitted
        }
        
        if self.is_fitted:
            summary['training_history'] = self.training_history
            summary['evaluation_results'] = self.evaluation_results
            
            # Add model-specific info
            if hasattr(self.model, 'weights') and self.model.weights is not None:
                summary['num_parameters'] = len(self.model.weights)
                summary['weights_norm'] = float(np.linalg.norm(self.model.weights))
                
        return summary 