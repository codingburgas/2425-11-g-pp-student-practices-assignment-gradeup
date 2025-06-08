# Linear regression and logistic regression models implemented from scratch

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import time
from .base import BaseTrainer, BaseMetric, TrainingResult
from .validators import DataValidator, ParameterValidator


class LinearRegression(BaseTrainer):
    """Linear regression model implemented from scratch using gradient descent."""
    
    def __init__(self, 
                 learning_rate: float = 0.01,
                 max_iterations: int = 1000,
                 tolerance: float = 1e-6,
                 fit_intercept: bool = True,
                 regularization: Optional[str] = None,
                 lambda_reg: float = 0.01,
                 random_seed: Optional[int] = 42):
        """
        Initialize linear regression model.
        
        Args:
            learning_rate: Step size for gradient descent
            max_iterations: Maximum number of iterations
            tolerance: Convergence tolerance
            fit_intercept: Whether to fit intercept term
            regularization: Type of regularization ('l1', 'l2', 'elastic', None)
            lambda_reg: Regularization strength
            random_seed: Random seed for reproducibility
        """
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.fit_intercept = fit_intercept
        self.regularization = regularization
        self.lambda_reg = lambda_reg
        self.random_seed = random_seed
        
        # Model parameters
        self.weights = None
        self.bias = None
        self.is_fitted = False
        
        # Training history
        self.training_history = {
            'loss': [],
            'weights_norm': [],
            'gradient_norm': []
        }
        
    def _add_intercept(self, X: np.ndarray) -> np.ndarray:
        """Add intercept term to feature matrix."""
        if self.fit_intercept:
            intercept = np.ones((X.shape[0], 1))
            return np.concatenate([intercept, X], axis=1)
        return X
        
    def _initialize_parameters(self, n_features: int):
        """Initialize model parameters."""
        if self.random_seed is not None:
            np.random.seed(self.random_seed)
            
        # Initialize weights with small random values
        self.weights = np.random.normal(0, 0.01, size=n_features)
        
        if self.fit_intercept:
            self.bias = 0.0
        else:
            self.bias = None
            
    def _compute_predictions(self, X: np.ndarray) -> np.ndarray:
        """Compute model predictions."""
        predictions = np.dot(X, self.weights)
        if self.fit_intercept and self.bias is not None:
            predictions += self.bias
        return predictions
        
    def _compute_loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute mean squared error loss with regularization."""
        predictions = self._compute_predictions(X)
        mse_loss = np.mean((y - predictions) ** 2)
        
        # Add regularization
        reg_term = 0.0
        if self.regularization == 'l1':
            reg_term = self.lambda_reg * np.sum(np.abs(self.weights))
        elif self.regularization == 'l2':
            reg_term = self.lambda_reg * np.sum(self.weights ** 2)
        elif self.regularization == 'elastic':
            l1_term = 0.5 * self.lambda_reg * np.sum(np.abs(self.weights))
            l2_term = 0.5 * self.lambda_reg * np.sum(self.weights ** 2)
            reg_term = l1_term + l2_term
            
        return mse_loss + reg_term
        
    def _compute_gradients(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, Optional[float]]:
        """Compute gradients for weights and bias."""
        n_samples = X.shape[0]
        predictions = self._compute_predictions(X)
        error = predictions - y
        
        # Gradient for weights
        weight_gradient = (2 / n_samples) * np.dot(X.T, error)
        
        # Add regularization to weight gradient
        if self.regularization == 'l1':
            weight_gradient += self.lambda_reg * np.sign(self.weights)
        elif self.regularization == 'l2':
            weight_gradient += 2 * self.lambda_reg * self.weights
        elif self.regularization == 'elastic':
            weight_gradient += self.lambda_reg * (np.sign(self.weights) + self.weights)
            
        # Gradient for bias
        bias_gradient = None
        if self.fit_intercept:
            bias_gradient = (2 / n_samples) * np.sum(error)
            
        return weight_gradient, bias_gradient
        
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> Dict[str, Any]:
        """
        Train the linear regression model.
        
        Args:
            X: Training features
            y: Training targets
            **kwargs: Additional arguments (verbose, etc.)
            
        Returns:
            Training history and metrics
        """
        # Validate inputs
        validator = DataValidator()
        validator.validate_input(X)
        validator.validate_labels(y)
        validator.validate_data_consistency(X, y)
        
        ParameterValidator.validate_learning_rate(self.learning_rate)
        ParameterValidator.validate_epochs(self.max_iterations)
        
        verbose = kwargs.get('verbose', False)
        start_time = time.time()
        
        # Initialize parameters
        self._initialize_parameters(X.shape[1])
        
        # Clear training history
        self.training_history = {'loss': [], 'weights_norm': [], 'gradient_norm': []}
        
        # Training loop
        for iteration in range(self.max_iterations):
            # Compute current loss
            current_loss = self._compute_loss(X, y)
            
            # Compute gradients
            weight_grad, bias_grad = self._compute_gradients(X, y)
            
            # Update parameters
            self.weights -= self.learning_rate * weight_grad
            if self.fit_intercept and bias_grad is not None:
                self.bias -= self.learning_rate * bias_grad
                
            # Record training metrics
            weights_norm = np.linalg.norm(self.weights)
            gradient_norm = np.linalg.norm(weight_grad)
            
            self.training_history['loss'].append(current_loss)
            self.training_history['weights_norm'].append(weights_norm)
            self.training_history['gradient_norm'].append(gradient_norm)
            
            # Check for convergence
            if gradient_norm < self.tolerance:
                if verbose:
                    print(f"Converged at iteration {iteration + 1}")
                break
                
            if verbose and (iteration + 1) % 100 == 0:
                print(f"Iteration {iteration + 1}: Loss = {current_loss:.6f}")
                
        self.is_fitted = True
        training_time = time.time() - start_time
        
        return {
            'training_time': training_time,
            'iterations': len(self.training_history['loss']),
            'final_loss': self.training_history['loss'][-1],
            'converged': gradient_norm < self.tolerance
        }
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions on new data."""
        if not self.is_fitted:
            raise ValueError("Model has not been fitted yet")
            
        validator = DataValidator()
        validator.validate_input(X)
        
        return self._compute_predictions(X)
        
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Evaluate model performance."""
        if not self.is_fitted:
            raise ValueError("Model has not been fitted yet")
            
        predictions = self.predict(X)
        
        # Compute metrics
        mse = np.mean((y - predictions) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y - predictions))
        
        # R-squared
        ss_res = np.sum((y - predictions) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        }


class LogisticRegression(BaseTrainer):
    """Logistic regression model implemented from scratch using gradient descent."""
    
    def __init__(self,
                 learning_rate: float = 0.01,
                 max_iterations: int = 1000,
                 tolerance: float = 1e-6,
                 fit_intercept: bool = True,
                 regularization: Optional[str] = None,
                 lambda_reg: float = 0.01,
                 random_seed: Optional[int] = 42):
        """
        Initialize logistic regression model.
        
        Args:
            learning_rate: Step size for gradient descent
            max_iterations: Maximum number of iterations
            tolerance: Convergence tolerance
            fit_intercept: Whether to fit intercept term
            regularization: Type of regularization ('l1', 'l2', None)
            lambda_reg: Regularization strength
            random_seed: Random seed for reproducibility
        """
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.fit_intercept = fit_intercept
        self.regularization = regularization
        self.lambda_reg = lambda_reg
        self.random_seed = random_seed
        
        # Model parameters
        self.weights = None
        self.bias = None
        self.is_fitted = False
        
        # Training history
        self.training_history = {
            'loss': [],
            'accuracy': [],
            'weights_norm': [],
            'gradient_norm': []
        }
        
    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        """Sigmoid activation function with numerical stability."""
        # Clip z to prevent overflow
        z_clipped = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z_clipped))
        
    def _initialize_parameters(self, n_features: int):
        """Initialize model parameters."""
        if self.random_seed is not None:
            np.random.seed(self.random_seed)
            
        # Initialize weights with small random values
        self.weights = np.random.normal(0, 0.01, size=n_features)
        
        if self.fit_intercept:
            self.bias = 0.0
        else:
            self.bias = None
            
    def _compute_predictions(self, X: np.ndarray) -> np.ndarray:
        """Compute model predictions (probabilities)."""
        z = np.dot(X, self.weights)
        if self.fit_intercept and self.bias is not None:
            z += self.bias
        return self._sigmoid(z)
        
    def _compute_loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute logistic loss with regularization."""
        predictions = self._compute_predictions(X)
        
        # Prevent log(0) by clipping predictions
        predictions_clipped = np.clip(predictions, 1e-15, 1 - 1e-15)
        
        # Binary cross-entropy loss
        log_loss = -np.mean(y * np.log(predictions_clipped) + (1 - y) * np.log(1 - predictions_clipped))
        
        # Add regularization
        reg_term = 0.0
        if self.regularization == 'l1':
            reg_term = self.lambda_reg * np.sum(np.abs(self.weights))
        elif self.regularization == 'l2':
            reg_term = self.lambda_reg * np.sum(self.weights ** 2)
            
        return log_loss + reg_term
        
    def _compute_gradients(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, Optional[float]]:
        """Compute gradients for weights and bias."""
        n_samples = X.shape[0]
        predictions = self._compute_predictions(X)
        error = predictions - y
        
        # Gradient for weights
        weight_gradient = (1 / n_samples) * np.dot(X.T, error)
        
        # Add regularization to weight gradient
        if self.regularization == 'l1':
            weight_gradient += self.lambda_reg * np.sign(self.weights)
        elif self.regularization == 'l2':
            weight_gradient += 2 * self.lambda_reg * self.weights
            
        # Gradient for bias
        bias_gradient = None
        if self.fit_intercept:
            bias_gradient = (1 / n_samples) * np.sum(error)
            
        return weight_gradient, bias_gradient
        
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> Dict[str, Any]:
        """
        Train the logistic regression model.
        
        Args:
            X: Training features
            y: Training targets (binary: 0 or 1)
            **kwargs: Additional arguments
            
        Returns:
            Training history and metrics
        """
        # Validate inputs
        validator = DataValidator()
        validator.validate_input(X)
        validator.validate_labels(y)
        validator.validate_data_consistency(X, y)
        
        # Check for binary classification
        unique_labels = np.unique(y)
        if len(unique_labels) != 2 or not all(label in [0, 1] for label in unique_labels):
            raise ValueError("Logistic regression requires binary labels (0 and 1)")
            
        ParameterValidator.validate_learning_rate(self.learning_rate)
        ParameterValidator.validate_epochs(self.max_iterations)
        
        verbose = kwargs.get('verbose', False)
        start_time = time.time()
        
        # Initialize parameters
        self._initialize_parameters(X.shape[1])
        
        # Clear training history
        self.training_history = {'loss': [], 'accuracy': [], 'weights_norm': [], 'gradient_norm': []}
        
        # Training loop
        for iteration in range(self.max_iterations):
            # Compute current loss
            current_loss = self._compute_loss(X, y)
            
            # Compute accuracy
            predictions = self._compute_predictions(X)
            predicted_classes = (predictions >= 0.5).astype(int)
            accuracy = np.mean(predicted_classes == y)
            
            # Compute gradients
            weight_grad, bias_grad = self._compute_gradients(X, y)
            
            # Update parameters
            self.weights -= self.learning_rate * weight_grad
            if self.fit_intercept and bias_grad is not None:
                self.bias -= self.learning_rate * bias_grad
                
            # Record training metrics
            weights_norm = np.linalg.norm(self.weights)
            gradient_norm = np.linalg.norm(weight_grad)
            
            self.training_history['loss'].append(current_loss)
            self.training_history['accuracy'].append(accuracy)
            self.training_history['weights_norm'].append(weights_norm)
            self.training_history['gradient_norm'].append(gradient_norm)
            
            # Check for convergence
            if gradient_norm < self.tolerance:
                if verbose:
                    print(f"Converged at iteration {iteration + 1}")
                break
                
            if verbose and (iteration + 1) % 100 == 0:
                print(f"Iteration {iteration + 1}: Loss = {current_loss:.6f}, Accuracy = {accuracy:.4f}")
                
        self.is_fitted = True
        training_time = time.time() - start_time
        
        return {
            'training_time': training_time,
            'iterations': len(self.training_history['loss']),
            'final_loss': self.training_history['loss'][-1],
            'final_accuracy': self.training_history['accuracy'][-1],
            'converged': gradient_norm < self.tolerance
        }
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make probability predictions on new data."""
        if not self.is_fitted:
            raise ValueError("Model has not been fitted yet")
            
        validator = DataValidator()
        validator.validate_input(X)
        
        return self._compute_predictions(X)
        
    def predict_classes(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Make class predictions on new data."""
        probabilities = self.predict(X)
        return (probabilities >= threshold).astype(int)
        
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Evaluate model performance."""
        if not self.is_fitted:
            raise ValueError("Model has not been fitted yet")
            
        probabilities = self.predict(X)
        predicted_classes = (probabilities >= 0.5).astype(int)
        
        # Compute metrics
        accuracy = np.mean(predicted_classes == y)
        
        # Precision, Recall, F1 for positive class (class 1)
        true_positives = np.sum((predicted_classes == 1) & (y == 1))
        false_positives = np.sum((predicted_classes == 1) & (y == 0))
        false_negatives = np.sum((predicted_classes == 0) & (y == 1))
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Log loss
        probabilities_clipped = np.clip(probabilities, 1e-15, 1 - 1e-15)
        log_loss = -np.mean(y * np.log(probabilities_clipped) + (1 - y) * np.log(1 - probabilities_clipped))
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'log_loss': log_loss
        } 