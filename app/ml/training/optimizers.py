# Custom gradient descent optimizers implemented from scratch

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from abc import ABC, abstractmethod


class BaseOptimizer(ABC):
    """Abstract base class for optimizers."""
    
    def __init__(self, learning_rate: float = 0.01):
        """
        Initialize optimizer.
        
        Args:
            learning_rate: Learning rate for parameter updates
        """
        self.learning_rate = learning_rate
        self.step_count = 0
        
    @abstractmethod
    def update(self, parameters: np.ndarray, gradients: np.ndarray) -> np.ndarray:
        """
        Update parameters using gradients.
        
        Args:
            parameters: Current parameter values
            gradients: Computed gradients
            
        Returns:
            Updated parameters
        """
        pass
        
    def reset(self):
        """Reset optimizer state."""
        self.step_count = 0


class SGDOptimizer(BaseOptimizer):
    """Stochastic Gradient Descent optimizer."""
    
    def __init__(self, learning_rate: float = 0.01):
        """
        Initialize SGD optimizer.
        
        Args:
            learning_rate: Learning rate for parameter updates
        """
        super().__init__(learning_rate)
        
    def update(self, parameters: np.ndarray, gradients: np.ndarray) -> np.ndarray:
        """Update parameters using basic gradient descent."""
        self.step_count += 1
        return parameters - self.learning_rate * gradients


class MomentumOptimizer(BaseOptimizer):
    """SGD with momentum optimizer."""
    
    def __init__(self, learning_rate: float = 0.01, momentum: float = 0.9):
        """
        Initialize momentum optimizer.
        
        Args:
            learning_rate: Learning rate for parameter updates
            momentum: Momentum coefficient (typically 0.9)
        """
        super().__init__(learning_rate)
        self.momentum = momentum
        self.velocity = None
        
    def update(self, parameters: np.ndarray, gradients: np.ndarray) -> np.ndarray:
        """Update parameters using momentum."""
        self.step_count += 1
        
        # Initialize velocity on first step
        if self.velocity is None:
            self.velocity = np.zeros_like(parameters)
            
        # Update velocity and parameters
        self.velocity = self.momentum * self.velocity + self.learning_rate * gradients
        return parameters - self.velocity
        
    def reset(self):
        """Reset optimizer state."""
        super().reset()
        self.velocity = None


class AdaGradOptimizer(BaseOptimizer):
    """AdaGrad optimizer with adaptive learning rates."""
    
    def __init__(self, learning_rate: float = 0.01, epsilon: float = 1e-8):
        """
        Initialize AdaGrad optimizer.
        
        Args:
            learning_rate: Initial learning rate
            epsilon: Small constant to prevent division by zero
        """
        super().__init__(learning_rate)
        self.epsilon = epsilon
        self.sum_squared_gradients = None
        
    def update(self, parameters: np.ndarray, gradients: np.ndarray) -> np.ndarray:
        """Update parameters using AdaGrad."""
        self.step_count += 1
        
        # Initialize accumulated gradients on first step
        if self.sum_squared_gradients is None:
            self.sum_squared_gradients = np.zeros_like(parameters)
            
        # Accumulate squared gradients
        self.sum_squared_gradients += gradients ** 2
        
        # Compute adaptive learning rate
        adaptive_lr = self.learning_rate / (np.sqrt(self.sum_squared_gradients) + self.epsilon)
        
        return parameters - adaptive_lr * gradients
        
    def reset(self):
        """Reset optimizer state."""
        super().reset()
        self.sum_squared_gradients = None


class RMSPropOptimizer(BaseOptimizer):
    """RMSProp optimizer with exponential moving average of squared gradients."""
    
    def __init__(self, learning_rate: float = 0.01, decay_rate: float = 0.9, epsilon: float = 1e-8):
        """
        Initialize RMSProp optimizer.
        
        Args:
            learning_rate: Learning rate for parameter updates
            decay_rate: Decay rate for moving average (typically 0.9)
            epsilon: Small constant to prevent division by zero
        """
        super().__init__(learning_rate)
        self.decay_rate = decay_rate
        self.epsilon = epsilon
        self.moving_avg_squared_gradients = None
        
    def update(self, parameters: np.ndarray, gradients: np.ndarray) -> np.ndarray:
        """Update parameters using RMSProp."""
        self.step_count += 1
        
        # Initialize moving average on first step
        if self.moving_avg_squared_gradients is None:
            self.moving_avg_squared_gradients = np.zeros_like(parameters)
            
        # Update moving average of squared gradients
        self.moving_avg_squared_gradients = (
            self.decay_rate * self.moving_avg_squared_gradients + 
            (1 - self.decay_rate) * gradients ** 2
        )
        
        # Compute adaptive learning rate
        adaptive_lr = self.learning_rate / (np.sqrt(self.moving_avg_squared_gradients) + self.epsilon)
        
        return parameters - adaptive_lr * gradients
        
    def reset(self):
        """Reset optimizer state."""
        super().reset()
        self.moving_avg_squared_gradients = None


class AdamOptimizer(BaseOptimizer):
    """Adam optimizer combining momentum and adaptive learning rates."""
    
    def __init__(self, 
                 learning_rate: float = 0.001,
                 beta1: float = 0.9,
                 beta2: float = 0.999,
                 epsilon: float = 1e-8):
        """
        Initialize Adam optimizer.
        
        Args:
            learning_rate: Learning rate for parameter updates
            beta1: Exponential decay rate for first moment estimates (typically 0.9)
            beta2: Exponential decay rate for second moment estimates (typically 0.999)
            epsilon: Small constant to prevent division by zero
        """
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None  # First moment estimate
        self.v = None  # Second moment estimate
        
    def update(self, parameters: np.ndarray, gradients: np.ndarray) -> np.ndarray:
        """Update parameters using Adam."""
        self.step_count += 1
        
        # Initialize moment estimates on first step
        if self.m is None:
            self.m = np.zeros_like(parameters)
            self.v = np.zeros_like(parameters)
            
        # Update biased first moment estimate
        self.m = self.beta1 * self.m + (1 - self.beta1) * gradients
        
        # Update biased second moment estimate
        self.v = self.beta2 * self.v + (1 - self.beta2) * gradients ** 2
        
        # Compute bias-corrected first moment estimate
        m_corrected = self.m / (1 - self.beta1 ** self.step_count)
        
        # Compute bias-corrected second moment estimate
        v_corrected = self.v / (1 - self.beta2 ** self.step_count)
        
        # Update parameters
        return parameters - self.learning_rate * m_corrected / (np.sqrt(v_corrected) + self.epsilon)
        
    def reset(self):
        """Reset optimizer state."""
        super().reset()
        self.m = None
        self.v = None


class LearningRateScheduler:
    """Learning rate scheduling strategies."""
    
    @staticmethod
    def step_decay(initial_lr: float, step: int, drop_rate: float = 0.5, epochs_drop: int = 100) -> float:
        """
        Step decay learning rate schedule.
        
        Args:
            initial_lr: Initial learning rate
            step: Current step/epoch
            drop_rate: Factor to multiply learning rate by
            epochs_drop: Number of epochs between drops
            
        Returns:
            Updated learning rate
        """
        return initial_lr * (drop_rate ** (step // epochs_drop))
        
    @staticmethod
    def exponential_decay(initial_lr: float, step: int, decay_rate: float = 0.95) -> float:
        """
        Exponential decay learning rate schedule.
        
        Args:
            initial_lr: Initial learning rate
            step: Current step/epoch
            decay_rate: Decay rate per step
            
        Returns:
            Updated learning rate
        """
        return initial_lr * (decay_rate ** step)
        
    @staticmethod
    def cosine_annealing(initial_lr: float, step: int, max_steps: int, min_lr: float = 0.0) -> float:
        """
        Cosine annealing learning rate schedule.
        
        Args:
            initial_lr: Initial learning rate
            step: Current step/epoch
            max_steps: Total number of steps
            min_lr: Minimum learning rate
            
        Returns:
            Updated learning rate
        """
        return min_lr + (initial_lr - min_lr) * (1 + np.cos(np.pi * step / max_steps)) / 2
        
    @staticmethod
    def polynomial_decay(initial_lr: float, step: int, max_steps: int, power: float = 1.0, min_lr: float = 0.0) -> float:
        """
        Polynomial decay learning rate schedule.
        
        Args:
            initial_lr: Initial learning rate
            step: Current step/epoch
            max_steps: Total number of steps
            power: Power of the polynomial
            min_lr: Minimum learning rate
            
        Returns:
            Updated learning rate
        """
        if step >= max_steps:
            return min_lr
        decay_factor = (1 - step / max_steps) ** power
        return (initial_lr - min_lr) * decay_factor + min_lr


class OptimizerFactory:
    """Factory for creating optimizers."""
    
    @staticmethod
    def create_optimizer(optimizer_name: str, **kwargs) -> BaseOptimizer:
        """
        Create optimizer by name.
        
        Args:
            optimizer_name: Name of optimizer ('sgd', 'momentum', 'adagrad', 'rmsprop', 'adam')
            **kwargs: Optimizer-specific parameters
            
        Returns:
            Initialized optimizer
        """
        optimizers = {
            'sgd': SGDOptimizer,
            'momentum': MomentumOptimizer,
            'adagrad': AdaGradOptimizer,
            'rmsprop': RMSPropOptimizer,
            'adam': AdamOptimizer
        }
        
        if optimizer_name.lower() not in optimizers:
            raise ValueError(f"Unknown optimizer: {optimizer_name}. Available: {list(optimizers.keys())}")
            
        return optimizers[optimizer_name.lower()](**kwargs)
        
    @staticmethod
    def get_optimizer_info() -> Dict[str, Dict[str, Any]]:
        """Get information about available optimizers."""
        return {
            'sgd': {
                'name': 'Stochastic Gradient Descent',
                'parameters': {'learning_rate': 'float'},
                'description': 'Basic gradient descent optimizer'
            },
            'momentum': {
                'name': 'SGD with Momentum',
                'parameters': {'learning_rate': 'float', 'momentum': 'float'},
                'description': 'SGD with momentum to accelerate convergence'
            },
            'adagrad': {
                'name': 'AdaGrad',
                'parameters': {'learning_rate': 'float', 'epsilon': 'float'},
                'description': 'Adaptive learning rate based on gradient history'
            },
            'rmsprop': {
                'name': 'RMSProp',
                'parameters': {'learning_rate': 'float', 'decay_rate': 'float', 'epsilon': 'float'},
                'description': 'RMSProp with exponential moving average'
            },
            'adam': {
                'name': 'Adam',
                'parameters': {'learning_rate': 'float', 'beta1': 'float', 'beta2': 'float', 'epsilon': 'float'},
                'description': 'Adam optimizer combining momentum and adaptive learning rates'
            }
        }


class GradientClipping:
    """Gradient clipping utilities."""
    
    @staticmethod
    def clip_by_norm(gradients: np.ndarray, max_norm: float) -> np.ndarray:
        """
        Clip gradients by global norm.
        
        Args:
            gradients: Gradients to clip
            max_norm: Maximum allowed norm
            
        Returns:
            Clipped gradients
        """
        grad_norm = np.linalg.norm(gradients)
        if grad_norm > max_norm:
            return gradients * (max_norm / grad_norm)
        return gradients
        
    @staticmethod
    def clip_by_value(gradients: np.ndarray, min_value: float = -1.0, max_value: float = 1.0) -> np.ndarray:
        """
        Clip gradients by value.
        
        Args:
            gradients: Gradients to clip
            min_value: Minimum allowed value
            max_value: Maximum allowed value
            
        Returns:
            Clipped gradients
        """
        return np.clip(gradients, min_value, max_value)


# Utility functions for optimization
def compute_gradient_norm(gradients: np.ndarray) -> float:
    """Compute the L2 norm of gradients."""
    return np.linalg.norm(gradients)


def check_convergence(gradients: np.ndarray, tolerance: float = 1e-6) -> bool:
    """Check if optimization has converged based on gradient norm."""
    return compute_gradient_norm(gradients) < tolerance


def estimate_optimal_learning_rate(model, X: np.ndarray, y: np.ndarray, 
                                 lr_range: Tuple[float, float] = (1e-6, 1e-1),
                                 num_steps: int = 100) -> float:
    """
    Estimate optimal learning rate using learning rate range test.
    
    Args:
        model: Model to train
        X: Training features
        y: Training targets
        lr_range: Range of learning rates to test
        num_steps: Number of steps to test
        
    Returns:
        Estimated optimal learning rate
    """
    min_lr, max_lr = lr_range
    lrs = np.logspace(np.log10(min_lr), np.log10(max_lr), num_steps)
    losses = []
    
    # Save original model state
    original_weights = model.weights.copy() if model.weights is not None else None
    original_bias = model.bias
    
    for lr in lrs:
        # Reset model
        if original_weights is not None:
            model.weights = original_weights.copy()
        model.bias = original_bias
        model.learning_rate = lr
        
        # Take one training step
        model.fit(X, y, max_iterations=1, verbose=False)
        
        # Compute loss
        if hasattr(model, '_compute_loss'):
            loss = model._compute_loss(X, y)
            losses.append(loss)
        else:
            # Fallback for models without _compute_loss method
            predictions = model.predict(X)
            loss = np.mean((y - predictions) ** 2)
            losses.append(loss)
    
    # Find learning rate with steepest decrease
    losses = np.array(losses)
    gradients = np.gradient(losses)
    optimal_idx = np.argmin(gradients)
    
    return lrs[optimal_idx] 