"""
Base classes and interfaces for the training system.

This module defines the fundamental interfaces and abstract base classes
that all training components should implement.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Tuple, Optional
import numpy as np


class BaseValidator(ABC):
    """Abstract base class for data validation components."""
    
    @abstractmethod
    def validate_input(self, data: np.ndarray) -> bool:
        """Validate input data format and content."""
        pass
    
    @abstractmethod
    def validate_labels(self, labels: np.ndarray) -> bool:
        """Validate label data format and content."""
        pass


class BaseTrainer(ABC):
    """Abstract base class for model training components."""
    
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> Dict[str, Any]:
        """Train the model on provided data."""
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions on new data."""
        pass
    
    @abstractmethod
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Evaluate model performance on test data."""
        pass


class BaseMetric(ABC):
    """Abstract base class for evaluation metrics."""
    
    @abstractmethod
    def calculate(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate the metric value."""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the metric name."""
        pass


class TrainingConfig:
    """Configuration class for training parameters."""
    
    def __init__(self, 
                 learning_rate: float = 0.01,
                 epochs: int = 100,
                 batch_size: int = 32,
                 validation_split: float = 0.2,
                 test_split: float = 0.2,
                 random_seed: int = 42,
                 early_stopping: bool = True,
                 patience: int = 10,
                 verbose: bool = True):
        """Initialize training configuration."""
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.test_split = test_split
        self.random_seed = random_seed
        self.early_stopping = early_stopping
        self.patience = patience
        self.verbose = verbose
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return vars(self)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'TrainingConfig':
        """Create configuration from dictionary."""
        return cls(**config_dict)


class TrainingResult:
    """Container for training results and metrics."""
    
    def __init__(self):
        """Initialize empty training result."""
        self.history: Dict[str, List[float]] = {}
        self.final_metrics: Dict[str, float] = {}
        self.best_epoch: int = 0
        self.training_time: float = 0.0
        self.model_path: Optional[str] = None
        
    def add_epoch_metric(self, metric_name: str, value: float):
        """Add a metric value for the current epoch."""
        if metric_name not in self.history:
            self.history[metric_name] = []
        self.history[metric_name].append(value)
    
    def set_final_metric(self, metric_name: str, value: float):
        """Set a final evaluation metric."""
        self.final_metrics[metric_name] = value
    
    def get_best_metric(self, metric_name: str) -> Tuple[float, int]:
        """Get the best value and epoch for a given metric."""
        if metric_name not in self.history:
            raise ValueError(f"Metric '{metric_name}' not found in history")
        
        values = self.history[metric_name]
        if 'loss' in metric_name.lower() or 'error' in metric_name.lower():
            # For loss/error metrics, lower is better
            best_value = min(values)
            best_epoch = values.index(best_value)
        else:
            # For accuracy/score metrics, higher is better
            best_value = max(values)
            best_epoch = values.index(best_value)
            
        return best_value, best_epoch
    
    def summary(self) -> Dict[str, Any]:
        """Get a summary of training results."""
        summary = {
            'training_time': self.training_time,
            'total_epochs': len(self.history.get('loss', [])),
            'best_epoch': self.best_epoch,
            'final_metrics': self.final_metrics.copy()
        }
        
        # Add best values for each metric
        for metric_name in self.history:
            try:
                best_value, best_epoch = self.get_best_metric(metric_name)
                summary[f'best_{metric_name}'] = best_value
                summary[f'best_{metric_name}_epoch'] = best_epoch
            except ValueError:
                continue
                
        return summary 