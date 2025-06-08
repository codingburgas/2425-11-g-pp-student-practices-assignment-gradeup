"""
Core Neural Network Models

This module contains the fundamental neural network components:
- Activation functions
- Loss functions
- Custom neural network implementation
"""

import numpy as np
import pickle
import os
from typing import List, Dict, Any, Optional


class ActivationFunctions:
    """Collection of activation functions and their derivatives."""
    
    @staticmethod
    def sigmoid(x: np.ndarray) -> np.ndarray:
        """Sigmoid activation function."""
        
        x_clipped = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x_clipped))
    
    @staticmethod
    def sigmoid_derivative(x: np.ndarray) -> np.ndarray:
        """Derivative of sigmoid function."""
        s = ActivationFunctions.sigmoid(x)
        return s * (1 - s)
    
    @staticmethod
    def relu(x: np.ndarray) -> np.ndarray:
        """ReLU activation function."""
        return np.maximum(0, x)
    
    @staticmethod
    def relu_derivative(x: np.ndarray) -> np.ndarray:
        """Derivative of ReLU function."""
        return (x > 0).astype(float)
    
    @staticmethod
    def tanh(x: np.ndarray) -> np.ndarray:
        """Tanh activation function."""
        return np.tanh(x)
    
    @staticmethod
    def tanh_derivative(x: np.ndarray) -> np.ndarray:
        """Derivative of tanh function."""
        return 1 - np.tanh(x) ** 2
    
    @staticmethod
    def softmax(x: np.ndarray) -> np.ndarray:
        """Softmax activation function for multiclass classification."""
        
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)


class LossFunctions:
    """Collection of loss functions and their derivatives."""
    
    @staticmethod
    def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Mean Squared Error loss function."""
        return np.mean((y_true - y_pred) ** 2)
    
    @staticmethod
    def mse_derivative(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Derivative of MSE loss function."""
        return 2 * (y_pred - y_true) / y_true.shape[0]
    
    @staticmethod
    def binary_crossentropy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Binary crossentropy loss function."""
        
        y_pred_clipped = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return -np.mean(y_true * np.log(y_pred_clipped) + (1 - y_true) * np.log(1 - y_pred_clipped))
    
    @staticmethod
    def categorical_crossentropy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Categorical crossentropy loss function."""
        
        y_pred_clipped = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return -np.mean(np.sum(y_true * np.log(y_pred_clipped), axis=1))


class CustomNeuralNetwork:
    """
    Custom Multi-Layer Perceptron Neural Network implementation from scratch.
    Supports multiple hidden layers, different activation functions, and various optimizers.
    """
    
    def __init__(self, input_size: int, hidden_sizes: List[int], output_size: int, 
                 activation: str = 'relu', output_activation: str = 'sigmoid',
                 learning_rate: float = 0.01, random_seed: int = 42):
        """
        Initialize the neural network.
        
        Args:
            input_size: Number of input features
            hidden_sizes: List of hidden layer sizes
            output_size: Number of output neurons
            activation: Activation function for hidden layers ('relu', 'sigmoid', 'tanh')
            output_activation: Activation function for output layer ('sigmoid', 'softmax')
            learning_rate: Learning rate for optimization
            random_seed: Random seed for reproducibility
        """
        np.random.seed(random_seed)
        
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.activation = activation
        self.output_activation = output_activation
        
        
        self.layers = []
        layer_sizes = [input_size] + hidden_sizes + [output_size]
        
        
        for i in range(len(layer_sizes) - 1):
            weight = np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * np.sqrt(2.0 / layer_sizes[i])
            bias = np.zeros((1, layer_sizes[i + 1]))
            self.layers.append({
                'weight': weight,
                'bias': bias,
                'z': None,  
                'a': None   
            })
        
        
        self.training_history = {
            'loss': [],
            'accuracy': [],
            'val_loss': [],
            'val_accuracy': []
        }
        
        
        self.activation_func = getattr(ActivationFunctions, activation)
        self.activation_derivative = getattr(ActivationFunctions, f"{activation}_derivative")
        self.output_activation_func = getattr(ActivationFunctions, output_activation)
    
    def forward_propagation(self, X: np.ndarray) -> np.ndarray:
        """
        Forward propagation through the network.
        
        Args:
            X: Input data of shape (batch_size, input_size)
            
        Returns:
            Output predictions of shape (batch_size, output_size)
        """
        current_input = X
        
        for i, layer in enumerate(self.layers):
            
            layer['z'] = np.dot(current_input, layer['weight']) + layer['bias']
            
            
            if i == len(self.layers) - 1:  
                layer['a'] = self.output_activation_func(layer['z'])
            else:  
                layer['a'] = self.activation_func(layer['z'])
            
            current_input = layer['a']
        
        return current_input
    
    def backward_propagation(self, X: np.ndarray, y: np.ndarray, y_pred: np.ndarray):
        """
        Backward propagation to compute gradients.
        
        Args:
            X: Input data
            y: True labels
            y_pred: Predicted values
        """
        m = X.shape[0]  
        
        
        if self.output_activation == 'softmax':
            
            error = y_pred - y
        else:
            
            error = LossFunctions.mse_derivative(y, y_pred)
            if self.output_activation == 'sigmoid':
                error *= ActivationFunctions.sigmoid_derivative(self.layers[-1]['z'])
        
        
        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]
            
            
            if i == 0:
                layer_input = X
            else:
                layer_input = self.layers[i - 1]['a']
            
            
            weight_gradient = np.dot(layer_input.T, error) / m
            bias_gradient = np.mean(error, axis=0, keepdims=True)
            
            
            layer['weight'] -= self.learning_rate * weight_gradient
            layer['bias'] -= self.learning_rate * bias_gradient
            
            
            if i > 0:
                error = np.dot(error, layer['weight'].T)
                error *= self.activation_derivative(self.layers[i - 1]['z'])
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None,
              epochs: int = 100, batch_size: int = 32, verbose: bool = True) -> Dict[str, List[float]]:
        """
        Train the neural network.
        
        Args:
            X_train: Training input data
            y_train: Training labels
            X_val: Validation input data (optional)
            y_val: Validation labels (optional)
            epochs: Number of training epochs
            batch_size: Batch size for mini-batch gradient descent
            verbose: Whether to print training progress
            
        Returns:
            Training history dictionary
        """
        n_samples = X_train.shape[0]
        
        for epoch in range(epochs):
            
            indices = np.random.permutation(n_samples)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]
            
            
            total_loss = 0
            for i in range(0, n_samples, batch_size):
                batch_X = X_shuffled[i:i + batch_size]
                batch_y = y_shuffled[i:i + batch_size]
                
                
                y_pred = self.forward_propagation(batch_X)
                self.backward_propagation(batch_X, batch_y, y_pred)
                
                
                if self.output_activation == 'softmax':
                    loss = LossFunctions.categorical_crossentropy(batch_y, y_pred)
                else:
                    loss = LossFunctions.mean_squared_error(batch_y, y_pred)
                total_loss += loss
            
            
            train_pred = self.predict(X_train)
            train_accuracy = self.calculate_accuracy(y_train, train_pred)
            avg_loss = total_loss / (n_samples // batch_size + 1)
            
            self.training_history['loss'].append(avg_loss)
            self.training_history['accuracy'].append(train_accuracy)
            
            
            if X_val is not None and y_val is not None:
                val_pred = self.predict(X_val)
                val_accuracy = self.calculate_accuracy(y_val, val_pred)
                
                if self.output_activation == 'softmax':
                    val_loss = LossFunctions.categorical_crossentropy(y_val, self.forward_propagation(X_val))
                else:
                    val_loss = LossFunctions.mean_squared_error(y_val, self.forward_propagation(X_val))
                
                self.training_history['val_loss'].append(val_loss)
                self.training_history['val_accuracy'].append(val_accuracy)
                
                if verbose and (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f} - "
                          f"Accuracy: {train_accuracy:.4f} - Val Loss: {val_loss:.4f} - "
                          f"Val Accuracy: {val_accuracy:.4f}")
            else:
                if verbose and (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f} - "
                          f"Accuracy: {train_accuracy:.4f}")
        
        return self.training_history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions on input data."""
        return self.forward_propagation(X)
    
    def predict_classes(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        predictions = self.predict(X)
        if self.output_size == 1:
            return (predictions > 0.5).astype(int).flatten()
        else:
            return np.argmax(predictions, axis=1)
    
    def calculate_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate accuracy score."""
        if self.output_size == 1:
            predictions = (y_pred > 0.5).astype(int)
            return np.mean(predictions.flatten() == y_true.flatten())
        else:
            predictions = np.argmax(y_pred, axis=1)
            if y_true.ndim > 1:
                y_true = np.argmax(y_true, axis=1)
            return np.mean(predictions == y_true)
    
    def save_model(self, filepath: str):
        """Save the trained model to a file."""
        model_data = {
            'layers': self.layers,
            'input_size': self.input_size,
            'hidden_sizes': self.hidden_sizes,
            'output_size': self.output_size,
            'activation': self.activation,
            'output_activation': self.output_activation,
            'learning_rate': self.learning_rate,
            'training_history': self.training_history
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    @classmethod
    def load_model(cls, filepath: str) -> 'CustomNeuralNetwork':
        """Load a trained model from a file."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        
        model = cls(
            input_size=model_data['input_size'],
            hidden_sizes=model_data['hidden_sizes'],
            output_size=model_data['output_size'],
            activation=model_data['activation'],
            output_activation=model_data['output_activation'],
            learning_rate=model_data['learning_rate']
        )
        
        
        model.layers = model_data['layers']
        model.training_history = model_data['training_history']
        
        return model 