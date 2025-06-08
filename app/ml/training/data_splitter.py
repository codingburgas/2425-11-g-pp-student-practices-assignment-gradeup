# Data splitting functionality for train/test/validation splits

import numpy as np
from typing import Tuple, List, Dict, Any, Optional, Union
from .validators import DataValidator, ParameterValidator


class DataSplitter:
    """Custom implementation of data splitting for machine learning."""
    
    def __init__(self, random_seed: Optional[int] = 42):
        """
        Initialize data splitter.
        
        Args:
            random_seed: Random seed for reproducible splits
        """
        self.random_seed = random_seed
        if random_seed is not None:
            np.random.seed(random_seed)
            
    def train_test_split(self, 
                        X: np.ndarray, 
                        y: np.ndarray,
                        test_size: float = 0.2,
                        shuffle: bool = True,
                        stratify: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split data into training and testing sets.
        
        Args:
            X: Input features
            y: Target labels
            test_size: Fraction of data to use for testing
            shuffle: Whether to shuffle data before splitting
            stratify: Whether to maintain class distribution in splits
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        # Validate inputs
        validator = DataValidator()
        validator.validate_input(X)
        validator.validate_labels(y)
        validator.validate_data_consistency(X, y)
        ParameterValidator.validate_split_ratio(test_size, "test_size")
        
        n_samples = X.shape[0]
        n_test = int(n_samples * test_size)
        n_train = n_samples - n_test
        
        if n_train < 1 or n_test < 1:
            raise ValueError(f"Split results in empty sets: train={n_train}, test={n_test}")
            
        if stratify and len(y.shape) == 1:
            return self._stratified_split(X, y, test_size, shuffle)
        
        # Generate indices
        indices = np.arange(n_samples)
        
        if shuffle:
            np.random.shuffle(indices)
            
        # Split indices
        train_indices = indices[:n_train]
        test_indices = indices[n_train:]
        
        return X[train_indices], X[test_indices], y[train_indices], y[test_indices]
        
    def train_validation_test_split(self,
                                  X: np.ndarray,
                                  y: np.ndarray,
                                  validation_size: float = 0.2,
                                  test_size: float = 0.2,
                                  shuffle: bool = True,
                                  stratify: bool = False) -> Tuple[np.ndarray, ...]:
        """
        Split data into training, validation, and testing sets.
        
        Args:
            X: Input features
            y: Target labels
            validation_size: Fraction of data to use for validation
            test_size: Fraction of data to use for testing
            shuffle: Whether to shuffle data before splitting
            stratify: Whether to maintain class distribution in splits
            
        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        # Validate inputs
        ParameterValidator.validate_split_ratio(validation_size, "validation_size")
        ParameterValidator.validate_split_ratio(test_size, "test_size")
        
        if validation_size + test_size >= 1.0:
            raise ValueError(f"validation_size + test_size must be < 1.0, got {validation_size + test_size}")
            
        # First split: separate out test set
        X_temp, X_test, y_temp, y_test = self.train_test_split(
            X, y, test_size=test_size, shuffle=shuffle, stratify=stratify
        )
        
        # Second split: separate train and validation from remaining data
        val_size_adjusted = validation_size / (1.0 - test_size)
        X_train, X_val, y_train, y_val = self.train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, shuffle=shuffle, stratify=stratify
        )
        
        return X_train, X_val, X_test, y_train, y_val, y_test
        
    def _stratified_split(self, 
                         X: np.ndarray, 
                         y: np.ndarray,
                         test_size: float,
                         shuffle: bool) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Perform stratified split to maintain class distribution.
        
        Args:
            X: Input features
            y: Target labels (1D array)
            test_size: Fraction of data to use for testing
            shuffle: Whether to shuffle data before splitting
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        unique_classes, class_counts = np.unique(y, return_counts=True)
        
        train_indices = []
        test_indices = []
        
        for class_label in unique_classes:
            # Get indices for this class
            class_indices = np.where(y == class_label)[0]
            
            if shuffle:
                np.random.shuffle(class_indices)
                
            # Calculate split for this class
            n_class_samples = len(class_indices)
            n_test_class = max(1, int(n_class_samples * test_size))
            n_train_class = n_class_samples - n_test_class
            
            # Split indices for this class
            train_indices.extend(class_indices[:n_train_class])
            test_indices.extend(class_indices[n_train_class:])
            
        # Convert to arrays and shuffle if requested
        train_indices = np.array(train_indices)
        test_indices = np.array(test_indices)
        
        if shuffle:
            np.random.shuffle(train_indices)
            np.random.shuffle(test_indices)
            
        return X[train_indices], X[test_indices], y[train_indices], y[test_indices]
        
    def time_series_split(self,
                         X: np.ndarray,
                         y: np.ndarray,
                         test_size: float = 0.2) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split time series data maintaining temporal order.
        
        Args:
            X: Input features (ordered by time)
            y: Target labels (ordered by time)
            test_size: Fraction of data to use for testing
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        validator = DataValidator()
        validator.validate_input(X)
        validator.validate_labels(y)
        validator.validate_data_consistency(X, y)
        
        n_samples = X.shape[0]
        split_index = int(n_samples * (1 - test_size))
        
        if split_index < 1 or split_index >= n_samples:
            raise ValueError(f"Invalid split index: {split_index}")
            
        X_train = X[:split_index]
        X_test = X[split_index:]
        y_train = y[:split_index]
        y_test = y[split_index:]
        
        return X_train, X_test, y_train, y_test
        
    def bootstrap_sample(self,
                        X: np.ndarray,
                        y: np.ndarray,
                        sample_size: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Create bootstrap sample from the data.
        
        Args:
            X: Input features
            y: Target labels
            sample_size: Size of bootstrap sample (default: same as original)
            
        Returns:
            Tuple of (X_bootstrap, y_bootstrap, out_of_bag_indices)
        """
        validator = DataValidator()
        validator.validate_input(X)
        validator.validate_labels(y)
        validator.validate_data_consistency(X, y)
        
        n_samples = X.shape[0]
        if sample_size is None:
            sample_size = n_samples
            
        # Sample with replacement
        bootstrap_indices = np.random.choice(n_samples, size=sample_size, replace=True)
        
        # Find out-of-bag samples
        all_indices = set(range(n_samples))
        bootstrap_set = set(bootstrap_indices)
        oob_indices = np.array(list(all_indices - bootstrap_set))
        
        X_bootstrap = X[bootstrap_indices]
        y_bootstrap = y[bootstrap_indices]
        
        return X_bootstrap, y_bootstrap, oob_indices
        
    def k_fold_indices(self,
                      n_samples: int,
                      k: int = 5,
                      shuffle: bool = True) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate k-fold cross-validation indices.
        
        Args:
            n_samples: Total number of samples
            k: Number of folds
            shuffle: Whether to shuffle indices before splitting
            
        Returns:
            List of (train_indices, val_indices) tuples for each fold
        """
        if k < 2:
            raise ValueError(f"Number of folds must be at least 2, got {k}")
        if k > n_samples:
            raise ValueError(f"Number of folds ({k}) cannot exceed number of samples ({n_samples})")
            
        indices = np.arange(n_samples)
        if shuffle:
            np.random.shuffle(indices)
            
        fold_sizes = np.full(k, n_samples // k, dtype=int)
        fold_sizes[:n_samples % k] += 1
        
        folds = []
        current = 0
        for fold_size in fold_sizes:
            start, stop = current, current + fold_size
            folds.append(indices[start:stop])
            current = stop
            
        # Generate train/validation indices for each fold
        fold_indices = []
        for i in range(k):
            val_indices = folds[i]
            train_indices = np.concatenate([folds[j] for j in range(k) if j != i])
            fold_indices.append((train_indices, val_indices))
            
        return fold_indices
        
    def stratified_k_fold_indices(self,
                                 y: np.ndarray,
                                 k: int = 5,
                                 shuffle: bool = True) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate stratified k-fold cross-validation indices.
        
        Args:
            y: Target labels for stratification
            k: Number of folds
            shuffle: Whether to shuffle indices before splitting
            
        Returns:
            List of (train_indices, val_indices) tuples for each fold
        """
        if len(y.shape) != 1:
            raise ValueError("Stratified k-fold requires 1D labels")
            
        unique_classes, class_counts = np.unique(y, return_counts=True)
        
        # Check if we have enough samples per class
        min_class_count = np.min(class_counts)
        if min_class_count < k:
            raise ValueError(f"Insufficient samples for {k}-fold split. "
                           f"Minimum class has only {min_class_count} samples")
                           
        class_fold_indices = {}
        
        # Generate folds for each class separately
        for class_label in unique_classes:
            class_indices = np.where(y == class_label)[0]
            
            if shuffle:
                np.random.shuffle(class_indices)
                
            # Split class indices into k folds
            class_fold_indices[class_label] = []
            n_class_samples = len(class_indices)
            fold_sizes = np.full(k, n_class_samples // k, dtype=int)
            fold_sizes[:n_class_samples % k] += 1
            
            current = 0
            for fold_size in fold_sizes:
                start, stop = current, current + fold_size
                class_fold_indices[class_label].append(class_indices[start:stop])
                current = stop
                
        # Combine folds across classes
        fold_indices = []
        for i in range(k):
            val_indices = np.concatenate([class_fold_indices[cls][i] for cls in unique_classes])
            train_indices = np.concatenate([
                np.concatenate([class_fold_indices[cls][j] for j in range(k) if j != i])
                for cls in unique_classes
            ])
            
            if shuffle:
                np.random.shuffle(val_indices)
                np.random.shuffle(train_indices)
                
            fold_indices.append((train_indices, val_indices))
            
        return fold_indices
        
    def get_split_info(self, X: np.ndarray, y: np.ndarray, split_result: Tuple) -> Dict[str, Any]:
        """
        Get information about the data split.
        
        Args:
            X: Original input features
            y: Original target labels
            split_result: Result from any split method
            
        Returns:
            Dictionary containing split information
        """
        original_shape = X.shape
        
        if len(split_result) == 4:  # train_test_split
            X_train, X_test, y_train, y_test = split_result
            splits = ['train', 'test']
            X_splits = [X_train, X_test]
            y_splits = [y_train, y_test]
        elif len(split_result) == 6:  # train_validation_test_split
            X_train, X_val, X_test, y_train, y_val, y_test = split_result
            splits = ['train', 'validation', 'test']
            X_splits = [X_train, X_val, X_test]
            y_splits = [y_train, y_val, y_test]
        else:
            raise ValueError("Unsupported split result format")
            
        info = {
            'original_shape': original_shape,
            'total_samples': original_shape[0],
            'splits': {}
        }
        
        for split_name, X_split, y_split in zip(splits, X_splits, y_splits):
            info['splits'][split_name] = {
                'samples': X_split.shape[0],
                'percentage': X_split.shape[0] / original_shape[0] * 100,
                'shape': X_split.shape
            }
            
            # Add class distribution if applicable
            if len(y_split.shape) == 1:
                unique, counts = np.unique(y_split, return_counts=True)
                info['splits'][split_name]['class_distribution'] = dict(zip(unique.astype(str), counts.astype(int)))
                
        return info 