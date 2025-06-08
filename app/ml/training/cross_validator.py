# Cross-validation framework implemented from scratch

import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
import time
import json
from .data_splitter import DataSplitter
from .validators import DataValidator
from .base import BaseTrainer, TrainingResult
from .linear_models import LinearRegression, LogisticRegression


class CrossValidationResult:
    """Container for cross-validation results."""
    
    def __init__(self, cv_strategy: str, n_folds: int):
        """
        Initialize CV result container.
        
        Args:
            cv_strategy: Cross-validation strategy used
            n_folds: Number of folds
        """
        self.cv_strategy = cv_strategy
        self.n_folds = n_folds
        self.fold_results: List[Dict[str, Any]] = []
        self.fold_models: List[Any] = []
        self.training_times: List[float] = []
        self.evaluation_times: List[float] = []
        
    def add_fold_result(self, fold_idx: int, metrics: Dict[str, float], 
                       model: Any, train_time: float, eval_time: float):
        """Add results from a single fold."""
        result = {
            'fold': fold_idx,
            'metrics': metrics.copy(),
            'train_time': train_time,
            'eval_time': eval_time
        }
        self.fold_results.append(result)
        self.fold_models.append(model)
        self.training_times.append(train_time)
        self.evaluation_times.append(eval_time)
        
    def get_mean_metrics(self) -> Dict[str, float]:
        """Get mean metrics across all folds."""
        if not self.fold_results:
            return {}
            
        # Get all metric names
        metric_names = set()
        for result in self.fold_results:
            metric_names.update(result['metrics'].keys())
            
        mean_metrics = {}
        for metric_name in metric_names:
            values = [result['metrics'].get(metric_name, 0) for result in self.fold_results]
            mean_metrics[f'mean_{metric_name}'] = np.mean(values)
            mean_metrics[f'std_{metric_name}'] = np.std(values)
            
        return mean_metrics
        
    def get_summary(self) -> Dict[str, Any]:
        """Get comprehensive summary of CV results."""
        mean_metrics = self.get_mean_metrics()
        
        return {
            'cv_strategy': self.cv_strategy,
            'n_folds': self.n_folds,
            'mean_metrics': mean_metrics,
            'total_training_time': sum(self.training_times),
            'total_evaluation_time': sum(self.evaluation_times),
            'mean_training_time_per_fold': np.mean(self.training_times),
            'mean_evaluation_time_per_fold': np.mean(self.evaluation_times),
            'fold_results': self.fold_results
        }


class CrossValidator:
    """Comprehensive cross-validation framework."""
    
    def __init__(self, random_seed: Optional[int] = 42):
        """
        Initialize cross-validator.
        
        Args:
            random_seed: Random seed for reproducibility
        """
        self.random_seed = random_seed
        self.data_splitter = DataSplitter(random_seed=random_seed)
        
    def k_fold_cv(self, 
                  model_class: type,
                  model_params: Dict[str, Any],
                  X: np.ndarray,
                  y: np.ndarray,
                  k: int = 5,
                  shuffle: bool = True,
                  stratified: bool = False,
                  verbose: bool = False) -> CrossValidationResult:
        """
        Perform k-fold cross-validation.
        
        Args:
            model_class: Class of model to train
            model_params: Parameters for model initialization
            X: Feature data
            y: Target data
            k: Number of folds
            shuffle: Whether to shuffle data before splitting
            stratified: Whether to use stratified splitting
            verbose: Whether to print progress
            
        Returns:
            CrossValidationResult object
        """
        # Validate inputs
        validator = DataValidator()
        validator.validate_input(X)
        validator.validate_labels(y)
        validator.validate_data_consistency(X, y)
        
        if verbose:
            print(f"Starting {k}-fold {'stratified ' if stratified else ''}cross-validation...")
            
        # Generate fold indices
        if stratified and len(y.shape) == 1:
            fold_indices = self.data_splitter.stratified_k_fold_indices(y, k, shuffle)
            cv_strategy = 'stratified_k_fold'
        else:
            fold_indices = self.data_splitter.k_fold_indices(X.shape[0], k, shuffle)
            cv_strategy = 'k_fold'
            
        cv_result = CrossValidationResult(cv_strategy, k)
        
        # Perform cross-validation
        for fold_idx, (train_indices, val_indices) in enumerate(fold_indices):
            if verbose:
                print(f"  Fold {fold_idx + 1}/{k}")
                
            # Split data
            X_train, X_val = X[train_indices], X[val_indices]
            y_train, y_val = y[train_indices], y[val_indices]
            
            # Initialize and train model
            model = model_class(**model_params)
            
            start_time = time.time()
            model.fit(X_train, y_train, verbose=False)
            train_time = time.time() - start_time
            
            # Evaluate model
            start_time = time.time()
            metrics = model.evaluate(X_val, y_val)
            eval_time = time.time() - start_time
            
            # Store results
            cv_result.add_fold_result(fold_idx, metrics, model, train_time, eval_time)
            
            if verbose:
                print(f"    Validation metrics: {metrics}")
                
        if verbose:
            mean_metrics = cv_result.get_mean_metrics()
            print(f"Mean CV metrics: {mean_metrics}")
            
        return cv_result
        
    def time_series_cv(self,
                      model_class: type,
                      model_params: Dict[str, Any],
                      X: np.ndarray,
                      y: np.ndarray,
                      n_splits: int = 5,
                      test_size: float = 0.2,
                      verbose: bool = False) -> CrossValidationResult:
        """
        Perform time series cross-validation.
        
        Args:
            model_class: Class of model to train
            model_params: Parameters for model initialization
            X: Feature data (ordered by time)
            y: Target data (ordered by time)
            n_splits: Number of train/test splits
            test_size: Fraction of data to use for testing in each split
            verbose: Whether to print progress
            
        Returns:
            CrossValidationResult object
        """
        # Validate inputs
        validator = DataValidator()
        validator.validate_input(X)
        validator.validate_labels(y)
        validator.validate_data_consistency(X, y)
        
        if verbose:
            print(f"Starting time series cross-validation with {n_splits} splits...")
            
        cv_result = CrossValidationResult('time_series', n_splits)
        n_samples = X.shape[0]
        test_samples = int(n_samples * test_size)
        
        # Generate time series splits
        for split_idx in range(n_splits):
            # Calculate split point (moving forward in time)
            split_point = int(n_samples * (split_idx + 1) / (n_splits + 1))
            train_end = split_point
            test_start = train_end
            test_end = min(test_start + test_samples, n_samples)
            
            if train_end < 10 or test_end - test_start < 5:  # Ensure minimum samples
                continue
                
            if verbose:
                print(f"  Split {split_idx + 1}/{n_splits} - Train: 0:{train_end}, Test: {test_start}:{test_end}")
                
            # Split data
            X_train, X_test = X[:train_end], X[test_start:test_end]
            y_train, y_test = y[:train_end], y[test_start:test_end]
            
            # Initialize and train model
            model = model_class(**model_params)
            
            start_time = time.time()
            model.fit(X_train, y_train, verbose=False)
            train_time = time.time() - start_time
            
            # Evaluate model
            start_time = time.time()
            metrics = model.evaluate(X_test, y_test)
            eval_time = time.time() - start_time
            
            # Store results
            cv_result.add_fold_result(split_idx, metrics, model, train_time, eval_time)
            
            if verbose:
                print(f"    Test metrics: {metrics}")
                
        if verbose:
            mean_metrics = cv_result.get_mean_metrics()
            print(f"Mean time series CV metrics: {mean_metrics}")
            
        return cv_result
        
    def leave_one_out_cv(self,
                        model_class: type,
                        model_params: Dict[str, Any],
                        X: np.ndarray,
                        y: np.ndarray,
                        max_samples: int = 100,
                        verbose: bool = False) -> CrossValidationResult:
        """
        Perform leave-one-out cross-validation.
        
        Args:
            model_class: Class of model to train
            model_params: Parameters for model initialization
            X: Feature data
            y: Target data
            max_samples: Maximum number of samples to use (for computational efficiency)
            verbose: Whether to print progress
            
        Returns:
            CrossValidationResult object
        """
        # Validate inputs
        validator = DataValidator()
        validator.validate_input(X)
        validator.validate_labels(y)
        validator.validate_data_consistency(X, y)
        
        n_samples = X.shape[0]
        if n_samples > max_samples:
            print(f"Warning: Using only first {max_samples} samples for LOOCV (out of {n_samples})")
            X = X[:max_samples]
            y = y[:max_samples]
            n_samples = max_samples
            
        if verbose:
            print(f"Starting leave-one-out cross-validation with {n_samples} samples...")
            
        cv_result = CrossValidationResult('leave_one_out', n_samples)
        
        # Perform LOOCV
        for i in range(n_samples):
            if verbose and (i + 1) % 10 == 0:
                print(f"  Sample {i + 1}/{n_samples}")
                
            # Create train/test split
            train_indices = np.concatenate([np.arange(i), np.arange(i + 1, n_samples)])
            test_indices = np.array([i])
            
            X_train, X_test = X[train_indices], X[test_indices]
            y_train, y_test = y[train_indices], y[test_indices]
            
            # Initialize and train model
            model = model_class(**model_params)
            
            start_time = time.time()
            model.fit(X_train, y_train, verbose=False)
            train_time = time.time() - start_time
            
            # Evaluate model
            start_time = time.time()
            metrics = model.evaluate(X_test, y_test)
            eval_time = time.time() - start_time
            
            # Store results
            cv_result.add_fold_result(i, metrics, model, train_time, eval_time)
            
        if verbose:
            mean_metrics = cv_result.get_mean_metrics()
            print(f"Mean LOOCV metrics: {mean_metrics}")
            
        return cv_result
        
    def repeated_cv(self,
                   model_class: type,
                   model_params: Dict[str, Any],
                   X: np.ndarray,
                   y: np.ndarray,
                   k: int = 5,
                   n_repeats: int = 3,
                   stratified: bool = False,
                   verbose: bool = False) -> List[CrossValidationResult]:
        """
        Perform repeated k-fold cross-validation.
        
        Args:
            model_class: Class of model to train
            model_params: Parameters for model initialization
            X: Feature data
            y: Target data
            k: Number of folds
            n_repeats: Number of repetitions
            stratified: Whether to use stratified splitting
            verbose: Whether to print progress
            
        Returns:
            List of CrossValidationResult objects, one per repeat
        """
        if verbose:
            print(f"Starting repeated {k}-fold CV with {n_repeats} repeats...")
            
        results = []
        
        for repeat_idx in range(n_repeats):
            if verbose:
                print(f"Repeat {repeat_idx + 1}/{n_repeats}")
                
            # Use different random seed for each repeat
            original_seed = self.random_seed
            if self.random_seed is not None:
                self.data_splitter = DataSplitter(random_seed=self.random_seed + repeat_idx)
                
            # Perform k-fold CV
            cv_result = self.k_fold_cv(
                model_class, model_params, X, y, k, 
                shuffle=True, stratified=stratified, verbose=verbose
            )
            
            results.append(cv_result)
            
            # Restore original random seed
            self.data_splitter = DataSplitter(random_seed=original_seed)
            
        return results
        
    def compare_models(self,
                      model_configs: Dict[str, Dict[str, Any]],
                      X: np.ndarray,
                      y: np.ndarray,
                      cv_strategy: str = 'k_fold',
                      cv_params: Optional[Dict[str, Any]] = None,
                      verbose: bool = False) -> Dict[str, CrossValidationResult]:
        """
        Compare multiple models using cross-validation.
        
        Args:
            model_configs: Dictionary mapping model names to (class, params) dict
            X: Feature data
            y: Target data
            cv_strategy: CV strategy ('k_fold', 'stratified_k_fold', 'time_series')
            cv_params: Parameters for CV strategy
            verbose: Whether to print progress
            
        Returns:
            Dictionary mapping model names to CV results
        """
        if cv_params is None:
            cv_params = {}
            
        if verbose:
            print(f"Comparing {len(model_configs)} models using {cv_strategy} CV...")
            
        results = {}
        
        for model_name, config in model_configs.items():
            if verbose:
                print(f"\nEvaluating model: {model_name}")
                
            model_class = config['class']
            model_params = config.get('params', {})
            
            if cv_strategy == 'k_fold':
                cv_result = self.k_fold_cv(
                    model_class, model_params, X, y, 
                    verbose=verbose, **cv_params
                )
            elif cv_strategy == 'stratified_k_fold':
                cv_result = self.k_fold_cv(
                    model_class, model_params, X, y,
                    stratified=True, verbose=verbose, **cv_params
                )
            elif cv_strategy == 'time_series':
                cv_result = self.time_series_cv(
                    model_class, model_params, X, y,
                    verbose=verbose, **cv_params
                )
            else:
                raise ValueError(f"Unknown CV strategy: {cv_strategy}")
                
            results[model_name] = cv_result
            
            if verbose:
                mean_metrics = cv_result.get_mean_metrics()
                print(f"  Mean metrics: {mean_metrics}")
                
        return results
        
    def nested_cv(self,
                 model_class: type,
                 param_grid: Dict[str, List[Any]],
                 X: np.ndarray,
                 y: np.ndarray,
                 outer_cv: int = 5,
                 inner_cv: int = 3,
                 scoring_metric: str = 'mse',
                 verbose: bool = False) -> Dict[str, Any]:
        """
        Perform nested cross-validation for hyperparameter tuning.
        
        Args:
            model_class: Class of model to train
            param_grid: Grid of parameters to search
            X: Feature data
            y: Target data
            outer_cv: Number of outer CV folds
            inner_cv: Number of inner CV folds
            scoring_metric: Metric to optimize
            verbose: Whether to print progress
            
        Returns:
            Dictionary with nested CV results
        """
        if verbose:
            print(f"Starting nested CV: outer={outer_cv}, inner={inner_cv}")
            
        # Generate outer CV folds
        outer_fold_indices = self.data_splitter.k_fold_indices(X.shape[0], outer_cv, shuffle=True)
        
        nested_results = {
            'outer_scores': [],
            'best_params_per_fold': [],
            'inner_cv_results': []
        }
        
        for outer_fold_idx, (train_idx, test_idx) in enumerate(outer_fold_indices):
            if verbose:
                print(f"Outer fold {outer_fold_idx + 1}/{outer_cv}")
                
            X_train_outer, X_test_outer = X[train_idx], X[test_idx]
            y_train_outer, y_test_outer = y[train_idx], y[test_idx]
            
            # Inner CV for hyperparameter selection
            best_params = None
            best_score = float('inf') if 'loss' in scoring_metric or 'error' in scoring_metric else float('-inf')
            
            # Generate all parameter combinations
            param_names = list(param_grid.keys())
            param_values = list(param_grid.values())
            
            from itertools import product
            param_combinations = list(product(*param_values))
            
            for param_combo in param_combinations:
                params = dict(zip(param_names, param_combo))
                
                # Inner CV
                inner_cv_result = self.k_fold_cv(
                    model_class, params, X_train_outer, y_train_outer,
                    k=inner_cv, verbose=False
                )
                
                # Get mean score
                mean_metrics = inner_cv_result.get_mean_metrics()
                score_key = f'mean_{scoring_metric}'
                
                if score_key in mean_metrics:
                    score = mean_metrics[score_key]
                    
                    # Check if this is the best score
                    is_better = (score < best_score if 'loss' in scoring_metric or 'error' in scoring_metric 
                               else score > best_score)
                    
                    if is_better:
                        best_score = score
                        best_params = params
                        
            # Train model with best parameters on full outer training set
            best_model = model_class(**best_params)
            best_model.fit(X_train_outer, y_train_outer, verbose=False)
            
            # Evaluate on outer test set
            outer_metrics = best_model.evaluate(X_test_outer, y_test_outer)
            outer_score = outer_metrics.get(scoring_metric, 0)
            
            nested_results['outer_scores'].append(outer_score)
            nested_results['best_params_per_fold'].append(best_params)
            
            if verbose:
                print(f"  Best params: {best_params}")
                print(f"  Outer score: {outer_score}")
                
        # Compute final nested CV score
        mean_outer_score = np.mean(nested_results['outer_scores'])
        std_outer_score = np.std(nested_results['outer_scores'])
        
        nested_results['mean_score'] = mean_outer_score
        nested_results['std_score'] = std_outer_score
        
        if verbose:
            print(f"Nested CV score: {mean_outer_score:.4f} ± {std_outer_score:.4f}")
            
        return nested_results 