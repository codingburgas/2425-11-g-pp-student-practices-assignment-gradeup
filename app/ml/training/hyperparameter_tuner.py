# Hyperparameter tuning utilities for model optimization

import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union, Type, Callable
import time
import json
from itertools import product
from dataclasses import dataclass

from .base import BaseTrainer
from .cross_validator import CrossValidator, CrossValidationResult
from .model_trainer import ModelTrainer
from .validators import DataValidator


@dataclass
class HyperparameterResult:
    """Container for hyperparameter tuning results."""
    
    def __init__(self):
        """Initialize hyperparameter result container."""
        self.param_combinations: List[Dict[str, Any]] = []
        self.scores: List[float] = []
        self.cv_results: List[CrossValidationResult] = []
        self.best_params: Optional[Dict[str, Any]] = None
        self.best_score: float = float('-inf')
        self.best_index: int = -1
        self.tuning_time: float = 0.0
        self.n_combinations: int = 0
        
    def add_result(self, params: Dict[str, Any], score: float, cv_result: CrossValidationResult):
        """Add a parameter combination result."""
        self.param_combinations.append(params.copy())
        self.scores.append(score)
        self.cv_results.append(cv_result)
        self.n_combinations += 1
        
        # Update best if this is better
        if score > self.best_score:
            self.best_score = score
            self.best_params = params.copy()
            self.best_index = len(self.scores) - 1
            
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of tuning results."""
        if not self.scores:
            return {'error': 'No results available'}
            
        return {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'n_combinations_tested': self.n_combinations,
            'tuning_time': self.tuning_time,
            'score_statistics': {
                'mean': float(np.mean(self.scores)),
                'std': float(np.std(self.scores)),
                'min': float(np.min(self.scores)),
                'max': float(np.max(self.scores))
            }
        }
        
    def get_top_k_results(self, k: int = 5) -> List[Dict[str, Any]]:
        """Get top k parameter combinations."""
        if not self.scores:
            return []
            
        # Sort by score (descending)
        sorted_indices = np.argsort(self.scores)[::-1]
        
        top_results = []
        for i in range(min(k, len(sorted_indices))):
            idx = sorted_indices[i]
            top_results.append({
                'rank': i + 1,
                'params': self.param_combinations[idx],
                'score': self.scores[idx],
                'cv_std': np.std([result.get_mean_metrics() for result in [self.cv_results[idx]]])
            })
            
        return top_results


class HyperparameterTuner:
    """Comprehensive hyperparameter tuning framework."""
    
    def __init__(self, 
                 model_class: Type[BaseTrainer],
                 scoring_metric: str = 'accuracy',
                 cv_folds: int = 5,
                 cv_strategy: str = 'k_fold',
                 random_seed: Optional[int] = 42,
                 n_jobs: int = 1,
                 verbose: bool = True):
        """
        Initialize hyperparameter tuner.
        
        Args:
            model_class: Class of model to tune
            scoring_metric: Metric to optimize (higher is better)
            cv_folds: Number of cross-validation folds
            cv_strategy: CV strategy ('k_fold', 'stratified_k_fold')
            random_seed: Random seed for reproducibility
            n_jobs: Number of parallel jobs (not implemented, placeholder)
            verbose: Whether to print progress
        """
        self.model_class = model_class
        self.scoring_metric = scoring_metric
        self.cv_folds = cv_folds
        self.cv_strategy = cv_strategy
        self.random_seed = random_seed
        self.n_jobs = n_jobs
        self.verbose = verbose
        
        # Initialize cross-validator
        self.cross_validator = CrossValidator(random_seed=random_seed)
        
    def _evaluate_params(self, 
                        params: Dict[str, Any], 
                        X: np.ndarray, 
                        y: np.ndarray) -> Tuple[float, CrossValidationResult]:
        """Evaluate a single parameter combination using cross-validation."""
        if self.cv_strategy == 'k_fold':
            cv_result = self.cross_validator.k_fold_cv(
                self.model_class, params, X, y,
                k=self.cv_folds, verbose=False
            )
        elif self.cv_strategy == 'stratified_k_fold':
            cv_result = self.cross_validator.k_fold_cv(
                self.model_class, params, X, y,
                k=self.cv_folds, stratified=True, verbose=False
            )
        else:
            raise ValueError(f"Unknown CV strategy: {self.cv_strategy}")
            
        # Extract score
        mean_metrics = cv_result.get_mean_metrics()
        score_key = f'mean_{self.scoring_metric}'
        
        if score_key in mean_metrics:
            score = mean_metrics[score_key]
        else:
            # Try variations of the metric name
            for key in mean_metrics:
                if self.scoring_metric in key:
                    score = mean_metrics[key]
                    break
            else:
                # Default to first available metric
                score = list(mean_metrics.values())[0] if mean_metrics else 0.0
                
        return score, cv_result
        
    def grid_search(self, 
                   param_grid: Dict[str, List[Any]], 
                   X: np.ndarray, 
                   y: np.ndarray) -> HyperparameterResult:
        """
        Perform grid search over parameter combinations.
        
        Args:
            param_grid: Dictionary mapping parameter names to lists of values
            X: Training features
            y: Training targets
            
        Returns:
            HyperparameterResult object
        """
        if self.verbose:
            print("Starting grid search hyperparameter tuning...")
            
        start_time = time.time()
        
        # Validate inputs
        validator = DataValidator()
        validator.validate_input(X)
        validator.validate_labels(y)
        validator.validate_data_consistency(X, y)
        
        # Generate all parameter combinations
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        param_combinations = list(product(*param_values))
        
        if self.verbose:
            print(f"Testing {len(param_combinations)} parameter combinations...")
            
        result = HyperparameterResult()
        
        for i, param_combo in enumerate(param_combinations):
            params = dict(zip(param_names, param_combo))
            
            if self.verbose:
                print(f"  Combination {i + 1}/{len(param_combinations)}: {params}")
                
            # Evaluate parameters
            score, cv_result = self._evaluate_params(params, X, y)
            result.add_result(params, score, cv_result)
            
            if self.verbose:
                print(f"    Score: {score:.4f}")
                
        result.tuning_time = time.time() - start_time
        
        if self.verbose:
            print(f"Grid search completed in {result.tuning_time:.2f} seconds")
            print(f"Best score: {result.best_score:.4f}")
            print(f"Best params: {result.best_params}")
            
        return result
        
    def random_search(self, 
                     param_distributions: Dict[str, Any], 
                     X: np.ndarray, 
                     y: np.ndarray,
                     n_iter: int = 100) -> HyperparameterResult:
        """
        Perform random search over parameter distributions.
        
        Args:
            param_distributions: Dictionary mapping parameter names to distributions
            X: Training features
            y: Training targets
            n_iter: Number of parameter combinations to test
            
        Returns:
            HyperparameterResult object
        """
        if self.verbose:
            print("Starting random search hyperparameter tuning...")
            
        start_time = time.time()
        
        # Validate inputs
        validator = DataValidator()
        validator.validate_input(X)
        validator.validate_labels(y)
        validator.validate_data_consistency(X, y)
        
        if self.random_seed is not None:
            np.random.seed(self.random_seed)
            
        if self.verbose:
            print(f"Testing {n_iter} random parameter combinations...")
            
        result = HyperparameterResult()
        
        for i in range(n_iter):
            # Sample parameters
            params = {}
            for param_name, distribution in param_distributions.items():
                if isinstance(distribution, list):
                    # Choose randomly from list
                    params[param_name] = np.random.choice(distribution)
                elif isinstance(distribution, dict):
                    # Handle distribution specification
                    if distribution['type'] == 'uniform':
                        params[param_name] = np.random.uniform(
                            distribution['low'], distribution['high']
                        )
                    elif distribution['type'] == 'log_uniform':
                        params[param_name] = np.exp(np.random.uniform(
                            np.log(distribution['low']), np.log(distribution['high'])
                        ))
                    elif distribution['type'] == 'choice':
                        params[param_name] = np.random.choice(distribution['choices'])
                    elif distribution['type'] == 'randint':
                        params[param_name] = np.random.randint(
                            distribution['low'], distribution['high']
                        )
                    else:
                        raise ValueError(f"Unknown distribution type: {distribution['type']}")
                else:
                    raise ValueError(f"Invalid distribution format for {param_name}")
                    
            if self.verbose:
                print(f"  Combination {i + 1}/{n_iter}: {params}")
                
            # Evaluate parameters
            score, cv_result = self._evaluate_params(params, X, y)
            result.add_result(params, score, cv_result)
            
            if self.verbose:
                print(f"    Score: {score:.4f}")
                
        result.tuning_time = time.time() - start_time
        
        if self.verbose:
            print(f"Random search completed in {result.tuning_time:.2f} seconds")
            print(f"Best score: {result.best_score:.4f}")
            print(f"Best params: {result.best_params}")
            
        return result
        
    def bayesian_optimization(self, 
                             param_space: Dict[str, Dict[str, Any]], 
                             X: np.ndarray, 
                             y: np.ndarray,
                             n_iter: int = 50,
                             n_initial_points: int = 10) -> HyperparameterResult:
        """
        Perform Bayesian optimization (simplified implementation).
        
        This is a simplified Bayesian optimization that uses Gaussian Process
        surrogate model approximation without external libraries.
        
        Args:
            param_space: Dictionary mapping parameter names to space definitions
            X: Training features
            y: Training targets
            n_iter: Number of optimization iterations
            n_initial_points: Number of random initial points
            
        Returns:
            HyperparameterResult object
        """
        if self.verbose:
            print("Starting Bayesian optimization...")
            print("Note: Using simplified BO implementation without GP libraries")
            
        start_time = time.time()
        
        # Validate inputs
        validator = DataValidator()
        validator.validate_input(X)
        validator.validate_labels(y)
        validator.validate_data_consistency(X, y)
        
        if self.random_seed is not None:
            np.random.seed(self.random_seed)
            
        result = HyperparameterResult()
        
        # Phase 1: Random exploration
        if self.verbose:
            print(f"Phase 1: Random exploration with {n_initial_points} points...")
            
        for i in range(n_initial_points):
            params = self._sample_from_space(param_space)
            
            if self.verbose:
                print(f"  Initial point {i + 1}/{n_initial_points}: {params}")
                
            score, cv_result = self._evaluate_params(params, X, y)
            result.add_result(params, score, cv_result)
            
            if self.verbose:
                print(f"    Score: {score:.4f}")
                
        # Phase 2: Bayesian optimization (simplified)
        if self.verbose:
            print(f"Phase 2: Bayesian optimization with {n_iter - n_initial_points} iterations...")
            
        for i in range(n_initial_points, n_iter):
            # Simplified acquisition function (Upper Confidence Bound approximation)
            params = self._acquire_next_point(result, param_space)
            
            if self.verbose:
                print(f"  BO iteration {i + 1}/{n_iter}: {params}")
                
            score, cv_result = self._evaluate_params(params, X, y)
            result.add_result(params, score, cv_result)
            
            if self.verbose:
                print(f"    Score: {score:.4f}")
                
        result.tuning_time = time.time() - start_time
        
        if self.verbose:
            print(f"Bayesian optimization completed in {result.tuning_time:.2f} seconds")
            print(f"Best score: {result.best_score:.4f}")
            print(f"Best params: {result.best_params}")
            
        return result
        
    def _sample_from_space(self, param_space: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Sample parameters from the defined space."""
        params = {}
        
        for param_name, space_def in param_space.items():
            if space_def['type'] == 'uniform':
                params[param_name] = np.random.uniform(space_def['low'], space_def['high'])
            elif space_def['type'] == 'log_uniform':
                params[param_name] = np.exp(np.random.uniform(
                    np.log(space_def['low']), np.log(space_def['high'])
                ))
            elif space_def['type'] == 'choice':
                params[param_name] = np.random.choice(space_def['choices'])
            elif space_def['type'] == 'randint':
                params[param_name] = np.random.randint(space_def['low'], space_def['high'])
            else:
                raise ValueError(f"Unknown space type: {space_def['type']}")
                
        return params
        
    def _acquire_next_point(self, 
                           result: HyperparameterResult, 
                           param_space: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Simplified acquisition function for next point selection.
        
        This uses a combination of exploitation (near best points) and 
        exploration (random sampling) without full Gaussian Process.
        """
        # 50% exploitation, 50% exploration
        if np.random.random() < 0.5 and result.best_params is not None:
            # Exploitation: perturb best parameters
            params = result.best_params.copy()
            
            # Add small random perturbation
            for param_name, space_def in param_space.items():
                if space_def['type'] in ['uniform', 'log_uniform']:
                    current_val = params[param_name]
                    noise_scale = (space_def['high'] - space_def['low']) * 0.1
                    
                    if space_def['type'] == 'log_uniform':
                        # Perturbation in log space
                        log_val = np.log(current_val)
                        log_noise = np.random.normal(0, np.log(1 + noise_scale))
                        params[param_name] = np.exp(log_val + log_noise)
                        # Clip to bounds
                        params[param_name] = np.clip(params[param_name], 
                                                   space_def['low'], space_def['high'])
                    else:
                        # Linear perturbation
                        noise = np.random.normal(0, noise_scale)
                        params[param_name] = np.clip(current_val + noise,
                                                   space_def['low'], space_def['high'])
                        
        else:
            # Exploration: random sampling
            params = self._sample_from_space(param_space)
            
        return params
        
    def tune_with_budget(self, 
                        param_space: Dict[str, Any], 
                        X: np.ndarray, 
                        y: np.ndarray,
                        max_time_minutes: float = 60.0,
                        strategy: str = 'random') -> HyperparameterResult:
        """
        Perform hyperparameter tuning with time budget.
        
        Args:
            param_space: Parameter space definition
            X: Training features
            y: Training targets
            max_time_minutes: Maximum time budget in minutes
            strategy: Tuning strategy ('random', 'grid', 'bayesian')
            
        Returns:
            HyperparameterResult object
        """
        max_time_seconds = max_time_minutes * 60
        start_time = time.time()
        
        if self.verbose:
            print(f"Starting {strategy} search with {max_time_minutes:.1f} minute budget...")
            
        result = HyperparameterResult()
        iteration = 0
        
        while (time.time() - start_time) < max_time_seconds:
            if strategy == 'random':
                params = self._sample_from_space(param_space) if isinstance(param_space, dict) and all(isinstance(v, dict) for v in param_space.values()) else self._sample_random_params(param_space)
            else:
                # For grid and bayesian, we'd need to implement iterative versions
                params = self._sample_random_params(param_space)
                
            if self.verbose:
                elapsed = time.time() - start_time
                print(f"  Iteration {iteration + 1} ({elapsed:.1f}s): {params}")
                
            score, cv_result = self._evaluate_params(params, X, y)
            result.add_result(params, score, cv_result)
            
            if self.verbose:
                print(f"    Score: {score:.4f}")
                
            iteration += 1
            
        result.tuning_time = time.time() - start_time
        
        if self.verbose:
            print(f"Budget search completed: {iteration} iterations in {result.tuning_time:.2f} seconds")
            print(f"Best score: {result.best_score:.4f}")
            print(f"Best params: {result.best_params}")
            
        return result
        
    def _sample_random_params(self, param_space: Dict[str, Any]) -> Dict[str, Any]:
        """Sample random parameters from space (handles both formats)."""
        params = {}
        
        for param_name, space in param_space.items():
            if isinstance(space, list):
                params[param_name] = np.random.choice(space)
            elif isinstance(space, dict):
                params[param_name] = self._sample_from_space({param_name: space})[param_name]
            else:
                # Assume it's a single value
                params[param_name] = space
                
        return params 