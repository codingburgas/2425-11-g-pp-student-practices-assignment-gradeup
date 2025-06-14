# End-to-end ML training pipeline combining all system components

import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union, Type
import time
import json
from pathlib import Path

from .base import BaseTrainer, TrainingConfig, TrainingResult
from .validators import DataValidator, ParameterValidator
from .preprocessing import DataPreprocessor
from .data_splitter import DataSplitter
from .model_trainer import ModelTrainer
from .cross_validator import CrossValidator, CrossValidationResult
from .hyperparameter_tuner import HyperparameterTuner, HyperparameterResult
from .performance_tracker import PerformanceTracker, ExperimentRun
from .linear_models import LinearRegression, LogisticRegression


class TrainingPipeline:
    """Comprehensive end-to-end machine learning training pipeline."""
    
    def __init__(self, 
                 experiment_name: str = "ml_pipeline_experiment",
                 log_dir: str = "experiments",
                 random_seed: Optional[int] = 42,
                 auto_tracking: bool = True):
        """
        Initialize training pipeline.
        
        Args:
            experiment_name: Name for the experiment
            log_dir: Directory for saving logs and results
            random_seed: Random seed for reproducibility
            auto_tracking: Whether to automatically track experiments
        """
        self.experiment_name = experiment_name
        self.log_dir = Path(log_dir)
        self.random_seed = random_seed
        self.auto_tracking = auto_tracking
        
        # Initialize components
        self.data_splitter = DataSplitter(random_seed=random_seed)
        self.cross_validator = CrossValidator(random_seed=random_seed)
        
        # Performance tracking
        if auto_tracking:
            self.performance_tracker = PerformanceTracker(
                experiment_name=experiment_name,
                log_dir=str(log_dir)
            )
        else:
            self.performance_tracker = None
            
        # Pipeline state
        self.current_data = None
        self.trained_models = {}
        self.pipeline_results = {}
        
    def prepare_data(self, 
                    X: np.ndarray, 
                    y: np.ndarray,
                    test_size: float = 0.2,
                    validation_size: float = 0.2,
                    preprocessing_config: Optional[Dict[str, Any]] = None,
                    stratify: bool = False) -> Dict[str, Any]:
        """
        Prepare data with preprocessing and splitting.
        
        Args:
            X: Input features
            y: Target labels
            test_size: Fraction for test set
            validation_size: Fraction for validation set
            preprocessing_config: Configuration for preprocessing
            stratify: Whether to use stratified splitting
            
        Returns:
            Dictionary containing prepared data splits
        """
        print("Preparing data...")
        start_time = time.time()
        
        # Validate inputs
        validator = DataValidator()
        validator.validate_input(X)
        validator.validate_labels(y)
        validator.validate_data_consistency(X, y)
        
        # Get data summary
        data_summary = validator.get_data_summary(X)
        class_distribution = validator.check_class_distribution(y)
        
        print(f"Dataset shape: {X.shape}")
        print(f"Features: {X.shape[1]}, Samples: {X.shape[0]}")
        if len(y.shape) == 1:
            print(f"Classes: {class_distribution.get('num_classes', 'N/A')}")
            
        # Initialize preprocessor if config provided
        preprocessor = None
        if preprocessing_config:
            preprocessor = DataPreprocessor(**preprocessing_config)
            X_processed = preprocessor.fit_transform(X)
            print(f"Applied preprocessing: {preprocessor.get_preprocessing_info()}")
        else:
            X_processed = X
            
        # Split data
        if validation_size > 0:
            X_train, X_val, X_test, y_train, y_val, y_test = self.data_splitter.train_validation_test_split(
                X_processed, y, 
                validation_size=validation_size,
                test_size=test_size,
                stratify=stratify
            )
            
            split_info = self.data_splitter.get_split_info(
                X_processed, y, (X_train, X_val, X_test, y_train, y_val, y_test)
            )
        else:
            X_train, X_test, y_train, y_test = self.data_splitter.train_test_split(
                X_processed, y,
                test_size=test_size,
                stratify=stratify
            )
            X_val, y_val = None, None
            
            split_info = self.data_splitter.get_split_info(
                X_processed, y, (X_train, X_test, y_train, y_test)
            )
            
        # Store data
        self.current_data = {
            'X_train': X_train,
            'X_val': X_val,
            'X_test': X_test,
            'y_train': y_train,
            'y_val': y_val,
            'y_test': y_test,
            'preprocessor': preprocessor,
            'data_summary': data_summary,
            'class_distribution': class_distribution,
            'split_info': split_info
        }
        
        prep_time = time.time() - start_time
        print(f"Data preparation completed in {prep_time:.2f} seconds")
        print(f"Split info: {split_info}")
        
        return self.current_data
        
    def train_model(self, 
                   model_class: Type[BaseTrainer],
                   model_params: Optional[Dict[str, Any]] = None,
                   training_config: Optional[TrainingConfig] = None,
                   model_name: Optional[str] = None,
                   track_performance: bool = True) -> ModelTrainer:
        """
        Train a single model with the prepared data.
        
        Args:
            model_class: Class of model to train
            model_params: Parameters for model initialization
            training_config: Training configuration
            model_name: Name for the model (auto-generated if None)
            track_performance: Whether to track performance
            
        Returns:
            Trained ModelTrainer instance
        """
        if self.current_data is None:
            raise ValueError("No data prepared. Call prepare_data() first.")
            
        if model_name is None:
            model_name = f"{model_class.__name__}_{len(self.trained_models)}"
            
        print(f"\nTraining model: {model_name}")
        
        # Start performance tracking
        run_id = None
        if track_performance and self.performance_tracker:
            run_id = self.performance_tracker.start_run(
                model_name=model_name,
                parameters=model_params or {},
                dataset_info={
                    'shape': self.current_data['X_train'].shape,
                    'split_info': self.current_data['split_info'],
                    'preprocessing': self.current_data['preprocessor'].get_preprocessing_info() if self.current_data['preprocessor'] else None
                }
            )
            
        # Initialize trainer
        trainer = ModelTrainer(
            model_class=model_class,
            model_params=model_params,
            training_config=training_config,
            random_seed=self.random_seed
        )
        
        # Prepare validation data
        validation_data = None
        if self.current_data['X_val'] is not None:
            validation_data = (self.current_data['X_val'], self.current_data['y_val'])
            
        # Train model
        training_result = trainer.train(
            self.current_data['X_train'], 
            self.current_data['y_train'],
            validation_data=validation_data,
            verbose=True
        )
        
        # Evaluate on test set
        test_metrics = trainer.evaluate(
            self.current_data['X_test'], 
            self.current_data['y_test']
        )
        
        print(f"Test metrics: {test_metrics}")
        
        # Track performance
        if track_performance and self.performance_tracker:
            self.performance_tracker.log_training_result(training_result, run_id)
            self.performance_tracker.end_run(
                final_metrics=test_metrics,
                model_name=model_name,
                parameters=model_params or {},
                training_time=training_result.training_time,
                dataset_info=self.current_data.get('data_summary', {}),
                run_id=run_id
            )
            
        # Store trained model
        self.trained_models[model_name] = {
            'trainer': trainer,
            'training_result': training_result,
            'test_metrics': test_metrics,
            'run_id': run_id
        }
        
        return trainer
        
    def compare_models(self, 
                      model_configs: Dict[str, Dict[str, Any]],
                      training_config: Optional[TrainingConfig] = None) -> Dict[str, Any]:
        """
        Train and compare multiple models.
        
        Args:
            model_configs: Dictionary mapping model names to configuration
            training_config: Shared training configuration
            
        Returns:
            Comparison results
        """
        if self.current_data is None:
            raise ValueError("No data prepared. Call prepare_data() first.")
            
        print(f"\nComparing {len(model_configs)} models...")
        
        comparison_results = {
            'models': {},
            'rankings': {},
            'summary': {}
        }
        
        # Train each model
        for model_name, config in model_configs.items():
            print(f"\n--- Training {model_name} ---")
            
            model_class = config['class']
            model_params = config.get('params', {})
            
            trainer = self.train_model(
                model_class=model_class,
                model_params=model_params,
                training_config=training_config,
                model_name=model_name
            )
            
            comparison_results['models'][model_name] = self.trained_models[model_name]
            
        # Compare using performance tracker
        if self.performance_tracker:
            run_ids = [self.trained_models[name]['run_id'] for name in model_configs.keys() 
                      if self.trained_models[name]['run_id']]
            
            if run_ids:
                tracker_comparison = self.performance_tracker.compare_models(run_ids)
                comparison_results['tracker_comparison'] = tracker_comparison
                
        # Manual comparison
        metrics_comparison = {}
        all_metrics = set()
        
        for model_name in model_configs.keys():
            model_metrics = self.trained_models[model_name]['test_metrics']
            all_metrics.update(model_metrics.keys())
            
        for metric_name in all_metrics:
            metric_values = {}
            for model_name in model_configs.keys():
                model_metrics = self.trained_models[model_name]['test_metrics']
                if metric_name in model_metrics:
                    metric_values[model_name] = model_metrics[metric_name]
                    
            if metric_values:
                # Determine if higher or lower is better
                is_loss_metric = any(term in metric_name.lower() for term in ['loss', 'error', 'mse', 'rmse'])
                
                sorted_models = sorted(metric_values.items(), 
                                     key=lambda x: x[1], 
                                     reverse=not is_loss_metric)
                
                metrics_comparison[metric_name] = {
                    'values': metric_values,
                    'ranking': [model for model, _ in sorted_models],
                    'best_model': sorted_models[0][0],
                    'best_value': sorted_models[0][1]
                }
                
        comparison_results['metrics_comparison'] = metrics_comparison
        
        # Print summary
        print("\n" + "="*60)
        print("MODEL COMPARISON SUMMARY")
        print("="*60)
        
        for metric_name, metric_data in metrics_comparison.items():
            print(f"\n{metric_name.upper()}:")
            for i, (model, value) in enumerate(zip(metric_data['ranking'], 
                                                  [metric_data['values'][m] for m in metric_data['ranking']])):
                print(f"  {i+1}. {model}: {value:.4f}")
                
        return comparison_results
        
    def hyperparameter_search(self, 
                             model_class: Type[BaseTrainer],
                             param_space: Dict[str, Any],
                             search_strategy: str = 'random',
                             n_iter: int = 50,
                             cv_folds: int = 5,
                             scoring_metric: str = 'accuracy',
                             model_name: Optional[str] = None) -> HyperparameterResult:
        """
        Perform hyperparameter search for a model.
        
        Args:
            model_class: Class of model to tune
            param_space: Parameter space definition
            search_strategy: Search strategy ('grid', 'random', 'bayesian')
            n_iter: Number of iterations for random/bayesian search
            cv_folds: Number of CV folds
            scoring_metric: Metric to optimize
            model_name: Name for tracking
            
        Returns:
            HyperparameterResult object
        """
        if self.current_data is None:
            raise ValueError("No data prepared. Call prepare_data() first.")
            
        if model_name is None:
            model_name = f"{model_class.__name__}_hypersearch"
            
        print(f"\nPerforming {search_strategy} hyperparameter search for {model_name}...")
        
        # Combine training and validation data for cross-validation
        if self.current_data['X_val'] is not None:
            X_search = np.vstack([self.current_data['X_train'], self.current_data['X_val']])
            y_search = np.hstack([self.current_data['y_train'], self.current_data['y_val']])
        else:
            X_search = self.current_data['X_train']
            y_search = self.current_data['y_train']
            
        # Initialize tuner
        tuner = HyperparameterTuner(
            model_class=model_class,
            scoring_metric=scoring_metric,
            cv_folds=cv_folds,
            random_seed=self.random_seed
        )
        
        # Perform search
        if search_strategy == 'grid':
            result = tuner.grid_search(param_space, X_search, y_search)
        elif search_strategy == 'random':
            result = tuner.random_search(param_space, X_search, y_search, n_iter=n_iter)
        elif search_strategy == 'bayesian':
            result = tuner.bayesian_optimization(param_space, X_search, y_search, n_iter=n_iter)
        else:
            raise ValueError(f"Unknown search strategy: {search_strategy}")
            
        # Train best model on full training data and evaluate
        print(f"\nTraining best model with parameters: {result.best_params}")
        
        best_trainer = self.train_model(
            model_class=model_class,
            model_params=result.best_params,
            model_name=f"{model_name}_best",
            track_performance=True
        )
        
        print(f"Hyperparameter search completed!")
        print(f"Best CV score: {result.best_score:.4f}")
        print(f"Best parameters: {result.best_params}")
        
        return result
        
    def cross_validate_model(self, 
                            model_class: Type[BaseTrainer],
                            model_params: Optional[Dict[str, Any]] = None,
                            cv_strategy: str = 'k_fold',
                            k: int = 5,
                            stratified: bool = False) -> CrossValidationResult:
        """
        Perform cross-validation on a model.
        
        Args:
            model_class: Class of model to validate
            model_params: Model parameters
            cv_strategy: CV strategy
            k: Number of folds
            stratified: Whether to use stratified CV
            
        Returns:
            CrossValidationResult object
        """
        if self.current_data is None:
            raise ValueError("No data prepared. Call prepare_data() first.")
            
        print(f"\nPerforming {cv_strategy} cross-validation...")
        
        # Use training data for CV
        X_cv = self.current_data['X_train']
        y_cv = self.current_data['y_train']
        
        # Perform cross-validation
        cv_result = self.cross_validator.k_fold_cv(
            model_class=model_class,
            model_params=model_params or {},
            X=X_cv,
            y=y_cv,
            k=k,
            stratified=stratified,
            verbose=True
        )
        
        # Print results
        mean_metrics = cv_result.get_mean_metrics()
        print(f"\nCross-validation results:")
        for metric_name, value in mean_metrics.items():
            print(f"  {metric_name}: {value:.4f}")
            
        return cv_result
        
    def full_pipeline(self, 
                     X: np.ndarray, 
                     y: np.ndarray,
                     model_configs: Dict[str, Dict[str, Any]],
                     preprocessing_config: Optional[Dict[str, Any]] = None,
                     training_config: Optional[TrainingConfig] = None,
                     hyperparameter_search: bool = False,
                     search_params: Optional[Dict[str, Any]] = None,
                     cross_validate: bool = True,
                     test_size: float = 0.2,
                     validation_size: float = 0.2) -> Dict[str, Any]:
        """
        Run complete ML pipeline from data preparation to model comparison.
        
        Args:
            X: Input features
            y: Target labels
            model_configs: Dictionary of model configurations
            preprocessing_config: Preprocessing configuration
            training_config: Training configuration
            hyperparameter_search: Whether to perform hyperparameter search
            search_params: Parameters for hyperparameter search
            cross_validate: Whether to perform cross-validation
            test_size: Test set size
            validation_size: Validation set size
            
        Returns:
            Complete pipeline results
        """
        print("="*60)
        print("STARTING FULL ML PIPELINE")
        print("="*60)
        
        pipeline_start_time = time.time()
        results = {}
        
        # Step 1: Data preparation
        data_info = self.prepare_data(
            X, y, 
            test_size=test_size, 
            validation_size=validation_size,
            preprocessing_config=preprocessing_config,
            stratify=True
        )
        results['data_preparation'] = data_info
        
        # Step 2: Cross-validation (if requested)
        if cross_validate:
            print("\n" + "="*40)
            print("CROSS-VALIDATION PHASE")
            print("="*40)
            
            cv_results = {}
            for model_name, config in model_configs.items():
                cv_result = self.cross_validate_model(
                    model_class=config['class'],
                    model_params=config.get('params', {}),
                    cv_strategy='stratified_k_fold' if len(y.shape) == 1 else 'k_fold'
                )
                cv_results[model_name] = cv_result
                
            results['cross_validation'] = cv_results
            
        # Step 3: Hyperparameter search (if requested)
        if hyperparameter_search and search_params:
            print("\n" + "="*40)
            print("HYPERPARAMETER SEARCH PHASE")
            print("="*40)
            
            search_results = {}
            for model_name, config in model_configs.items():
                if model_name in search_params:
                    search_config = search_params[model_name]
                    search_result = self.hyperparameter_search(
                        model_class=config['class'],
                        param_space=search_config.get('param_space', {}),
                        search_strategy=search_config.get('strategy', 'random'),
                        n_iter=search_config.get('n_iter', 50),
                        scoring_metric=search_config.get('scoring_metric', 'accuracy'),
                        model_name=model_name
                    )
                    search_results[model_name] = search_result
                    
                    # Update model config with best parameters
                    model_configs[model_name]['params'] = search_result.best_params
                    
            results['hyperparameter_search'] = search_results
            
        # Step 4: Model training and comparison
        print("\n" + "="*40)
        print("MODEL TRAINING PHASE")
        print("="*40)
        
        comparison_results = self.compare_models(model_configs, training_config)
        results['model_comparison'] = comparison_results
        
        # Step 5: Final summary
        total_time = time.time() - pipeline_start_time
        
        print("\n" + "="*60)
        print("PIPELINE COMPLETED")
        print("="*60)
        print(f"Total time: {total_time:.2f} seconds")
        
        if self.performance_tracker:
            summary = self.performance_tracker.get_experiment_summary()
            print(f"Total experiments tracked: {summary['total_runs']}")
            results['experiment_summary'] = summary
            
        results['total_time'] = total_time
        self.pipeline_results = results
        
        return results
        
    def save_pipeline_results(self, filename: Optional[str] = None) -> str:
        """Save complete pipeline results to file."""
        if not self.pipeline_results:
            raise ValueError("No pipeline results to save. Run full_pipeline() first.")
            
        if filename is None:
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            filename = f"{self.experiment_name}_pipeline_results_{timestamp}.json"
            
        filepath = self.log_dir / filename
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = self._make_json_serializable(self.pipeline_results)
        
        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=2)
            
        print(f"Pipeline results saved to: {filepath}")
        return str(filepath)
        
    def _make_json_serializable(self, obj):
        """Recursively convert numpy arrays and other non-serializable objects."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif hasattr(obj, 'to_dict'):
            return self._make_json_serializable(obj.to_dict())
        else:
            return obj
            
    def get_best_model(self, metric: str = 'accuracy') -> Tuple[str, ModelTrainer]:
        """
        Get the best performing model based on a metric.
        
        Args:
            metric: Metric to use for comparison
            
        Returns:
            Tuple of (model_name, trainer)
        """
        if not self.trained_models:
            raise ValueError("No trained models available.")
            
        best_model_name = None
        best_score = float('-inf') if 'accuracy' in metric or 'f1' in metric else float('inf')
        is_higher_better = 'accuracy' in metric or 'f1' in metric or 'r2' in metric
        
        for model_name, model_data in self.trained_models.items():
            test_metrics = model_data['test_metrics']
            if metric in test_metrics:
                score = test_metrics[metric]
                
                if is_higher_better:
                    if score > best_score:
                        best_score = score
                        best_model_name = model_name
                else:
                    if score < best_score:
                        best_score = score
                        best_model_name = model_name
                        
        if best_model_name is None:
            raise ValueError(f"Metric '{metric}' not found in any model.")
            
        return best_model_name, self.trained_models[best_model_name]['trainer'] 