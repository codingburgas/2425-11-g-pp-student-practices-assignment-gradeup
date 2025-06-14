# Performance tracking and visualization for model training

import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
import time
import json
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime

from .base import TrainingResult
from .cross_validator import CrossValidationResult
from .hyperparameter_tuner import HyperparameterResult


@dataclass
class ExperimentRun:
    """Container for a single experiment run."""
    
    run_id: str
    timestamp: str
    model_name: str
    parameters: Dict[str, Any]
    metrics: Dict[str, float]
    training_time: float
    dataset_info: Dict[str, Any]
    notes: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExperimentRun':
        """Create from dictionary."""
        return cls(**data)


class PerformanceTracker:
    """Comprehensive performance tracking and analysis system."""
    
    def __init__(self, 
                 experiment_name: str = "ml_experiment",
                 log_dir: str = "experiments",
                 auto_save: bool = True):
        """
        Initialize performance tracker.
        
        Args:
            experiment_name: Name of the experiment
            log_dir: Directory to save experiment logs
            auto_save: Whether to automatically save results
        """
        self.experiment_name = experiment_name
        self.log_dir = Path(log_dir)
        self.auto_save = auto_save
        
        # Create log directory
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Experiment tracking
        self.experiment_runs: List[ExperimentRun] = []
        self.current_run_id = None
        
        # Metrics tracking
        self.training_histories: Dict[str, Dict[str, List[float]]] = {}
        self.model_comparisons: Dict[str, Dict[str, Any]] = {}
        
        # Load existing experiments
        self._load_existing_experiments()
        
    def _load_existing_experiments(self):
        """Load existing experiment data if available."""
        experiment_file = self.log_dir / f"{self.experiment_name}_experiments.json"
        
        if experiment_file.exists():
            try:
                with open(experiment_file, 'r') as f:
                    data = json.load(f)
                    self.experiment_runs = [ExperimentRun.from_dict(run) for run in data]
            except Exception as e:
                print(f"Warning: Could not load existing experiments: {e}")
                
    def _save_experiments(self):
        """Save experiment data to disk."""
        if not self.auto_save:
            return
            
        experiment_file = self.log_dir / f"{self.experiment_name}_experiments.json"
        
        try:
            with open(experiment_file, 'w') as f:
                json.dump([run.to_dict() for run in self.experiment_runs], f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save experiments: {e}")
            
    def start_run(self, 
                  model_name: str,
                  parameters: Dict[str, Any],
                  dataset_info: Optional[Dict[str, Any]] = None,
                  notes: str = "") -> str:
        """
        Start a new experiment run.
        
        Args:
            model_name: Name of the model being trained
            parameters: Model parameters and configuration
            dataset_info: Information about the dataset
            notes: Additional notes about the experiment
            
        Returns:
            Run ID for this experiment
        """
        run_id = f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.current_run_id = run_id
        
        # Initialize tracking for this run
        self.training_histories[run_id] = {}
        
        print(f"Started experiment run: {run_id}")
        return run_id
        
    def log_metrics(self, 
                   metrics: Dict[str, float],
                   step: Optional[int] = None,
                   run_id: Optional[str] = None):
        """
        Log metrics for current or specified run.
        
        Args:
            metrics: Dictionary of metric names and values
            step: Optional step/epoch number
            run_id: Run ID (uses current if not specified)
        """
        if run_id is None:
            run_id = self.current_run_id
            
        if run_id is None:
            raise ValueError("No active run. Call start_run() first.")
            
        if run_id not in self.training_histories:
            self.training_histories[run_id] = {}
            
        # Log each metric
        for metric_name, value in metrics.items():
            if metric_name not in self.training_histories[run_id]:
                self.training_histories[run_id][metric_name] = []
            self.training_histories[run_id][metric_name].append(value)
            
    def log_training_result(self, 
                           training_result: TrainingResult,
                           run_id: Optional[str] = None):
        """
        Log results from a TrainingResult object.
        
        Args:
            training_result: TrainingResult from model training
            run_id: Run ID (uses current if not specified)
        """
        if run_id is None:
            run_id = self.current_run_id
            
        if run_id is None:
            raise ValueError("No active run. Call start_run() first.")
            
        # Log training history
        for metric_name, values in training_result.history.items():
            if run_id not in self.training_histories:
                self.training_histories[run_id] = {}
            self.training_histories[run_id][metric_name] = values
            
        # Log final metrics
        for metric_name, value in training_result.final_metrics.items():
            self.log_metrics({metric_name: value}, run_id=run_id)
            
    def end_run(self,
               final_metrics: Dict[str, float],
               model_name: str,
               parameters: Dict[str, Any],
               training_time: float,
               dataset_info: Optional[Dict[str, Any]] = None,
               notes: str = "",
               run_id: Optional[str] = None):
        """
        End current experiment run and save results.
        
        Args:
            final_metrics: Final evaluation metrics
            model_name: Name of the model
            parameters: Model parameters
            training_time: Total training time
            dataset_info: Dataset information
            notes: Additional notes
            run_id: Run ID (uses current if not specified)
        """
        if run_id is None:
            run_id = self.current_run_id
            
        if run_id is None:
            raise ValueError("No active run.")
            
        # Create experiment run record
        experiment_run = ExperimentRun(
            run_id=run_id,
            timestamp=datetime.now().isoformat(),
            model_name=model_name,
            parameters=parameters,
            metrics=final_metrics,
            training_time=training_time,
            dataset_info=dataset_info or {},
            notes=notes
        )
        
        self.experiment_runs.append(experiment_run)
        
        # Save to disk
        self._save_experiments()
        
        print(f"Completed experiment run: {run_id}")
        print(f"Final metrics: {final_metrics}")
        
        # Reset current run
        if run_id == self.current_run_id:
            self.current_run_id = None
            
    def compare_models(self, 
                      run_ids: List[str],
                      metric_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Compare performance across multiple model runs.
        
        Args:
            run_ids: List of run IDs to compare
            metric_names: Specific metrics to compare (all if None)
            
        Returns:
            Dictionary with comparison results
        """
        if not run_ids:
            return {}
            
        # Get experiment runs
        runs = {run.run_id: run for run in self.experiment_runs if run.run_id in run_ids}
        
        if not runs:
            return {'error': 'No matching runs found'}
            
        # Collect metrics
        if metric_names is None:
            # Get all metrics from all runs
            all_metrics = set()
            for run in runs.values():
                all_metrics.update(run.metrics.keys())
            metric_names = list(all_metrics)
            
        comparison = {
            'runs': len(run_ids),
            'metrics': {},
            'rankings': {},
            'summary': {}
        }
        
        # Compare each metric
        for metric_name in metric_names:
            metric_values = {}
            for run_id, run in runs.items():
                if metric_name in run.metrics:
                    metric_values[run_id] = run.metrics[metric_name]
                    
            if metric_values:
                # Sort by metric value (assume higher is better for most metrics)
                is_loss_metric = any(term in metric_name.lower() for term in ['loss', 'error', 'mse', 'rmse'])
                sorted_runs = sorted(metric_values.items(), 
                                   key=lambda x: x[1], 
                                   reverse=not is_loss_metric)
                
                comparison['metrics'][metric_name] = metric_values
                comparison['rankings'][metric_name] = [run_id for run_id, _ in sorted_runs]
                
                # Statistics
                values = list(metric_values.values())
                comparison['summary'][metric_name] = {
                    'best_run': sorted_runs[0][0],
                    'best_value': sorted_runs[0][1],
                    'worst_run': sorted_runs[-1][0],
                    'worst_value': sorted_runs[-1][1],
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'range': float(np.max(values) - np.min(values))
                }
                
        return comparison
        
    def get_training_curve(self, 
                          run_id: str,
                          metric_name: str) -> Tuple[List[int], List[float]]:
        """
        Get training curve data for plotting.
        
        Args:
            run_id: Run ID
            metric_name: Name of metric to plot
            
        Returns:
            Tuple of (epochs, values)
        """
        if run_id not in self.training_histories:
            return [], []
            
        if metric_name not in self.training_histories[run_id]:
            return [], []
            
        values = self.training_histories[run_id][metric_name]
        epochs = list(range(1, len(values) + 1))
        
        return epochs, values
        
    def plot_training_curve_ascii(self, 
                                 run_id: str,
                                 metric_name: str,
                                 width: int = 80,
                                 height: int = 20) -> str:
        """
        Create ASCII plot of training curve.
        
        Args:
            run_id: Run ID
            metric_name: Metric to plot
            width: Plot width in characters
            height: Plot height in characters
            
        Returns:
            ASCII plot as string
        """
        epochs, values = self.get_training_curve(run_id, metric_name)
        
        if not values:
            return f"No data for {metric_name} in run {run_id}"
            
        # Normalize values to plot height
        min_val, max_val = min(values), max(values)
        if max_val == min_val:
            normalized = [height // 2] * len(values)
        else:
            normalized = [(v - min_val) / (max_val - min_val) * (height - 1) for v in values]
            
        # Create plot grid
        plot = [[' ' for _ in range(width)] for _ in range(height)]
        
        # Plot points
        for i, norm_val in enumerate(normalized):
            x = int(i * (width - 1) / (len(values) - 1)) if len(values) > 1 else width // 2
            y = height - 1 - int(norm_val)
            if 0 <= x < width and 0 <= y < height:
                plot[y][x] = '*'
                
        # Convert to string
        result = [f"{metric_name} Training Curve - Run: {run_id}"]
        result.append("+" + "-" * (width - 2) + "+")
        
        for i, row in enumerate(plot):
            # Add y-axis labels
            if i == 0:
                label = f"{max_val:.3f}"
            elif i == height - 1:
                label = f"{min_val:.3f}"
            else:
                label = " " * 7
                
            result.append(f"|{''.join(row)}| {label}")
            
        result.append("+" + "-" * (width - 2) + "+")
        result.append(f"Epochs: {len(values)}, Min: {min_val:.4f}, Max: {max_val:.4f}")
        
        return "\n".join(result)
        
    def compare_training_curves_ascii(self, 
                                    run_ids: List[str],
                                    metric_name: str,
                                    width: int = 80,
                                    height: int = 20) -> str:
        """
        Create ASCII plot comparing training curves from multiple runs.
        
        Args:
            run_ids: List of run IDs to compare
            metric_name: Metric to plot
            width: Plot width in characters
            height: Plot height in characters
            
        Returns:
            ASCII comparison plot as string
        """
        if len(run_ids) > 5:
            return "Too many runs to compare (max 5)"
            
        symbols = ['*', '+', 'o', 'x', '#'][:len(run_ids)]
        
        # Get all curves
        all_curves = {}
        max_epochs = 0
        all_values = []
        
        for run_id in run_ids:
            epochs, values = self.get_training_curve(run_id, metric_name)
            if values:
                all_curves[run_id] = values
                max_epochs = max(max_epochs, len(values))
                all_values.extend(values)
                
        if not all_values:
            return f"No data for {metric_name} in any run"
            
        # Normalize all values
        min_val, max_val = min(all_values), max(all_values)
        
        # Create plot grid
        plot = [[' ' for _ in range(width)] for _ in range(height)]
        
        # Plot each curve
        for i, (run_id, values) in enumerate(all_curves.items()):
            symbol = symbols[i]
            
            if max_val == min_val:
                normalized = [height // 2] * len(values)
            else:
                normalized = [(v - min_val) / (max_val - min_val) * (height - 1) for v in values]
                
            for j, norm_val in enumerate(normalized):
                x = int(j * (width - 1) / max(max_epochs - 1, 1))
                y = height - 1 - int(norm_val)
                if 0 <= x < width and 0 <= y < height:
                    plot[y][x] = symbol
                    
        # Convert to string
        result = [f"{metric_name} Training Curves Comparison"]
        result.append("+" + "-" * (width - 2) + "+")
        
        for i, row in enumerate(plot):
            # Add y-axis labels
            if i == 0:
                label = f"{max_val:.3f}"
            elif i == height - 1:
                label = f"{min_val:.3f}"
            else:
                label = " " * 7
                
            result.append(f"|{''.join(row)}| {label}")
            
        result.append("+" + "-" * (width - 2) + "+")
        
        # Legend
        result.append("Legend:")
        for i, run_id in enumerate(all_curves.keys()):
            result.append(f"  {symbols[i]}: {run_id}")
            
        return "\n".join(result)
        
    def get_experiment_summary(self) -> Dict[str, Any]:
        """Get summary of all experiments."""
        if not self.experiment_runs:
            return {'total_runs': 0}
            
        # Collect statistics
        models = [run.model_name for run in self.experiment_runs]
        training_times = [run.training_time for run in self.experiment_runs]
        
        # Get all unique metrics
        all_metrics = set()
        for run in self.experiment_runs:
            all_metrics.update(run.metrics.keys())
            
        # Metrics statistics
        metrics_stats = {}
        for metric in all_metrics:
            values = [run.metrics[metric] for run in self.experiment_runs if metric in run.metrics]
            if values:
                metrics_stats[metric] = {
                    'count': len(values),
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values))
                }
                
        return {
            'total_runs': len(self.experiment_runs),
            'unique_models': len(set(models)),
            'model_counts': {model: models.count(model) for model in set(models)},
            'total_training_time': sum(training_times),
            'avg_training_time': np.mean(training_times) if training_times else 0,
            'metrics_statistics': metrics_stats,
            'experiment_name': self.experiment_name
        }
        
    def export_results(self, 
                      format: str = 'json',
                      filename: Optional[str] = None) -> str:
        """
        Export experiment results to file.
        
        Args:
            format: Export format ('json', 'csv')
            filename: Output filename (auto-generated if None)
            
        Returns:
            Path to exported file
        """
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{self.experiment_name}_results_{timestamp}.{format}"
            
        filepath = self.log_dir / filename
        
        if format == 'json':
            export_data = {
                'experiment_name': self.experiment_name,
                'export_timestamp': datetime.now().isoformat(),
                'summary': self.get_experiment_summary(),
                'runs': [run.to_dict() for run in self.experiment_runs],
                'training_histories': self.training_histories
            }
            
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2)
                
        elif format == 'csv':
            # Simple CSV export of run results
            import csv
            
            with open(filepath, 'w', newline='') as f:
                if self.experiment_runs:
                    # Get all metric names
                    all_metrics = set()
                    for run in self.experiment_runs:
                        all_metrics.update(run.metrics.keys())
                        
                    fieldnames = ['run_id', 'timestamp', 'model_name', 'training_time'] + list(all_metrics)
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    
                    for run in self.experiment_runs:
                        row = {
                            'run_id': run.run_id,
                            'timestamp': run.timestamp,
                            'model_name': run.model_name,
                            'training_time': run.training_time
                        }
                        row.update(run.metrics)
                        writer.writerow(row)
        else:
            raise ValueError(f"Unsupported format: {format}")
            
        print(f"Results exported to: {filepath}")
        return str(filepath)
        
    def clear_experiments(self):
        """Clear all experiment data."""
        self.experiment_runs.clear()
        self.training_histories.clear()
        self.model_comparisons.clear()
        self.current_run_id = None
        self._save_experiments()
        print("All experiment data cleared") 