"""
Data Preprocessing Pipeline

Main pipeline that orchestrates the complete data preprocessing workflow:
- Data cleaning and missing value handling
- Data normalization and encoding
- Feature engineering
- Data validation and quality checks
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging
from datetime import datetime
import json
import os

from .data_cleaner import DataCleaner
from .normalizer import DataNormalizer
from .feature_engineer import FeatureEngineer

logger = logging.getLogger(__name__)


class PreprocessingPipeline:
    """
    Complete data preprocessing pipeline for survey data.
    """
    
    def __init__(self,
                 # Data cleaning parameters
                 missing_threshold: float = 0.3,
                 outlier_method: str = 'iqr',
                 outlier_factor: float = 1.5,
                 
                 # Normalization parameters
                 numerical_method: str = 'standard',
                 categorical_method: str = 'onehot',
                 
                 # Feature engineering parameters
                 create_interactions: bool = True,
                 create_polynomials: bool = False,
                 polynomial_degree: int = 2,
                 create_aggregations: bool = True,
                 create_domain_features: bool = True,
                 
                 # General parameters
                 save_artifacts: bool = True,
                 artifacts_dir: str = 'preprocessing_artifacts'):
        """
        Initialize the preprocessing pipeline.
        
        Args:
            missing_threshold: Threshold for dropping columns with too many missing values
            outlier_method: Method for outlier detection ('iqr', 'zscore')
            outlier_factor: Factor for outlier detection
            numerical_method: Method for numerical normalization
            categorical_method: Method for categorical encoding
            create_interactions: Whether to create interaction features
            create_polynomials: Whether to create polynomial features
            polynomial_degree: Degree for polynomial features
            create_aggregations: Whether to create aggregation features
            create_domain_features: Whether to create domain-specific features
            save_artifacts: Whether to save preprocessing artifacts
            artifacts_dir: Directory to save artifacts
        """
        # Initialize components
        self.data_cleaner = DataCleaner(
            missing_threshold=missing_threshold,
            outlier_method=outlier_method,
            outlier_factor=outlier_factor
        )
        
        self.normalizer = DataNormalizer(
            numerical_method=numerical_method,
            categorical_method=categorical_method,
            save_scalers=save_artifacts
        )
        
        self.feature_engineer = FeatureEngineer(
            create_interactions=create_interactions,
            create_polynomials=create_polynomials,
            polynomial_degree=polynomial_degree,
            create_aggregations=create_aggregations,
            create_domain_features=create_domain_features
        )
        
        # Pipeline settings
        self.save_artifacts = save_artifacts
        self.artifacts_dir = artifacts_dir
        self.pipeline_report = {}
        self.is_fitted = False
        
        # Create artifacts directory
        if self.save_artifacts and not os.path.exists(self.artifacts_dir):
            os.makedirs(self.artifacts_dir)
    
    def fit_transform(self, data: pd.DataFrame, target_column: Optional[str] = None) -> pd.DataFrame:
        """
        Fit the pipeline on training data and transform it.
        
        Args:
            data: Training data DataFrame
            target_column: Target column for supervised feature selection
            
        Returns:
            Processed DataFrame
        """
        logger.info("Starting complete preprocessing pipeline (fit_transform)")
        
        start_time = datetime.now()
        
        # Track original data info
        original_shape = data.shape
        self.pipeline_report['original_shape'] = original_shape
        self.pipeline_report['original_columns'] = list(data.columns)
        self.pipeline_report['target_column'] = target_column
        
        # Step 1: Data Quality Assessment
        quality_report = self._assess_data_quality(data)
        self.pipeline_report['data_quality'] = quality_report
        
        # Step 2: Data Cleaning
        logger.info("Step 1/4: Data cleaning")
        cleaned_data = self.data_cleaner.clean_survey_data(data)
        self.pipeline_report['cleaning'] = self.data_cleaner.get_cleaning_report()
        
        # Step 3: Feature Engineering (before normalization to work with original scales)
        logger.info("Step 2/4: Feature engineering")
        engineered_data = self.feature_engineer.engineer_features(cleaned_data, target_column)
        self.pipeline_report['feature_engineering'] = self.feature_engineer.get_feature_engineering_report()
        
        # Step 4: Data Normalization
        logger.info("Step 3/4: Data normalization")
        normalized_data = self.normalizer.normalize_data(engineered_data, fit=True)
        self.pipeline_report['normalization'] = self.normalizer.get_normalization_report()
        
        # Step 5: Final validation
        logger.info("Step 4/4: Final validation")
        validation_report = self._validate_processed_data(normalized_data)
        self.pipeline_report['validation'] = validation_report
        
        # Track final info and timing
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        final_shape = normalized_data.shape
        self.pipeline_report['final_shape'] = final_shape
        self.pipeline_report['final_columns'] = list(normalized_data.columns)
        self.pipeline_report['processing_time_seconds'] = processing_time
        self.pipeline_report['timestamp'] = end_time.isoformat()
        
        # Mark as fitted
        self.is_fitted = True
        
        # Save artifacts
        if self.save_artifacts:
            self._save_artifacts()
        
        logger.info(f"Pipeline completed. Shape: {original_shape} -> {final_shape}")
        logger.info(f"Processing time: {processing_time:.2f} seconds")
        
        return normalized_data
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Transform new data using fitted pipeline.
        
        Args:
            data: New data DataFrame
            
        Returns:
            Processed DataFrame
        """
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted before transforming new data. Use fit_transform() first.")
        
        logger.info("Transforming new data using fitted pipeline")
        
        # Step 1: Data Cleaning (using same parameters, no fitting)
        cleaned_data = self.data_cleaner.clean_survey_data(data)
        
        # Step 2: Feature Engineering (create same features, no fitting)
        # Note: For new data, we don't do feature selection as that requires a target
        engineered_data = self._transform_features(cleaned_data)
        
        # Step 3: Data Normalization (using fitted scalers)
        normalized_data = self.normalizer.normalize_data(engineered_data, fit=False)
        
        logger.info(f"New data transformed. Shape: {data.shape} -> {normalized_data.shape}")
        
        return normalized_data
    
    def _transform_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transform features for new data without refitting."""
        # Create feature engineer with same settings but without feature selection
        temp_engineer = FeatureEngineer(
            create_interactions=self.feature_engineer.create_interactions,
            create_polynomials=self.feature_engineer.create_polynomials,
            polynomial_degree=self.feature_engineer.polynomial_degree,
            create_aggregations=self.feature_engineer.create_aggregations,
            create_domain_features=self.feature_engineer.create_domain_features
        )
        
        # Apply feature engineering without feature selection
        engineered_data = temp_engineer.engineer_features(data, target_column=None)
        
        return engineered_data
    
    def _assess_data_quality(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Assess data quality before processing."""
        logger.info("Assessing data quality")
        
        quality_report = {}
        
        # Basic statistics
        quality_report['shape'] = data.shape
        quality_report['memory_usage_mb'] = data.memory_usage(deep=True).sum() / 1024 / 1024
        
        # Missing values analysis
        missing_info = {}
        total_missing = data.isnull().sum().sum()
        missing_percentage = (total_missing / (data.shape[0] * data.shape[1])) * 100
        
        missing_info['total_missing_values'] = int(total_missing)
        missing_info['overall_missing_percentage'] = float(missing_percentage)
        missing_info['columns_with_missing'] = int((data.isnull().sum() > 0).sum())
        missing_info['completely_missing_columns'] = int((data.isnull().sum() == len(data)).sum())
        
        quality_report['missing_values'] = missing_info
        
        # Data types analysis
        dtype_info = {}
        dtype_counts = data.dtypes.value_counts()
        for dtype, count in dtype_counts.items():
            dtype_info[str(dtype)] = int(count)
        
        quality_report['data_types'] = dtype_info
        
        # Numerical columns analysis
        numerical_columns = data.select_dtypes(include=[np.number]).columns
        if len(numerical_columns) > 0:
            numerical_info = {}
            numerical_info['count'] = len(numerical_columns)
            numerical_info['columns'] = list(numerical_columns)
            
            # Check for potential issues
            inf_columns = []
            zero_var_columns = []
            
            for col in numerical_columns:
                if np.isinf(data[col]).any():
                    inf_columns.append(col)
                if data[col].var() == 0:
                    zero_var_columns.append(col)
            
            numerical_info['columns_with_inf'] = inf_columns
            numerical_info['zero_variance_columns'] = zero_var_columns
            
            quality_report['numerical_analysis'] = numerical_info
        
        # Categorical columns analysis
        categorical_columns = data.select_dtypes(include=['object', 'category']).columns
        if len(categorical_columns) > 0:
            categorical_info = {}
            categorical_info['count'] = len(categorical_columns)
            categorical_info['columns'] = list(categorical_columns)
            
            # Cardinality analysis
            high_cardinality_cols = []
            for col in categorical_columns:
                unique_count = data[col].nunique()
                if unique_count > 0.5 * len(data):  # High cardinality threshold
                    high_cardinality_cols.append({'column': col, 'unique_count': unique_count})
            
            categorical_info['high_cardinality_columns'] = high_cardinality_cols
            
            quality_report['categorical_analysis'] = categorical_info
        
        # Duplicate analysis
        duplicate_info = {}
        exact_duplicates = data.duplicated().sum()
        duplicate_info['exact_duplicates'] = int(exact_duplicates)
        duplicate_info['duplicate_percentage'] = float((exact_duplicates / len(data)) * 100)
        
        quality_report['duplicates'] = duplicate_info
        
        return quality_report
    
    def _validate_processed_data(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Validate the processed data."""
        logger.info("Validating processed data")
        
        validation_report = {}
        
        # Check for remaining issues
        issues = []
        
        # Check for missing values
        remaining_missing = data.isnull().sum().sum()
        if remaining_missing > 0:
            issues.append(f"Still has {remaining_missing} missing values")
        
        # Check for infinite values
        if data.select_dtypes(include=[np.number]).apply(lambda x: np.isinf(x).any()).any():
            issues.append("Contains infinite values")
        
        # Check for very high cardinality after encoding
        high_dim_warning = False
        if data.shape[1] > 1000:
            high_dim_warning = True
            issues.append(f"High dimensionality: {data.shape[1]} features")
        
        # Check data types
        type_info = {}
        for dtype in data.dtypes.unique():
            count = (data.dtypes == dtype).sum()
            type_info[str(dtype)] = int(count)
        
        validation_report['data_types'] = type_info
        validation_report['issues'] = issues
        validation_report['high_dimensionality_warning'] = high_dim_warning
        validation_report['is_valid'] = len(issues) == 0
        
        if validation_report['is_valid']:
            logger.info("Data validation passed")
        else:
            logger.warning(f"Data validation found issues: {issues}")
        
        return validation_report
    
    def _save_artifacts(self):
        """Save preprocessing artifacts."""
        logger.info("Saving preprocessing artifacts")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save scalers and encoders
        scalers_path = os.path.join(self.artifacts_dir, f"scalers_{timestamp}.joblib")
        self.normalizer.save_scalers(scalers_path)
        
        # Save pipeline report
        report_path = os.path.join(self.artifacts_dir, f"pipeline_report_{timestamp}.json")
        self.save_pipeline_report(report_path)
        
        # Save individual component reports
        cleaning_report_path = os.path.join(self.artifacts_dir, f"cleaning_report_{timestamp}.json")
        self.data_cleaner.save_cleaning_report(cleaning_report_path)
        
        normalization_report_path = os.path.join(self.artifacts_dir, f"normalization_report_{timestamp}.json")
        self.normalizer.save_normalization_report(normalization_report_path)
        
        feature_report_path = os.path.join(self.artifacts_dir, f"feature_engineering_report_{timestamp}.json")
        self.feature_engineer.save_feature_engineering_report(feature_report_path)
        
        logger.info(f"Artifacts saved to {self.artifacts_dir}")
    
    def load_artifacts(self, artifacts_dir: Optional[str] = None, timestamp: Optional[str] = None):
        """Load preprocessing artifacts."""
        if artifacts_dir is None:
            artifacts_dir = self.artifacts_dir
        
        if timestamp is None:
            # Find the most recent artifacts
            files = os.listdir(artifacts_dir)
            scaler_files = [f for f in files if f.startswith("scalers_") and f.endswith(".joblib")]
            if not scaler_files:
                raise FileNotFoundError("No scaler artifacts found")
            
            # Get the most recent
            scaler_files.sort(reverse=True)
            latest_file = scaler_files[0]
            timestamp = latest_file.replace("scalers_", "").replace(".joblib", "")
        
        # Load scalers
        scalers_path = os.path.join(artifacts_dir, f"scalers_{timestamp}.joblib")
        self.normalizer.load_scalers(scalers_path)
        
        # Mark as fitted
        self.is_fitted = True
        
        logger.info(f"Artifacts loaded from {artifacts_dir} (timestamp: {timestamp})")
    
    def get_pipeline_report(self) -> Dict[str, Any]:
        """Get the complete pipeline report."""
        return self.pipeline_report
    
    def save_pipeline_report(self, filepath: str):
        """Save pipeline report to a file."""
        # Convert numpy types to native Python types for JSON serialization
        def convert_numpy_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (list, tuple)):
                return [convert_numpy_types(item) for item in obj]
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            return obj
        
        # Create a JSON-serializable version of the report
        serializable_report = convert_numpy_types(self.pipeline_report)
        
        with open(filepath, 'w') as f:
            json.dump(serializable_report, f, indent=2)
        
        logger.info(f"Pipeline report saved to {filepath}")
    
    def get_summary(self) -> str:
        """Get a human-readable summary of the preprocessing pipeline."""
        if not self.pipeline_report:
            return "Pipeline has not been run yet."
        
        summary_lines = []
        summary_lines.append("=== Data Preprocessing Pipeline Summary ===")
        summary_lines.append("")
        
        # Original data info
        if 'original_shape' in self.pipeline_report:
            orig_shape = self.pipeline_report['original_shape']
            summary_lines.append(f"Original data shape: {orig_shape[0]} rows × {orig_shape[1]} columns")
        
        # Final data info
        if 'final_shape' in self.pipeline_report:
            final_shape = self.pipeline_report['final_shape']
            summary_lines.append(f"Final data shape: {final_shape[0]} rows × {final_shape[1]} columns")
        
        # Processing time
        if 'processing_time_seconds' in self.pipeline_report:
            time_sec = self.pipeline_report['processing_time_seconds']
            summary_lines.append(f"Processing time: {time_sec:.2f} seconds")
        
        summary_lines.append("")
        
        # Data quality issues
        if 'data_quality' in self.pipeline_report:
            quality = self.pipeline_report['data_quality']
            if 'missing_values' in quality:
                missing = quality['missing_values']
                summary_lines.append(f"Missing values: {missing['total_missing_values']} ({missing['overall_missing_percentage']:.1f}%)")
        
        # Cleaning results
        if 'cleaning' in self.pipeline_report:
            cleaning = self.pipeline_report['cleaning']
            if 'duplicates_removed' in cleaning:
                summary_lines.append(f"Duplicates removed: {cleaning['duplicates_removed']}")
        
        # Feature engineering results
        if 'feature_engineering' in self.pipeline_report:
            fe = self.pipeline_report['feature_engineering']
            if 'created_features' in fe:
                created_count = len(fe['created_features'])
                summary_lines.append(f"New features created: {created_count}")
        
        # Validation
        if 'validation' in self.pipeline_report:
            validation = self.pipeline_report['validation']
            if validation.get('is_valid', False):
                summary_lines.append("✓ Data validation passed")
            else:
                issues = validation.get('issues', [])
                summary_lines.append(f"⚠ Validation issues: {len(issues)}")
                for issue in issues:
                    summary_lines.append(f"  - {issue}")
        
        return "\n".join(summary_lines) 