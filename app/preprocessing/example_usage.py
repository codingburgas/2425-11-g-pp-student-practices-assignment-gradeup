"""
Example Usage of Data Preprocessing Pipeline

This script demonstrates how to use the preprocessing pipeline
with sample survey data.
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime

from .pipeline import PreprocessingPipeline
from .utils import create_sample_survey_data, validate_dataframe_for_preprocessing, get_column_info, export_processed_data


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_basic_preprocessing_example():
    """
    Run a basic example of the preprocessing pipeline.
    """
    print("=" * 60)
    print("DATA PREPROCESSING PIPELINE EXAMPLE")
    print("=" * 60)
    
    
    print("\n1. Creating sample survey data...")
    sample_data = create_sample_survey_data(num_responses=200, num_questions=15)
    print(f"Sample data shape: {sample_data.shape}")
    print(f"Columns: {list(sample_data.columns)}")
    
    
    print("\n2. Validating data for preprocessing...")
    validation_report = validate_dataframe_for_preprocessing(sample_data)
    print(f"Data is valid: {validation_report['is_valid']}")
    if validation_report['warnings']:
        print(f"Warnings: {validation_report['warnings']}")
    
    
    print("\n3. Initializing preprocessing pipeline...")
    pipeline = PreprocessingPipeline(
        
        missing_threshold=0.3,
        outlier_method='iqr',
        
        
        numerical_method='standard',
        categorical_method='onehot',
        
        
        create_interactions=True,
        create_polynomials=False,
        create_aggregations=True,
        create_domain_features=True,
        
        
        save_artifacts=True,
        artifacts_dir='preprocessing_artifacts'
    )
    
    
    print("\n4. Running preprocessing pipeline...")
    processed_data = pipeline.fit_transform(
        data=sample_data,
        target_column='recommendation_score'
    )
    
    print(f"Processed data shape: {processed_data.shape}")
    print(f"Original shape: {sample_data.shape}")
    
    
    print("\n5. Pipeline Summary:")
    print(pipeline.get_summary())
    
    
    print("\n6. Exporting processed data...")
    export_processed_data(
        df=processed_data,
        filepath='processed_survey_data.csv',
        format='csv'
    )
    
    print("\n✓ Preprocessing pipeline completed successfully!")
    
    return pipeline, processed_data


def run_advanced_preprocessing_example():
    """
    Run an advanced example with custom parameters and analysis.
    """
    print("\n" + "=" * 60)
    print("ADVANCED PREPROCESSING PIPELINE EXAMPLE")
    print("=" * 60)
    
    
    print("\n1. Creating complex sample data...")
    sample_data = create_sample_survey_data(num_responses=500, num_questions=20)
    
    
    sample_data['user_income'] = np.random.lognormal(10, 1, len(sample_data))
    sample_data['user_years_experience'] = np.random.poisson(8, len(sample_data))
    sample_data['response_time_minutes'] = np.random.gamma(2, 10, len(sample_data))
    
    print(f"Complex data shape: {sample_data.shape}")
    
    
    print("\n2. Analyzing column information...")
    column_info = get_column_info(sample_data)
    print("\nColumn Information Summary:")
    print(f"- Total columns: {len(column_info)}")
    print(f"- Columns with missing values: {(column_info['null_count'] > 0).sum()}")
    print(f"- High cardinality columns (>50% unique): {(column_info['unique_percentage'] > 50).sum()}")
    
    
    print("\n3. Configuring advanced preprocessing pipeline...")
    advanced_pipeline = PreprocessingPipeline(
        
        missing_threshold=0.2,
        outlier_method='iqr',
        outlier_factor=1.5,
        
        
        numerical_method='robust',  
        categorical_method='label',  
        
        
        create_interactions=True,
        create_polynomials=True,
        polynomial_degree=2,
        create_aggregations=True,
        create_domain_features=True,
        
        
        save_artifacts=True,
        artifacts_dir='advanced_preprocessing_artifacts'
    )
    
    
    print("\n4. Running advanced preprocessing...")
    start_time = datetime.now()
    
    processed_data = advanced_pipeline.fit_transform(
        data=sample_data,
        target_column='recommendation_score'
    )
    
    end_time = datetime.now()
    processing_time = (end_time - start_time).total_seconds()
    
    print(f"Processing completed in {processing_time:.2f} seconds")
    print(f"Data shape: {sample_data.shape} -> {processed_data.shape}")
    
    
    print("\n5. Analyzing preprocessing results...")
    pipeline_report = advanced_pipeline.get_pipeline_report()
    
    print(f"- Data quality score: {pipeline_report['validation']['is_valid']}")
    print(f"- Features created: {len(pipeline_report.get('feature_engineering', {}).get('created_features', []))}")
    print(f"- Missing values handled: {pipeline_report.get('cleaning', {}).get('missing_values', {})}")
    
    
    print("\n6. Testing transform on new data...")
    new_sample_data = create_sample_survey_data(num_responses=50, num_questions=20)
    new_sample_data['user_income'] = np.random.lognormal(10, 1, len(new_sample_data))
    new_sample_data['user_years_experience'] = np.random.poisson(8, len(new_sample_data))
    new_sample_data['response_time_minutes'] = np.random.gamma(2, 10, len(new_sample_data))
    
    new_processed_data = advanced_pipeline.transform(new_sample_data)
    print(f"New data transformed: {new_sample_data.shape} -> {new_processed_data.shape}")
    
    
    print("\n7. Exporting processed data in multiple formats...")
    
    export_processed_data(processed_data, 'advanced_processed_data.csv', 'csv')
    export_processed_data(processed_data, 'advanced_processed_data.json', 'json')
    
    print("\n✓ Advanced preprocessing pipeline completed successfully!")
    
    return advanced_pipeline, processed_data


def compare_preprocessing_methods():
    """
    Compare different preprocessing methods on the same data.
    """
    print("\n" + "=" * 60)
    print("PREPROCESSING METHODS COMPARISON")
    print("=" * 60)
    
    
    test_data = create_sample_survey_data(num_responses=300, num_questions=12)
    
    
    print("\n1. Standard preprocessing...")
    standard_pipeline = PreprocessingPipeline(
        numerical_method='standard',
        categorical_method='onehot',
        create_interactions=False,
        create_polynomials=False,
        save_artifacts=False
    )
    
    standard_result = standard_pipeline.fit_transform(test_data, target_column='recommendation_score')
    
    
    print("\n2. Robust preprocessing...")
    robust_pipeline = PreprocessingPipeline(
        numerical_method='robust',
        categorical_method='label',
        outlier_method='iqr',
        create_interactions=True,
        create_polynomials=False,
        save_artifacts=False
    )
    
    robust_result = robust_pipeline.fit_transform(test_data, target_column='recommendation_score')
    
    
    print("\n3. Feature-rich preprocessing...")
    feature_rich_pipeline = PreprocessingPipeline(
        numerical_method='minmax',
        categorical_method='onehot',
        create_interactions=True,
        create_polynomials=True,
        polynomial_degree=2,
        create_aggregations=True,
        save_artifacts=False
    )
    
    feature_rich_result = feature_rich_pipeline.fit_transform(test_data, target_column='recommendation_score')
    
    
    print("\n4. Comparison Results:")
    print(f"Original data shape: {test_data.shape}")
    print(f"Standard preprocessing: {standard_result.shape}")
    print(f"Robust preprocessing: {robust_result.shape}")
    print(f"Feature-rich preprocessing: {feature_rich_result.shape}")
    
    
    methods = [
        ("Standard", standard_pipeline),
        ("Robust", robust_pipeline),
        ("Feature-rich", feature_rich_pipeline)
    ]
    
    print("\nProcessing time comparison:")
    for method_name, pipeline in methods:
        report = pipeline.get_pipeline_report()
        time_sec = report.get('processing_time_seconds', 0)
        print(f"- {method_name}: {time_sec:.3f} seconds")
    
    print("\n✓ Preprocessing methods comparison completed!")


if __name__ == "__main__":
    
    try:
        
        basic_pipeline, basic_data = run_basic_preprocessing_example()
        
        
        advanced_pipeline, advanced_data = run_advanced_preprocessing_example()
        
        
        compare_preprocessing_methods()
        
        print("\n" + "=" * 60)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
    except Exception as e:
        logger.error(f"Example failed: {e}")
        raise 