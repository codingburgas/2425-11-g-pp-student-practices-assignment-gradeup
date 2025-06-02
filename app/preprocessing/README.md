# Data Preprocessing Module

This module provides comprehensive data preprocessing functionality for survey data, including data cleaning, normalization, and feature engineering.

## Overview

The data preprocessing pipeline is designed to handle survey data from the application database, clean and transform it for machine learning and analysis purposes.

### Key Features

- **Data Cleaning**: Missing value handling, outlier detection, duplicate removal
- **Data Normalization**: Multiple scaling methods, categorical encoding
- **Feature Engineering**: Interaction features, polynomial features, domain-specific features
- **Quality Assessment**: Data validation and quality reporting
- **Artifact Management**: Save and load preprocessing artifacts for consistency

## Module Structure

```
app/preprocessing/
├── __init__.py          # Module initialization
├── data_cleaner.py      # Data cleaning operations
├── normalizer.py        # Data normalization and encoding
├── feature_engineer.py  # Feature engineering operations
├── pipeline.py          # Main preprocessing pipeline
├── utils.py            # Utility functions for data loading
├── example_usage.py    # Usage examples
└── README.md           # This documentation
```

## Quick Start

### Basic Usage

```python
from app.preprocessing import PreprocessingPipeline
from app.preprocessing.utils import load_survey_responses_to_dataframe

# Load survey data
df = load_survey_responses_to_dataframe()

# Initialize pipeline
pipeline = PreprocessingPipeline()

# Run preprocessing
processed_df = pipeline.fit_transform(df, target_column='recommendation_score')

# Get summary
print(pipeline.get_summary())
```

### Web Interface

Access the data preprocessing dashboard at `/data-preprocessing` (admin only).

The web interface provides:
- Data quality assessment
- Configurable preprocessing parameters
- Real-time processing status
- Export functionality

## Core Components

### 1. DataCleaner

Handles data cleaning operations:

```python
from app.preprocessing.data_cleaner import DataCleaner

cleaner = DataCleaner(
    missing_threshold=0.3,    # Drop columns with >30% missing
    outlier_method='iqr',     # Use IQR for outlier detection
    outlier_factor=1.5        # IQR multiplier
)

cleaned_data = cleaner.clean_survey_data(raw_data)
```

**Features:**
- Missing value imputation (mode, median, forward fill)
- Outlier detection and treatment (IQR, Z-score)
- Duplicate removal
- Data type validation and conversion
- Text response cleaning

### 2. DataNormalizer

Handles data normalization and encoding:

```python
from app.preprocessing.normalizer import DataNormalizer

normalizer = DataNormalizer(
    numerical_method='standard',    # StandardScaler
    categorical_method='onehot',    # One-hot encoding
    save_scalers=True              # Save for later use
)

normalized_data = normalizer.normalize_data(data, fit=True)
```

**Features:**
- Multiple scaling methods: standard, minmax, robust, unit
- Categorical encoding: one-hot, label, target
- Boolean handling
- Scaler persistence for new data

### 3. FeatureEngineer

Creates new features from existing data:

```python
from app.preprocessing.feature_engineer import FeatureEngineer

engineer = FeatureEngineer(
    create_interactions=True,      # Feature interactions
    create_polynomials=False,      # Polynomial features
    create_aggregations=True,      # Group-based features
    create_domain_features=True    # Survey-specific features
)

engineered_data = engineer.engineer_features(data, target_column='score')
```

**Features:**
- Interaction features (multiplication, division)
- Polynomial features (degree 2+)
- Aggregation features (group statistics)
- Domain-specific survey features
- Statistical features (mean, std, skew, etc.)
- Feature selection based on importance

### 4. PreprocessingPipeline

Orchestrates the complete preprocessing workflow:

```python
from app.preprocessing.pipeline import PreprocessingPipeline

pipeline = PreprocessingPipeline(
    # Data cleaning
    missing_threshold=0.3,
    outlier_method='iqr',
    
    # Normalization  
    numerical_method='standard',
    categorical_method='onehot',
    
    # Feature engineering
    create_interactions=True,
    create_domain_features=True,
    
    # Artifacts
    save_artifacts=True,
    artifacts_dir='preprocessing_artifacts'
)

# Fit on training data
processed_data = pipeline.fit_transform(training_data, target_column='target')

# Transform new data
new_processed_data = pipeline.transform(new_data)
```

## Configuration Options

### Data Cleaning Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `missing_threshold` | 0.3 | Drop columns with missing % above this |
| `outlier_method` | 'iqr' | Method for outlier detection ('iqr', 'zscore') |
| `outlier_factor` | 1.5 | Factor for outlier bounds |

### Normalization Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `numerical_method` | 'standard' | Scaling method ('standard', 'minmax', 'robust', 'unit') |
| `categorical_method` | 'onehot' | Encoding method ('onehot', 'label', 'target') |
| `save_scalers` | True | Whether to save fitted scalers |

### Feature Engineering Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `create_interactions` | True | Create feature interactions |
| `create_polynomials` | False | Create polynomial features |
| `polynomial_degree` | 2 | Degree for polynomial features |
| `create_aggregations` | True | Create group-based features |
| `create_domain_features` | True | Create survey-specific features |

## Domain-Specific Features

The pipeline creates specialized features for survey data:

### Rating/Satisfaction Features
- `overall_satisfaction_score`: Average of all rating questions
- `rating_consistency`: Standard deviation across ratings
- `extreme_responses_count`: Count of very low/high ratings
- `rating_range`: Difference between max and min ratings

### Response Quality Features
- `response_completeness`: Percentage of answered questions
- `response_time_minutes`: Time taken to complete survey (if available)

### User Profile Features
- Preference-based features from user profiles
- Demographic aggregations
- Historical response patterns

## API Endpoints

### `/api/preprocess-data` (POST)

Run the preprocessing pipeline on survey data.

**Request:**
```json
{
  "survey_id": 1,
  "config": {
    "missing_threshold": 0.3,
    "numerical_method": "standard",
    "create_interactions": true
  }
}
```

**Response:**
```json
{
  "success": true,
  "report": {
    "original_shape": [1000, 25],
    "final_shape": [980, 67],
    "processing_time": 2.45,
    "features_created": 42,
    "data_quality": true
  },
  "export_filename": "processed_survey_data_20241209_143022.csv",
  "summary": "=== Data Preprocessing Pipeline Summary ===\n..."
}
```

### `/api/preprocessing-sample` (POST)

Create sample data for testing.

**Request:**
```json
{
  "num_responses": 100,
  "num_questions": 10
}
```

### `/api/preprocessing-status/<survey_id>` (GET)

Get preprocessing status for a specific survey.

## Utility Functions

### Data Loading

```python
from app.preprocessing.utils import (
    load_survey_responses_to_dataframe,
    load_survey_data_with_recommendations,
    prepare_survey_data_for_ml
)

# Load basic survey responses
df = load_survey_responses_to_dataframe(survey_id=1)

# Load with recommendation data
df_with_recs = load_survey_data_with_recommendations(survey_id=1)

# Prepare for ML
ml_data, target_col = prepare_survey_data_for_ml(survey_id=1)
```

### Data Export

```python
from app.preprocessing.utils import export_processed_data

# Export to different formats
export_processed_data(df, 'output.csv', format='csv')
export_processed_data(df, 'output.xlsx', format='excel')
export_processed_data(df, 'output.json', format='json')
export_processed_data(df, 'output.parquet', format='parquet')
```

### Data Validation

```python
from app.preprocessing.utils import validate_dataframe_for_preprocessing

validation_report = validate_dataframe_for_preprocessing(df)
print(f"Valid: {validation_report['is_valid']}")
print(f"Warnings: {validation_report['warnings']}")
```

## Error Handling

The pipeline includes comprehensive error handling:

- **Data Loading Errors**: Graceful handling of empty datasets
- **Processing Errors**: Detailed error messages and partial results
- **Validation Errors**: Clear feedback on data quality issues
- **Export Errors**: File system and format validation

## Performance Considerations

### Memory Usage
- Large datasets are processed in chunks where possible
- Unnecessary data is dropped early in the pipeline
- Memory-efficient pandas operations are used

### Processing Time
- Feature creation is optimized for large datasets
- Parallel processing for independent operations
- Progress tracking for long-running operations

### Scalability
- Pipeline can handle datasets with thousands of rows
- Feature selection prevents dimensionality explosion
- Configurable thresholds for different dataset sizes

## Examples

### Basic Preprocessing

```python
# Run the basic example
from app.preprocessing.example_usage import run_basic_preprocessing_example

pipeline, processed_data = run_basic_preprocessing_example()
```

### Advanced Preprocessing

```python
# Run the advanced example
from app.preprocessing.example_usage import run_advanced_preprocessing_example

pipeline, processed_data = run_advanced_preprocessing_example()
```

### Method Comparison

```python
# Compare different preprocessing methods
from app.preprocessing.example_usage import compare_preprocessing_methods

compare_preprocessing_methods()
```

## Dependencies

Required packages (automatically added to requirements.txt):
- `pandas>=2.2.0`: Data manipulation
- `numpy>=1.26.3`: Numerical operations
- `scikit-learn>=1.4.0`: Machine learning preprocessing
- `joblib`: Model persistence

## Best Practices

1. **Always validate data** before preprocessing
2. **Save preprocessing artifacts** for production consistency
3. **Use appropriate scaling methods** based on data distribution
4. **Monitor feature explosion** when creating interactions
5. **Test preprocessing pipeline** on sample data first
6. **Document preprocessing decisions** for reproducibility

## Troubleshooting

### Common Issues

1. **"No survey data found"**
   - Check database connectivity
   - Verify survey_id exists
   - Ensure user has submitted responses

2. **"High dimensionality warning"**
   - Reduce interaction features
   - Increase feature selection threshold
   - Use label encoding instead of one-hot

3. **"Memory error during processing"**
   - Process data in smaller batches
   - Reduce feature engineering complexity
   - Increase available memory

4. **"Artifacts not found"**
   - Check artifacts directory permissions
   - Verify artifact files weren't deleted
   - Re-run fit_transform to regenerate

### Debug Mode

Enable debug logging for detailed information:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Contributing

When extending the preprocessing functionality:

1. Add comprehensive docstrings
2. Include error handling
3. Add unit tests
4. Update this documentation
5. Follow existing code patterns

## License

This module is part of the main application and follows the same license terms. 