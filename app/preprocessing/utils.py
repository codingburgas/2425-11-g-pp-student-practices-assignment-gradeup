"""
Data Preprocessing Utilities

Utility functions for:
- Loading survey data from database
- Converting between database models and pandas DataFrames
- Data export functions
- Preprocessing helpers
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
from datetime import datetime
import json

from app import db
from app.models import Survey, SurveyResponse, User, School, Program

logger = logging.getLogger(__name__)


def load_survey_responses_to_dataframe(survey_id: Optional[int] = None,
                                     user_id: Optional[int] = None,
                                     include_user_info: bool = True,
                                     include_timestamps: bool = True) -> pd.DataFrame:
    """
    Load survey responses from database into a pandas DataFrame.
    
    Args:
        survey_id: Specific survey ID to load (None for all surveys)
        user_id: Specific user ID to load (None for all users)
        include_user_info: Whether to include user information
        include_timestamps: Whether to include timestamp columns
        
    Returns:
        DataFrame with survey responses
    """
    logger.info(f"Loading survey responses (survey_id={survey_id}, user_id={user_id})")
    
    
    query = db.session.query(SurveyResponse)
    
    if survey_id:
        query = query.filter(SurveyResponse.survey_id == survey_id)
    
    if user_id:
        query = query.filter(SurveyResponse.user_id == user_id)
    
    
    responses = query.all()
    
    if not responses:
        logger.warning("No survey responses found")
        return pd.DataFrame()
    
    
    data_records = []
    
    for response in responses:
        record = {
            'response_id': response.id,
            'user_id': response.user_id,
            'survey_id': response.survey_id
        }
        
        
        if include_timestamps:
            record['created_at'] = response.created_at
        
        
        if include_user_info and response.user:
            record['user_username'] = response.user.username
            record['user_email'] = response.user.email
            record['user_location'] = response.user.location
            record['user_bio'] = response.user.bio
            
            
            user_prefs = response.user.get_preferences()
            for pref_key, pref_value in user_prefs.items():
                record[f'user_pref_{pref_key}'] = pref_value
        
        
        answers = response.get_answers()
        for question_key, answer_value in answers.items():
            
            clean_key = question_key.replace(' ', '_').replace('?', '').replace('(', '').replace(')', '').lower()
            record[f'q_{clean_key}'] = answer_value
        
        data_records.append(record)
    
    df = pd.DataFrame(data_records)
    
    logger.info(f"Loaded {len(df)} survey responses with {len(df.columns)} columns")
    
    return df


def load_survey_data_with_recommendations(survey_id: Optional[int] = None,
                                        include_school_program_info: bool = True) -> pd.DataFrame:
    """
    Load survey data with associated recommendations.
    
    Args:
        survey_id: Specific survey ID to load
        include_school_program_info: Whether to include school and program details
        
    Returns:
        DataFrame with survey responses and recommendations
    """
    logger.info(f"Loading survey data with recommendations (survey_id={survey_id})")
    
    
    df = load_survey_responses_to_dataframe(survey_id=survey_id)
    
    if df.empty:
        return df
    
    
    recommendation_records = []
    
    for _, row in df.iterrows():
        response_id = row['response_id']
        
        
        from app.models import Recommendation
        recommendations = Recommendation.query.filter_by(survey_response_id=response_id).all()
        
        for rec in recommendations:
            rec_record = row.to_dict()  
            
            
            rec_record['recommendation_id'] = rec.id
            rec_record['recommendation_score'] = rec.score
            rec_record['program_id'] = rec.program_id
            
            
            if include_school_program_info and rec.program:
                program = rec.program
                rec_record['program_name'] = program.name
                rec_record['program_description'] = program.description
                rec_record['program_duration'] = program.duration
                rec_record['program_degree_type'] = program.degree_type
                rec_record['program_tuition_fee'] = program.tuition_fee
                
                if program.school:
                    school = program.school
                    rec_record['school_id'] = school.id
                    rec_record['school_name'] = school.name
                    rec_record['school_location'] = school.location
                    rec_record['school_website'] = school.website
            
            recommendation_records.append(rec_record)
    
    if recommendation_records:
        recommendations_df = pd.DataFrame(recommendation_records)
        logger.info(f"Loaded {len(recommendations_df)} recommendation records")
        return recommendations_df
    else:
        logger.info("No recommendations found, returning survey data only")
        return df


def prepare_survey_data_for_ml(survey_id: Optional[int] = None,
                             target_column: str = 'recommendation_score',
                             min_responses_per_user: int = 1) -> Tuple[pd.DataFrame, Optional[str]]:
    """
    Prepare survey data for machine learning.
    
    Args:
        survey_id: Specific survey ID to prepare
        target_column: Target column for supervised learning
        min_responses_per_user: Minimum responses required per user
        
    Returns:
        Tuple of (prepared_dataframe, target_column_name)
    """
    logger.info(f"Preparing survey data for ML (survey_id={survey_id})")
    
    
    df = load_survey_data_with_recommendations(survey_id=survey_id)
    
    if df.empty:
        logger.warning("No data available for ML preparation")
        return pd.DataFrame(), None
    
    
    user_response_counts = df['user_id'].value_counts()
    valid_users = user_response_counts[user_response_counts >= min_responses_per_user].index
    
    df_filtered = df[df['user_id'].isin(valid_users)]
    
    logger.info(f"Filtered data: {len(df_filtered)} records from {len(valid_users)} users")
    
    
    if target_column not in df_filtered.columns:
        logger.warning(f"Target column '{target_column}' not found. Available columns: {df_filtered.columns.tolist()}")
        target_column = None
    
    return df_filtered, target_column


def export_processed_data(df: pd.DataFrame, 
                         filepath: str,
                         format: str = 'csv',
                         include_index: bool = False) -> None:
    """
    Export processed data to various formats.
    
    Args:
        df: DataFrame to export
        filepath: Output file path
        format: Export format ('csv', 'excel', 'json', 'parquet')
        include_index: Whether to include DataFrame index
    """
    logger.info(f"Exporting data to {filepath} (format: {format})")
    
    try:
        if format.lower() == 'csv':
            df.to_csv(filepath, index=include_index)
        elif format.lower() == 'excel':
            df.to_excel(filepath, index=include_index)
        elif format.lower() == 'json':
            df.to_json(filepath, orient='records', indent=2)
        elif format.lower() == 'parquet':
            df.to_parquet(filepath, index=include_index)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Data exported successfully to {filepath}")
        
    except Exception as e:
        logger.error(f"Failed to export data: {e}")
        raise


def create_sample_survey_data(num_responses: int = 100,
                            num_questions: int = 10) -> pd.DataFrame:
    """
    Create sample survey data for testing preprocessing pipeline.
    
    Args:
        num_responses: Number of survey responses to generate
        num_questions: Number of questions in the survey
        
    Returns:
        DataFrame with sample survey data
    """
    logger.info(f"Creating sample survey data ({num_responses} responses, {num_questions} questions)")
    
    np.random.seed(42)  
    
    data_records = []
    
    for i in range(num_responses):
        record = {
            'response_id': i + 1,
            'user_id': np.random.randint(1, min(50, num_responses // 2) + 1),
            'survey_id': 1,
            'created_at': datetime.now()
        }
        
        
        record['user_age'] = np.random.randint(18, 65)
        record['user_location'] = np.random.choice(['New York', 'California', 'Texas', 'Florida', 'Illinois'])
        record['user_education'] = np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'])
        
        
        for q in range(1, num_questions + 1):
            if q <= 3:
                
                record[f'q_rating_{q}'] = np.random.randint(1, 6)
            elif q <= 6:
                
                record[f'q_binary_{q}'] = np.random.choice(['Yes', 'No'])
            elif q <= 8:
                
                record[f'q_choice_{q}'] = np.random.choice(['Option A', 'Option B', 'Option C', 'Option D'])
            else:
                
                record[f'q_satisfaction_{q}'] = np.random.randint(1, 11)
        
        
        missing_fields = np.random.choice(list(record.keys()), size=max(1, len(record) // 10), replace=False)
        for field in missing_fields:
            if not field.endswith('_id') and field != 'created_at':  
                record[field] = np.nan
        
        data_records.append(record)
    
    df = pd.DataFrame(data_records)
    
    
    
    rating_cols = [col for col in df.columns if 'rating' in col or 'satisfaction' in col]
    if rating_cols:
        df['recommendation_score'] = df[rating_cols].mean(axis=1) + np.random.normal(0, 0.5, len(df))
        df['recommendation_score'] = df['recommendation_score'].clip(1, 10)
    else:
        df['recommendation_score'] = np.random.uniform(1, 10, len(df))
    
    logger.info(f"Created sample data with shape {df.shape}")
    
    return df


def validate_dataframe_for_preprocessing(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Validate a DataFrame for preprocessing pipeline.
    
    Args:
        df: DataFrame to validate
        
    Returns:
        Validation report dictionary
    """
    logger.info("Validating DataFrame for preprocessing")
    
    validation_report = {
        'is_valid': True,
        'warnings': [],
        'errors': [],
        'recommendations': []
    }
    
    
    if df.empty:
        validation_report['errors'].append("DataFrame is empty")
        validation_report['is_valid'] = False
        return validation_report
    
    if len(df) < 10:
        validation_report['warnings'].append(f"Very small dataset: only {len(df)} rows")
    
    
    if len(df.select_dtypes(include=[np.number]).columns) == 0:
        validation_report['warnings'].append("No numerical columns found")
    
    
    missing_percentage = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
    if missing_percentage > 50:
        validation_report['warnings'].append(f"High missing value percentage: {missing_percentage:.1f}%")
    
    
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        unique_count = df[col].nunique()
        if unique_count > 0.8 * len(df):
            validation_report['warnings'].append(f"High cardinality column: {col} ({unique_count} unique values)")
    
    
    constant_cols = []
    for col in df.columns:
        if df[col].nunique() <= 1:
            constant_cols.append(col)
    
    if constant_cols:
        validation_report['warnings'].append(f"Constant columns found: {constant_cols}")
    
    
    if len(df) < 100:
        validation_report['recommendations'].append("Consider collecting more data for better preprocessing results")
    
    if missing_percentage > 20:
        validation_report['recommendations'].append("Consider investigating data collection process to reduce missing values")
    
    
    validation_report['summary'] = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'missing_percentage': missing_percentage,
        'numerical_columns': len(df.select_dtypes(include=[np.number]).columns),
        'categorical_columns': len(df.select_dtypes(include=['object']).columns),
        'constant_columns': len(constant_cols)
    }
    
    if validation_report['errors']:
        validation_report['is_valid'] = False
    
    logger.info(f"Validation completed. Valid: {validation_report['is_valid']}")
    
    return validation_report


def get_column_info(df: pd.DataFrame) -> pd.DataFrame:
    """
    Get detailed information about DataFrame columns.
    
    Args:
        df: DataFrame to analyze
        
    Returns:
        DataFrame with column information
    """
    column_info = []
    
    for col in df.columns:
        info = {
            'column_name': col,
            'data_type': str(df[col].dtype),
            'non_null_count': df[col].count(),
            'null_count': df[col].isnull().sum(),
            'null_percentage': (df[col].isnull().sum() / len(df)) * 100,
            'unique_count': df[col].nunique(),
            'unique_percentage': (df[col].nunique() / len(df)) * 100
        }
        
        
        if df[col].dtype in ['int64', 'float64']:
            info.update({
                'min_value': df[col].min(),
                'max_value': df[col].max(),
                'mean_value': df[col].mean(),
                'std_value': df[col].std()
            })
        elif df[col].dtype == 'object':
            
            if df[col].count() > 0:
                mode_value = df[col].mode()
                info['most_common'] = mode_value[0] if not mode_value.empty else None
                info['most_common_count'] = (df[col] == info['most_common']).sum() if info['most_common'] else 0
        
        column_info.append(info)
    
    return pd.DataFrame(column_info) 