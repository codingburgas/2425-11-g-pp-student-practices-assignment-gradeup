"""
Feature Engineering Module

Handles feature engineering operations including:
- Creating interaction features
- Polynomial features
- Aggregation features
- Domain-specific survey features
- Statistical features
- Time-based features
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Callable
import logging
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from itertools import combinations
import re

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Handles feature engineering operations for survey data.
    """
    
    def __init__(self, 
                 create_interactions: bool = True,
                 create_polynomials: bool = False,
                 polynomial_degree: int = 2,
                 create_aggregations: bool = True,
                 create_domain_features: bool = True):
        """
        Initialize FeatureEngineer.
        
        Args:
            create_interactions: Whether to create interaction features
            create_polynomials: Whether to create polynomial features
            polynomial_degree: Degree for polynomial features
            create_aggregations: Whether to create aggregation features
            create_domain_features: Whether to create domain-specific features
        """
        self.create_interactions = create_interactions
        self.create_polynomials = create_polynomials
        self.polynomial_degree = polynomial_degree
        self.create_aggregations = create_aggregations
        self.create_domain_features = create_domain_features
        self.feature_engineering_report = {}
        self.created_features = []
        
    def engineer_features(self, data: pd.DataFrame, target_column: Optional[str] = None) -> pd.DataFrame:
        """
        Main method to engineer features from survey data.
        
        Args:
            data: Input DataFrame
            target_column: Target column for supervised feature selection
            
        Returns:
            DataFrame with engineered features
        """
        logger.info("Starting feature engineering process")
        
        
        engineered_data = data.copy()
        
        
        original_shape = engineered_data.shape
        self.feature_engineering_report['original_shape'] = original_shape
        self.feature_engineering_report['original_features'] = list(engineered_data.columns)
        
        
        if self.create_interactions:
            engineered_data = self._create_interaction_features(engineered_data)
        
        
        if self.create_polynomials:
            engineered_data = self._create_polynomial_features(engineered_data)
        
        
        if self.create_aggregations:
            engineered_data = self._create_aggregation_features(engineered_data)
        
        
        if self.create_domain_features:
            engineered_data = self._create_domain_features(engineered_data)
        
        
        engineered_data = self._create_statistical_features(engineered_data)
        
        
        engineered_data = self._create_ratio_difference_features(engineered_data)
        
        
        if target_column and target_column in engineered_data.columns:
            engineered_data = self._select_features(engineered_data, target_column)
        
        
        final_shape = engineered_data.shape
        self.feature_engineering_report['final_shape'] = final_shape
        self.feature_engineering_report['final_features'] = list(engineered_data.columns)
        self.feature_engineering_report['features_added'] = final_shape[1] - original_shape[1]
        self.feature_engineering_report['created_features'] = self.created_features
        
        logger.info(f"Feature engineering completed. Shape changed from {original_shape} to {final_shape}")
        logger.info(f"Created {len(self.created_features)} new features")
        
        return engineered_data
    
    def _create_interaction_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features between numerical columns."""
        numerical_columns = data.select_dtypes(include=[np.number]).columns.tolist()
        
        
        if len(numerical_columns) > 10:
            
            corr_matrix = data[numerical_columns].corr().abs()
            
            avg_corr = corr_matrix.mean().sort_values(ascending=False)
            numerical_columns = avg_corr.head(10).index.tolist()
        
        interaction_features = []
        interaction_info = {}
        
        
        for col1, col2 in combinations(numerical_columns, 2):
            
            interaction_name = f"{col1}_x_{col2}"
            data[interaction_name] = data[col1] * data[col2]
            interaction_features.append(interaction_name)
            
            
            if (data[col2] != 0).all():
                division_name = f"{col1}_div_{col2}"
                data[division_name] = data[col1] / data[col2]
                interaction_features.append(division_name)
        
        if interaction_features:
            interaction_info['created_features'] = interaction_features
            interaction_info['feature_count'] = len(interaction_features)
            self.feature_engineering_report['interaction_features'] = interaction_info
            self.created_features.extend(interaction_features)
            logger.info(f"Created {len(interaction_features)} interaction features")
        
        return data
    
    def _create_polynomial_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create polynomial features."""
        numerical_columns = data.select_dtypes(include=[np.number]).columns.tolist()
        
        
        if len(numerical_columns) > 5:
            numerical_columns = numerical_columns[:5]
        
        polynomial_features = []
        polynomial_info = {}
        
        for column in numerical_columns:
            for degree in range(2, self.polynomial_degree + 1):
                poly_name = f"{column}_poly_{degree}"
                data[poly_name] = data[column] ** degree
                polynomial_features.append(poly_name)
        
        if polynomial_features:
            polynomial_info['created_features'] = polynomial_features
            polynomial_info['feature_count'] = len(polynomial_features)
            polynomial_info['degree'] = self.polynomial_degree
            self.feature_engineering_report['polynomial_features'] = polynomial_info
            self.created_features.extend(polynomial_features)
            logger.info(f"Created {len(polynomial_features)} polynomial features")
        
        return data
    
    def _create_aggregation_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create aggregation features grouped by categorical variables."""
        numerical_columns = data.select_dtypes(include=[np.number]).columns.tolist()
        categorical_columns = data.select_dtypes(include=['object', 'category']).columns.tolist()
        
        aggregation_features = []
        aggregation_info = {}
        
        for cat_col in categorical_columns:
            for num_col in numerical_columns:
                
                grouped = data.groupby(cat_col)[num_col]
                
                
                mean_name = f"{num_col}_mean_by_{cat_col}"
                group_means = grouped.transform('mean')
                data[mean_name] = group_means
                aggregation_features.append(mean_name)
                
                
                std_name = f"{num_col}_std_by_{cat_col}"
                group_stds = grouped.transform('std').fillna(0)
                data[std_name] = group_stds
                aggregation_features.append(std_name)
                
                
                deviation_name = f"{num_col}_deviation_from_{cat_col}_mean"
                data[deviation_name] = data[num_col] - group_means
                aggregation_features.append(deviation_name)
        
        if aggregation_features:
            aggregation_info['created_features'] = aggregation_features
            aggregation_info['feature_count'] = len(aggregation_features)
            self.feature_engineering_report['aggregation_features'] = aggregation_info
            self.created_features.extend(aggregation_features)
            logger.info(f"Created {len(aggregation_features)} aggregation features")
        
        return data
    
    def _create_domain_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create domain-specific features for survey data."""
        domain_features = []
        domain_info = {}
        
        
        rating_columns = [col for col in data.columns 
                         if any(keyword in col.lower() for keyword in ['rating', 'score', 'satisfaction', 'importance'])]
        
        if rating_columns:
            
            satisfaction_name = "overall_satisfaction_score"
            data[satisfaction_name] = data[rating_columns].mean(axis=1)
            domain_features.append(satisfaction_name)
            
            
            consistency_name = "rating_consistency"
            data[consistency_name] = data[rating_columns].std(axis=1).fillna(0)
            domain_features.append(consistency_name)
            
            
            if data[rating_columns].max().max() <= 10:  
                extreme_name = "extreme_responses_count"
                extreme_responses = ((data[rating_columns] <= 2) | (data[rating_columns] >= 9)).sum(axis=1)
                data[extreme_name] = extreme_responses
                domain_features.append(extreme_name)
            
            
            range_name = "rating_range"
            data[range_name] = data[rating_columns].max(axis=1) - data[rating_columns].min(axis=1)
            domain_features.append(range_name)
        
        
        preference_columns = [col for col in data.columns 
                            if any(keyword in col.lower() for keyword in ['prefer', 'choice', 'select', 'option'])]
        
        if preference_columns:
            
            preferences_count_name = "total_preferences_count"
            
            data[preferences_count_name] = data[preference_columns].sum(axis=1)
            domain_features.append(preferences_count_name)
        
        
        completeness_name = "response_completeness"
        data[completeness_name] = (data.notna().sum(axis=1) / len(data.columns)) * 100
        domain_features.append(completeness_name)
        
        
        timestamp_columns = [col for col in data.columns 
                           if any(keyword in col.lower() for keyword in ['time', 'date', 'created', 'timestamp'])]
        
        if len(timestamp_columns) >= 2:
            
            for i in range(len(timestamp_columns) - 1):
                try:
                    start_col = timestamp_columns[i]
                    end_col = timestamp_columns[i + 1]
                    
                    
                    start_time = pd.to_datetime(data[start_col], errors='coerce')
                    end_time = pd.to_datetime(data[end_col], errors='coerce')
                    
                    duration_name = f"duration_{start_col}_to_{end_col}"
                    data[duration_name] = (end_time - start_time).dt.total_seconds() / 60  
                    domain_features.append(duration_name)
                except:
                    logger.warning(f"Could not calculate duration between {start_col} and {end_col}")
        
        if domain_features:
            domain_info['created_features'] = domain_features
            domain_info['feature_count'] = len(domain_features)
            domain_info['rating_columns_used'] = rating_columns
            domain_info['preference_columns_used'] = preference_columns
            self.feature_engineering_report['domain_features'] = domain_info
            self.created_features.extend(domain_features)
            logger.info(f"Created {len(domain_features)} domain-specific features")
        
        return data
    
    def _create_statistical_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create statistical features."""
        numerical_columns = data.select_dtypes(include=[np.number]).columns.tolist()
        
        statistical_features = []
        statistical_info = {}
        
        if len(numerical_columns) >= 3:
            
            row_stats = {
                'row_mean': data[numerical_columns].mean(axis=1),
                'row_std': data[numerical_columns].std(axis=1).fillna(0),
                'row_median': data[numerical_columns].median(axis=1),
                'row_min': data[numerical_columns].min(axis=1),
                'row_max': data[numerical_columns].max(axis=1),
                'row_skew': data[numerical_columns].skew(axis=1).fillna(0),
                'row_kurt': data[numerical_columns].kurtosis(axis=1).fillna(0)
            }
            
            for stat_name, stat_values in row_stats.items():
                data[stat_name] = stat_values
                statistical_features.append(stat_name)
        
        if statistical_features:
            statistical_info['created_features'] = statistical_features
            statistical_info['feature_count'] = len(statistical_features)
            self.feature_engineering_report['statistical_features'] = statistical_info
            self.created_features.extend(statistical_features)
            logger.info(f"Created {len(statistical_features)} statistical features")
        
        return data
    
    def _create_ratio_difference_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create ratio and difference features between related columns."""
        numerical_columns = data.select_dtypes(include=[np.number]).columns.tolist()
        
        ratio_diff_features = []
        ratio_diff_info = {}
        
        
        column_groups = self._group_similar_columns(numerical_columns)
        
        for group_name, columns in column_groups.items():
            if len(columns) >= 2:
                for i in range(len(columns)):
                    for j in range(i + 1, len(columns)):
                        col1, col2 = columns[i], columns[j]
                        
                        
                        if (data[col2] != 0).all():
                            ratio_name = f"ratio_{col1}_to_{col2}"
                            data[ratio_name] = data[col1] / data[col2]
                            ratio_diff_features.append(ratio_name)
                        
                        
                        diff_name = f"diff_{col1}_minus_{col2}"
                        data[diff_name] = data[col1] - data[col2]
                        ratio_diff_features.append(diff_name)
        
        if ratio_diff_features:
            ratio_diff_info['created_features'] = ratio_diff_features
            ratio_diff_info['feature_count'] = len(ratio_diff_features)
            ratio_diff_info['column_groups'] = {k: v for k, v in column_groups.items() if len(v) >= 2}
            self.feature_engineering_report['ratio_difference_features'] = ratio_diff_info
            self.created_features.extend(ratio_diff_features)
            logger.info(f"Created {len(ratio_diff_features)} ratio and difference features")
        
        return data
    
    def _group_similar_columns(self, columns: List[str]) -> Dict[str, List[str]]:
        """Group columns with similar names/patterns."""
        groups = {}
        
        
        for col in columns:
            
            base_name = re.sub(r'(_\d+|_score|_rating|_level|_value)$', '', col.lower())
            base_name = re.sub(r'\d+$', '', base_name)
            
            if base_name not in groups:
                groups[base_name] = []
            groups[base_name].append(col)
        
        
        return {k: v for k, v in groups.items() if len(v) > 1}
    
    def _select_features(self, data: pd.DataFrame, target_column: str, k: int = 50) -> pd.DataFrame:
        """Select top k features using statistical tests."""
        feature_columns = [col for col in data.columns if col != target_column]
        
        if len(feature_columns) <= k:
            logger.info(f"Number of features ({len(feature_columns)}) is less than k ({k}). Keeping all features.")
            return data
        
        
        X = data[feature_columns]
        y = data[target_column]
        
        
        if y.dtype == 'object':
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            y = le.fit_transform(y)
        
        
        X = X.fillna(X.median())
        
        try:
            
            selector = SelectKBest(score_func=mutual_info_classif, k=min(k, len(feature_columns)))
            X_selected = selector.fit_transform(X, y)
            
            
            selected_features = [feature_columns[i] for i in selector.get_support(indices=True)]
            selected_features.append(target_column)  
            
            
            selected_data = data[selected_features].copy()
            
            feature_selection_info = {
                'method': 'mutual_info_classif',
                'k': k,
                'original_feature_count': len(feature_columns),
                'selected_feature_count': len(selected_features) - 1,  
                'selected_features': selected_features[:-1]  
            }
            
            self.feature_engineering_report['feature_selection'] = feature_selection_info
            logger.info(f"Selected {len(selected_features) - 1} features out of {len(feature_columns)}")
            
            return selected_data
            
        except Exception as e:
            logger.warning(f"Feature selection failed: {e}. Returning all features.")
            return data
    
    def get_feature_importance(self, data: pd.DataFrame, target_column: str) -> pd.DataFrame:
        """Get feature importance scores."""
        feature_columns = [col for col in data.columns if col != target_column]
        
        X = data[feature_columns].fillna(data[feature_columns].median())
        y = data[target_column]
        
        
        if y.dtype == 'object':
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            y = le.fit_transform(y)
        
        try:
            
            mi_scores = mutual_info_classif(X, y)
            
            
            importance_df = pd.DataFrame({
                'feature': feature_columns,
                'mutual_info_score': mi_scores
            }).sort_values('mutual_info_score', ascending=False)
            
            return importance_df
            
        except Exception as e:
            logger.error(f"Could not calculate feature importance: {e}")
            return pd.DataFrame()
    
    def get_feature_engineering_report(self) -> Dict[str, Any]:
        """Get the feature engineering report."""
        return self.feature_engineering_report
    
    def save_feature_engineering_report(self, filepath: str):
        """Save feature engineering report to a file."""
        import json
        
        with open(filepath, 'w') as f:
            json.dump(self.feature_engineering_report, f, indent=2)
        
        logger.info(f"Feature engineering report saved to {filepath}")
    
    def get_created_features(self) -> List[str]:
        """Get list of all created features."""
        return self.created_features 