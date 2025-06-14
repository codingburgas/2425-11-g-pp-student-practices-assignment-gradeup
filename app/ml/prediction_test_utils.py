"""
Test utilities for the Advanced Prediction System

Provides helper functions and test data for testing the prediction system
"""

import json
import random
from typing import Dict, List, Any
from datetime import datetime, timedelta

from app import db
from app.models import User, SurveyResponse, Survey, Program, School, PredictionHistory
from .prediction_system import advanced_prediction_system


class PredictionSystemTestUtils:
    """Utility class for testing the prediction system."""
    
    @staticmethod
    def create_test_survey_data(complexity: str = 'simple') -> Dict[str, Any]:
        """
        Create test survey data for prediction testing.
        
        Args:
            complexity: 'simple', 'medium', or 'complex'
            
        Returns:
            Dictionary with survey response data
        """
        if complexity == 'simple':
            return {
                'career_interests': ['technology', 'programming'],
                'favorite_subjects': ['mathematics', 'computer_science'],
                'preferred_work_environment': 'office',
                'career_goals': 'Software Developer',
                'study_preferences': 'theoretical'
            }
        
        elif complexity == 'medium':
            return {
                'career_interests': ['business', 'marketing', 'management'],
                'favorite_subjects': ['economics', 'psychology', 'communications'],
                'preferred_work_environment': 'mixed',
                'career_goals': 'Marketing Manager for tech company',
                'study_preferences': 'practical',
                'work_style': 'collaborative',
                'industry_preference': 'technology',
                'salary_importance': 7,
                'work_life_balance_importance': 8
            }
        
        else:  # complex
            return {
                'career_interests': ['healthcare', 'research', 'helping_others'],
                'favorite_subjects': ['biology', 'chemistry', 'psychology', 'statistics'],
                'preferred_work_environment': 'hospital',
                'career_goals': 'Medical researcher specializing in neurology',
                'study_preferences': 'mixed',
                'work_style': 'independent',
                'industry_preference': 'healthcare',
                'salary_importance': 6,
                'work_life_balance_importance': 5,
                'research_interest': 9,
                'patient_interaction_preference': 4,
                'academic_strength': ['analytical_thinking', 'problem_solving'],
                'extracurricular_activities': ['science_club', 'volunteer_work'],
                'geographic_preference': 'urban',
                'school_size_preference': 'large'
            }
    
    @staticmethod
    def create_batch_test_data(count: int = 5) -> List[Dict[str, Any]]:
        """
        Create batch test data for testing batch predictions.
        
        Args:
            count: Number of test requests to create
            
        Returns:
            List of prediction request dictionaries
        """
        complexities = ['simple', 'medium', 'complex']
        test_requests = []
        
        for i in range(count):
            complexity = random.choice(complexities)
            survey_data = PredictionSystemTestUtils.create_test_survey_data(complexity)
            
            request = {
                'survey_data': survey_data,
                'top_k': random.randint(3, 8),
                'user_id': 1  # Assuming test user with id 1 exists
            }
            test_requests.append(request)
        
        return test_requests
    
    @staticmethod
    def test_individual_prediction(user_id: int = 1, 
                                 complexity: str = 'medium') -> Dict[str, Any]:
        """
        Test individual prediction functionality.
        
        Args:
            user_id: ID of user to test with
            complexity: Complexity level of test data
            
        Returns:
            Prediction result
        """
        survey_data = PredictionSystemTestUtils.create_test_survey_data(complexity)
        
        result = advanced_prediction_system.predict_with_confidence(
            survey_data=survey_data,
            user_id=user_id,
            top_k=5,
            store_history=True
        )
        
        return result
    
    @staticmethod
    def test_batch_prediction(count: int = 3) -> List[Dict[str, Any]]:
        """
        Test batch prediction functionality.
        
        Args:
            count: Number of predictions in batch
            
        Returns:
            List of prediction results
        """
        test_requests = PredictionSystemTestUtils.create_batch_test_data(count)
        
        results = advanced_prediction_system.batch_predict(
            prediction_requests=test_requests,
            store_history=True
        )
        
        return results
    
    @staticmethod
    def test_prediction_history(user_id: int = 1) -> Dict[str, Any]:
        """
        Test prediction history functionality.
        
        Args:
            user_id: ID of user to test with
            
        Returns:
            Dictionary with history and analysis results
        """
        # Get prediction history
        history = advanced_prediction_system.get_prediction_history(
            user_id=user_id,
            limit=10
        )
        
        # Get pattern analysis
        analysis = advanced_prediction_system.analyze_prediction_patterns(user_id)
        
        return {
            'history': history,
            'analysis': analysis,
            'history_count': len(history)
        }
    
    @staticmethod
    def run_comprehensive_test(user_id: int = 1) -> Dict[str, Any]:
        """
        Run a comprehensive test of all prediction system features.
        
        Args:
            user_id: ID of user to test with
            
        Returns:
            Dictionary with all test results
        """
        test_results = {}
        
        try:
            # Test 1: Individual predictions
            print("Testing individual predictions...")
            individual_simple = PredictionSystemTestUtils.test_individual_prediction(
                user_id, 'simple'
            )
            individual_complex = PredictionSystemTestUtils.test_individual_prediction(
                user_id, 'complex'
            )
            
            test_results['individual_predictions'] = {
                'simple': individual_simple,
                'complex': individual_complex
            }
            
            # Test 2: Batch predictions
            print("Testing batch predictions...")
            batch_results = PredictionSystemTestUtils.test_batch_prediction(3)
            test_results['batch_predictions'] = batch_results
            
            # Test 3: History and analysis
            print("Testing prediction history and analysis...")
            history_results = PredictionSystemTestUtils.test_prediction_history(user_id)
            test_results['history_and_analysis'] = history_results
            
            # Test 4: System information
            print("Testing system information...")
            system_info = {
                'model_version': advanced_prediction_system.model_version,
                'confidence_threshold': advanced_prediction_system.confidence_threshold,
                'ml_service_trained': advanced_prediction_system.ml_service.is_trained
            }
            test_results['system_info'] = system_info
            
            # Summary statistics
            total_predictions = 0
            successful_predictions = 0
            
            # Count individual predictions
            for test_type in ['simple', 'complex']:
                result = test_results['individual_predictions'][test_type]
                total_predictions += 1
                if result.get('predictions'):
                    successful_predictions += 1
            
            # Count batch predictions
            for batch_result in test_results['batch_predictions']:
                total_predictions += 1
                if batch_result.get('predictions'):
                    successful_predictions += 1
            
            test_results['summary'] = {
                'total_tests': total_predictions,
                'successful_tests': successful_predictions,
                'success_rate': successful_predictions / max(1, total_predictions),
                'test_timestamp': datetime.utcnow().isoformat()
            }
            
            print(f"Test completed: {successful_predictions}/{total_predictions} successful")
            
        except Exception as e:
            test_results['error'] = str(e)
            print(f"Test failed with error: {e}")
        
        return test_results
    
    @staticmethod
    def create_demo_data() -> Dict[str, Any]:
        """
        Create demo data for showcasing the prediction system.
        
        Returns:
            Dictionary with demo results
        """
        demo_scenarios = [
            {
                'name': 'Aspiring Software Engineer',
                'survey_data': {
                    'career_interests': ['technology', 'programming', 'innovation'],
                    'favorite_subjects': ['mathematics', 'computer_science', 'physics'],
                    'career_goals': 'Senior Software Engineer at a tech company',
                    'study_preferences': 'hands_on',
                    'work_style': 'collaborative',
                    'salary_importance': 8,
                    'work_life_balance_importance': 7
                }
            },
            {
                'name': 'Future Business Leader',
                'survey_data': {
                    'career_interests': ['business', 'leadership', 'entrepreneurship'],
                    'favorite_subjects': ['economics', 'business_studies', 'communications'],
                    'career_goals': 'Start my own company',
                    'study_preferences': 'theoretical_and_practical',
                    'work_style': 'leadership',
                    'salary_importance': 9,
                    'work_life_balance_importance': 6
                }
            },
            {
                'name': 'Healthcare Professional',
                'survey_data': {
                    'career_interests': ['healthcare', 'helping_others', 'medical_research'],
                    'favorite_subjects': ['biology', 'chemistry', 'psychology'],
                    'career_goals': 'Pediatric nurse practitioner',
                    'study_preferences': 'practical',
                    'work_style': 'caring',
                    'salary_importance': 6,
                    'work_life_balance_importance': 8
                }
            }
        ]
        
        demo_results = {}
        
        for scenario in demo_scenarios:
            try:
                result = advanced_prediction_system.predict_with_confidence(
                    survey_data=scenario['survey_data'],
                    user_id=1,  # Demo user
                    top_k=3,
                    store_history=False  # Don't store demo data
                )
                demo_results[scenario['name']] = result
            except Exception as e:
                demo_results[scenario['name']] = {'error': str(e)}
        
        return demo_results
    
    @staticmethod
    def validate_prediction_quality(predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Validate the quality of predictions.
        
        Args:
            predictions: List of prediction dictionaries
            
        Returns:
            Quality validation results
        """
        if not predictions:
            return {'status': 'No predictions to validate'}
        
        quality_metrics = {
            'total_predictions': len(predictions),
            'high_confidence_predictions': 0,
            'medium_confidence_predictions': 0,
            'low_confidence_predictions': 0,
            'predictions_with_details': 0,
            'average_confidence': 0
        }
        
        confidence_scores = []
        
        for pred in predictions:
            confidence = pred.get('enhanced_confidence', pred.get('confidence', 0))
            confidence_scores.append(confidence)
            
            if confidence >= 0.7:
                quality_metrics['high_confidence_predictions'] += 1
            elif confidence >= 0.4:
                quality_metrics['medium_confidence_predictions'] += 1
            else:
                quality_metrics['low_confidence_predictions'] += 1
            
            if pred.get('confidence_factors'):
                quality_metrics['predictions_with_details'] += 1
        
        if confidence_scores:
            quality_metrics['average_confidence'] = sum(confidence_scores) / len(confidence_scores)
        
        quality_metrics['confidence_distribution'] = {
            'high': quality_metrics['high_confidence_predictions'] / len(predictions),
            'medium': quality_metrics['medium_confidence_predictions'] / len(predictions),
            'low': quality_metrics['low_confidence_predictions'] / len(predictions)
        }
        
        return quality_metrics


def run_prediction_system_demo():
    """
    Run a complete demonstration of the prediction system.
    """
    print("="*60)
    print("🎓 GradeUp Advanced Prediction System Demo")
    print("="*60)
    
    utils = PredictionSystemTestUtils()
    
    try:
        # Initialize the prediction system
        print("\n1. Initializing prediction system...")
        
        # Create demo data
        print("\n2. Running demo predictions...")
        demo_results = utils.create_demo_data()
        
        for scenario_name, result in demo_results.items():
            print(f"\n📊 {scenario_name}:")
            if 'error' in result:
                print(f"   ❌ Error: {result['error']}")
            else:
                predictions = result.get('predictions', [])
                print(f"   ✅ Generated {len(predictions)} recommendations")
                if predictions:
                    top_pred = predictions[0]
                    print(f"   🏆 Top recommendation: {top_pred.get('program_name', 'N/A')}")
                    print(f"   🎯 Confidence: {top_pred.get('enhanced_confidence', 0):.3f}")
        
        # Run comprehensive test
        print("\n3. Running comprehensive system test...")
        test_results = utils.run_comprehensive_test()
        
        if 'error' not in test_results:
            summary = test_results.get('summary', {})
            print(f"   ✅ Test completed successfully!")
            print(f"   📈 Success rate: {summary.get('success_rate', 0):.2%}")
            print(f"   📊 Tests passed: {summary.get('successful_tests', 0)}/{summary.get('total_tests', 0)}")
        else:
            print(f"   ❌ Test failed: {test_results['error']}")
        
        print("\n4. System is ready for use! 🚀")
        print("   Available API endpoints:")
        print("   • POST /api/prediction/predict - Individual predictions")
        print("   • POST /api/prediction/batch-predict - Batch predictions")
        print("   • GET /api/prediction/history - Prediction history")
        print("   • GET /api/prediction/analyze-patterns - Pattern analysis")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    run_prediction_system_demo()