#!/usr/bin/env python3
"""
Test script for the dual AI prediction system
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app import create_app, db
from app.models import User, SurveyResponse
from app.main.routes import _get_main_method_predictions, _get_backup_method_predictions, _find_consensus_recommendations

def test_dual_system():
    """Test the dual prediction system"""
    
    app = create_app()
    
    with app.app_context():
        print("🧪 Testing Dual AI Prediction System")
        print("=" * 50)
        
        # Sample survey data
        survey_data = {
            'math_interest': 8,
            'science_interest': 7,
            'art_interest': 4,
            'sports_interest': 3,
            'preferred_study_method': 'interactive',
            'career_goal': 'technology',
            'budget_range': 'moderate',
            'location_preference': 'urban',
            'university_size': 'large',
            'academic_focus': 0.7,
            'social_life_importance': 0.5,
            'research_interest': 0.6
        }
        
        print("📊 Sample Survey Data:")
        for key, value in survey_data.items():
            print(f"   {key}: {value}")
        
        print("\n🧠 Testing Main AI Method (Neural Network)...")
        main_predictions = _get_main_method_predictions(survey_data, 1)
        
        print(f"✅ Main method returned {len(main_predictions)} predictions:")
        for i, pred in enumerate(main_predictions[:3]):
            confidence_pct = pred['confidence'] * 100 if pred['confidence'] <= 1 else pred['confidence']
            print(f"   {i+1}. {pred['program_name']} at {pred['school_name']} ({confidence_pct:.1f}%)")
        
        print("\n📈 Testing Backup Method (Statistical)...")
        backup_predictions = _get_backup_method_predictions(survey_data)
        
        print(f"✅ Backup method returned {len(backup_predictions)} predictions:")
        for i, pred in enumerate(backup_predictions[:3]):
            confidence_pct = pred['confidence'] * 100 if pred['confidence'] <= 1 else pred['confidence']
            print(f"   {i+1}. {pred['program_name']} at {pred['school_name']} ({confidence_pct:.1f}%)")
        
        print("\n🤝 Testing Consensus Recommendations...")
        consensus = _find_consensus_recommendations(main_predictions, backup_predictions)
        
        print(f"✅ Found {len(consensus)} consensus recommendations:")
        for i, rec in enumerate(consensus):
            print(f"   {i+1}. {rec['program_name']} - Average: {rec['avg_confidence']:.2%}")
            print(f"      Main: {rec['main_confidence']:.2%}, Backup: {rec['backup_confidence']:.2%}")
        
        print("\n🎉 Dual AI System Test Completed Successfully!")
        print("=" * 50)
        
        # Test different interest profiles
        print("\n🔄 Testing Different Interest Profiles...")
        
        test_profiles = [
            {
                'name': 'Art-focused student',
                'data': {**survey_data, 'art_interest': 9, 'math_interest': 3, 'career_goal': 'design'}
            },
            {
                'name': 'Science-focused student', 
                'data': {**survey_data, 'science_interest': 9, 'math_interest': 6, 'career_goal': 'research'}
            },
            {
                'name': 'Business-focused student',
                'data': {**survey_data, 'math_interest': 6, 'science_interest': 5, 'career_goal': 'management'}
            }
        ]
        
        for profile in test_profiles:
            print(f"\n👤 {profile['name']}:")
            main_preds = _get_main_method_predictions(profile['data'], 1)
            backup_preds = _get_backup_method_predictions(profile['data'])
            
            if main_preds:
                main_conf = main_preds[0]['confidence'] * 100 if main_preds[0]['confidence'] <= 1 else main_preds[0]['confidence']
                print(f"   Main top pick: {main_preds[0]['program_name']} ({main_conf:.1f}%)")
            if backup_preds:
                backup_conf = backup_preds[0]['confidence'] * 100 if backup_preds[0]['confidence'] <= 1 else backup_preds[0]['confidence']
                print(f"   Backup top pick: {backup_preds[0]['program_name']} ({backup_conf:.1f}%)")
        
        print("\n✅ All tests completed successfully!")


if __name__ == "__main__":
    test_dual_system() 