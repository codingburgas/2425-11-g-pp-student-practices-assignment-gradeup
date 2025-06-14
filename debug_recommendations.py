#!/usr/bin/env python3
"""
Debug script to test recommendations with actual survey data
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import create_app, db
from app.models import SurveyResponse, User
from app.ml.recommendation_engine import recommendation_engine

def test_recommendations_with_real_data():
    """Test recommendations with actual survey data from database."""
    
    app = create_app()
    with app.app_context():
        print("=== TESTING RECOMMENDATIONS WITH REAL DATA ===")
        
        # Get the user who has taken the survey (user ID 2)
        user_with_survey = User.query.get(2)  # bbf user
        if not user_with_survey:
            print("❌ User ID 2 not found")
            return
            
        print(f"✅ Testing recommendations for user: {user_with_survey.username}")
        
        # Get their survey response
        survey_response = SurveyResponse.query.filter_by(user_id=2).first()
        if not survey_response:
            print("❌ No survey response found for user ID 2")
            return
            
        print(f"✅ Found survey response from {survey_response.created_at}")
        
        # Get the survey data
        survey_data = survey_response.get_answers()
        print(f"📊 Survey data keys: {list(survey_data.keys())}")
        print(f"📊 Survey data sample: {dict(list(survey_data.items())[:3])}")
        
        # Test program recommendations
        print("\n🔍 Testing Program Recommendations...")
        try:
            program_recs = recommendation_engine.recommend_programs(
                user_id=2,
                survey_data=survey_data,
                user_preferences=user_with_survey.get_preferences() if user_with_survey.preferences else None,
                top_k=5
            )
            
            print(f"✅ Found {len(program_recs)} program recommendations")
            for i, rec in enumerate(program_recs, 1):
                print(f"  {i}. {rec.get('program_name', 'N/A')} - {rec.get('school_name', 'N/A')}")
                print(f"     Match Score: {rec.get('match_score', 0):.2f}")
                print(f"     Reasons: {rec.get('recommendation_reasons', [])}")
                print()
                
        except Exception as e:
            print(f"❌ Error in program recommendations: {e}")
            import traceback
            traceback.print_exc()
        
        # Test university recommendations
        print("\n🏫 Testing University Recommendations...")
        try:
            university_recs = recommendation_engine.match_universities(
                user_preferences=user_with_survey.get_preferences() if user_with_survey.preferences else {},
                survey_data=survey_data,
                top_k=5
            )
            
            print(f"✅ Found {len(university_recs)} university recommendations")
            for i, rec in enumerate(university_recs, 1):
                print(f"  {i}. {rec.get('school_name', 'N/A')} - {rec.get('location', 'N/A')}")
                print(f"     Match Score: {rec.get('match_score', 0):.2f}")
                print(f"     Reasons: {rec.get('match_reasons', [])}")
                print()
                
        except Exception as e:
            print(f"❌ Error in university recommendations: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    test_recommendations_with_real_data() 