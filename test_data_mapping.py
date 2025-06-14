#!/usr/bin/env python3
"""
Test script to verify the survey data mapping function
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import create_app, db
from app.models import SurveyResponse
from app.ml.utils import map_survey_data_to_recommendation_format
from app.ml.recommendation_engine import recommendation_engine

def test_data_mapping():
    """Test the survey data mapping and recommendations."""
    
    app = create_app()
    with app.app_context():
        print("=== TESTING DATA MAPPING ===")
        
        # Get the survey response
        response = SurveyResponse.query.first()
        if not response:
            print("❌ No survey response found")
            return
            
        print(f"✅ Testing with survey response from User {response.user_id}")
        
        # Get raw survey data
        raw_data = response.get_answers()
        print(f"📊 Raw survey data: {raw_data}")
        
        # Map the data
        mapped_data = map_survey_data_to_recommendation_format(raw_data)
        print(f"\n🔄 Mapped data: {mapped_data}")
        
        # Test program recommendations with mapped data
        print("\n🎯 Testing Program Recommendations with Mapped Data...")
        try:
            program_recs = recommendation_engine.recommend_programs(
                user_id=response.user_id,
                survey_data=mapped_data,
                user_preferences=None,
                top_k=5
            )
            
            print(f"✅ Found {len(program_recs)} program recommendations")
            for i, rec in enumerate(program_recs, 1):
                print(f"  {i}. {rec.get('program_name', 'N/A')} at {rec.get('school_name', 'N/A')}")
                print(f"     Match Score: {rec.get('match_score', 0):.3f} ({rec.get('match_score', 0)*100:.1f}%)")
                print(f"     Reasons: {rec.get('recommendation_reasons', [])}")
                print()
                
        except Exception as e:
            print(f"❌ Error in program recommendations: {e}")
            import traceback
            traceback.print_exc()
        
        # Test university recommendations with mapped data
        print("\n🏫 Testing University Recommendations with Mapped Data...")
        try:
            university_recs = recommendation_engine.match_universities(
                user_preferences={},
                survey_data=mapped_data,
                top_k=5
            )
            
            print(f"✅ Found {len(university_recs)} university recommendations")
            for i, rec in enumerate(university_recs, 1):
                print(f"  {i}. {rec.get('school_name', 'N/A')} - {rec.get('location', 'N/A')}")
                print(f"     Match Score: {rec.get('match_score', 0):.3f} ({rec.get('match_score', 0)*100:.1f}%)")
                print(f"     Reasons: {rec.get('match_reasons', [])}")
                print()
                
        except Exception as e:
            print(f"❌ Error in university recommendations: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    test_data_mapping() 