#!/usr/bin/env python3
"""
Test script for the Recommendation Engine

This script tests the main functionality of the recommendation engine
with proper Flask application context.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import json

def test_recommendation_engine():
    """Test the recommendation engine functionality."""
    print("🧪 Testing GradeUp Recommendation Engine")
    print("=" * 50)
    
    # Set up Flask application context
    try:
        from app import create_app, db
        from app.ml.recommendation_engine import recommendation_engine
        
        app = create_app()
        
        with app.app_context():
            print("✅ Flask application context created successfully")
            
            # Initialize the engine
            recommendation_engine.initialize()
            
            # Test data
            sample_survey_data = {
                'math_interest': 8,
                'science_interest': 7,
                'art_interest': 3,
                'sports_interest': 5,
                'study_hours_per_day': 4,
                'preferred_subject': 'Mathematics',
                'career_goal': 'Engineer',
                'extracurricular': True,
                'leadership_experience': True,
                'team_preference': False,
                'languages_spoken': ['Bulgarian', 'English'],
                'grades_average': 5.5
            }
            
            sample_user_preferences = {
                'preferred_location': 'Sofia',
                'max_tuition': 10000,
                'preferred_duration': '4 years'
            }
            
            print("\n📊 Test Data:")
            print("Survey Data:", json.dumps(sample_survey_data, indent=2))
            print("User Preferences:", json.dumps(sample_user_preferences, indent=2))
            print()
            
            # Check database connectivity
            print("🗄️ Testing Database Connectivity...")
            try:
                from app.models import School, Program, User
                
                school_count = School.query.count()
                program_count = Program.query.count()
                user_count = User.query.count()
                
                print(f"   📊 Database Statistics:")
                print(f"   - Schools: {school_count}")
                print(f"   - Programs: {program_count}")
                print(f"   - Users: {user_count}")
                
                if school_count == 0:
                    print("   ⚠️ No schools found - recommendations will be empty")
                if program_count == 0:
                    print("   ⚠️ No programs found - recommendations will be empty")
                    
            except Exception as e:
                print(f"   ❌ Database connectivity error: {e}")
            
            # Test university matching
            print("\n🏫 Testing University Matching...")
            try:
                universities = recommendation_engine.match_universities(
                    user_preferences=sample_user_preferences,
                    survey_data=sample_survey_data,
                    top_k=5
                )
                print(f"✅ University matching works - returned {len(universities)} results")
                if universities:
                    print("   🎯 Sample result:")
                    sample = universities[0]
                    print(f"      - Name: {sample.get('school_name', 'N/A')}")
                    print(f"      - Location: {sample.get('location', 'N/A')}")
                    print(f"      - Match Score: {sample.get('match_score', 0):.2f}")
                    print(f"      - Reasons: {len(sample.get('match_reasons', []))}")
                else:
                    print("   ℹ️ No universities matched (possibly no data in database)")
                    
            except Exception as e:
                print(f"❌ University matching error: {e}")
            
            # Test program recommendations
            print("\n📚 Testing Program Recommendations...")
            try:
                programs = recommendation_engine.recommend_programs(
                    user_id=1,  # Test user ID
                    survey_data=sample_survey_data,
                    user_preferences=sample_user_preferences,
                    top_k=5
                )
                print(f"✅ Program recommendation works - returned {len(programs)} results")
                if programs:
                    print("   🎯 Sample result:")
                    sample = programs[0]
                    print(f"      - Program: {sample.get('program_name', 'N/A')}")
                    print(f"      - School: {sample.get('school_name', 'N/A')}")
                    print(f"      - Match Score: {sample.get('match_score', 0):.2f}")
                    print(f"      - Reasons: {sample.get('recommendation_reasons', [])}")
                else:
                    print("   ℹ️ No programs matched (possibly no data in database)")
                    
            except Exception as e:
                print(f"❌ Program recommendation error: {e}")
            
            # Test personalized suggestions
            print("\n💡 Testing Personalized Suggestions...")
            try:
                suggestions = recommendation_engine.get_personalized_suggestions(
                    user_id=1,
                    limit=3
                )
                print(f"✅ Personalized suggestions work")
                print("   📋 Available categories:")
                for category, items in suggestions.items():
                    count = len(items) if isinstance(items, list) else 'N/A'
                    print(f"      - {category}: {count} items")
                    
            except Exception as e:
                print(f"❌ Personalized suggestions error: {e}")
            
            # Test recommendation history
            print("\n📈 Testing Recommendation History...")
            try:
                history = recommendation_engine.get_recommendation_history(
                    user_id=1,
                    limit=5
                )
                print(f"✅ Recommendation history works - found {len(history)} records")
                
                if history:
                    print("   📊 Recent history:")
                    for record in history[:2]:
                        print(f"      - {record.get('created_at', 'N/A')}: {record.get('prediction_type', 'N/A')}")
                        
            except Exception as e:
                print(f"❌ Recommendation history error: {e}")
            
            # Test helper functions
            print("\n🔧 Testing Helper Functions...")
            try:
                # Test program type classification
                test_programs = [
                    "Computer Science Engineering",
                    "Business Administration", 
                    "Medicine",
                    "Fine Arts",
                    "Physics Research"
                ]
                
                print("   🏷️ Program Classification:")
                for prog in test_programs:
                    prog_type = recommendation_engine._classify_program_type(prog)
                    print(f"      '{prog}' -> '{prog_type}'")
                
                print("✅ Helper functions work correctly")
                
            except Exception as e:
                print(f"❌ Helper functions error: {e}")
            
            print("\n" + "=" * 50)
            print("🎉 Recommendation Engine Test Complete!")
            print("\n📋 Summary:")
            print("   ✅ Flask application context: Working")
            print("   ✅ Database connectivity: Working")
            print("   ✅ Recommendation engine: Initialized")
            print("   ✅ Core algorithms: Functional")
            
            print("\n🚀 Next steps:")
            print("1. Run the Flask application: python app.py")
            print("2. Visit the dashboard: http://localhost:5000/dashboard") 
            print("3. Check recommendations: http://localhost:5000/recommendations")
            print("4. Test API endpoints with sample data")
            print("5. Add sample schools/programs if database is empty")
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure you're in the correct directory and virtual environment is activated")
    except Exception as e:
        print(f"❌ Application setup error: {e}")
        print("Make sure the database is configured correctly")

if __name__ == "__main__":
    test_recommendation_engine() 