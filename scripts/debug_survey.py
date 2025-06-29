#!/usr/bin/env python3
"""
Debug script to check survey and database status
"""
import os
import sys

# Set environment variables
os.environ['FLASK_ENV'] = 'development'

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from app import create_app, db
    from app.models import User, Survey, SurveyResponse
    
    app = create_app()
    
    with app.app_context():
        print("=== DATABASE STATUS ===")
        
        # Check if tables exist
        try:
            user_count = User.query.count()
            survey_count = Survey.query.count()
            response_count = SurveyResponse.query.count()
            
            print(f"✅ Database connected successfully")
            print(f"📊 Users: {user_count}")
            print(f"📋 Surveys: {survey_count}")
            print(f"💬 Survey Responses: {response_count}")
            
        except Exception as e:
            print(f"❌ Database connection error: {e}")
            
        print("\n=== SURVEY DATA ===")
        
        # List all surveys
        try:
            surveys = Survey.query.all()
            if surveys:
                for survey in surveys:
                    print(f"Survey ID: {survey.id}")
                    print(f"  Title: {survey.title}")
                    print(f"  Active: {survey.is_active}")
                    print(f"  Questions: {len(survey.get_questions())} questions")
                    print(f"  Created: {survey.created_at}")
                    
                    # Show first few questions
                    questions = survey.get_questions()
                    for i, q in enumerate(questions[:3]):
                        print(f"    Q{i+1}: {q.get('text', 'No text')[:50]}...")
                    
                    if len(questions) > 3:
                        print(f"    ... and {len(questions)-3} more questions")
                    print()
            else:
                print("❌ No surveys found in database")
                
        except Exception as e:
            print(f"❌ Error querying surveys: {e}")
            
        print("\n=== RECENT RESPONSES ===")
        
        # List recent responses
        try:
            responses = SurveyResponse.query.order_by(SurveyResponse.created_at.desc()).limit(5).all()
            if responses:
                for resp in responses:
                    print(f"Response ID: {resp.id}")
                    print(f"  User ID: {resp.user_id}")
                    print(f"  Survey ID: {resp.survey_id}")
                    print(f"  Created: {resp.created_at}")
                    answers = resp.get_answers()
                    print(f"  Answers: {len(answers)} questions answered")
                    print()
            else:
                print("❌ No responses found in database")
                
        except Exception as e:
            print(f"❌ Error querying responses: {e}")
            
        print("\n=== USER DATA ===")
        
        # List users
        try:
            users = User.query.all()
            if users:
                for user in users:
                    user_responses = SurveyResponse.query.filter_by(user_id=user.id).count()
                    print(f"User: {user.username} (ID: {user.id})")
                    print(f"  Email: {user.email}")
                    print(f"  Responses: {user_responses}")
                    print(f"  Created: {user.created_at}")
                    print()
            else:
                print("❌ No users found in database")
                
        except Exception as e:
            print(f"❌ Error querying users: {e}")

except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure Flask and SQLAlchemy are installed")
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc() 