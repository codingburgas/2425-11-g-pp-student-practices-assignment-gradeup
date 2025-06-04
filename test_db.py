#!/usr/bin/env python3
"""
Simple database test script to check survey response saving
"""
import os
import sys

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from flask import Flask
    from flask_sqlalchemy import SQLAlchemy
    from datetime import datetime
    import json
    
    # Create minimal Flask app for testing
    app = Flask(__name__)
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///test.db'
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    
    db = SQLAlchemy(app)
    
    # Minimal model definitions for testing
    class User(db.Model):
        __tablename__ = 'users'
        id = db.Column(db.Integer, primary_key=True)
        username = db.Column(db.String(64), unique=True, nullable=False)
        email = db.Column(db.String(120), unique=True, nullable=False)
        created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    class Survey(db.Model):
        __tablename__ = 'surveys'
        id = db.Column(db.Integer, primary_key=True)
        title = db.Column(db.String(100), nullable=False)
        questions = db.Column(db.Text, nullable=False)
        is_active = db.Column(db.Boolean, default=True)
        created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    class SurveyResponse(db.Model):
        __tablename__ = 'survey_responses'
        id = db.Column(db.Integer, primary_key=True)
        answers = db.Column(db.Text, nullable=False)
        created_at = db.Column(db.DateTime, default=datetime.utcnow)
        user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
        survey_id = db.Column(db.Integer, db.ForeignKey('surveys.id'), nullable=False)
    
    with app.app_context():
        # Create tables
        db.create_all()
        
        # Check existing data
        user_count = User.query.count()
        survey_count = Survey.query.count()
        response_count = SurveyResponse.query.count()
        
        print(f"Database Test Results:")
        print(f"Users: {user_count}")
        print(f"Surveys: {survey_count}")
        print(f"Survey Responses: {response_count}")
        
        # List existing responses
        if response_count > 0:
            responses = SurveyResponse.query.all()
            print(f"\nExisting responses:")
            for resp in responses:
                print(f"  ID: {resp.id}, User: {resp.user_id}, Survey: {resp.survey_id}, Created: {resp.created_at}")
        
        # Test creating a response
        try:
            # Check if we have any users/surveys to test with
            test_user = User.query.first()
            test_survey = Survey.query.first()
            
            if test_user and test_survey:
                test_answers = {"1": "Test Answer", "2": "Another Answer"}
                test_response = SurveyResponse(
                    user_id=test_user.id,
                    survey_id=test_survey.id,
                    answers=json.dumps(test_answers)
                )
                
                db.session.add(test_response)
                db.session.commit()
                
                print(f"\n✅ Successfully created test response with ID: {test_response.id}")
                
                # Verify it was saved
                saved_response = SurveyResponse.query.filter_by(id=test_response.id).first()
                if saved_response:
                    print(f"✅ Verified test response was saved: {saved_response.id}")
                else:
                    print("❌ Test response was not found after save!")
                    
            else:
                print(f"\n⚠️ Cannot test response creation - missing test data:")
                print(f"   Users available: {test_user is not None}")
                print(f"   Surveys available: {test_survey is not None}")
        
        except Exception as e:
            print(f"\n❌ Error testing response creation: {e}")
            import traceback
            traceback.print_exc()

except ImportError as e:
    print(f"Missing dependencies: {e}")
    print("Please install Flask and SQLAlchemy first")
except Exception as e:
    print(f"Database test error: {e}")
    import traceback
    traceback.print_exc() 