#!/usr/bin/env python3
"""
Debug script to examine survey questions and data mapping
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import create_app, db
from app.models import Survey, SurveyResponse, User

def debug_survey_mapping():
    """Debug the survey question mapping."""
    
    app = create_app()
    with app.app_context():
        print("=== SURVEY QUESTION MAPPING DEBUG ===")
        
        # Get the survey
        survey = Survey.query.first()
        if not survey:
            print("❌ No survey found")
            return
            
        print(f"✅ Survey: {survey.title}")
        
        # Get the questions
        questions = survey.get_questions()
        print(f"📋 Questions ({len(questions)}):")
        for i, question in enumerate(questions, 1):
            print(f"  Q{i} (ID: {question.get('id', 'N/A')}): {question.get('text', 'No text')[:100]}...")
            print(f"      Type: {question.get('type', 'N/A')}")
            if question.get('options'):
                print(f"      Options: {question.get('options')}")
            print()
        
        # Get the survey response
        response = SurveyResponse.query.first()
        if not response:
            print("❌ No survey response found")
            return
            
        print(f"✅ Survey Response from User {response.user_id}")
        
        # Get the raw answers
        answers = response.get_answers()
        print(f"📊 Raw answers: {answers}")
        
        # Map answers to questions
        print("\n🔍 Question-Answer Mapping:")
        for question_id, answer in answers.items():
            # Find the question with this ID
            question = next((q for q in questions if str(q.get('id')) == str(question_id)), None)
            if question:
                print(f"  Q{question_id}: {question.get('text', 'N/A')[:60]}...")
                print(f"         Answer: {answer}")
                print(f"         Type: {question.get('type', 'N/A')}")
            else:
                print(f"  Q{question_id}: [Question not found]")
                print(f"         Answer: {answer}")
            print()

if __name__ == "__main__":
    debug_survey_mapping() 